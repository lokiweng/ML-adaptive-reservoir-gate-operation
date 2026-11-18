import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import DQN, PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

st.set_page_config(
    page_title="G6 Reservoir Flood RL Simulator",
    page_icon="🌊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. Procedural Storm Generator & Reservoir Environment (matches Colab notebook)
# ---------------------------------------------------------------------------
def generate_flood_hydrograph(seed: int, T: int = 72, dt_hours: float = 1.0):
    rng = np.random.default_rng(seed)
    baseflow = rng.uniform(30.0, 80.0)
    peak_add = rng.uniform(350.0, 950.0)
    t_peak = rng.uniform(14.0, 30.0)
    shape_alpha = rng.uniform(2.2, 4.5)

    t = np.arange(T, dtype=np.float32) * dt_hours + 1.0
    ratio = np.clip(t / t_peak, 1e-3, None)
    storm_wave = peak_add * (ratio ** shape_alpha) * np.exp(shape_alpha * (1.0 - ratio))
    noise = rng.normal(0.0, 12.0, size=T)
    inflow = np.clip(baseflow + storm_wave + noise, 20.0, None)

    local_base = rng.uniform(15.0, 45.0)
    local_peak = rng.uniform(60.0, 180.0)
    local_tp = max(6.0, t_peak - rng.uniform(2.0, 6.0))
    local_ratio = np.clip(t / local_tp, 1e-3, None)
    q_local = np.clip(
        local_base + local_peak * (local_ratio ** 2.5) * np.exp(2.5 * (1.0 - local_ratio)),
        10.0,
        None,
    )
    return inflow.astype(np.float32), q_local.astype(np.float32)


class ReservoirFloodEnv(gym.Env):
    def __init__(
        self,
        T=72,
        dt_hours=1.0,
        forecast_horizon=3,
        forecast_noise_std=0.0,
        n_discrete_actions=11,
        continuous_action=False,
        max_delta_a=0.3,
        w_D=1.0,
        w_peak=1.5,
        w_crest=1.0,
        w_S=5.0,
        w_L=2.0,
        w_delta=0.01,
    ):
        super().__init__()
        self.T = T
        self.dt_hours = dt_hours
        self.dt_sec = dt_hours * 3600.0
        self.forecast_horizon = forecast_horizon
        self.forecast_noise_std = forecast_noise_std
        self.n_discrete_actions = n_discrete_actions
        self.continuous_action = continuous_action
        self.max_delta_a = max_delta_a

        self.w_D = w_D
        self.w_peak = w_peak
        self.w_crest = w_crest
        self.w_S = w_S
        self.w_L = w_L
        self.w_delta = w_delta

        self.S_min = 15.0e6
        self.S_init = 35.0e6
        self.S_crest = 55.0e6
        self.S_max = 65.0e6
        self.S_abs_max = 80.0e6

        self.C_d = 0.72
        self.A_max = 42.0
        self.g = 9.81
        self.D_ref = 50.0

        obs_dim = 5 + self.forecast_horizon
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(obs_dim,), dtype=np.float32)
        if self.continuous_action:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.n_discrete_actions)
            self.action_levels = np.linspace(0.0, 1.0, self.n_discrete_actions, dtype=np.float32)

    def _storage_to_head(self, S):
        return max(1.0, 4.0 + 18.0 * (S / self.S_max) ** 0.75)

    def _discharge_to_stage(self, Q):
        return 1.5 + 0.14 * (max(0.0, Q) ** 0.55)

    def _stage_to_damage(self, stage):
        if stage <= 4.5:
            return 0.0
        elif stage <= 6.0:
            return 12.0 * (stage - 4.5)
        elif stage <= 7.5:
            return 18.0 + 35.0 * (stage - 6.0)
        else:
            return 70.5 + 80.0 * (stage - 7.5)

    def _get_obs(self):
        S_norm = np.clip(self.S / self.S_abs_max, 0.0, 1.5)
        I_norm = np.clip(self.inflow[self.t] / 1200.0, 0.0, 1.5)
        forecasts = []
        for k in range(1, self.forecast_horizon + 1):
            idx = min(self.t + k, self.T - 1)
            forecasts.append(np.clip(self.inflow_forecast[idx] / 1200.0, 0.0, 1.5))
        R_norm = np.clip(self.R_prev / 1200.0, 0.0, 1.5)
        D_norm = np.clip(self.D_curr / 100.0, 0.0, 1.5)
        t_norm = self.t / float(self.T)
        return np.array([S_norm, I_norm, *forecasts, R_norm, D_norm, t_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        event_seed = int(options["event_seed"]) if (options and "event_seed" in options) else (seed or 3000)
        self.inflow, self.q_local = generate_flood_hydrograph(event_seed, T=self.T, dt_hours=self.dt_hours)
        if self.forecast_noise_std > 0.0:
            rng = np.random.default_rng(event_seed + 99999)
            noise = rng.normal(0.0, self.forecast_noise_std, size=self.T)
            self.inflow_forecast = np.clip(self.inflow * (1.0 + noise), 10.0, None)
        else:
            self.inflow_forecast = self.inflow.copy()

        self.t = 0
        self.S = self.S_init
        self.S0 = self.S_init
        self.a_prev = 0.2
        self.R_prev = 50.0
        self.D_curr = 0.0
        self.total_inflow_vol = 0.0
        self.total_release_vol = 0.0
        return self._get_obs(), {"event_seed": event_seed}

    def step(self, action):
        if self.continuous_action:
            a_target = float(np.clip(action[0], 0.0, 1.0))
        else:
            a_target = float(self.action_levels[int(action)])

        a_t = float(np.clip(a_target, self.a_prev - self.max_delta_a, self.a_prev + self.max_delta_a))
        delta_a = abs(a_t - self.a_prev)

        H_t = self._storage_to_head(self.S)
        R_gate = self.C_d * a_t * self.A_max * np.sqrt(2.0 * self.g * H_t)
        R_spill = 0.0
        if self.S > self.S_crest:
            excess_m = (self.S - self.S_crest) / 1.0e6
            R_spill = 35.0 * (excess_m ** 1.5)

        R_t = R_gate + R_spill
        I_t = float(self.inflow[self.t])
        S_next = max(0.0, self.S + (I_t - R_t) * self.dt_sec)

        Q_t = R_t + float(self.q_local[self.t])
        stage_t = self._discharge_to_stage(Q_t)
        D_t = self._stage_to_damage(stage_t)

        d_ratio = D_t / self.D_ref
        over_crest = max(0.0, S_next - self.S_crest) / self.S_crest
        over_max = max(0.0, S_next - self.S_max) / self.S_max
        under_min = max(0.0, self.S_min - S_next) / self.S_min
        reward = -float(
            self.w_D * d_ratio
            + self.w_peak * (d_ratio ** 2)
            + self.w_crest * over_crest
            + self.w_S * over_max
            + self.w_L * under_min
            + self.w_delta * delta_a
        )

        self.total_inflow_vol += I_t * self.dt_sec
        self.total_release_vol += R_t * self.dt_sec
        self.S = S_next
        self.a_prev = a_t
        self.R_prev = R_t
        self.D_curr = D_t
        self.t += 1

        terminated = self.t >= self.T
        info = {
            "hour": self.t,
            "inflow": I_t,
            "release": R_t,
            "spill": R_spill,
            "storage_mcm": self.S / 1.0e6,
            "damage": D_t,
            "gate_opening": a_t,
            "mass_balance_err_m3": abs(self.S - self.S0 - (self.total_inflow_vol - self.total_release_vol)),
        }
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, False, info


# ---------------------------------------------------------------------------
# 2. Policies & Rollout Helper
# ---------------------------------------------------------------------------
def rule_curve_policy(obs, env):
    storage_ratio = obs[0] * env.S_abs_max / env.S_max
    inflow_curr = obs[1] * 1200.0
    if storage_ratio > 0.85 or inflow_curr > 600.0:
        a_target = 0.9
    elif storage_ratio > 0.65 or inflow_curr > 350.0:
        a_target = 0.6
    elif storage_ratio > 0.45:
        a_target = 0.3
    else:
        a_target = 0.1
    return int(np.argmin(np.abs(env.action_levels - a_target)))


def forecast_prerelease_policy(obs, env):
    """Built-in anticipatory heuristic when no trained SB3 .zip is uploaded."""
    storage_mcm = obs[0] * env.S_abs_max / 1.0e6
    forecast_vals = obs[2:2 + env.forecast_horizon] * 1200.0 if env.forecast_horizon > 0 else [obs[1] * 1200.0]
    max_fc = float(np.max(forecast_vals))
    if storage_mcm > 50.0 or max_fc > 650.0:
        a_target = 0.8
    elif storage_mcm > 42.0 or max_fc > 400.0:
        a_target = 0.6
    elif storage_mcm > 30.0:
        a_target = 0.4
    else:
        a_target = 0.1
    return int(np.argmin(np.abs(env.action_levels - a_target)))


def run_episode(policy_fn, seed, horizon, noise_std, continuous=False):
    env = ReservoirFloodEnv(
        forecast_horizon=horizon,
        forecast_noise_std=noise_std,
        continuous_action=continuous,
    )
    obs, _ = env.reset(options={"event_seed": seed})
    done = False
    rows = []
    total_reward = 0.0
    spill_mcm = 0.0
    while not done:
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        spill_mcm += info["spill"] * env.dt_sec / 1.0e6
        rows.append(info)
    df = pd.DataFrame(rows)
    summary = {
        "reward": total_reward,
        "peak_damage": float(df["damage"].max()),
        "int_damage": float(df["damage"].sum()),
        "spill_mcm": float(spill_mcm),
        "mb_err": float(df["mass_balance_err_m3"].iloc[-1]),
    }
    return df, summary


# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------
st.title("🌊 Adaptive Reservoir Flood Control — Pure Simulation Prototype")
st.caption("BMDS2114 Machine Learning Assignment (Group 6) | Interactive Storm & Policy Visualizer")

with st.sidebar:
    st.header("Simulation Controls")
    event_seed = st.slider("Unseen Test Event Seed", min_value=3000, max_value=3029, value=3005, step=1)
    horizon = st.selectbox("Forecast Horizon (hours)", options=[0, 3, 6], index=1)
    noise_pct = st.selectbox("Forecast Noise (%)", options=[0, 10, 20], index=0)
    noise_std = noise_pct / 100.0

    st.divider()
    st.header("Trained RL Model (Optional)")
    uploaded_zip = st.file_uploader(
        "Upload SB3 model (.zip from Colab)",
        type=["zip"],
        help="In Colab, run: best_dqn_model.save('best_rl_model.zip') and upload it here.",
    )
    algo_type = st.radio("Model Type", options=["DQN / Double DQN", "PPO (Continuous)"], index=0)

# Load uploaded SB3 model if provided
rl_label = "Forecast Pre-Release (Built-in)"
rl_policy_fn = forecast_prerelease_policy
rl_continuous = False

if uploaded_zip is not None and SB3_AVAILABLE:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded_zip.read())
        tmp_path = tmp.name
    try:
        if algo_type == "PPO (Continuous)":
            model = PPO.load(tmp_path)
            rl_continuous = True
            rl_label = "Uploaded PPO Agent"
        else:
            model = DQN.load(tmp_path)
            rl_continuous = False
            rl_label = "Uploaded DQN / DDQN Agent"
        rl_policy_fn = lambda obs, env: model.predict(obs, deterministic=True)[0]
    except Exception as exc:
        st.sidebar.error(f"Could not load model: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Run policies on selected storm
df_zero, sum_zero = run_episode(lambda o, e: 0, event_seed, horizon, noise_std)
df_fixed, sum_fixed = run_episode(lambda o, e: 5, event_seed, horizon, noise_std)
df_rule, sum_rule = run_episode(rule_curve_policy, event_seed, horizon, noise_std)
df_rl, sum_rl = run_episode(rl_policy_fn, event_seed, horizon, noise_std, continuous=rl_continuous)

# KPI Row
dmg_red = 100.0 * (sum_zero["peak_damage"] - sum_rl["peak_damage"]) / max(sum_zero["peak_damage"], 1e-6)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{rl_label} Peak Damage", f"${sum_rl['peak_damage']:.1f}M", f"{dmg_red:+.1f}% vs Zero-Opening")
c2.metric("Integrated Damage", f"${sum_rl['int_damage']:.1f}M·hr", f"Rule-Curve: ${sum_rule['int_damage']:.1f}M·hr")
c3.metric("Spillway Volume", f"{sum_rl['spill_mcm']:.2f} MCM", f"Zero-Open: {sum_zero['spill_mcm']:.2f} MCM")
c4.metric("Mass-Balance Residual", f"{sum_rl['mb_err']:.2e} m³", "Exact conservation")

# Comparison Table
st.subheader(f"Event {event_seed} — Policy Scorecard")
score_df = pd.DataFrame([
    {"Method": "Zero-Opening", **sum_zero},
    {"Method": "Fixed-Opening (50%)", **sum_fixed},
    {"Method": "Rule-Curve", **sum_rule},
    {"Method": rl_label, **sum_rl},
])
st.dataframe(score_df.round(3), use_container_width=True, hide_index=True)

# 2x2 Hydrograph Plot
st.subheader(f"72-Hour Storm Simulation Trajectory (Seed {event_seed})")
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)

axes[0, 0].plot(df_rl["hour"], df_rl["inflow"], color="black", linestyle="--", linewidth=1.8, label="Inflow I_t")
axes[0, 0].plot(df_zero["hour"], df_zero["release"], color="#7f8c8d", linewidth=1.8, label="Zero-Opening")
axes[0, 0].plot(df_rule["hour"], df_rule["release"], color="#3498db", linewidth=1.8, label="Rule-Curve")
axes[0, 0].plot(df_rl["hour"], df_rl["release"], color="#2ecc71", linewidth=2.2, label=rl_label)
axes[0, 0].set_title("Inflow vs. Downstream Release (m³/s)")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(df_zero["hour"], df_zero["storage_mcm"], color="#7f8c8d", linewidth=1.8, label="Zero-Opening")
axes[0, 1].plot(df_rule["hour"], df_rule["storage_mcm"], color="#3498db", linewidth=1.8, label="Rule-Curve")
axes[0, 1].plot(df_rl["hour"], df_rl["storage_mcm"], color="#2ecc71", linewidth=2.2, label=rl_label)
axes[0, 1].axhline(65.0, color="red", linestyle=":", label="S_max (65 MCM)")
axes[0, 1].axhline(55.0, color="purple", linestyle=":", label="S_crest (55 MCM)")
axes[0, 1].axhline(15.0, color="orange", linestyle=":", label="S_min (15 MCM)")
axes[0, 1].set_title("Reservoir Storage (MCM)")
axes[0, 1].legend(fontsize=8)

axes[1, 0].plot(df_zero["hour"], df_zero["damage"], color="#7f8c8d", linewidth=1.8, label="Zero-Opening")
axes[1, 0].plot(df_rule["hour"], df_rule["damage"], color="#3498db", linewidth=1.8, label="Rule-Curve")
axes[1, 0].plot(df_rl["hour"], df_rl["damage"], color="#2ecc71", linewidth=2.2, label=rl_label)
axes[1, 0].set_title("Instantaneous Flood Damage ($M)")
axes[1, 0].set_xlabel("Hour")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(df_rule["hour"], df_rule["gate_opening"] * 100.0, color="#3498db", linewidth=1.8, label="Rule-Curve")
axes[1, 1].plot(df_rl["hour"], df_rl["gate_opening"] * 100.0, color="#2ecc71", linewidth=2.2, label=rl_label)
axes[1, 1].set_title("Gate Opening (%)")
axes[1, 1].set_xlabel("Hour")
axes[1, 1].set_ylim(-5, 105)
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
st.pyplot(fig)

with st.expander("View Hourly Simulation Data & Download CSV"):
    st.dataframe(df_rl.round(3), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Event CSV",
        data=df_rl.to_csv(index=False).encode("utf-8"),
        file_name=f"reservoir_event_{event_seed}.csv",
        mime="text/csv",
    )