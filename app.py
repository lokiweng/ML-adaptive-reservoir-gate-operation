import json
from io import BytesIO
from pathlib import Path
from pickle import UnpicklingError
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from reservoir_environment import (
    EXPLORED_SEEDS,
    ReservoirFloodEnv,
    forecast_prerelease_policy,
    rule_curve_policy,
)

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


def run_episode(policy_fn, seed, horizon, noise_std, continuous=False):
    env = ReservoirFloodEnv(
        seed_pool=(seed,),
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
        "int_damage": float(df["damage"].sum() * env.dt_hours),
        "spill_mcm": float(spill_mcm),
        "storage_violation_hours": float((df["over_max"].sum() + df["under_min"].sum()) * env.dt_hours),
        "mean_gate_movement": float(df["delta_a"].mean()),
        "mb_err": float(df["mass_balance_err_m3"].max()),
    }
    return df, summary


# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------
st.title("🌊 Adaptive Reservoir Flood Control — Pure Simulation Prototype")
st.caption("BMDS2114 Machine Learning Assignment (Group 6) | Interactive Storm & Policy Visualizer")

with st.sidebar:
    st.header("Simulation Controls")
    event_seed = st.select_slider("Exploratory Storm ID", options=EXPLORED_SEEDS, value=3005)
    noise_pct = st.selectbox("Forecast Noise (%)", options=[0, 10, 20], index=0)
    noise_std = noise_pct / 100.0

    st.divider()
    st.header("Trained Model")
    uploaded_zip = st.file_uploader(
        "Upload a checkpoint from the revised notebook",
        type=["zip"],
    )
    uploaded_algorithm = st.radio("Uploaded Model Type", options=["DQN / Double DQN", "PPO"])

rl_label = "Forecast Heuristic"
rl_policy_fn = forecast_prerelease_policy
rl_continuous = False
model_horizon = None
checkpoint_dir = Path(__file__).resolve().parent / "bmds_results"
checkpoint = checkpoint_dir / "assignment_model.zip"
checkpoint_metadata = checkpoint_dir / "assignment_model.json"

if uploaded_zip is not None or checkpoint.is_file():
    if not SB3_AVAILABLE:
        st.sidebar.error("Stable-Baselines3 is required to load the trained model.")
    else:
        try:
            if uploaded_zip is not None:
                model_source = BytesIO(uploaded_zip.getvalue())
                algorithm = uploaded_algorithm
                model_label = f"Uploaded {algorithm}"
            else:
                metadata = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
                model_source = checkpoint
                algorithm = metadata["algorithm"]
                model_label = metadata["method"]

            model_cls = PPO if algorithm == "PPO" else DQN
            model = model_cls.load(model_source, device="cpu")
            obs_shape = model.observation_space.shape
            if obs_shape is None or len(obs_shape) != 1 or obs_shape[0] - 5 not in (0, 3, 6):
                raise ValueError(f"Unsupported observation shape {obs_shape}")
            if model_cls is PPO and model.action_space.shape != (1,):
                raise ValueError("The PPO checkpoint must have one continuous gate action")
            if model_cls is DQN and getattr(model.action_space, "n", None) != 11:
                raise ValueError("The DQN checkpoint must have 11 discrete gate actions")

            model_horizon = obs_shape[0] - 5
            rl_continuous = model_cls is PPO
            rl_label = model_label
            rl_policy_fn = lambda obs, env: model.predict(obs, deterministic=True)[0]
        except (OSError, ValueError, RuntimeError, KeyError, TypeError,
                AttributeError, AssertionError, EOFError, BadZipFile, UnpicklingError) as exc:
            st.sidebar.error(f"Could not use the checkpoint: {exc}")

if model_horizon is None:
    horizon = st.sidebar.selectbox("Forecast Horizon (hours)", options=[0, 3, 6], index=1)
    st.sidebar.caption("The comparison uses the built-in forecast heuristic.")
else:
    horizon = model_horizon
    st.sidebar.caption(f"{rl_label} · fixed forecast horizon of {horizon} hours.")

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
c4.metric("Mass-Balance Residual", f"{sum_rl['mb_err']:.2e} m³", "Maximum running residual")

# Comparison Table
st.subheader(f"Event {event_seed} — Policy Scorecard")
score_df = pd.DataFrame([
    {"Method": "Zero-Opening", **sum_zero},
    {"Method": "Fixed-Opening (50%)", **sum_fixed},
    {"Method": "Rule-Curve", **sum_rule},
    {"Method": rl_label, **sum_rl},
]).rename(columns={
    "reward": "Episode reward",
    "peak_damage": "Peak damage ($M)",
    "int_damage": "Integrated damage ($M·hr)",
    "spill_mcm": "Spill (MCM)",
    "storage_violation_hours": "Storage violations (hr)",
    "mean_gate_movement": "Mean absolute gate movement",
    "mb_err": "Max mass-balance residual (m³)",
})
st.dataframe(score_df.round(3), use_container_width=True, hide_index=True)

# 2x2 Hydrograph Plot
st.subheader(f"72-Hour Storm Simulation Trajectory (Seed {event_seed})")
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)

axes[0, 0].plot(df_rl["hour"], df_rl["inflow"], color="black", linestyle="--", linewidth=1.8, label="Inflow I_t")
axes[0, 0].plot(df_zero["hour"], df_zero["release"], color="#7f8c8d", linewidth=1.8, label="Zero-Opening")
axes[0, 0].plot(df_rule["hour"], df_rule["release"], color="#3498db", linewidth=1.8, label="Rule-Curve")
axes[0, 0].plot(df_rl["hour"], df_rl["release"], color="#2ecc71", linewidth=2.2, label=rl_label)
axes[0, 0].set_title("Reservoir Inflow and Release (m³/s)")
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
plt.close(fig)

with st.expander("View Hourly Simulation Data & Download CSV"):
    st.dataframe(df_rl.round(3), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Event CSV",
        data=df_rl.to_csv(index=False).encode("utf-8"),
        file_name=f"reservoir_event_{event_seed}.csv",
        mime="text/csv",
    )
