"""Shared synthetic reservoir simulator and policies for the notebook and Streamlit demo."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TRAIN_SEEDS = tuple(range(1000, 1200))
VAL_SEEDS = tuple(range(2000, 2030))
EXPLORED_SEEDS = tuple(range(3000, 3030))
FINAL_TEST_SEEDS = tuple(range(6000, 6030))


def generate_flood_hydrograph(seed, T=72, dt_hours=1.0):
    """
    Procedurally generates a synthetic storm inflow hydrograph I_t (m^3/s)
    and downstream local tributary inflow Q_local_t (m^3/s) for one episode.
    """
    rng = np.random.default_rng(seed)

    # Randomized storm parameters
    baseflow = rng.uniform(30.0, 80.0)          # m^3/s
    peak_add = rng.uniform(350.0, 950.0)        # m^3/s above baseflow
    t_peak = rng.uniform(14.0, 30.0)            # hour of peak inflow
    shape_alpha = rng.uniform(2.2, 4.5)         # skewness of flood wave

    t = np.arange(T, dtype=np.float32) * dt_hours + 1.0
    ratio = np.clip(t / t_peak, 1e-3, None)
    storm_wave = peak_add * (ratio ** shape_alpha) * np.exp(shape_alpha * (1.0 - ratio))

    # Independent hourly storm noise
    noise = rng.normal(0.0, 12.0, size=T)
    inflow = np.clip(baseflow + storm_wave + noise, 20.0, None)

    # Downstream local tributary inflow (smaller, slightly earlier or coincident)
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
    """
    Custom Gymnasium environment for Pure-Simulation Reservoir Flood Control.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed_pool=None,
        T=72,
        dt_hours=1.0,
        forecast_horizon=3,
        forecast_noise_std=0.0,
        n_discrete_actions=11,
        continuous_action=False,
        max_delta_a=0.3,
        w_D=1.0,
        w_peak=0.8,
        w_S=5.0,
        w_L=2.0,
        w_delta=0.01,
    ):
        super().__init__()
        self.seed_pool = tuple(seed_pool) if seed_pool is not None else TRAIN_SEEDS
        if not self.seed_pool:
            raise ValueError("seed_pool must contain at least one storm ID")
        self.T = T
        self.dt_hours = dt_hours
        self.dt_sec = dt_hours * 3600.0
        self.forecast_horizon = forecast_horizon
        self.forecast_noise_std = forecast_noise_std
        self.n_discrete_actions = n_discrete_actions
        self.continuous_action = continuous_action
        self.max_delta_a = max_delta_a

        # Reward weights
        self.w_D = w_D
        self.w_peak = w_peak
        self.w_S = w_S
        self.w_L = w_L
        self.w_delta = w_delta

        # Physical reservoir parameters (1e6 m^3 = MCM)
        self.S_min = 15.0e6
        self.S_init = 35.0e6
        self.S_crest = 55.0e6
        self.S_max = 65.0e6
        self.S_abs_max = 80.0e6

        # Gate hydraulics
        self.C_d = 0.72
        self.A_max = 42.0       # m^2 max gate area
        self.g = 9.81
        self.D_ref = 50.0       # Reference damage ($M) for reward scaling

        # Observation: [S_norm, I_norm, forecast_I_norm (h), R_prev_norm, D_norm, t/T]
        obs_dim = 5 + self.forecast_horizon
        self.observation_space = spaces.Box(
            low=0.0, high=1.5, shape=(obs_dim,), dtype=np.float32
        )

        if self.continuous_action:
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(self.n_discrete_actions)
            self.action_levels = np.linspace(0.0, 1.0, self.n_discrete_actions, dtype=np.float32)

        self._episode_idx = 0

    def _storage_to_head(self, S):
        # Monotonic storage-head curve (m above gate centerline)
        return max(1.0, 4.0 + 18.0 * (S / self.S_max) ** 0.75)

    def _gate_release(self, storage, opening):
        head = self._storage_to_head(storage)
        return float(self.C_d * opening * self.A_max * np.sqrt(2.0 * self.g * head))

    def _discharge_to_stage(self, Q):
        # Rating curve Phi(Q): downstream river stage (m)
        return 1.5 + 0.14 * (max(0.0, Q) ** 0.55)

    def _stage_to_damage(self, stage):
        # Piecewise stage-damage curve f_dmg(h) in $M
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
        I_curr = self.inflow[self.t]
        I_norm = np.clip(I_curr / 1200.0, 0.0, 1.5)

        forecasts = []
        for k in range(1, self.forecast_horizon + 1):
            idx = min(self.t + k, self.T - 1)
            f_val = self.inflow_forecast[idx]
            forecasts.append(np.clip(f_val / 1200.0, 0.0, 1.5))

        R_norm = np.clip(self.R_prev / 1200.0, 0.0, 1.5)
        D_norm = np.clip(self.D_curr / 100.0, 0.0, 1.5)
        t_norm = self.t / float(self.T)

        obs = np.array([S_norm, I_norm, *forecasts, R_norm, D_norm, t_norm], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # The Gymnasium seed controls repeatability; storm IDs always come from the split.
        if seed is not None:
            self._episode_idx = 0
        if options is not None and "event_seed" in options:
            event_seed = int(options["event_seed"])
            if event_seed not in self.seed_pool:
                raise ValueError(f"Storm {event_seed} is outside this environment's seed pool")
        else:
            event_seed = self.seed_pool[self._episode_idx % len(self.seed_pool)]
            self._episode_idx += 1

        self.current_event_seed = event_seed
        self.inflow, self.q_local = generate_flood_hydrograph(
            event_seed, T=self.T, dt_hours=self.dt_hours
        )

        # Optional forecast noise for robustness testing
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
        self.R_prev = self._gate_release(self.S, self.a_prev)
        self.D_curr = 0.0
        self.total_inflow_vol = 0.0
        self.total_release_vol = 0.0
        self.total_spill_vol = 0.0

        return self._get_obs(), {"event_seed": event_seed}

    def step(self, action):
        if self.continuous_action:
            a_target = float(np.clip(np.asarray(action).item(), 0.0, 1.0))
        else:
            a_target = float(self.action_levels[int(action)])

        # Enforce gate rate-of-change limit |a_t - a_{t-1}| <= max_delta_a
        a_t = float(np.clip(a_target, self.a_prev - self.max_delta_a, self.a_prev + self.max_delta_a))
        delta_a = abs(a_t - self.a_prev)

        # Hydraulics: controlled gate release + uncontrolled spillway if S > S_crest
        R_gate = self._gate_release(self.S, a_t)
        R_spill = 0.0
        if self.S > self.S_crest:
            excess_mcm = (self.S - self.S_crest) / 1.0e6
            R_spill = 35.0 * (excess_mcm ** 1.5)

        I_t = float(self.inflow[self.t])

        # Cap the requested release at the water available this step. Any spill
        # is uncontrolled, so it takes priority over the requested gate release.
        available_volume = self.S + I_t * self.dt_sec
        spill_volume = min(R_spill * self.dt_sec, available_volume)
        gate_volume = min(R_gate * self.dt_sec, available_volume - spill_volume)
        release_volume = spill_volume + gate_volume
        R_spill = spill_volume / self.dt_sec
        R_t = release_volume / self.dt_sec
        S_next = available_volume - release_volume

        # Downstream stage & flood damage
        Q_t = R_t + float(self.q_local[self.t])
        stage_t = self._discharge_to_stage(Q_t)
        D_t = self._stage_to_damage(stage_t)

        # Reward computation (linear + quadratic peak-damage shaping)
        d_ratio = D_t / self.D_ref
        over_max = max(0.0, S_next - self.S_max) / self.S_max
        under_min = max(0.0, self.S_min - S_next) / self.S_min
        penalty = (
            self.w_D * d_ratio
            + self.w_peak * (d_ratio ** 2)
            + self.w_S * over_max
            + self.w_L * under_min
            + self.w_delta * delta_a
        )
        reward = -float(penalty)

        # Track episode totals
        self.total_inflow_vol += I_t * self.dt_sec
        self.total_release_vol += release_volume
        self.total_spill_vol += spill_volume

        # Advance state
        self.S = S_next
        self.a_prev = a_t
        self.R_prev = R_t
        self.D_curr = D_t
        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        info = {
            "hour": self.t,
            "inflow": I_t,
            "release": R_t,
            "spill": R_spill,
            "gate_release": gate_volume / self.dt_sec,
            "storage_mcm": self.S / 1.0e6,
            "damage": D_t,
            "stage": stage_t,
            "gate_opening": a_t,
            "delta_a": delta_a,
            "over_max": int(self.S > self.S_max),
            "under_min": int(self.S < self.S_min),
            "mass_balance_err_m3": abs(
                self.S - self.S0 - (self.total_inflow_vol - self.total_release_vol)
            ),
        }

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, info


def rule_curve_policy(obs, env):
    """Simple domain-inspired heuristic rule curve based on storage and inflow."""
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
    # Map to nearest discrete action index
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
