"""Reproduce the report's paired storm and seed-42 hourly diagnostics without retraining."""

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from reservoir_environment import FINAL_TEST_SEEDS, ReservoirFloodEnv, rule_curve_policy


def replay(storm, model=None):
    env = ReservoirFloodEnv(seed_pool=(storm,), forecast_horizon=3, continuous_action=model is not None)
    obs, _ = env.reset(options={"event_seed": storm})
    rows = []
    for _ in range(env.T):
        action = rule_curve_policy(obs, env) if model is None else model.predict(obs, deterministic=True)[0]
        obs, _, terminated, truncated, info = env.step(action)
        rows.append(info)
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(rows)


def main():
    results = Path(__file__).resolve().parent / "bmds_results"
    rows = pd.read_csv(results / "holdout_event_scores.csv")
    metrics = ["peak_damage", "int_damage", "spill_mcm"]
    means = rows.groupby(["method", "storm_id"])[metrics].mean()
    rule = means.loc["Rule-Curve"]
    paired = means.loc["Tuned PPO"] - rule
    print("Tuned PPO minus the rule curve, averaged over training seeds within each storm")
    print(paired.round(2).to_string())
    print("\nStorms where tuned PPO is lower")
    print(paired.lt(0).sum().to_string())

    cases = {
        int(rule.peak_damage.idxmax()),
        int(paired.peak_damage.idxmax()),
        int(paired.peak_damage.idxmin()),
    }
    model = PPO.load(results / "assignment_model.zip", device="cpu")
    records = []
    for name, policy in [("Rule-Curve", None), ("Tuned PPO", model)]:
        reference = rows[rows.method == name]
        if policy is not None:
            reference = reference[reference.training_seed == 42]
        reference = reference.set_index("storm_id")
        for storm in FINAL_TEST_SEEDS:
            trace = replay(storm, policy)
            # Each step represents one hour. Convert m³/s to MCM per step.
            spill_mcm = trace.spill.sum() * 3600 / 1e6
            np.testing.assert_allclose(
                [trace.damage.max(), trace.damage.sum(), spill_mcm],
                reference.loc[storm, metrics].to_numpy(dtype=float),
                rtol=2e-6,
            )
            records.append({
                "policy": name,
                "final_storage_mcm": trace.storage_mcm.iloc[-1],
                "max_storage_mcm": trace.storage_mcm.max(),
                "gate_volume_mcm": trace.gate_release.sum() * 3600 / 1e6,
                "spill_mcm": spill_mcm,
                "hours_spilling": trace.spill.gt(0).sum(),
                "hours_with_damage": trace.damage.gt(0).sum(),
                "hours_above_6m": trace.stage.gt(6).sum(),
                "hours_above_7_5m": trace.stage.gt(7.5).sum(),
            })
            if storm in cases:
                peak = trace.loc[trace.damage.idxmax(), ["hour", "damage", "storage_mcm", "gate_release", "spill"]]
                window = (trace.hour - 1) // 24 + 1
                by_window = trace.assign(spill_mcm=trace.spill * 3600 / 1e6).groupby(window)[["damage", "spill_mcm"]].sum()
                print(f"\nStorm {storm}, {name}, peak-hour measurements")
                print(peak.round(2).to_string())
                print("Successive 24-hour blocks (integrated damage and spill volume)")
                print(by_window.round(2).to_string())
    print("\nMean over 30 storms (the tuned policy here is seed 42)")
    print(pd.DataFrame(records).groupby("policy").mean().round(2).to_string())


if __name__ == "__main__":
    main()
