"""
Exact Reproducibility Audit Script for ResQNet Sionna RT ↔ PPO Optimization.
Runs two identical trials from scratch and performs an exact floating-point audit across all steps.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

import sionna_feedback_engine
from ml.run_reproducible_experiment import run_experiment_trial


def audit_reproducibility():
    print("=" * 85)
    print("RESQNET: EXACT REPRODUCIBILITY VERIFICATION AUDIT")
    print("=" * 85)

    seed = 42
    start_pos = [-45.0, 50.0, 12.0]
    rss_thresh = -50.0
    tx_power = 30.0
    freq = 2.1e9
    noise_power = -94.0
    scene_name = "city_damaged.xml"

    # --- TRIAL 1 ---
    print("\n[Trial 1] Executing from clean state (seed=42)...")
    sionna_feedback_engine._GLOBAL_ENGINE = None
    records_1, actions, cum_reward_1 = run_experiment_trial(
        seed=seed,
        episode_id=1,
        start_pos=start_pos,
        rss_threshold_dbm=rss_thresh,
        tx_power_dbm=tx_power,
        freq=freq,
        noise_power_dbm=noise_power,
        scene_name=scene_name
    )

    # --- TRIAL 2 ---
    print("\n[Trial 2] Executing identical second pass from clean state (seed=42)...")
    sionna_feedback_engine._GLOBAL_ENGINE = None
    records_2, _, cum_reward_2 = run_experiment_trial(
        seed=seed,
        episode_id=2,
        start_pos=start_pos,
        rss_threshold_dbm=rss_thresh,
        tx_power_dbm=tx_power,
        freq=freq,
        noise_power_dbm=noise_power,
        scene_name=scene_name
    )

    # --- SIDE-BY-SIDE NUMERICAL COMPARISON ---
    print("\n" + "=" * 95)
    print("STEP-BY-STEP SIDE-BY-SIDE COMPARISON (TRIAL 1 vs. TRIAL 2)")
    print("=" * 95)
    print(f"{'Step':<5} | {'Action':<15} | {'UAV Position':<22} | {'Conn (T1/T2)':<14} | {'RSS T1 (dBm)':<13} | {'RSS T2 (dBm)':<13} | {'Reward (T1/T2)':<18}")
    print("-" * 105)

    diffs = []
    for r1, r2 in zip(records_1, records_2):
        step = r1["step"]
        act_name = r1["action_name"]
        pos_str = f"[{r1['uav_x']}, {r1['uav_y']}, {r1['uav_z']}]"
        conn_str = f"{r1['connected_users']} / {r2['connected_users']}"
        rss_1 = f"{r1['mean_rss']:.2f}"
        rss_2 = f"{r2['mean_rss']:.2f}"
        rew_str = f"{r1['reward']:+.4f} / {r2['reward']:+.4f}"

        print(f"{step:<5} | {act_name:<15} | {pos_str:<22} | {conn_str:<14} | {rss_1:<13} | {rss_2:<13} | {rew_str:<18}")

        # Check all keys
        for key in ["action", "uav_x", "uav_y", "uav_z", "connected_users", "mean_rss", "reward", "cumulative_reward"]:
            val1 = r1[key]
            val2 = r2[key]
            if val1 != val2:
                diffs.append({
                    "step": step,
                    "metric": key,
                    "trial_1_value": val1,
                    "trial_2_value": val2,
                    "delta": val2 - val1 if isinstance(val1, (int, float)) else "N/A"
                })

    print("\n" + "=" * 85)
    print("REPRODUCIBILITY AUDIT VERDICT")
    print("=" * 85)

    if len(diffs) == 0:
        print("EXACT REPRODUCIBILITY: PASS")
        print("All steps, trajectories, connected user counts, mean RSS, and rewards are 100% identical.")
    else:
        print("EXACT REPRODUCIBILITY: FAIL")
        print(f"Number of differing metric entries: {len(diffs)}")
        print("\nDiffering Metrics Detail:")
        for d in diffs:
            print(f"  * Step {d['step']} - {d['metric']}: Trial 1 = {d['trial_1_value']}, Trial 2 = {d['trial_2_value']} (Delta = {d['delta']})")

    return len(diffs) == 0, records_1, records_2, diffs


if __name__ == "__main__":
    audit_reproducibility()
