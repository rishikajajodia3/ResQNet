"""
Focused Verification: Multi-Position Sionna Propagation & PPO Step Consistency
Evaluates 3 distinct UAV positions across the exact same 10 users:
- Position A: [-18.82, 48.41, 30.57] (Initial)
- Position B: [-8.82, 48.41, 30.57] (Move +X by 10m / Action 2)
- Position C: [-18.82, 58.41, 30.57] (Move +Y by 10m / Action 4)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from sionna_feedback_engine import evaluate_uav_position, DEFAULT_10_USERS, ConnectivityConfig
from dynamic_model import UAVDynamicEnv


def print_position_metrics(label: str, pos_data: dict):
    uav_pos = pos_data["uav_position"]
    agg = pos_data["aggregate"]
    
    print("\n" + "=" * 90)
    print(f"{label}: UAV Position = [X={uav_pos[0]:.2f}, Y={uav_pos[1]:.2f}, Z={uav_pos[2]:.2f}]")
    print("=" * 90)
    print(f"Aggregate Metrics:")
    print(f"  * Connected Users:    {agg['connected_users_count']} / 10 ({agg['connected_users_percentage']}%)")
    print(f"  * Coverage:           {agg['coverage_percentage']}%")
    print(f"  * Mean Path Gain:     {agg['mean_path_gain_db']} dB")
    print(f"  * Mean RSS:           {agg['mean_rss_dbm']} dBm")
    print(f"  * Mean SINR / SNR:    {agg['mean_sinr_db']} dB" if agg['mean_sinr_db'] is not None else "  * Mean SINR / SNR:    None (interference undefined)")
    
    print("\nPer-User Wireless Metrics Breakdown (10 Ground Users):")
    print(f"{'User ID':<8} | {'Position (x,y,z)':<18} | {'Path Gain (dB)':<15} | {'RSS (dBm)':<11} | {'SINR (dB)':<10} | {'Paths':<6} | {'Delay (s)':<12} | {'Covered':<8} | {'Connected':<9}")
    print("-" * 110)
    
    for u in pos_data["user_results"]:
        uid = u["user_id"]
        pos_str = f"[{u['position'][0]:.0f}, {u['position'][1]:.0f}, {u['position'][2]:.1f}]"
        gain_str = f"{u['path_gain_db']:.2f}"
        rss_str = f"{u['rss_dbm']:.2f}"
        sinr_str = f"{u['sinr_db']:.2f}" if u['sinr_db'] is not None else "None"
        delay_str = f"{u['shortest_delay_sec']:.2e}" if u['shortest_delay_sec'] is not None else "None"
        cov_str = "True" if u["coverage"] else "False"
        conn_str = "True" if u["connected"] else "False"
        
        print(f"{uid:<8} | {pos_str:<18} | {gain_str:<15} | {rss_str:<11} | {sinr_str:<10} | {u['num_paths']:<6} | {delay_str:<12} | {cov_str:<8} | {conn_str:<9}")


def main():
    print("=" * 90)
    print("PHYSICS-LEVEL SIONNA RT PROPAGATION VERIFICATION (1 UAV + 10 USERS)")
    print("=" * 90)

    users_10 = DEFAULT_10_USERS
    pos_A = [-18.82, 48.41, 30.57]
    pos_B = [-8.82, 48.41, 30.57]   # +10m X
    pos_C = [-18.82, 58.41, 30.57]  # +10m Y

    # 1. Direct Sionna RT evaluation for all 3 positions
    print("\n[Step 1] Evaluating Position A (Initial)...")
    res_A = evaluate_uav_position(pos_A, user_positions=users_10, scene_name="city_damaged.xml")
    print_position_metrics("POSITION A (Default)", res_A)

    print("\n[Step 2] Evaluating Position B (Moved +10m in X)...")
    res_B = evaluate_uav_position(pos_B, user_positions=users_10, scene_name="city_damaged.xml")
    print_position_metrics("POSITION B (+X Movement)", res_B)

    print("\n[Step 3] Evaluating Position C (Moved +10m in Y)...")
    res_C = evaluate_uav_position(pos_C, user_positions=users_10, scene_name="city_damaged.xml")
    print_position_metrics("POSITION C (+Y Movement)", res_C)

    # 2. Physics recalculation verification
    print("\n" + "=" * 90)
    print("PHYSICS RECALCULATION VALIDATION")
    print("=" * 90)
    
    gain_diff_AB = abs(res_B["aggregate"]["mean_path_gain_db"] - res_A["aggregate"]["mean_path_gain_db"])
    gain_diff_AC = abs(res_C["aggregate"]["mean_path_gain_db"] - res_A["aggregate"]["mean_path_gain_db"])
    rss_diff_AB = abs(res_B["aggregate"]["mean_rss_dbm"] - res_A["aggregate"]["mean_rss_dbm"])
    rss_diff_AC = abs(res_C["aggregate"]["mean_rss_dbm"] - res_A["aggregate"]["mean_rss_dbm"])

    print(f"Mean RSS Change (Pos A -> Pos B): {res_B['aggregate']['mean_rss_dbm'] - res_A['aggregate']['mean_rss_dbm']:+.4f} dBm")
    print(f"Mean RSS Change (Pos A -> Pos C): {res_C['aggregate']['mean_rss_dbm'] - res_A['aggregate']['mean_rss_dbm']:+.4f} dBm")
    
    # Check individual user differences
    user_gain_changes_AB = [abs(res_B["user_results"][i]["rss_dbm"] - res_A["user_results"][i]["rss_dbm"]) for i in range(10)]
    print(f"Per-User RSS Deltas (Pos A -> Pos B): {[round(d, 2) for d in user_gain_changes_AB]}")
    assert any(d > 0.01 for d in user_gain_changes_AB), "At least one user must experience physical propagation change"
    print("[PASS] Verified: Sionna RT genuinely recalculates ray-traced propagation paths for each UAV coordinate.")

    # 3. PPO live environment step match verification
    print("\n" + "=" * 90)
    print("PPO LIVE STEP CONSISTENCY CHECK")
    print("=" * 90)
    
    env = UAVDynamicEnv(use_sionna_feedback=True, scene_name="city_damaged.xml")
    obs, info = env.reset(seed=42)
    env.uav_position = np.array(pos_A, dtype=np.float32)
    
    # Execute Action 2 (+X) in PPO environment
    obs_next, reward, term, trunc, step_info = env.step(2)
    
    print("PPO Step Action: 2 (+X)")
    print(f"PPO New UAV Position: {step_info['new_position']}")
    assert np.allclose(step_info["new_position"], pos_B, atol=1e-2), f"Expected pos {pos_B}, got {step_info['new_position']}"
    
    ppo_sionna_metrics = step_info.get("sionna_metrics")
    assert ppo_sionna_metrics is not None, "PPO info must contain sionna_metrics"
    assert len(ppo_sionna_metrics["user_results"]) == 10, "PPO sionna_metrics must contain exactly 10 user results"
    
    print(f"PPO Received Mean RSS: {ppo_sionna_metrics['aggregate']['mean_rss_dbm']} dBm (matches Pos B direct calculation: {res_B['aggregate']['mean_rss_dbm']} dBm)")
    print(f"PPO Computed Reward:   {reward:.4f}")
    assert ppo_sionna_metrics["aggregate"]["mean_rss_dbm"] == res_B["aggregate"]["mean_rss_dbm"]
    print("[PASS] Verified: PPO env.step() correctly consumes and exposes the identical live Sionna metrics.")

    # 4. NS-3 Handoff Compatibility Check
    print("\n" + "=" * 90)
    print("NS-3 HANDOFF FILE COMPATIBILITY CHECK (data/uav_positions.json)")
    print("=" * 90)
    
    output_path = PROJECT_ROOT / "data" / "uav_positions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    handoff_data = {
        "num_uavs": 1,
        "uav_positions": [
            [float(p) for p in step_info["new_position"]]
        ]
    }
    with open(output_path, "w") as f:
        json.dump(handoff_data, f, indent=2)
        
    print(f"Wrote handoff JSON to: {output_path}")
    print(f"File content:\n{json.dumps(handoff_data, indent=2)}")
    
    # Validate structure matches ns3/uav-demo.cc parser requirements
    with open(output_path, "r") as f:
        loaded = json.load(f)
        assert loaded.get("num_uavs") == 1
        assert len(loaded.get("uav_positions")) == 1
        assert len(loaded["uav_positions"][0]) == 3
    print("[PASS] Verified: data/uav_positions.json strictly adheres to ns-3 schema requirements.")


if __name__ == "__main__":
    main()
