"""
Test determinism of Sionna RT ray tracing across multiple evaluations of the same trajectory.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

from sionna_feedback_engine import evaluate_uav_position, DEFAULT_10_USERS, ConnectivityConfig

config = ConnectivityConfig(rss_threshold_dbm=-50.0)

trajectory = [
    [-45.0, 50.0, 12.0],  # Step 0
    [-35.0, 50.0, 12.0],  # Step 1
    [-25.0, 50.0, 12.0],  # Step 2
    [-25.0, 40.0, 12.0],  # Step 3
    [-15.0, 40.0, 12.0],  # Step 4
    [-15.0, 30.0, 12.0],  # Step 5
    [-15.0, 30.0, 12.0],  # Step 6
]

def run_pass(pass_id):
    results = []
    for step_idx, pos in enumerate(trajectory):
        res = evaluate_uav_position(pos, user_positions=DEFAULT_10_USERS, scene_name="city_damaged.xml", connectivity_config=config)
        agg = res["aggregate"]
        results.append({
            "step": step_idx,
            "pos": pos,
            "connected": agg["connected_users_count"],
            "rss": agg["mean_rss_dbm"],
            "path_gain": agg["mean_path_gain_db"]
        })
    return results

print("Running Pass 1...")
p1 = run_pass(1)

print("Running Pass 2...")
p2 = run_pass(2)

print("\nComparing Pass 1 vs Pass 2:")
print(f"{'Step':<5} | {'Pass 1 Conn':<12} | {'Pass 2 Conn':<12} | {'Pass 1 RSS':<12} | {'Pass 2 RSS':<12} | {'Match?':<8}")
print("-" * 75)

matches = []
for r1, r2 in zip(p1, p2):
    match = (r1["connected"] == r2["connected"] and abs(r1["rss"] - r2["rss"]) < 1e-4)
    matches.append(match)
    print(f"{r1['step']:<5} | {r1['connected']:<12} | {r2['connected']:<12} | {r1['rss']:<12.2f} | {r2['rss']:<12.2f} | {'YES' if match else 'NO':<8}")

print(f"\nAll steps match perfectly: {all(matches)}")
