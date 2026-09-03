"""
Test and demonstration of Normal (city_baseline.xml) vs Damaged (city_damaged.xml) scene comparison.
"""

import os
import sys
import json

# Ensure sionna directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sionna_feedback_engine import (
    compare_normal_vs_damaged,
    DEFAULT_10_USERS
)


def run_comparison():
    print("=" * 80)
    print("RESQNET: NORMAL (BASELINE) VS DAMAGED CITY SCENE PROPAGATION COMPARISON")
    print("=" * 80)

    uav_pos = [-18.82, 48.41, 30.57]
    user_positions = DEFAULT_10_USERS

    print(f"\nUAV Transmitter Position: {uav_pos}")
    print(f"Number of Evaluated Ground Users: {len(user_positions)}")
    print("\nRunning live Sionna RT PathSolver on both scenes...")

    results = compare_normal_vs_damaged(
        uav_position=uav_pos,
        user_positions=user_positions
    )

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Baseline (Normal)':<18} | {'Damaged City':<18} | {'Delta (Damaged - Base)':<18}")
    print("-" * 95)
    
    b_agg = results["baseline_aggregate"]
    d_agg = results["damaged_aggregate"]
    deltas = results["aggregate_deltas"]

    print(f"{'Connected Users Count':<35} | {b_agg['connected_users_count']:<18} | {d_agg['connected_users_count']:<18} | {deltas['delta_connected_users_count']:+d}")
    print(f"{'Connected Users %':<35} | {b_agg['connected_users_percentage']:<18.1f} | {d_agg['connected_users_percentage']:<18.1f} | {deltas['delta_connected_users_percentage']:+.1f}%")
    print(f"{'Coverage %':<35} | {b_agg['coverage_percentage']:<18.1f} | {d_agg['coverage_percentage']:<18.1f} | {deltas['delta_coverage_percentage']:+.1f}%")
    print(f"{'Mean Path Gain (dB)':<35} | {b_agg['mean_path_gain_db']:<18.2f} | {d_agg['mean_path_gain_db']:<18.2f} | {deltas['delta_mean_path_gain_db']:+.2f} dB")
    print(f"{'Mean RSS (dBm)':<35} | {b_agg['mean_rss_dbm']:<18.2f} | {d_agg['mean_rss_dbm']:<18.2f} | {deltas['delta_mean_rss_dbm']:+.2f} dBm")
    if b_agg['mean_sinr_db'] is not None and d_agg['mean_sinr_db'] is not None:
        print(f"{'Mean SINR/SNR (dB)':<35} | {b_agg['mean_sinr_db']:<18.2f} | {d_agg['mean_sinr_db']:<18.2f} | {deltas['delta_mean_sinr_db']:+.2f} dB")

    print("\n" + "=" * 80)
    print("PER-USER WIRELESS PROPAGATION BREAKDOWN (ALL 10 USERS)")
    print("=" * 80)
    print(f"{'User ID':<8} | {'Position (x,y,z)':<20} | {'Base RSS':<10} | {'Damaged RSS':<12} | {'Delta RSS':<10} | {'Paths (B/D)':<11} | {'Conn (B/D)':<10}")
    print("-" * 95)

    for u in results["user_comparisons"]:
        uid = u["user_id"]
        pos_str = f"[{u['position'][0]:.0f}, {u['position'][1]:.0f}, {u['position'][2]:.1f}]"
        b_rss = f"{u['baseline']['rss_dbm']:.2f}"
        d_rss = f"{u['damaged']['rss_dbm']:.2f}"
        delta_rss = f"{u['deltas']['delta_rss_dbm']:+.2f} dB"
        paths_str = f"{u['baseline']['num_paths']} / {u['damaged']['num_paths']}"
        conn_str = f"{'Y' if u['baseline']['connected'] else 'N'} -> {'Y' if u['damaged']['connected'] else 'N'}"
        
        print(f"{uid:<8} | {pos_str:<20} | {b_rss:<10} | {d_rss:<12} | {delta_rss:<10} | {paths_str:<11} | {conn_str:<10}")

    print("\n" + "=" * 80)
    print("PHYSICAL PROPAGATION VALIDATION CONCLUSION")
    print("=" * 80)
    print(f"Physical propagation changed due to geometry: {results['physical_propagation_changed']}")
    print(f"Number of users with altered propagation paths: {results['num_users_with_propagation_change']} / 10")
    print(f"Users with connectivity state changed: {results['connectivity_changed_users'] if results['connectivity_changed_users'] else 'None (all maintained signal above threshold)'}")


if __name__ == "__main__":
    run_comparison()
