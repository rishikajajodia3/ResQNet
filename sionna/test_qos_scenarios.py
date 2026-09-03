"""
Test realistic emergency QoS thresholds and transmit powers (e.g. handheld IoT / low-power emergency payload).
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

# 1. Test QoS threshold (e.g. -50 dBm for high-throughput video uplink or lower Tx power)
qos_configs = [
    # Tx Power 30 dBm, RSS threshold -50 dBm (high QoS / video stream)
    (30.0, ConnectivityConfig(rss_threshold_dbm=-50.0, path_gain_threshold_db=-80.0), "QoS High-Throughput (Threshold -50 dBm)"),
    # Tx Power 15 dBm (Handheld / low power emergency UAV), RSS threshold -65 dBm
    (15.0, ConnectivityConfig(rss_threshold_dbm=-65.0, path_gain_threshold_db=-80.0), "Low-Power Payload (15 dBm Tx, -65 dBm Sens)"),
    # Standard 30 dBm with distant / canyon positions
    (30.0, ConnectivityConfig(rss_threshold_dbm=-55.0, path_gain_threshold_db=-85.0), "Standard 30 dBm (Threshold -55 dBm)")
]

candidate_positions = [
    [-45.0, 50.0, 12.0],
    [50.0, -40.0, 15.0],
    [-18.82, 48.41, 30.57],
    [0.0, -10.0, 25.0],
    [10.0, 10.0, 25.0],
    [-20.0, -20.0, 20.0],
]

for tx_p, cfg, label in qos_configs:
    print(f"\n--- Testing Scenario: {label} ---")
    print(f"{'Position (X, Y, Z)':<25} | {'Connected Users':<18} | {'Coverage %':<12} | {'Mean RSS (dBm)':<15}")
    print("-" * 75)
    for pos in candidate_positions:
        res = evaluate_uav_position(pos, user_positions=DEFAULT_10_USERS, scene_name="city_damaged.xml", tx_power_dbm=tx_p, connectivity_config=cfg)
        agg = res["aggregate"]
        pos_str = f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        print(f"{pos_str:<25} | {agg['connected_users_count']} / 10 ({agg['connected_users_percentage']}%) | {agg['coverage_percentage']:.1f}%        | {agg['mean_rss_dbm']:.2f} dBm")
