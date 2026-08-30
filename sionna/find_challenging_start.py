"""
Broader search for occluded / partial connectivity positions in city_damaged.xml
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

from sionna_feedback_engine import evaluate_uav_position, DEFAULT_10_USERS, ConnectivityConfig

config = ConnectivityConfig(rss_threshold_dbm=-75.0, sinr_threshold_db=-3.0)  # Standard higher sensitivity or test positions

candidates = [
    # Peripheral / Far positions
    [-80.0, -80.0, 15.0],
    [-90.0, 30.0, 12.0],
    [90.0, -90.0, 15.0],
    [100.0, 100.0, 20.0],
    [-100.0, -50.0, 10.0],
    [-60.0, -60.0, 10.0],
    [-50.0, -70.0, 12.0],
    [-70.0, 70.0, 15.0],
    [-80.0, -30.0, 10.0],
    [-30.0, -80.0, 10.0],
    [80.0, -50.0, 12.0],
]

print("Evaluating peripheral candidate starting positions...")
print(f"{'Position (X, Y, Z)':<25} | {'Conn (-95dBm)':<14} | {'Conn (-75dBm)':<14} | {'Coverage %':<12} | {'Mean RSS (dBm)':<15}")
print("-" * 85)

cfg_std = ConnectivityConfig(rss_threshold_dbm=-95.0)
cfg_high = ConnectivityConfig(rss_threshold_dbm=-75.0)

for pos in candidates:
    res_std = evaluate_uav_position(pos, user_positions=DEFAULT_10_USERS, scene_name="city_damaged.xml", connectivity_config=cfg_std)
    res_high = evaluate_uav_position(pos, user_positions=DEFAULT_10_USERS, scene_name="city_damaged.xml", connectivity_config=cfg_high)
    
    conn_std = f"{res_std['aggregate']['connected_users_count']} / 10"
    conn_high = f"{res_high['aggregate']['connected_users_count']} / 10"
    cov_str = f"{res_std['aggregate']['coverage_percentage']:.1f}%"
    rss_str = f"{res_std['aggregate']['mean_rss_dbm']:.2f} dBm"
    pos_str = f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
    print(f"{pos_str:<25} | {conn_std:<14} | {conn_high:<14} | {cov_str:<12} | {rss_str:<15}")
