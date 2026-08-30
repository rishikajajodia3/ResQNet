"""
Test positions near building blocks in city_damaged.xml to identify non-100% coverage states.
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

# Let's test standard users vs clustered users behind tall buildings
print("Testing various locations and thresholds...")

# 1. Test DEFAULT_10_USERS at low altitude or edge positions
test_configs = [
    # Low altitude / shadowed positions
    ([-18.82, 48.41, 5.0], "Low altitude (Z=5m)"),
    ([-18.82, 48.41, 8.0], "Low altitude (Z=8m)"),
    ([60.0, -10.0, 10.0], "Edge Canyon East (Z=10m)"),
    ([-60.0, 20.0, 10.0], "Edge Canyon West (Z=10m)"),
    ([0.0, -80.0, 12.0], "South Blockage (Z=12m)"),
    ([100.0, 50.0, 15.0], "Far East (Z=15m)"),
    ([-50.0, -50.0, 5.0], "South-West Low (Z=5m)"),
]

cfg = ConnectivityConfig(rss_threshold_dbm=-95.0, sinr_threshold_db=-3.0)

for pos, desc in test_configs:
    res = evaluate_uav_position(pos, user_positions=DEFAULT_10_USERS, scene_name="city_damaged.xml", connectivity_config=cfg)
    agg = res["aggregate"]
    print(f"{desc:<25} | Pos: {pos} | Conn: {agg['connected_users_count']}/10 | Mean RSS: {agg['mean_rss_dbm']} dBm | Paths per user: {[u['num_paths'] for u in res['user_results']]}")
