"""
Integration Test: UAVDynamicEnv in Surrogate mode and Live Sionna mode
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

from dynamic_model import UAVDynamicEnv


def run_tests():
    print("=" * 70)
    print("RUNNING PPO <-> SIONNA INTEGRATION VERIFICATION TESTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST 1: Default Surrogate Mode (use_sionna_feedback=False)
    # -------------------------------------------------------------
    print("\n--- [TEST 1] Testing Default Surrogate Mode (use_sionna_feedback=False) ---")
    env_surrogate = UAVDynamicEnv(use_sionna_feedback=False)
    obs_s, info_s = env_surrogate.reset(seed=42)
    
    print("Reset Observation shape:", obs_s.shape)
    assert obs_s.shape == (6,), f"Expected shape (6,), got {obs_s.shape}"
    print("Observation values [x, y, z, cov, power, num_users]:", obs_s)
    
    # Take step with action 2 (+X)
    obs_s_next, r_s, term_s, trunc_s, info_s_next = env_surrogate.step(2)
    assert obs_s_next.shape == (6,), f"Expected shape (6,), got {obs_s_next.shape}"
    assert info_s_next["action"] == 2
    assert info_s_next["new_position"][0] == info_s_next["old_position"][0] + 10.0
    print("Surrogate step reward:", r_s)
    print("Surrogate new observation:", obs_s_next)
    print("[PASS] TEST 1: Default surrogate mode successfully verified.")

    # -------------------------------------------------------------
    # TEST 2 & 3: Live Sionna Mode Initialization & Step Evaluation
    # -------------------------------------------------------------
    print("\n--- [TEST 2 & 3] Testing Live Sionna Mode (use_sionna_feedback=True) ---")
    env_sionna = UAVDynamicEnv(use_sionna_feedback=True, scene_name="city_damaged.xml")
    obs_sionna, info_sionna = env_sionna.reset(seed=42)
    
    print("Sionna mode Reset Obs shape:", obs_sionna.shape)
    assert obs_sionna.shape == (6,), f"Expected shape (6,), got {obs_sionna.shape}"
    assert info_sionna.get("sionna_metrics") is not None, "info['sionna_metrics'] must be present on reset in Sionna mode"
    print("Initial Sionna coverage:", obs_sionna[3], "%")
    print("Initial Sionna mean power:", obs_sionna[4], "dBm")

    # Take step with action 4 (+Y movement)
    print("\nExecuting action 4 (+Y movement)...")
    old_pos = obs_sionna[:3].copy()
    obs_next, reward, term, trunc, info = env_sionna.step(4)
    new_pos = obs_next[:3]

    print(f"UAV Position moved from {old_pos} -> {new_pos}")
    assert new_pos[1] == old_pos[1] + 10.0, "Y position should increase by 10m"

    # -------------------------------------------------------------
    # TEST 4: 10-User Wireless Metrics in info["sionna_metrics"]
    # -------------------------------------------------------------
    print("\n--- [TEST 4] Verifying 10-User Sionna Metrics in info ---")
    sionna_metrics = info.get("sionna_metrics")
    assert sionna_metrics is not None, "info['sionna_metrics'] must not be None"
    user_results = sionna_metrics["user_results"]
    assert len(user_results) == 10, f"Expected exactly 10 user results, got {len(user_results)}"
    
    print(f"Number of evaluated users in info['sionna_metrics']: {len(user_results)}")
    print(f"Connected users: {sionna_metrics['aggregate']['connected_users_count']}/10")
    print(f"Mean Path Gain: {sionna_metrics['aggregate']['mean_path_gain_db']} dB")
    print(f"Mean RSS: {sionna_metrics['aggregate']['mean_rss_dbm']} dBm")
    print(f"Reward calculated: {reward:.4f}")

    # -------------------------------------------------------------
    # TEST 5: Observation Invariance
    # -------------------------------------------------------------
    print("\n--- [TEST 5] Verifying Observation Shape & Invariance ---")
    assert obs_next.shape == (6,), f"Expected observation shape (6,), got {obs_next.shape}"
    assert env_sionna.observation_space.contains(obs_next.astype(np.float32)), "Observation must fall within observation_space bounds"
    print("Observation vector:", obs_next)
    print("[PASS] TEST 5: Observation shape is strictly (6,) and conforms to observation_space.")

    print("\n" + "=" * 70)
    print("ALL PPO <-> SIONNA INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
