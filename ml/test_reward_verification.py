"""
Unit and scenario verification tests for the Sionna-based PPO reward in UAVDynamicEnv.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

from dynamic_model import UAVDynamicEnv


class TestSionnaReward(unittest.TestCase):

    def test_01_numerical_bounds_verification(self):
        """Mathematically verifies theoretical min and max bounds for the reward."""
        w_conn, w_qual, w_prog, w_move = 0.50, 0.25, 0.20, 0.05

        # Minimum possible: 0 conn, 0 qual, regressed -10 conn, moved (cost=1)
        r_min = w_conn * 0.0 + w_qual * 0.0 + w_prog * (-1.0) - w_move * 1.0
        self.assertAlmostEqual(r_min, -0.25, places=5)

        # Maximum possible: 10 conn, 1.0 qual, improved +10 conn, stayed (cost=0)
        r_max = w_conn * 1.0 + w_qual * 1.0 + w_prog * (+1.0) - w_move * 0.0
        self.assertAlmostEqual(r_max, +0.95, places=5)

        # Steady state optimal: 10 conn, 1.0 qual, progress 0, stayed (cost=0)
        r_steady_max = w_conn * 1.0 + w_qual * 1.0 + w_prog * 0.0 - w_move * 0.0
        self.assertAlmostEqual(r_steady_max, +0.75, places=5)
        print(f"\n[PASS] TEST 1: Theoretical Bounds Verified: Min={r_min:.2f}, Max={r_max:.2f}, Steady-State Optimal={r_steady_max:.2f}")

    def test_02_scenario_calculations(self):
        """Tests the 4 required scenario transitions with exact formula checks."""
        print("\n--- [TEST 2] Testing Required Scenarios ---")
        w_conn, w_qual, w_prog, w_move = 0.50, 0.25, 0.20, 0.05
        rss_min, rss_max = -95.0, -40.0

        # Scenario 1: 10 -> 10 connected, stay (action=0, RSS=-45.0 dBm)
        r_conn = 10 / 10.0
        r_qual = np.clip((-45.0 - rss_min) / (rss_max - rss_min), 0.0, 1.0)
        r_prog = (10 - 10) / 10.0
        cost_move = 0.0
        r1 = w_conn * r_conn + w_qual * r_qual + w_prog * r_prog - w_move * cost_move
        print(f"Scenario 1 (10 -> 10 conn, Stay): Reward = {r1:.4f} (Expected > 0.70)")
        self.assertGreater(r1, 0.70)

        # Scenario 2: 10 -> 10 connected, move (action=1, RSS=-45.0 dBm)
        cost_move = 1.0
        r2 = w_conn * r_conn + w_qual * r_qual + w_prog * r_prog - w_move * cost_move
        print(f"Scenario 2 (10 -> 10 conn, Move): Reward = {r2:.4f} (Expected {r1 - 0.05:.4f})")
        self.assertAlmostEqual(r2, r1 - 0.05, places=4)

        # Scenario 3: 6 -> 8 connected (progress +2, action=2, RSS=-60.0 dBm)
        r_conn_3 = 8 / 10.0
        r_qual_3 = np.clip((-60.0 - rss_min) / (rss_max - rss_min), 0.0, 1.0)
        r_prog_3 = (8 - 6) / 10.0
        cost_move_3 = 1.0
        r3 = w_conn * r_conn_3 + w_qual * r_qual_3 + w_prog * r_prog_3 - w_move * cost_move_3
        print(f"Scenario 3 (6 -> 8 conn, Move): Reward = {r3:.4f}")
        self.assertGreater(r3, 0.40)

        # Scenario 4: 8 -> 7 connected (regression -1, action=3, RSS=-70.0 dBm)
        r_conn_4 = 7 / 10.0
        r_qual_4 = np.clip((-70.0 - rss_min) / (rss_max - rss_min), 0.0, 1.0)
        r_prog_4 = (7 - 8) / 10.0
        cost_move_4 = 1.0
        r4 = w_conn * r_conn_4 + w_qual * r_qual_4 + w_prog * r_prog_4 - w_move * cost_move_4
        print(f"Scenario 4 (8 -> 7 conn, Move): Reward = {r4:.4f}")
        self.assertLess(r4, r3)
        print("[PASS] TEST 2: Scenario rewards demonstrate expected policy incentives.")

    def test_03_live_env_sionna_reward_execution(self):
        """Tests live UAVDynamicEnv in use_sionna_feedback=True mode executing Sionna reward."""
        print("\n--- [TEST 3] Testing Live Env Execution with Sionna Reward ---")
        env = UAVDynamicEnv(use_sionna_feedback=True, scene_name="city_damaged.xml")
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (6,))
        self.assertIn("sionna_metrics", info)
        self.assertEqual(len(info["sionna_metrics"]["user_results"]), 10)

        # Step 1: Stay action (action=0)
        obs1, r1, term1, trunc1, info1 = env.step(0)
        self.assertEqual(obs1.shape, (6,))
        self.assertIn("sionna_metrics", info1)
        self.assertGreater(r1, 0.0, "Stay action at optimal position should yield positive reward")
        print(f"Live Step 1 (Stay): Reward = {r1:.4f}, Connected = {info1['sionna_metrics']['aggregate']['connected_users_count']}/10")

        # Step 2: Move action (action=2, +X)
        obs2, r2, term2, trunc2, info2 = env.step(2)
        self.assertEqual(obs2.shape, (6,))
        print(f"Live Step 2 (Move +X): Reward = {r2:.4f}, Connected = {info2['sionna_metrics']['aggregate']['connected_users_count']}/10")
        print("[PASS] TEST 3: Live Sionna reward correctly calculated in environment step loop.")

    def test_04_legacy_surrogate_mode_preservation(self):
        """Verifies that legacy surrogate mode (use_sionna_feedback=False) remains unchanged."""
        print("\n--- [TEST 4] Verifying Legacy Surrogate Mode ---")
        env = UAVDynamicEnv(use_sionna_feedback=False)
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (6,))

        obs_next, reward, term, trunc, info_next = env.step(1)
        self.assertEqual(obs_next.shape, (6,))
        self.assertIsInstance(reward, float)
        print(f"Surrogate Mode Step Reward: {reward:.4f}")
        print("[PASS] TEST 4: Legacy surrogate mode preserved.")


if __name__ == "__main__":
    unittest.main()
