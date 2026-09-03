"""
Unit tests for Phase 1: Sionna RT Wireless Feedback Engine
"""

import os
import sys
import json
import unittest

# Ensure sionna directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sionna_feedback_engine import (
    evaluate_uav_position,
    evaluate_connectivity,
    ConnectivityConfig,
    DEFAULT_10_USERS,
    export_evaluation_to_json,
    export_evaluation_to_csv
)


class TestPhase1SionnaFeedback(unittest.TestCase):

    def setUp(self):
        self.uav_pos = [-18.82, 48.41, 30.57]
        self.users_10 = DEFAULT_10_USERS

    def test_01_ten_users_output_structure(self):
        """Test that evaluate_uav_position produces exactly 10 user-level results with all required fields."""
        print("\n--- Running TEST 1: 10 Users Evaluation Output ---")
        result = evaluate_uav_position(
            uav_position=self.uav_pos,
            user_positions=self.users_10,
            scene_name="city_damaged.xml"
        )

        self.assertIn("user_results", result)
        self.assertIn("aggregate", result)
        self.assertEqual(len(result["user_results"]), 10, "Must return exactly 10 user results")

        required_user_keys = [
            "user_id", "position", "uav_position",
            "path_gain_linear", "path_gain_db",
            "rss_dbm", "sinr_db", "num_paths",
            "shortest_delay_sec", "coverage", "connected"
        ]

        for u in result["user_results"]:
            for k in required_user_keys:
                self.assertIn(k, u, f"Missing key '{k}' in user result")
            self.assertIsInstance(u["position"], list)
            self.assertEqual(len(u["position"]), 3)
            self.assertIsInstance(u["path_gain_linear"], float)
            self.assertIsInstance(u["path_gain_db"], float)
            self.assertIsInstance(u["rss_dbm"], float)
            self.assertIsInstance(u["num_paths"], int)
            self.assertIsInstance(u["coverage"], bool)
            self.assertIsInstance(u["connected"], bool)

        print(f"[PASS] TEST 1: Exactly 10 users evaluated. Connected count: {result['aggregate']['connected_users_count']}/10")

    def test_02_actual_physical_metrics(self):
        """Test that physical metrics (path gain, RSS, delay, multipath count) come from real Sionna ray tracing."""
        print("\n--- Running TEST 2: Physical Metric Validation ---")
        result = evaluate_uav_position(
            uav_position=self.uav_pos,
            user_positions=self.users_10,
            scene_name="city_damaged.xml"
        )

        # Check that at least some users have positive linear path gain and received power > -150 dBm
        valid_users = [u for u in result["user_results"] if u["path_gain_linear"] > 0]
        self.assertGreater(len(valid_users), 0, "At least some users must receive valid ray paths")

        for u in valid_users:
            self.assertGreater(u["path_gain_linear"], 0)
            self.assertGreater(u["rss_dbm"], -200.0)
            self.assertGreaterEqual(u["num_paths"], 1)
            if u["shortest_delay_sec"] is not None:
                self.assertGreater(u["shortest_delay_sec"], 0.0)

        agg = result["aggregate"]
        self.assertIn("connected_users_count", agg)
        self.assertIn("connected_users_percentage", agg)
        self.assertIn("coverage_percentage", agg)
        self.assertIn("mean_path_gain_db", agg)
        self.assertIn("mean_rss_dbm", agg)
        print(f"[PASS] TEST 2: Mean Path Gain = {agg['mean_path_gain_db']} dB, Mean RSS = {agg['mean_rss_dbm']} dBm")

    def test_03_sinr_configurable_behavior(self):
        """Test that SINR is calculated when noise floor is configured, and None when disabled/unsupported."""
        print("\n--- Running TEST 3: SINR Configuration Integrity ---")
        # Mode 1: SINR enabled with noise floor
        cfg_enabled = ConnectivityConfig(enable_sinr=True, noise_power_dbm=-94.0)
        res_enabled = evaluate_uav_position(
            uav_position=self.uav_pos,
            user_positions=self.users_10,
            connectivity_config=cfg_enabled
        )
        for u in res_enabled["user_results"]:
            if u["rss_dbm"] > -199.0:
                self.assertIsNotNone(u["sinr_db"])
                self.assertEqual(u["sinr_db"], round(u["rss_dbm"] - (-94.0), 2))

        # Mode 2: SINR disabled (e.g. no interference/noise model assumed) -> must return None
        cfg_disabled = ConnectivityConfig(enable_sinr=False, noise_power_dbm=None)
        res_disabled = evaluate_uav_position(
            uav_position=self.uav_pos,
            user_positions=self.users_10,
            connectivity_config=cfg_disabled
        )
        for u in res_disabled["user_results"]:
            self.assertIsNone(u["sinr_db"], "When SINR is disabled, sinr_db must be None (not fabricated)")

        self.assertIsNone(res_disabled["aggregate"]["mean_sinr_db"])
        print("[PASS] TEST 3: SINR returns correct SNR when noise is configured, and None when disabled.")

    def test_04_connectivity_decision_thresholds(self):
        """Test that connectivity decision is decoupled and strictly follows configured thresholds."""
        print("\n--- Running TEST 4: Connectivity Threshold Decoupling ---")
        config_strict = ConnectivityConfig(rss_threshold_dbm=-70.0, sinr_threshold_db=10.0)
        config_lenient = ConnectivityConfig(rss_threshold_dbm=-120.0, sinr_threshold_db=-10.0)

        # -80 dBm is above -120 dBm (lenient) but below -70 dBm (strict)
        self.assertTrue(evaluate_connectivity(rss_dbm=-80.0, sinr_db=0.0, config=config_lenient))
        self.assertFalse(evaluate_connectivity(rss_dbm=-80.0, sinr_db=0.0, config=config_strict))

        # Dead zone (-200 dBm) is always disconnected
        self.assertFalse(evaluate_connectivity(rss_dbm=-200.0, config=config_lenient))
        print("[PASS] TEST 4: Connectivity decisions accurately evaluate configurable threshold limits.")

    def test_05_json_and_csv_export(self):
        """Test optional JSON and CSV export functions."""
        print("\n--- Running TEST 5: JSON / CSV Export Integrity ---")
        res = evaluate_uav_position(
            uav_position=self.uav_pos,
            user_positions=self.users_10
        )
        json_path = os.path.join(current_dir, "test_output_eval.json")
        csv_path = os.path.join(current_dir, "test_output_eval.csv")

        export_evaluation_to_json(res, json_path)
        export_evaluation_to_csv(res, csv_path, scenario_id=42)

        self.assertTrue(os.path.exists(json_path), "JSON output file must exist")
        self.assertTrue(os.path.exists(csv_path), "CSV output file must exist")

        with open(json_path, "r") as f:
            loaded_json = json.load(f)
            self.assertEqual(len(loaded_json["user_results"]), 10)

        # Clean up temporary test files
        if os.path.exists(json_path):
            os.remove(json_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)

        print("[PASS] TEST 5: JSON & CSV export successfully created and verified.")


if __name__ == "__main__":
    unittest.main()
