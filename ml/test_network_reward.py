"""Regression tests for the network-consistent link model and reward.

    cd ml && python test_network_reward.py -v
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "ml", PROJECT_ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import network_model as nm  # noqa: E402
import scenarios  # noqa: E402
from multiuav_env import MultiUAVEnv, OBS_LEN, surrogate_multi_uav  # noqa: E402


def _surrogate_net(uavs, users):
    rf = surrogate_multi_uav(uavs, users, connectivity_config=None)
    return rf, nm.evaluate_network(rf, uavs, users)


class TestNetworkLinkModel(unittest.TestCase):
    def test_surrogate_reproduces_ns3_received_power(self):
        """Free-space RF has no excess loss, so the network RSS must equal
        ns-3's LogDistancePropagationLossModel default exactly."""
        for d in (5.0, 40.0, 120.0, 300.0):
            uavs, users = [[0.0, 0.0, 30.0]], [[d, 0.0, 1.5]]
            _, net = _surrogate_net(uavs, users)
            dist = math.dist(uavs[0], users[0])
            expected = 30.0 - (46.6777 + 30.0 * math.log10(dist))
            self.assertAlmostEqual(net["per_user"][0]["net_rss_dbm"], expected, places=1)

    def test_obstruction_loss_from_rf_is_kept(self):
        d, f = 100.0, 2.1e9
        fs = -nm.free_space_loss_db(d, f)
        self.assertAlmostEqual(nm.network_rss_dbm(fs - 15.0, d, f), nm.network_rss_dbm(fs, d, f) - 15.0, places=6)
        # multipath gain above free space is not credited (ns-3 has none)
        self.assertAlmostEqual(nm.network_rss_dbm(fs + 6.0, d, f), nm.network_rss_dbm(fs, d, f), places=6)
        self.assertEqual(nm.network_rss_dbm(-200.0, d, f), -200.0)

    def test_usable_range_matches_measured_ns3_cutoff(self):
        """ns-3 sweep (UAV at 40 m): 130 m horizontal delivered, 150 m did not."""
        for horiz, usable in ((130.0, True), (150.0, False)):
            _, net = _surrogate_net([[0.0, 0.0, 40.0]], [[horiz, 0.0, 1.5]])
            self.assertEqual(net["per_user"][0]["usable"], usable, horiz)

    def test_rate_tables(self):
        self.assertEqual(nm.mcs_rate_mbps(0.0), 0.0)
        self.assertEqual(nm.mcs_rate_mbps(50.0), 65.0)
        snrs = np.linspace(-5, 40, 200)
        smooth = [nm.smooth_rate_mbps(s) for s in snrs]
        self.assertTrue(all(b >= a for a, b in zip(smooth, smooth[1:])), "smooth rate must be monotone")
        self.assertLessEqual(max(smooth), nm.MAX_RATE_MBPS)
        for thr, rate in zip(nm.HT_SNR_THRESH_DB, nm.HT_RATES_MBPS):
            self.assertAlmostEqual(nm.smooth_rate_mbps(thr + 1e-9), rate, places=3)
        # between thresholds the smooth rate never exceeds the next MCS ns-3 could reach
        for s in snrs:
            nxt = nm.HT_RATES_MBPS[np.searchsorted(nm.HT_SNR_THRESH_DB, s)] \
                if s < nm.HT_SNR_THRESH_DB[-1] else nm.MAX_RATE_MBPS
            self.assertLessEqual(nm.smooth_rate_mbps(s), nxt + 1e-9)

    def test_airtime_saturation(self):
        self.assertGreater(nm.airtime_us(6.5), nm.airtime_us(26.0))
        self.assertGreater(nm.airtime_us(26.0), nm.airtime_us(65.0))
        uavs = [[0.0, 0.0, 30.0]]
        few = [[float(i), 0.0, 1.5] for i in range(10)]
        many = [[float(i % 10), float(i // 10), 1.5] for i in range(50)]
        self.assertEqual(_surrogate_net(uavs, few)[1]["aggregate"]["served_fraction"], 1.0)
        sat = _surrogate_net(uavs, many)[1]["aggregate"]
        self.assertGreater(sat["airtime_utilisation"], 1.0)
        self.assertLess(sat["served_fraction"], 1.0)
        self.assertAlmostEqual(sat["predicted_pdr"], sat["served_fraction"], places=9)


class TestNetworkReward(unittest.TestCase):
    def _env(self, sid="wide_area", n=30, **kw):
        env = MultiUAVEnv(num_uavs=3, num_users=n, scenario=sid, mode="surrogate", seed=7,
                          reward_mode="network", **kw)
        env.reset(seed=7)
        return env

    def test_default_reward_mode_is_legacy(self):
        env = MultiUAVEnv(num_uavs=3, num_users=20, scenario="dense_urban", mode="surrogate", seed=3)
        self.assertEqual(env.reward_mode, "legacy")
        with self.assertRaises(ValueError):
            MultiUAVEnv(reward_mode="bogus")

    def test_legacy_mode_unaffected_by_network_fields(self):
        a = MultiUAVEnv(num_uavs=3, num_users=20, scenario="dense_urban", mode="surrogate", seed=3)
        b = MultiUAVEnv(num_uavs=3, num_users=20, scenario="dense_urban", mode="surrogate", seed=3,
                        random_start_prob=0.0)
        oa, _ = a.reset(seed=3); ob, _ = b.reset(seed=3)
        np.testing.assert_array_equal(oa, ob)
        ra = a.step([1, 2, 3, 0, 0])[1]; rb = b.step([1, 2, 3, 0, 0])[1]
        self.assertEqual(ra, rb)

    def test_obs_shape_and_bounds_in_network_mode(self):
        env = self._env()
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs.shape, (OBS_LEN,))
        self.assertTrue(env.observation_space.contains(obs))
        obs, *_ = env.step([0, 1, 2, 3, 4])
        self.assertTrue(env.observation_space.contains(obs))

    def test_stacking_uavs_is_penalised(self):
        env = self._env()
        spread = env.state_reward()
        env.uav_positions[:] = env.uav_positions[0]
        env._refresh()
        self.assertLess(env.state_reward(), spread - 0.1)

    def test_abandoning_users_is_not_rewarded(self):
        env = self._env()
        base, base_conn = env.state_reward(), env.last_net["aggregate"]["net_connected_count"]
        reg = scenarios.get_scenario("wide_area")["uav_region"]
        env.uav_positions[2, :2] = [reg["x"][1], reg["y"][1]]   # park UAV 3 in an empty corner
        env._refresh()
        self.assertLess(env.last_net["aggregate"]["net_connected_count"], base_conn)
        self.assertLess(env.state_reward(), base)

    def test_reaching_an_unserved_user_increases_reward(self):
        """An idle UAV moving onto an out-of-range user must raise the reward.
        (Moving a UAV that is serving others need not - it abandons them.)"""
        env = self._env(n=3)
        env.uav_positions = np.array([[0.0, 0.0, 40.0], [60.0, 0.0, 40.0], [-60.0, 0.0, 40.0]])
        env.user_positions = np.array([[5.0, 0.0, 1.5], [55.0, 0.0, 1.5], [0.0, 170.0, 1.5]])
        env._refresh()
        self.assertEqual([p["usable"] for p in env.last_net["per_user"]], [True, True, False])
        before = env.state_reward()
        env.uav_positions[2, :2] = [0.0, 165.0]          # UAV 3 served nobody
        env._refresh()
        self.assertEqual([p["usable"] for p in env.last_net["per_user"]], [True, True, True])
        self.assertGreater(env.state_reward(), before)

    def test_reward_bounds(self):
        env = self._env(n=50)
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.integers(0, 5, size=5)
            _, r, *_ = env.step(a)
            self.assertGreaterEqual(r, -(env.w["w_overlap"] + env.w["w_move"] + env.w["w_prog"]))
            self.assertLessEqual(r, env.w["w_conn"] + env.w["w_pdr"] + env.w["w_rate"] + env.w["w_prog"])

    def test_random_start_prob(self):
        env = self._env(random_start_prob=1.0)
        fixed = scenarios.initial_uav_positions(3, "wide_area")
        self.assertFalse(np.allclose(env.uav_positions, fixed))
        np.testing.assert_allclose(env.uav_positions[:, 2], fixed[:, 2])   # altitude unchanged
        reg = scenarios.get_scenario("wide_area")["uav_region"]
        self.assertTrue(np.all(env.uav_positions[:, 0] >= reg["x"][0]))
        self.assertTrue(np.all(env.uav_positions[:, 0] <= reg["x"][1]))


class TestNetworkTrainingConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((PROJECT_ROOT / "configs" / "training_network.json").read_text())

    def test_train_and_heldout_are_disjoint(self):
        self.assertFalse(set(self.cfg["train"]["scenarios"]) & set(self.cfg["eval"]["scenarios"]))
        self.assertEqual(set(self.cfg["eval"]["scenarios"]), set(scenarios.eval_scenarios()))
        self.assertEqual(self.cfg["train"]["reward_mode"], "network")

    def test_extra_scenarios_are_training_only(self):
        extra = self.cfg["extra_train_scenarios"]
        self.assertFalse(set(extra) & set(scenarios.scenario_ids()))
        scenarios.register_scenarios(extra)
        for sid in extra:
            self.assertEqual(scenarios.get_scenario(sid), extra[sid])
            self.assertNotIn(sid, scenarios.scenario_ids())       # dashboard list unchanged
            self.assertIn(sid, self.cfg["train"]["scenarios"])
            self.assertNotEqual(extra[sid]["user_distribution"],
                                scenarios.get_scenario("wide_area")["user_distribution"])
        with self.assertRaises(ValueError):
            scenarios.register_scenarios({"wide_area": extra[next(iter(extra))]})


FAKE_NS3 = """#!/usr/bin/env python3
print("ML UAV Position: (0, 0, 30)")
print("\\n====================================\\n       UAV NETWORK RESULTS\\n====================================")
print("\\nFlow: 10.1.1.2 -> 10.1.1.1\\nPackets Sent: 800\\nPackets Received: 800\\nThroughput: 0.432 Mbps\\n"
      "Average Delay: 2 ms\\nPacket Loss: 0 %")
print("\\nFlow: 10.1.1.4 -> 10.1.1.1\\nPackets Sent: 800\\nPackets Received: 400\\nThroughput: 0.216 Mbps\\n"
      "Average Delay: 10 ms\\nPacket Loss: 50 %")
print("\\nFlow: 10.1.1.3 -> 10.1.1.1\\nPackets Sent: 800\\nNo packets received.")
print("\\n====================================")
"""


class TestEvaluatorNs3Parsing(unittest.TestCase):
    def test_metrics_include_failed_users(self):
        import evaluate_network as ev
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "fake_ns3"
            fake.write_text(FAKE_NS3)
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            handoff = {"num_uavs": 1, "uav_positions": [[0, 0, 30]], "num_users": 3,
                       "user_positions": [[0, 0, 1.5]] * 3}
            r = ev.run_ns3(fake, handoff)
        self.assertEqual(r["per_user_rx"], [800, 0, 400])          # ordered by user, not by flow
        self.assertEqual(r["delivering_users"], 2)
        self.assertEqual(r["users_pdr_ge_90"], 1)
        self.assertAlmostEqual(r["pdr"], 1200 / 2400)
        self.assertAlmostEqual(r["mean_latency_ms"], (2 * 800 + 10 * 400) / 1200)
        per_user = [800 * 540 * 8 / 8.0 / 1e6, 0.0, 400 * 540 * 8 / 8.0 / 1e6]
        self.assertAlmostEqual(r["mean_user_throughput_mbps"], float(np.mean(per_user)))
        self.assertEqual(r["min_user_throughput_mbps"], 0.0)

    def test_paired_statistics(self):
        import evaluate_network as ev
        p = ev.paired([1, 2, 3, 3], [0, 2, 1, 4])
        self.assertEqual((p["wins"], p["ties"], p["losses"]), (2, 1, 1))
        self.assertEqual(ev.paired([1, 1], [1, 1])["wilcoxon_p"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
