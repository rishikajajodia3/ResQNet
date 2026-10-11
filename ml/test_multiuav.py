"""Test suite for the ResQNet multi-UAV / multi-scenario system.

Run:  python ml/test_multiuav.py
Slow Sionna / NS-3 tests are skipped unless RESQNET_SLOW_TESTS=1.

Tests are added phase by phase. Each phase's gate is the subset of this
file that its classes cover.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "ml", PROJECT_ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SLOW = os.environ.get("RESQNET_SLOW_TESTS") == "1"

import scenarios  # noqa: E402
import multiuav_io as mio  # noqa: E402


# ======================================================================
# PHASE 2 - scenario system & deterministic generation
# ======================================================================
class TestScenarioConfig(unittest.TestCase):
    def test_five_scenarios_present(self):
        ids = scenarios.scenario_ids()
        for expected in (
            "open_area",
            "dense_urban",
            "blocked_urban",
            "clustered_survivors",
            "wide_area",
        ):
            self.assertIn(expected, ids)

    def test_each_scenario_well_formed(self):
        for sid in scenarios.scenario_ids():
            s = scenarios.get_scenario(sid)
            self.assertIn("scene", s)
            self.assertIn(s["user_distribution"], scenarios.VALID_DISTRIBUTIONS)
            for reg_key in ("user_region", "uav_region"):
                reg = s[reg_key]
                self.assertLess(reg["x"][0], reg["x"][1])
                self.assertLess(reg["y"][0], reg["y"][1])
            self.assertLess(s["uav_region"]["z"][0], s["uav_region"]["z"][1])

    def test_train_eval_disjoint(self):
        scenarios.assert_train_eval_disjoint()
        self.assertEqual(set(scenarios.train_scenarios()),
                         {"open_area", "dense_urban", "clustered_survivors"})
        self.assertEqual(set(scenarios.eval_scenarios()),
                         {"blocked_urban", "wide_area"})


class TestUserReproducibility(unittest.TestCase):
    def test_same_seed_same_users(self):
        for sid in scenarios.scenario_ids():
            for n in (10, 20, 30, 50):
                a = scenarios.generate_users(sid, n, seed=42)
                b = scenarios.generate_users(sid, n, seed=42)
                np.testing.assert_array_equal(a, b, f"{sid} n={n} not reproducible")
                self.assertEqual(a.shape, (n, 3))

    def test_different_seed_different_users(self):
        for sid in scenarios.scenario_ids():
            a = scenarios.generate_users(sid, 30, seed=42)
            c = scenarios.generate_users(sid, 30, seed=7)
            self.assertFalse(np.array_equal(a, c), f"{sid}: seed change had no effect")

    def test_users_inside_region(self):
        for sid in scenarios.scenario_ids():
            reg = scenarios.get_scenario(sid)["user_region"]
            u = scenarios.generate_users(sid, 50, seed=123)
            self.assertTrue((u[:, 0] >= reg["x"][0] - 1e-6).all())
            self.assertTrue((u[:, 0] <= reg["x"][1] + 1e-6).all())
            self.assertTrue((u[:, 1] >= reg["y"][0] - 1e-6).all())
            self.assertTrue((u[:, 1] <= reg["y"][1] + 1e-6).all())
            np.testing.assert_allclose(u[:, 2], scenarios.user_height())

    def test_all_distributions_work(self):
        for dist in scenarios.VALID_DISTRIBUTIONS:
            u = scenarios.generate_users("blocked_urban", 40, seed=5, distribution=dist)
            self.assertEqual(u.shape, (40, 3))
            self.assertTrue(np.isfinite(u).all())

    @staticmethod
    def _mean_nn_dist(pts: np.ndarray) -> float:
        xy = pts[:, :2]
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).mean())

    def test_clustered_actually_clusters(self):
        # In a clustered layout points bunch together, so the mean
        # nearest-neighbour distance is much smaller than for a uniform draw
        # over the same region.
        clu = scenarios.generate_users("clustered_survivors", 60, seed=42)
        uni = scenarios.generate_users("clustered_survivors", 60, seed=42,
                                       distribution="uniform")
        self.assertLess(self._mean_nn_dist(clu), 0.6 * self._mean_nn_dist(uni))


class TestInitialUavPositions(unittest.TestCase):
    def test_counts_and_bounds(self):
        for sid in scenarios.scenario_ids():
            reg = scenarios.get_scenario(sid)["uav_region"]
            for k in (1, 2, 3, 5):
                p = scenarios.initial_uav_positions(k, sid)
                self.assertEqual(p.shape, (k, 3))
                self.assertTrue((p[:, 0] >= reg["x"][0]).all() and (p[:, 0] <= reg["x"][1]).all())
                self.assertTrue((p[:, 1] >= reg["y"][0]).all() and (p[:, 1] <= reg["y"][1]).all())
                self.assertTrue((p[:, 2] >= reg["z"][0]).all() and (p[:, 2] <= reg["z"][1]).all())

    def test_no_duplicate_positions_for_k_ge_2(self):
        for sid in scenarios.scenario_ids():
            for k in (2, 3, 4, 5):
                p = scenarios.initial_uav_positions(k, sid)
                dists = [
                    np.linalg.norm(p[i] - p[j])
                    for i in range(k)
                    for j in range(i + 1, k)
                ]
                self.assertGreater(min(dists), 15.0,
                                   f"{sid} k={k}: UAVs too close ({min(dists):.1f} m)")

    def test_deterministic(self):
        a = scenarios.initial_uav_positions(3, "wide_area")
        b = scenarios.initial_uav_positions(3, "wide_area")
        np.testing.assert_array_equal(a, b)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            scenarios.initial_uav_positions(6, "open_area")
        with self.assertRaises(ValueError):
            scenarios.initial_uav_positions(0, "open_area")


# ======================================================================
# PHASE 3 - multi-UAV data structures (K=1 must stay identical)
# ======================================================================
LEGACY_HANDOFF_KEYS = {"num_uavs", "uav_positions", "num_users", "user_positions"}
LEGACY_STEP_KEYS = {
    "step", "action", "position", "connected_users", "coverage_percent",
    "mean_path_gain_db", "mean_rss_dbm", "mean_sinr_db", "reward", "user_results",
}
LEGACY_TRAJ_KEYS = {
    "scene", "mode", "num_uavs", "num_users", "num_steps", "total_reward",
    "initial_position", "final_position", "trajectory",
}


def _fake_user_results(n, uav_positions):
    out = []
    for i in range(n):
        out.append({
            "user_id": f"user_{i+1}",
            "position": [float(i), float(-i), 1.5],
            "rss_dbm": -60.0, "sinr_db": 34.0, "path_gain_db": -90.0,
            "num_paths": 3, "shortest_delay_sec": 1e-7,
            "coverage": True, "connected": True,
        })
    return out


class TestHandoffSchemaBackwardCompat(unittest.TestCase):
    def test_k1_handoff_identical_to_legacy(self):
        uavs = [[-29.31, 22.64, 25.97]]
        users = [[float(i), float(i), 1.5] for i in range(10)]
        d = mio.build_uav_positions(uavs, users)  # no associations
        self.assertEqual(set(d.keys()), LEGACY_HANDOFF_KEYS)
        self.assertEqual(d["num_uavs"], 1)
        self.assertEqual(len(d["uav_positions"]), 1)
        self.assertEqual(len(d["uav_positions"][0]), 3)
        self.assertEqual(d["num_users"], 10)
        # trivial all-zero associations must also collapse to legacy form
        d2 = mio.build_uav_positions(uavs, users, associations=[0] * 10)
        self.assertEqual(set(d2.keys()), LEGACY_HANDOFF_KEYS)

    def test_k1_step_identical_to_legacy(self):
        step = mio.build_trajectory_step(
            step=1, num_uavs=1, uav_positions=[[1.0, 2.0, 25.0]], actions=[2],
            connected_users=10, coverage_percent=100.0, mean_path_gain_db=-73.0,
            mean_rss_dbm=-43.0, mean_sinr_db=51.0, reward=0.68,
            user_results=_fake_user_results(10, None),
        )
        self.assertEqual(set(step.keys()), LEGACY_STEP_KEYS)
        self.assertEqual(step["action"], 2)
        self.assertEqual(step["position"], [1.0, 2.0, 25.0])

    def test_k1_trajectory_identical_to_legacy(self):
        steps = [mio.build_trajectory_step(
            step=s, num_uavs=1, uav_positions=[[0.0, 0.0, 25.0]], actions=[None if s == 0 else 1],
            connected_users=10, coverage_percent=100.0, mean_path_gain_db=-73.0,
            mean_rss_dbm=-43.0, mean_sinr_db=51.0, reward=0.0,
            user_results=_fake_user_results(10, None),
        ) for s in range(3)]
        doc = mio.build_trajectory(
            scene="city_damaged.xml", mode="sionna", num_uavs=1, num_users=10,
            num_steps=2, total_reward=1.0,
            initial_positions=[[0.0, 0.0, 25.0]], final_positions=[[0.0, 0.0, 25.0]],
            steps=steps,
        )
        self.assertEqual(set(doc.keys()), LEGACY_TRAJ_KEYS)


class TestHandoffSchemaMultiUav(unittest.TestCase):
    def test_k3_handoff_has_list_and_associations(self):
        uavs = [[-30, 0, 25], [0, 0, 25], [30, 0, 25]]
        users = [[float(i * 5 - 20), 0.0, 1.5] for i in range(9)]
        assoc = mio.nearest_uav_association(uavs, users)
        d = mio.build_uav_positions(uavs, users, associations=assoc)
        self.assertEqual(d["num_uavs"], 3)
        self.assertEqual(len(d["uav_positions"]), 3)
        self.assertEqual(d["num_users"], 9)
        self.assertIn("associations", d)
        self.assertEqual(len(d["associations"]), 9)
        self.assertTrue(all(0 <= a < 3 for a in d["associations"]))

    def test_k3_step_has_lists_plus_scalar_mirror(self):
        step = mio.build_trajectory_step(
            step=2, num_uavs=3,
            uav_positions=[[1, 1, 25], [2, 2, 25], [3, 3, 25]], actions=[0, 2, 4],
            connected_users=25, coverage_percent=83.3, mean_path_gain_db=-80.0,
            mean_rss_dbm=-50.0, mean_sinr_db=20.0, reward=0.5,
            user_results=_fake_user_results(25, None),
        )
        self.assertEqual(step["uav_positions"], [[1, 1, 25], [2, 2, 25], [3, 3, 25]])
        self.assertEqual(step["actions"], [0, 2, 4])
        self.assertEqual(step["position"], [1, 1, 25])   # scalar mirror = UAV 0
        self.assertEqual(step["action"], 0)

    def test_read_uav_positions_fills_missing_association(self):
        import tempfile
        uavs = [[-30, 0, 25], [30, 0, 25]]
        users = [[-25, 0, 1.5], [25, 0, 1.5]]
        legacy = {"num_uavs": 2, "uav_positions": uavs, "num_users": 2,
                  "user_positions": users}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            import json as _j
            _j.dump(legacy, f)
            fn = f.name
        got = mio.read_uav_positions(fn)
        self.assertEqual(got["associations"], [0, 1])

    def test_validate_consistency_detects_mismatch(self):
        uavs = [[0, 0, 25], [50, 50, 25]]
        users = [[1, 1, 1.5], [2, 2, 1.5]]
        handoff = mio.build_uav_positions(uavs, users, associations=[0, 1])
        steps = [mio.build_trajectory_step(
            step=0, num_uavs=2, uav_positions=uavs, actions=[None, None],
            connected_users=2, coverage_percent=100.0, mean_path_gain_db=-70.0,
            mean_rss_dbm=-40.0, mean_sinr_db=54.0, reward=0.0,
            user_results=[{"user_id": "user_1", "position": [1, 1, 1.5]},
                          {"user_id": "user_2", "position": [2, 2, 1.5]}],
        )]
        traj_ok = mio.build_trajectory(
            scene="s", mode="surrogate", num_uavs=2, num_users=2, num_steps=0,
            total_reward=0.0, initial_positions=uavs, final_positions=uavs, steps=steps,
        )
        self.assertEqual(mio.validate_consistency(handoff, traj_ok), [])

        # break UAV 1's final position
        bad = mio.build_trajectory(
            scene="s", mode="surrogate", num_uavs=2, num_users=2, num_steps=0,
            total_reward=0.0, initial_positions=uavs,
            final_positions=[[0, 0, 25], [99, 99, 25]], steps=steps,
        )
        self.assertTrue(mio.validate_consistency(handoff, bad))

    def test_validate_detects_duplicate_uavs(self):
        uavs = [[10, 10, 25], [10, 10, 25]]
        users = [[0, 0, 1.5]]
        handoff = mio.build_uav_positions(uavs, users, associations=[0])
        steps = [mio.build_trajectory_step(
            step=0, num_uavs=2, uav_positions=uavs, actions=[None, None],
            connected_users=1, coverage_percent=100.0, mean_path_gain_db=-70.0,
            mean_rss_dbm=-40.0, mean_sinr_db=54.0, reward=0.0,
            user_results=[{"user_id": "user_1", "position": [0, 0, 1.5]}],
        )]
        traj = mio.build_trajectory(
            scene="s", mode="surrogate", num_uavs=2, num_users=1, num_steps=0,
            total_reward=0.0, initial_positions=uavs, final_positions=uavs, steps=steps,
        )
        probs = mio.validate_consistency(handoff, traj)
        self.assertTrue(any("same position" in p for p in probs))


# ======================================================================
# PHASE 4 - Sionna multi-UAV evaluation
# ======================================================================
class TestSionnaApiSurface(unittest.TestCase):
    """Non-slow: the legacy single-UAV entry point must be untouched."""

    def test_evaluate_uav_position_signature_unchanged(self):
        import inspect
        import sionna_feedback_engine as sfe
        params = list(inspect.signature(sfe.evaluate_uav_position).parameters)
        self.assertEqual(
            params,
            ["uav_position", "user_positions", "scene_name", "freq",
             "tx_power_dbm", "connectivity_config", "seed"],
        )
        self.assertTrue(hasattr(sfe, "evaluate_multi_uav"))
        self.assertTrue(hasattr(sfe, "DEFAULT_10_USERS"))
        self.assertEqual(len(sfe.DEFAULT_10_USERS), 10)

    def test_missing_scene_raises_instead_of_falling_back(self):
        """A missing scene used to be silently replaced by simple_street_canyon,
        producing metrics for the wrong geometry under the requested name."""
        import sionna_feedback_engine as sfe
        import sionna_pipeline as sp
        with self.assertRaises(FileNotFoundError):
            sfe.safe_load_scene_robust("no_such_scene.xml")
        with self.assertRaises(FileNotFoundError):
            sp.safe_load_scene("no_such_scene.xml")
        with self.assertRaises(FileNotFoundError):
            sfe.evaluate_multi_uav([[0.0, 0.0, 30.0]], scene_name="no_such_scene.xml")

    def test_scenario_scenes_load_the_city_mesh(self):
        """Every configured scenario scene loads the city mesh, not the placeholder."""
        import sionna_feedback_engine as sfe
        import sionna_pipeline as sp
        for name in sorted({s["scene"] for s in scenarios.load_scenarios()["scenarios"].values()}):
            scene = sfe.safe_load_scene_robust(name)
            self.assertEqual(list(scene.objects), ["merged-shapes"], name)
        # relative path used by sionna_pipeline's __main__, run from the repo root
        cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            scene = sp.safe_load_scene("scenes/city_damaged.xml")
        finally:
            os.chdir(cwd)
        self.assertEqual(list(scene.objects), ["merged-shapes"])


@unittest.skipUnless(SLOW, "set RESQNET_SLOW_TESTS=1 to run Sionna ray tracing")
class TestSionnaMultiUav(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sionna_feedback_engine as sfe
        cls.sfe = sfe
        cls.users30 = scenarios.generate_users("blocked_urban", 30, seed=42).tolist()
        cls.uavs3 = scenarios.initial_uav_positions(3, "blocked_urban").tolist()

    def test_sionna_multiuav_eval(self):
        """PHASE 4 GATE: real multi-UAV Sionna ray tracing, 3 UAVs x 30 users."""
        r = self.sfe.evaluate_multi_uav(
            self.uavs3, user_positions=self.users30,
            scene_name="city_damaged.xml", seed=42,
        )
        # real ray tracing over K transmitters
        self.assertEqual(r["num_uavs"], 3)
        self.assertEqual(r["num_users"], 30)
        self.assertEqual(len(r["user_results"]), 30)
        self.assertEqual(r["interference_model"], "co_channel")

        for u in r["user_results"]:
            # per-user, per-UAV metrics preserved (not averaged)
            self.assertEqual(len(u["per_uav"]), 3)
            for pu in u["per_uav"]:
                for k in ("uav_id", "rss_dbm", "snr_db", "path_gain_db",
                          "num_paths", "shortest_delay_sec"):
                    self.assertIn(k, pu)
            # best-RSS serving association
            self.assertIn(u["serving_uav"], (1, 2, 3))
            serving_rss = u["per_uav"][u["serving_uav"] - 1]["rss_dbm"]
            self.assertEqual(serving_rss, max(pu["rss_dbm"] for pu in u["per_uav"]))
            self.assertEqual(u["best_rss_dbm"], serving_rss)
            # both SNR and SINR present
            self.assertIn("best_snr_db", u)
            self.assertIn("best_sinr_db", u)
            self.assertIsNotNone(u["best_snr_db"])
            self.assertIsNotNone(u["best_sinr_db"])

        agg = r["aggregate"]
        # users_per_uav aggregate: keyed per UAV, sums to connected users
        self.assertEqual(set(agg["users_per_uav"].keys()), {"uav_1", "uav_2", "uav_3"})
        self.assertEqual(sum(agg["users_per_uav"].values()), agg["connected_users_count"])
        self.assertLessEqual(agg["connected_users_count"], 30)
        for key in ("mean_best_rss_dbm", "mean_best_snr_db",
                    "mean_best_sinr_db", "mean_best_path_gain_db"):
            self.assertIn(key, agg)

    def test_1uav_sinr_equals_snr(self):
        r = self.sfe.evaluate_multi_uav(
            [self.uavs3[0]], user_positions=self.users30,
            scene_name="city_damaged.xml", seed=42,
        )
        for u in r["user_results"]:
            if u["best_snr_db"] is not None and u["best_sinr_db"] is not None:
                self.assertAlmostEqual(u["best_sinr_db"], u["best_snr_db"], delta=0.05)

    def test_cochannel_sinr_not_above_snr(self):
        r = self.sfe.evaluate_multi_uav(
            self.uavs3, user_positions=self.users30,
            scene_name="city_damaged.xml", seed=42, interference_model="co_channel",
        )
        for u in r["user_results"]:
            if u["best_snr_db"] is not None and u["best_sinr_db"] is not None:
                self.assertLessEqual(u["best_sinr_db"], u["best_snr_db"] + 0.05)

    def test_orthogonal_model_gives_snr(self):
        r = self.sfe.evaluate_multi_uav(
            self.uavs3, user_positions=self.users30,
            scene_name="city_damaged.xml", seed=42, interference_model="orthogonal",
        )
        for u in r["user_results"]:
            if u["best_snr_db"] is not None and u["best_sinr_db"] is not None:
                self.assertAlmostEqual(u["best_sinr_db"], u["best_snr_db"], delta=0.05)

    def test_reproducible(self):
        a = self.sfe.evaluate_multi_uav(self.uavs3, user_positions=self.users30,
                                        scene_name="city_damaged.xml", seed=42)
        b = self.sfe.evaluate_multi_uav(self.uavs3, user_positions=self.users30,
                                        scene_name="city_damaged.xml", seed=42)
        self.assertEqual(a["aggregate"]["connected_users_count"],
                         b["aggregate"]["connected_users_count"])
        self.assertEqual([u["serving_uav"] for u in a["user_results"]],
                         [u["serving_uav"] for u in b["user_results"]])


# ======================================================================
# PHASE 5 - multi-UAV PPO environment
# ======================================================================
class TestMultiUavEnv(unittest.TestCase):
    def setUp(self):
        from multiuav_env import MultiUAVEnv, OBS_LEN, MAX_UAVS
        self.MultiUAVEnv = MultiUAVEnv
        self.OBS_LEN = OBS_LEN
        self.MAX_UAVS = MAX_UAVS

    def test_obs_fixed_length_across_users_and_uavs(self):
        for n_users in (10, 20, 30, 50):
            for n_uavs in (2, 3, 5):
                env = self.MultiUAVEnv(
                    num_uavs=n_uavs, num_users=n_users,
                    scenario="blocked_urban", mode="surrogate", seed=1,
                    max_steps=4,
                )
                obs, info = env.reset(seed=1)
                self.assertEqual(obs.shape, (self.OBS_LEN,),
                                 f"n_users={n_users} n_uavs={n_uavs}")
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(info["num_uavs"], n_uavs)
                self.assertEqual(info["num_users"], n_users)
                self.assertEqual(len(info["eval"]["user_results"]), n_users)

    def test_action_space_is_fixed_multidiscrete(self):
        env = self.MultiUAVEnv(num_uavs=3, num_users=20, mode="surrogate")
        self.assertEqual(list(env.action_space.nvec), [5] * self.MAX_UAVS)

    def test_step_runs_and_terminates(self):
        env = self.MultiUAVEnv(num_uavs=3, num_users=25, mode="surrogate",
                               scenario="open_area", max_steps=5, seed=2)
        obs, _ = env.reset(seed=2)
        rng = np.random.default_rng(0)
        done = False
        steps = 0
        while not done and steps < 20:
            a = rng.integers(0, 5, size=self.MAX_UAVS)
            obs, r, term, trunc, info = env.step(a)
            self.assertEqual(obs.shape, (self.OBS_LEN,))
            self.assertIsInstance(r, float)
            self.assertEqual(len(info["actions"]), 3)
            done = term or trunc
            steps += 1
        self.assertTrue(done)
        self.assertEqual(steps, 5)

    def test_reproducible_reset(self):
        e1 = self.MultiUAVEnv(num_uavs=3, num_users=30, scenario="clustered_survivors",
                              mode="surrogate", seed=7)
        e2 = self.MultiUAVEnv(num_uavs=3, num_users=30, scenario="clustered_survivors",
                              mode="surrogate", seed=7)
        o1, i1 = e1.reset(seed=7)
        o2, i2 = e2.reset(seed=7)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(
            np.array(i1["eval"]["uav_positions"]),
            np.array(i2["eval"]["uav_positions"]),
        )

    def test_overlap_penalty_responds_to_spread(self):
        env = self.MultiUAVEnv(num_uavs=3, num_users=30, scenario="open_area",
                               mode="surrogate", seed=3, max_steps=8)
        env.reset(seed=3)
        # force all UAVs to the same spot -> maximal overlap penalty
        env.uav_positions[:] = env.uav_positions[0]
        env.last_eval = env._evaluate()
        stacked = env._overlap_penalty()
        # spread them far apart
        env.uav_positions[0] += np.array([120.0, 0, 0])
        env.uav_positions[1] += np.array([0, 120.0, 0])
        spread = env._overlap_penalty()
        self.assertGreater(stacked, spread)
        self.assertAlmostEqual(spread, 0.0, places=6)

    def test_surrogate_matches_multiuav_result_shape(self):
        from multiuav_env import surrogate_multi_uav
        import sionna_feedback_engine as sfe
        uavs = scenarios.initial_uav_positions(3, "open_area").tolist()
        users = scenarios.generate_users("open_area", 15, seed=9).tolist()
        s = surrogate_multi_uav(uavs, users, connectivity_config=sfe.ConnectivityConfig())
        for key in ("uav_positions", "num_uavs", "num_users", "interference_model",
                    "user_results", "aggregate"):
            self.assertIn(key, s)
        self.assertEqual(len(s["user_results"]), 15)
        for u in s["user_results"]:
            self.assertEqual(len(u["per_uav"]), 3)
            self.assertIn("best_snr_db", u)
            self.assertIn("best_sinr_db", u)
        self.assertIn("users_per_uav", s["aggregate"])
        self.assertEqual(s["scene_name"], "surrogate:free_space")


class TestMultiUavTrainingSmoke(unittest.TestCase):
    def test_ppo_learns_and_predicts_on_multidiscrete(self):
        from stable_baselines3 import PPO
        from multiuav_env import MultiUAVEnv, MAX_UAVS, OBS_LEN
        env = MultiUAVEnv(num_uavs=3, num_users=20, scenario="dense_urban",
                          mode="surrogate", seed=11, max_steps=8)
        model = PPO("MlpPolicy", env, seed=11, n_steps=128, batch_size=64, verbose=0)
        model.learn(total_timesteps=256)
        obs, _ = env.reset(seed=11)
        action, _ = model.predict(obs, deterministic=True)
        self.assertEqual(np.asarray(action).shape, (MAX_UAVS,))
        self.assertTrue(all(0 <= int(a) < 5 for a in np.asarray(action).ravel()))
        # one env step with the policy action
        o, r, term, trunc, info = env.step(action)
        self.assertEqual(o.shape, (OBS_LEN,))
        self.assertEqual(len(info["actions"]), 3)

    def test_training_sampler_covers_only_train_scenarios(self):
        from train_multiuav import build_sampler
        cfg = scenarios.load_training_config()["train"]
        sampler, combos = build_sampler(cfg, base_seed=42)
        seen = {sampler(i)["scenario"] for i in range(200)}
        self.assertTrue(seen.issubset(set(scenarios.train_scenarios())))
        self.assertNotIn("blocked_urban", seen)   # held out
        self.assertNotIn("wide_area", seen)       # held out


# ======================================================================
# PHASE 6 - JSON handoff wiring in run_ppo.py
# ======================================================================
class TestJsonHandoffMultiUav(unittest.TestCase):
    def test_multiuav_run_writes_consistent_extended_json(self):
        import tempfile, json as _json
        from run_ppo import run_multiuav_episode
        import multiuav_io as mio

        with tempfile.TemporaryDirectory() as td:
            up = Path(td) / "uav_positions.json"
            tp = Path(td) / "ppo_trajectory.json"
            run_multiuav_episode(
                scenario="dense_urban", num_uavs=3, num_users=30,
                mode="surrogate", max_steps=5, seed=42,
                out_uav_positions=str(up), out_trajectory=str(tp),
            )
            handoff = _json.loads(up.read_text())
            traj = _json.loads(tp.read_text())

        self.assertEqual(handoff["num_uavs"], 3)
        self.assertEqual(len(handoff["uav_positions"]), 3)
        self.assertEqual(handoff["num_users"], 30)
        self.assertEqual(len(handoff["user_positions"]), 30)
        self.assertIn("associations", handoff)
        self.assertEqual(len(handoff["associations"]), 30)
        self.assertTrue(all(0 <= a < 3 for a in handoff["associations"]))

        self.assertEqual(traj["num_uavs"], 3)
        self.assertEqual(traj["num_users"], 30)
        self.assertEqual(traj["scenario"], "dense_urban")
        self.assertEqual(traj["interference_model"], "co_channel")
        for step in traj["trajectory"]:
            self.assertEqual(len(step["uav_positions"]), 3)
            self.assertEqual(len(step["actions"]), 3)
            self.assertEqual(len(step["user_results"]), 30)
            self.assertEqual(step["position"], step["uav_positions"][0])   # scalar mirror
            self.assertIn("mean_snr_db", step)
            self.assertIn("users_per_uav", step)
            for u in step["user_results"]:
                self.assertIn("serving_uav", u)
                self.assertIn("per_uav", u)
                self.assertEqual(len(u["per_uav"]), 3)

        self.assertEqual(mio.validate_consistency(handoff, traj), [])

    def test_consistency_pipeline_positions_match_everywhere(self):
        import tempfile, json as _json
        from run_ppo import run_multiuav_episode
        import multiuav_io as mio
        with tempfile.TemporaryDirectory() as td:
            up = Path(td) / "u.json"
            tp = Path(td) / "t.json"
            run_multiuav_episode(
                scenario="clustered_survivors", num_uavs=3, num_users=40,
                mode="surrogate", max_steps=4, seed=99,
                out_uav_positions=str(up), out_trajectory=str(tp),
            )
            handoff = mio.read_uav_positions(up)
            traj = _json.loads(tp.read_text())

        # handoff UAV positions == trajectory final_positions
        np.testing.assert_allclose(
            np.array(handoff["uav_positions"]),
            np.array(traj["final_positions"]), atol=0.02,
        )
        # user positions identical between handoff and every trajectory step
        hu = np.array(handoff["user_positions"])
        for step in traj["trajectory"]:
            su = np.array([u["position"] for u in step["user_results"]])
            np.testing.assert_allclose(hu, su, atol=0.02)
        # no duplicate UAV positions (K=3 configured distinct)
        p = np.array(handoff["uav_positions"])
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertGreater(np.linalg.norm(p[i] - p[j]), 1.0)

    def test_k1_legacy_path_untouched_by_subprocess(self):
        """`python ml/run_ppo.py` with no multi-UAV args must produce the
        legacy 1-UAV files. Runs as a subprocess; restores data/*.json after."""
        import subprocess, shutil, json as _json
        data = PROJECT_ROOT / "data"
        bak = {}
        for name in ("uav_positions.json", "ppo_trajectory.json"):
            src = data / name
            if src.exists():
                bak[name] = src.read_bytes()
        try:
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "ml" / "run_ppo.py"),
                 "--mode", "surrogate", "--steps", "2"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            up = _json.loads((data / "uav_positions.json").read_text())
            tp = _json.loads((data / "ppo_trajectory.json").read_text())
            self.assertEqual(set(up.keys()), LEGACY_HANDOFF_KEYS)
            self.assertEqual(up["num_uavs"], 1)
            self.assertEqual(len(up["uav_positions"]), 1)
            self.assertEqual(set(tp.keys()), LEGACY_TRAJ_KEYS)
            self.assertEqual(tp["num_uavs"], 1)
            for step in tp["trajectory"]:
                self.assertNotIn("uav_positions", step)
                self.assertNotIn("actions", step)
        finally:
            for name, blob in bak.items():
                (data / name).write_bytes(blob)


# ======================================================================
# PHASE 7 - multi-UAV NS-3 topology
# ======================================================================
NS3_ROOT = Path.home() / "ns-allinone-3.47" / "ns-3.47"


def _run_ns3_with_handoff(handoff: dict, args: str = "") -> str:
    import subprocess, shutil, json as _json
    (NS3_ROOT / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "ns3" / "uav-demo.cc",
                 NS3_ROOT / "scratch" / "uav-demo.cc")
    (NS3_ROOT / "data" / "uav_positions.json").write_text(_json.dumps(handoff, indent=2))
    program = f"scratch/uav-demo {args}".strip()
    r = subprocess.run(["/usr/bin/python3.12", "./ns3", "run", program],
                       cwd=str(NS3_ROOT), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout


class TestOrchestratorArgparseBackCompat(unittest.TestCase):
    def test_help_lists_new_args_without_breaking(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_full_pipeline.py"), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(r.returncode, 0)
        for flag in ("--num-uavs", "--num-users", "--scenario", "--seed",
                     "--interference-model", "--steps", "--mode", "--ns3-path"):
            self.assertIn(flag, r.stdout)


@unittest.skipUnless(SLOW and NS3_ROOT.exists(),
                     "needs RESQNET_SLOW_TESTS=1 and ~/ns-allinone-3.47/ns-3.47")
class TestNs3MultiUavTopology(unittest.TestCase):
    def test_k3_topology(self):
        import multiuav_io as mio
        uavs = [[10.0, 28.0, 30.0], [-8.0, -8.0, 30.0], [28.0, -8.0, 30.0]]
        users = scenarios.generate_users("dense_urban", 18, seed=5).tolist()
        assoc = mio.nearest_uav_association(uavs, users)
        handoff = mio.build_uav_positions(uavs, users, associations=assoc)
        out = _run_ns3_with_handoff(handoff)

        self.assertIn("ML UAV Positions: 3", out)
        for k in (1, 2, 3):
            self.assertRegex(out, rf"UAV {k}: \(")
            self.assertRegex(out, rf"UAV {k} address: ")
        self.assertIn("Loaded 18 ground-user positions", out)

        addrs = re.findall(r"UAV \d+ address:\s+(\S+)", out)
        self.assertEqual(len(addrs), 3)
        flows = re.findall(r"^Flow:\s+\S+\s+->\s+(\S+)", out, re.M)
        self.assertEqual(len(flows), 18, "one FlowMonitor flow per user")
        self.assertTrue(set(flows).issubset(set(addrs)),
                        f"flow destinations {set(flows)} not all UAV addresses {set(addrs)}")
        # at least two different UAVs actually receive flows
        self.assertGreaterEqual(len(set(flows)), 2)

    def test_k1_stdout_shape_unchanged(self):
        import multiuav_io as mio
        uavs = [[-29.31, 22.64, 25.97]]
        users = [[float(i), float(-i), 1.5] for i in range(10)]
        handoff = mio.build_uav_positions(uavs, users)   # legacy shape, no associations
        out = _run_ns3_with_handoff(handoff)
        self.assertRegex(out, r"ML UAV Position: \(-29\.31, 22\.64, 25\.97\)")
        self.assertNotIn("ML UAV Positions:", out)     # multi-UAV line absent for K=1
        self.assertNotIn("UAV 1:", out)
        self.assertNotIn("UAV 1 address:", out)
        self.assertIn("Loaded 10 ground-user positions", out)
        flows = re.findall(r"^Flow:\s+\S+\s+->\s+(\S+)", out, re.M)
        self.assertTrue(all(d == "10.1.1.1" for d in flows))  # all users -> the one UAV

    def test_in_range_users_all_deliver(self):
        """Regression: with dynamic ARP, synchronized ARP retries collided and
        ns-3 blackholed whole flows for ArpCache::DeadTimeout (100 s). Every
        user here is < 100 m from its UAV, well inside ns-3's range, so every
        flow must deliver; the legacy --populateArp=false run must not."""
        import multiuav_io as mio
        uavs = scenarios.initial_uav_positions(3, "blocked_urban").tolist()
        users = scenarios.generate_users("blocked_urban", 30, seed=42).tolist()
        assoc = mio.nearest_uav_association(uavs, users)
        for u, a in zip(users, assoc):
            self.assertLess(float(np.linalg.norm(np.subtract(u, uavs[a]))), 100.0)
        handoff = mio.build_uav_positions(uavs, users, associations=assoc)

        def delivering(out):
            return len(re.findall(r"^Packets Received:\s+[1-9]", out, re.M))

        out = _run_ns3_with_handoff(handoff)
        self.assertEqual(len(re.findall(r"^Flow:", out, re.M)), 30)
        self.assertEqual(delivering(out), 30, "an in-range user delivered nothing")

        legacy = _run_ns3_with_handoff(handoff, "--populateArp=false")
        self.assertLess(delivering(legacy), 30,
                        "legacy dynamic-ARP run no longer reproduces the blackhole")


# ======================================================================
# PHASE 8 - dashboard API (multi-UAV aware)
# ======================================================================
class TestDashboardApiMultiUav(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warnings
        warnings.filterwarnings("ignore")
        sys.path.insert(0, str(PROJECT_ROOT / "dashboard" / "backend"))
        import app as _app
        from fastapi.testclient import TestClient
        cls.client = TestClient(_app.app)

    def test_scenarios_endpoint(self):
        r = self.client.get("/api/scenarios")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        ids = {s["id"] for s in d["scenarios"]}
        self.assertEqual(ids, {"open_area", "dense_urban", "blocked_urban",
                               "clustered_survivors", "wide_area"})
        splits = {s["id"]: s["split"] for s in d["scenarios"]}
        self.assertEqual(splits["blocked_urban"], "eval (held out)")
        self.assertEqual(splits["open_area"], "train")
        self.assertIn(1, d["uav_counts"])
        self.assertIn(3, d["uav_counts"])
        self.assertIn("co-channel", d["interference_note"])

    def test_training_summary_endpoint(self):
        r = self.client.get("/api/training/summary")
        # 200 if a model exists, 404 otherwise - both are valid
        self.assertIn(r.status_code, (200, 404))
        if r.status_code == 200:
            d = r.json()
            self.assertIn("episodes", d)
            self.assertIn("surrogate", d["rf_model"].lower())

    def test_experiments_endpoint_shape(self):
        r = self.client.get("/api/experiments")
        self.assertEqual(r.status_code, 200)
        self.assertIn("experiments", r.json())
        self.assertIsInstance(r.json()["experiments"], list)

    def test_run_request_accepts_multiuav_params(self):
        # validation only - do not actually launch (no NS-3 in unit context)
        r = self.client.post("/api/run", json={
            "mode": "surrogate", "steps": 3,
            "scenario": "does_not_exist", "num_uavs": 3, "num_users": 30,
        })
        self.assertEqual(r.status_code, 422)   # unknown scenario rejected
        r = self.client.post("/api/run", json={
            "mode": "surrogate", "steps": 3, "num_uavs": 99,
        })
        self.assertEqual(r.status_code, 422)   # num_uavs out of range

    def test_scene_meta_multiuav_fields(self):
        r = self.client.get("/api/scene/meta")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("num_uavs", d)
        self.assertIn("uav_trajectories", d)
        self.assertEqual(len(d["uav_trajectories"]), d["num_uavs"])

    def test_raw_passthroughs_still_work(self):
        for path in ("/api/ppo/trajectory", "/api/ns3/results", "/api/health"):
            self.assertEqual(self.client.get(path).status_code, 200)


# ======================================================================
# PHASE 9 - baselines / experiments
# ======================================================================
class TestEvaluationBaselines(unittest.TestCase):
    def test_fixed_random_ppo_placements_and_index(self):
        import tempfile, importlib
        import evaluate_multiuav as ev
        importlib.reload(ev)
        with tempfile.TemporaryDirectory() as td:
            ev.EXPERIMENTS_DIR = Path(td)
            ev.INDEX_FILE = Path(td) / "index.json"

            sid = "dense_urban"
            users = scenarios.generate_users(sid, 20, seed=3).tolist()
            fixed = scenarios.initial_uav_positions(3, sid).tolist()
            rand = ev.random_positions(3, sid, 3)
            ppo_final, ppo_init = ev.ppo_positions(3, 20, sid, "surrogate", 6, 3, "co_channel")

            # placements differ (random != fixed), all have 3 UAVs
            self.assertEqual(len(fixed), 3)
            self.assertEqual(len(rand), 3)
            self.assertEqual(len(ppo_final), 3)
            self.assertNotEqual(fixed, rand)

            rows = []
            for name, (s, e) in {"fixed": (fixed, fixed), "random": (rand, rand),
                                 "ppo": (ppo_init, ppo_final)}.items():
                res = ev.rf_eval(e, users, "surrogate", sid, "co_channel", 3)
                agg = res["aggregate"]
                rows.append({
                    "placement": name,
                    "coverage_pct": agg["coverage_percentage"],
                    "connected_users": agg["connected_users_count"],
                    "mean_best_sinr_db": agg["mean_best_sinr_db"],
                    "movement_m": round(ev.movement_m(s, e), 2),
                })
            names = {r["placement"] for r in rows}
            self.assertEqual(names, {"fixed", "random", "ppo"})
            for r in rows:
                self.assertGreaterEqual(r["coverage_pct"], 0)
                self.assertLessEqual(r["coverage_pct"], 100)
            self.assertEqual(next(r for r in rows if r["placement"] == "fixed")["movement_m"], 0.0)

    def test_scenario_split_is_reported(self):
        import evaluate_multiuav as ev  # noqa: F401
        self.assertIn("blocked_urban", scenarios.eval_scenarios())
        self.assertIn("dense_urban", scenarios.train_scenarios())


class TestExistingVerifyScriptStillPasses(unittest.TestCase):
    """sionna/verify_positions_and_handoff.py asserts num_uavs==1 &
    len(uav_positions)==1 on data/uav_positions.json - our K=1 builder
    must keep satisfying that."""

    def test_k1_builder_satisfies_legacy_asserts(self):
        d = mio.build_uav_positions([[1.0, 2.0, 3.0]],
                                    [[0.0, 0.0, 1.5]] * 10)
        self.assertEqual(d.get("num_uavs"), 1)
        self.assertEqual(len(d.get("uav_positions")), 1)
        self.assertEqual(len(d["uav_positions"][0]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
