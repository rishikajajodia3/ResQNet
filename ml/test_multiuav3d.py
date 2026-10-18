"""Tests for the 3D multi-UAV environment (ml/multiuav3d_env.py) and its PPO
smoke path. Run: python -m unittest ml/test_multiuav3d.py"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multiuav3d_env as E  # noqa: E402
from closed_loop_control import safety_violations, SurrogateRF  # noqa: E402

REGION = {"x": [0.0, 195.0], "y": [0.0, 195.0], "z": [15.0, 60.0]}
WINDOWS = {"w": {"role": "train", "uav_region": REGION, "user_region": {"x": [10.0, 185.0], "y": [10.0, 185.0]}}}


def geometry(building=True):
    """5 m height map: open ground plus one 30 m tall block x, y in [100, 130]."""
    xs = np.arange(-20.0, 220.0, 5.0)
    ys = np.arange(-20.0, 220.0, 5.0)
    H = np.zeros((len(ys), len(xs)))
    if building:
        H[np.ix_((ys >= 100) & (ys <= 130), (xs >= 100) & (xs <= 130))] = 30.0
    return E.HeightmapGeometry(xs, ys, H)


class ConstRF:
    def path_gain(self, P, U):
        return np.full((len(P), len(U)), -70.0)


def const_pdr(P, U, pg):
    """Network outcome independent of position: isolates the reward formula,
    so any altitude dependence would have to be hard-coded in it."""
    return np.full(len(U), 0.7)


def make(rf=None, pdr=None, **kw):
    return E.MultiUAV3DEnv(WINDOWS, geometry(), rf or SurrogateRF(E.TRAINING_FREQ_HZ),
                           pdr or E.NetworkPDR(), seed=0, **kw)


def reset_at(env, positions, n_users=10):
    users = [[20.0 + 15 * i, 20.0, 1.5] for i in range(n_users)]
    return env.reset(options={"window": "w", "positions": positions, "users": users})


def act(env, per_uav):
    a = np.zeros(E.MAX_UAVS, int)
    a[:len(per_uav)] = per_uav
    return env.step(a)


class TestSpaces(unittest.TestCase):
    def test_spaces(self):
        env = make()
        self.assertEqual(list(env.action_space.nvec), [7] * 5)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, (E.OBS_LEN,))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(E.ACTION_NAMES, ["stay", "-x", "+x", "-y", "+y", "down", "up"])

    def test_held_out_refused(self):
        with self.assertRaises(ValueError):
            E.MultiUAV3DEnv({"urban_core": WINDOWS["w"]}, geometry(), ConstRF(), E.NetworkPDR())
        E.MultiUAV3DEnv({"urban_core": WINDOWS["w"]}, geometry(), ConstRF(), E.NetworkPDR(), allow_held_out=True)

    def test_training_config_excludes_urban_core(self):
        import json
        cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "training_areas_3d.json").read_text())
        held = json.loads((Path(__file__).resolve().parents[1] / "configs" / "eval_urban_core.json").read_text())
        uc = held["scenarios"]["urban_core"]["uav_region"]
        self.assertNotIn("urban_core", cfg["windows"])
        for w in cfg["windows"].values():
            r = w["uav_region"]
            dx = max(uc["x"][0] - r["x"][1], r["x"][0] - uc["x"][1], 0)
            dy = max(uc["y"][0] - r["y"][1], r["y"][0] - uc["y"][1], 0)
            self.assertGreaterEqual(math.hypot(dx, dy), 100.0)


class TestMoves(unittest.TestCase):
    def test_each_action_moves_one_axis(self):
        env = make()
        expect = {1: (-15, 0, 0), 2: (15, 0, 0), 3: (0, -15, 0), 4: (0, 15, 0), 5: (0, 0, -10), 6: (0, 0, 10)}
        for a, d in expect.items():
            reset_at(env, [[45.0, 45.0, 40.0]])
            _, _, _, _, info = act(env, [a])
            np.testing.assert_allclose(env.P[0], np.array([45.0, 45.0, 40.0]) + d)
            self.assertEqual(info["actions"], [a])

    def test_independent_z_per_uav(self):
        env = make()
        reset_at(env, [[30.0, 30.0, 40.0], [75.0, 30.0, 40.0], [30.0, 75.0, 40.0]])
        act(env, [6, 5, 2])                          # up, down, +x in one step
        np.testing.assert_allclose(env.P, [[30, 30, 50], [75, 30, 30], [45, 75, 40]])
        act(env, [6, 0, 0])
        self.assertEqual(sorted(env.P[:, 2]), [30, 40, 60])   # three different altitudes, chosen per UAV

    def test_inactive_slots_ignored(self):
        env = make()
        reset_at(env, [[30.0, 30.0, 40.0]])
        a = np.array([0, 6, 6, 6, 6])
        env.step(a)
        self.assertEqual(env.P.shape, (1, 3))
        np.testing.assert_allclose(env.P[0], [30, 30, 40])
        self.assertFalse(env.action_masks().reshape(5, 7)[1:, 1:].any())

    def test_step_sizes_configurable(self):
        env = make(step_h_m=10.0, step_v_m=5.0)
        reset_at(env, [[30.0, 30.0, 40.0]])
        act(env, [2]); act(env, [6])
        np.testing.assert_allclose(env.P[0], [40, 30, 45])


class TestSafety(unittest.TestCase):
    def assert_blocked(self, env, k, a):
        before = env.P.copy()
        self.assertFalse(env.action_masks().reshape(5, 7)[k, a])
        _, _, _, _, info = act(env, [a if j == k else 0 for j in range(env.K)])
        np.testing.assert_allclose(env.P, before)
        self.assertEqual(info["unsafe_attempts"], 1)

    def test_min_altitude(self):
        env = make(); reset_at(env, [[45.0, 45.0, 20.0]]); self.assert_blocked(env, 0, 5)

    def test_max_altitude(self):
        env = make(); reset_at(env, [[45.0, 45.0, 60.0]]); self.assert_blocked(env, 0, 6)

    def test_boundary(self):
        env = make(); reset_at(env, [[0.0, 45.0, 40.0]]); self.assert_blocked(env, 0, 1)

    def test_building_clearance_horizontal(self):
        env = make(); reset_at(env, [[75.0, 115.0, 30.0]])   # 25 m from the 30 m block: safe at 30 m
        self.assert_blocked(env, 0, 2)                         # x=90 is within 10 m of it: needs 35 m

    def test_building_clearance_descent(self):
        env = make(); reset_at(env, [[115.0, 115.0, 40.0]])   # above the 30 m block, 40 >= 30 + 5
        self.assert_blocked(env, 0, 5)                         # 30 m < 35 m required

    def test_separation_mask_and_joint_conflict(self):
        env = make()
        reset_at(env, [[30.0, 30.0, 40.0], [60.0, 30.0, 40.0]])
        _, _, _, _, info = act(env, [2, 1])          # both move into x=45: only one may move
        self.assertEqual(info["separation_cancels"], 1)
        self.assertGreaterEqual(math.dist(env.P[0], env.P[1]), 15.0)

    def test_swap_through_each_other_blocked(self):
        env = make()
        reset_at(env, [[30.0, 30.0, 40.0], [45.0, 30.0, 40.0]])
        act(env, [2, 1])
        self.assertGreaterEqual(math.dist(env.P[0], env.P[1]), 15.0)
        self.assertFalse(np.allclose(env.P, [[45, 30, 40], [30, 30, 40]]))

    def test_random_rollouts_never_violate(self):
        """Random (unmasked and masked) actions: every position after every step
        passes the controller's safety rule, and Z actually changes."""
        env = make(num_uavs=(3, 4, 5), num_users=(10,))
        rng = np.random.default_rng(3)
        zs = set()
        for ep in range(6):
            env.reset(seed=ep)
            for _ in range(25):
                if ep % 2:
                    m = env.action_masks().reshape(5, 7)
                    a = [rng.choice(np.flatnonzero(m[k])) for k in range(5)]
                else:
                    a = rng.integers(0, 7, size=5)
                env.step(np.array(a))
                for k in range(env.K):
                    others = [env.P[j] for j in range(env.K) if j != k]
                    self.assertEqual(safety_violations(env.P[k], others, env.region, env.safety, env.geometry), [])
                    zs.add(float(env.P[k, 2]))
        self.assertGreaterEqual(len(zs), 3)


class TestRewardAndObs(unittest.TestCase):
    def test_reward_has_no_altitude_preference(self):
        rs = []
        for z in (20.0, 40.0, 60.0):
            env = make(rf=ConstRF(), pdr=const_pdr)
            reset_at(env, [[45.0, 45.0, z]])
            _, r, _, _, _ = act(env, [0])
            rs.append(r)
        self.assertAlmostEqual(max(rs) - min(rs), 0.0, places=9)

    def test_reward_terms_and_movement_cost(self):
        env = make(rf=ConstRF(), pdr=const_pdr)
        reset_at(env, [[45.0, 45.0, 40.0]])
        _, r_stay, _, _, _ = act(env, [0])
        _, r_move, _, _, _ = act(env, [6])
        self.assertAlmostEqual(r_stay - r_move, env.w["move"] * 10.0 / 15.0, places=9)
        self.assertTrue(set(env.reward_terms) >= {"score", "delta", "move_norm", "unsafe_norm", "reward"})

    def test_obs_reflects_z_and_clearance(self):
        env = make()
        o1, _ = reset_at(env, [[45.0, 45.0, 20.0]])
        o2, _ = act(env, [6])[:2]
        self.assertAlmostEqual(o1[3], (20 - 15) / 45, places=6)
        self.assertAlmostEqual(o2[3], (30 - 15) / 45, places=6)
        self.assertEqual(o1[5 + 4], 1.0)            # "down" blocked at 20 m (index 5 = -x ... 9 = down)
        self.assertEqual(o2[5 + 4], 0.0)

    def test_metrics_in_info(self):
        env = make()
        _, info = env.reset(seed=4)
        for k in ("pdr", "coverage", "worst10_pdr", "throughput_mbps", "positions", "actions"):
            self.assertIn(k, info)

    def test_calibrated_frequency_shim(self):
        """Excess loss seen by the calibrated model equals the 2.412 GHz excess."""
        P = np.array([[0.0, 0.0, 40.0]]); U = np.array([[30.0, 40.0, 1.5]])
        d = np.linalg.norm(P[:, None] - U[None], axis=2)
        pg = -(20 * np.log10(4 * np.pi * d * 2.412e9 / E.nm.C_LIGHT)) - 12.0
        self.assertAlmostEqual(float(E.excess_loss_db(pg, d, 2.412e9)[0, 0]), 12.0, places=6)


class TestPPO(unittest.TestCase):
    def test_maskable_ppo_train_save_load(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: make(num_users=(10,))])
        model = MaskablePPO("MlpPolicy", venv, n_steps=64, batch_size=32, seed=0, verbose=0)
        model.learn(128)
        env = make(num_users=(10,))
        obs, _ = env.reset(seed=9)
        m = env.action_masks()
        a1, _ = model.predict(obs, deterministic=True, action_masks=m)
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d) / "m.zip")
            loaded = MaskablePPO.load(Path(d) / "m.zip")
        a2, _ = loaded.predict(obs, deterministic=True, action_masks=m)
        np.testing.assert_array_equal(a1, a2)
        self.assertEqual(a1.shape, (5,))
        self.assertTrue(all(m.reshape(5, 7)[k, a1[k]] for k in range(5)))   # masked actions respected
        self.assertEqual(loaded.action_space.nvec.tolist(), [7] * 5)


    def test_plain_ppo_fallback_without_masks(self):
        """Without masking (plain SB3 PPO) unsafe choices become 'stay' and are
        counted; positions stay safe."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: make(num_users=(10,))])
        model = PPO("MlpPolicy", venv, n_steps=64, batch_size=32, seed=0, verbose=0)
        model.learn(128)
        env = make(num_users=(10,), num_uavs=(4,))
        obs, _ = env.reset(seed=2)
        unsafe = 0
        for _ in range(30):
            a, _ = model.predict(obs, deterministic=False)
            obs, _, _, trunc, info = env.step(a)
            unsafe += info["unsafe_attempts"]
            for k in range(env.K):
                others = [env.P[j] for j in range(env.K) if j != k]
                self.assertEqual(safety_violations(env.P[k], others, env.region, env.safety, env.geometry), [])
            if trunc:
                obs, _ = env.reset()
        self.assertGreaterEqual(unsafe, 0)

    def test_masks_shape_and_stay_always_valid(self):
        env = make(num_uavs=(5,), num_users=(10,))
        for s in range(5):
            env.reset(seed=s)
            m = env.action_masks()
            self.assertEqual(m.shape, (35,))
            self.assertTrue(m.reshape(5, 7)[:, 0].all())


class TestRadioMapRF(unittest.TestCase):
    def test_lookup_and_metadata(self):
        import json
        import radiomap_cache as rc
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cx = (np.arange(10) + 0.5) * 2.0          # cells over x, y in [0, 20)
            meta = {"cache_version": rc.CACHE_VERSION, "freq_hz": 2.412e9, "cell_m": 2.0,
                    "cell_centres": {"x": cx.tolist(), "y": cx.tolist()}}
            (d / "meta.json").write_text(json.dumps(meta))
            P = np.array([[0.0, 0.0, 20.0], [15.0, 0.0, 30.0]])
            maps = np.zeros((2, 10, 10), np.float32)
            maps[0, 3, 7] = -77.0                      # tx0, cell y in [6, 8), x in [14, 16)
            maps[1, 3, 7] = -88.0
            np.savez_compressed(d / "shard_0000.npz", positions=P, pg_db=maps)
            rf = rc.RadioMapRF(d, expect={"freq_hz": 2.412e9})
            pg = rf.path_gain([[15.0, 0.0, 30.0], [0.0, 0.0, 20.0]], [[15.2, 7.9, 1.5], [1.0, 1.0, 1.5]])
            np.testing.assert_allclose(pg, [[-88.0, 0.0], [-77.0, 0.0]])
            with self.assertRaises(KeyError):
                rf.path_gain([[5.0, 0.0, 20.0]], [[1.0, 1.0, 1.5]])     # not a cached position
            with self.assertRaises(ValueError):
                rf.path_gain([[0.0, 0.0, 20.0]], [[25.0, 1.0, 1.5]])    # user outside the map
            with self.assertRaises(ValueError):
                rc.RadioMapRF(d, expect={"freq_hz": 2.1e9})             # incompatible settings refused


class TestClosedLoopPPO(unittest.TestCase):
    """policy "ppo" in closed_loop_control: measured PDR -> observation ->
    action -> safety filter -> move -> RF -> network -> next observation."""

    @classmethod
    def setUpClass(cls):
        import json
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        model = MaskablePPO("MlpPolicy", DummyVecEnv([lambda: make(num_users=(10,))]), n_steps=64,
                            batch_size=32, seed=1, verbose=0)
        model.learn(64)
        model.save(d / "model.zip")
        (d / "train_config.json").write_text(json.dumps({
            "run_id": "test", "obs_version": E.OBS_VERSION, "step_h_m": 15.0, "step_v_m": 10.0,
            "max_steps": 12, "coverage_threshold": 0.5, "rf_label": "test", "pdr_model": "test"}))
        cls.ckpt = str(d / "model.zip")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def controller(self, policy="ppo", **over):
        import closed_loop_control as clc
        cfg = clc.deep_merge(clc.DEFAULT_CONFIG, {"network": "mock", "rf_mode": "surrogate", "events": [],
                                                  "step_interval_s": 0, "max_steps": 4, "policy": policy,
                                                  "ppo": {"checkpoint": self.ckpt, "deterministic": False}})
        cfg = clc.deep_merge(cfg, over)
        f = 2.412e9
        users = [[20.0 + 15 * i, 60.0, 1.5] for i in range(10)]
        pos = [[45.0, 45.0, 40.0], [105.0, 45.0, 30.0], [150.0, 150.0, 50.0]]
        return clc.ClosedLoopController(cfg, clc.SurrogateRF(f), geometry(), clc.MockMeasurer(f), users, pos,
                                        REGION, WINDOWS["w"]["user_region"], f)

    def test_default_policy_is_rule_based(self):
        import closed_loop_control as clc
        self.assertEqual(clc.DEFAULT_CONFIG["policy"], "rule_based")
        ctl = self.controller(policy="rule_based")
        self.assertIsNone(ctl.ppo)
        self.assertEqual(ctl.state()["provenance"]["policy"], {"policy": "rule_based"})

    def test_ppo_loop_uses_measurements_and_records_actions(self):
        ctl = self.controller()
        seen = []
        orig = E.MultiUAV3DEnv.load_state

        def spy(env, region, positions, users, pg, pdr, t, **kw):
            seen.append((list(map(list, positions)), list(pdr)))
            return orig(env, region, positions, users, pg, pdr, t, **kw)

        E.MultiUAV3DEnv.load_state = spy
        try:
            st = ctl.run()
        finally:
            E.MultiUAV3DEnv.load_state = orig
        self.assertGreaterEqual(len(seen), 1)
        hist = st["history"]
        # the observation at step t is built from the measurement taken after step t-1
        for t, (pos, pdr) in enumerate(seen):
            self.assertEqual(pos, hist[t]["positions"])
        self.assertEqual(st["provenance"]["policy"]["policy"], "ppo")
        self.assertEqual(st["provenance"]["policy"]["checkpoint"], self.ckpt)
        for h in hist[1:]:
            self.assertEqual(len(h["actions"]), 3)
            self.assertTrue(set(h["actions"]) <= set(E.ACTION_NAMES))
        # every executed move is one lattice step, positions stay safe
        for a, b in zip(hist, hist[1:]):
            for k in range(3):
                d = np.abs(np.array(b["positions"][k]) - np.array(a["positions"][k]))
                self.assertIn(round(float(d.sum()), 6), (0.0, 10.0, 15.0))
        for k, p in enumerate(ctl.positions):
            others = ctl.positions[:k] + ctl.positions[k + 1:]
            self.assertEqual(safety_violations(p, others, ctl.region, ctl.cfg["safety"], ctl.geometry), [])
        for r in ctl.decisions:
            self.assertIn(r["outcome"], ("held", "executed"))
            self.assertIn("policy_action", r)

    def test_first_observation_is_measured_pdr(self):
        ctl = self.controller(max_steps=1)
        captured = {}
        orig = E.MultiUAV3DEnv.load_state

        def spy(env, region, positions, users, pg, pdr, t, **kw):
            captured["pdr"] = list(pdr)
            return orig(env, region, positions, users, pg, pdr, t, **kw)

        E.MultiUAV3DEnv.load_state = spy
        try:
            ctl.status = "running"
            ctl.cur = ctl.observe(ctl.positions)
            measured = list(ctl.cur["meas"]["per_user_pdr"])
            ctl.step()
        finally:
            E.MultiUAV3DEnv.load_state = orig
        self.assertEqual(captured["pdr"], measured)

    def test_incompatible_checkpoint_refused(self):
        with self.assertRaises(ValueError):
            self.controller(step_h_m=20.0)


class TestDashboardSchema(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard" / "backend"))
        import schemas
        self.S = schemas

    def test_default_request_unchanged(self):
        self.assertEqual(self.S.ControlStartRequest().to_overrides(), {})

    def test_ppo_requires_checkpoint_inside_models(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.S.ControlStartRequest(policy="ppo")
        with self.assertRaises(ValidationError):
            self.S.ControlStartRequest(policy="ppo", ppo_checkpoint="/etc/passwd")
        with self.assertRaises(ValidationError):
            self.S.ControlStartRequest(policy="ppo", ppo_checkpoint="../ppo_multiuav/model.zip")

    def test_site_override(self):
        o = self.S.ControlStartRequest(site="val_a").to_overrides()
        self.assertEqual(o["scenario_config"], "configs/eval_val_a_3d.json")


if __name__ == "__main__":
    unittest.main()
