"""Tests for the ns-3-calibrated model, calibrated reward, placement optimisers
and closed-loop evaluation helpers.

    cd ml && python test_closed_loop.py -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "ml", PROJECT_ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import calibrated_model as cm  # noqa: E402
import placement_search as ps  # noqa: E402
import scenarios  # noqa: E402
from multiuav_env import MultiUAVEnv, OBS_LEN, surrogate_multi_uav  # noqa: E402

HAVE_MODEL = cm.DEFAULT_MODEL.exists()


def _features(uavs, users):
    rf = surrogate_multi_uav(uavs, users, connectivity_config=None)
    X, usable, _ = cm.user_features(rf, uavs, users)
    return {f: X[:, i] for i, f in enumerate(cm.FEATURES)}, usable


class TestHiddenNodeFeatures(unittest.TestCase):
    def test_users_on_opposite_sides_are_hidden(self):
        """Both reach the UAV (100 m away each) but are 200 m apart: below the
        -82 dBm carrier-sense floor, so each is hidden from the other."""
        f, usable = _features([[0.0, 0.0, 40.0]], [[-100.0, 0.0, 1.5], [100.0, 0.0, 1.5]])
        self.assertTrue(usable.all())
        self.assertTrue((f["hidden_load"] > 0).all())
        self.assertTrue((f["sensed_load"] == 0).all())

    def test_nearby_users_sense_each_other(self):
        f, _ = _features([[0.0, 0.0, 40.0]], [[-10.0, 0.0, 1.5], [10.0, 0.0, 1.5]])
        self.assertTrue((f["hidden_load"] == 0).all())
        self.assertTrue((f["sensed_load"] > 0).all())

    def test_unusable_hidden_user_is_counted(self):
        # user 1 is out of its UAV's range but still reaches UAV 0's receiver?
        # place a far user that UAV 0 cannot hear: it must NOT count as hidden for user 0
        f, usable = _features([[0.0, 0.0, 40.0]], [[0.0, 0.0, 1.5], [400.0, 0.0, 1.5]])
        self.assertEqual(usable.tolist(), [True, False])
        self.assertEqual(f["hidden_unusable"][0], 0.0)


@unittest.skipUnless(HAVE_MODEL, "calibrated model not fitted yet")
class TestCalibratedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = cm.CalibratedNetworkModel()

    def test_monotone_in_hidden_and_contention_load(self):
        base = np.array([[25.0, 0.03, 0.2, 0.0, 0.5, 0.6, 30.0]])
        p0 = self.model.model.predict(base)[0]
        for col in (1, 2, 3, 4, 5):                       # every load-type feature
            x = base.copy(); x[0, col] *= 3.0; x[0, col] += 0.5
            self.assertLessEqual(self.model.model.predict(x)[0], p0 + 1e-9, cm.FEATURES[col])
        x = base.copy(); x[0, 0] = 40.0                    # better SNR never hurts
        self.assertGreaterEqual(self.model.model.predict(x)[0], p0 - 1e-9)

    def test_unusable_users_predicted_zero(self):
        uavs, users = [[0.0, 0.0, 40.0]], [[0.0, 0.0, 1.5], [400.0, 0.0, 1.5]]
        rf = surrogate_multi_uav(uavs, users, connectivity_config=None)
        pdr, _ = self.model.predict_users(rf, uavs, users)
        self.assertEqual(pdr[1], 0.0)
        self.assertGreater(pdr[0], 0.5)

    def test_report_documents_validation(self):
        rep = json.loads((cm.DEFAULT_MODEL.parent / "report.json").read_text())
        self.assertFalse(set(rep["train_scenarios"]) & set(rep["validation_scenarios"]))
        self.assertIn("held_out_scenarios", rep)


@unittest.skipUnless(HAVE_MODEL, "calibrated model not fitted yet")
class TestCalibratedReward(unittest.TestCase):
    def _env(self, n=30, sid="wide_area"):
        env = MultiUAVEnv(num_uavs=3, num_users=n, scenario=sid, mode="surrogate", seed=7,
                          reward_mode="calibrated")
        env.reset(seed=7)
        return env

    def test_obs_shape_and_bounds(self):
        env = self._env()
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs.shape, (OBS_LEN,))
        self.assertTrue(env.observation_space.contains(obs))
        obs, r, *_ = env.step([1, 2, 3, 0, 0])
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(np.isfinite(r))

    def test_stacking_penalised(self):
        env = self._env()
        spread = env.state_reward()
        env.uav_positions[:] = env.uav_positions[0]
        env._refresh()
        self.assertLess(env.state_reward(), spread)

    def test_abandoning_users_penalised(self):
        env = self._env()
        base = env.state_reward()
        reg = scenarios.get_scenario("wide_area")["uav_region"]
        env.uav_positions[2, :2] = [reg["x"][1], reg["y"][1]]
        env._refresh()
        self.assertLess(env.state_reward(), base)

    def test_legacy_default_unchanged(self):
        env = MultiUAVEnv(num_uavs=3, num_users=20, scenario="dense_urban", mode="surrogate", seed=3)
        self.assertEqual(env.reward_mode, "legacy")
        self.assertIsNone(env.calibrated)


@unittest.skipUnless(HAVE_MODEL, "calibrated model not fitted yet")
class TestSurrogateOptimisers(unittest.TestCase):
    def test_never_worse_than_start_and_respects_constraints(self):
        sid = "wide_area"
        users = scenarios.generate_users(sid, 20, 3).tolist()
        fixed = scenarios.initial_uav_positions(3, sid)
        reg = scenarios.get_scenario(sid)["uav_region"]
        for name, fn in ps.SURROGATE_OPTIMISERS.items():
            obj = ps.SurrogateObjective(sid, users, 3)
            start_val = obj(fixed)
            pos, val = fn(obj, fixed, budget=300)
            self.assertGreaterEqual(val, start_val - 1e-9, name)
            self.assertLessEqual(obj.evals, 300 + 1, name)
            np.testing.assert_allclose(pos[:, 2], fixed[:, 2], err_msg=name)     # altitude fixed
            self.assertTrue(np.all((pos[:, 0] >= reg["x"][0]) & (pos[:, 0] <= reg["x"][1])), name)
            self.assertAlmostEqual(obj(pos), val, places=9, msg=name)            # reported value is real


class TestNs3RefineLogic(unittest.TestCase):
    def test_accepts_measured_improvements_only(self):
        lo, hi = np.array([-100.0, -100.0]), np.array([100.0, 100.0])
        target = np.array([30.0, 0.0])

        def measure(jobs):   # fake "ns-3": PDR highest when UAV 0 sits at target
            return [1.0 - np.linalg.norm(p[0, :2] - target) / 1000.0 for _, p in jobs]

        start_a = np.array([[0.0, 0.0, 40.0], [50.0, 50.0, 40.0]])
        start_b = np.array([[-50.0, 0.0, 40.0], [50.0, 50.0, 40.0]])
        res = ps.ns3_refine_batch([{"candidates": [start_a, start_b], "lo": lo, "hi": hi}], measure,
                                  steps_m=(20.0, 10.0), min_gain=0.0)
        best = res[0]["pos"]
        self.assertEqual(best[0, 0], 30.0)                  # 0 -> 20 -> 30
        self.assertEqual(res[0]["runs"], 2 + 8 + 8)         # 2 starts + 2 x (2 UAVs x 4 moves)
        np.testing.assert_allclose(best[:, 2], 40.0)

    def test_noise_below_min_gain_is_rejected(self):
        lo, hi = np.array([-100.0, -100.0]), np.array([100.0, 100.0])
        start = np.array([[0.0, 0.0, 40.0]])
        res = ps.ns3_refine_batch([{"candidates": [start], "lo": lo, "hi": hi}],
                                  lambda jobs: [0.5 + 0.001 * (j > 0) for j, _ in enumerate(jobs)],
                                  min_gain=0.002)
        np.testing.assert_allclose(res[0]["pos"], start)

    def _far_target(self, max_rounds):
        lo, hi = np.array([-200.0, -200.0]), np.array([200.0, 200.0])
        target = np.array([90.0, 0.0])
        calls = []

        def measure(jobs):
            calls.append(len(jobs))
            return [1.0 - np.linalg.norm(p[0, :2] - target) / 1000.0 for _, p in jobs]
        start = np.array([[0.0, 0.0, 40.0]])
        res = ps.ns3_refine_batch([{"candidates": [start], "lo": lo, "hi": hi}], measure,
                                  min_gain=0.0, max_rounds=max_rounds)
        return res[0], calls

    def test_single_round_is_default(self):
        """Default = the original search: one round per step size, so a target
        90 m away is not reached (0 -> 20 -> 30)."""
        r, calls = self._far_target(max_rounds=1)
        self.assertEqual(r["pos"][0, 0], 30.0)
        self.assertEqual(calls, [1, 4, 4])
        self.assertEqual(ps.ns3_refine_batch.__defaults__[-1], 1)

    def test_iterated_rounds_follow_measured_gain_and_stop(self):
        """Iterated: keeps moving while the measurement improves, stops after
        the first round without gain, never changes altitude."""
        r, calls = self._far_target(max_rounds=10)
        self.assertEqual(r["pos"][0, 0], 90.0)               # 0 -> 20 -> ... -> 80 (20 m), -> 90 (10 m)
        np.testing.assert_allclose(r["pos"][:, 2], 40.0)
        # 20 m: 4 improving rounds + 1 without gain; 10 m: 1 improving + 1 without gain
        self.assertEqual(calls, [1] + [4] * 5 + [4] * 2)
        self.assertEqual(r["runs"], sum(calls))
        self.assertEqual(len(r["history"]), 1 + 4 + 1)

    def test_iterated_rounds_respect_round_cap(self):
        r, calls = self._far_target(max_rounds=2)
        self.assertEqual(calls, [1, 4, 4, 4, 4])
        self.assertEqual(r["pos"][0, 0], 60.0)               # 0 -> 20 -> 40 (20 m cap), -> 50 -> 60 (10 m cap)

    def test_inactive_instances_are_not_remeasured(self):
        lo, hi = np.array([-200.0, -200.0]), np.array([200.0, 200.0])
        seen = []

        def measure(jobs):
            seen.append(sorted({i for i, _ in jobs}))
            # instance 0 improves towards x=60; instance 1 is flat (never improves)
            return [1.0 - abs(p[0, 0] - 60.0) / 1000.0 if i == 0 else 0.5 for i, p in jobs]
        start = np.array([[0.0, 0.0, 40.0]])
        ps.ns3_refine_batch([{"candidates": [start], "lo": lo, "hi": hi}] * 2, measure,
                            min_gain=0.0, max_rounds=5)
        self.assertEqual(seen[0], [0, 1])                     # start candidates
        self.assertEqual(seen[1], [0, 1])                     # first 20 m round
        self.assertTrue(all(s == [0] for s in seen[2:4]))     # later 20 m rounds: only the improving one


class TestNoLeakage(unittest.TestCase):
    def test_eval_scenarios_never_in_training_configs(self):
        evc = json.loads((PROJECT_ROOT / "configs" / "eval_network.json").read_text())
        held_out = set(evc["eval_scenarios"])
        self.assertIn("flood_shelters", held_out)
        for cfg in ("training.json", "training_network.json", "training_calibrated.json"):
            tr = json.loads((PROJECT_ROOT / "configs" / cfg).read_text())
            self.assertFalse(set(tr["train"]["scenarios"]) & held_out, cfg)
            self.assertNotIn("flood_shelters", tr.get("extra_train_scenarios", {}), cfg)
        self.assertNotIn("flood_shelters", scenarios.scenario_ids())


class TestEvaluationHelpers(unittest.TestCase):
    def test_score_and_fairness(self):
        import evaluate_closed_loop as ecl
        runs = [{"per_user_rx": [800, 400, 0], "per_user_sent": [800, 800, 800], "delivering_users": 2,
                 "mean_latency_ms": 10.0, "p95_flow_latency_ms": 20.0}] * 2
        s = ecl.score(runs)
        self.assertAlmostEqual(s["pdr"], 0.5)
        self.assertAlmostEqual(s["loss_pct"], 50.0)
        self.assertEqual(s["users_pdr_ge_90"], 1)
        t = np.array([800, 400, 0]) * 540 * 8 / 8.0 / 1e6
        self.assertAlmostEqual(s["jain_fairness"], t.sum() ** 2 / (3 * (t ** 2).sum()))
        self.assertEqual(s["min_user_tput_mbps"], 0.0)

    def test_paired_stats_direction(self):
        import evaluate_closed_loop as ecl
        st = ecl.paired_stats([10, 12, 9], [12, 12, 12], higher_better=False)   # latency: lower is better
        self.assertEqual((st["wins"], st["ties"], st["losses"]), (2, 1, 0))
        self.assertLess(st["ci95"][0], st["ci95"][1] + 1e-12)


FAKE_COUNTING_NS3 = """#!/usr/bin/env python3
import json, os, sys
open(os.environ["FAKE_NS3_COUNTER"], "a").write(" ".join(sys.argv[1:]) + "\\n")
h = json.load(open("data/uav_positions.json"))
print("\\n====================================\\n       UAV NETWORK RESULTS\\n====================================")
for i in range(h["num_users"]):
    print(f"\\nFlow: 10.1.1.{h['num_uavs'] + i + 1} -> 10.1.1.1\\nPackets Sent: 800\\nPackets Received: 700\\n"
          "Throughput: 0.4 Mbps\\nAverage Delay: 5 ms\\nPacket Loss: 12.5 %")
print("\\n====================================")
"""


class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        import os, stat, tempfile
        import checkpoint as ck
        import evaluate_network as ev
        self.ck, self.ev = ck, ev
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.fake = self.dir / "fake_ns3"
        self.fake.write_text(FAKE_COUNTING_NS3)
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IEXEC)
        self.counter = self.dir / "calls.txt"
        os.environ["FAKE_NS3_COUNTER"] = str(self.counter)
        self.jobs = [({"num_uavs": 1, "uav_positions": [[float(i), 0, 30]], "num_users": 2,
                       "user_positions": [[0, 0, 1.5], [5, 0, 1.5]]}, ("--RngRun=11",)) for i in range(6)]

    def tearDown(self):
        self.td.cleanup()

    def calls(self):
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def runner(self, run_fn=None, **kw):
        fn = run_fn or (lambda h, a: self.ev.run_ns3(self.fake, h, a))
        return self.ck.CachedNs3Runner(self.dir / "cache", "binhash", fn, workers=2,
                                       log=lambda m: None, **kw)

    def test_atomic_write_leaves_no_temp_files(self):
        p = self.dir / "sub" / "r.json"
        self.ck.atomic_write_json(p, {"a": 1})
        self.ck.atomic_write_json(p, {"a": 2})
        self.assertEqual(self.ck.read_json(p), {"a": 2})
        self.assertEqual([x.name for x in p.parent.iterdir()], ["r.json"])

    def test_completed_simulations_are_not_rerun(self):
        r1 = self.runner().map(self.jobs)
        self.assertEqual(self.calls(), 6)
        run2 = self.runner()
        r2 = run2.map(self.jobs)
        self.assertEqual(self.calls(), 6, "a resumed run must not re-simulate cached jobs")
        self.assertEqual(r1, r2)
        self.assertEqual(run2.stats["cached"], 6)

    def test_corrupt_or_different_entries_are_resimulated(self):
        r = self.runner()
        r.map(self.jobs[:1])
        k = r.key(*self.jobs[0])
        r._path(k).write_text("{truncated")                     # corrupt entry
        self.runner().map(self.jobs[:1])
        self.assertEqual(self.calls(), 2)
        h, _ = self.jobs[0]
        self.runner().map([(h, ("--RngRun=12",))])               # different args -> new job
        self.assertEqual(self.calls(), 3)
        other = self.ck.CachedNs3Runner(self.dir / "cache", "otherbinary",
                                        lambda h, a: self.ev.run_ns3(self.fake, h, a), log=lambda m: None)
        other.map(self.jobs[:1])                                 # different ns-3 binary -> new job
        self.assertEqual(self.calls(), 4)

    def test_stop_mid_run_keeps_finished_jobs_and_resume_completes(self):
        holder = {}

        def fn(h, a):
            res = self.ev.run_ns3(self.fake, h, a)
            holder["r"].request_stop("test stop")
            return res
        r = self.runner(run_fn=fn, )
        r.workers = 1
        holder["r"] = r
        with self.assertRaises(self.ck.StopRequested):
            r.map(self.jobs)
        done = self.calls()
        self.assertGreaterEqual(done, 1)
        self.assertLess(done, 6)
        res = self.runner().map(self.jobs)                       # resume
        self.assertEqual(self.calls(), 6)
        self.assertTrue(all(x is not None for x in res))

    def test_low_memory_stops_without_simulating(self):
        r = self.runner(mem_fn=lambda: 100.0, min_free_mb=2048.0, low_mem_wait_s=0.0)
        with self.assertRaises(self.ck.StopRequested):
            r.map(self.jobs)
        self.assertEqual(self.calls(), 0)
        self.assertEqual(r.stats["min_mem_available_mb"], 100.0)

    def test_config_mismatch_is_refused(self):
        out = self.dir / "exp"
        out.mkdir()
        fp = {"settings": {"seeds": 20}, "files": {"model": "aaa"}}
        self.assertFalse(self.ck.check_or_write_config(out, fp))
        self.assertTrue(self.ck.check_or_write_config(out, fp))
        with self.assertRaises(SystemExit):
            self.ck.check_or_write_config(out, {"settings": {"seeds": 5}, "files": {"model": "aaa"}})
        with self.assertRaises(SystemExit):
            self.ck.check_or_write_config(out, {"settings": {"seeds": 20}, "files": {"model": "bbb"}})

    def test_fallback_cache_is_reused_read_only(self):
        old = self.ck.CachedNs3Runner(self.dir / "old", "binhash",
                                      lambda h, a: self.ev.run_ns3(self.fake, h, a), log=lambda m: None)
        r_old = old.map(self.jobs[:3])
        self.assertEqual(self.calls(), 3)
        before = sorted(p.name for p in (self.dir / "old").rglob("*.json"))
        new = self.runner(fallback_dirs=[self.dir / "old"])
        r_new = new.map(self.jobs)
        self.assertEqual(self.calls(), 6, "only the 3 jobs missing from the old cache are simulated")
        self.assertEqual(r_new[:3], r_old)
        self.assertEqual(sorted(p.name for p in (self.dir / "old").rglob("*.json")), before,
                         "the fallback cache must never be written")
        self.assertEqual(len(list((self.dir / "cache").rglob("*.json"))), 6, "hits are copied in")
        other = self.ck.CachedNs3Runner(self.dir / "c2", "otherbinary",
                                        lambda h, a: self.ev.run_ns3(self.fake, h, a), log=lambda m: None,
                                        fallback_dirs=[self.dir / "old"])
        other.map(self.jobs[:1])                                 # different binary -> no reuse
        self.assertEqual(self.calls(), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
