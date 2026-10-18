"""Tests for the optional ns-3 obstruction-loss mode and the urban_core scenario.

    python -m unittest ml/test_obstruction.py -v

ns-3 tests need the built uav-demo binary (current source) and, for the
default-behaviour regression test, the ORIGINAL binary saved before the
rebuild (RESQNET_NS3_BINARY_ORIG, default ~/.cache/resqnet_ns3_bin/uav-demo-orig).
Sionna tests need sionna-rt. Missing prerequisites skip the test with a reason.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import multiuav_io as mio  # noqa: E402
import network_model as nm  # noqa: E402
import obstruction as ob  # noqa: E402
import scenarios  # noqa: E402

CONFIG = _ROOT / "configs" / "eval_urban_core.json"
ORIG_SHA = "3b6c957815f4d75759e63cac90fce3ee1ef251c481050d7842c6168bbf7b0e7d"
F = 2.412e9


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def fake_rf(uavs, users, pg):
    """Minimal RF result (same shape as Sionna / surrogate) with given path gains pg[k][i]."""
    ur = []
    for i, u in enumerate(users):
        per = [{"uav_id": f"uav_{k+1}", "uav_position": [round(c, 2) for c in uavs[k]],
                "path_gain_db": pg[k][i], "num_paths": 1} for k in range(len(uavs))]
        ur.append({"position": [round(c, 2) for c in u], "per_uav": per, "serving_uav": 1})
    return {"num_uavs": len(uavs), "num_users": len(users), "user_results": ur}


def fspl(a, b, f=F):
    return nm.free_space_loss_db(float(np.linalg.norm(np.asarray(a) - np.asarray(b))), f)


def new_binary():
    env = os.environ.get("RESQNET_NS3_BINARY")
    if env:
        return Path(env)
    import evaluate_obstruction as eo
    root = Path(os.environ.get("NS3_PATH") or Path.home() / "ns-allinone-3.47" / "ns-3.47")
    try:
        return eo.find_binary(root)
    except SystemExit as e:
        raise unittest.SkipTest(f"no current ns-3 binary: {e}")


def orig_binary():
    b = Path(os.environ.get("RESQNET_NS3_BINARY_ORIG") or Path.home() / ".cache" / "resqnet_ns3_bin" / "uav-demo-orig")
    if not b.exists():
        raise unittest.SkipTest(f"original binary backup not found: {b}")
    import checkpoint as ck
    if ck.file_sha256(b) != ORIG_SHA:
        raise AssertionError(f"{b} is not the original binary (sha mismatch)")
    return b


def run_raw(binary, handoff, args=("--RngRun=1",), text=None):
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "data").mkdir()
        (Path(td) / "data" / "uav_positions.json").write_text(text if text is not None else json.dumps(handoff))
        r = subprocess.run([str(binary), *args], cwd=td, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout, r.stderr


def flows(stdout, K):
    """Per-user (sent, received) from uav-demo stdout, by user index."""
    import evaluate_network as ev
    out = {}
    for m in ev.FLOW_RE.finditer(stdout):
        idx = int(m.group(1).split(".")[-1]) - K - 1
        blk = m.group(3)
        sent = int(__import__("re").search(r"Packets Sent:\s+(\d+)", blk).group(1))
        rx = __import__("re").search(r"Packets Received:\s+(\d+)", blk)
        out[idx] = (sent, int(rx.group(1)) if rx else 0)
    return out


def ns3_freq_hz(binary):
    import evaluate_obstruction as eo
    return eo.ns3_channel(binary)["frequency_mhz"] * 1e6


def regression_handoffs():
    """Three handoffs exercising the K=1 legacy path, K=3 and urban_core coordinates."""
    rng = np.random.default_rng(3)
    k1_users = [[round(float(10 * math.cos(a)), 2), round(float(10 * math.sin(a)), 2), 1.5]
                for a in np.linspace(0, 2 * math.pi, 10, endpoint=False)]
    h1 = mio.build_uav_positions([[0.0, 0.0, 26.0]], k1_users)
    users3 = scenarios.generate_users("blocked_urban", 30, 0).tolist()
    uavs3 = scenarios.initial_uav_positions(3, "blocked_urban").tolist()
    h3 = mio.build_uav_positions(uavs3, users3, mio.nearest_uav_association(uavs3, users3))
    uavs_c = [[500.0, -20.0, 30.0], [470.0, -80.0, 30.0], [530.0, -80.0, 30.0]]
    users_c = [[round(float(x), 2), round(float(y), 2), 1.5]
               for x, y in zip(rng.uniform(410, 590, 30), rng.uniform(-140, 40, 30))]
    hc = mio.build_uav_positions(uavs_c, users_c, mio.nearest_uav_association(uavs_c, users_c))
    return {"k1_legacy": h1, "k3_blocked_urban": h3, "k3_urban_core": hc}


# ----------------------------------------------------------------------
# pure Python
# ----------------------------------------------------------------------
class TestObstructionMatrix(unittest.TestCase):
    uavs = [[0.0, 0.0, 30.0], [100.0, 0.0, 30.0]]
    users = [[10.0, 5.0, 1.5], [60.0, -20.0, 1.5], [90.0, 40.0, 1.5]]

    def pg(self, offsets):
        """path gain = -(FSPL + offset); offset None -> no path."""
        return [[-200.0 if offsets[k][i] is None else -(fspl(self.uavs[k], self.users[i]) + offsets[k][i])
                 for i in range(3)] for k in range(2)]

    def test_excess_over_free_space_clamp_and_no_path(self):
        rf = fake_rf(self.uavs, self.users, self.pg([[0.0, 12.0, -1.0], [None, 3.5, 25.0]]))
        M, st = ob.obstruction_matrix_db(rf, self.uavs, self.users, F)
        np.testing.assert_allclose(M[0], [0.0, 12.0, 0.0], atol=0.02)      # LOS 0, obstructed 12, constructive clamped
        self.assertEqual(M[1][0], ob.NO_PATH_LOSS_DB)
        np.testing.assert_allclose(M[1][1:], [3.5, 25.0], atol=0.02)
        self.assertEqual(st["no_path_links"], 1)
        self.assertEqual(st["clamped_negative_links"], 1)
        self.assertEqual(st["links"], 6)

    def test_shape_and_order_is_uav_by_user(self):
        rf = fake_rf(self.uavs, self.users, self.pg([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        M, _ = ob.obstruction_matrix_db(rf, self.uavs, self.users, F)
        self.assertEqual((len(M), len(M[0])), (2, 3))
        np.testing.assert_allclose(M, [[1, 2, 3], [4, 5, 6]], atol=0.02)

    def test_distance_not_counted_twice(self):
        # a LOS link far away still has ~0 obstruction: the matrix carries no distance term
        far_users = [[1000.0, 0.0, 1.5]]
        rf = fake_rf(self.uavs, far_users, [[-fspl(self.uavs[0], far_users[0])], [-fspl(self.uavs[1], far_users[0])]])
        M, _ = ob.obstruction_matrix_db(rf, self.uavs, far_users, F)
        self.assertLess(max(M[0][0], M[1][0]), 0.02)

    def test_frequency_consistency_matters(self):
        # the same path gain read at a different frequency gives a different excess:
        # free-space loss grows with frequency, so the excess at 2.412 GHz is smaller
        rf = fake_rf(self.uavs, self.users, self.pg([[5.0] * 3, [5.0] * 3]))
        M24, _ = ob.obstruction_matrix_db(rf, self.uavs, self.users, F)
        M21, _ = ob.obstruction_matrix_db(rf, self.uavs, self.users, 2.1e9)
        self.assertAlmostEqual(M24[0][0] - M21[0][0], -20 * math.log10(F / 2.1e9), delta=0.02)

    def test_geometry_mismatch_raises(self):
        rf = fake_rf(self.uavs, self.users, self.pg([[0.0] * 3, [0.0] * 3]))
        moved = copy.deepcopy(self.uavs)
        moved[1][2] = 45.0
        with self.assertRaises(ValueError):
            ob.obstruction_matrix_db(rf, moved, self.users, F)
        with self.assertRaises(ValueError):
            ob.obstruction_matrix_db(rf, self.uavs, self.users[:2], F)


class TestHandoff(unittest.TestCase):
    def test_with_obstruction_copies_and_validates(self):
        h = mio.build_uav_positions([[0, 0, 20], [50, 0, 20]], [[1, 1, 1.5], [2, 2, 1.5], [3, 3, 1.5]], [0, 0, 1])
        before = json.dumps(h, sort_keys=True)
        h2 = ob.with_obstruction(h, [[0, 1, 2], [3, 4, 5]], F)
        self.assertEqual(json.dumps(h, sort_keys=True), before)       # input untouched
        self.assertEqual(h2["obstruction_loss_db"], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(h2["obstruction_freq_hz"], F)
        for bad in ([[0, 1], [3, 4]], [[0, 1, 2]], [[0, -1, 2], [3, 4, 5]], [[0, float("nan"), 2], [3, 4, 5]],
                    [[0, 1, 2], [3, 4, 301]]):
            with self.assertRaises(ValueError):
                ob.with_obstruction(h, bad, F)

    def test_default_handoff_unchanged(self):
        h = mio.build_uav_positions([[0, 0, 20]], [[1, 1, 1.5]])
        self.assertEqual(set(h), {"num_uavs", "uav_positions", "num_users", "user_positions"})


class TestMatchedLatency(unittest.TestCase):
    def mk(self, rng, rx, delay, failed=False):
        return {"rng_run": rng, "per_user_rx": rx, "per_user_delay_ms": delay, **({"failed": True} if failed else {})}

    def test_only_users_delivering_at_both_altitudes_same_run(self):
        import evaluate_obstruction as eo
        a = [self.mk(11, [10, 10, 0, 10], [5.0, 10.0, None, 20.0]),
             self.mk(12, [10, 10, 10, 0], [7.0, 9.0, 3.0, None]),
             self.mk(13, [10, 10, 10, 10], [1.0, 1.0, 1.0, 1.0], failed=True)]   # failed: ignored
        b = [self.mk(12, [10, 10, 10, 10], [1.0, 2.0, 3.0, 4.0]),                 # order does not matter
             self.mk(11, [10, 0, 10, 10], [15.0, None, 30.0, 40.0])]
        m = eo.matched_latency(a, b)
        # run 11: users 0, 3 (1 dead in b, 2 dead in a); run 12: users 0, 1, 2 (3 dead in a)
        self.assertEqual(m["matched_users_per_run"], [2, 3])
        self.assertAlmostEqual(m["a_ms"], ((5 + 20) / 2 + (7 + 9 + 3) / 3) / 2)
        self.assertAlmostEqual(m["b_ms"], ((15 + 40) / 2 + (1 + 2 + 3) / 3) / 2)
        self.assertAlmostEqual(m["rel_change"], (m["a_ms"] - m["b_ms"]) / m["b_ms"])
        self.assertEqual(m["num_users"], 4)
        self.assertEqual(m["runs_paired"], 2)

    def test_survivor_bias_removed(self):
        # altitude B serves only user 0 (fast); A serves all users (slower overall, user 0 equally fast)
        import evaluate_obstruction as eo
        a = [self.mk(11, [10, 10, 10], [5.0, 80.0, 90.0])]
        b = [self.mk(11, [10, 0, 0], [5.0, None, None])]
        m = eo.matched_latency(a, b)
        self.assertEqual(m["matched_users_per_run"], [1])
        self.assertAlmostEqual(m["rel_change"], 0.0)

    def test_no_match_gives_none(self):
        import evaluate_obstruction as eo
        self.assertIsNone(eo.matched_latency([self.mk(11, [10, 0], [5.0, None])],
                                             [self.mk(11, [0, 10], [None, 5.0])]))
        self.assertIsNone(eo.matched_latency([self.mk(11, [10], [5.0])], [self.mk(12, [10], [5.0])]))


class TestScenarioIsolation(unittest.TestCase):
    def test_urban_core_not_in_existing_configs(self):
        self.assertNotIn("urban_core", scenarios.load_scenarios()["scenarios"])
        self.assertNotIn("urban_core", scenarios.train_scenarios())
        self.assertNotIn("urban_core", scenarios.eval_scenarios())
        evn = json.loads((_ROOT / "configs" / "eval_network.json").read_text())
        self.assertNotIn("urban_core", evn["extra_eval_scenarios"])
        self.assertNotIn("urban_core", evn["eval_scenarios"])
        for f in (_ROOT / "configs").glob("training*.json"):
            self.assertNotIn("urban_core", f.read_text(), f.name)

    def test_config_well_formed_and_registers(self):
        cfg = json.loads(CONFIG.read_text())
        s = cfg["scenarios"]["urban_core"]
        self.assertEqual(s["centre"], [500.0, -50.0])
        for reg in (s["user_region"], s["uav_region"]):
            self.assertLess(reg["x"][0], 500.0)
            self.assertGreater(reg["x"][1], 500.0)
        self.assertEqual(s["uav_region"]["z"], [15.0, 60.0])
        for z in cfg["experiment"]["altitudes_m"]:
            self.assertTrue(s["uav_region"]["z"][0] <= z <= s["uav_region"]["z"][1])
        had = "urban_core" in scenarios._EXTRA_SCENARIOS
        try:
            scenarios.register_scenarios({"urban_core": s})
            self.assertEqual(scenarios.get_scenario("urban_core")["scene"], "city_damaged.xml")
        finally:
            if not had:
                scenarios._EXTRA_SCENARIOS.pop("urban_core", None)


# ----------------------------------------------------------------------
# Sionna scene geometry
# ----------------------------------------------------------------------
class TestSceneGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import sionna_feedback_engine as sfe
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"sionna not available: {e}")
        cls.sfe = sfe
        cls.engine = sfe.SionnaFeedbackEngine(scene_name="city_damaged.xml", freq=F)
        cls.mi_scene = cls.engine.scene.mi_scene
        cls.cfg = json.loads(CONFIG.read_text())
        cls.scen = cls.cfg["scenarios"]["urban_core"]
        cls.exp = cls.cfg["experiment"]

    def test_alignment_dense_site_vs_operating_region(self):
        # urban_core window is built up; the existing operating region is not (same coordinates as ns-3)
        g = np.arange(-90, 91, 10.0)
        gx, gy = np.meshgrid(g, g)
        core = ob.surface_height(self.mi_scene, 500 + gx.ravel(), -50 + gy.ravel())
        ops = ob.surface_height(self.mi_scene, 10 + gx.ravel() * 0.6, 10 + gy.ravel() * 0.6)
        self.assertGreater((core > 5).mean(), 0.3)
        self.assertLess(ops.max(), 10.0)

    def test_street_level_users_reproducible_and_open(self):
        reg = self.scen["user_region"]
        a = ob.street_level_users(self.mi_scene, reg, 30, 7000)
        b = ob.street_level_users(self.mi_scene, reg, 30, 7000)
        c = ob.street_level_users(self.mi_scene, reg, 30, 7001)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        xy = np.asarray(a)[:, :2]
        self.assertTrue(((xy[:, 0] >= reg["x"][0]) & (xy[:, 0] <= reg["x"][1])).all())
        self.assertTrue(((xy[:, 1] >= reg["y"][0]) & (xy[:, 1] <= reg["y"][1])).all())
        self.assertTrue(all(u[2] == 1.5 for u in a))
        self.assertLess(ob.surface_height(self.mi_scene, xy[:, 0], xy[:, 1]).max(), 0.5)

    def test_uavs_clear_roofs_at_every_altitude(self):
        uav = ob.roof_clear_uav_xy(self.mi_scene, self.scen["centre"], self.exp["uav_offsets_m"],
                                   search_m=self.exp["uav_snap_search_m"], radius_m=self.exp["roof_radius_m"])
        self.assertEqual(len(uav), 3)
        lowest = min(self.exp["altitudes_m"])
        for u in uav:
            roof = ob.max_height_within(self.mi_scene, u["x"], u["y"], self.exp["roof_radius_m"])
            self.assertAlmostEqual(roof, u["local_roof_m"], places=3)
            self.assertGreaterEqual(lowest, roof + self.exp["roof_clearance_m"],
                                    f"UAV at {u} needs an altitude floor above {lowest} m")

    def test_sionna_geometry_equals_handoff(self):
        uavs = [[500.0, -20.0, 30.0], [470.0, -80.0, 30.0], [530.0, -80.0, 30.0]]
        users = ob.street_level_users(self.mi_scene, self.scen["user_region"], 5, 7000)
        rf = self.engine.evaluate_multi_uav(uavs, users)
        M, st = ob.obstruction_matrix_db(rf, uavs, users, F)       # raises on any position mismatch
        self.assertEqual((len(M), len(M[0])), (3, 5))
        h = mio.build_uav_positions(uavs, users, [u["serving_uav"] - 1 for u in rf["user_results"]])
        for k in range(3):
            pos = np.array(self.engine.scene.transmitters[f"uav_{k+1}"].position).ravel()
            np.testing.assert_allclose(pos, h["uav_positions"][k], atol=1e-3)
        for i in range(5):
            pos = np.array(self.engine.scene.receivers[f"user_{i+1}"].position).ravel()
            np.testing.assert_allclose(pos, h["user_positions"][i], atol=1e-3)


# ----------------------------------------------------------------------
# ns-3
# ----------------------------------------------------------------------
class TestNs3DefaultUnchanged(unittest.TestCase):
    def test_default_stdout_byte_identical_to_original_binary(self):
        new, old = new_binary(), orig_binary()
        for name, h in regression_handoffs().items():
            with self.subTest(handoff=name):
                rn, on, en = run_raw(new, h)
                ro, oo, eo_ = run_raw(old, h)
                self.assertEqual((rn, ro), (0, 0), en + eo_)
                self.assertEqual(on, oo, f"{name}: default-mode stdout differs from the original binary")

    def test_zero_matrix_identical_to_default(self):
        b = new_binary()
        f = ns3_freq_hz(b)
        h = regression_handoffs()["k3_urban_core"]
        r0, s0, e0 = run_raw(b, h)
        hz = ob.with_obstruction(h, [[0.0] * h["num_users"] for _ in range(h["num_uavs"])], f)
        r1, s1, e1 = run_raw(b, hz)
        self.assertEqual((r0, r1), (0, 0), e0 + e1)
        self.assertIn("Obstruction loss: 3 x 30 matrix applied", s1)
        strip = lambda s: "\n".join(l for l in s.splitlines()  # noqa: E731
                                    if not l.startswith(("Obstruction loss:", "CHANNEL ")))
        self.assertEqual(strip(s1), strip(s0))


class TestNs3ObstructionApplied(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = new_binary()
        cls.f = ns3_freq_hz(cls.b)

    def test_exact_db_and_no_double_counting(self):
        # LogDistance (exp 3, 46.6777 dB @1 m, 30 dBm) gives -70 dBm at d; detection threshold -82 dBm
        d = 10 ** ((30.0 - 46.6777 + 70.0) / 30.0)
        h = mio.build_uav_positions([[0.0, 0.0, 1.5]], [[round(d, 2), 0.0, 1.5]])
        got = {}
        for loss in (0.0, 8.0, 16.0):
            rc, out, err = run_raw(self.b, ob.with_obstruction(h, [[loss]], self.f))
            self.assertEqual(rc, 0, err)
            got[loss] = flows(out, 1)[0]
        self.assertGreater(got[0.0][1], 0.99 * got[0.0][0])      # -70 dBm: delivers
        self.assertGreater(got[8.0][1], 0.99 * got[8.0][0])      # -78 dBm: still above -82 (fails if FSPL were added again)
        self.assertEqual(got[16.0][1], 0)                         # -86 dBm: below detection -> nothing delivered

    def test_loss_goes_to_the_right_uav_user_pair(self):
        uavs = [[0.0, 0.0, 20.0], [200.0, 0.0, 20.0], [0.0, 200.0, 20.0]]
        users = [[5.0, 0.0, 1.5], [0.0, 205.0, 1.5], [205.0, 0.0, 1.5]]
        assoc = [0, 2, 1]                                  # user 1 -> UAV 2, user 2 -> UAV 1
        h = mio.build_uav_positions(uavs, users, assoc)
        M = [[0.0] * 3 for _ in range(3)]
        M[1][2] = 200.0                                    # UAV 1 <-> user 2 (a serving link)
        rc, out, err = run_raw(self.b, ob.with_obstruction(h, M, self.f))
        self.assertEqual(rc, 0, err)
        fl = flows(out, 3)
        self.assertEqual(fl[2][1], 0, "user 2's serving link was blocked")
        self.assertGreater(fl[1][1], 0.99 * fl[1][0], "transposed matrix would block user 1")
        self.assertGreater(fl[0][1], 0.99 * fl[0][0])

    def test_detailed_runner_matches_project_runner(self):
        import evaluate_network as ev
        import evaluate_obstruction as eo
        h = regression_handoffs()["k3_urban_core"]
        a = ev.run_ns3(self.b, h, ("--RngRun=11",))
        d = eo.run_ns3_detailed(self.b, h, ("--RngRun=11",))
        for k, v in a.items():
            self.assertEqual(d[k], v, k)
        self.assertFalse(d["obstruction_applied"])
        self.assertEqual([x is None for x in d["per_user_delay_ms"]], [rx == 0 for rx in d["per_user_rx"]])
        # with a matrix: flagged as applied, and a blocked user has no delay
        uavs = [[0.0, 0.0, 20.0], [200.0, 0.0, 20.0], [0.0, 200.0, 20.0]]
        users = [[5.0, 0.0, 1.5], [0.0, 205.0, 1.5], [205.0, 0.0, 1.5]]
        M = [[0.0] * 3 for _ in range(3)]
        M[1][2] = 200.0
        d2 = eo.run_ns3_detailed(self.b, ob.with_obstruction(mio.build_uav_positions(uavs, users, [0, 2, 1]), M, self.f))
        self.assertTrue(d2["obstruction_applied"])
        self.assertIsNone(d2["per_user_delay_ms"][2])
        self.assertIsNotNone(d2["per_user_delay_ms"][0])

    def test_malformed_matrix_is_an_error(self):
        h = mio.build_uav_positions([[0, 0, 20], [50, 0, 20]], [[1, 1, 1.5], [40, 2, 1.5]], [0, 1])
        good = ob.with_obstruction(h, [[0, 0], [0, 0]], self.f)
        cases = {
            "rows": {**good, "obstruction_loss_db": [[0, 0]]},
            "cols": {**good, "obstruction_loss_db": [[0, 0, 0], [0, 0, 0]]},
            "negative": {**good, "obstruction_loss_db": [[0, -1], [0, 0]]},
            "too_large": {**good, "obstruction_loss_db": [[0, 301], [0, 0]]},
            "freq_missing": {k: v for k, v in good.items() if k != "obstruction_freq_hz"},
            "freq_mismatch": {**good, "obstruction_freq_hz": self.f + 50e6},
        }
        raw = {"nan": json.dumps(good).replace('"obstruction_loss_db": [[0.0, 0.0]', '"obstruction_loss_db": [[NaN, 0.0]'),
               "text": json.dumps(good).replace('"obstruction_loss_db": [[0.0, 0.0]', '"obstruction_loss_db": [["x", 0.0]'),
               "not_array": json.dumps(good).replace('"obstruction_loss_db": [[0.0, 0.0], [0.0, 0.0]]',
                                                     '"obstruction_loss_db": 5')}
        for name, text in raw.items():
            self.assertNotEqual(text, json.dumps(good), name)
        for name, hd in cases.items():
            with self.subTest(case=name):
                rc, out, err = run_raw(self.b, hd)
                self.assertEqual(rc, 2, f"{name}: rc={rc} stderr={err[-300:]}")
                self.assertIn("ERROR", err)
        for name, text in raw.items():
            with self.subTest(case=name):
                rc, out, err = run_raw(self.b, None, text=text)
                self.assertEqual(rc, 2, f"{name}: rc={rc} stderr={err[-300:]}")
                self.assertIn("ERROR", err)


if __name__ == "__main__":
    unittest.main()
