"""Tests for the closed-loop per-UAV controller and its dashboard API.

    .venv/bin/python -m unittest ml/test_closed_loop_control.py -v

Fast: uses scripted RF models and the MOCK network measurer (no Sionna, no
ns-3). The real Sionna + ns-3 path is exercised by the end-to-end demo and by
ml/test_obstruction.py (obstruction matrix applied by ns-3, default behaviour
byte-identical).
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna", _ROOT / "dashboard" / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import network_model as nm  # noqa: E402
import closed_loop_control as clc  # noqa: E402

FREQ = 2.412e9
REGION = {"x": [-50.0, 350.0], "y": [-100.0, 100.0], "z": [10.0, 60.0]}
SAFETY = {"min_altitude_m": 10.0, "max_altitude_m": 60.0, "building_clearance_m": 5.0,
          "clearance_radius_m": 10.0, "min_separation_m": 15.0}


def cfg_for(**over):
    c = clc.deep_merge(clc.DEFAULT_CONFIG, {"safety": SAFETY, "step_interval_s": 0, "max_steps": 6,
                                           "network": "mock", "rf_mode": "surrogate", "events": []})
    return clc.deep_merge(c, over)


class SplitRF:
    """Scripted radio: users with x < 150 ('west') are blocked by 40 dB below
    30 m; users with x >= 150 ('east') are blocked by 40 dB above 20 m.
    Otherwise free space. Forces opposite altitude needs for two UAVs."""
    label = "scripted test RF"
    has_buildings = False

    def __init__(self):
        self.calls = 0

    def path_gain(self, positions, users):
        self.calls += 1
        P, U = np.asarray(positions, float), np.asarray(users, float)
        out = np.zeros((len(P), len(U)))
        for k, p in enumerate(P):
            for i, u in enumerate(U):
                pg = -nm.free_space_loss_db(math.dist(p, u), FREQ)
                west = u[0] < 150
                if (west and p[2] < 30) or (not west and p[2] > 20):
                    pg -= 40.0
                out[k, i] = pg
        return out


def split_users():
    west = [[x, y, 1.5] for x in (-20.0, 0.0, 20.0) for y in (-20.0, 20.0)]
    east = [[x, y, 1.5] for x in (280.0, 300.0, 320.0) for y in (-20.0, 20.0)]
    return west + east


def make(rf=None, measurer=None, positions=None, users=None, geometry=None, out_dir=None, **over):
    return clc.ClosedLoopController(
        cfg_for(**over), rf or SplitRF(), geometry or clc.FlatGeometry(), measurer or clc.MockMeasurer(FREQ),
        users or split_users(), positions or [[0.0, 0.0, 25.0], [300.0, 0.0, 25.0]], REGION,
        {"x": [-50, 350], "y": [-100, 100]}, FREQ, out_dir=out_dir, log=lambda m: None)


class RecordingMeasurer(clc.MockMeasurer):
    def __init__(self, *a, worse_if_moved_from=None, **k):
        super().__init__(FREQ)
        self.handoffs = []
        self.worse_if_moved_from = worse_if_moved_from

    def measure(self, handoff):
        self.handoffs.append(copy.deepcopy(handoff))
        m = super().measure(handoff)
        if self.worse_if_moved_from is not None and handoff["uav_positions"] != self.worse_if_moved_from:
            m = dict(m)
            m["per_user_rx"] = [0.0 for _ in m["per_user_rx"]]
            m["per_user_pdr"] = [0.0 for _ in m["per_user_pdr"]]
            m["per_user_latency_ms"] = [None for _ in m["per_user_latency_ms"]]
        return m


# ----------------------------------------------------------------------
class TestSafety(unittest.TestCase):
    def setUp(self):
        # a 30 m building over x in [100, 140], y in [-20, 20]
        self.geo = clc.FlatGeometry(lambda x, y: 30.0 if 100 <= x <= 140 and -20 <= y <= 20 else 0.0)

    def v(self, pos, others=(), origin=None):
        return clc.safety_violations(pos, list(others), REGION, SAFETY, self.geo, origin=origin)

    def test_safe_position(self):
        self.assertEqual(self.v([0, 0, 30]), [])

    def test_boundary(self):
        self.assertTrue(any("boundary" in r for r in self.v([400, 0, 30])))
        self.assertTrue(any("boundary" in r for r in self.v([0, -150, 30])))

    def test_min_and_max_altitude(self):
        self.assertTrue(any("minimum safe altitude" in r for r in self.v([0, 0, 9])))
        self.assertTrue(any("maximum altitude" in r for r in self.v([0, 0, 61])))

    def test_building_clearance(self):
        self.assertTrue(any("building clearance" in r for r in self.v([120, 0, 34])))
        self.assertEqual(self.v([120, 0, 35]), [])
        # roof within the clearance radius counts too
        self.assertTrue(any("building clearance" in r for r in self.v([148, 0, 20])))

    def test_midpoint_of_move_checked(self):
        # destination clear, but the path crosses the building's surroundings
        self.assertEqual(self.v([160, 0, 20]), [])
        self.assertTrue(any("building clearance" in r for r in self.v([160, 0, 20], origin=[80, 0, 20])))

    def test_separation(self):
        self.assertTrue(any("too close" in r for r in self.v([0, 0, 30], others=[[0, 10, 30]])))
        self.assertEqual(self.v([0, 0, 30], others=[[0, 20, 30]]), [])

    def test_unsafe_initial_position_rejected(self):
        with self.assertRaises(ValueError):
            make(positions=[[0.0, 0.0, 5.0], [300.0, 0.0, 25.0]])

    def test_controller_never_leaves_safe_set(self):
        """Every position the loop ever puts a UAV at (including trial moves) is safe."""
        geo = clc.FlatGeometry(lambda x, y: 28.0 if -40 <= x <= 40 and 30 <= y <= 60 else 0.0)
        ctl = make(geometry=geo, max_steps=8)
        seen = []
        orig = ctl.measurer.measure

        def spy(h):
            seen.append([list(p) for p in h["uav_positions"]])
            return orig(h)
        ctl.measurer.measure = spy
        ctl.run()
        self.assertGreater(len(seen), 1)
        for pos in seen:
            for k, p in enumerate(pos):
                self.assertEqual(clc.safety_violations(p, pos[:k] + pos[k + 1:], REGION, SAFETY, geo), [], p)


class TestObjective(unittest.TestCase):
    oc = clc.DEFAULT_CONFIG["objective"]

    def test_metrics(self):
        m = clc.network_metrics([1.0, 0.5, 0.0, 0.9], [10.0, 30.0, None, 20.0], [100, 50, 0, 90],
                                [100, 100, 100, 100], self.oc)
        self.assertAlmostEqual(m["coverage"], 0.75)
        self.assertAlmostEqual(m["pdr"], 0.6)
        self.assertAlmostEqual(m["worst_decile_pdr"], 0.0)
        self.assertAlmostEqual(m["mean_latency_ms"], 20.0)   # users that delivered only
        w = clc.PRESETS["balanced"]
        self.assertAlmostEqual(m["objective"], w["coverage"] * 0.75 + w["pdr"] * 0.6 - w["latency"] * 0.2)

    def test_priorities_change_the_ranking(self):
        """Wide coverage with high latency vs fewer users with low latency."""
        wide = dict(per_user_pdr=[0.95] * 10, per_user_latency=[120.0] * 10, rx=[95] * 10, sent=[100] * 10)
        narrow = dict(per_user_pdr=[0.95] * 8 + [0.0] * 2, per_user_latency=[10.0] * 8 + [None] * 2,
                      rx=[95] * 8 + [0] * 2, sent=[100] * 10)

        def j(d, preset):
            oc = dict(self.oc, preset=preset, weights=None)
            return clc.network_metrics(d["per_user_pdr"], d["per_user_latency"], d["rx"], d["sent"], oc)["objective"]
        self.assertGreater(j(wide, "coverage_first"), j(narrow, "coverage_first"))
        self.assertLess(j(wide, "latency_first"), j(narrow, "latency_first"))

    def test_custom_weights_validated(self):
        with self.assertRaises(ValueError):
            clc.weights_of({"weights": {"coverage": 1.0}})
        with self.assertRaises(ValueError):
            clc.weights_of({"weights": {"coverage": -1, "pdr": 0, "fairness": 0, "latency": 0}})


class TestIndependentAltitude(unittest.TestCase):
    def test_uavs_choose_different_altitudes(self):
        ctl = make(max_steps=6)
        st = ctl.run()
        z0, z1 = ctl.positions[0][2], ctl.positions[1][2]
        self.assertGreaterEqual(z0, 30.0, "west UAV must climb above the 30 m blockage")
        self.assertLessEqual(z1, 20.0, "east UAV must descend below 20 m")
        first = [d for d in ctl.decisions if d["step"] == 1]
        self.assertEqual({d["uav_id"]: d["action"] for d in first}, {"UAV-1": "up", "UAV-2": "down"})
        self.assertTrue(all(d["outcome"] == "accepted" for d in first))
        self.assertGreater(st["after"]["metrics"]["coverage"], st["before"]["metrics"]["coverage"])
        self.assertEqual(st["after"]["metrics"]["coverage"], 1.0)

    def test_horizontal_and_vertical_moves_available_per_uav(self):
        self.assertEqual(set(clc.ACTIONS), {"stay", "+x", "-x", "+y", "-y", "up", "down"})
        self.assertEqual(clc.ACTIONS["up"], (0, 0, 1))


class TestFeedbackLoop(unittest.TestCase):
    def test_phase_order_each_interval(self):
        ctl = make(max_steps=2)
        ctl.run()
        phases_step1 = [p for s, p in ctl.trace if s == 1]
        self.assertEqual(phases_step1, ["observe", "evaluate", "select", "apply", "measure", "feedback"])

    def test_measurement_after_every_move(self):
        meas = RecordingMeasurer()
        ctl = make(measurer=meas, max_steps=4)
        ctl.run()
        moved_steps = [h for h in ctl.history if h["moved"]]
        self.assertEqual(len(meas.handoffs), 1 + len(moved_steps))
        # the measured positions of each move are the positions after the move
        for h, hand in zip(moved_steps, meas.handoffs[1:]):
            self.assertEqual(hand["uav_positions"], h["positions"])

    def test_worse_measurement_is_reverted_and_banned(self):
        start = [[0.0, 0.0, 35.0], [300.0, 0.0, 25.0]]   # UAV-2 needs to move
        meas = RecordingMeasurer(worse_if_moved_from=start)
        ctl = make(measurer=meas, positions=start, max_steps=3)
        ctl.run()
        self.assertEqual(ctl.positions, start)
        d1 = [d for d in ctl.decisions if d["step"] == 1 and d["action"] != "stay"]
        self.assertTrue(d1 and all(d["outcome"] == "reverted" for d in d1))
        self.assertLess(d1[0]["measured_delta"]["objective"], 0)
        # the reverted moves are not retried while banned
        d2 = [d for d in ctl.decisions if d["step"] == 2]
        for d in d2:
            prev = next(x for x in d1 if x["uav"] == d["uav"]) if any(x["uav"] == d["uav"] for x in d1) else None
            if prev:
                self.assertNotEqual(d["action"], prev["action"])
        # learned correction now penalises the reverted actions
        self.assertLess(ctl._correction(d1[0]["action"]), 0)

    def test_stops_when_converged(self):
        ctl = make(max_steps=20, patience=2)
        st = ctl.run()
        self.assertEqual(st["status"], "converged")
        self.assertLess(st["step"], 20)

    def test_event_triggers_re_measurement(self):
        ev = [{"step": 2, "type": "add_users", "count": 3, "seed": 5,
               "region": {"x": [-30.0, 30.0], "y": [-30.0, 30.0]}}]
        meas = RecordingMeasurer()
        ctl = make(measurer=meas, max_steps=4, events=ev)
        st = ctl.run()
        self.assertEqual(len(ctl.users), 15)
        self.assertEqual(st["events"][0]["count"], 3)
        self.assertTrue(any(h["num_users"] == 15 for h in meas.handoffs))
        self.assertEqual(len(st["users"]), 15)

    def test_failed_event_is_recorded_not_fatal(self):
        class NoRoom(clc.FlatGeometry):
            def users(self, *a, **k):
                raise RuntimeError("placed only 2/6 street-level users")
        ev = [{"step": 1, "type": "add_users", "count": 6, "seed": 5, "region": {"x": [0, 1], "y": [0, 1]}}]
        ctl = make(geometry=NoRoom(), max_steps=3, events=ev)
        st = ctl.run()
        self.assertIn(st["status"], ("converged", "finished"))
        self.assertEqual(len(ctl.users), 12)
        self.assertIn("placed only", st["events"][0]["failed"])

    def test_pause_and_stop_via_control_file(self):
        import checkpoint as ck
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            ck.atomic_write_json(out / "control.json", {"command": "stop"})
            ctl = make(out_dir=out)
            st = ctl.run()
            self.assertEqual(st["status"], "stopped")
            self.assertEqual(json.loads((out / "state.json").read_text())["status"], "stopped")


class TestObstructionLinks(unittest.TestCase):
    def test_matrix_on_intended_links_and_distance_mode_unchanged(self):
        users = split_users()
        pos = [[0.0, 0.0, 25.0], [300.0, 0.0, 25.0]]
        for mode in ("obstruction", "distance"):
            meas = RecordingMeasurer()
            ctl = make(measurer=meas, positions=pos, users=users, channel_mode=mode, max_steps=1)
            ctl.cur = ctl.observe(ctl.positions)
            h = meas.handoffs[0]
            if mode == "distance":
                self.assertNotIn("obstruction_loss_db", h)
                self.assertNotIn("obstruction_freq_hz", h)
                continue
            M = h["obstruction_loss_db"]
            self.assertEqual(h["obstruction_freq_hz"], FREQ)
            self.assertEqual((len(M), len(M[0])), (2, len(users)))   # row = UAV, column = user
            for k, p in enumerate(pos):
                for i, u in enumerate(users):
                    west = u[0] < 150
                    blocked = (west and p[2] < 30) or (not west and p[2] > 20)
                    self.assertAlmostEqual(M[k][i], 40.0 if blocked else 0.0, places=2)

    def test_distance_mode_prediction_ignores_obstruction(self):
        users = split_users()
        pos = [[0.0, 0.0, 25.0], [300.0, 0.0, 25.0]]
        pg = SplitRF().path_gain(pos, users)
        po = clc.predict(pos, users, pg, "obstruction", FREQ)
        pd = clc.predict(pos, users, pg, "distance", FREQ)
        self.assertFalse(any(p["usable"] for p in po["net"]["per_user"]))
        self.assertTrue(all(p["usable"] for p in pd["net"]["per_user"]))
        self.assertEqual(po["assoc"], pd["assoc"])   # same RF association in both modes


# ----------------------------------------------------------------------
class TestDashboardApi(unittest.TestCase):
    """Backend <-> controller state synchronisation through the real API,
    with a real controller subprocess (surrogate RF + mock network)."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import app as dash_app
        import control
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cfg = json.loads((_ROOT / "configs" / "closed_loop_control.json").read_text())
        cfg.update({"rf_mode": "surrogate", "network": "mock", "step_interval_s": 0.4, "max_steps": 8, "patience": 100,
                    "events": [{"step": 2, "type": "add_users", "count": 4, "seed": 3,
                                "region": {"x": [560.0, 590.0], "y": [-140.0, -110.0]}}]})
        (base / "cfg.json").write_text(json.dumps(cfg))
        cls.mgr = control.ControlManager(config_path=base / "cfg.json", base_dir=base / "runs")
        cls._orig = control.manager
        control.manager = cls.mgr
        cls.client = TestClient(dash_app.app)

    @classmethod
    def tearDownClass(cls):
        import control
        cls.mgr.reset()
        control.manager = cls._orig
        cls.tmp.cleanup()

    def wait(self, pred, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.client.get("/api/control/state").json()
            if pred(st):
                return st
            time.sleep(0.2)
        self.fail(f"timeout; last state status={st.get('status')} step={st.get('step')}")

    def test_lifecycle_and_state_sync(self):
        c = self.client
        self.assertEqual(c.get("/api/control/state").json()["status"], "idle")
        r = c.post("/api/control/start", json={"channel_mode": "obstruction", "preset": "balanced"})
        self.assertEqual(r.status_code, 202, r.text)
        run_dir = Path(r.json()["out_dir"])
        self.assertEqual(c.post("/api/control/start", json={}).status_code, 409)   # one at a time

        st = self.wait(lambda s: s.get("step", 0) >= 1)
        # pause: the version stops advancing
        c.post("/api/control/pause")
        st = self.wait(lambda s: s["status"] == "paused")
        v = st["version"]
        time.sleep(1.5)
        self.assertEqual(c.get("/api/control/state").json()["version"], v)
        # resume: runs to the end
        c.post("/api/control/resume")
        st = self.wait(lambda s: s["status"] in ("converged", "finished", "failed"))
        self.assertIn(st["status"], ("converged", "finished"), st.get("error"))

        # the API returns exactly what the controller wrote
        disk = json.loads((run_dir / "state.json").read_text())
        api = {k: v for k, v in st.items() if k != "session"}
        self.assertEqual(api, disk)
        # internal consistency of the served state
        self.assertEqual([u["position"] for u in st["uavs"]], st["history"][-1]["positions"])
        self.assertEqual(len(st["users"]), st["after"]["num_users"])
        self.assertEqual(st["after"]["metrics"], st["metrics"])
        self.assertTrue(st["provenance"]["mock"])
        self.assertFalse(st["provenance"]["live_physical_data"])
        self.assertEqual(st["events"][0]["count"], 4)
        self.assertGreaterEqual(len(st["decisions"]), 2 * st["step"])
        for d in st["decisions"]:
            self.assertIn(d["outcome"], ("held", "accepted", "reverted"))
            self.assertTrue(d["reason"])

        r = c.post("/api/control/reset").json()
        self.assertEqual(r["status"], "idle")
        self.assertTrue((run_dir / "state.json").exists(), "reset must keep the run on disk")
        self.assertTrue(any(x["run_id"] == run_dir.name for x in c.get("/api/control/runs").json()["runs"]))

    def test_reset_kills_running_session(self):
        c = self.client
        r = c.post("/api/control/start", json={"max_steps": 40})
        self.assertEqual(r.status_code, 202, r.text)
        self.wait(lambda s: s.get("step", 0) >= 1)
        proc = self.mgr.session.proc
        c.post("/api/control/reset")
        self.assertIsNotNone(proc.poll())
        self.assertEqual(c.get("/api/control/state").json()["status"], "idle")
        self.assertEqual(c.post("/api/control/pause").status_code, 404)

    def test_invalid_requests_rejected(self):
        c = self.client
        self.assertEqual(c.post("/api/control/start", json={"channel_mode": "bogus"}).status_code, 422)
        self.assertEqual(c.post("/api/control/start", json={"max_steps": 0}).status_code, 422)
        self.assertEqual(c.post("/api/control/start", json={"weights": {"coverage": -1, "pdr": 0, "fairness": 0,
                                                                       "latency": 0}}).status_code, 422)

    def test_existing_endpoints_unchanged(self):
        c = self.client
        h = c.get("/api/health")
        self.assertEqual(h.status_code, 200)
        self.assertEqual(set(h.json()), {"status", "data_files", "scene_assets", "ns3_path", "ns3_path_exists"})
        self.assertEqual(c.get("/api/scenarios").status_code, 200)
        self.assertIn(c.get("/api/run/latest").status_code, (200, 404))


if __name__ == "__main__":
    unittest.main()
