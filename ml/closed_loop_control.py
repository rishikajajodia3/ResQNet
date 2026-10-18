"""Closed-loop, per-UAV 3-D positioning controller (simulated digital twin).

Every control interval the controller runs one full loop:

    observe   current UAV / user positions; Sionna RT links at the current
              positions; ns-3 measured per-user delivery and latency
    evaluate  the objective J (below) and per-UAV diagnostics -> which UAVs
              need to move, and why
    select    each flagged UAV independently picks one move from
              {stay, +x, -x, +y, -y, up, down}; candidates that break a safety
              constraint are discarded; the rest are ranked by a Sionna-based
              prediction (+ a correction learned from earlier measurements)
    apply     move the selected UAVs (different UAVs may pick different
              altitudes and horizontal moves)
    measure   ray trace the new positions with Sionna and simulate them in ns-3
    feedback  keep the new positions if the MEASURED objective did not drop by
              more than the tolerance, otherwise fly the moved UAVs back and
              ban those moves for a few intervals; update the learned correction

The objective is computed from ns-3 measurements only:

    J = w_cov * coverage            fraction of users with PDR >= coverage threshold
      + w_pdr * pdr                 aggregate packet delivery ratio
      + w_fair * worst_decile_pdr   mean PDR of the worst 10 % of users
      - w_lat * latency_score       mean per-user latency / latency_ref (capped)

Latency is not predicted by any model in this project; candidate ranking
therefore leaves it out, and its effect enters only through the measured
acceptance test and the learned per-move correction. Nothing assumes that a
higher (or lower) altitude is better.

Channel modes: "obstruction" hands ns-3 the Sionna obstruction-loss matrix
(ml/obstruction.py, applied by ns3/uav-demo.cc to exactly the UAV-user links);
"distance" runs the unchanged distance-only ns-3 channel.

Everything here is SIMULATED (Sionna ray tracing + ns-3). No physical UAV or
live telemetry is involved. The "mock" network measurer and the "surrogate" RF
model exist for tests and quick UI checks and are labelled as such in the state.

    python ml/closed_loop_control.py --config configs/closed_loop_control.json \
        --out ~/.cache/resqnet_control/demo
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import network_model as nm  # noqa: E402
import obstruction as ob  # noqa: E402
import multiuav_io as mio  # noqa: E402
import checkpoint as ck  # noqa: E402

SCHEMA_VERSION = 1

ACTIONS: dict[str, tuple[int, int, int]] = {
    "stay": (0, 0, 0),
    "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "+y": (0, 1, 0), "-y": (0, -1, 0),
    "up": (0, 0, 1), "down": (0, 0, -1),
}

PRESETS: dict[str, dict[str, float]] = {
    "balanced": {"coverage": 0.35, "pdr": 0.30, "fairness": 0.15, "latency": 0.20},
    "coverage_first": {"coverage": 0.50, "pdr": 0.30, "fairness": 0.15, "latency": 0.05},
    "latency_first": {"coverage": 0.25, "pdr": 0.20, "fairness": 0.10, "latency": 0.45},
}

DEFAULT_CONFIG: dict[str, Any] = {
    "scenario_config": "configs/eval_urban_core.json",
    "rf_mode": "sionna",            # sionna | surrogate (free space, no buildings)
    "network": "ns3",               # ns3 | mock (tests / UI checks only)
    "channel_mode": "obstruction",  # obstruction | distance
    "num_uavs": 3,
    "num_users": 30,
    "user_seed": 7000,
    "initial_altitude_m": 40.0,
    "max_steps": 12,
    "patience": 2,                  # stop after this many intervals with no move and no pending event
    "step_interval_s": 0.5,         # pause between intervals (UI readability)
    "step_h_m": 15.0,
    "step_v_m": 10.0,
    "safety": {
        "min_altitude_m": 15.0,
        "max_altitude_m": 60.0,
        "building_clearance_m": 5.0,
        "clearance_radius_m": 10.0,
        "min_separation_m": 15.0,
    },
    "objective": {
        "preset": "balanced",
        "weights": None,            # overrides the preset when given
        "coverage_pdr_threshold": 0.5,
        "latency_ref_ms": 100.0,
        "latency_cap": 3.0,
        "pdr_target": 0.9,
        "latency_target_ms": 50.0,
        "margin_weight": 0.05,      # predicted link-margin tie-breaker (prediction only)
        "move_cost_per_m": 0.0005,
    },
    "acceptance_tolerance": 0.005,
    "min_predicted_gain": 0.002,
    "probe_tolerance": 0.01,
    "tabu_steps": 3,
    "ns3": {"traffic_seconds": 3.0, "rng_runs": [11], "workers": 2},
    "events": [],
    "policy": "rule_based",         # rule_based | ppo (3D MaskablePPO, ml/ppo_controller.py)
    "ppo": {"checkpoint": None, "deterministic": True},
}


def deep_merge(base: dict, over: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def weights_of(obj_cfg: dict) -> dict[str, float]:
    w = obj_cfg.get("weights") or PRESETS[obj_cfg.get("preset", "balanced")]
    missing = {"coverage", "pdr", "fairness", "latency"} - set(w)
    if missing:
        raise ValueError(f"objective weights missing {sorted(missing)}")
    if any(float(v) < 0 for v in w.values()):
        raise ValueError("objective weights must be >= 0")
    return {k: float(w[k]) for k in ("coverage", "pdr", "fairness", "latency")}


# ----------------------------------------------------------------------
# objective (measured)
# ----------------------------------------------------------------------
def network_metrics(per_user_pdr: Sequence[float], per_user_latency: Sequence[Optional[float]],
                    rx: Sequence[float], sent: Sequence[float], obj_cfg: dict) -> dict:
    """Headline metrics and J from per-user measurements."""
    pdr_u = np.asarray(per_user_pdr, dtype=float)
    n = len(pdr_u)
    total_sent = float(np.sum(sent))
    worst_n = max(1, int(math.ceil(0.10 * n)))
    lats = [float(l) for l, p in zip(per_user_latency, pdr_u) if l is not None and p > 0]
    lat = float(np.mean(lats)) if lats else None
    lat_score = obj_cfg["latency_cap"] if lat is None else min(lat / obj_cfg["latency_ref_ms"], obj_cfg["latency_cap"])
    m = {
        "num_users": n,
        "coverage": float(np.mean(pdr_u >= obj_cfg["coverage_pdr_threshold"])) if n else 0.0,
        "pdr": float(np.sum(rx)) / total_sent if total_sent else 0.0,
        "worst_decile_pdr": float(np.mean(np.sort(pdr_u)[:worst_n])) if n else 0.0,
        "mean_latency_ms": lat,
        "latency_score": float(lat_score),
        "users_pdr_ge_target": int(np.sum(pdr_u >= obj_cfg["pdr_target"])),
        "users_no_delivery": int(np.sum(pdr_u <= 0.0)),
    }
    w = weights_of(obj_cfg)
    m["objective"] = (w["coverage"] * m["coverage"] + w["pdr"] * m["pdr"]
                      + w["fairness"] * m["worst_decile_pdr"] - w["latency"] * m["latency_score"])
    return m


METRIC_KEYS = ("objective", "coverage", "pdr", "worst_decile_pdr", "mean_latency_ms")


def metric_delta(new: dict, old: dict) -> dict:
    out = {}
    for k in METRIC_KEYS:
        a, b = new.get(k), old.get(k)
        out[k] = None if a is None or b is None else round(float(a) - float(b), 6)
    return out


# ----------------------------------------------------------------------
# radio: per-link path gain (Sionna RT or free-space surrogate)
# ----------------------------------------------------------------------
class SionnaRF:
    """Sionna RT on the scenario scene at the ns-3 operating frequency."""

    label = "Sionna RT ray tracing (simulated)"
    has_buildings = True

    def __init__(self, scene_name: str, freq_hz: float, seed: int = 42):
        import sionna_feedback_engine as sfe
        self.sfe = sfe
        self.scene_name, self.freq_hz, self.seed = scene_name, float(freq_hz), seed
        self.conn = sfe.ConnectivityConfig()
        self._mi_scene = None

    def path_gain(self, positions: Sequence[Sequence[float]], users: Sequence[Sequence[float]]) -> np.ndarray:
        rf = self.sfe.evaluate_multi_uav([list(map(float, p)) for p in positions], user_positions=users,
                                         scene_name=self.scene_name, freq=self.freq_hz,
                                         connectivity_config=self.conn, seed=self.seed)
        return np.array([[l["path_gain_db"] for l in u["per_uav"]] for u in rf["user_results"]], dtype=float).T

    @property
    def mi_scene(self):
        if self._mi_scene is None:
            self.path_gain([[0.0, 0.0, 100.0]], [[0.0, 5.0, 1.5]])
            self._mi_scene = self.sfe._GLOBAL_ENGINE.scene.mi_scene
        return self._mi_scene


class SurrogateRF:
    """Free-space surrogate (no buildings). For tests / quick UI checks."""

    label = "free-space surrogate (no buildings; NOT Sionna)"
    has_buildings = False

    def __init__(self, freq_hz: float):
        self.freq_hz = float(freq_hz)

    def path_gain(self, positions, users) -> np.ndarray:
        P = np.asarray(positions, float)
        U = np.asarray(users, float)
        d = np.linalg.norm(P[:, None, :] - U[None, :, :], axis=2)
        return -np.vectorize(nm.free_space_loss_db)(d, self.freq_hz)


def compose_rf(positions: Sequence[Sequence[float]], users: Sequence[Sequence[float]],
               pg: np.ndarray, serving: Optional[Sequence[int]] = None) -> dict:
    """RF-result dict (the shape of sionna_feedback_engine.evaluate_multi_uav,
    fields used by obstruction.py / network_model.py) for K positions with the
    given K x N path gains. Best-server association = strongest path gain
    (all UAVs transmit at the same power)."""
    K, N = len(positions), len(users)
    ur = []
    for i in range(N):
        col = pg[:, i]
        s = int(np.argmax(col)) if serving is None else int(serving[i])
        ur.append({
            "position": [round(float(c), 2) for c in users[i]],
            "serving_uav": s + 1,
            "best_path_gain_db": float(col[s]),
            "per_uav": [{"uav_id": f"uav_{k+1}", "uav_position": [round(float(c), 2) for c in positions[k]],
                         "path_gain_db": float(col[k])} for k in range(K)],
        })
    return {"num_uavs": K, "num_users": N, "user_results": ur}


def predict(positions, users, pg: np.ndarray, channel_mode: str, freq_hz: float) -> dict:
    """Link-level prediction for one configuration (network_model link budget,
    the same budget ns-3 applies). In distance mode the obstruction is dropped,
    because the ns-3 channel will not see it. In both modes users are assigned
    to the UAV with the strongest RF path gain (as in the existing pipeline)."""
    pos = np.asarray(positions, float)
    us = np.asarray(users, float)
    pg = np.array(pg, float)
    serving = [int(s) for s in np.argmax(pg, axis=0)]
    if channel_mode == "distance":
        d = np.linalg.norm(pos[:, None, :] - us[None, :, :], axis=2)
        pg = np.where(pg > -199.0, -np.vectorize(nm.free_space_loss_db)(d, freq_hz), pg)
    rf = compose_rf(positions, users, pg, serving)
    net = nm.evaluate_network(rf, positions, users, freq_hz=freq_hz)
    served = net["aggregate"]["served_fraction"]
    pdr_u = np.array([served if p["usable"] else 0.0 for p in net["per_user"]])
    snr = np.array([p["net_snr_db"] for p in net["per_user"]])
    return {"rf": rf, "net": net, "pdr_u": pdr_u,
            "margin": np.clip((snr - nm.MIN_SNR_DB) / 20.0, -1.0, 1.0),
            "assoc": serving}


def predicted_score(pred: dict, obj_cfg: dict) -> float:
    """Predicted J without the latency term (latency is not predicted)."""
    w = weights_of(obj_cfg)
    p = pred["pdr_u"]
    n = len(p)
    worst = np.sort(p)[:max(1, int(math.ceil(0.10 * n)))]
    return float(w["coverage"] * np.mean(p >= obj_cfg["coverage_pdr_threshold"])
                 + w["pdr"] * np.mean(p) + w["fairness"] * np.mean(worst)
                 + obj_cfg["margin_weight"] * np.mean(pred["margin"]))


# ----------------------------------------------------------------------
# geometry and safety
# ----------------------------------------------------------------------
class SceneGeometry:
    """Building heights from ray casts on the loaded Sionna (Mitsuba) scene."""

    def __init__(self, mi_scene):
        self.mi_scene = mi_scene
        self._cache: dict = {}

    def roof_height(self, x: float, y: float, radius: float) -> float:
        key = (round(x, 1), round(y, 1), radius)
        if key not in self._cache:
            self._cache[key] = ob.max_height_within(self.mi_scene, x, y, radius)
        return self._cache[key]

    def users(self, region: dict, n: int, seed: int, user_z: float, clearance: float) -> list:
        return ob.street_level_users(self.mi_scene, region, n, seed, user_z=user_z, clearance_m=clearance)


class FlatGeometry:
    """Height map from a function (default: open ground). Tests / surrogate mode."""

    def __init__(self, height_fn: Callable[[float, float], float] = lambda x, y: 0.0):
        self.height_fn = height_fn

    def roof_height(self, x, y, radius):
        g = np.linspace(-radius, radius, 5)
        return float(max(self.height_fn(x + dx, y + dy) for dx in g for dy in g if dx * dx + dy * dy <= radius ** 2))

    def users(self, region, n, seed, user_z, clearance):
        rng = np.random.default_rng(seed)
        return [[round(float(rng.uniform(*region["x"])), 2), round(float(rng.uniform(*region["y"])), 2), float(user_z)]
                for _ in range(n)]


def safety_violations(pos: Sequence[float], others: Sequence[Sequence[float]], region: dict,
                      safety: dict, geometry, origin: Optional[Sequence[float]] = None) -> list[str]:
    """Reasons a UAV may not be at `pos` (empty list = safe). Checks scenario
    boundaries, minimum / maximum altitude, clearance above the highest roof
    within clearance_radius_m (at the destination and, for a move, the
    midpoint), and separation from every other UAV."""
    x, y, z = map(float, pos)
    out = []
    if not (region["x"][0] <= x <= region["x"][1] and region["y"][0] <= y <= region["y"][1]):
        out.append(f"outside scenario boundary ({x:.0f}, {y:.0f})")
    zmin = max(safety["min_altitude_m"], region.get("z", [-1e9, 1e9])[0])
    zmax = min(safety["max_altitude_m"], region.get("z", [-1e9, 1e9])[1])
    if z < zmin - 1e-9:
        out.append(f"below minimum safe altitude ({z:.1f} < {zmin:.1f} m)")
    if z > zmax + 1e-9:
        out.append(f"above maximum altitude ({z:.1f} > {zmax:.1f} m)")
    pts = [(x, y)]
    if origin is not None:
        pts.append(((x + origin[0]) / 2.0, (y + origin[1]) / 2.0))
    for px, py in pts:
        roof = geometry.roof_height(px, py, safety["clearance_radius_m"])
        if z < roof + safety["building_clearance_m"] - 1e-9:
            out.append(f"building clearance: {z:.1f} m < roof {roof:.1f} m + {safety['building_clearance_m']:.0f} m")
            break
    for o in others:
        d = math.dist((x, y, z), tuple(map(float, o)))
        if d < safety["min_separation_m"] - 1e-9:
            out.append(f"too close to another UAV ({d:.1f} m < {safety['min_separation_m']:.0f} m)")
            break
    return out


# ----------------------------------------------------------------------
# network measurement
# ----------------------------------------------------------------------
class Ns3Measurer:
    """ns-3 (ns3/uav-demo.cc) per-user delivery and latency; averaged over rng_runs."""

    label = "ns-3 packet-level simulation (simulated)"
    mock = False

    def __init__(self, binary: Path, freq_hz: float, traffic_seconds: float, rng_runs: Sequence[int],
                 workers: int = 2):
        import evaluate_obstruction as eo
        self.eo = eo
        self.binary, self.freq_hz = binary, float(freq_hz)
        self.traffic_seconds, self.rng_runs, self.workers = float(traffic_seconds), list(rng_runs), workers
        self._cache: dict[str, dict] = {}

    def measure(self, handoff: dict) -> dict:
        key = hashlib.sha256(json.dumps(handoff, sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        t0 = time.time()
        args = [(f"--RngRun={r}", f"--trafficSeconds={self.traffic_seconds:g}") for r in self.rng_runs]
        with cf.ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(args)))) as ex:
            runs = list(ex.map(lambda a: self.eo.run_ns3_detailed(self.binary, handoff, a), args))
        rx = np.mean([r["per_user_rx"] for r in runs], axis=0)
        sent = np.mean([r["per_user_sent"] for r in runs], axis=0)
        lat = []
        for i in range(handoff["num_users"]):
            v = [r["per_user_delay_ms"][i] for r in runs if r["per_user_delay_ms"][i] is not None]
            lat.append(float(np.mean(v)) if v else None)
        out = {"per_user_rx": rx.tolist(), "per_user_sent": sent.tolist(),
               "per_user_pdr": (rx / np.maximum(sent, 1)).tolist(), "per_user_latency_ms": lat,
               "obstruction_applied": all(r["obstruction_applied"] for r in runs),
               "rng_runs": self.rng_runs, "seconds": round(time.time() - t0, 2), "cached": False}
        self._cache[key] = out
        return out


class MockMeasurer:
    """NOT a network simulation. Deterministic stand-in for tests and UI checks:
    a user delivers 98 % of packets if the network_model link budget for its
    serving link (incl. obstruction when present) is usable, else 0 %; latency
    grows with the number of usable users sharing a UAV."""

    label = "MOCK network (deterministic stand-in, not ns-3)"
    mock = True

    def __init__(self, freq_hz: float, latency_per_user_ms: float = 2.0):
        self.freq_hz, self.lpu = float(freq_hz), latency_per_user_ms

    def measure(self, handoff: dict) -> dict:
        uavs, users = handoff["uav_positions"], handoff["user_positions"]
        assoc = handoff.get("associations", [0] * len(users))
        M = handoff.get("obstruction_loss_db")
        K, N = len(uavs), len(users)
        pg = np.zeros((K, N))
        for k in range(K):
            for i in range(N):
                d = math.dist(uavs[k], users[i])
                pg[k, i] = -nm.free_space_loss_db(d, self.freq_hz) - (M[k][i] if M else 0.0)
        ok = []
        for i in range(N):
            k = assoc[i]
            rss = nm.network_rss_dbm(pg[k, i], math.dist(uavs[k], users[i]), self.freq_hz)
            ok.append(rss >= nm.MIN_RSS_DBM and rss - nm.NS3_NOISE_DBM >= nm.MIN_SNR_DB)
        load = [sum(1 for i in range(N) if ok[i] and assoc[i] == k) for k in range(K)]
        sent = [100.0] * N
        rx = [98.0 if ok[i] else 0.0 for i in range(N)]
        lat = [5.0 + self.lpu * load[assoc[i]] if ok[i] else None for i in range(N)]
        return {"per_user_rx": rx, "per_user_sent": sent, "per_user_pdr": [r / 100.0 for r in rx],
                "per_user_latency_ms": lat, "obstruction_applied": M is not None,
                "rng_runs": [], "seconds": 0.0, "cached": False}


# ----------------------------------------------------------------------
# controller
# ----------------------------------------------------------------------
class StopLoop(Exception):
    pass


class ClosedLoopController:
    def __init__(self, cfg: dict, rf, geometry, measurer, users: list, uav_positions: list,
                 region: dict, user_region: dict, freq_hz: float, out_dir: Optional[Path] = None,
                 meta: Optional[dict] = None, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.rf, self.geometry, self.measurer = rf, geometry, measurer
        self.users = [list(map(float, u)) for u in users]
        self.region, self.user_region, self.freq_hz = region, user_region, float(freq_hz)
        self.out_dir = Path(out_dir) if out_dir else None
        self.log = log
        self.positions = [[round(float(c), 2) for c in p] for p in uav_positions]
        self.K = len(self.positions)
        self.step_idx = 0
        self.trace: list[tuple[int, str]] = []
        self.history: list[dict] = []
        self.decisions: list[dict] = []
        self.event_log: list[dict] = []
        self.tabu: dict[tuple[int, str], int] = {}
        self.residual: dict[str, list[float]] = {a: [] for a in ACTIONS}
        self.uav_status = [{"status": "initial", "last_action": None, "reason": "initial position"}
                           for _ in range(self.K)]
        self.version = 0
        self.status = "starting"
        self._prev_status = "running"
        self.phase = "init"
        self.phase_detail = ""
        self.error = None
        self.before = None
        self.meta = meta or {}
        self.timing = {"rf_s": 0.0, "network_s": 0.0, "rf_calls": 0, "network_calls": 0}
        self.idle_steps = 0
        self.cur = None   # current observation
        self.last_actions = None
        policy = cfg.get("policy", "rule_based")
        if policy not in ("rule_based", "ppo"):
            raise ValueError(f"policy must be rule_based|ppo, got {policy}")
        self.ppo = None
        if policy == "ppo":
            import ppo_controller
            self.ppo = ppo_controller.PPOPolicy(cfg.get("ppo") or {}, cfg)
        for k, p in enumerate(self.positions):
            v = safety_violations(p, self.positions[:k] + self.positions[k + 1:], region, cfg["safety"], geometry)
            if v:
                raise ValueError(f"initial position of UAV-{k+1} {p} is unsafe: {v}")

    # ---------------- control commands (file-based, checked between phases) --
    def _command(self) -> str:
        if not self.out_dir:
            return "run"
        c = ck.read_json(self.out_dir / "control.json") or {}
        return c.get("command", "run")

    def _checkpoint(self, phase: str, detail: str = ""):
        self.phase, self.phase_detail = phase, detail
        self.trace.append((self.step_idx, phase))
        self._write_state()
        while True:
            cmd = self._command()
            if cmd == "stop":
                raise StopLoop()
            if cmd != "pause":
                break
            if self.status != "paused":
                self._prev_status, self.status = self.status, "paused"
                self._write_state()
            time.sleep(0.2)
        if self.status == "paused":
            self.status = self._prev_status
            self._write_state()

    # ---------------- observation -------------------------------------------
    def observe(self, positions: list) -> dict:
        """Sionna (or surrogate) at `positions` + network measurement."""
        t0 = time.time()
        pg = self.rf.path_gain(positions, self.users)
        self.timing["rf_s"] += time.time() - t0
        self.timing["rf_calls"] += 1
        pred = predict(positions, self.users, pg, self.cfg["channel_mode"], self.freq_hz)
        assoc = pred["assoc"]
        M, mstats = ob.obstruction_matrix_db(compose_rf(positions, self.users, pg, assoc),
                                             positions, self.users, self.freq_hz)
        handoff = mio.build_uav_positions(positions, self.users, associations=assoc)
        if self.cfg["channel_mode"] == "obstruction":
            handoff = ob.with_obstruction(handoff, M, self.freq_hz)
        t0 = time.time()
        meas = self.measurer.measure(handoff)
        self.timing["network_s"] += time.time() - t0
        self.timing["network_calls"] += 1
        if meas["obstruction_applied"] != (self.cfg["channel_mode"] == "obstruction"):
            raise RuntimeError(f"obstruction applied={meas['obstruction_applied']} but channel_mode="
                               f"{self.cfg['channel_mode']}")
        metrics = network_metrics(meas["per_user_pdr"], meas["per_user_latency_ms"],
                                  meas["per_user_rx"], meas["per_user_sent"], self.cfg["objective"])
        return {"positions": [list(p) for p in positions], "users": [list(u) for u in self.users],
                "pg": pg, "pred": pred, "assoc": assoc,
                "matrix": M, "matrix_stats": mstats, "meas": meas, "metrics": metrics}

    # ---------------- evaluation --------------------------------------------
    def diagnose(self, obs: dict) -> list[dict]:
        oc = self.cfg["objective"]
        pdr = obs["meas"]["per_user_pdr"]
        lat = obs["meas"]["per_user_latency_ms"]
        out = []
        for k in range(self.K):
            S = [i for i, a in enumerate(obs["assoc"]) if a == k]
            poor = [i for i in S if pdr[i] < oc["pdr_target"]]
            dead = [i for i in S if pdr[i] <= 0.0]
            lats = [lat[i] for i in S if lat[i] is not None and pdr[i] > 0]
            ml = float(np.mean(lats)) if lats else None
            reasons = []
            if poor:
                reasons.append(f"{len(poor)}/{len(S)} assigned users below PDR target {oc['pdr_target']:.0%}"
                               + (f" ({len(dead)} with no delivery)" if dead else ""))
            if ml is not None and ml > oc["latency_target_ms"]:
                reasons.append(f"mean latency {ml:.0f} ms > target {oc['latency_target_ms']:.0f} ms")
            served = [i for i in S if pdr[i] >= oc["pdr_target"]]
            foot = max((math.dist(obs["positions"][k][:2], obs["users"][i][:2]) for i in served), default=0.0)
            out.append({"uav": k, "assigned": S, "poor": poor, "dead": dead, "mean_latency_ms": ml,
                        "mean_pdr": float(np.mean([pdr[i] for i in S])) if S else None,
                        "needs_move": bool(reasons), "latency_only": bool(reasons) and not poor,
                        "reason": "; ".join(reasons) if reasons else "all assigned users meet targets",
                        "footprint_m": round(foot, 1)})
        return out

    # ---------------- selection ---------------------------------------------
    def _correction(self, action: str) -> float:
        r = self.residual[action]
        return float(np.mean(r) * len(r) / (len(r) + 1)) if r else 0.0

    def select(self, obs: dict, diag: list[dict]) -> tuple[dict, list[dict]]:
        """Per-UAV action choice. Returns ({k: action}, decision records)."""
        if self.ppo is not None:
            self._checkpoint("select", "PPO policy on the measured state")
            return self.ppo.select(self, obs)
        cfg, oc = self.cfg, self.cfg["objective"]
        flagged = [d["uav"] for d in diag if d["needs_move"]]
        records = {}
        cands: dict[int, list] = {}
        for k in range(self.K):
            if k not in flagged:
                records[k] = {"uav": k, "action": "stay", "outcome": "held", "reason": diag[k]["reason"]}
                continue
            others = self.positions[:k] + self.positions[k + 1:]
            lst, blocked = [], []
            for a, (dx, dy, dz) in ACTIONS.items():
                if a == "stay":
                    continue
                p = self.positions[k]
                q = [round(p[0] + dx * cfg["step_h_m"], 2), round(p[1] + dy * cfg["step_h_m"], 2),
                     round(p[2] + dz * cfg["step_v_m"], 2)]
                if self.tabu.get((k, a), -1) >= self.step_idx:
                    blocked.append(f"{a}: recently reverted")
                    continue
                v = safety_violations(q, others, self.region, cfg["safety"], self.geometry, origin=p)
                if v:
                    blocked.append(f"{a}: {v[0]}")
                    continue
                lst.append((a, q))
            cands[k] = lst
            records[k] = {"uav": k, "blocked": blocked, "diag_reason": diag[k]["reason"]}
        if not any(cands.values()):
            for k in flagged:
                records[k].update({"action": "stay", "outcome": "held",
                                   "reason": diag[k]["reason"] + "; no safe candidate move"})
            return {}, [records[k] for k in range(self.K)]

        # one RF call for the current positions + every candidate position
        allpos = [list(p) for p in self.positions]
        index = {}
        for k, lst in cands.items():
            for a, q in lst:
                index[(k, a)] = len(allpos)
                allpos.append(q)
        self._checkpoint("select", f"ray tracing {len(allpos) - self.K} candidate positions")
        t0 = time.time()
        pg_all = self.rf.path_gain(allpos, self.users)
        self.timing["rf_s"] += time.time() - t0
        self.timing["rf_calls"] += 1
        cur_rows = list(range(self.K))
        base = predicted_score(predict(self.positions, self.users, pg_all[cur_rows],
                                       cfg["channel_mode"], self.freq_hz), oc)

        def score(rows, pos):
            return predicted_score(predict(pos, self.users, pg_all[rows], cfg["channel_mode"], self.freq_hz), oc)

        choice: dict[int, tuple] = {}
        for k, lst in cands.items():
            best = None
            for a, q in lst:
                rows, pos = cur_rows.copy(), [list(p) for p in self.positions]
                rows[k], pos[k] = index[(k, a)], q
                model = score(rows, pos) - base
                corr = self._correction(a)
                cost = oc["move_cost_per_m"] * math.dist(q, self.positions[k])
                est = model + corr - cost
                entry = (est, a, q, model, corr, cost)
                if best is None or est > best[0]:
                    best = entry
                records[k].setdefault("candidates", []).append(
                    {"action": a, "to": q, "predicted_model": round(model, 4), "correction": round(corr, 4),
                     "move_cost": round(cost, 4), "estimate": round(est, 4)})
            if best is None:
                continue
            est, a, q, model, corr, cost = best
            if est >= cfg["min_predicted_gain"]:
                choice[k] = (a, q, est, model, corr, "predicted gain")
            elif diag[k]["latency_only"]:
                # latency is not predicted: try the move whose predicted effect on
                # coverage/PDR/fairness is least harmful, judged by measurement
                safe = [c for c in records[k]["candidates"] if c["predicted_model"] >= -cfg["probe_tolerance"]]
                if safe:
                    c = max(safe, key=lambda c: c["estimate"])
                    choice[k] = (c["action"], c["to"], c["estimate"], c["predicted_model"], c["correction"],
                                 "trial (latency not predicted)")
        # at most one latency trial per interval, and only if nothing else moves
        trials = [k for k, c in choice.items() if c[5].startswith("trial")]
        if trials and len(trials) < len(choice):
            for k in trials:
                choice.pop(k)
        elif len(trials) > 1:
            keep = max(trials, key=lambda k: choice[k][2])
            for k in trials:
                if k != keep:
                    choice.pop(k)

        # joint feasibility: separation between moved UAVs, and joint prediction
        moves = dict(choice)
        for k in sorted(moves, key=lambda k: moves[k][2]):
            new = [moves[j][1] if j in moves else self.positions[j] for j in range(self.K)]
            if safety_violations(new[k], new[:k] + new[k + 1:], self.region, cfg["safety"], self.geometry,
                                 origin=self.positions[k]):
                moves.pop(k)
                records[k]["joint_note"] = "dropped: conflicts with another UAV's move"
        joint_pred = None
        if len(moves) > 1:
            rows, pos = cur_rows.copy(), [list(p) for p in self.positions]
            for k, (a, q, *_rest) in moves.items():
                rows[k], pos[k] = index[(k, a)], q
            joint_pred = score(rows, pos) - base
            if joint_pred <= 0 and not all(m[5].startswith("trial") for m in moves.values()):
                keep = max(moves, key=lambda k: moves[k][2])
                for k in list(moves):
                    if k != keep:
                        records[k]["joint_note"] = (f"dropped: joint prediction {joint_pred:+.4f} <= 0; "
                                                    f"only UAV-{keep+1} moves")
                        moves.pop(k)

        for k in range(self.K):
            r = records[k]
            if "outcome" in r:
                continue
            if k in moves:
                a, q, est, model, corr, kind = moves[k]
                p = self.positions[k]
                r.update({"action": a, "from": list(p), "to": q, "outcome": "pending", "kind": kind,
                          "predicted_delta_objective": round(est, 4), "predicted_model": round(model, 4),
                          "correction": round(corr, 4),
                          "reason": f"{r['diag_reason']}; {kind}: {a} predicted {est:+.4f}"
                                    f" (Sionna/link model {model:+.4f}, learned correction {corr:+.4f})"})
            else:
                best = max(r.get("candidates", []), key=lambda c: c["estimate"], default=None)
                why = r.get("joint_note") or (f"best candidate {best['action']} predicted {best['estimate']:+.4f}"
                                             f" < {cfg['min_predicted_gain']}" if best else "no safe candidate move")
                r.update({"action": "stay", "outcome": "held", "reason": f"{r['diag_reason']}; {why}"})
        if joint_pred is not None:
            for k in moves:
                records[k]["joint_predicted_model"] = round(joint_pred, 4)
        return {k: moves[k][0] for k in moves}, [records[k] for k in range(self.K)]

    # ---------------- events -------------------------------------------------
    def apply_events(self) -> bool:
        changed = False
        for ev in self.cfg.get("events", []):
            if int(ev.get("step", -1)) != self.step_idx or ev.get("_done"):
                continue
            if ev["type"] == "add_users":
                try:
                    new = self.geometry.users(ev["region"], int(ev["count"]), int(ev["seed"]),
                                              self.meta.get("user_z", 1.5), self.meta.get("user_clearance", 3.0))
                except RuntimeError as e:   # e.g. not enough open ground: recorded, not hidden
                    ev["_done"] = True
                    rec = {"step": self.step_idx, "type": "add_users", "count": 0, "failed": str(e),
                           "description": f"event not applied: {e}"}
                    self.event_log.append(rec)
                    self.log(f"[step {self.step_idx}] EVENT FAILED {e}")
                    continue
                self.users += [list(map(float, u)) for u in new]
                rec = {"step": self.step_idx, "type": "add_users", "count": len(new), "region": ev["region"],
                       "description": ev.get("description", f"{len(new)} new survivors detected")}
            elif ev["type"] == "remove_users":
                n = int(ev["count"])
                self.users = self.users[:-n] if n < len(self.users) else self.users
                rec = {"step": self.step_idx, "type": "remove_users", "count": n,
                       "description": ev.get("description", f"{n} survivors evacuated")}
            else:
                raise ValueError(f"unknown event type {ev['type']}")
            ev["_done"] = True
            self.event_log.append(rec)
            self.log(f"[step {self.step_idx}] EVENT {rec['description']}")
            changed = True
        return changed

    def _pending_events(self) -> bool:
        return any(not e.get("_done") and int(e.get("step", -1)) >= self.step_idx for e in self.cfg.get("events", []))

    # ---------------- one control interval -----------------------------------
    def step(self) -> dict:
        cfg = self.cfg
        self.step_idx += 1
        t = self.step_idx
        if self.apply_events():
            self._checkpoint("observe", "event: re-measuring the current positions")
            self.cur = self.observe(self.positions)
            self.history.append(self._hist_entry("event", [], True))
        self._checkpoint("observe", "current state")
        obs = self.cur
        self._checkpoint("evaluate", "objective and per-UAV diagnostics")
        diag = self.diagnose(obs)
        self._diag = diag
        moves, records = self.select(obs, diag)
        self.last_actions = [r["action"] for r in records]
        for r in records:
            r["step"] = t
            r["uav_id"] = f"UAV-{r['uav'] + 1}"
            r["metrics_before"] = {k: obs["metrics"].get(k) for k in METRIC_KEYS}
        if not moves:
            for r in records:
                r["measured_delta"] = None
                self.uav_status[r["uav"]] = {"status": "holding", "last_action": "stay", "reason": r["reason"]}
            self.decisions += records
            self.idle_steps += 1
            self.history.append(self._hist_entry("hold", [], True))
            self._checkpoint("feedback", "no UAV moved; measurement unchanged (deterministic simulation)")
            return {"moved": [], "accepted": True}
        self.idle_steps = 0
        old_pos = [list(p) for p in self.positions]
        for k, a in moves.items():
            self.uav_status[k] = {"status": "moving", "last_action": a,
                                  "reason": next(r["reason"] for r in records if r["uav"] == k)}
        new_pos = [next(r["to"] for r in records if r["uav"] == k) if k in moves else old_pos[k]
                   for k in range(self.K)]
        self._checkpoint("apply", ", ".join(f"UAV-{k+1} {a}" for k, a in moves.items()))
        self.positions = new_pos
        self._checkpoint("measure", "Sionna + network at the new positions")
        new = self.observe(self.positions)
        d = metric_delta(new["metrics"], obs["metrics"])
        # PPO moves are executed as chosen; accept / revert is part of the rule-based method
        accepted = self.ppo is not None or d["objective"] >= -cfg["acceptance_tolerance"]
        moved_recs = [r for r in records if r["uav"] in moves]
        joint_model = moved_recs[0].get("joint_predicted_model",
                                        sum(r.get("predicted_model", 0.0) for r in moved_recs))
        residual = d["objective"] - joint_model
        for k, a in moves.items():
            self.residual[a].append(residual / len(moves))
        for r in records:
            r["measured_delta"] = d if r["uav"] in moves else None
            if r["uav"] in moves and self.ppo is not None:
                r["outcome"] = "executed"
                r["result"] = f"measured objective {d['objective']:+.4f} (PPO: executed, not reverted)"
            elif r["uav"] in moves:
                r["outcome"] = "accepted" if accepted else "reverted"
                r["result"] = (f"measured objective {d['objective']:+.4f} "
                               + ("kept" if accepted else f"< -{cfg['acceptance_tolerance']}: flown back, "
                                  f"move banned for {cfg['tabu_steps']} intervals"))
        if accepted:
            self.cur = new
            for k, a in moves.items():
                self.uav_status[k]["status"] = "moved"
        else:
            self.positions = old_pos
            for k, a in moves.items():
                self.tabu[(k, a)] = t + cfg["tabu_steps"]
                self.uav_status[k] = {"status": "reverted", "last_action": a,
                                      "reason": f"{a} reverted: measured objective {d['objective']:+.4f}"}
        self.decisions += records
        self.history.append(self._hist_entry("accepted" if accepted else "reverted", list(moves), accepted,
                                             trial_metrics=new["metrics"]))
        self._checkpoint("feedback", ("accepted" if accepted else "reverted") +
                         f": measured objective {d['objective']:+.4f}")
        return {"moved": list(moves), "accepted": accepted, "delta": d}

    def _hist_entry(self, kind, moved, accepted, trial_metrics=None) -> dict:
        return {"step": self.step_idx, "kind": kind, "moved": [f"UAV-{k+1}" for k in moved],
                "accepted": accepted, "positions": [list(p) for p in self.positions],
                "num_users": len(self.users), "metrics": dict(self.cur["metrics"]),
                "actions": list(self.last_actions) if kind != "initial" and self.last_actions else None,
                "trial_metrics": dict(trial_metrics) if trial_metrics else None,
                "time": round(time.time(), 2)}

    # ---------------- main loop ----------------------------------------------
    def run(self) -> dict:
        try:
            self.status = "running"
            self._checkpoint("observe", "initial state")
            self.cur = self.observe(self.positions)
            self.before = {"positions": [list(p) for p in self.positions], "metrics": dict(self.cur["metrics"]),
                           "num_users": len(self.users)}
            self.history.append(self._hist_entry("initial", [], True))
            while self.step_idx < self.cfg["max_steps"]:
                self.step()
                if self.idle_steps >= self.cfg["patience"] and not self._pending_events():
                    self.status = "converged"
                    break
                time.sleep(self.cfg.get("step_interval_s", 0))
            if self.status == "running":
                self.status = "finished"
            self.phase, self.phase_detail = "done", ""
        except StopLoop:
            self.status = "stopped"
        except Exception as e:  # recorded in the state, re-raised for the CLI
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
            self._write_state()
            raise
        self._write_state()
        return self.state()

    # ---------------- state ---------------------------------------------------
    def state(self) -> dict:
        cur = self.cur
        diag = self.diagnose(cur) if cur else None
        users, uavs = [], []
        if cur:
            m = cur["meas"]
            for i, u in enumerate(cur["users"]):
                k = cur["assoc"][i]
                users.append({"id": f"S-{i+1}", "position": u, "serving_uav": k + 1,
                              "pdr": round(m["per_user_pdr"][i], 4),
                              "latency_ms": None if m["per_user_latency_ms"][i] is None
                              else round(m["per_user_latency_ms"][i], 2),
                              "path_gain_db": round(float(cur["pg"][k, i]), 2),
                              "obstruction_db": round(float(cur["matrix"][k][i]), 2),
                              "predicted_usable": bool(cur["pred"]["net"]["per_user"][i]["usable"]),
                              "net_snr_db": cur["pred"]["net"]["per_user"][i]["net_snr_db"]})
        for k in range(self.K):
            e = {"id": f"UAV-{k+1}", "position": self.positions[k], "altitude_m": self.positions[k][2],
                 "measured_at": cur["positions"][k] if cur else None,
                 **self.uav_status[k]}
            if diag:
                d = diag[k]
                e.update({"assigned_users": len(d["assigned"]), "poor_users": len(d["poor"]),
                          "no_delivery_users": len(d["dead"]), "mean_pdr": d["mean_pdr"],
                          "mean_latency_ms": d["mean_latency_ms"], "needs_move": d["needs_move"],
                          "diagnosis": d["reason"], "footprint_m": d["footprint_m"]})
            uavs.append(e)
        prov = {
            "rf": self.rf.label, "network": self.measurer.label, "mock": bool(self.measurer.mock),
            "live_physical_data": False,
            "policy": self.ppo.info if self.ppo is not None else {"policy": "rule_based"},
            "note": "All values come from simulation (ray tracing + network simulation) of a digital twin. "
                    "No physical UAV or live telemetry is connected.",
        }
        return {
            "schema": SCHEMA_VERSION, "version": self.version, "run_id": self.meta.get("run_id"),
            "status": self.status, "phase": self.phase, "phase_detail": self.phase_detail,
            "step": self.step_idx, "max_steps": self.cfg["max_steps"], "error": self.error,
            "provenance": prov, "channel_mode": self.cfg["channel_mode"],
            "config": {k: v for k, v in self.cfg.items() if k != "events"},
            "weights": weights_of(self.cfg["objective"]),
            "scene": {"region": self.region, "user_region": self.user_region,
                      "scene": self.meta.get("scene"), "freq_hz": self.freq_hz,
                      "ns3_channel": self.meta.get("ns3_channel")},
            "uavs": uavs, "users": users,
            "metrics": dict(cur["metrics"]) if cur else None,
            "obstruction": cur["matrix_stats"] if cur else None,
            "history": self.history, "decisions": self.decisions[-300:], "num_decisions": len(self.decisions),
            "events": self.event_log,
            "pending_events": [{"step": e["step"], "type": e["type"], "description": e.get("description")}
                               for e in self.cfg.get("events", []) if not e.get("_done")],
            "before": self.before,
            "after": {"positions": [list(p) for p in self.positions], "metrics": dict(cur["metrics"]),
                      "num_users": len(self.users)} if cur else None,
            "timing": {k: round(v, 2) if isinstance(v, float) else v for k, v in self.timing.items()},
            "updated_at": time.time(),
        }

    def _write_state(self):
        self.version += 1
        if self.out_dir:
            ck.atomic_write_json(self.out_dir / "state.json", self.state())


# ----------------------------------------------------------------------
# construction from a config file
# ----------------------------------------------------------------------
def build(cfg: dict, out_dir: Optional[Path], run_id: str, log=print, ns3_root: Optional[Path] = None):
    scen_cfg = json.loads((_ROOT / cfg["scenario_config"]).read_text())
    exp = scen_cfg["experiment"]
    sid = exp["scenario"]
    scen = scen_cfg["scenarios"][sid]
    meta = {"run_id": run_id, "scene": scen["scene"], "scenario": sid,
            "user_z": exp["user_z_m"], "user_clearance": exp["user_clearance_m"]}

    binary = None
    if cfg["network"] == "ns3":
        import evaluate_obstruction as eo
        root = ns3_root or Path(os.environ.get("NS3_PATH") or Path.home() / "ns-allinone-3.47" / "ns-3.47")
        binary = eo.find_binary(root)
        chan = eo.ns3_channel(binary)
        freq_hz = chan["frequency_mhz"] * 1e6
        meta["ns3_channel"] = chan
        measurer = Ns3Measurer(binary, freq_hz, cfg["ns3"]["traffic_seconds"], cfg["ns3"]["rng_runs"],
                               cfg["ns3"].get("workers", 2))
    elif cfg["network"] == "mock":
        freq_hz = 2.412e9
        measurer = MockMeasurer(freq_hz)
    else:
        raise ValueError(f"network must be ns3|mock, got {cfg['network']}")

    if cfg["rf_mode"] == "sionna":
        rf = SionnaRF(scen["scene"], freq_hz)
        geometry = SceneGeometry(rf.mi_scene)
        xy = ob.roof_clear_uav_xy(rf.mi_scene, scen["centre"], exp["uav_offsets_m"][:cfg["num_uavs"]],
                                  search_m=exp["uav_snap_search_m"], radius_m=exp["roof_radius_m"])
        xy = [(u["x"], u["y"]) for u in xy]
    elif cfg["rf_mode"] == "surrogate":
        rf = SurrogateRF(freq_hz)
        geometry = FlatGeometry()
        xy = [(scen["centre"][0] + o[0], scen["centre"][1] + o[1]) for o in exp["uav_offsets_m"][:cfg["num_uavs"]]]
    else:
        raise ValueError(f"rf_mode must be sionna|surrogate, got {cfg['rf_mode']}")
    if cfg["channel_mode"] not in ("obstruction", "distance"):
        raise ValueError("channel_mode must be obstruction|distance")

    users = geometry.users(scen["user_region"], cfg["num_users"], cfg["user_seed"],
                           exp["user_z_m"], exp["user_clearance_m"])
    s = cfg["safety"]
    region = scen["uav_region"]
    positions = []
    for x, y in xy:
        floor = geometry.roof_height(x, y, s["clearance_radius_m"]) + s["building_clearance_m"]
        z = max(cfg["initial_altitude_m"], s["min_altitude_m"], floor)
        if z != cfg["initial_altitude_m"]:
            log(f"initial altitude at ({x}, {y}) raised to {z:.1f} m (safety floor)")
        positions.append([x, y, z])
    return ClosedLoopController(cfg, rf, geometry, measurer, users, positions, region, scen["user_region"],
                                freq_hz, out_dir=out_dir, meta=meta, log=log)


def load_config(path: Optional[str], overrides: Optional[dict] = None) -> dict:
    cfg = DEFAULT_CONFIG
    if path:
        cfg = deep_merge(cfg, json.loads(Path(path).read_text()))
    cfg = deep_merge(cfg, overrides)
    weights_of(cfg["objective"])   # validate
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "closed_loop_control.json"))
    ap.add_argument("--override", default=None, help="JSON object merged over the config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "state.json").exists():
        raise SystemExit(f"{out} already holds a run; choose a new --out (results are never overwritten)")
    cfg = load_config(args.config, json.loads(args.override) if args.override else None)
    run_id = args.run_id or out.name
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    ck.atomic_write_json(out / "config.json", cfg)
    if not (out / "control.json").exists():
        ck.atomic_write_json(out / "control.json", {"command": "run"})
    ctl = build(cfg, out, run_id, log=log)
    log(f"START {run_id}: rf={ctl.rf.label}; network={ctl.measurer.label}; channel={cfg['channel_mode']}; "
        f"{ctl.K} UAVs, {len(ctl.users)} users")

    orig_step = ctl.step

    def logged_step():
        r = orig_step()
        for d in ctl.decisions[-ctl.K:]:
            log(f"[step {d['step']}] {d['uav_id']} {d['action']:>5} {d['outcome']:>8}: {d['reason']}"
                + (f" -> {d['result']}" if d.get("result") else ""))
        m = ctl.cur["metrics"]
        log(f"[step {ctl.step_idx}] J={m['objective']:.4f} cov={m['coverage']:.3f} pdr={m['pdr']:.3f} "
            f"worst10={m['worst_decile_pdr']:.3f} lat={m['mean_latency_ms']} pos={ctl.positions}")
        return r

    ctl.step = logged_step
    st = ctl.run()
    log(f"END status={st['status']} before={st['before']['metrics']} after={st['after']['metrics']}")


if __name__ == "__main__":
    main()
