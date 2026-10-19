"""3D multi-UAV positioning environment: every UAV independently chooses one of
7 discrete moves per step (stay, -X, +X, -Y, +Y, down, up).

Differences from multiuav_env.MultiUAVEnv (which stays unchanged):
  * 7 actions per UAV, horizontal step `step_h_m` and vertical step `step_v_m`
    (both configurable); UAV positions live on that lattice.
  * Safety is the closed-loop controller's own rule (closed_loop_control.
    safety_violations): scenario boundary, min / max altitude, building
    clearance at the destination and the move midpoint, UAV separation.
    Unsafe single moves are masked (action_masks(), for MaskablePPO) and, if
    chosen anyway, replaced by "stay" and counted. Simultaneous moves that
    would bring two UAVs (or their move midpoints) closer than the minimum
    separation are resolved by cancelling the higher-index UAV's move.
  * Reward (all terms in [0, 1], weights in DEFAULT_REWARD_WEIGHTS):
        score = w_pdr * mean PDR + w_cov * coverage(PDR >= thr) + w_fair * worst-10 % PDR
                - w_lat * latency score             (only if w_lat > 0, see below)
        r     = score + w_delta * (score - previous score)
                - w_move * metres moved / (active UAVs * max step)
                - w_unsafe * unsafe attempts / active UAVs
    No term depends on altitude. Latency score = mean predicted per-user delay /
    LATENCY_REF_MS, capped at LATENCY_CAP: the closed-loop objective's term
    (closed_loop_control.network_metrics), from the latency predictor that passed
    its pre-registered validation (ml/fit_latency_model.py). w_lat is 0 by default
    (earlier runs); latency is not in the observation. Throughput under the fixed
    offered load is offered * PDR, so it is logged, not rewarded.
  * Per-user PDR comes from a pluggable network model (calibrated model by
    default) on the path gains of a pluggable RF backend: rf.path_gain(P, U)
    -> [K, N] dB (free space for tests / smoke; Sionna radio maps for training).
    In the closed loop the same observation is built from ns-3 measurements.

Observation (OBS_LEN floats, slots for MAX_UAVS UAVs, zero-padded):
  per UAV (PER_UAV): active, x, y, z (normalised to the region), clearance
    above the highest roof within the clearance radius, 6 move-blocked flags,
    share of users served, mean / worst PDR of its users, mean obstruction
    excess loss of its links, share of its links with > 10 dB excess,
    distance to the nearest other UAV
  global (6): mean PDR, coverage, worst-10 % PDR, users / MAX_USERS,
    UAVs / MAX_UAVS, step / max_steps
  grids (2 x GRID x GRID): user density, density of users with PDR < threshold
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import network_model as nm  # noqa: E402
from closed_loop_control import safety_violations, compose_rf  # noqa: E402

OBS_VERSION = "3d-v1"
MAX_UAVS = 5
MAX_USERS = 50
GRID = 6
PER_UAV = 17
OBS_LEN = PER_UAV * MAX_UAVS + 6 + 2 * GRID * GRID

ACTION_NAMES = ["stay", "-x", "+x", "-y", "+y", "down", "up"]
# unit move per action: (dx, dy, dz) in units of (step_h, step_h, step_v)
ACTION_DELTAS = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)
N_ACTIONS = len(ACTION_NAMES)

DEFAULT_SAFETY = {"min_altitude_m": 15.0, "max_altitude_m": 60.0, "building_clearance_m": 5.0,
                  "clearance_radius_m": 10.0, "min_separation_m": 15.0}
DEFAULT_REWARD_WEIGHTS = {"pdr": 0.35, "coverage": 0.20, "fairness": 0.25, "delta": 0.10,
                          "move": 0.03, "unsafe": 0.05, "latency": 0.0}
LATENCY_REF_MS = 100.0              # = closed-loop objective latency_ref_ms / latency_cap
LATENCY_CAP = 3.0
TRAINING_FREQ_HZ = 2.412e9          # ns-3 Wi-Fi channel 1 (uav-demo.cc --printChannel)
CALIBRATED_FREQ_HZ = 2.1e9          # calibrated_model computes excess loss at this frequency


# ----------------------------------------------------------------------
# geometry from a height map (fast stand-in for closed_loop_control.SceneGeometry)
# ----------------------------------------------------------------------
class HeightmapGeometry:
    """Roof heights and street-level users from a regular height map
    (ml/scan_training_areas.py output: xs, ys, H[ny, nx]). Same interface as
    closed_loop_control.SceneGeometry, so safety_violations works unchanged.
    roof_height takes the highest map cell within the radius (plus the cell
    under the point), so it can only be as fine as the map resolution; the
    closed loop re-checks every move with ray casts on the real scene."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, H: np.ndarray):
        self.xs, self.ys, self.H = np.asarray(xs, float), np.asarray(ys, float), np.asarray(H, float)
        self.res = float(self.xs[1] - self.xs[0])
        self._cache: dict = {}

    @classmethod
    def load(cls, path: str | Path) -> "HeightmapGeometry":
        d = np.load(Path(path).expanduser())
        return cls(d["xs"], d["ys"], d["H"])

    def _ix(self, v, axis):
        return np.clip(np.round((np.asarray(v, float) - axis[0]) / self.res).astype(int), 0, len(axis) - 1)

    def height(self, x: float, y: float) -> float:
        return float(self.H[self._ix(y, self.ys), self._ix(x, self.xs)])

    def roof_height(self, x: float, y: float, radius: float) -> float:
        key = (round(x, 1), round(y, 1), radius)
        if key not in self._cache:
            r = int(math.ceil(radius / self.res))
            ix, iy = int(self._ix(x, self.xs)), int(self._ix(y, self.ys))
            best = float(self.H[iy, ix])
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < len(self.xs) and 0 <= jy < len(self.ys) and \
                            math.hypot(self.xs[jx] - x, self.ys[jy] - y) <= radius:
                        best = max(best, float(self.H[jy, jx]))
            self._cache[key] = best
        return self._cache[key]

    def users(self, region: dict, n: int, seed: int, user_z: float = 1.5, clearance: float = 3.0,
              min_spacing: float = 4.0) -> list:
        """n users on open ground (map cell < 0.5 m and every cell within
        `clearance` also open), uniformly over such cells, jittered inside the cell."""
        rng = np.random.default_rng(seed)
        ix0, ix1 = self._ix(region["x"][0], self.xs), self._ix(region["x"][1], self.xs)
        iy0, iy1 = self._ix(region["y"][0], self.ys), self._ix(region["y"][1], self.ys)
        r = int(math.ceil(clearance / self.res))
        sub = self.H[max(0, iy0 - r):iy1 + r + 1, max(0, ix0 - r):ix1 + r + 1] < 0.5
        oy, ox = iy0 - max(0, iy0 - r), ix0 - max(0, ix0 - r)
        ok = []
        for jy in range(iy1 - iy0 + 1):
            for jx in range(ix1 - ix0 + 1):
                if sub[oy + jy - r:oy + jy + r + 1, ox + jx - r:ox + jx + r + 1].all():
                    ok.append((ix0 + jx, iy0 + jy))
        if len(ok) < n:
            raise ValueError(f"only {len(ok)} open cells in region for {n} users")
        out: list = []
        for j in rng.permutation(len(ok)):
            jx, jy = ok[j]
            p = [float(self.xs[jx] + rng.uniform(-self.res / 4, self.res / 4)),
                 float(self.ys[jy] + rng.uniform(-self.res / 4, self.res / 4)), float(user_z)]
            if all(math.dist(p[:2], q[:2]) >= min_spacing for q in out):
                out.append([round(p[0], 2), round(p[1], 2), p[2]])
            if len(out) == n:
                return out
        raise ValueError("could not place users with the required spacing")


# ----------------------------------------------------------------------
# network model
# ----------------------------------------------------------------------
def excess_loss_db(pg: np.ndarray, d: np.ndarray, freq_hz: float) -> np.ndarray:
    """Obstruction loss beyond free space (the quantity ns-3 receives); 200 dB if no path."""
    fspl = 20.0 * np.log10(4.0 * np.pi * np.maximum(d, 1.0) * freq_hz / nm.C_LIGHT)
    return np.where(pg <= -199.0, 200.0, np.maximum(0.0, -pg - fspl))


class NetworkPDR:
    """Per-user PDR from network_model (link budget + shared-channel airtime)."""
    name = "network_model"

    def __init__(self, freq_hz: float = TRAINING_FREQ_HZ):
        self.freq_hz = freq_hz

    def __call__(self, P, U, pg) -> np.ndarray:
        rf = compose_rf(P.tolist(), U.tolist(), pg)
        net = nm.evaluate_network(rf, P, U, freq_hz=self.freq_hz)
        served = net["aggregate"]["served_fraction"]
        return np.array([served if p["usable"] else 0.0 for p in net["per_user"]])


class CalibratedPDR:
    """Per-user PDR from the ns-3-calibrated model (models/network_surrogate_v1).
    calibrated_model evaluates the link budget at 2.1 GHz; path gains at
    freq_hz are re-expressed so that the obstruction excess it computes equals
    the excess at freq_hz (exact; the ns-3 distance law is frequency-free)."""
    name = "calibrated_v1"

    def __init__(self, freq_hz: float = TRAINING_FREQ_HZ, path: Optional[str] = None,
                 latency_path: Optional[str] = None):
        """latency_path: optional latency predictor (ml/latency_model.py joblib); it
        shares this model's features, so predict() returns both at little extra cost."""
        import calibrated_model as cm
        self.cm = cm
        self.model = cm.CalibratedNetworkModel(path) if path else cm.CalibratedNetworkModel()
        self.freq_hz = freq_hz
        if path:
            self.name = f"calibrated:{path}"
        self.latency = None
        if latency_path:
            import latency_model as lm               # imports this module: load lazily
            self.latency = lm.LatencyModel(latency_path)

    def predict(self, P, U, pg) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """(per-user PDR, per-user mean delay [ms] or None without a latency model;
        NaN for users the link model says are unusable)."""
        d = np.linalg.norm(P[:, None, :] - U[None, :, :], axis=2)
        exc = excess_loss_db(pg, d, self.freq_hz)
        fspl_ref = 20.0 * np.log10(4.0 * np.pi * np.maximum(d, 1.0) * CALIBRATED_FREQ_HZ / nm.C_LIGHT)
        pg_ref = np.where(pg <= -199.0, -200.0, -(fspl_ref + exc))
        serving = np.argmax(pg, axis=0)               # association by the real path gain
        rf = compose_rf(P.tolist(), U.tolist(), pg_ref, serving.tolist())
        X, usable, net = self.cm.user_features(rf, P, U)
        pdr, lat = np.zeros(len(X)), None
        if usable.any():
            pdr[usable] = np.clip(self.model.model.predict(X[usable]), 0.0, 1.0)
        if self.latency is not None:                  # same features as latency_model.user_features
            cnt = np.bincount(serving, minlength=len(P))[serving]
            rate = np.array([p["mcs_rate_mbps"] for p in net["per_user"]])
            lat = np.full(len(X), np.nan)
            if usable.any():
                lat[usable] = np.exp(self.latency.model.predict(np.column_stack([X, cnt, rate])[usable]))
        return pdr, lat

    def __call__(self, P, U, pg) -> np.ndarray:
        return self.predict(P, U, pg)[0]


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------
class MultiUAV3DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, windows: dict, geometry, rf, pdr_model, *,
                 step_h_m: float = 15.0, step_v_m: float = 10.0, z_levels: Sequence[float] = (20, 30, 40, 50, 60),
                 safety: Optional[dict] = None, reward_weights: Optional[dict] = None,
                 coverage_threshold: float = 0.5, max_steps: int = 12,
                 num_uavs: Sequence[int] = (2, 3, 4), num_users: Sequence[int] = (20, 30, 40),
                 random_start_prob: float = 0.5, pdr_obs_noise: float = 0.0,
                 allow_held_out: bool = False, seed: Optional[int] = None):
        """windows: {name: {"uav_region": {...}, "user_region": {...}, "role": ...}}.
        Episodes sample a window uniformly. Held-out windows (role "held_out" or
        name "urban_core") are refused unless allow_held_out (evaluation only)."""
        super().__init__()
        for name, w in windows.items():
            if (name == "urban_core" or w.get("role") == "held_out") and not allow_held_out:
                raise ValueError(f"window {name!r} is held out for final evaluation")
        self.windows = windows
        self.geometry, self.rf, self.pdr_model = geometry, rf, pdr_model
        self.step_h, self.step_v = float(step_h_m), float(step_v_m)
        self.z_levels = [float(z) for z in z_levels]
        self.safety = {**DEFAULT_SAFETY, **(safety or {})}
        self.w = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        if self.w["latency"] > 0 and pdr_model is not None and getattr(pdr_model, "latency", None) is None:
            raise ValueError("latency reward weight > 0 needs a PDR model with a latency predictor")
        self.cov_thr = float(coverage_threshold)
        self.max_steps = int(max_steps)
        self.num_uavs_choices, self.num_users_choices = list(num_uavs), list(num_users)
        self.random_start_prob, self.pdr_obs_noise = float(random_start_prob), float(pdr_obs_noise)
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * MAX_UAVS)
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(OBS_LEN,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.K = 0

    # -- episode setup ---------------------------------------------------
    def _lattice(self, region):
        xs = np.arange(region["x"][0], region["x"][1] + 1e-6, self.step_h)
        ys = np.arange(region["y"][0], region["y"][1] + 1e-6, self.step_h)
        return xs, ys

    def _safe_static(self, p, origin=None) -> list:
        return safety_violations(p, [], self.region, self.safety, self.geometry, origin=origin)

    def _start_positions(self, K: int) -> np.ndarray:
        xs, ys = self._lattice(self.region)
        for _ in range(2000):
            if self._rng.random() < self.random_start_prob:      # independent random safe positions
                P = np.array([[self._rng.choice(xs), self._rng.choice(ys), self._rng.choice(self.z_levels)]
                              for _ in range(K)])
            else:                                                # compact formation at one random altitude
                cx = xs[len(xs) // 2 + self._rng.integers(-3, 4)]
                cy = ys[len(ys) // 2 + self._rng.integers(-3, 4)]
                z = self._rng.choice(self.z_levels)
                offs = [(0, 2), (-2, -2), (2, -2), (-2, 2), (2, 2)]
                P = np.array([[cx + offs[k][0] * self.step_h, cy + offs[k][1] * self.step_h, z] for k in range(K)])
            if all(not self._safe_static(p) for p in P) and \
                    all(math.dist(P[i], P[j]) >= self.safety["min_separation_m"]
                        for i in range(K) for j in range(i + 1, K)):
                return P
        raise RuntimeError("no safe start found")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        o = options or {}
        self.window = o.get("window") or str(self._rng.choice(sorted(self.windows)))
        w = self.windows[self.window]
        self.region = {**w["uav_region"], "z": w["uav_region"].get("z", [self.safety["min_altitude_m"],
                                                                         self.safety["max_altitude_m"]])}
        self.K = int(o.get("num_uavs") or self._rng.choice(self.num_uavs_choices))
        N = int(o.get("num_users") or self._rng.choice(self.num_users_choices))
        user_seed = int(o.get("user_seed", self._rng.integers(0, 2**31 - 1)))
        self.users = np.array(o["users"] if "users" in o else
                              self.geometry.users(w["user_region"], N, user_seed), float)
        self.P = np.array(o["positions"], float) if "positions" in o else self._start_positions(self.K)
        self.K = len(self.P)
        self.t = 0
        self._evaluate()
        self.prev_score = self.score
        return self._obs(), self._info(moved_m=0.0, unsafe=0, cancelled=0, actions=[0] * self.K)

    # -- dynamics --------------------------------------------------------
    def target(self, k: int, a: int) -> np.ndarray:
        d = ACTION_DELTAS[a]
        return self.P[k] + np.array([d[0] * self.step_h, d[1] * self.step_h, d[2] * self.step_v])

    def _move_ok(self, k: int, a: int) -> bool:
        if a == 0:
            return True
        t = self.target(k, a)
        others = [self.P[j] for j in range(self.K) if j != k]
        return not safety_violations(t, others, self.region, self.safety, self.geometry, origin=self.P[k])

    def action_masks(self) -> np.ndarray:
        """[MAX_UAVS * 7] bool. A move is allowed if it is safe against the
        other UAVs' current positions; inactive slots may only 'stay'."""
        m = np.zeros((MAX_UAVS, N_ACTIONS), bool)
        m[:, 0] = True
        for k in range(self.K):
            for a in range(1, N_ACTIONS):
                m[k, a] = self._move_ok(k, a)
        return m.ravel()

    def _apply(self, actions) -> tuple[np.ndarray, list, int, int]:
        acts = [int(actions[k]) for k in range(self.K)]
        unsafe = 0
        for k, a in enumerate(acts):                       # single-move safety
            if a and not self._move_ok(k, a):
                acts[k], unsafe = 0, unsafe + 1
        cancelled = 0
        sep = self.safety["min_separation_m"] - 1e-9
        while True:                                         # joint separation (ends and midpoints)
            T = np.array([self.target(k, a) for k, a in enumerate(acts)])
            M = (T + self.P) / 2.0
            bad = [(i, j) for i in range(self.K) for j in range(i + 1, self.K)
                   if (acts[i] or acts[j]) and (math.dist(T[i], T[j]) < sep or math.dist(M[i], M[j]) < sep)]
            if not bad:
                break
            i, j = bad[0]
            acts[j if acts[j] else i] = 0
            cancelled += 1
        return T, acts, unsafe, cancelled

    def step(self, action):
        action = np.asarray(action).reshape(-1)
        T, acts, unsafe, cancelled = self._apply(action)
        moved = float(np.linalg.norm(T - self.P, axis=1).sum())
        self.P = T
        self.t += 1
        self._evaluate()
        w = self.w
        move_norm = moved / (self.K * max(self.step_h, self.step_v))
        unsafe_norm = (unsafe + cancelled) / self.K
        delta = self.score - self.prev_score
        r = self.score + w["delta"] * delta - w["move"] * move_norm - w["unsafe"] * unsafe_norm
        self.reward_terms = {"score": self.score, "delta": delta, "move_norm": move_norm,
                             "unsafe_norm": unsafe_norm, "reward": r}
        self.prev_score = self.score
        trunc = self.t >= self.max_steps
        return self._obs(), float(r), False, trunc, self._info(moved, unsafe, cancelled, acts)

    # -- network evaluation / observation --------------------------------
    def predict_network(self, P) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """(path gain [K, N], per-user PDR, per-user delay [ms] or None) at positions P."""
        pg = np.asarray(self.rf.path_gain(P, self.users), float)
        if self.w["latency"] > 0:
            pdr, lat = self.pdr_model.predict(P, self.users, pg)
        else:
            pdr, lat = self.pdr_model(P, self.users, pg), None
        return pg, pdr, lat

    def score_of(self, metrics: dict) -> float:
        w = self.w
        s = w["pdr"] * metrics["pdr"] + w["coverage"] * metrics["coverage"] + w["fairness"] * metrics["worst10_pdr"]
        return s - w["latency"] * metrics["latency_score"] if w["latency"] > 0 else s

    def _evaluate(self) -> None:
        pg, pdr, lat = self.predict_network(self.P)
        self._set_network(pg, pdr, lat)

    def load_state(self, region: dict, positions, users, pg, per_user_pdr, t: int,
                   window: str = "closed_loop") -> np.ndarray:
        """Set the state from outside (closed loop: Sionna path gains and ns-3
        measured per-user PDR at the current positions) and return the
        observation the policy was trained on."""
        self.window = window
        self.region = {**region, "z": region.get("z", [self.safety["min_altitude_m"], self.safety["max_altitude_m"]])}
        self.P = np.asarray(positions, float)
        self.K = len(self.P)
        if self.K > MAX_UAVS:
            raise ValueError(f"policy supports at most {MAX_UAVS} UAVs")
        self.users = np.asarray(users, float)
        self.t = int(t)
        self._set_network(np.asarray(pg, float), per_user_pdr)
        return self._obs()

    def _set_network(self, pg: np.ndarray, pdr, latency_ms=None) -> None:
        self.pg = pg
        self.pdr = np.clip(np.asarray(pdr, float), 0.0, 1.0)
        self.serving = np.argmax(self.pg, axis=0)
        d = np.linalg.norm(self.P[:, None, :] - self.users[None, :, :], axis=2)
        self.excess = excess_loss_db(self.pg, d, TRAINING_FREQ_HZ)
        self.metrics = self.metrics_from(self.pdr, latency_ms)
        self.score = self.score_of(self.metrics)

    def metrics_from(self, pdr: np.ndarray, latency_ms=None) -> dict:
        n = len(pdr)
        worst = np.sort(pdr)[:max(1, int(math.ceil(0.10 * n)))]
        offered = nm.PACKETS_PER_SEC * nm.IP_PACKET_BYTES * 8 / 1e6
        m = {"pdr": float(pdr.mean()), "coverage": float(np.mean(pdr >= self.cov_thr)),
             "worst10_pdr": float(worst.mean()), "throughput_mbps": float(offered * pdr.mean()),
             "mean_latency_ms": None, "latency_score": 0.0}
        if latency_ms is not None:                    # as the closed loop: mean over delivering users
            lat = np.asarray(latency_ms, float)
            lat = lat[~np.isnan(lat) & (pdr > 0)]
            m["mean_latency_ms"] = float(lat.mean()) if len(lat) else None
            m["latency_score"] = LATENCY_CAP if not len(lat) else float(min(lat.mean() / LATENCY_REF_MS, LATENCY_CAP))
        return m

    def _obs(self) -> np.ndarray:
        rx, ry = self.region["x"], self.region["y"]
        zlo, zhi = self.safety["min_altitude_m"], self.safety["max_altitude_m"]
        pdr = self.pdr
        if self.pdr_obs_noise > 0:
            pdr = np.clip(pdr + self._rng.normal(0, self.pdr_obs_noise, len(pdr)), 0, 1)
        o = np.zeros(OBS_LEN, np.float32)
        mask = self.action_masks().reshape(MAX_UAVS, N_ACTIONS)
        N = len(self.users)
        for k in range(self.K):
            x, y, z = self.P[k]
            mine = self.serving == k
            roof = self.geometry.roof_height(x, y, self.safety["clearance_radius_m"])
            others = [math.dist(self.P[k], self.P[j]) for j in range(self.K) if j != k]
            exc = np.minimum(self.excess[k, mine], 60.0)
            o[k * PER_UAV:(k + 1) * PER_UAV] = [
                1.0,
                (x - rx[0]) / (rx[1] - rx[0]), (y - ry[0]) / (ry[1] - ry[0]), (z - zlo) / (zhi - zlo),
                np.clip((z - roof) / zhi, -1, 1),
                *(~mask[k, 1:]).astype(float),
                mine.mean(),
                pdr[mine].mean() if mine.any() else 0.0,
                pdr[mine].min() if mine.any() else 0.0,
                exc.mean() / 60.0 if mine.any() else 0.0,
                (self.excess[k, mine] > 10.0).mean() if mine.any() else 0.0,
                min(others) / 200.0 if others else 1.0,
            ]
        g = PER_UAV * MAX_UAVS
        m = self.metrics_from(pdr)
        o[g:g + 6] = [m["pdr"], m["coverage"], m["worst10_pdr"], N / MAX_USERS, self.K / MAX_UAVS,
                      self.t / self.max_steps]
        gx = np.clip(((self.users[:, 0] - rx[0]) / (rx[1] - rx[0]) * GRID).astype(int), 0, GRID - 1)
        gy = np.clip(((self.users[:, 1] - ry[0]) / (ry[1] - ry[0]) * GRID).astype(int), 0, GRID - 1)
        dens, poor = np.zeros((GRID, GRID)), np.zeros((GRID, GRID))
        np.add.at(dens, (gy, gx), 1.0 / N)
        np.add.at(poor, (gy, gx), (pdr < self.cov_thr) / N)
        o[g + 6:g + 6 + GRID * GRID] = dens.ravel()
        o[g + 6 + GRID * GRID:] = poor.ravel()
        return np.clip(o, -1.0, 2.0)

    def _info(self, moved_m, unsafe, cancelled, actions) -> dict:
        return {"window": self.window, "positions": self.P.tolist(), "actions": list(actions),
                "action_names": [ACTION_NAMES[a] for a in actions], "moved_m": moved_m,
                "unsafe_attempts": unsafe, "separation_cancels": cancelled, **self.metrics,
                "score": self.score}
