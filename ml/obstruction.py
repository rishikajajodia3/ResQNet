"""Building-obstruction loss from Sionna RT for the ns-3 handoff (opt-in).

ns3/uav-demo.cc models distance only (LogDistancePropagationLossModel). When
the handoff carries "obstruction_loss_db" (K x N, row = UAV k, column =
user i, handoff order) and "obstruction_freq_hz", ns-3 adds each entry to that
UAV-user link on top of LogDistance, symmetrically. Without these keys the
handoff - and the simulation - is unchanged.

Each entry is the RF model's loss IN EXCESS of free space at the same 3-D
distance and frequency, so ns-3's own distance term is not counted twice:

    L[k][i] = max(0, -path_gain_db[k,i] - FSPL(d_ki, f))

* a clear line-of-sight link gives ~0 dB, i.e. exactly the distance-only link;
* negative excess (constructive multipath) is clamped to 0: obstruction can
  only add loss, as in network_model.network_rss_dbm;
* a link for which Sionna finds no path at all gets NO_PATH_LOSS_DB.

The frequency must be the ns-3 Wi-Fi operating channel's (ns-3 refuses a
mismatch of more than 1 MHz), and the RF result must be computed at that
frequency.

Also provides building-aware geometry for dense scenes, using ray casts on the
Sionna scene that is ray traced: street-level users and UAV positions with roof
clearance.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np

import network_model as nm

NO_PATH_LOSS_DB = 200.0          # Sionna found no path: link unusable in ns-3
MAX_LOSS_DB = 300.0              # upper bound accepted by uav-demo.cc
POSITION_TOL_M = 0.01


def obstruction_matrix_db(
    rf_result: dict,
    uav_positions: Sequence[Sequence[float]],
    user_positions: Sequence[Sequence[float]],
    freq_hz: float,
) -> tuple[list[list[float]], dict]:
    """K x N obstruction loss from one RF result (Sionna or surrogate dict).

    The RF result must describe exactly these positions (checked), so the
    matrix and the ns-3 handoff use the same geometry.
    Returns (matrix, stats).
    """
    uavs = np.asarray(uav_positions, dtype=float)
    users = np.asarray(user_positions, dtype=float)
    K, N = len(uavs), len(users)
    if rf_result.get("num_uavs") != K or rf_result.get("num_users") != N:
        raise ValueError(f"RF result is {rf_result.get('num_uavs')}x{rf_result.get('num_users')}, "
                         f"positions are {K}x{N}")
    ur = rf_result["user_results"]
    M = [[0.0] * N for _ in range(K)]
    excess_raw, clamped, no_path = [], 0, 0
    for i, u in enumerate(ur):
        if np.max(np.abs(np.asarray(u["position"], float) - users[i])) > POSITION_TOL_M:
            raise ValueError(f"user {i}: RF position {u['position']} != handoff {users[i].tolist()}")
        if len(u["per_uav"]) != K:
            raise ValueError(f"user {i}: RF result has {len(u['per_uav'])} UAV links, expected {K}")
        for k, link in enumerate(u["per_uav"]):
            if np.max(np.abs(np.asarray(link["uav_position"], float) - uavs[k])) > POSITION_TOL_M:
                raise ValueError(f"UAV {k}: RF position {link['uav_position']} != handoff {uavs[k].tolist()}")
            pg = float(link["path_gain_db"])
            if not math.isfinite(pg) or pg <= -199.0:
                M[k][i] = NO_PATH_LOSS_DB
                no_path += 1
                continue
            d = float(np.linalg.norm(users[i] - uavs[k]))
            ex = -pg - nm.free_space_loss_db(d, freq_hz)
            excess_raw.append(ex)
            if ex < 0.0:
                clamped += 1
            M[k][i] = round(min(max(ex, 0.0), MAX_LOSS_DB), 3)
    flat = np.array([v for row in M for v in row if v < NO_PATH_LOSS_DB])
    stats = {
        "freq_hz": float(freq_hz),
        "links": K * N,
        "no_path_links": no_path,
        "clamped_negative_links": clamped,
        "mean_loss_db": float(flat.mean()) if len(flat) else None,
        "median_loss_db": float(np.median(flat)) if len(flat) else None,
        "max_loss_db": float(flat.max()) if len(flat) else None,
        "links_over_10db": int((flat > 10).sum()),
        "mean_raw_excess_db": float(np.mean(excess_raw)) if excess_raw else None,
    }
    return M, stats


def with_obstruction(handoff: dict, matrix: Sequence[Sequence[float]], freq_hz: float) -> dict:
    """Copy of a handoff dict (multiuav_io.build_uav_positions) with the
    obstruction keys added. The input dict is not modified."""
    K, N = handoff["num_uavs"], handoff["num_users"]
    if len(matrix) != K or any(len(r) != N for r in matrix):
        raise ValueError(f"matrix must be {K}x{N}")
    for r in matrix:
        for v in r:
            if not (math.isfinite(v) and 0.0 <= v <= MAX_LOSS_DB):
                raise ValueError(f"matrix values must be finite and in [0, {MAX_LOSS_DB}] dB, got {v}")
    out = copy.deepcopy(handoff)
    out["obstruction_loss_db"] = [[float(v) for v in r] for r in matrix]
    out["obstruction_freq_hz"] = float(freq_hz)
    return out


# ----------------------------------------------------------------------
# building-aware geometry (ray casts on the loaded Sionna / Mitsuba scene)
# ----------------------------------------------------------------------
RAY_TOP_M = 1000.0


def surface_height(mi_scene, xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
    """Height of the top surface under each (x, y): a ray cast straight down
    from RAY_TOP_M. 0 where nothing is hit (open ground)."""
    import mitsuba as mi
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    ray = mi.Ray3f(o=mi.Point3f(mi.Float(xs), mi.Float(ys), mi.Float(np.full(n, RAY_TOP_M))),
                   d=mi.Vector3f(mi.Float(np.zeros(n)), mi.Float(np.zeros(n)), mi.Float(-np.ones(n))))
    si = mi_scene.ray_intersect(ray)
    t = np.array(si.t, dtype=float)
    valid = np.array(si.is_valid(), dtype=bool)
    return np.where(valid, RAY_TOP_M - t, 0.0)


def max_height_within(mi_scene, x: float, y: float, radius: float, step: float = 2.5) -> float:
    """Highest surface within `radius` of (x, y), sampled on a `step` grid."""
    g = np.arange(-radius, radius + 1e-9, step)
    gx, gy = np.meshgrid(g, g)
    keep = gx ** 2 + gy ** 2 <= radius ** 2
    return float(surface_height(mi_scene, x + gx[keep], y + gy[keep]).max())


def street_level_users(mi_scene, region: dict, n: int, seed: int, user_z: float = 1.5,
                       clearance_m: float = 3.0, max_ground_m: float = 0.5,
                       min_spacing_m: float = 4.0, max_tries: int = 50000) -> list[list[float]]:
    """n reproducible users on open ground: the surface at the user and on a
    circle of radius clearance_m around it is below max_ground_m."""
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = region["x"], region["y"]
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    users: list[list[float]] = []
    tries = 0
    while len(users) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"placed only {len(users)}/{n} street-level users in {max_tries} tries")
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if any(math.hypot(x - u[0], y - u[1]) < min_spacing_m for u in users):
            continue
        hs = surface_height(mi_scene, np.r_[x, x + clearance_m * np.cos(ang)],
                            np.r_[y, y + clearance_m * np.sin(ang)])
        if hs.max() < max_ground_m:
            users.append([round(float(x), 2), round(float(y), 2), float(user_z)])
    return users


def roof_clear_uav_xy(mi_scene, centre: Sequence[float], offsets: Sequence[Sequence[float]],
                      search_m: float = 25.0, step_m: float = 5.0,
                      radius_m: float = 10.0) -> list[dict]:
    """For each nominal position centre + offset, the nearby spot (within
    search_m, on a step_m grid) whose surroundings (radius_m) are lowest;
    ties go to the spot closest to the nominal one."""
    out = []
    g = np.arange(-search_m, search_m + 1e-9, step_m)
    for ox, oy in offsets:
        bx, by = centre[0] + ox, centre[1] + oy
        best = None
        for dx in g:
            for dy in g:
                h = max_height_within(mi_scene, bx + dx, by + dy, radius_m)
                key = (round(h, 3), math.hypot(dx, dy))
                if best is None or key < best[0]:
                    best = (key, bx + dx, by + dy)
        out.append({"x": round(float(best[1]), 2), "y": round(float(best[2]), 2),
                    "local_roof_m": best[0][0], "nominal": [bx, by]})
    return out


def line_of_sight(mi_scene, tx: Sequence[Sequence[float]], rx: Sequence[Sequence[float]]) -> np.ndarray:
    """Geometric line of sight (unblocked straight segment) for each tx/rx pair."""
    import mitsuba as mi
    tx = np.asarray(tx, dtype=float)
    rx = np.asarray(rx, dtype=float)
    d = rx - tx
    L = np.linalg.norm(d, axis=1)
    d = d / L[:, None]
    o = tx + d * 0.01
    ray = mi.Ray3f(o=mi.Point3f(*[mi.Float(o[:, j]) for j in range(3)]),
                   d=mi.Vector3f(*[mi.Float(d[:, j]) for j in range(3)]))
    ray.maxt = mi.Float(L - 0.02)
    return ~np.array(mi_scene.ray_test(ray), dtype=bool)
