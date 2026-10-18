"""Stage-0 benchmark: cost of one Sionna RT radio map on city_damaged.xml.

Measures wall time and peak RSS of RadioMapSolver for a ground-level
(1.5 m) measurement plane under UAV transmitters, at the ns-3 operating
frequency, for one configuration per process (so peak memory is per config).
Also reports run-to-run noise (two seeds) and agreement with PathSolver
(the solver used by the closed loop) at outdoor street-level cells.

The window is chosen >= --min-dist-from-held-out metres from urban_core so the
held-out evaluation area is never touched.

Usage:
  python ml/radiomap_bench.py --samples 1000000 --num-tx 1 --out ~/.cache/resqnet_3d/bench
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sionna"))
sys.path.insert(0, str(ROOT / "ml"))

HELD_OUT_CENTRE = (500.0, -50.0)   # urban_core (configs/eval_urban_core.json)
NS3_FREQ_HZ = 2.412e9              # ns-3 Wi-Fi channel 1


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pick_window(mi_scene, size: float, min_dist: float) -> tuple[float, float]:
    """Densest-built window centre on a coarse grid, far from urban_core."""
    from obstruction import surface_height
    bb = mi_scene.bbox()
    xs = np.arange(bb.min[0] + size / 2, bb.max[0] - size / 2 + 1e-6, 50.0)
    ys = np.arange(bb.min[1] + size / 2, bb.max[1] - size / 2 + 1e-6, 50.0)
    best, best_c = -1.0, None
    g = np.arange(-size / 2, size / 2 + 1e-6, 10.0)
    gx, gy = np.meshgrid(g, g)
    for cx in xs:
        for cy in ys:
            if np.hypot(cx - HELD_OUT_CENTRE[0], cy - HELD_OUT_CENTRE[1]) < min_dist:
                continue
            h = surface_height(mi_scene, (cx + gx).ravel(), (cy + gy).ravel())
            built = float(np.mean(h > 3.0))
            if built > best:
                best, best_c = built, (float(cx), float(cy))
    if best_c is None:
        raise SystemExit("no window far enough from urban_core")
    return best_c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1_000_000, help="samples_per_tx")
    ap.add_argument("--num-tx", type=int, default=1, help="transmitters in one solver call")
    ap.add_argument("--size", type=float, default=200.0)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--uav-z", type=float, default=40.0)
    ap.add_argument("--min-dist-from-held-out", type=float, default=300.0)
    ap.add_argument("--n-check", type=int, default=30, help="street cells checked against PathSolver")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    tag = f"s{a.samples}_tx{a.num_tx}_c{a.cell:g}"
    dest = out / f"bench_{tag}.json"
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")

    rec: dict = {"config": vars(a), "freq_hz": NS3_FREQ_HZ}
    rec["rss_mb_start"] = peak_rss_mb()

    t0 = time.perf_counter()
    import mitsuba as mi  # noqa: F401
    from sionna.rt import RadioMapSolver, PathSolver, Transmitter, Receiver, PlanarArray
    from sionna_feedback_engine import safe_load_scene_robust
    from obstruction import surface_height
    scene = safe_load_scene_robust("city_damaged.xml")
    scene.frequency = NS3_FREQ_HZ
    for attr in ("tx_array", "rx_array"):
        setattr(scene, attr, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                         horizontal_spacing=0.5, pattern="iso", polarization="V"))
    rec["t_scene_load_s"] = time.perf_counter() - t0
    rec["rss_mb_after_scene"] = peak_rss_mb()
    bb = scene.mi_scene.bbox()
    rec["scene_bbox"] = [list(map(float, bb.min)), list(map(float, bb.max))]

    cx, cy = pick_window(scene.mi_scene, a.size, a.min_dist_from_held_out)
    rec["window_centre"] = [cx, cy]
    rec["window_dist_to_urban_core_m"] = float(np.hypot(cx - HELD_OUT_CENTRE[0], cy - HELD_OUT_CENTRE[1]))

    # transmitters on a small grid inside the window, all above local roofs
    rng = np.random.default_rng(0)
    txs = []
    while len(txs) < a.num_tx:
        x, y = cx + rng.uniform(-60, 60), cy + rng.uniform(-60, 60)
        if surface_height(scene.mi_scene, [x], [y])[0] + 5.0 < a.uav_z:
            txs.append([float(x), float(y), a.uav_z])
    for k, p in enumerate(txs):
        scene.add(Transmitter(name=f"tx{k}", position=p))
    rec["tx_positions"] = txs

    solver = RadioMapSolver()
    kw = dict(center=[cx, cy, 1.5], orientation=[0.0, 0.0, 0.0], size=[a.size, a.size],
              cell_size=[a.cell, a.cell], samples_per_tx=a.samples, max_depth=a.max_depth)
    runs, maps = [], []
    for seed in (1, 2, 3):   # first call includes JIT compilation
        t = time.perf_counter()
        rm = solver(scene, seed=seed, **kw)
        pg = np.array(rm.path_gain, dtype=np.float64)   # [num_tx, cells_y, cells_x], linear
        runs.append(time.perf_counter() - t)
        maps.append(pg)
    rec["t_radiomap_s"] = runs
    rec["t_radiomap_steady_s"] = float(np.mean(runs[1:]))
    rec["t_per_tx_steady_s"] = rec["t_radiomap_steady_s"] / a.num_tx
    rec["rss_mb_after_radiomap"] = peak_rss_mb()
    rec["map_shape"] = list(maps[0].shape)
    rec["map_bytes_float32"] = int(np.prod(maps[0].shape) * 4)

    centres = np.array(rm.cell_centers, dtype=np.float64)           # [cells_y, cells_x, 3]
    ground = surface_height(scene.mi_scene, centres[..., 0].ravel(), centres[..., 1].ravel())
    street = (ground.reshape(centres.shape[:2]) < 0.5)
    rec["street_cell_fraction"] = float(street.mean())

    # run-to-run noise (seed 2 vs 3) on street cells under tx0, where both have a path
    def db(x):
        with np.errstate(divide="ignore"):
            return 10 * np.log10(x)
    m2, m3 = maps[1][0], maps[2][0]
    both = street & (m2 > 0) & (m3 > 0)
    d = np.abs(db(m2[both]) - db(m3[both]))
    rec["seed_noise_db"] = {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                            "cells": int(both.sum())}
    rec["street_cells_no_path_frac"] = float(((m2 == 0) & street).sum() / max(1, street.sum()))

    # agreement with PathSolver at street cells under tx0
    iy, ix = np.nonzero(street)
    pick = rng.choice(len(iy), size=min(a.n_check, len(iy)), replace=False)
    for name in list(scene.transmitters.keys())[1:]:
        scene.remove(name)
    pts = []
    for j, i in enumerate(pick):
        p = centres[iy[i], ix[i]]
        scene.add(Receiver(name=f"rx{j}", position=[float(p[0]), float(p[1]), 1.5]))
        pts.append((iy[i], ix[i]))
    t = time.perf_counter()
    paths = PathSolver()(scene, seed=42, max_depth=a.max_depth)
    rec["t_pathsolver_s"] = time.perf_counter() - t
    ar, ai = paths.a
    amp = np.asarray(ar.numpy()) ** 2 + np.asarray(ai.numpy()) ** 2   # [rx, rx_ant, tx, tx_ant, path]
    ps = amp.sum(axis=(1, 3, 4))[:, 0]
    rm_vals = np.array([maps[1][0][y, x] for (y, x) in pts])
    ok = (ps > 0) & (rm_vals > 0)
    err = db(rm_vals[ok]) - db(ps[ok])
    rec["vs_pathsolver"] = {
        "n": len(pts), "both_have_path": int(ok.sum()),
        "pathsolver_only": int(((ps > 0) & (rm_vals == 0)).sum()),
        "radiomap_only": int(((ps == 0) & (rm_vals > 0)).sum()),
        "median_abs_err_db": float(np.median(np.abs(err))) if ok.any() else None,
        "p90_abs_err_db": float(np.percentile(np.abs(err), 90)) if ok.any() else None,
        "median_bias_db": float(np.median(err)) if ok.any() else None,
    }
    rec["rss_mb_peak"] = peak_rss_mb()
    dest.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("config", "t_scene_load_s", "t_radiomap_s", "t_per_tx_steady_s",
                                          "rss_mb_after_scene", "rss_mb_peak", "map_shape", "seed_noise_db",
                                          "street_cell_fraction", "street_cells_no_path_frac", "vs_pathsolver",
                                          "window_centre", "window_dist_to_urban_core_m")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
