"""Stage-1 scan: find dense built-up training / validation windows in
city_damaged.xml, keeping urban_core strictly held out.

Builds a height map of the scene by straight-down ray casts on a regular grid,
then scores every 200 m candidate window (25 m stride) whose edge is at least
--gap-from-held-out metres from the urban_core window. urban_core itself is
never sampled; its density is only quoted from configs/eval_urban_core.json
for reference.

Per window it reports built fraction (cells with surface > 5 m, the same
definition as the urban_core note), roof heights, street (open-ground) area
available for users, and how many UAV grid positions satisfy the controller's
building-clearance rule at each altitude.

Windows are chosen greedily by built fraction, mutually non-overlapping.
The 2nd-densest becomes the validation window, the others training windows.

Usage:
  python ml/scan_training_areas.py --out ~/.cache/resqnet_3d/scan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sionna"))
sys.path.insert(0, str(ROOT / "ml"))

HELD_OUT = json.loads((ROOT / "configs" / "eval_urban_core.json").read_text())["scenarios"]["urban_core"]


def window_gap(c, size, region) -> float:
    """Edge-to-edge gap between a size x size window at c and a region (0 if overlapping)."""
    x0, x1 = c[0] - size / 2, c[0] + size / 2
    y0, y1 = c[1] - size / 2, c[1] + size / 2
    dx = max(region["x"][0] - x1, x0 - region["x"][1], 0.0)
    dy = max(region["y"][0] - y1, y0 - region["y"][1], 0.0)
    return float(np.hypot(dx, dy))


def disk_max(h: np.ndarray, radius_cells: int) -> np.ndarray:
    """Max of h over a disk of the given radius (in cells), edge-padded."""
    out = np.full_like(h, -np.inf)
    p = np.pad(h, radius_cells, mode="edge")
    n0, n1 = h.shape
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                out = np.maximum(out, p[radius_cells + dy: radius_cells + dy + n0,
                                        radius_cells + dx: radius_cells + dx + n1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=5.0, help="height-map cell (m)")
    ap.add_argument("--size", type=float, default=200.0, help="window side (m), = urban_core uav_region")
    ap.add_argument("--stride", type=float, default=25.0)
    ap.add_argument("--gap-from-held-out", type=float, default=100.0)
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--min-street-cells", type=int, default=150, help="5 m open-ground cells for users")
    ap.add_argument("--step-h", type=float, default=15.0)
    ap.add_argument("--altitudes", type=float, nargs="+", default=[15, 20, 30, 40, 50, 60])
    ap.add_argument("--clearance", type=float, default=5.0)
    ap.add_argument("--clearance-radius", type=float, default=10.0)
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "windows.json").exists():
        raise SystemExit(f"refusing to overwrite {out / 'windows.json'}")

    from sionna_feedback_engine import safe_load_scene_robust
    from obstruction import surface_height
    t0 = time.perf_counter()
    scene = safe_load_scene_robust("city_damaged.xml")
    bb = scene.mi_scene.bbox()
    xs = np.arange(np.floor(bb.min[0]), np.ceil(bb.max[0]) + 1e-6, a.res)
    ys = np.arange(np.floor(bb.min[1]), np.ceil(bb.max[1]) + 1e-6, a.res)
    gx, gy = np.meshgrid(xs, ys)                      # [ny, nx]
    H = surface_height(scene.mi_scene, gx.ravel(), gy.ravel()).reshape(gx.shape)
    t_hmap = time.perf_counter() - t0
    np.savez_compressed(out / "heightmap.npz", xs=xs, ys=ys, H=H)

    roof_r = disk_max(H, int(round(a.clearance_radius / a.res)))   # highest roof within radius
    street = H < 0.5
    # users need 3 m open ground around them: approximate with 1-cell erosion of street
    street_core = street & ~disk_max((~street).astype(float), 1).astype(bool)

    def idx(v, axis):
        return np.clip(np.round((v - axis[0]) / a.res).astype(int), 0, len(axis) - 1)

    half = a.size / 2
    cands = []
    for cx in np.arange(bb.min[0] + half, bb.max[0] - half + 1e-6, a.stride):
        for cy in np.arange(bb.min[1] + half, bb.max[1] - half + 1e-6, a.stride):
            gap = window_gap((cx, cy), a.size, HELD_OUT["uav_region"])
            if gap < a.gap_from_held_out:
                continue
            ix0, ix1 = idx(cx - half, xs), idx(cx + half, xs)
            iy0, iy1 = idx(cy - half, ys), idx(cy + half, ys)
            w = H[iy0:iy1 + 1, ix0:ix1 + 1]
            built = w > 5.0
            # user region = window shrunk by 10 m (as urban_core)
            u = street_core[idx(cy - half + 10, ys):idx(cy + half - 10, ys) + 1,
                            idx(cx - half + 10, xs):idx(cx + half - 10, xs) + 1]
            # UAV grid on step_h
            ux = np.arange(cx - half, cx + half + 1e-6, a.step_h)
            uy = np.arange(cy - half, cy + half + 1e-6, a.step_h)
            R = roof_r[np.ix_(idx(uy, ys), idx(ux, xs))]
            safe = {f"{z:g}": int((R + a.clearance <= z).sum()) for z in a.altitudes}
            cands.append({
                "centre": [float(cx), float(cy)],
                "gap_to_urban_core_m": round(gap, 1),
                "built_frac_gt5m": round(float(built.mean()), 4),
                "roof_mean_m": round(float(w[built].mean()), 2) if built.any() else 0.0,
                "roof_p90_m": round(float(np.percentile(w[built], 90)), 2) if built.any() else 0.0,
                "roof_max_m": round(float(w.max()), 2),
                "street_cells": int(u.sum()),
                "uav_grid_xy": int(R.size),
                "safe_uav_xy_per_altitude": safe,
            })

    ok = [c for c in cands if c["street_cells"] >= a.min_street_cells]
    ok.sort(key=lambda c: -c["built_frac_gt5m"])
    chosen = []
    for c in ok:
        if all(max(abs(c["centre"][0] - o["centre"][0]), abs(c["centre"][1] - o["centre"][1])) >= a.size
               for o in chosen):
            chosen.append(c)
        if len(chosen) == a.n_windows:
            break
    train_names = iter(f"train_{ch}" for ch in "abcdefgh")
    for i, c in enumerate(chosen):
        c["role"] = "validation" if i == 1 else "train"
        c["name"] = "val_a" if i == 1 else next(train_names)
        c["uav_region"] = {"x": [c["centre"][0] - half, c["centre"][0] + half],
                           "y": [c["centre"][1] - half, c["centre"][1] + half], "z": [15.0, 60.0]}
        c["user_region"] = {"x": [c["centre"][0] - half + 10, c["centre"][0] + half - 10],
                            "y": [c["centre"][1] - half + 10, c["centre"][1] + half - 10]}

    rec = {
        "scene": "city_damaged.xml",
        "scene_bbox": [list(map(float, bb.min)), list(map(float, bb.max))],
        "params": vars(a),
        "t_heightmap_s": round(t_hmap, 2),
        "heightmap_shape": list(H.shape),
        "scene_built_frac_gt5m": round(float((H > 5).mean()), 4),
        "held_out": {"name": "urban_core", "uav_region": HELD_OUT["uav_region"],
                     "reference_only": "72 % of 5 m cells > 5 m (quoted from configs/eval_urban_core.json; not re-sampled)"},
        "n_candidates": len(cands),
        "n_candidates_with_streets": len(ok),
        "built_frac_quantiles": {q: round(float(np.quantile([c["built_frac_gt5m"] for c in cands], q / 100)), 4)
                                 for q in (50, 90, 99)} if cands else {},
        "top10": ok[:10],
        "windows": chosen,
    }
    (out / "windows.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("scene_bbox", "t_heightmap_s", "heightmap_shape", "scene_built_frac_gt5m",
                                          "n_candidates", "n_candidates_with_streets", "built_frac_quantiles")}))
    for c in chosen:
        print(c["name"], c["role"], c["centre"], "gap", c["gap_to_urban_core_m"], "built", c["built_frac_gt5m"],
              "roof mean/p90/max", c["roof_mean_m"], c["roof_p90_m"], c["roof_max_m"],
              "street", c["street_cells"], "safe xy/alt", c["safe_uav_xy_per_altitude"], "of", c["uav_grid_xy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
