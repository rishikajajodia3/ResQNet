"""Precomputed Sionna RT radio maps for 3D PPO training (stage 2) and their
lookup (RadioMapRF), plus the accuracy check against PathSolver (stage 3).

One radio map per UAV lattice position of a window: a transmitter at the UAV
position, a horizontal measurement plane at user height (1.5 m) covering the
window, cell size `cell` m, RadioMapSolver at the ns-3 operating frequency.
Path gain is reciprocal, so map[cell] is the UAV<->user path gain for a user in
that cell. Maps are stored in dB (-200 where Sionna found no path, the value
the rest of the code uses for "no path").

Cache layout (never overwritten; resumable):
  <root>/<window>/<tag>/meta.json            scene / frequency / grid / solver settings
  <root>/<window>/<tag>/shard_XXXX.npz       positions [B,3], pg_db [B,ny,nx] float32
A cache is reused only if meta.json matches the requested settings exactly.

Usage:
  python ml/radiomap_cache.py precompute --window train_a --samples 10000000
  python ml/radiomap_cache.py check --window train_a --links 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "ml", ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CACHE_VERSION = 1
DEFAULT_ROOT = "~/.cache/resqnet_3d/radiomaps"
NS3_FREQ_HZ = 2.412e9
USER_Z = 1.5
NO_PATH_DB = -200.0


def scene_fingerprint(scene_name: str = "city_damaged.xml") -> dict:
    from sionna_feedback_engine import resolve_scene_path
    xml = Path(resolve_scene_path(scene_name))
    h = hashlib.sha256()
    for f in sorted(xml.parent.glob(xml.stem + ".*")):
        if f.suffix in (".xml", ".obj", ".mtl"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return {"scene": scene_name, "scene_sha256": h.hexdigest()}


def windows_cfg() -> dict:
    return json.loads((ROOT / "configs" / "training_areas_3d.json").read_text())


def lattice(region: dict, step_h: float, z_levels) -> np.ndarray:
    xs = np.arange(region["x"][0], region["x"][1] + 1e-6, step_h)
    ys = np.arange(region["y"][0], region["y"][1] + 1e-6, step_h)
    return np.array([[x, y, z] for z in z_levels for y in ys for x in xs], float)


def make_meta(window: str, w: dict, a) -> dict:
    return {"cache_version": CACHE_VERSION, **scene_fingerprint(), "window": window,
            "uav_region": w["uav_region"], "freq_hz": NS3_FREQ_HZ, "user_z_m": USER_Z,
            "cell_m": a.cell, "map_size_m": [w["uav_region"]["x"][1] - w["uav_region"]["x"][0],
                                             w["uav_region"]["y"][1] - w["uav_region"]["y"][0]],
            "samples_per_tx": a.samples, "max_depth": a.max_depth, "solver": "RadioMapSolver",
            "los": True, "specular_reflection": True, "refraction": True, "diffraction": False,
            "antenna": "iso, V, 1x1", "step_h_m": a.step_h, "z_levels": list(map(float, a.z_levels)),
            "clearance_filter": {"heightmap": a.heightmap, "building_clearance_m": 5.0, "radius_m": 10.0}}


def cache_dir(root, window, a) -> Path:
    return Path(os.path.expanduser(root)) / window / f"v{CACHE_VERSION}_s{a.samples}_d{a.max_depth}_c{a.cell:g}"


# ----------------------------------------------------------------------
def precompute(a) -> int:
    import multiuav3d_env as E
    cfg = windows_cfg()
    w = cfg["windows"][a.window]
    if a.window == "urban_core" or w.get("role") == "held_out":
        raise SystemExit("urban_core is held out")
    d = cache_dir(a.root, a.window, a)
    d.mkdir(parents=True, exist_ok=True)
    meta = make_meta(a.window, w, a)
    mp = d / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        old.pop("cell_centres", None)
        if old != meta:
            raise SystemExit(f"existing cache {d} has different settings; refusing to mix")
    geom = E.HeightmapGeometry.load(a.heightmap)
    P = lattice(w["uav_region"], a.step_h, a.z_levels)
    safe = np.array([geom.roof_height(p[0], p[1], 10.0) + 5.0 <= p[2] for p in P])
    P = P[safe]
    shards = [P[i:i + a.batch] for i in range(0, len(P), a.batch)]
    print(f"{a.window}: {len(P)} safe lattice positions ({(~safe).sum()} unsafe skipped), {len(shards)} shards -> {d}",
          flush=True)

    from sionna.rt import RadioMapSolver, Transmitter, PlanarArray
    from sionna_feedback_engine import safe_load_scene_robust
    scene = safe_load_scene_robust("city_damaged.xml")
    scene.frequency = NS3_FREQ_HZ
    for attr in ("tx_array", "rx_array"):
        setattr(scene, attr, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                         pattern="iso", polarization="V"))
    r = w["uav_region"]
    centre = [(r["x"][0] + r["x"][1]) / 2, (r["y"][0] + r["y"][1]) / 2, USER_Z]
    solver = RadioMapSolver()
    t0, done = time.time(), 0
    for i, S in enumerate(shards):
        f = d / f"shard_{i:04d}.npz"
        if f.exists():
            continue
        for name in list(scene.transmitters):
            scene.remove(name)
        for k, p in enumerate(S):
            scene.add(Transmitter(name=f"tx{k}", position=[float(v) for v in p]))
        rm = solver(scene, center=centre, orientation=[0.0, 0.0, 0.0], size=meta["map_size_m"],
                    cell_size=[a.cell, a.cell], samples_per_tx=a.samples, max_depth=a.max_depth, seed=1 + i)
        pg = np.array(rm.path_gain, dtype=np.float64)
        with np.errstate(divide="ignore"):
            pg_db = np.where(pg > 0, 10 * np.log10(np.maximum(pg, 1e-300)), NO_PATH_DB).astype(np.float32)
        if not mp.exists():
            cc = np.array(rm.cell_centers, dtype=np.float64)
            mp.write_text(json.dumps({**meta, "cell_centres": {"x": cc[0, :, 0].tolist(), "y": cc[:, 0, 1].tolist()}}))
        tmp = d / f".shard_{i:04d}.tmp.npz"
        np.savez_compressed(tmp, positions=S, pg_db=pg_db)
        tmp.rename(f)
        done += 1
        el = time.time() - t0
        print(f"shard {i + 1}/{len(shards)}  {el:.0f}s  eta {el / done * (len(shards) - i - 1):.0f}s  "
              f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB", flush=True)
    print("done", d, flush=True)
    return 0


# ----------------------------------------------------------------------
class RadioMapRF:
    """path_gain(P [K,3], U [N,3]) -> [K,N] dB from the precomputed maps.
    UAV positions must be lattice points of the cache (exact match within
    1 cm); users are looked up in the cell that contains them."""

    label = "Sionna RT radio maps (precomputed, simulated)"
    has_buildings = True

    def __init__(self, directory: str | Path, expect: dict | None = None):
        d = Path(os.path.expanduser(directory))
        self.meta = json.loads((d / "meta.json").read_text())
        if self.meta.get("cache_version") != CACHE_VERSION:
            raise ValueError("radio-map cache version mismatch")
        for k, v in (expect or {}).items():
            if self.meta.get(k) != v:
                raise ValueError(f"radio-map cache {k}={self.meta.get(k)!r}, expected {v!r}")
        pos, maps = [], []
        for f in sorted(d.glob("shard_*.npz")):
            z = np.load(f)
            pos.append(z["positions"]); maps.append(z["pg_db"])
        if not pos:
            raise ValueError(f"no shards in {d}")
        self.positions = np.concatenate(pos)
        self.maps = np.concatenate(maps)
        self.index = {self._key(p): i for i, p in enumerate(self.positions)}
        cx, cy = np.array(self.meta["cell_centres"]["x"]), np.array(self.meta["cell_centres"]["y"])
        self.cell = float(self.meta["cell_m"])
        self.x0, self.y0 = cx[0] - self.cell / 2, cy[0] - self.cell / 2
        self.nx, self.ny = len(cx), len(cy)
        self.freq_hz = float(self.meta["freq_hz"])

    @staticmethod
    def _key(p):
        return (round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2))

    def map_index(self, p) -> int:
        try:
            return self.index[self._key(p)]
        except KeyError:
            raise KeyError(f"UAV position {tuple(p)} is not in the radio-map cache") from None

    def user_cells(self, U) -> tuple[np.ndarray, np.ndarray]:
        U = np.asarray(U, float)
        ix = np.floor((U[:, 0] - self.x0) / self.cell).astype(int)
        iy = np.floor((U[:, 1] - self.y0) / self.cell).astype(int)
        if (ix < 0).any() or (ix >= self.nx).any() or (iy < 0).any() or (iy >= self.ny).any():
            raise ValueError("user outside the radio-map area")
        return iy, ix

    def path_gain(self, P, U) -> np.ndarray:
        iy, ix = self.user_cells(U)
        idx = [self.map_index(p) for p in np.asarray(P, float)]
        return self.maps[idx][:, iy, ix].astype(float)


# ----------------------------------------------------------------------
def check(a) -> int:
    """Stage 3: radio-map lookup vs PathSolver at the exact user points."""
    import multiuav3d_env as E
    import network_model as nm
    from sionna.rt import PathSolver, Transmitter, Receiver, PlanarArray
    from sionna_feedback_engine import safe_load_scene_robust
    d = cache_dir(a.root, a.window, a)
    rf = RadioMapRF(d)
    w = windows_cfg()["windows"][a.window]
    geom = E.HeightmapGeometry.load(a.heightmap)
    rng = np.random.default_rng(a.seed)
    scene = safe_load_scene_robust("city_damaged.xml")
    scene.frequency = rf.freq_hz
    for attr in ("tx_array", "rx_array"):
        setattr(scene, attr, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                         pattern="iso", polarization="V"))
    solver = PathSolver()
    rows = []
    per_tx = 10
    n_tx = int(math.ceil(a.links / per_tx))
    tx_idx = rng.choice(len(rf.positions), size=n_tx, replace=False)
    t0 = time.time()
    for j, ti in enumerate(tx_idx):
        p = rf.positions[ti]
        users = geom.users(w["user_region"], per_tx, int(rng.integers(1 << 30)))
        for name in list(scene.transmitters) + list(scene.receivers):
            scene.remove(name)
        scene.add(Transmitter(name="tx", position=[float(v) for v in p]))
        for i, u in enumerate(users):
            scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in u]))
        paths = solver(scene, seed=42, max_depth=int(rf.meta["max_depth"]))
        ar, ai = paths.a
        amp = (np.asarray(ar.numpy()) ** 2 + np.asarray(ai.numpy()) ** 2).sum(axis=(1, 3, 4))[:, 0]
        with np.errstate(divide="ignore"):
            ps = np.where(amp > 0, 10 * np.log10(np.maximum(amp, 1e-300)), NO_PATH_DB)
        lk = rf.path_gain([p], users)[0]
        for i, u in enumerate(users):
            dist = math.dist(p, u)
            use_ps = nm.network_rss_dbm(ps[i], dist, rf.freq_hz)
            use_lk = nm.network_rss_dbm(lk[i], dist, rf.freq_hz)
            ok = lambda rss: rss >= nm.MIN_RSS_DBM and rss - nm.NS3_NOISE_DBM >= nm.MIN_SNR_DB  # noqa: E731
            rows.append({"uav": p.tolist(), "user": u, "dist_m": dist, "pathsolver_db": float(ps[i]),
                         "radiomap_db": float(lk[i]), "usable_pathsolver": bool(ok(use_ps)),
                         "usable_radiomap": bool(ok(use_lk))})
    both = [r for r in rows if r["pathsolver_db"] > NO_PATH_DB and r["radiomap_db"] > NO_PATH_DB]
    err = np.array([r["radiomap_db"] - r["pathsolver_db"] for r in both])
    flips = sum(r["usable_pathsolver"] != r["usable_radiomap"] for r in rows)
    res = {"window": a.window, "cache": str(d), "links": len(rows), "t_s": round(time.time() - t0, 1),
           "both_have_path": len(both),
           "pathsolver_only_path": sum(r["pathsolver_db"] > NO_PATH_DB >= r["radiomap_db"] for r in rows),
           "radiomap_only_path": sum(r["radiomap_db"] > NO_PATH_DB >= r["pathsolver_db"] for r in rows),
           "median_abs_err_db": float(np.median(np.abs(err))) if len(err) else None,
           "p90_abs_err_db": float(np.percentile(np.abs(err), 90)) if len(err) else None,
           "median_bias_db": float(np.median(err)) if len(err) else None,
           "usable_flips": int(flips), "usable_flip_frac": flips / len(rows),
           "usable_frac_pathsolver": float(np.mean([r["usable_pathsolver"] for r in rows])),
           "thresholds": {"median_abs_err_db": 3.0, "usable_flip_frac": 0.05}}
    res["pass"] = bool(res["median_abs_err_db"] is not None and res["median_abs_err_db"] <= 3.0
                       and res["usable_flip_frac"] <= 0.05)
    out = d.parent / f"accuracy_{d.name}_seed{a.seed}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    print(json.dumps(res))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["precompute", "check"])
    ap.add_argument("--window", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--samples", type=int, default=10_000_000)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--step-h", type=float, default=15.0)
    ap.add_argument("--z-levels", type=float, nargs="+", default=[20, 30, 40, 50, 60])
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--heightmap", default="~/.cache/resqnet_3d/scan/heightmap.npz")
    ap.add_argument("--links", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    return precompute(a) if a.cmd == "precompute" else check(a)


if __name__ == "__main__":
    sys.exit(main())
