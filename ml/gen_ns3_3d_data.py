"""Stage 4: obstruction-aware ns-3 runs on the 3D training / validation windows
(never urban_core), for checking the PDR model and fitting / validating a
latency predictor.

Per window: `--layouts` random safe layouts (2-4 UAVs on the 3D lattice,
street-level users) and, for each, a one-move neighbour (one UAV moved by one
safe action, including up / down). Per layout: Sionna PathSolver at the ns-3
operating frequency -> obstruction matrix (obstruction.py) -> ns-3
(uav-demo.cc, matrix applied and verified) — the same path as
closed_loop_control.observe.

Outputs (resumable, never overwritten): <out>/<window>/<layout>_<base|nbr>.json

Usage: python ml/gen_ns3_3d_data.py --out ~/.cache/resqnet_3d/ns3_data --layouts 40
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "ml", ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import multiuav3d_env as E  # noqa: E402
import obstruction as ob  # noqa: E402
import multiuav_io as mio  # noqa: E402
from closed_loop_control import compose_rf  # noqa: E402


class _Zero:
    def path_gain(self, P, U):
        return np.zeros((len(P), len(U)))


class PathSolverRF:
    """Sionna PathSolver, same settings as sionna_feedback_engine (iso V
    antennas, max_depth 3, 1e6 samples, seed 42), scene loaded once."""

    def __init__(self, freq_hz: float):
        from sionna.rt import PathSolver, PlanarArray
        from sionna_feedback_engine import safe_load_scene_robust
        self.scene = safe_load_scene_robust("city_damaged.xml")
        self.scene.frequency = freq_hz
        for attr in ("tx_array", "rx_array"):
            setattr(self.scene, attr, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                                  horizontal_spacing=0.5, pattern="iso", polarization="V"))
        self.solver = PathSolver()

    def path_gain(self, P, U) -> np.ndarray:
        from sionna.rt import Transmitter, Receiver
        for name in list(self.scene.transmitters) + list(self.scene.receivers):
            self.scene.remove(name)
        for k, p in enumerate(P):
            self.scene.add(Transmitter(name=f"uav_{k+1}", position=[float(v) for v in p]))
        for i, u in enumerate(U):
            self.scene.add(Receiver(name=f"user_{i+1}", position=[float(v) for v in u]))
        paths = self.solver(self.scene, seed=42)
        ar, ai = paths.a
        amp = (np.asarray(ar.numpy()) ** 2 + np.asarray(ai.numpy()) ** 2).sum(axis=(1, 3, 4))   # [rx, tx]
        with np.errstate(divide="ignore"):
            pg = np.where(amp > 0, 10 * np.log10(np.maximum(amp, 1e-300)), -200.0)
        return pg.T                                                                         # [K, N]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/.cache/resqnet_3d/ns3_data")
    ap.add_argument("--windows", nargs="+", default=["train_a", "train_b", "train_c", "val_a"])
    ap.add_argument("--layouts", type=int, default=40)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--traffic-seconds", type=float, default=3.0)
    ap.add_argument("--rng-run", type=int, default=11)
    ap.add_argument("--heightmap", default="~/.cache/resqnet_3d/scan/heightmap.npz")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()

    import evaluate_obstruction as eo
    root = Path(os.environ.get("NS3_PATH") or Path.home() / "ns-allinone-3.47" / "ns-3.47")
    binary = eo.find_binary(root)                      # read-only check: no copy, no rebuild
    chan = eo.ns3_channel(binary)
    freq = chan["frequency_mhz"] * 1e6
    print("ns-3 binary", binary, "channel", chan, flush=True)

    cfg = json.loads((ROOT / "configs" / "training_areas_3d.json").read_text())["windows"]
    geom = E.HeightmapGeometry.load(a.heightmap)
    rf = PathSolverRF(freq)
    out = Path(os.path.expanduser(a.out))
    pool = cf.ThreadPoolExecutor(max_workers=a.workers)
    pending = []
    t0 = time.time()

    def finish(fut, path, rec):
        rec["ns3"] = fut.result()
        rec["finished_at"] = time.time()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(path)

    for w in a.windows:
        if w == "urban_core" or w not in cfg:
            raise SystemExit(f"window {w!r} not allowed / unknown")
        (out / w).mkdir(parents=True, exist_ok=True)
        env = E.MultiUAV3DEnv({w: cfg[w]}, geom, _Zero(), lambda P, U, pg: np.zeros(len(U)), seed=0)
        rng = np.random.default_rng([a.seed, a.windows.index(w)])
        for li in range(a.layouts):
            K = int(rng.choice([2, 3, 4]))
            N = int(rng.choice([20, 30]))
            env._rng = np.random.default_rng([a.seed, a.windows.index(w), li])
            env.reset(options={"window": w, "num_uavs": K, "num_users": N,
                               "user_seed": int(rng.integers(1 << 30))})
            P0 = env.P.copy()
            m = env.action_masks().reshape(E.MAX_UAVS, E.N_ACTIONS)
            opts = [(k, j) for k in range(env.K) for j in range(1, E.N_ACTIONS) if m[k, j]]
            k, j = opts[int(rng.integers(len(opts)))]
            P1 = P0.copy()
            P1[k] = env.target(k, j)
            for role, P in (("base", P0), ("nbr", P1)):
                path = out / w / f"{li:03d}_{role}.json"
                if path.exists():
                    continue
                users = env.users.tolist()
                pg = rf.path_gain(P, users)
                assoc = [int(s) for s in np.argmax(pg, axis=0)]
                M, mstats = ob.obstruction_matrix_db(compose_rf(P.tolist(), users, pg, assoc), P.tolist(), users, freq)
                handoff = ob.with_obstruction(mio.build_uav_positions(P.tolist(), users, associations=assoc), M, freq)
                rec = {"window": w, "layout": li, "role": role, "moved_uav": k, "move": E.ACTION_NAMES[j],
                       "positions": P.tolist(), "users": users, "path_gain_db": pg.tolist(), "assoc": assoc,
                       "obstruction_stats": mstats, "freq_hz": freq,
                       "ns3_args": [f"--RngRun={a.rng_run}", f"--trafficSeconds={a.traffic_seconds:g}"]}
                fut = pool.submit(eo.run_ns3_detailed, binary, handoff, tuple(rec["ns3_args"]))
                pending.append((fut, path, rec))
            while len(pending) > 2 * a.workers:
                fut, path, rec = pending.pop(0)
                finish(fut, path, rec)
            print(f"{w} layout {li + 1}/{a.layouts}  {time.time() - t0:.0f}s", flush=True)
    for fut, path, rec in pending:
        finish(fut, path, rec)
    pool.shutdown()
    print("done", time.time() - t0, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
