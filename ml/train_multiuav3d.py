"""Train MaskablePPO on MultiUAV3DEnv (7 actions per UAV: stay, -X, +X, -Y, +Y, down, up).

Training windows come from configs/training_areas_3d.json (role "train");
model selection uses the "validation" window only. urban_core is refused by the
environment. Each run writes to a NEW directory and never overwrites:
  models/ppo_multiuav_3d/<run_id>/   (or --out-root, e.g. a cache dir for smoke runs)
    model.zip, train_config.json, train_log.csv, val_log.csv, summary.json

RF backends: "free_space" (smoke / tests; NOT Sionna) or "radiomap" (precomputed
Sionna radio maps, ml/radiomap_cache.py; stage 2).

Smoke example (~2k steps, free-space RF, real building height map for safety):
  python ml/train_multiuav3d.py --smoke --out-root ~/.cache/resqnet_3d/smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

import multiuav3d_env as E  # noqa: E402

DEFAULT_HEIGHTMAP = "~/.cache/resqnet_3d/scan/heightmap.npz"


def load_windows(role: str, only=None) -> dict:
    cfg = json.loads((ROOT / "configs" / "training_areas_3d.json").read_text())
    return {n: w for n, w in cfg["windows"].items() if w["role"] == role and (not only or n in only)}


class MultiWindowRF:
    """Dispatch to the radio-map cache of the window the positions are in."""
    label = "Sionna RT radio maps (precomputed, simulated)"

    def __init__(self, windows: dict, samples: int):
        import radiomap_cache as rc
        self.rfs = {}
        for n, w in windows.items():
            d = Path(os.path.expanduser(rc.DEFAULT_ROOT)) / n / f"v{rc.CACHE_VERSION}_s{samples}_d3_c2"
            self.rfs[n] = (w["uav_region"], rc.RadioMapRF(d, expect={"freq_hz": E.TRAINING_FREQ_HZ,
                                                                      "window": n}))
            for arr in (self.rfs[n][1].maps, self.rfs[n][1].positions):   # shared between envs: read-only
                arr.flags.writeable = False

    def path_gain(self, P, U):
        x, y = float(P[0][0]), float(P[0][1])
        for reg, rf in self.rfs.values():
            if reg["x"][0] <= x <= reg["x"][1] and reg["y"][0] <= y <= reg["y"][1]:
                return rf.path_gain(P, U)
        raise KeyError("positions outside every cached window")


_RF_SHARED: dict = {}   # one MultiWindowRF per (window set, samples); pure lookup, so envs can share it


def make_env(args, role: str, seed: int):
    from closed_loop_control import SurrogateRF
    geom = E.HeightmapGeometry.load(args.heightmap)
    windows = load_windows(role, args.train_windows if role == "train" else args.val_windows)
    if args.rf == "free_space":
        rf = SurrogateRF(E.TRAINING_FREQ_HZ)
    else:
        key = (tuple(sorted(windows)), args.radiomap_samples)
        if key not in _RF_SHARED:
            _RF_SHARED[key] = MultiWindowRF(windows, args.radiomap_samples)
        rf = _RF_SHARED[key]
    if args.pdr_model == "calibrated":
        pdr = E.CalibratedPDR(path=args.pdr_model_path)
    else:
        pdr = E.NetworkPDR()
    return E.MultiUAV3DEnv(windows, geom, rf, pdr, max_steps=args.max_steps,
                           pdr_obs_noise=args.pdr_obs_noise, seed=seed,
                           reward_weights=json.loads(args.reward_weights) if args.reward_weights else None)


def provenance(args) -> dict:
    """Code revision and input hashes, so a run can be reproduced (the 3D files are untracked, so HEAD alone is
    not enough)."""
    import hashlib
    import subprocess

    def sha(p):
        p = Path(p).expanduser()
        p = p if p.is_absolute() else ROOT / p
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    import calibrated_model as cm
    import radiomap_cache as rc
    files = ["ml/multiuav3d_env.py", "ml/train_multiuav3d.py", "ml/radiomap_cache.py", "ml/closed_loop_control.py",
             "configs/training_areas_3d.json"]
    caches = {n: str(Path(os.path.expanduser(rc.DEFAULT_ROOT)) / n / f"v{rc.CACHE_VERSION}_s{args.radiomap_samples}_d3_c2")
              for n in {**load_windows("train", args.train_windows), **load_windows("validation", args.val_windows)}}
    return {"git_head": head, "source_sha256": {f: sha(ROOT / f) for f in files},
            "pdr_model_sha256": sha(args.pdr_model_path or cm.DEFAULT_MODEL), "heightmap_sha256": sha(args.heightmap),
            "radiomap_caches": caches,
            "radiomap_meta_sha256": {n: sha(Path(d) / "meta.json") for n, d in caches.items()}}


def run_policy(model, env, episodes: int, seed: int) -> dict:
    """Deterministic rollouts; returns mean final metrics and Z-action stats."""
    finals, z_moves, moves = [], 0, 0
    alts = []
    for e in range(episodes):
        obs, info = env.reset(seed=seed + e)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            acts = info["actions"]
            z_moves += sum(1 for x in acts if x in (5, 6))
            moves += sum(1 for x in acts if x)
        finals.append([info["pdr"], info["coverage"], info["worst10_pdr"], info["score"]])
        alts += [p[2] for p in info["positions"]]
    f = np.mean(finals, axis=0)
    return {"pdr": f[0], "coverage": f[1], "worst10_pdr": f[2], "score": f[3],
            "z_move_share": z_moves / max(1, moves), "final_altitudes": alts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-root", default=str(ROOT / "models" / "ppo_multiuav_3d"))
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--smoke", action="store_true", help="2048 steps, tiny validation")
    ap.add_argument("--rf", choices=["free_space", "radiomap"], default="radiomap")
    ap.add_argument("--pdr-model", choices=["calibrated", "network"], default="calibrated")
    ap.add_argument("--pdr-model-path", default=None, help="calibrated-model joblib (default: v1)")
    ap.add_argument("--radiomap-samples", type=int, default=10_000_000)
    ap.add_argument("--train-windows", nargs="+", default=None, help="subset of the train windows")
    ap.add_argument("--val-windows", nargs="+", default=None)
    ap.add_argument("--reward-weights", default=None, help="JSON overrides of DEFAULT_REWARD_WEIGHTS")
    ap.add_argument("--checkpoint-every", type=int, default=25_000)
    ap.add_argument("--resume-from", default=None, help="checkpoint .zip to continue from (new run dir)")
    ap.add_argument("--heightmap", default=DEFAULT_HEIGHTMAP)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--pdr-obs-noise", type=float, default=0.0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--val-every", type=int, default=20_000)
    ap.add_argument("--val-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.timesteps, a.val_every, a.val_episodes, a.n_steps, a.batch_size = 2048, 1024, 2, 128, 64
        if a.rf == "radiomap":
            a.rf = "free_space"

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    run_id = a.run_id or (("smoke_" if a.smoke else "") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = Path(os.path.expanduser(a.out_root)) / run_id
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing run directory {out}")
    out.mkdir(parents=True)

    from stable_baselines3.common.monitor import Monitor
    venv = DummyVecEnv([lambda i=i: Monitor(make_env(a, "train", a.seed * 100 + i)) for i in range(a.n_envs)])
    val_env = make_env(a, "validation", 10_000 + a.seed)
    cfg = {**vars(a), "run_id": run_id, "obs_version": E.OBS_VERSION, "obs_len": E.OBS_LEN,
           "action_names": E.ACTION_NAMES, "max_uavs": E.MAX_UAVS,
           "step_h_m": val_env.step_h, "step_v_m": val_env.step_v, "z_levels": val_env.z_levels,
           "safety": val_env.safety, "reward_weights": val_env.w, "coverage_threshold": val_env.cov_thr,
           "train_windows": sorted(load_windows("train", a.train_windows)),
           "validation_windows": sorted(load_windows("validation", a.val_windows)),
           "rf_label": "free-space surrogate (NOT Sionna)" if a.rf == "free_space" else "Sionna RT radio maps",
           "pdr_model": val_env.pdr_model.name, "latency_in_reward": False, "held_out": ["urban_core"],
           "area_sampling": "uniform over train_windows, re-drawn every episode (MultiUAV3DEnv.reset)",
           "radiomap_sharing": "one read-only MultiWindowRF per window set, shared by the envs of the process",
           "provenance": provenance(a)}
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))

    if a.resume_from:   # continue training into this NEW run directory; the source is only read
        model = MaskablePPO.load(Path(a.resume_from).expanduser(), env=venv, device="cpu")
    else:
        model = MaskablePPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=a.batch_size, learning_rate=a.lr,
                            ent_coef=a.ent_coef, gamma=0.95, seed=a.seed, verbose=0, device="cpu")

    class Log(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0, self.rows, self.val_rows, self.best = time.time(), [], [], -1e9
            self.act_hist = np.zeros(E.N_ACTIONS, int)
            self.next_val = a.val_every
            self.next_prog = 0

        def _progress(self):          # persistent progress log, appended while training runs
            import resource
            f = out / "progress.csv"
            new = not f.exists()
            recent = [r[1] for r in self.rows[-200:]]
            with open(f, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["timesteps", "elapsed_s", "steps_per_s", "episodes", "mean_ep_return_last200",
                                "maxrss_mb"])
                el = time.time() - self.t0
                w.writerow([self.num_timesteps, round(el, 1), round(self.num_timesteps / max(el, 1e-9), 1),
                            len(self.rows), round(float(np.mean(recent)), 4) if recent else "",
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024])

        def _on_step(self):
            if self.num_timesteps >= self.next_prog:
                self.next_prog += 2048
                self._progress()
            for info in self.locals["infos"]:
                for x in info["actions"]:
                    self.act_hist[x] += 1
                if "episode" in info:
                    self.rows.append([self.num_timesteps, info["episode"]["r"], info["pdr"], info["coverage"],
                                      info["worst10_pdr"], info["window"]])
            if a.checkpoint_every and self.num_timesteps % a.checkpoint_every < a.n_envs:
                self.model.save(out / f"checkpoint_{self.num_timesteps:07d}.zip")
            if self.num_timesteps >= self.next_val:
                self.next_val += a.val_every
                v = run_policy(self.model, val_env, a.val_episodes, 50_000)
                self.val_rows.append([self.num_timesteps, v["score"], v["pdr"], v["coverage"],
                                      v["worst10_pdr"], v["z_move_share"]])
                with open(out / "val_progress.csv", "a", newline="") as fh:   # survives an interrupted run
                    csv.writer(fh).writerow(self.val_rows[-1])
                if v["score"] > self.best:          # model selection on the validation window only
                    self.best = v["score"]
                    self.model.save(out / "best_val_model.zip")
            return True

    cb = Log()
    t0 = time.time()
    model.learn(total_timesteps=a.timesteps, callback=cb, reset_num_timesteps=not a.resume_from)
    train_s = time.time() - t0
    cb._progress()
    model.save(out / "model.zip")
    with open(out / "train_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "ep_return", "pdr", "coverage", "worst10_pdr", "window"])
        w.writerows(cb.rows)
    with open(out / "val_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "score", "pdr", "coverage", "worst10_pdr", "z_move_share"])
        w.writerows(cb.val_rows)
    summary = {"run_id": run_id, "timesteps": int(model.num_timesteps), "train_seconds": round(train_s, 1),
               "steps_per_second": round(model.num_timesteps / train_s, 1),
               "action_histogram": dict(zip(E.ACTION_NAMES, cb.act_hist.tolist())),
               "episodes": len(cb.rows), "best_val_score": cb.best, "val_log": cb.val_rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, default=float))
    print("saved to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
