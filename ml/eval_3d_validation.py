"""Offline comparison on the VALIDATION window (val_a; never urban_core) in the
training model (precomputed Sionna radio maps + PDR model). These are
PREDICTED network metrics, not ns-3 measurements; they are for model
selection only. The real comparison is closed-loop Sionna + ns-3.

Methods, all from identical start states / users per seed:
  fixed          : stay at the start positions
  random_actions : uniformly random safe action per UAV and step
  random_place   : one random safe lattice placement per seed (no movement cost counted)
  greedy_model   : per step, the single safe one-UAV move with the best predicted score
                   (the rule-based controller's idea, on the training model)
  ppo            : the checkpoint, deterministic
  ppo_no_z       : the same checkpoint with up / down masked (tests whether altitude choices help)

Usage: python ml/eval_3d_validation.py --checkpoint models/ppo_multiuav_3d/<run>/model.zip --episodes 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import multiuav3d_env as E  # noqa: E402
import train_multiuav3d as T  # noqa: E402
from closed_loop_control import safety_violations  # noqa: E402


def rollout(env, seed, policy) -> dict:
    obs, info = env.reset(seed=seed)
    start = env.P.copy()
    h = v = 0.0
    viol = 0
    first = dict(info)
    done = False
    while not done:
        a = policy(env, obs)
        prev = env.P.copy()
        obs, r, term, trunc, info = env.step(a)
        d = env.P - prev
        h += float(np.linalg.norm(d[:, :2], axis=1).sum())
        v += float(np.abs(d[:, 2]).sum())
        for k in range(env.K):
            others = [env.P[j] for j in range(env.K) if j != k]
            viol += bool(safety_violations(env.P[k], others, env.region, env.safety, env.geometry))
        done = term or trunc
    return {"seed": seed, "K": env.K, "N": len(env.users), "start": start.tolist(), "final": env.P.tolist(),
            "start_metrics": {k: first[k] for k in ("pdr", "coverage", "worst10_pdr", "score")},
            **{k: info[k] for k in ("pdr", "coverage", "worst10_pdr", "throughput_mbps", "score")},
            "horizontal_m": h, "vertical_m": v, "safety_violations": viol,
            "altitudes": env.P[:, 2].tolist(), "distinct_altitudes": len(set(env.P[:, 2].round(1)))}


def stay(env, obs):
    return np.zeros(E.MAX_UAVS, int)


def make_random(rng):
    def pol(env, obs):
        m = env.action_masks().reshape(E.MAX_UAVS, E.N_ACTIONS)
        return np.array([rng.choice(np.flatnonzero(m[k])) for k in range(E.MAX_UAVS)])
    return pol


def greedy(env, obs):
    m = env.action_masks().reshape(E.MAX_UAVS, E.N_ACTIONS)
    P0 = env.P.copy()
    best, best_a = env.score, None
    for k in range(env.K):
        for j in range(1, E.N_ACTIONS):
            if not m[k, j]:
                continue
            P = P0.copy()
            P[k] = env.target(k, j)
            pg = np.asarray(env.rf.path_gain(P, env.users))
            pdr = np.clip(env.pdr_model(P, env.users, pg), 0, 1)
            mt = env.metrics_from(pdr)
            s = env.w["pdr"] * mt["pdr"] + env.w["coverage"] * mt["coverage"] + env.w["fairness"] * mt["worst10_pdr"]
            s -= env.w["move"] * (math.dist(P[k], P0[k]) / (env.K * max(env.step_h, env.step_v)))
            if s > best + 1e-6:
                best, best_a = s, (k, j)
    a = np.zeros(E.MAX_UAVS, int)
    if best_a:
        a[best_a[0]] = best_a[1]
    return a


def make_ppo(model, no_z=False):
    def pol(env, obs):
        m = env.action_masks().reshape(E.MAX_UAVS, E.N_ACTIONS).copy()
        if no_z:
            m[:, 5:] = False
        a, _ = model.predict(obs, deterministic=True, action_masks=m.ravel())
        return a
    return pol


def summarise(rows) -> dict:
    keys = ["pdr", "coverage", "worst10_pdr", "throughput_mbps", "score", "horizontal_m", "vertical_m",
            "safety_violations", "distinct_altitudes"]
    s = {k: {"mean": float(np.mean([r[k] for r in rows])),
             "ci95": float(1.96 * np.std([r[k] for r in rows], ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0}
         for k in keys}
    alts = np.concatenate([r["altitudes"] for r in rows])
    s["altitude_hist"] = {f"{z:g}": int((alts == z).sum()) for z in sorted(set(alts.tolist()))}
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=90_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rf", default="radiomap", choices=["radiomap", "free_space"])
    ap.add_argument("--pdr-model", default="calibrated", choices=["calibrated", "network"])
    ap.add_argument("--pdr-model-path", default=None)
    ap.add_argument("--radiomap-samples", type=int, default=10_000_000)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--val-windows", nargs="+", default=None)
    ap.add_argument("--role", default="validation", choices=["validation", "train"],
                    help="train = diagnostic run on a training window (never used for model selection)")
    ap.add_argument("--methods", nargs="+", default=["fixed", "random_actions", "random_place", "greedy_model",
                                                      "ppo", "ppo_no_z"])
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    ns = argparse.Namespace(heightmap=T.DEFAULT_HEIGHTMAP, rf=a.rf, pdr_model=a.pdr_model,
                            pdr_model_path=a.pdr_model_path, radiomap_samples=a.radiomap_samples,
                            max_steps=a.max_steps, pdr_obs_noise=0.0, train_windows=None,
                            val_windows=a.val_windows, reward_weights=None)
    ns.train_windows = a.val_windows
    env = T.make_env(ns, a.role, 0)
    model = None
    if a.checkpoint:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(Path(a.checkpoint).expanduser(), device="cpu")
    pols = {"fixed": stay, "random_actions": make_random(np.random.default_rng(1)), "greedy_model": greedy}
    if model is not None:
        pols["ppo"], pols["ppo_no_z"] = make_ppo(model), make_ppo(model, no_z=True)
    res, t0 = {}, time.time()
    for name in a.methods:
        if name == "random_place":
            rows = []
            for e in range(a.episodes):
                env.reset(seed=a.seed0 + e)
                users, K = env.users.tolist(), env.K
                env.random_start_prob = 1.0
                env._rng = np.random.default_rng(a.seed0 + e + 7)
                P = env._start_positions(K)
                env.random_start_prob = 0.5
                _, info = env.reset(options={"window": env.window, "users": users, "positions": P.tolist()})
                rows.append({"seed": a.seed0 + e, "K": K, "final": P.tolist(),
                             **{k: info[k] for k in ("pdr", "coverage", "worst10_pdr", "throughput_mbps", "score")},
                             "horizontal_m": 0.0, "vertical_m": 0.0, "safety_violations": 0,
                             "altitudes": P[:, 2].tolist(), "distinct_altitudes": len(set(P[:, 2].round(1)))})
        elif name in pols:
            rows = [rollout(env, a.seed0 + e, pols[name]) for e in range(a.episodes)]
        else:
            continue
        res[name] = {"summary": summarise(rows), "episodes": rows}
        print(name, json.dumps({k: round(v["mean"], 4) for k, v in res[name]["summary"].items()
                                if isinstance(v, dict) and "mean" in v}), f"{time.time() - t0:.0f}s", flush=True)
    meta = {"role": a.role, "checkpoint": a.checkpoint, "episodes": a.episodes, "seed0": a.seed0, "windows": sorted(env.windows),
            "rf": a.rf, "pdr_model": env.pdr_model.name, "pdr_model_path": a.pdr_model_path,
            "note": "PREDICTED metrics (training model on the validation window), not ns-3 measurements"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": res}, indent=1, default=float))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
