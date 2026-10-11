"""Multi-seed network evaluation: FIXED vs RANDOM vs PPO (legacy and network reward).

Every placement is judged under identical conditions:

  * same scenario, user positions (seeded), UAV count and altitude;
  * the same RF model (Sionna RT or surrogate) for the final placement;
  * the network-layer prediction (ml/network_model.py); and
  * an actual ns-3 run of ns3/uav-demo.cc with the best-RF association,
    exactly as scripts/run_full_pipeline.py hands it over.

Placements:
  fixed        - scenarios.initial_uav_positions, no movement
  random       - seeded uniform X/Y in the UAV region, same altitude as fixed
                 (the policies cannot change altitude either)
  ppo_legacy   - models/ppo_multiuav/model.zip rolled out with the legacy reward
  ppo_network  - models/ppo_multiuav_net/model.zip rolled out with the network reward

PPO rollouts start from the fixed layout and use the same RF model as the
final evaluation. ns-3 metrics are the ground truth for network claims.

    python ml/evaluate_network.py --scenarios blocked_urban wide_area \
        --seeds 20 --rf-mode surrogate --out data/experiments_network/surrogate_n30

Writes rows.json, summary.json and summary.md to --out (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import scenarios  # noqa: E402
import multiuav_io as mio  # noqa: E402
import network_model as nm  # noqa: E402
from multiuav_env import MultiUAVEnv  # noqa: E402

MODELS = {
    "ppo_legacy": (_ROOT / "models" / "ppo_multiuav" / "model.zip", "legacy"),
    "ppo_network": (_ROOT / "models" / "ppo_multiuav_net" / "model.zip", "network"),
}
PLACEMENTS = ["fixed", "random", "ppo_legacy", "ppo_network"]
# --extra-baselines (diagnostics, same 12-step / 10 m action budget as PPO):
#   greedy_oracle            - from the fixed layout, each step tries all 5^K joint
#                              actions and keeps the best network-reward move
#   ppo_*_from_random        - the policies started from the random layout
EXTRA_PLACEMENTS = ["greedy_oracle", "ppo_network_from_random", "ppo_legacy_from_random"]
PAIRS = [("ppo_network", "fixed"), ("ppo_network", "random"), ("ppo_network", "ppo_legacy")]
EXTRA_PAIRS = [("greedy_oracle", "fixed"), ("ppo_network", "greedy_oracle"),
               ("ppo_network_from_random", "random"), ("ppo_legacy_from_random", "random")]
SIM_SECONDS = 8.0  # uav-demo.cc: clients send from 2 s to 10 s

FLOW_RE = re.compile(r"Flow:\s+(\S+)\s+->\s+(\S+)(.*?)(?=\nFlow:|\n=+\n|\Z)", re.DOTALL)


# ----------------------------------------------------------------------
# placements
# ----------------------------------------------------------------------
def random_xy_positions(num_uavs, scenario, seed):
    reg = scenarios.get_scenario(scenario)["uav_region"]
    z = float(scenarios.initial_uav_positions(num_uavs, scenario)[0, 2])
    rng = np.random.default_rng(seed + 9999)
    return [[round(float(rng.uniform(*reg["x"])), 2), round(float(rng.uniform(*reg["y"])), 2), z]
            for _ in range(num_uavs)]


_LOADED = {}


def _start_env(sid, num_uavs, num_users, rf_mode, steps, seed, reward_mode, start):
    env = MultiUAVEnv(num_uavs=num_uavs, num_users=num_users, scenario=sid, mode=rf_mode,
                      seed=seed, max_steps=steps, reward_mode=reward_mode)
    obs, _ = env.reset(seed=seed)
    if start is not None:
        env.uav_positions = np.asarray(start, dtype=float)
        env.initial_positions = env.uav_positions.copy()
        env._refresh()
        obs = env._obs()
    return env, obs


def ppo_rollout(name, sid, num_uavs, num_users, rf_mode, steps, seed, start=None):
    from stable_baselines3 import PPO
    path, reward_mode = MODELS[name]
    if name not in _LOADED:
        _LOADED[name] = PPO.load(str(path))
    env, obs = _start_env(sid, num_uavs, num_users, rf_mode, steps, seed, reward_mode, start)
    actions = []
    for _ in range(steps):
        a, _ = _LOADED[name].predict(obs, deterministic=True)
        a = np.asarray(a).ravel()
        actions.append([int(x) for x in a[:num_uavs]])
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return env.uav_positions.tolist(), actions


def greedy_rollout(sid, num_uavs, num_users, rf_mode, steps, seed):
    """Exhaustive one-step lookahead on the network reward, same action budget."""
    import itertools
    env, _ = _start_env(sid, num_uavs, num_users, rf_mode, steps, seed, "network", None)
    actions = []
    for _ in range(steps):
        base_pos = env.uav_positions.copy()
        best, best_a = env.state_reward(), None
        for a in itertools.product(range(5), repeat=num_uavs):
            if not any(a):
                continue
            env.uav_positions = base_pos.copy()
            moved = env._apply(list(a) + [0] * (5 - num_uavs))
            env._refresh()
            r = env.state_reward() - env.w["w_move"] * moved
            if r > best + 1e-9:
                best, best_a = r, a
        env.uav_positions = base_pos.copy()
        if best_a is None:
            env._refresh()
            break
        env._apply(list(best_a) + [0] * (5 - num_uavs))
        env._refresh()
        actions.append(list(best_a))
    return env.uav_positions.tolist(), actions


# ----------------------------------------------------------------------
# evaluation of one final placement
# ----------------------------------------------------------------------
def rf_eval(uavs, users, rf_mode, sid, seed):
    import sionna_feedback_engine as sfe
    if rf_mode == "sionna":
        return sfe.evaluate_multi_uav(uavs, user_positions=users,
                                      scene_name=scenarios.get_scenario(sid)["scene"], seed=seed)
    from multiuav_env import surrogate_multi_uav
    return surrogate_multi_uav(uavs, users, connectivity_config=sfe.ConnectivityConfig())


def state_rewards(sid, uavs, users, rf, net):
    out = {}
    for mode in ("network", "legacy"):
        env = MultiUAVEnv(num_uavs=len(uavs), num_users=len(users), scenario=sid, reward_mode=mode)
        env.uav_positions = np.asarray(uavs, dtype=float)
        env.user_positions = np.asarray(users, dtype=float)
        env.last_eval, env.last_net = rf, net
        out[mode] = env.state_reward()
    return out


def ns3_binary(ns3_root: Path) -> Path:
    shutil.copy2(_ROOT / "ns3" / "uav-demo.cc", ns3_root / "scratch" / "uav-demo.cc")
    r = subprocess.run(["/usr/bin/python3.12", "./ns3", "build", "scratch/uav-demo"],
                       cwd=str(ns3_root), capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("ns-3 build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    hits = glob.glob(str(ns3_root / "build" / "scratch" / "*uav-demo*"))
    if not hits:
        raise FileNotFoundError("built uav-demo binary not found")
    return Path(hits[0])


def run_ns3(binary: Path, handoff: dict, args: tuple = ()) -> dict:
    """Run the built binary in its own working directory (it reads
    data/uav_positions.json relative to cwd), so runs can be parallel.
    ``args`` are extra ns-3 command-line arguments (e.g. --RngRun=3)."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "data").mkdir()
        (Path(td) / "data" / "uav_positions.json").write_text(json.dumps(handoff))
        r = subprocess.run([str(binary), *args], cwd=td, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    K = handoff["num_uavs"]
    per_user = {}
    for m in FLOW_RE.finditer(r.stdout):
        blk = m.group(3)
        g = lambda pat: re.search(pat, blk)  # noqa: E731
        sent, rx, dl = g(r"Packets Sent:\s+(\d+)"), g(r"Packets Received:\s+(\d+)"), g(r"Average Delay:\s+([\d.eE+-]+)")
        idx = int(m.group(1).split(".")[-1]) - K - 1
        per_user[idx] = {
            "sent": int(sent.group(1)) if sent else 0,
            "rx": int(rx.group(1)) if rx else 0,
            "delay_ms": float(dl.group(1)) if dl else None,
        }
    n = handoff["num_users"]
    users = [per_user.get(i, {"sent": 0, "rx": 0, "delay_ms": None}) for i in range(n)]
    sent = sum(u["sent"] for u in users)
    rx = sum(u["rx"] for u in users)
    delivering = [u for u in users if u["rx"] > 0]
    pdr_u = [u["rx"] / u["sent"] if u["sent"] else 0.0 for u in users]
    goodput = [u["rx"] * nm.IP_PACKET_BYTES * 8 / SIM_SECONDS / 1e6 for u in users]
    lat = (sum(u["delay_ms"] * u["rx"] for u in delivering) / rx) if rx else None
    return {
        "flows": len(per_user),
        "delivering_users": len(delivering),
        "users_pdr_ge_90": sum(1 for p in pdr_u if p >= 0.9),
        "pdr": rx / sent if sent else 0.0,
        "mean_user_throughput_mbps": float(np.mean(goodput)),
        "min_user_throughput_mbps": float(np.min(goodput)),
        "mean_latency_ms": lat,
        "p95_flow_latency_ms": float(np.percentile([u["delay_ms"] for u in delivering], 95)) if delivering else None,
        "per_user_rx": [u["rx"] for u in users],
        "per_user_sent": [u["sent"] for u in users],
    }


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return {"mean": float(a.mean()), "sd": 0.0, "ci95": 0.0, "n": len(a)}
    from scipy import stats
    half = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "ci95": half, "n": len(a)}


def paired(a, b):
    """a - b per seed: mean diff, wins/ties/losses, Wilcoxon signed-rank p."""
    d = np.asarray([x - y for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    p = 1.0
    if wins + losses > 0:
        from scipy import stats
        p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    return {"mean_diff": float(d.mean()) if len(d) else None, "wins": wins,
            "ties": int(len(d) - wins - losses), "losses": losses, "wilcoxon_p": p}


METRICS = [
    ("rf_coverage_pct", "RF coverage % (legacy PG>=-100 dB)"),
    ("rf_connected", "RF connected users (legacy thresholds)"),
    ("net_usable", "predicted usable ns-3 links"),
    ("ns3_delivering_users", "ns-3 users with any delivery"),
    ("ns3_users_pdr_ge_90", "ns-3 users with PDR >= 90%"),
    ("ns3_pdr", "ns-3 aggregate PDR"),
    ("ns3_mean_user_throughput_mbps", "ns-3 mean per-user goodput (Mbps)"),
    ("ns3_min_user_throughput_mbps", "ns-3 worst-user goodput (Mbps)"),
    ("ns3_mean_latency_ms", "ns-3 mean latency (ms)"),
    ("reward_network", "network reward (state)"),
    ("reward_legacy", "legacy reward (state)"),
    ("movement_m", "mean UAV movement (m)"),
]
HIGHER_IS_BETTER = {k: k not in ("ns3_mean_latency_ms", "movement_m") for k, _ in METRICS}


def summarise(rows, scen_list, placements=PLACEMENTS, pairs=PAIRS):
    summary = {}
    for sid in scen_list:
        srows = [r for r in rows if r["scenario"] == sid]
        seeds = sorted({r["seed"] for r in srows})
        by = {(r["placement"], r["seed"]): r for r in srows}
        s = {"seeds": seeds, "placements": {}, "paired": {}}
        for pl in placements:
            s["placements"][pl] = {k: mean_ci([by[(pl, sd)][k] for sd in seeds if (pl, sd) in by])
                                   for k, _ in METRICS}
        for cand, base in pairs:
            s["paired"][f"{cand}_vs_{base}"] = {
                k: paired([by[(cand, sd)][k] for sd in seeds], [by[(base, sd)][k] for sd in seeds])
                for k, _ in METRICS
            }
        # network-model vs ns-3 agreement (per user, all placements)
        agree = total = 0
        for r in srows:
            for u, rx in zip(r["net_usable_per_user"], r["ns3_per_user_rx"]):
                agree += int(u == (rx > 0)); total += 1
        s["model_vs_ns3_user_agreement"] = agree / total if total else None
        from scipy import stats
        pp = [r["net_predicted_pdr"] for r in srows]; mp = [r["ns3_pdr"] for r in srows]
        s["model_vs_ns3_pdr_spearman"] = (float(stats.spearmanr(pp, mp).statistic)
                                          if len(set(pp)) > 1 and len(set(mp)) > 1 else None)
        summary[sid] = s
    return summary


def fmt(ci, digits=3):
    if ci is None:
        return "—"
    return f"{ci['mean']:.{digits}f} ± {ci['ci95']:.{digits}f}"


PAIRED_KEYS = ["net_usable", "ns3_delivering_users", "ns3_users_pdr_ge_90", "ns3_pdr",
               "ns3_mean_user_throughput_mbps", "ns3_mean_latency_ms", "reward_network"]


def to_markdown(summary, meta):
    placements = meta["placements"]
    lines = [f"# Network evaluation — {meta['label']}", "",
             f"RF model: **{meta['rf_mode']}** · users: {meta['num_users']} · UAVs: {meta['num_uavs']} · "
             f"PPO steps: {meta['steps']} · seeds per scenario: {meta['num_seeds']} · "
             "values are mean ± 95% CI over seeds.", ""]
    labels = dict(METRICS)
    for sid, s in summary.items():
        lines += [f"## {sid}", "", "| metric | " + " | ".join(placements) + " |",
                  "|---|" + "---|" * len(placements)]
        for k, label in METRICS:
            lines.append(f"| {label} | " + " | ".join(fmt(s["placements"][pl][k]) for pl in placements) + " |")
        lines += ["", "Paired per-seed differences (A − B), wins/ties/losses for A in the metric's good "
                  "direction, Wilcoxon signed-rank p:", "",
                  "| comparison | " + " | ".join(labels[k] for k in PAIRED_KEYS) + " |",
                  "|---|" + "---|" * len(PAIRED_KEYS)]
        for pair, ps in s["paired"].items():
            cells = []
            for k in PAIRED_KEYS:
                p = ps[k]
                w, l = (p["wins"], p["losses"]) if HIGHER_IS_BETTER[k] else (p["losses"], p["wins"])
                cells.append(f"{p['mean_diff']:+.3f} ({w}/{p['ties']}/{l}, p={p['wilcoxon_p']:.2g})"
                             if p["mean_diff"] is not None else "—")
            lines.append(f"| {pair.replace('_vs_', ' − ')} | " + " | ".join(cells) + " |")
        lines += ["", f"Network model vs ns-3: per-user usable-link agreement "
                  f"{s['model_vs_ns3_user_agreement']:.3f}; Spearman(predicted PDR, ns-3 PDR) = "
                  f"{s['model_vs_ns3_pdr_spearman'] if s['model_vs_ns3_pdr_spearman'] is None else round(s['model_vs_ns3_pdr_spearman'], 3)}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["blocked_urban", "wide_area"])
    ap.add_argument("--seeds", type=int, default=20, help="evaluation seeds 0..N-1")
    ap.add_argument("--num-users", type=int, default=30)
    ap.add_argument("--num-uavs", type=int, default=3)
    ap.add_argument("--rf-mode", choices=["surrogate", "sionna"], default="surrogate")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6, help="parallel ns-3 runs")
    ap.add_argument("--extra-baselines", action="store_true",
                    help="also run greedy_oracle and the PPO policies started from the random layout")
    ap.add_argument("--ns3-path", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists - refusing to overwrite")
    for name, (path, _) in MODELS.items():
        if not path.exists():
            raise SystemExit(f"missing model for {name}: {path}")
    ns3_root = Path(args.ns3_path or os.environ.get("NS3_PATH")
                    or (Path.home() / "ns-allinone-3.47" / "ns-3.47")).expanduser()
    binary = ns3_binary(ns3_root)
    placements = PLACEMENTS + (EXTRA_PLACEMENTS if args.extra_baselines else [])
    pairs = PAIRS + (EXTRA_PAIRS if args.extra_baselines else [])

    t0 = time.time()
    cases = []
    for sid in args.scenarios:
        for seed in range(args.seeds):
            users = scenarios.generate_users(sid, args.num_users, seed).tolist()
            fixed = scenarios.initial_uav_positions(args.num_uavs, sid).tolist()
            rand = random_xy_positions(args.num_uavs, sid, seed)
            finals = {"fixed": (fixed, []), "random": (rand, [])}
            starts = {pl: fixed for pl in placements}
            starts["random"] = rand
            for name in MODELS:
                finals[name] = ppo_rollout(name, sid, args.num_uavs, args.num_users,
                                           args.rf_mode, args.steps, seed)
            if args.extra_baselines:
                finals["greedy_oracle"] = greedy_rollout(sid, args.num_uavs, args.num_users,
                                                         args.rf_mode, args.steps, seed)
                for name in MODELS:
                    finals[f"{name}_from_random"] = ppo_rollout(
                        name, sid, args.num_uavs, args.num_users, args.rf_mode, args.steps, seed, start=rand)
                    starts[f"{name}_from_random"] = rand
            for pl in placements:
                uavs, actions = finals[pl]
                rf = rf_eval(uavs, users, args.rf_mode, sid, seed)
                net = nm.evaluate_network(rf, uavs, users)
                rew = state_rewards(sid, uavs, users, rf, net)
                assoc = [u["serving_uav"] - 1 for u in rf["user_results"]]
                start = starts[pl]
                cases.append({
                    "scenario": sid, "seed": seed, "placement": pl, "rf_mode": args.rf_mode,
                    "uav_positions": [[round(c, 2) for c in p] for p in uavs], "actions": actions,
                    "movement_m": float(np.mean([math.dist(a, b) for a, b in zip(start, uavs)])),
                    "rf_coverage_pct": rf["aggregate"]["coverage_percentage"],
                    "rf_connected": rf["aggregate"]["connected_users_count"],
                    "net_usable": net["aggregate"]["net_connected_count"],
                    "net_predicted_pdr": net["aggregate"]["predicted_pdr"],
                    "net_usable_per_user": [p["usable"] for p in net["per_user"]],
                    "reward_network": rew["network"], "reward_legacy": rew["legacy"],
                    "_handoff": mio.build_uav_positions(uavs, users, associations=assoc),
                })
            print(f"[{time.time() - t0:6.0f}s] {sid} seed {seed}: placements ready", flush=True)

    print(f"running {len(cases)} ns-3 simulations with {args.workers} workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda c: run_ns3(binary, c["_handoff"]), cases))
    rows = []
    for c, n in zip(cases, results):
        c = {k: v for k, v in c.items() if k != "_handoff"}
        c.update({
            "ns3_flows": n["flows"], "ns3_delivering_users": n["delivering_users"],
            "ns3_users_pdr_ge_90": n["users_pdr_ge_90"], "ns3_pdr": n["pdr"],
            "ns3_mean_user_throughput_mbps": n["mean_user_throughput_mbps"],
            "ns3_min_user_throughput_mbps": n["min_user_throughput_mbps"],
            "ns3_mean_latency_ms": n["mean_latency_ms"], "ns3_p95_flow_latency_ms": n["p95_flow_latency_ms"],
            "ns3_per_user_rx": n["per_user_rx"],
        })
        rows.append(c)

    meta = {"label": out.name, "rf_mode": args.rf_mode, "num_users": args.num_users,
            "num_uavs": args.num_uavs, "steps": args.steps, "num_seeds": args.seeds,
            "scenarios": args.scenarios, "models": {k: str(v[0]) for k, v in MODELS.items()},
            "placements": placements,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_sec": round(time.time() - t0, 1)}
    summary = summarise(rows, args.scenarios, placements, pairs)
    out.mkdir(parents=True)
    (out / "rows.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    (out / "summary.md").write_text(to_markdown(summary, meta))
    print(to_markdown(summary, meta))
    print(f"\nsaved -> {out}  ({meta['wall_time_sec']} s)")


if __name__ == "__main__":
    main()
