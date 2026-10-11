"""Final evaluation of placement methods on held-out scenarios, scored in ns-3.

Every method sees the same scenario, users (seed), UAV count, altitude, radio
settings and traffic. Final placements are scored with the full 8 s ns-3 run
under TWO ns-3 random streams (RngRun 11 and 12) that are never used by any
optimiser (ns3_refine searches with RngRun 101 and 3 s traffic; the calibrated
model was fitted on RngRun 1-5 of training scenarios), and the two runs are
averaged.

Methods
  fixed, random            baselines (random = seeded X/Y, same altitude)
  ppo_network              previous candidate (models/ppo_multiuav_net)
  ppo_calibrated           PPO trained on the ns-3-calibrated reward (models/ppo_multiuav_cal)
  greedy_joint, coordinate, evolution
                           surrogate optimisers on the same calibrated objective,
                           equal budget of surrogate evaluations
  ns3_refine               measured-feedback local search in ns-3, started from
                           the best of {fixed, coordinate, ppo_calibrated}
  ns3_refine_iter          (opt-in via --methods) the same search, iterated per
                           step size until no measured gain (<= REFINE_ITER_ROUNDS)

    python ml/evaluate_closed_loop.py --seeds 20 --workers 4 --out data/experiments_closed_loop/main

--reuse-cache DIR looks up simulations already finished in another
experiment's ns3_runs (read-only; the key covers binary, handoff and args).

Checkpointing: every ns-3 simulation is cached in --out/ns3_runs as soon as it
finishes (atomic writes, content-addressed), placements per instance in
--out/placements, finished per-(scenario, seed, method) results in
--out/results. Rerunning the same command resumes and only simulates what is
missing; config.json pins the settings, code, model and ns-3 hashes so
incompatible results are never combined. SIGINT/SIGTERM or sustained low
memory stops safely (exit code 3). Final rows.json, summary.json and
summary.md are written when every job is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import scenarios  # noqa: E402
import multiuav_io as mio  # noqa: E402
import evaluate_network as ev  # noqa: E402
import placement_search as ps  # noqa: E402
import calibrated_model as cm  # noqa: E402

ev.MODELS["ppo_calibrated"] = (_ROOT / "models" / "ppo_multiuav_cal" / "model.zip", "calibrated")
FINAL_RNG_RUNS = (11, 12)
SURROGATE_BUDGET = 1500
METHODS = ["fixed", "random", "ppo_network", "ppo_calibrated", "ppo_calibrated_30",
           "greedy_joint", "coordinate", "evolution", "ns3_refine"]
PRIMARY = "pdr"
REFINE_ITER_ROUNDS = 5
NS3_REFINERS = {"ns3_refine": 1, "ns3_refine_iter": REFINE_ITER_ROUNDS}   # method -> rounds per step size

METRICS = [
    ("delivering_users", "users with any delivery", True),
    ("users_pdr_ge_90", "users with PDR >= 90%", True),
    ("pdr", "aggregate PDR", True),
    ("loss_pct", "aggregate packet loss %", False),
    ("mean_user_tput_mbps", "mean per-user goodput (Mbps)", True),
    ("min_user_tput_mbps", "worst-user goodput (Mbps)", True),
    ("jain_fairness", "Jain fairness (per-user goodput)", True),
    ("mean_latency_ms", "mean latency (ms)", False),
    ("p95_flow_latency_ms", "p95 per-flow latency (ms)", False),
    ("calibrated_pdr", "calibrated-model PDR (predicted)", True),
    ("movement_m", "mean UAV movement (m)", False),
]


def register_eval_scenarios():
    cfg = json.loads((_ROOT / "configs" / "eval_network.json").read_text())
    new = {k: v for k, v in cfg["extra_eval_scenarios"].items() if k not in scenarios._EXTRA_SCENARIOS}
    scenarios.register_scenarios(new)
    return cfg["eval_scenarios"]


def handoff_for(uavs, users, sid, seed, rf_mode="surrogate"):
    rf = ev.rf_eval(uavs, users, rf_mode, sid, seed)
    return rf, mio.build_uav_positions(uavs, users, associations=[u["serving_uav"] - 1 for u in rf["user_results"]])


def score(results: list[dict]) -> dict:
    """Average the final ns-3 runs of one placement into per-placement metrics."""
    rx = np.mean([r["per_user_rx"] for r in results], axis=0)
    sent = np.mean([r["per_user_sent"] for r in results], axis=0)
    pdr_u = rx / np.maximum(sent, 1)
    tput = rx * ev.nm.IP_PACKET_BYTES * 8 / ev.SIM_SECONDS / 1e6
    lat = [r["mean_latency_ms"] for r in results if r["mean_latency_ms"] is not None]
    p95 = [r["p95_flow_latency_ms"] for r in results if r["p95_flow_latency_ms"] is not None]
    jain = float(tput.sum() ** 2 / (len(tput) * (tput ** 2).sum())) if tput.sum() > 0 else 0.0
    pdr = float(rx.sum() / sent.sum())
    return {
        "delivering_users": float(np.mean([r["delivering_users"] for r in results])),
        "users_pdr_ge_90": int((pdr_u >= 0.9).sum()),
        "pdr": pdr, "loss_pct": 100 * (1 - pdr),
        "mean_user_tput_mbps": float(tput.mean()), "min_user_tput_mbps": float(tput.min()),
        "jain_fairness": jain,
        "mean_latency_ms": float(np.mean(lat)) if lat else None,
        "p95_flow_latency_ms": float(np.mean(p95)) if p95 else None,
        "per_user_pdr": pdr_u.round(4).tolist(),
    }


def paired_stats(a, b, higher_better, n_boot=5000, seed=0):
    d = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d))).mean(1)
    wins = int((d > 1e-9).sum()) if higher_better else int((d < -1e-9).sum())
    losses = int((d < -1e-9).sum()) if higher_better else int((d > 1e-9).sum())
    p = 1.0
    if (np.abs(d) > 1e-12).any():
        from scipy import stats
        p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": wins, "ties": int(len(d) - wins - losses), "losses": losses, "wilcoxon_p": p}


def summarise(rows, scen_list, methods, candidate):
    summary = {}
    for sid in scen_list:
        sr = [r for r in rows if r["scenario"] == sid]
        seeds = sorted({r["seed"] for r in sr})
        by = {(r["method"], r["seed"]): r for r in sr}
        s = {"seeds": seeds, "methods": {}, "paired": {}}
        for m in methods:
            s["methods"][m] = {}
            for k, _, _ in METRICS:
                vals = [by[(m, sd)][k] for sd in seeds if (m, sd) in by and by[(m, sd)][k] is not None]
                if vals:
                    a = np.array(vals, dtype=float)
                    from scipy import stats
                    half = float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
                    s["methods"][m][k] = {"mean": float(a.mean()), "ci95": half}
            s["methods"][m]["budget"] = {
                "surrogate_evals": float(np.mean([by[(m, sd)]["surrogate_evals"] for sd in seeds if (m, sd) in by])),
                "ns3_search_runs": float(np.mean([by[(m, sd)]["ns3_search_runs"] for sd in seeds if (m, sd) in by])),
            }
        for base in [m for m in methods if m != candidate]:
            s["paired"][f"{candidate}_vs_{base}"] = {
                k: paired_stats([by[(candidate, sd)][k] for sd in seeds], [by[(base, sd)][k] for sd in seeds], hb)
                for k, _, hb in METRICS}
        summary[sid] = s
    # Holm correction of the primary metric (candidate vs fixed) across scenarios
    ps_ = sorted(((summary[sid]["paired"][f"{candidate}_vs_fixed"][PRIMARY]["wilcoxon_p"], sid) for sid in scen_list))
    m = len(ps_)
    running = 0.0
    for i, (p, sid) in enumerate(ps_):
        running = max(running, min(1.0, (m - i) * p))
        summary[sid]["paired"][f"{candidate}_vs_fixed"][PRIMARY]["holm_p"] = running
    return summary


def to_markdown(summary, meta):
    methods, cand = meta["methods"], meta["candidate"]
    L = [f"# Closed-loop placement evaluation — {meta['label']}", "",
         f"RF: **{meta['rf_mode']}** · {meta['num_users']} users · {meta['num_uavs']} UAVs · seeds {meta['num_seeds']} · "
         f"final ns-3 scoring: 8 s traffic, RngRun {list(FINAL_RNG_RUNS)} averaged · mean ± 95% CI over seeds", ""]
    for sid, s in summary.items():
        L += [f"## {sid}", "", "| metric | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
        for k, label, _ in METRICS:
            cells = []
            for m in methods:
                v = s["methods"][m].get(k)
                cells.append(f"{v['mean']:.3f} ± {v['ci95']:.3f}" if v else "—")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("| budget: surrogate evals / ns-3 search runs | " + " | ".join(
            f"{s['methods'][m]['budget']['surrogate_evals']:.0f} / {s['methods'][m]['budget']['ns3_search_runs']:.0f}"
            for m in methods) + " |")
        L += ["", f"Paired per-seed differences, **{cand}** − baseline: mean [bootstrap 95% CI] "
              "(wins/ties/losses, Wilcoxon p):", "",
              "| baseline | " + " | ".join(lbl for k, lbl, _ in METRICS[:8]) + " |", "|---|" + "---|" * 8]
        for pair, st in s["paired"].items():
            cells = []
            for k, _, _ in METRICS[:8]:
                q = st[k]
                if q is None:
                    cells.append("—"); continue
                extra = f", Holm p={q['holm_p']:.2g}" if "holm_p" in q else ""
                cells.append(f"{q['mean_diff']:+.3f} [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}] "
                             f"({q['wins']}/{q['ties']}/{q['losses']}, p={q['wilcoxon_p']:.2g}{extra})")
            L.append(f"| {pair.split('_vs_')[1]} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


EVAL_PROTOCOL = 2   # bump when the evaluation procedure itself changes


def result_files() -> dict:
    """Files whose content determines this evaluation's results."""
    ml = _ROOT / "ml"
    files = {f"code/{n}": ml / n for n in (
        "multiuav_env.py", "network_model.py", "calibrated_model.py", "placement_search.py",
        "scenarios.py", "evaluate_network.py", "multiuav_io.py")}
    files.update({
        "code/surrogate_rf": _ROOT / "sionna" / "sionna_feedback_engine.py",
        "config/scenarios.json": _ROOT / "configs" / "scenarios.json",
        "config/eval_network.json": _ROOT / "configs" / "eval_network.json",
        "ns3/uav-demo.cc": _ROOT / "ns3" / "uav-demo.cc",
        "model/calibrated": cm.DEFAULT_MODEL,
    })
    for name, (path, _mode) in ev.MODELS.items():
        files[f"model/{name}"] = path
    return files


def _log(out: Path):
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(out / "run.log", "a") as f:      # the log file first: stdout may be gone
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except (BrokenPipeError, OSError):          # e.g. the wsl.exe wrapper was killed (SIGHUP)
            pass
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--num-users", type=int, default=30)
    ap.add_argument("--num-uavs", type=int, default=3)
    ap.add_argument("--rf-mode", choices=["surrogate", "sionna"], default="surrogate")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--candidate", default="ns3_refine")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel ns-3 simulations (default 4; each 30-user run needs ~100-200 MB)")
    ap.add_argument("--min-free-mb", type=float, default=2048.0,
                    help="do not start a simulation while MemAvailable is below this; stop safely if "
                         "it stays below for --low-mem-wait seconds")
    ap.add_argument("--low-mem-wait", type=float, default=180.0)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="testing aid: stop safely after this many new simulations")
    ap.add_argument("--out", required=True,
                    help="results directory; rerunning the same command resumes it")
    ap.add_argument("--reuse-cache", nargs="*", default=[],
                    help="read-only ns3_runs directories of earlier experiments to reuse")
    args = ap.parse_args()

    out = Path(args.out)
    scen_list = args.scenarios or register_eval_scenarios()
    register_eval_scenarios()
    ns3_root = Path(os.environ.get("NS3_PATH") or Path.home() / "ns-allinone-3.47" / "ns-3.47")
    binary = ev.ns3_binary(ns3_root)

    import checkpoint as ck
    settings = {"protocol": EVAL_PROTOCOL, "scenarios": scen_list, "seeds": args.seeds,
                "num_users": args.num_users, "num_uavs": args.num_uavs, "rf_mode": args.rf_mode,
                "methods": args.methods, "candidate": args.candidate,
                "final_rng_runs": list(FINAL_RNG_RUNS), "search_rng_run": ps.SEARCH_RNG_RUN,
                "search_traffic_seconds": ps.SEARCH_TRAFFIC_SECONDS, "surrogate_budget": SURROGATE_BUDGET,
                "refine_rounds": {m: r for m, r in NS3_REFINERS.items() if m in args.methods}}
    fp = ck.fingerprint(settings, {**result_files(), "ns3/binary": binary})
    out.mkdir(parents=True, exist_ok=True)
    resuming = ck.check_or_write_config(out, fp)
    log = _log(out)
    log(("RESUMING" if resuming else "STARTING") + f" {out} with {args.workers} workers")

    new_sims = {"n": 0}

    def run_fn(handoff, run_args):
        r = ev.run_ns3(binary, handoff, run_args)
        new_sims["n"] += 1
        if args.stop_after is not None and new_sims["n"] >= args.stop_after:
            runner.request_stop(f"--stop-after {args.stop_after} reached")
        return r

    runner = ck.CachedNs3Runner(out / "ns3_runs", fp["files"]["ns3/binary"], run_fn,
                                workers=args.workers, min_free_mb=args.min_free_mb,
                                low_mem_wait_s=args.low_mem_wait, log=log,
                                fallback_dirs=[Path(d) for d in args.reuse_cache])
    runner.install_signal_handlers()
    t0 = time.time()

    def progress(stage: str, extra: dict | None = None):
        ck.atomic_write_json(out / "progress.json", {
            "stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 1), "runner": runner.stats, **(extra or {})})

    try:
        # ---------------- placements (cached per instance) ----------------
        inst = []
        for sid in scen_list:
            for seed in range(args.seeds):
                path = out / "placements" / f"{sid}_s{seed}.json"
                saved = ck.read_json(path)
                if saved is None:
                    if runner.stop.is_set():
                        raise ck.StopRequested(runner.stop_reason)
                    users = scenarios.generate_users(sid, args.num_users, seed).tolist()
                    fixed = scenarios.initial_uav_positions(args.num_uavs, sid).tolist()
                    pl, budget = {"fixed": fixed}, {"fixed": (0, 0)}
                    pl["random"], budget["random"] = ev.random_xy_positions(args.num_uavs, sid, seed), (0, 0)
                    for name in ("ppo_network", "ppo_calibrated", "ppo_calibrated_30"):
                        if name in args.methods:
                            # *_30: same policy, 30 decision steps instead of the 12 it was trained with
                            model_name, steps = (name[:-3], 30) if name.endswith("_30") else (name, 12)
                            pl[name] = ev.ppo_rollout(model_name, sid, args.num_uavs, args.num_users,
                                                      args.rf_mode, steps, seed)[0]
                            budget[name] = (steps, 0)
                    for name, fn in ps.SURROGATE_OPTIMISERS.items():
                        if name in args.methods or (name == "coordinate" and any(m in args.methods for m in NS3_REFINERS)):
                            obj = ps.SurrogateObjective(sid, users, args.num_uavs)
                            pos, _ = fn(obj, fixed, budget=SURROGATE_BUDGET)
                            pl[name], budget[name] = pos.round(2).tolist(), (obj.evals, 0)
                    saved = {"scenario": sid, "seed": seed, "users": users, "fixed": fixed,
                             "pl": pl, "budget": budget}
                    ck.atomic_write_json(path, saved)
                inst.append(saved)
            log(f"{sid}: placements ready")
        progress("placements done")

        # ---------------- ns-3 in the loop (every measurement cached) -----
        search_args = (f"--RngRun={ps.SEARCH_RNG_RUN}", f"--trafficSeconds={ps.SEARCH_TRAFFIC_SECONDS}")

        def measure(jobs):
            hand = [handoff_for(pos.tolist(), inst[i]["users"], inst[i]["scenario"], inst[i]["seed"],
                                args.rf_mode)[1] for i, pos in jobs]
            res = runner.map([(h, search_args) for h in hand])
            progress("ns3_refine")
            return [r["pdr"] for r in res]

        for name, rounds in NS3_REFINERS.items():
            if name not in args.methods:
                continue
            rinst = []
            for x in inst:
                starts = [x["fixed"], x["pl"]["coordinate"]] + (
                    [x["pl"]["ppo_calibrated"]] if "ppo_calibrated" in x["pl"] else [])
                lo, hi = ps.region_bounds(x["scenario"])
                rinst.append({"candidates": starts, "lo": lo, "hi": hi})
            res = ps.ns3_refine_batch(rinst, measure, max_rounds=rounds)
            suffix = "" if name == "ns3_refine" else f"_{name}"
            for x, r in zip(inst, res):
                x["pl"][name] = r["pos"].round(2).tolist()
                x["budget"][name] = (x["budget"]["coordinate"][0]
                                     + (12 if "ppo_calibrated" in x["pl"] else 0), r["runs"])
                x[f"{name}_history"] = r["history"]
                ck.atomic_write_json(out / "refine" / f"{x['scenario']}_s{x['seed']}{suffix}.json",
                                     {"pos": x["pl"][name], "runs": r["runs"], "history": r["history"]})
            log(f"{name}: ns-3 in-the-loop refinement done ({runner.stats})")
        progress("refine done")

        # ---------------- final scoring (incremental per scenario/seed/method) ----
        cal_model = cm.CalibratedNetworkModel()
        jobs, meta_jobs = [], []
        for x in inst:
            for m in args.methods:
                rf, h = handoff_for(x["pl"][m], x["users"], x["scenario"], x["seed"], args.rf_mode)
                cal = cal_model.evaluate(rf, x["pl"][m], x["users"])["aggregate"]["calibrated_pdr"]
                for rr in FINAL_RNG_RUNS:
                    jobs.append((h, (f"--RngRun={rr}",)))
                    meta_jobs.append((x, m, cal))
        got: dict = {}

        def on_done(i, result):
            x, m, cal = meta_jobs[i]
            k = (x["scenario"], x["seed"], m)
            got.setdefault(k, []).append(result)
            if len(got[k]) == len(FINAL_RNG_RUNS):
                rec = {"scenario": k[0], "seed": k[1], "method": m, **score(got[k]), "calibrated_pdr": cal,
                       "uav_positions": x["pl"][m],
                       "movement_m": float(np.mean([math.dist(a, b) for a, b in zip(x["fixed"], x["pl"][m])])),
                       "surrogate_evals": x["budget"][m][0], "ns3_search_runs": x["budget"][m][1],
                       "ns3_refine_history": x.get(f"{m}_history") if m in NS3_REFINERS else None}
                ck.atomic_write_json(out / "results" / f"{k[0]}_s{k[1]}_{m}.json", rec)

        log(f"final scoring: {len(jobs)} ns-3 runs ({sum(runner.cached(h, a) is not None for h, a in jobs)} cached)")
        runner.map(jobs, on_done=on_done)
    except ck.StopRequested as e:
        progress("stopped", {"reason": str(e)})
        log(f"STOPPED SAFELY: {e}. Completed simulations are saved in {out / 'ns3_runs'}; "
            "rerun the same command to resume.")
        sys.exit(3)

    final = [ck.read_json(p) for p in sorted((out / "results").glob("*.json"))]
    expected = len(scen_list) * args.seeds * len(args.methods)
    if len(final) != expected:
        raise SystemExit(f"expected {expected} results, found {len(final)}")
    meta = {"label": out.name, "rf_mode": args.rf_mode, "num_users": args.num_users, "num_uavs": args.num_uavs,
            "num_seeds": args.seeds, "scenarios": scen_list, "methods": args.methods, "candidate": args.candidate,
            "final_rng_runs": list(FINAL_RNG_RUNS), "search_rng_run": ps.SEARCH_RNG_RUN,
            "surrogate_budget": SURROGATE_BUDGET, "runner": runner.stats,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    summary = summarise(final, scen_list, args.methods, args.candidate)
    ck.atomic_write_json(out / "rows.json", {"meta": meta, "rows": final})
    ck.atomic_write_json(out / "summary.json", {"meta": meta, "summary": summary})
    md = to_markdown(summary, meta)
    (out / "summary.md.tmp").write_text(md)
    os.replace(out / "summary.md.tmp", out / "summary.md")
    progress("complete")
    try:
        print(md)
    except (BrokenPipeError, OSError):
        pass
    log(f"COMPLETE -> {out} ({runner.stats})")


if __name__ == "__main__":
    main()
