"""Shared-altitude experiment: distance-only vs obstruction-aware ns-3 (urban_core).

For every user layout (seed) and shared UAV altitude, the SAME geometry is
ray traced once with Sionna RT and simulated in ns-3 twice:

  distance     - the unchanged ns-3 channel (LogDistance only)
  obstruction  - the same plus Sionna's per UAV-user obstruction loss
                 (ml/obstruction.py: loss in excess of free space, so the
                 distance term is not counted twice)

Both modes use the same Sionna best-server association, positions, traffic and
ns-3 random runs, so the obstruction matrix is the only difference. Sionna is
run at the ns-3 Wi-Fi operating frequency, read from the built binary.

    python ml/evaluate_obstruction.py --config configs/eval_urban_core.json \
        --seeds 10 --out ~/.cache/resqnet_obstruction/main

MEASURED: ns-3 PDR / goodput / latency (per run, averaged over --rng-runs).
Latency is reported two ways: over all delivered packets (as elsewhere in the
project), and - for altitude comparisons and the decision criteria - over
MATCHED users only: users that deliver packets at both altitudes in the same
layout and ns-3 run. The first rewards altitudes that serve fewer users (less
contention among the survivors); the second does not.
RAY TRACED: Sionna path gain / obstruction loss / geometric line of sight.
PREDICTED: network_model and the calibrated PDR model (labelled as such).

Every ns-3 run is cached under --out/ns3_runs (resumable; config.json pins the
settings, code and binary). Failed runs are recorded, never dropped silently.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
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
import network_model as nm  # noqa: E402
import evaluate_network as ev  # noqa: E402
import checkpoint as ck  # noqa: E402
import obstruction as ob  # noqa: E402

SIONNA_SEED = 42          # path-solver sampling seed (fixed: geometry varies, not the solver)
EXPERIMENT_PROTOCOL = 1
WORST_FRACTION = 0.10     # "worst-decile user PDR" = mean PDR of the worst 10 % of users

# Decision criteria (fixed before the experiment). A higher altitude is a
# meaningful improvement over a baseline altitude (15 m and 20 m separately),
# in obstruction mode, only if ALL of:
#   1. at least one headline gain: mean PDR >= +0.02, worst-decile PDR >= +0.10,
#      or mean latency <= -20 % (latency over MATCHED users - delivering at both
#      altitudes in the same layout and ns-3 run; chosen after the smoke test,
#      thresholds unchanged);
#   2. that metric's paired bootstrap 95 % CI excludes 0 in the favourable direction;
#   3. it improves in >= 80 % of seeds;
#   4. no headline metric worsens by at least its threshold;
#   5. the same metric's mean gain in distance-only mode is < 50 % of the
#      obstruction-mode gain (the gain is due to obstruction).
CRIT = {"pdr": 0.02, "worst_decile_pdr": 0.10, "mean_latency_rel": -0.20,
        "min_win_frac": 0.80, "attribution_frac": 0.50}


# ----------------------------------------------------------------------
# ns-3 binary
# ----------------------------------------------------------------------
def find_binary(ns3_root: Path) -> Path:
    """The built uav-demo binary; refuses one not built from this repo's source."""
    src_repo = _ROOT / "ns3" / "uav-demo.cc"
    src_ns3 = ns3_root / "scratch" / "uav-demo.cc"
    if not src_ns3.exists() or ck.file_sha256(src_ns3) != ck.file_sha256(src_repo):
        raise SystemExit(f"{src_ns3} differs from {src_repo}: build the current source first")
    hits = sorted(glob.glob(str(ns3_root / "build" / "scratch" / "*uav-demo*")))
    if not hits:
        raise SystemExit("built uav-demo binary not found")
    b = Path(hits[0])
    if b.stat().st_mtime < src_ns3.stat().st_mtime:
        raise SystemExit(f"{b} is older than {src_ns3}: rebuild it")
    return b


CHANNEL_RE = re.compile(r"CHANNEL number=(\d+) frequency_mhz=([\d.]+) width_mhz=([\d.]+) band=(\S+)")


def ns3_channel(binary: Path) -> dict:
    """Ask the binary for its Wi-Fi operating channel (tiny 1 UAV / 1 user run)."""
    h = mio.build_uav_positions([[0.0, 0.0, 10.0]], [[5.0, 0.0, 1.5]])
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "data").mkdir()
        (Path(td) / "data" / "uav_positions.json").write_text(json.dumps(h))
        r = subprocess.run([str(binary), "--printChannel=1", "--trafficSeconds=0.1"], cwd=td,
                           capture_output=True, text=True, timeout=300)
    m = CHANNEL_RE.search(r.stdout)
    if r.returncode != 0 or not m:
        raise SystemExit("could not read the ns-3 operating channel:\n" + r.stdout[-1000:] + r.stderr[-1000:])
    return {"number": int(m.group(1)), "frequency_mhz": float(m.group(2)),
            "width_mhz": float(m.group(3)), "band": m.group(4)}


def run_ns3_detailed(binary: Path, handoff: dict, args: tuple = ()) -> dict:
    """evaluate_network.run_ns3 plus per-user mean delay, and a check that ns-3
    applied the obstruction matrix exactly when the handoff carries one."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "data").mkdir()
        (Path(td) / "data" / "uav_positions.json").write_text(json.dumps(handoff))
        r = subprocess.run([str(binary), *args], cwd=td, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"ns-3 exit {r.returncode}: {r.stderr[-2000:]}")
    applied = "Obstruction loss:" in r.stdout
    if applied != ("obstruction_loss_db" in handoff):
        raise RuntimeError(f"obstruction matrix in handoff={'obstruction_loss_db' in handoff} "
                           f"but applied by ns-3={applied}")
    K, n = handoff["num_uavs"], handoff["num_users"]
    per_user = {}
    for m in ev.FLOW_RE.finditer(r.stdout):
        blk = m.group(3)
        g = lambda pat: re.search(pat, blk)  # noqa: E731
        sent, rx, dl = g(r"Packets Sent:\s+(\d+)"), g(r"Packets Received:\s+(\d+)"), g(r"Average Delay:\s+([\d.eE+-]+)")
        idx = int(m.group(1).split(".")[-1]) - K - 1
        per_user[idx] = {"sent": int(sent.group(1)) if sent else 0, "rx": int(rx.group(1)) if rx else 0,
                         "delay_ms": float(dl.group(1)) if dl else None}
    users = [per_user.get(i, {"sent": 0, "rx": 0, "delay_ms": None}) for i in range(n)]
    sent = sum(u["sent"] for u in users)
    rx = sum(u["rx"] for u in users)
    delivering = [u for u in users if u["rx"] > 0]
    pdr_u = [u["rx"] / u["sent"] if u["sent"] else 0.0 for u in users]
    goodput = [u["rx"] * nm.IP_PACKET_BYTES * 8 / ev.SIM_SECONDS / 1e6 for u in users]
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
        "per_user_delay_ms": [u["delay_ms"] if u["rx"] > 0 else None for u in users],
        "obstruction_applied": applied,
    }


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def matched_latency(runs_a: list[dict], runs_b: list[dict]) -> dict | None:
    """Mean per-user latency of A and B over MATCHED users: users delivering
    packets in both, compared within the same ns-3 run (paired by rng_run).
    Per run: unweighted mean of the matched users' mean delays; then averaged
    over runs with at least one matched user. None if nothing matches."""
    b_by_run = {r["rng_run"]: r for r in runs_b if not r.get("failed")}
    la, lb, counts, n_users = [], [], [], None
    for ra in runs_a:
        rb = b_by_run.get(ra["rng_run"])
        if ra.get("failed") or rb is None:
            continue
        da, db = ra["per_user_delay_ms"], rb["per_user_delay_ms"]
        n_users = len(da)
        idx = [i for i in range(len(da)) if da[i] is not None and db[i] is not None
               and ra["per_user_rx"][i] > 0 and rb["per_user_rx"][i] > 0]
        counts.append(len(idx))
        if idx:
            la.append(float(np.mean([da[i] for i in idx])))
            lb.append(float(np.mean([db[i] for i in idx])))
    if not la:
        return None
    return {"a_ms": float(np.mean(la)), "b_ms": float(np.mean(lb)),
            "rel_change": float(np.mean(la) - np.mean(lb)) / float(np.mean(lb)) if np.mean(lb) else None,
            "matched_users_per_run": counts, "matched_users_mean": float(np.mean(counts)),
            "num_users": n_users, "runs_paired": len(counts)}


def point_metrics(runs: list[dict]) -> dict:
    """Average the successful ns-3 runs of one (seed, altitude, mode)."""
    ok = [r for r in runs if not r.get("failed")]
    out = {"runs_ok": len(ok), "runs_failed": len(runs) - len(ok),
           "errors": [r["error"] for r in runs if r.get("failed")]}
    if not ok:
        return out
    rx = np.mean([r["per_user_rx"] for r in ok], axis=0)
    sent = np.mean([r["per_user_sent"] for r in ok], axis=0)
    pdr_u = rx / np.maximum(sent, 1)
    tput = rx * nm.IP_PACKET_BYTES * 8 / ev.SIM_SECONDS / 1e6
    n_worst = max(1, int(round(WORST_FRACTION * len(pdr_u))))
    lat = [r["mean_latency_ms"] for r in ok if r["mean_latency_ms"] is not None]
    p95 = [r["p95_flow_latency_ms"] for r in ok if r["p95_flow_latency_ms"] is not None]
    out.update({
        "pdr": float(rx.sum() / max(sent.sum(), 1)),
        "worst_decile_pdr": float(np.sort(pdr_u)[:n_worst].mean()),
        "users_pdr_ge_90": int((pdr_u >= 0.9).sum()),
        "delivering_users": float(np.mean([r["delivering_users"] for r in ok])),
        "mean_user_tput_mbps": float(tput.mean()),
        "min_user_tput_mbps": float(tput.min()),
        "mean_latency_ms": float(np.mean(lat)) if lat else None,
        "p95_flow_latency_ms": float(np.mean(p95)) if p95 else None,
        "per_user_pdr": pdr_u.round(4).tolist(),
        "per_run_pdr": [r["pdr"] for r in ok],
    })
    return out


METRICS = [  # key, label, higher_is_better
    ("pdr", "aggregate PDR", True),
    ("worst_decile_pdr", "worst-decile user PDR", True),
    ("users_pdr_ge_90", "users with PDR >= 90 %", True),
    ("mean_user_tput_mbps", "mean per-user goodput (Mbps)", True),
    ("min_user_tput_mbps", "worst-user goodput (Mbps)", True),
    ("mean_latency_ms", "mean latency (ms)", False),
    ("p95_flow_latency_ms", "p95 per-flow latency (ms)", False),
]


def ci95(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return 0.0
    from scipy import stats
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))


def paired(a, b, higher_better, n_boot=5000, seed=0):
    """a - b over seeds where both exist."""
    d = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d))).mean(1)
    better = (d > 1e-9) if higher_better else (d < -1e-9)
    worse = (d < -1e-9) if higher_better else (d > 1e-9)
    p = 1.0
    if (np.abs(d) > 1e-12).any():
        from scipy import stats
        p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int(better.sum()), "losses": int(worse.sum()), "ties": int(len(d) - better.sum() - worse.sum()),
            "wilcoxon_p": p}


def matched_rel_latency(runs_by, mode, z, base, seeds):
    """Per seed: relative change of matched-user latency (z vs base), and the
    matched-user details."""
    rel, det = [], []
    for s in seeds:
        m = matched_latency(runs_by.get((mode, z, s), []), runs_by.get((mode, base, s), []))
        rel.append(None if m is None else m["rel_change"])
        det.append(m)
    return rel, det


def decide(by, runs_by, seeds, z, base, modes):
    """Evaluate the predefined criteria for altitude z against altitude `base`.
    Latency is compared over matched users (delivering at both altitudes)."""
    def vals(mode, alt, key):
        return [by.get((mode, alt, s), {}).get(key) for s in seeds]

    res = {"altitude": z, "baseline": base, "gains": {}, "met": False, "reasons": []}
    if "obstruction" not in modes:
        res["reasons"].append("obstruction mode not run")
        return res
    rel_lat, det = matched_rel_latency(runs_by, "obstruction", z, base, seeds)
    res["matched_users"] = {"per_seed_mean": [None if d is None else d["matched_users_mean"] for d in det],
                            "num_users": next((d["num_users"] for d in det if d), None)}
    cand = {
        "pdr": (vals("obstruction", z, "pdr"), vals("obstruction", base, "pdr"), True, CRIT["pdr"]),
        "worst_decile_pdr": (vals("obstruction", z, "worst_decile_pdr"),
                             vals("obstruction", base, "worst_decile_pdr"), True, CRIT["worst_decile_pdr"]),
        "mean_latency_rel": (rel_lat, [0.0 if r is not None else None for r in rel_lat], False,
                             CRIT["mean_latency_rel"]),
    }
    worsened = []
    for key, (a, b, hb, thr) in cand.items():
        st = paired(a, b, hb)
        if st is None:
            continue
        gain = st["mean_diff"]
        big = gain >= thr if hb else gain <= thr
        ci_ok = st["ci95"][0] > 0 if hb else st["ci95"][1] < 0
        win_ok = st["wins"] >= CRIT["min_win_frac"] * st["n"]
        bad = gain <= -thr if hb else gain >= -thr
        if bad:
            worsened.append(key)
        # attribution: the same gain in distance-only mode
        attr = None
        if "distance" in modes:
            if key == "mean_latency_rel":
                dd = [x for x in matched_rel_latency(runs_by, "distance", z, base, seeds)[0] if x is not None]
            else:
                dd = [x - y for x, y in zip(vals("distance", z, key), vals("distance", base, key))
                      if x is not None and y is not None]
            dgain = float(np.mean(dd)) if dd else 0.0
            attr = abs(dgain) < CRIT["attribution_frac"] * abs(gain) or (dgain < 0 if hb else dgain > 0)
        res["gains"][key] = {**st, "threshold": thr, "large_enough": bool(big), "ci_excludes_0": bool(ci_ok),
                             "wins_ok": bool(win_ok), "attributable_to_obstruction": attr}
    passing = [k for k, g in res["gains"].items()
               if g["large_enough"] and g["ci_excludes_0"] and g["wins_ok"] and g["attributable_to_obstruction"] is not False]
    res["passing_metrics"] = passing
    res["worsened_metrics"] = worsened
    res["met"] = bool(passing) and not worsened
    if not passing:
        res["reasons"].append("no headline metric meets size + CI + win-rate + attribution")
    if worsened:
        res["reasons"].append(f"worsened: {worsened}")
    return res


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "eval_urban_core.json"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--altitudes", type=float, nargs="+", default=None)
    ap.add_argument("--rng-runs", type=int, nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", choices=["distance", "obstruction"], default=["distance", "obstruction"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--ns3-root", default=os.environ.get("NS3_PATH") or str(Path.home() / "ns-allinone-3.47" / "ns-3.47"))
    ap.add_argument("--min-free-mb", type=float, default=2048.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    exp = cfg["experiment"]
    sid = exp["scenario"]
    scen = cfg["scenarios"][sid]
    if sid not in scenarios._EXTRA_SCENARIOS:
        scenarios.register_scenarios({sid: scen})
    alts = [float(z) for z in (args.altitudes or exp["altitudes_m"])]
    rng_runs = list(args.rng_runs or exp["rng_runs"])
    seeds = list(range(args.seeds))

    out = Path(args.out).expanduser()
    binary = find_binary(Path(args.ns3_root))
    chan = ns3_channel(binary)
    freq_hz = chan["frequency_mhz"] * 1e6

    import sionna_feedback_engine as sfe
    settings = {"protocol": EXPERIMENT_PROTOCOL, "scenario": sid, "seeds": seeds, "altitudes": alts,
                "rng_runs": rng_runs, "modes": args.modes, "freq_hz": freq_hz, "ns3_channel": chan,
                "sionna_seed": SIONNA_SEED, "experiment": exp, "criteria": CRIT}
    files = {f"code/{n}": _HERE / n for n in ("obstruction.py", "evaluate_obstruction.py", "network_model.py",
                                             "multiuav_io.py", "evaluate_network.py", "calibrated_model.py")}
    files.update({"code/sionna_feedback_engine.py": _ROOT / "sionna" / "sionna_feedback_engine.py",
                  "config/eval_urban_core.json": cfg_path, "ns3/uav-demo.cc": _ROOT / "ns3" / "uav-demo.cc",
                  "ns3/binary": binary, "scene/xml": _ROOT / "sionna" / "scenes" / scen["scene"],
                  "scene/obj": _ROOT / "sionna" / "scenes" / scen["scene"].replace(".xml", ".obj")})
    fp = ck.fingerprint(settings, files)
    out.mkdir(parents=True, exist_ok=True)
    resuming = ck.check_or_write_config(out, fp)
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except (BrokenPipeError, OSError):
            pass

    log(("RESUMING" if resuming else "STARTING") + f" {out}; ns-3 channel {chan}; Sionna at {freq_hz/1e6:g} MHz")

    # ---------------- geometry (Sionna scene, same coordinates as ns-3) -----
    conn = sfe.ConnectivityConfig()
    probe_uav = [[scen["centre"][0], scen["centre"][1], 100.0]]
    sfe.evaluate_multi_uav(probe_uav, user_positions=[[scen["centre"][0], scen["centre"][1] + 5, 1.5]],
                           scene_name=scen["scene"], freq=freq_hz, connectivity_config=conn, seed=SIONNA_SEED)
    engine = sfe._GLOBAL_ENGINE
    mi_scene = engine.scene.mi_scene

    geo_path = out / "geometry.json"
    geo = ck.read_json(geo_path)
    if geo is None:
        uav_xy = ob.roof_clear_uav_xy(mi_scene, scen["centre"], exp["uav_offsets_m"],
                                      search_m=exp["uav_snap_search_m"], radius_m=exp["roof_radius_m"])
        floor = max(u["local_roof_m"] for u in uav_xy) + exp["roof_clearance_m"]
        z_eff = {str(z): max(z, round(floor, 2)) for z in alts}
        users = {str(s): ob.street_level_users(mi_scene, scen["user_region"], exp["num_users"],
                                              exp["user_seed_base"] + s, user_z=exp["user_z_m"],
                                              clearance_m=exp["user_clearance_m"]) for s in seeds}
        geo = {"uav_xy": uav_xy, "roof_floor_m": floor, "altitude_effective": z_eff, "users": users}
        ck.atomic_write_json(geo_path, geo)
    adjusted = {z: e for z, e in geo["altitude_effective"].items() if float(z) != e}
    log(f"UAV x/y {[(u['x'], u['y'], u['local_roof_m']) for u in geo['uav_xy']]}; "
        f"altitude floor {geo['roof_floor_m']:.1f} m; adjusted altitudes: {adjusted or 'none'}")

    # ---------------- Sionna + predictions per (seed, altitude) -------------
    import calibrated_model as cm
    cal = cm.CalibratedNetworkModel()
    points = {}
    for s in seeds:
        users = geo["users"][str(s)]
        for z in alts:
            ze = geo["altitude_effective"][str(z)]
            uavs = [[u["x"], u["y"], ze] for u in geo["uav_xy"]]
            rf_path = out / "rf" / f"s{s}_z{z:g}.json"
            rec = ck.read_json(rf_path)
            if rec is None:
                t0 = time.time()
                rf = sfe.evaluate_multi_uav(uavs, user_positions=users, scene_name=scen["scene"], freq=freq_hz,
                                            connectivity_config=conn, seed=SIONNA_SEED)
                M, mstats = ob.obstruction_matrix_db(rf, uavs, users, freq_hz)
                tx = [uavs[k] for k in range(len(uavs)) for _ in users]
                rx = [u for _ in uavs for u in users]
                los = ob.line_of_sight(mi_scene, tx, rx).reshape(len(uavs), len(users))
                serving = [u["serving_uav"] - 1 for u in rf["user_results"]]
                serv_loss = [M[serving[i]][i] for i in range(len(users))]
                net = nm.evaluate_network(rf, uavs, users, freq_hz=freq_hz)["aggregate"]
                calr = cal.evaluate(rf, uavs, users)["aggregate"]
                rec = {"seed": s, "altitude": z, "altitude_effective": ze, "uav_positions": uavs,
                       "sionna_seconds": round(time.time() - t0, 2), "matrix": M, "matrix_stats": mstats,
                       "associations": serving, "los_fraction": float(los.mean()),
                       "serving_los_fraction": float(np.mean([los[serving[i], i] for i in range(len(users))])),
                       "serving_mean_loss_db": float(np.mean([v for v in serv_loss if v < ob.NO_PATH_LOSS_DB] or [0])),
                       "serving_no_path": int(sum(v >= ob.NO_PATH_LOSS_DB for v in serv_loss)),
                       "best_serving_pg_db": float(np.mean([u["best_path_gain_db"] for u in rf["user_results"]])),
                       "predicted_network_model": {k: net[k] for k in ("net_connected_fraction", "predicted_pdr")},
                       "predicted_calibrated": {k: calr[k] for k in ("calibrated_pdr", "worst_decile_pdr")}}
                ck.atomic_write_json(rf_path, rec)
            points[(s, z)] = rec
        log(f"seed {s}: Sionna done")

    # ---------------- ns-3 ------------------------------------------------
    def run_fn(handoff, run_args):
        try:
            return run_ns3_detailed(binary, handoff, run_args)
        except Exception as e:  # recorded, not dropped
            return {"failed": True, "error": str(e)[-800:]}

    runner = ck.CachedNs3Runner(out / "ns3_runs", fp["files"]["ns3/binary"], run_fn,
                                workers=args.workers, min_free_mb=args.min_free_mb, log=log)
    runner.install_signal_handlers()
    jobs, meta = [], []
    for (s, z), rec in points.items():
        users = geo["users"][str(s)]
        base = mio.build_uav_positions(rec["uav_positions"], users, associations=rec["associations"])
        for mode in args.modes:
            h = base if mode == "distance" else ob.with_obstruction(base, rec["matrix"], freq_hz)
            for r in rng_runs:
                jobs.append((h, (f"--RngRun={r}",)))
                meta.append((s, z, mode, r))
    log(f"ns-3: {len(jobs)} runs ({sum(runner.cached(h, a) is not None for h, a in jobs)} cached)")
    try:
        results = runner.map(jobs)
    except ck.StopRequested as e:
        log(f"STOPPED SAFELY: {e}; rerun the same command to resume")
        sys.exit(3)

    # ---------------- aggregate ---------------------------------------------
    grouped = {}
    for (s, z, mode, r), res in zip(meta, results):
        grouped.setdefault((mode, z, s), []).append({**res, "rng_run": r})
    by, rows, failures = {}, [], []
    runs_by = grouped
    for (mode, z, s), runs in sorted(grouped.items()):
        m = point_metrics(runs)
        rec = points[(s, z)]
        row = {"mode": mode, "altitude": z, "altitude_effective": rec["altitude_effective"], "seed": s, **m,
               "sionna": {k: rec[k] for k in ("matrix_stats", "los_fraction", "serving_los_fraction",
                                              "serving_mean_loss_db", "serving_no_path", "best_serving_pg_db")},
               "predicted_network_model": rec["predicted_network_model"],
               "predicted_calibrated": rec["predicted_calibrated"]}
        rows.append(row)
        by[(mode, z, s)] = row
        for rr in runs:
            if rr.get("failed"):
                failures.append({"mode": mode, "altitude": z, "seed": s, "rng_run": rr["rng_run"], "error": rr["error"]})

    summary = {"modes": {}, "paired": {}, "decisions": [], "failures": failures}
    for mode in args.modes:
        summary["modes"][mode] = {}
        for z in alts:
            d = {}
            for k, _, _ in METRICS:
                v = [by[(mode, z, s)].get(k) for s in seeds if by.get((mode, z, s), {}).get(k) is not None]
                if v:
                    d[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                            "ci95": ci95(v), "n": len(v)}
            summary["modes"][mode][str(z)] = d
        for base in [b for b in (15.0, 20.0) if b in alts]:
            for z in alts:
                if z == base:
                    continue
                st = {k: paired([by.get((mode, z, s), {}).get(k) for s in seeds],
                                [by.get((mode, base, s), {}).get(k) for s in seeds], hb)
                      for k, _, hb in METRICS}
                rel, det = matched_rel_latency(runs_by, mode, z, base, seeds)
                st["matched_latency_ms"] = paired([None if d is None else d["a_ms"] for d in det],
                                                  [None if d is None else d["b_ms"] for d in det], False)
                st["matched_latency_rel"] = paired(rel, [None if r is None else 0.0 for r in rel], False)
                st["matched_users"] = {"per_seed_mean": [None if d is None else d["matched_users_mean"] for d in det],
                                       "num_users": next((d["num_users"] for d in det if d), None)}
                summary["paired"][f"{mode}:{z:g}_vs_{base:g}"] = st
    for base in [b for b in (15.0, 20.0) if b in alts]:
        for z in alts:
            if z > 20.0:
                summary["decisions"].append(decide(by, runs_by, seeds, z, base, args.modes))
    sionna = {}
    for z in alts:
        recs = [points[(s, z)] for s in seeds]
        sionna[str(z)] = {
            "mean_loss_db": float(np.mean([r["matrix_stats"]["mean_loss_db"] for r in recs])),
            "median_loss_db": float(np.mean([r["matrix_stats"]["median_loss_db"] for r in recs])),
            "links_over_10db_frac": float(np.mean([r["matrix_stats"]["links_over_10db"] / r["matrix_stats"]["links"] for r in recs])),
            "no_path_links": int(sum(r["matrix_stats"]["no_path_links"] for r in recs)),
            "clamped_links": int(sum(r["matrix_stats"]["clamped_negative_links"] for r in recs)),
            "los_fraction": float(np.mean([r["los_fraction"] for r in recs])),
            "serving_los_fraction": float(np.mean([r["serving_los_fraction"] for r in recs])),
            "serving_mean_loss_db": float(np.mean([r["serving_mean_loss_db"] for r in recs])),
            "serving_no_path": int(sum(r["serving_no_path"] for r in recs)),
            "best_serving_pg_db": float(np.mean([r["best_serving_pg_db"] for r in recs])),
            "predicted_pdr_network_model": float(np.mean([r["predicted_network_model"]["predicted_pdr"] for r in recs])),
            "predicted_pdr_calibrated": float(np.mean([r["predicted_calibrated"]["calibrated_pdr"] for r in recs])),
        }
    summary["sionna"] = sionna
    meta_out = {"scenario": sid, "seeds": seeds, "altitudes": alts, "altitude_effective": geo["altitude_effective"],
                "rng_runs": rng_runs, "modes": args.modes, "ns3_channel": chan, "freq_hz": freq_hz,
                "uav_xy": geo["uav_xy"], "criteria": CRIT, "runner": runner.stats,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ck.atomic_write_json(out / "rows.json", {"meta": meta_out, "rows": rows})
    ck.atomic_write_json(out / "summary.json", {"meta": meta_out, "summary": summary})
    md = to_markdown(meta_out, summary, by, seeds, alts, args.modes)
    (out / "summary.md").write_text(md)
    try:
        print(md)
    except (BrokenPipeError, OSError):
        pass
    log(f"COMPLETE -> {out} ({runner.stats}); failed ns-3 runs: {len(failures)}")


def to_markdown(meta, summary, by, seeds, alts, modes) -> str:
    L = [f"# Obstruction-aware altitude experiment — {meta['scenario']}", "",
         f"ns-3 channel {meta['ns3_channel']['number']} at {meta['ns3_channel']['frequency_mhz']:g} MHz; "
         f"Sionna run at the same frequency. Seeds {len(seeds)}, ns-3 RngRun {meta['rng_runs']} averaged per seed. "
         f"Effective altitudes: {meta['altitude_effective']}. Failed ns-3 runs: {len(summary['failures'])}.", "",
         "## Sionna (ray traced, not a network measurement)", "",
         "| z | mean obstruction dB (all links) | median | links >10 dB | no-path links | clamped | geometric LOS (all / serving) | serving-link obstruction dB | serving no-path | mean serving path gain dB | PREDICTED PDR network_model / calibrated |",
         "|---" * 11 + "|"]
    for z in alts:
        s = summary["sionna"][str(z)]
        L.append(f"| {z:g} | {s['mean_loss_db']:.2f} | {s['median_loss_db']:.2f} | {100*s['links_over_10db_frac']:.0f}% | "
                 f"{s['no_path_links']} | {s['clamped_links']} | {100*s['los_fraction']:.0f}% / {100*s['serving_los_fraction']:.0f}% | "
                 f"{s['serving_mean_loss_db']:.2f} | {s['serving_no_path']} | {s['best_serving_pg_db']:.1f} | "
                 f"{s['predicted_pdr_network_model']:.3f} / {s['predicted_pdr_calibrated']:.3f} |")
    for mode in modes:
        L += ["", f"## ns-3 MEASURED — {mode} mode (mean ± 95% CI over seeds; sd)", "",
              "| z | " + " | ".join(lbl for _, lbl, _ in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for z in alts:
            d = summary["modes"][mode][str(z)]
            L.append(f"| {z:g} | " + " | ".join(
                f"{d[k]['mean']:.4f} ± {d[k]['ci95']:.4f} (sd {d[k]['sd']:.4f})" if k in d else "—"
                for k, _, _ in METRICS) + " |")
        L += ["", f"Per-seed aggregate PDR / worst-decile PDR / mean latency ms ({mode}):", "",
              "| seed | " + " | ".join(f"z{z:g}" for z in alts) + " |", "|---" * (len(alts) + 1) + "|"]
        for s in seeds:
            cells = []
            for z in alts:
                r = by.get((mode, z, s), {})
                cells.append("FAILED" if "pdr" not in r else
                             f"{r['pdr']:.4f} / {r['worst_decile_pdr']:.3f} / "
                             f"{r['mean_latency_ms'] if r['mean_latency_ms'] is None else round(r['mean_latency_ms'], 1)}"
                             + (f" ({r['runs_failed']} failed)" if r.get("runs_failed") else ""))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "## Paired differences (altitude − baseline), mean [bootstrap 95% CI] (wins/ties/losses, Wilcoxon p)", "",
          "mean_latency_ms = over all delivered packets; matched_latency_ms / _rel = over users delivering at "
          "both altitudes in the same layout and ns-3 run (count in the last column).", ""]
    keys = ["pdr", "worst_decile_pdr", "users_pdr_ge_90", "mean_latency_ms", "matched_latency_ms",
            "matched_latency_rel", "min_user_tput_mbps"]
    L += ["| comparison | " + " | ".join(keys) + " | matched users per run (mean over seeds, range) |",
          "|---" * (len(keys) + 2) + "|"]
    for name, st in summary["paired"].items():
        cells = []
        for k in keys:
            q = st.get(k)
            cells.append("—" if q is None else f"{q['mean_diff']:+.4f} [{q['ci95'][0]:+.4f}, {q['ci95'][1]:+.4f}] "
                                                f"({q['wins']}/{q['ties']}/{q['losses']}, p={q['wilcoxon_p']:.2g})")
        mu = [v for v in st["matched_users"]["per_seed_mean"] if v is not None]
        cells.append(f"{np.mean(mu):.1f} of {st['matched_users']['num_users']} ({min(mu):.1f}–{max(mu):.1f})" if mu else "—")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## Decision criteria (obstruction mode; thresholds fixed before the run, latency over matched users)", "",
          f"Thresholds: {CRIT}", ""]
    for d in summary["decisions"]:
        mu = [v for v in d.get("matched_users", {}).get("per_seed_mean", []) if v is not None]
        matched = f"matched users {np.mean(mu):.1f}/{d['matched_users']['num_users']}; " if mu else "no matched users; "
        L.append(f"- z={d['altitude']:g} vs {d['baseline']:g} m: **{'MET' if d['met'] else 'NOT met'}** — "
                 f"passing {d.get('passing_metrics')}, worsened {d.get('worsened_metrics')}; "
                 + matched + "; ".join(d["reasons"]))
        for k, g in d["gains"].items():
            L.append(f"  - {k}: mean {g['mean_diff']:+.4f} CI [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] "
                     f"wins {g['wins']}/{g['n']}; large={g['large_enough']} ci={g['ci_excludes_0']} "
                     f"wins_ok={g['wins_ok']} attributable={g['attributable_to_obstruction']}")
    if summary["failures"]:
        L += ["", "## Failed ns-3 runs", ""] + [f"- {f}" for f in summary["failures"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
