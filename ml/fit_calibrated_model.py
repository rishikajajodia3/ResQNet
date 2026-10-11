"""Generate ns-3 ground truth and fit the calibrated per-user PDR model.

    python ml/fit_calibrated_model.py --train-runs 280 --val-runs 60

* training data: ns-3 runs on the TRAINING scenarios of
  configs/training_network.json only (seeds 5000+, varied placements);
* validation: ns-3 runs on the held-out scenarios of configs/eval_network.json
  with seeds 900+ (disjoint from the evaluation seeds 0..N-1);
* output: models/network_surrogate_v1/{model.joblib, report.json} and the raw
  per-user dataset in data/network_surrogate_v1/ (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json
import sys
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
import calibrated_model as cm  # noqa: E402
import evaluate_network as ev  # noqa: E402


def register_all():
    tr = json.loads((_ROOT / "configs" / "training_network.json").read_text())
    evc = json.loads((_ROOT / "configs" / "eval_network.json").read_text())
    for defs in (tr["extra_train_scenarios"], evc["extra_eval_scenarios"]):
        scenarios.register_scenarios({k: v for k, v in defs.items() if k not in scenarios._EXTRA_SCENARIOS})
    return tr["train"]["scenarios"], evc["eval_scenarios"]


def sample_placement(kind, sid, k, users, rng):
    reg = scenarios.get_scenario(sid)["uav_region"]
    fixed = scenarios.initial_uav_positions(k, sid)
    z = fixed[0, 2]
    lo = np.array([reg["x"][0], reg["y"][0]]); hi = np.array([reg["x"][1], reg["y"][1]])
    if kind == "fixed":
        xy = fixed[:, :2]
    elif kind == "random":
        xy = rng.uniform(lo, hi, size=(k, 2))
    elif kind == "jitter":
        xy = fixed[:, :2] + rng.normal(0, 30.0, size=(k, 2))
    elif kind == "user_kmeans":      # roughly user-centred, plus noise
        u = np.asarray(users)[:, :2]
        c = u[rng.choice(len(u), k, replace=False)]
        for _ in range(10):
            lab = np.argmin(((u[:, None] - c[None]) ** 2).sum(-1), axis=1)
            c = np.array([u[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        xy = c + rng.normal(0, 15.0, size=(k, 2))
    else:                            # partly stacked
        xy = rng.uniform(lo, hi, size=(k, 2))
        xy[1] = xy[0] + rng.normal(0, 8.0, size=2)
    xy = np.clip(xy, lo, hi)
    return np.column_stack([xy, np.full(k, z)]).round(2).tolist()


KINDS = ["fixed", "random", "jitter", "user_kmeans", "stacked"]


def make_cases(scen_list, n_runs, seed0, rng):
    cases = []
    for r in range(n_runs):
        sid = scen_list[r % len(scen_list)]
        n = int(rng.choice([10, 20, 30, 50]))
        seed = seed0 + r
        users = scenarios.generate_users(sid, n, seed).tolist()
        kind = KINDS[(r // len(scen_list)) % len(KINDS)]
        uavs = sample_placement(kind, sid, 3, users, rng)
        rf = ev.rf_eval(uavs, users, "surrogate", sid, seed)
        assoc = [u["serving_uav"] - 1 for u in rf["user_results"]]
        cases.append({"scenario": sid, "seed": seed, "kind": kind, "uavs": uavs, "users": users, "rf": rf,
                      "rng_run": int(rng.integers(1, 6)),
                      "handoff": mio.build_uav_positions(uavs, users, associations=assoc)})
    return cases


def simulate(binary, cases, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda c: ev.run_ns3(binary, c["handoff"], (f"--RngRun={c['rng_run']}",)), cases))


def to_rows(cases, results):
    X, y, usable, run_id, pred_old = [], [], [], [], []
    for r, (c, n) in enumerate(zip(cases, results)):
        Xi, ui, net = cm.user_features(c["rf"], c["uavs"], c["users"])
        pdr = np.array(n["per_user_rx"]) / np.maximum(np.array(n["per_user_sent"]), 1)
        X.append(Xi); y.append(pdr); usable.append(ui); run_id += [r] * len(pdr)
        pred_old.append(net["aggregate"]["predicted_pdr"])
    return np.vstack(X), np.concatenate(y), np.concatenate(usable), np.array(run_id), np.array(pred_old)


def placement_metrics(y, pred_user, run_id, pred_old, usable=None):
    runs = np.unique(run_id)
    true = np.array([y[run_id == r].mean() for r in runs])
    new = np.array([pred_user[run_id == r].mean() for r in runs])
    from scipy import stats
    if usable is not None and pred_old is not None:
        # the hard regime: every user has a usable link, so placements differ
        # only through contention / hidden nodes
        full = np.array([usable[run_id == r].all() for r in runs])
        if full.sum() >= 5:
            sub = {"n_placements": int(full.sum()),
                   "calibrated_spearman": float(stats.spearmanr(new[full], true[full]).statistic),
                   "calibrated_mae": float(np.abs(new[full] - true[full]).mean()),
                   "old_airtime_spearman": float(stats.spearmanr(pred_old[full], true[full]).statistic),
                   "old_airtime_mae": float(np.abs(pred_old[full] - true[full]).mean())}
        else:
            sub = {"n_placements": int(full.sum())}
    out = {"n_placements": int(len(runs)), "true_mean_pdr": float(true.mean()),
           "calibrated": {"mae": float(np.abs(new - true).mean()),
                          "spearman": float(stats.spearmanr(new, true).statistic),
                          "pearson": float(stats.pearsonr(new, true).statistic)},
           "per_user_mae_calibrated": float(np.abs(pred_user - y).mean())}
    if pred_old is not None:
        out["old_airtime_model"] = {"mae": float(np.abs(pred_old - true).mean()),
                                    "spearman": float(stats.spearmanr(pred_old, true).statistic),
                                    "pearson": float(stats.pearsonr(pred_old, true).statistic)}
        if usable is not None:
            out["all_users_usable_subset"] = sub
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-runs", type=int, default=280)
    ap.add_argument("--val-runs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--model-dir", default=str(_ROOT / "models" / "network_surrogate_v1"))
    ap.add_argument("--data-dir", default=str(_ROOT / "data" / "network_surrogate_v1"))
    args = ap.parse_args()
    model_dir, data_dir = Path(args.model_dir), Path(args.data_dir)
    for d in (model_dir, data_dir):
        if d.exists():
            raise SystemExit(f"{d} already exists - refusing to overwrite")

    train_scen, eval_scen = register_all()
    rng = np.random.default_rng(2024)
    binary = ev.ns3_binary(Path.home() / "ns-allinone-3.47" / "ns-3.47")
    t0 = time.time()
    tr_cases = make_cases(train_scen, args.train_runs, 5000, rng)
    va_cases = make_cases(eval_scen, args.val_runs, 900, rng)
    print(f"simulating {len(tr_cases)} training + {len(va_cases)} validation placements ...", flush=True)
    tr_res = simulate(binary, tr_cases, args.workers)
    va_res = simulate(binary, va_cases, args.workers)
    print(f"ns-3 done in {time.time() - t0:.0f} s", flush=True)

    Xtr, ytr, utr, rtr, _ = to_rows(tr_cases, tr_res)
    Xva, yva, uva, rva, oldva = to_rows(va_cases, va_res)

    from sklearn.ensemble import HistGradientBoostingRegressor
    # in-distribution check: hold out 20 % of training placements (grouped by run)
    hold = np.isin(rtr, rng.choice(np.unique(rtr), len(np.unique(rtr)) // 5, replace=False))

    def fit(mask):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          monotonic_cst=cm.MONOTONE, random_state=0)
        m.fit(Xtr[mask & utr], ytr[mask & utr])
        return m

    def predict(m, X, u):
        p = np.zeros(len(X)); p[u] = np.clip(m.predict(X[u]), 0, 1); return p

    m_cv = fit(~hold)
    in_dist = placement_metrics(ytr[hold], predict(m_cv, Xtr[hold], utr[hold]), rtr[hold], None)
    model = fit(np.ones(len(Xtr), bool))
    held_out = placement_metrics(yva, predict(model, Xva, uva), rva, oldva, uva)
    unusable_check = {"unusable_users": int((~utr).sum()),
                      "unusable_users_with_delivery": int(((~utr) & (ytr > 0)).sum())}

    report = {"features": cm.FEATURES, "monotone": cm.MONOTONE,
              "train_scenarios": train_scen, "validation_scenarios": eval_scen,
              "train_placements": len(tr_cases), "train_user_rows": int(len(ytr)),
              "in_distribution_holdout": in_dist, "held_out_scenarios": held_out,
              "unusable_rule_check": unusable_check, "wall_time_sec": round(time.time() - t0, 1)}
    import joblib
    model_dir.mkdir(parents=True); data_dir.mkdir(parents=True)
    joblib.dump({"model": model, "features": cm.FEATURES, "meta": report}, model_dir / "model.joblib")
    (model_dir / "report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(data_dir / "dataset.npz", Xtr=Xtr, ytr=ytr, utr=utr, rtr=rtr,
                        Xva=Xva, yva=yva, uva=uva, rva=rva)
    (data_dir / "placements.json").write_text(json.dumps(
        [{k: c[k] for k in ("scenario", "seed", "kind", "uavs", "rng_run")} | {"split": s}
         for s, cs in (("train", tr_cases), ("validation", va_cases)) for c in cs]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
