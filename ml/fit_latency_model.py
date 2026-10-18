"""Stage 5: fit and validate the latency predictor, and check the calibrated
PDR model under obstruction, on the stage-4 ns-3 runs (ml/gen_ns3_3d_data.py).

Split: fit on the training windows (train_*), test on the validation window
(val_a, never seen in fitting). urban_core data are never read.

Pre-registered criteria (fixed before looking at the data):
  latency  : Spearman rho >= 0.6 between predicted and measured per-UAV mean
             latency across placements on val_a, AND >= 70 % correct sign of
             the latency change for one-move placement pairs on val_a.
  PDR (v1) : models/network_surrogate_v1 on val_a placements: mean-PDR
             MAE <= 0.05 AND Spearman rho >= 0.6. If v1 fails, a v2 with the
             same features / monotone constraints is fitted on the training
             windows and used only if it passes the same test on val_a.
If the latency model fails, it is saved for the record but marked
"use_in_training": false, and latency stays out of the PPO observation / reward.

Usage: python ml/fit_latency_model.py --data ~/.cache/resqnet_3d/ns3_data --out models/latency_surrogate_v1/<run>
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
for p in (ROOT / "ml", ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import calibrated_model as cm  # noqa: E402
import latency_model as lm  # noqa: E402
import multiuav3d_env as E  # noqa: E402

LAT_RHO, LAT_PAIR = 0.6, 0.70
PDR_MAE, PDR_RHO = 0.05, 0.6


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def load(data: Path) -> list[dict]:
    recs = []
    for w in sorted(p for p in data.iterdir() if p.is_dir()):
        if "urban_core" in w.name:
            raise SystemExit("urban_core data must not be used")
        for f in sorted(w.glob("*.json")):
            r = json.loads(f.read_text())
            n = r["ns3"]
            X, usable, serving = lm.user_features(r["positions"], r["users"], r["path_gain_db"], r["freq_hz"])
            r.update({"X": X, "usable": usable, "serving": serving,
                      "pdr_u": np.array(n["per_user_rx"]) / np.maximum(np.array(n["per_user_sent"]), 1),
                      "delay_u": np.array([np.nan if v is None else v for v in n["per_user_delay_ms"]], float)})
            recs.append(r)
    return recs


def placement_latency(delay, users) -> float:
    v = delay[users]
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else np.nan


def lat_eval(model, recs) -> dict:
    items_m, items_p, pairs = [], [], {}
    base_cnt_m, base_cnt_p = [], []
    for r in recs:
        pred = np.full(len(r["X"]), np.nan)
        if r["usable"].any():
            pred[r["usable"]] = np.exp(model.predict(r["X"][r["usable"]]))
        r["lat_pred"] = pred
        for k in range(len(r["positions"])):
            mine = r["serving"] == k
            m, p = placement_latency(r["delay_u"], mine), placement_latency(pred, mine)
            if not (np.isnan(m) or np.isnan(p)):
                items_m.append(m); items_p.append(p)
                base_cnt_m.append(m); base_cnt_p.append(int(mine.sum()))
        allu = np.ones(len(r["X"]), bool)
        pairs.setdefault((r["window"], r["layout"]), {})[r["role"]] = (
            placement_latency(r["delay_u"], allu), placement_latency(pred, allu))
    ok = tot = 0
    for v in pairs.values():
        if "base" in v and "nbr" in v and not any(np.isnan(x) for x in (*v["base"], *v["nbr"])):
            dm, dp = v["nbr"][0] - v["base"][0], v["nbr"][1] - v["base"][1]
            tot += 1
            ok += int(np.sign(dm) == np.sign(dp))
    return {"per_uav_items": len(items_m), "per_uav_spearman": spearman(items_m, items_p) if len(items_m) > 2 else None,
            "baseline_user_count_spearman": spearman(base_cnt_m, base_cnt_p) if len(items_m) > 2 else None,
            "pairs": tot, "pair_sign_accuracy": ok / tot if tot else None,
            "median_abs_log_err": float(np.nanmedian(np.abs(np.log(np.concatenate(
                [r["lat_pred"] for r in recs])) - np.log(np.concatenate([r["delay_u"] for r in recs])))))}


def pdr_eval(predict_users, recs) -> dict:
    m, p, pairs = [], [], {}
    for r in recs:
        pu = predict_users(r)
        m.append(float(r["pdr_u"].mean())); p.append(float(pu.mean()))
        pairs.setdefault((r["window"], r["layout"]), {})[r["role"]] = (m[-1], p[-1])
    ok = tot = 0
    for v in pairs.values():
        if "base" in v and "nbr" in v:
            dm, dp = v["nbr"][0] - v["base"][0], v["nbr"][1] - v["base"][1]
            if abs(dm) > 1e-9:
                tot += 1
                ok += int(np.sign(dm) == np.sign(dp))
    m, p = np.array(m), np.array(p)
    return {"placements": len(m), "mae": float(np.abs(m - p).mean()), "bias": float((p - m).mean()),
            "spearman": spearman(m, p) if len(m) > 2 else None, "measured_mean": float(m.mean()),
            "measured_std": float(m.std()), "pair_sign_accuracy": ok / tot if tot else None, "pairs": tot}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/.cache/resqnet_3d/ns3_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-window", default="val_a")
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    recs = load(Path(os.path.expanduser(a.data)))
    train = [r for r in recs if r["window"].startswith("train_")]
    val = [r for r in recs if r["window"] == a.val_window]
    if not train or not val:
        raise SystemExit("need training-window and validation-window runs")
    report = {"data": a.data, "n_train_runs": len(train), "n_val_runs": len(val),
              "train_windows": sorted({r["window"] for r in train}), "val_window": a.val_window,
              "criteria": {"latency": {"per_uav_spearman_min": LAT_RHO, "pair_sign_accuracy_min": LAT_PAIR},
                           "pdr_v1": {"mae_max": PDR_MAE, "spearman_min": PDR_RHO}},
              "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------- latency ----------------
    def lat_xy(rs):
        X, y = [], []
        for r in rs:
            keep = r["usable"] & ~np.isnan(r["delay_u"]) & (r["delay_u"] > 0)
            X.append(r["X"][keep]); y.append(np.log(r["delay_u"][keep]))
        return np.concatenate(X), np.concatenate(y)

    def lat_fit(rs):
        X, y = lat_xy(rs)
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=30, l2_regularization=1.0,
                                             random_state=0).fit(X, y), len(y)

    # leave-one-training-window-out (generalisation within the training data)
    lowo = {}
    for w in report["train_windows"]:
        mdl, _ = lat_fit([r for r in train if r["window"] != w])
        lowo[w] = lat_eval(mdl, [r for r in train if r["window"] == w])
    lat_model, n_users = lat_fit(train)
    lat_val = lat_eval(lat_model, val)
    lat_pass = bool(lat_val["per_uav_spearman"] is not None and lat_val["per_uav_spearman"] >= LAT_RHO
                    and lat_val["pair_sign_accuracy"] is not None and lat_val["pair_sign_accuracy"] >= LAT_PAIR)
    report["latency"] = {"train_user_samples": n_users, "leave_one_window_out": lowo, "validation": lat_val,
                         "pass": lat_pass, "use_in_training": lat_pass}

    # ---------------- PDR model under obstruction ----------------
    v1 = cm.CalibratedNetworkModel()

    def v1_users(r):
        pu = np.zeros(len(r["X"]))
        if r["usable"].any():
            pu[r["usable"]] = np.clip(v1.model.predict(r["X"][r["usable"]][:, :len(cm.FEATURES)]), 0, 1)
        return pu

    pdr = {"v1_all_windows": {w: pdr_eval(v1_users, [r for r in recs if r["window"] == w])
                              for w in sorted({r["window"] for r in recs})}}
    pv = pdr["v1_all_windows"][a.val_window]
    pdr["v1_pass"] = bool(pv["mae"] <= PDR_MAE and pv["spearman"] is not None and pv["spearman"] >= PDR_RHO)
    if not pdr["v1_pass"]:
        X = np.concatenate([r["X"][r["usable"]][:, :len(cm.FEATURES)] for r in train])
        y = np.concatenate([r["pdr_u"][r["usable"]] for r in train])
        v2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           monotonic_cst=cm.MONOTONE, random_state=0).fit(X, y)

        def v2_users(r):
            pu = np.zeros(len(r["X"]))
            if r["usable"].any():
                pu[r["usable"]] = np.clip(v2.predict(r["X"][r["usable"]][:, :len(cm.FEATURES)]), 0, 1)
            return pu
        pdr["v2_validation"] = pdr_eval(v2_users, val)
        q = pdr["v2_validation"]
        pdr["v2_pass"] = bool(q["mae"] <= PDR_MAE and q["spearman"] is not None and q["spearman"] >= PDR_RHO)
        pdr["v2_train_samples"] = int(len(y))
    report["pdr"] = pdr

    out.mkdir(parents=True)
    joblib.dump({"model": lat_model, "features": lm.FEATURES, "meta": report}, out / "model.joblib")
    if not pdr["v1_pass"]:
        joblib.dump({"model": v2, "features": cm.FEATURES,
                     "meta": {"fitted_on": report["train_windows"], "validation": pdr["v2_validation"],
                              "pass": pdr["v2_pass"], "obstruction_aware": True}}, out / "pdr_v2.joblib")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({"latency": {"validation": lat_val, "pass": lat_pass},
                      "pdr_v1_val": pv, "pdr_v1_pass": pdr["v1_pass"],
                      "pdr_v2_val": pdr.get("v2_validation"), "pdr_v2_pass": pdr.get("v2_pass")}, default=float))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
