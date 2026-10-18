"""Per-user ns-3 latency predictor (candidate PPO observation / reward input).

Features: the calibrated PDR model's contention / hidden-node features
(calibrated_model.user_features, evaluated with the obstruction excess at the
operating frequency) plus the serving UAV's user count and the link rate.
Target: log of the user's mean packet delay in ns-3 (delivering users only).

It is used in training ONLY if ml/fit_latency_model.py reports that it passed
the pre-registered validation criteria on an unseen window (Spearman >= 0.6
for per-UAV mean latency, >= 70 % correct ordering for one-move pairs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import calibrated_model as cm
import network_model as nm
import multiuav3d_env as E
from closed_loop_control import compose_rf

EXTRA = ["serving_uav_users", "mcs_rate_mbps"]
FEATURES = cm.FEATURES + EXTRA


def user_features(P, U, pg, freq_hz: float = E.TRAINING_FREQ_HZ):
    """(X [N, len(FEATURES)], usable [N], serving [N]) for one placement."""
    P, U, pg = np.asarray(P, float), np.asarray(U, float), np.asarray(pg, float)
    d = np.linalg.norm(P[:, None, :] - U[None, :, :], axis=2)
    exc = E.excess_loss_db(pg, d, freq_hz)
    fspl_ref = 20.0 * np.log10(4.0 * np.pi * np.maximum(d, 1.0) * E.CALIBRATED_FREQ_HZ / nm.C_LIGHT)
    pg_ref = np.where(pg <= -199.0, -200.0, -(fspl_ref + exc))
    serving = np.argmax(pg, axis=0)
    rf = compose_rf(P.tolist(), U.tolist(), pg_ref, serving.tolist())
    X, usable, net = cm.user_features(rf, P, U)
    cnt = np.bincount(serving, minlength=len(P))[serving]
    rate = np.array([p["mcs_rate_mbps"] for p in net["per_user"]])
    return np.column_stack([X, cnt, rate]), usable, serving


class LatencyModel:
    def __init__(self, path: str | Path):
        import joblib
        blob = joblib.load(path)
        if blob["features"] != FEATURES:
            raise ValueError("latency model trained with a different feature set")
        self.model, self.meta = blob["model"], blob.get("meta", {})

    def predict_users(self, P, U, pg) -> np.ndarray:
        """Predicted mean delay [ms] per user; NaN for users the link model says are unusable."""
        X, usable, _ = user_features(P, U, pg)
        out = np.full(len(X), np.nan)
        if usable.any():
            out[usable] = np.exp(self.model.predict(X[usable]))
        return out
