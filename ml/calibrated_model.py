"""ns-3-calibrated per-user packet-delivery model (hidden nodes + contention).

Why: ns-3 measurements (see ml/fit_calibrated_model.py) show that on large
areas most packet loss is NOT link failure. Users that cannot hear each
other (< -82 dBm) but reach the same UAV collide at the UAV receiver
(hidden nodes); the failed attempts trigger retransmissions and backoff,
queues build up and 88 % of lost packets expire in the MAC queue (500 ms
lifetime). The analytic airtime model in network_model.py ignores this, so
it predicted PDR 0.93 where ns-3 measured 0.75.

This module computes physically meaningful per-user features from the
geometry using ns-3's own propagation law (so "who hears whom" matches the
simulator exactly) and maps them to per-user PDR with a monotone-constrained
gradient-boosting model fitted to ns-3 runs on TRAINING scenarios only.
Monotone constraints (PDR never increases with hidden / contending load,
never decreases with link SNR) keep the model from rewarding non-physical
placements when it is used as a training reward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

import network_model as nm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "network_surrogate_v1" / "model.joblib"

FEATURES = [
    "snr_db",              # link SNR to the serving UAV (ns-3 law + RF obstruction excess)
    "own_airtime",         # fraction of time the user's own frames occupy the channel
    "hidden_load",         # airtime of users the UAV hears but this user cannot sense
    "hidden_unusable",     # count of hidden users that can never be ACKed (retry storms)
    "sensed_load",         # airtime of users this user defers to (carrier sense)
    "uav_load",            # airtime of all users the serving UAV hears
    "num_users",
]
# +1: PDR non-decreasing in the feature, -1: non-increasing, 0: unconstrained
MONOTONE = [1, -1, -1, -1, -1, -1, 0]

UNUSABLE_AIRTIME = 7 * nm.airtime_us(6.5) * 1e-6 * nm.PACKETS_PER_SEC  # retries at the lowest rate


def _ns3_rss(dist: np.ndarray) -> np.ndarray:
    d = np.maximum(dist, 1.0)
    return nm.NS3_TX_POWER_DBM - (nm.NS3_REF_LOSS_DB + 10 * nm.NS3_PATH_LOSS_EXPONENT * np.log10(d))


def user_features(
    rf_result: dict,
    uav_positions: Sequence[Sequence[float]],
    user_positions: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X [N, len(FEATURES)], usable mask [N], network_model result)."""
    net = nm.evaluate_network(rf_result, uav_positions, user_positions)
    uavs = np.asarray(uav_positions, dtype=float)
    users = np.asarray(user_positions, dtype=float)
    n = len(users)
    usable = np.array([p["usable"] for p in net["per_user"]], dtype=bool)
    serving = np.array([p["serving_uav"] - 1 for p in net["per_user"]])
    snr = np.array([p["net_snr_db"] for p in net["per_user"]], dtype=float)
    air = np.array([
        nm.PACKETS_PER_SEC * nm.airtime_us(p["mcs_rate_mbps"]) * 1e-6 if p["usable"] else UNUSABLE_AIRTIME
        for p in net["per_user"]
    ])

    uu = np.linalg.norm(users[:, None, :] - users[None, :, :], axis=-1)
    sensed = _ns3_rss(uu) >= nm.MIN_RSS_DBM                     # [i, j]: i senses j
    np.fill_diagonal(sensed, False)
    uk = np.linalg.norm(users[:, None, :] - uavs[None, :, :], axis=-1)
    heard_by_uav = _ns3_rss(uk) >= nm.MIN_RSS_DBM               # [j, k]: UAV k detects j

    X = np.zeros((n, len(FEATURES)))
    for i in range(n):
        k = serving[i]
        others = np.arange(n) != i
        at_uav = heard_by_uav[:, k] & others
        hidden = at_uav & ~sensed[i]
        X[i] = [
            snr[i],
            air[i],
            air[hidden].sum(),
            float((hidden & ~usable).sum()),
            air[sensed[i]].sum(),
            air[at_uav].sum(),
            n,
        ]
    return X, usable, net


class CalibratedNetworkModel:
    """Per-user PDR predictor. Unusable links are 0 by construction (the link
    model agrees with ns-3 on usability for ~99.9 % of users)."""

    def __init__(self, path: str | Path = DEFAULT_MODEL):
        import joblib
        blob = joblib.load(path)
        if blob["features"] != FEATURES:
            raise ValueError("model was trained with a different feature set")
        self.model = blob["model"]
        self.meta = blob.get("meta", {})

    def predict_users(self, rf_result, uav_positions, user_positions) -> tuple[np.ndarray, dict]:
        X, usable, net = user_features(rf_result, uav_positions, user_positions)
        pdr = np.zeros(len(X))
        if usable.any():
            pdr[usable] = np.clip(self.model.predict(X[usable]), 0.0, 1.0)
        return pdr, net

    def evaluate(self, rf_result, uav_positions, user_positions) -> dict:
        pdr, net = self.predict_users(rf_result, uav_positions, user_positions)
        return {
            "per_user_pdr": pdr,
            "aggregate": {
                "calibrated_pdr": float(pdr.mean()),
                "users_pdr_ge_90": int((pdr >= 0.9).sum()),
                "net_connected_fraction": net["aggregate"]["net_connected_fraction"],
                "worst_decile_pdr": float(np.sort(pdr)[: max(1, len(pdr) // 10)].mean()),
            },
            "network": net,
        }
