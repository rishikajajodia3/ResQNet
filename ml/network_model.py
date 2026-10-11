"""Network-consistent link and throughput model for the multi-UAV reward.

The RF layer (Sionna RT, or the free-space surrogate) and the network layer
(ns3/uav-demo.cc) disagree about received power: Sionna works at 2.1 GHz
with free-space-like decay over this near-open terrain, while ns-3 uses its
default LogDistancePropagationLossModel (exponent 3, 46.6777 dB at 1 m,
i.e. Friis at 5.15 GHz). At 150 m the two differ by ~30 dB, so a user can be
"connected" in Sionna and unreachable in ns-3.

This module predicts what the network layer will see:

* received power = ns-3 log-distance law + any obstruction loss the RF model
  reports in excess of free space (0 for the surrogate, building shadowing
  for Sionna). For the surrogate this reproduces ns-3's received power.
* a link is usable only if ns-3 can detect it: RSS >= -82 dBm and
  SNR >= 4 dB (ThresholdPreambleDetectionModel defaults; -82 dBm is also
  the IEEE 802.11n MCS0 minimum sensitivity).
* the link rate follows ns-3's IdealWifiManager MCS table (HT, 20 MHz,
  800 ns GI, 1 spatial stream; SNR thresholds at BER 1e-6, read from ns-3).
* all UAVs and users share ONE 802.11 channel (single IBSS), so delivery is
  limited by total airtime; under saturation DCF gives every station the
  same packet rate (the 802.11 "performance anomaly").

Co-channel full-buffer SINR is deliberately NOT used here: ns-3's CSMA/CA
network does not have simultaneous co-channel transmissions, so rewarding it
would let the policy optimise a quantity the network never experiences.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

C_LIGHT = 299_792_458.0

# --- ns-3 radio defaults used by ns3/uav-demo.cc (ns-3.47) ------------------
NS3_TX_POWER_DBM = 30.0            # uav-demo.cc TxPowerStart/End
NS3_REF_LOSS_DB = 46.6777          # LogDistancePropagationLossModel::ReferenceLoss
NS3_PATH_LOSS_EXPONENT = 3.0       # LogDistancePropagationLossModel::Exponent
NS3_NOISE_DBM = -174.0 + 10.0 * math.log10(20e6) + 7.0   # 20 MHz, RxNoiseFigure 7 dB
MIN_RSS_DBM = -82.0                # ThresholdPreambleDetectionModel::MinimumRssi
MIN_SNR_DB = 4.0                   # ThresholdPreambleDetectionModel::Threshold

# HtMcs0..7 (20 MHz, 800 ns GI, 1 SS): rate [Mbps], IdealWifiManager SNR threshold (linear)
HT_RATES_MBPS = np.array([6.5, 13.0, 19.5, 26.0, 39.0, 52.0, 58.5, 65.0])
HT_SNR_THRESH_DB = 10.0 * np.log10(
    np.array([1.4012, 2.80221, 4.99459, 10.9018, 22.2075, 61.1646, 81.7523, 119.536])
)
MAX_RATE_MBPS = float(HT_RATES_MBPS[-1])

# --- traffic offered by every user in uav-demo.cc --------------------------
PACKETS_PER_SEC = 100.0            # UdpClient Interval 10 ms
IP_PACKET_BYTES = 512 + 28         # payload + IPv4/UDP headers
MAC_FRAME_BYTES = IP_PACKET_BYTES + 8 + 26 + 4   # LLC/SNAP + QoS MAC header + FCS

# --- 802.11n @ 2.4 GHz (ns-3 default band for WIFI_STANDARD_80211n) timing --
SLOT_US, SIFS_US = 9.0, 10.0
AIFS_US = SIFS_US + 3 * SLOT_US    # AC_BE, AIFSN 3
MEAN_BACKOFF_US = 7.5 * SLOT_US    # CWmin 15
HT_PREAMBLE_US = 36.0              # HT-mixed: L-STF/LTF/SIG + HT-SIG/STF/LTF
SIGNAL_EXT_US = 6.0                # 2.4 GHz OFDM signal extension
ACK_US = 20.0 + 8.0 + SIGNAL_EXT_US  # legacy preamble + 14 B ACK @ 24 Mbps
MAX_UTILISATION = 1.0              # contention overhead is in MEAN_BACKOFF_US


def ns3_path_loss_db(distance_m: float) -> float:
    return NS3_REF_LOSS_DB + 10.0 * NS3_PATH_LOSS_EXPONENT * math.log10(max(distance_m, 1.0))


def free_space_loss_db(distance_m: float, freq_hz: float) -> float:
    d = max(distance_m, 1.0)
    return 20.0 * math.log10(4.0 * math.pi * d * freq_hz / C_LIGHT)


def network_rss_dbm(rf_path_gain_db: float, distance_m: float, freq_hz: float) -> float:
    """ns-3-equivalent received power, keeping the RF model's obstruction loss."""
    if rf_path_gain_db <= -199.0:          # RF model found no path at all
        return -200.0
    excess = max(0.0, -rf_path_gain_db - free_space_loss_db(distance_m, freq_hz))
    return NS3_TX_POWER_DBM - ns3_path_loss_db(distance_m) - excess


def mcs_rate_mbps(snr_db: float) -> float:
    """Rate IdealWifiManager would pick (0 if below HtMcs0)."""
    ok = HT_SNR_THRESH_DB < snr_db
    return float(HT_RATES_MBPS[ok][-1]) if ok.any() else 0.0


def smooth_rate_mbps(snr_db: float) -> float:
    """Continuous, monotone version of mcs_rate_mbps: linear in dB between the
    ns-3 MCS thresholds, 0 below HtMcs0, capped at HtMcs7. Equal to the
    staircase at every threshold, so it never promises more than the top MCS."""
    if snr_db <= HT_SNR_THRESH_DB[0]:
        return 0.0
    return float(np.interp(snr_db, HT_SNR_THRESH_DB, HT_RATES_MBPS))


def airtime_us(rate_mbps: float) -> float:
    """Channel time for one data frame + ACK exchange at the given HT rate."""
    bits = 16 + 8 * MAC_FRAME_BYTES + 6
    n_dbps = rate_mbps * 4.0                     # data bits per 4 us symbol
    data_us = 4.0 * math.ceil(bits / n_dbps)
    return AIFS_US + MEAN_BACKOFF_US + HT_PREAMBLE_US + data_us + SIGNAL_EXT_US + SIFS_US + ACK_US


def evaluate_network(
    rf_result: dict,
    uav_positions: Sequence[Sequence[float]],
    user_positions: Sequence[Sequence[float]],
    freq_hz: float = 2.1e9,
) -> dict:
    """Network-layer view of one RF evaluation (Sionna or surrogate result dict).

    Each user keeps the serving UAV chosen by the RF layer, because that is
    the association handed to ns-3.
    """
    uavs = np.asarray(uav_positions, dtype=float)
    users = np.asarray(user_positions, dtype=float)
    n = len(users)
    per_user = []
    airtime_total = 0.0
    for i, u in enumerate(rf_result["user_results"]):
        k = int(u["serving_uav"]) - 1
        d = float(np.linalg.norm(users[i] - uavs[k]))
        rss = network_rss_dbm(float(u["per_uav"][k]["path_gain_db"]), d, freq_hz)
        snr = rss - NS3_NOISE_DBM
        usable = bool(rss >= MIN_RSS_DBM and snr >= MIN_SNR_DB)
        rate = mcs_rate_mbps(snr) if usable else 0.0
        if usable and rate > 0:
            airtime_total += PACKETS_PER_SEC * airtime_us(rate) * 1e-6
        per_user.append({
            "serving_uav": k + 1,
            "distance_m": round(d, 2),
            "net_rss_dbm": round(rss, 2),
            "net_snr_db": round(snr, 2),
            "usable": usable and rate > 0,
            "mcs_rate_mbps": rate,
            "smooth_rate_mbps": round(smooth_rate_mbps(snr), 3) if usable else 0.0,
        })

    connected = sum(1 for p in per_user if p["usable"])
    served = min(1.0, MAX_UTILISATION / airtime_total) if airtime_total > 0 else 1.0
    offered_mbps = PACKETS_PER_SEC * IP_PACKET_BYTES * 8 / 1e6
    return {
        "per_user": per_user,
        "aggregate": {
            "net_connected_count": connected,
            "net_connected_fraction": connected / n if n else 0.0,
            "airtime_utilisation": airtime_total,
            "served_fraction": served,
            "predicted_pdr": served * connected / n if n else 0.0,
            "predicted_throughput_per_user_mbps": offered_mbps * served * connected / n if n else 0.0,
            "mean_rate_norm": float(np.mean([p["smooth_rate_mbps"] for p in per_user])) / MAX_RATE_MBPS
            if n else 0.0,
        },
    }
