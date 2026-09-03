"""
ResQNet - Sionna RT Wireless Feedback Engine
Phase 1: 1 UAV + 10 Ground Users Wireless Propagation Evaluator

This module turns Sionna RT into a live wireless feedback engine for PPO
and network validation without duplicating ray-tracing pipelines.
"""

import os
import json
import csv
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import sionna
from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray, PathSolver
)


@dataclass
class ConnectivityConfig:
    """
    Configurable engineering thresholds for connectivity decisions.
    
    Engineering Assumptions:
    - rss_threshold_dbm: -95.0 dBm (Standard reference sensitivity for edge emergency reception in LTE/5G).
    - sinr_threshold_db: -3.0 dB (Minimum QPSK 1/3 decoding threshold).
    - path_gain_threshold_db: -100.0 dB (Threshold for spatial coverage detection).
    - noise_power_dbm: -94.0 dBm (Standard thermal noise floor for 20 MHz bandwidth with 7 dB receiver noise figure:
                       -174 dBm/Hz + 10*log10(20 MHz) + 7 dB = -94 dBm).
    - enable_sinr: True if noise floor is configured to compute SNR (SINR with I=0 for single UAV). Set False if SINR should not be computed.
    """
    rss_threshold_dbm: float = -95.0
    sinr_threshold_db: float = -3.0
    path_gain_threshold_db: float = -100.0
    noise_power_dbm: Optional[float] = -94.0
    enable_sinr: bool = True


# Default 10 ground survivor coordinates across the damaged city center
DEFAULT_10_USERS = [
    [-20.0, -30.0, 1.5],
    [-10.0, -20.0, 1.5],
    [0.0, -10.0, 1.5],
    [10.0, 0.0, 1.5],
    [20.0, 10.0, 1.5],
    [-15.0, 25.0, 1.5],
    [5.0, 35.0, 1.5],
    [25.0, -15.0, 1.5],
    [-30.0, -10.0, 1.5],
    [0.0, 20.0, 1.5]
]


def resolve_scene_path(scene_name: Optional[str] = None) -> str:
    """
    Resolves the scene file path, prioritizing city_damaged.xml.
    """
    if scene_name is None or scene_name == "city_damaged.xml":
        candidates = [
            os.path.join(os.path.dirname(__file__), "scenes", "city_damaged.xml"),
            os.path.join("sionna", "scenes", "city_damaged.xml"),
            os.path.join("scenes", "city_damaged.xml"),
            "city_damaged.xml"
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return "city_damaged.xml"
    elif scene_name == "city_baseline.xml":
        candidates = [
            os.path.join(os.path.dirname(__file__), "scenes", "city_baseline.xml"),
            os.path.join("sionna", "scenes", "city_baseline.xml"),
            os.path.join("scenes", "city_baseline.xml"),
            "city_baseline.xml"
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return "city_baseline.xml"
    
    if os.path.isabs(scene_name) and os.path.exists(scene_name):
        return scene_name
    
    local_path = os.path.join(os.path.dirname(__file__), "scenes", scene_name)
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
        
    return scene_name


def safe_load_scene_robust(scene_path: Optional[str] = None) -> Any:
    """
    Loads scene from path with material validation and fallback.
    """
    resolved = resolve_scene_path(scene_path)
    try:
        scene = load_scene(resolved)
        for name, obj in scene.objects.items():
            if obj.radio_material is None:
                obj.radio_material = "itu_concrete"
        return scene
    except Exception as e:
        print(f"[WARN] Failed loading '{resolved}': {e}. Falling back to default street canyon.")
        return load_scene(sionna.rt.scene.simple_street_canyon)


def evaluate_connectivity(
    rss_dbm: float,
    sinr_db: Optional[float] = None,
    path_gain_db: Optional[float] = None,
    config: Optional[ConnectivityConfig] = None
) -> bool:
    """
    Phase 2: Evaluates physical-layer results to decide connected status (True/False).
    
    Decoupled from ray tracing.
    """
    if config is None:
        config = ConnectivityConfig()
        
    if rss_dbm <= -199.0:
        return False
        
    # Check RSS threshold
    rss_ok = rss_dbm >= config.rss_threshold_dbm
    
    # Check SINR threshold if SINR is available
    if sinr_db is not None:
        sinr_ok = sinr_db >= config.sinr_threshold_db
        return bool(rss_ok and sinr_ok)
        
    # Fallback to RSS and path gain check if SINR is unavailable
    if path_gain_db is not None:
        gain_ok = path_gain_db >= config.path_gain_threshold_db
        return bool(rss_ok and gain_ok)
        
    return bool(rss_ok)


class SionnaFeedbackEngine:
    """
    Stateful evaluation engine that holds loaded scene in memory for high-performance
    repeated evaluations during PPO stepping.
    """
    def __init__(
        self,
        scene_name: str = "city_damaged.xml",
        freq: float = 2.1e9,
        tx_power_dbm: float = 30.0,
        connectivity_config: Optional[ConnectivityConfig] = None,
        seed: int = 42
    ):
        self.scene_name = scene_name
        self.freq = freq
        self.tx_power_dbm = tx_power_dbm
        self.config = connectivity_config or ConnectivityConfig()
        self.seed = seed
        
        self.scene = safe_load_scene_robust(self.scene_name)
        self.scene.frequency = self.freq
        
        # Configure standard PlanarArray antennas
        self.scene.tx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )
        self.scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )
        self.solver = PathSolver()
        self._last_uav_pos = None
        self._last_user_positions = None

    def evaluate(
        self,
        uav_position: List[float],
        user_positions: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Runs ray tracing for 1 UAV and 10 users and returns structured per-user & aggregate metrics.
        """
        if user_positions is None:
            user_positions = DEFAULT_10_USERS

        num_users = len(user_positions)
        
        # Clear previous transmitters and receivers
        for name in list(self.scene.transmitters.keys()):
            self.scene.remove(name)
        for name in list(self.scene.receivers.keys()):
            self.scene.remove(name)

        # Add single UAV transmitter with standard Python floats
        uav_tx = Transmitter(name="uav_1", position=[float(x) for x in uav_position])
        self.scene.add(uav_tx)

        # Add all 10 ground receivers with standard Python floats
        for idx, pos in enumerate(user_positions):
            rx = Receiver(name=f"user_{idx+1}", position=[float(x) for x in pos])
            self.scene.add(rx)

        # Run ray tracing PathSolver with explicit deterministic seed
        paths = self.solver(self.scene, seed=self.seed)

        # Extract complex amplitudes: shape (num_rx, 1, 1, 1, num_paths)
        a_real, a_imag = paths.a
        a_complex = a_real.numpy() + 1j * a_imag.numpy()
        taus = paths.tau.numpy()  # shape (num_rx, 1, num_paths)

        tx_power_watts = 10 ** ((self.tx_power_dbm - 30.0) / 10.0)
        
        user_results = []
        path_gains_db_list = []
        rss_dbm_list = []
        sinr_db_list = []
        connected_count = 0
        covered_count = 0

        for u_idx, u_pos in enumerate(user_positions):
            # Squeeze to get 1D path array for this user
            user_a = a_complex[u_idx, 0, 0, 0, :]
            user_tau = taus[u_idx, 0, :]

            # Linear path gain: sum of squared path amplitudes
            path_gains_linear = np.abs(user_a) ** 2
            total_gain_linear = float(np.sum(path_gains_linear))

            # Path gain in dB
            if total_gain_linear > 1e-20:
                path_gain_db = round(float(10.0 * np.log10(total_gain_linear)), 2)
            else:
                path_gain_db = -200.0

            # Received signal strength (RSS in dBm)
            rx_power_linear = total_gain_linear * tx_power_watts
            if rx_power_linear > 1e-20:
                rss_dbm = round(float(10.0 * np.log10(rx_power_linear * 1000.0)), 2)
            else:
                rss_dbm = -200.0

            # SINR / SNR computation
            # Single UAV scenario: I = 0. If noise floor is configured, compute SNR = RSS - Noise
            if self.config.enable_sinr and self.config.noise_power_dbm is not None:
                if rss_dbm > -199.0:
                    sinr_db = round(float(rss_dbm - self.config.noise_power_dbm), 2)
                else:
                    sinr_db = -50.0  # deep dead zone SNR floor
            else:
                sinr_db = None

            # Shortest propagation delay
            valid_delays = user_tau[user_tau > 0]
            min_delay = float(np.min(valid_delays)) if len(valid_delays) > 0 else None

            # Number of multipath rays reaching user
            num_paths = int(np.count_nonzero(path_gains_linear > 1e-20))

            # Coverage threshold check
            coverage = bool(path_gain_db >= self.config.path_gain_threshold_db)
            if coverage:
                covered_count += 1

            # Connectivity decision
            connected = evaluate_connectivity(
                rss_dbm=rss_dbm,
                sinr_db=sinr_db,
                path_gain_db=path_gain_db,
                config=self.config
            )
            if connected:
                connected_count += 1

            if path_gain_db > -199.0:
                path_gains_db_list.append(path_gain_db)
                rss_dbm_list.append(rss_dbm)
                if sinr_db is not None:
                    sinr_db_list.append(sinr_db)

            user_results.append({
                "user_id": f"user_{u_idx+1}",
                "position": [round(float(p), 2) for p in u_pos],
                "uav_position": [round(float(p), 2) for p in uav_position],
                "path_gain_linear": float(total_gain_linear),
                "path_gain_db": path_gain_db,
                "rss_dbm": rss_dbm,
                "sinr_db": sinr_db,
                "num_paths": num_paths,
                "shortest_delay_sec": min_delay,
                "coverage": coverage,
                "connected": connected
            })

        # Calculate unbiased aggregate metrics
        mean_path_gain_db = round(float(np.mean(path_gains_db_list)), 2) if path_gains_db_list else -200.0
        mean_rss_dbm = round(float(np.mean(rss_dbm_list)), 2) if rss_dbm_list else -200.0
        mean_sinr_db = round(float(np.mean(sinr_db_list)), 2) if sinr_db_list else (None if not self.config.enable_sinr else -50.0)
        
        connected_pct = round((connected_count / num_users) * 100.0, 2)
        coverage_pct = round((covered_count / num_users) * 100.0, 2)

        return {
            "uav_position": [round(float(p), 2) for p in uav_position],
            "scene_name": self.scene_name,
            "num_users": num_users,
            "user_results": user_results,
            "aggregate": {
                "connected_users_count": connected_count,
                "connected_users_percentage": connected_pct,
                "coverage_percentage": coverage_pct,
                "mean_path_gain_db": mean_path_gain_db,
                "mean_rss_dbm": mean_rss_dbm,
                "mean_sinr_db": mean_sinr_db
            }
        }


# Global singleton engine instance for fast reuse across steps
_GLOBAL_ENGINE = None


def evaluate_uav_position(
    uav_position: List[float],
    user_positions: Optional[List[List[float]]] = None,
    scene_name: str = "city_damaged.xml",
    freq: float = 2.1e9,
    tx_power_dbm: float = 30.0,
    connectivity_config: Optional[ConnectivityConfig] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Phase 1 primary interface:
    Evaluates 1 UAV + 10 users on Sionna RT and returns per-user & aggregate metrics.
    """
    global _GLOBAL_ENGINE
    if (_GLOBAL_ENGINE is None or 
        _GLOBAL_ENGINE.scene_name != scene_name or 
        _GLOBAL_ENGINE.freq != freq or 
        _GLOBAL_ENGINE.tx_power_dbm != tx_power_dbm or
        _GLOBAL_ENGINE.seed != seed):
        _GLOBAL_ENGINE = SionnaFeedbackEngine(
            scene_name=scene_name,
            freq=freq,
            tx_power_dbm=tx_power_dbm,
            connectivity_config=connectivity_config,
            seed=seed
        )
    elif connectivity_config is not None:
        _GLOBAL_ENGINE.config = connectivity_config

    return _GLOBAL_ENGINE.evaluate(uav_position, user_positions)


def export_evaluation_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Exports evaluation results to formatted JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def export_evaluation_to_csv(data: Dict[str, Any], filepath: str, scenario_id: int = 1) -> None:
    """
    Exports per-user evaluation results to flattened CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    uav_pos = data["uav_position"]
    rows = []
    for u in data["user_results"]:
        rows.append({
            "scenario_id": scenario_id,
            "uav_x": uav_pos[0],
            "uav_y": uav_pos[1],
            "uav_z": uav_pos[2],
            "user_id": u["user_id"],
            "user_x": u["position"][0],
            "user_y": u["position"][1],
            "user_z": u["position"][2],
            "path_gain_linear": u["path_gain_linear"],
            "path_gain_db": u["path_gain_db"],
            "rss_dbm": u["rss_dbm"],
            "sinr_db": u["sinr_db"] if u["sinr_db"] is not None else "",
            "num_paths": u["num_paths"],
            "shortest_delay_sec": u["shortest_delay_sec"] if u["shortest_delay_sec"] is not None else "",
            "coverage": u["coverage"],
            "connected": u["connected"]
        })
    if rows:
        fieldnames = list(rows[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def compare_normal_vs_damaged(
    uav_position: List[float],
    user_positions: Optional[List[List[float]]] = None,
    freq: float = 2.1e9,
    tx_power_dbm: float = 30.0,
    connectivity_config: Optional[ConnectivityConfig] = None
) -> Dict[str, Any]:
    """
    Phase 5 / Scene Comparison:
    Evaluates the exact same UAV position and 10 user positions in both
    city_baseline.xml (intact) and city_damaged.xml (damaged), returning
    side-by-side per-user and aggregate differences.
    """
    if user_positions is None:
        user_positions = DEFAULT_10_USERS

    # 1. Evaluate baseline (intact) environment
    baseline_res = evaluate_uav_position(
        uav_position=uav_position,
        user_positions=user_positions,
        scene_name="city_baseline.xml",
        freq=freq,
        tx_power_dbm=tx_power_dbm,
        connectivity_config=connectivity_config
    )

    # 2. Evaluate damaged environment
    damaged_res = evaluate_uav_position(
        uav_position=uav_position,
        user_positions=user_positions,
        scene_name="city_damaged.xml",
        freq=freq,
        tx_power_dbm=tx_power_dbm,
        connectivity_config=connectivity_config
    )

    # 3. Compute per-user deltas
    user_comparisons = []
    connectivity_changed_users = []
    propagation_changed_flags = []

    for b_u, d_u in zip(baseline_res["user_results"], damaged_res["user_results"]):
        delta_gain_db = round(d_u["path_gain_db"] - b_u["path_gain_db"], 2)
        delta_rss_dbm = round(d_u["rss_dbm"] - b_u["rss_dbm"], 2)
        
        delta_sinr_db = None
        if d_u["sinr_db"] is not None and b_u["sinr_db"] is not None:
            delta_sinr_db = round(d_u["sinr_db"] - b_u["sinr_db"], 2)

        delta_num_paths = d_u["num_paths"] - b_u["num_paths"]
        
        delta_delay = None
        if d_u["shortest_delay_sec"] is not None and b_u["shortest_delay_sec"] is not None:
            delta_delay = d_u["shortest_delay_sec"] - b_u["shortest_delay_sec"]

        conn_changed = (d_u["connected"] != b_u["connected"])
        cov_changed = (d_u["coverage"] != b_u["coverage"])

        if conn_changed:
            connectivity_changed_users.append(d_u["user_id"])

        # Check if actual propagation metric changed
        has_prop_change = abs(delta_gain_db) > 1e-3 or delta_num_paths != 0
        propagation_changed_flags.append(has_prop_change)

        user_comparisons.append({
            "user_id": d_u["user_id"],
            "position": d_u["position"],
            "baseline": {
                "path_gain_db": b_u["path_gain_db"],
                "rss_dbm": b_u["rss_dbm"],
                "sinr_db": b_u["sinr_db"],
                "num_paths": b_u["num_paths"],
                "shortest_delay_sec": b_u["shortest_delay_sec"],
                "coverage": b_u["coverage"],
                "connected": b_u["connected"]
            },
            "damaged": {
                "path_gain_db": d_u["path_gain_db"],
                "rss_dbm": d_u["rss_dbm"],
                "sinr_db": d_u["sinr_db"],
                "num_paths": d_u["num_paths"],
                "shortest_delay_sec": d_u["shortest_delay_sec"],
                "coverage": d_u["coverage"],
                "connected": d_u["connected"]
            },
            "deltas": {
                "delta_path_gain_db": delta_gain_db,
                "delta_rss_dbm": delta_rss_dbm,
                "delta_sinr_db": delta_sinr_db,
                "delta_num_paths": delta_num_paths,
                "delta_shortest_delay_sec": delta_delay,
                "connectivity_changed": conn_changed,
                "coverage_changed": cov_changed
            }
        })

    b_agg = baseline_res["aggregate"]
    d_agg = damaged_res["aggregate"]

    aggregate_deltas = {
        "delta_connected_users_count": d_agg["connected_users_count"] - b_agg["connected_users_count"],
        "delta_connected_users_percentage": round(d_agg["connected_users_percentage"] - b_agg["connected_users_percentage"], 2),
        "delta_coverage_percentage": round(d_agg["coverage_percentage"] - b_agg["coverage_percentage"], 2),
        "delta_mean_path_gain_db": round(d_agg["mean_path_gain_db"] - b_agg["mean_path_gain_db"], 2),
        "delta_mean_rss_dbm": round(d_agg["mean_rss_dbm"] - b_agg["mean_rss_dbm"], 2),
        "delta_mean_sinr_db": round(d_agg["mean_sinr_db"] - b_agg["mean_sinr_db"], 2) if (d_agg["mean_sinr_db"] is not None and b_agg["mean_sinr_db"] is not None) else None
    }

    return {
        "uav_position": [round(float(p), 2) for p in uav_position],
        "num_users": len(user_positions),
        "baseline_aggregate": b_agg,
        "damaged_aggregate": d_agg,
        "aggregate_deltas": aggregate_deltas,
        "connectivity_changed_users": connectivity_changed_users,
        "physical_propagation_changed": any(propagation_changed_flags),
        "num_users_with_propagation_change": sum(propagation_changed_flags),
        "user_comparisons": user_comparisons
    }
