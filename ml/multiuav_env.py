"""Multi-UAV reinforcement-learning environment for ResQNet.

Separate from the 1-UAV ``UAVDynamicEnv`` (ml/dynamic_model.py), which is
left untouched. ``run_ppo.py --num-uavs 1`` still uses the old env and the
old model; this env is used for num_uavs >= 2.

Action space (fixed, so one policy works for any K <= MAX_UAVS):
    MultiDiscrete([5] * MAX_UAVS)   per slot: 0 stay, 1 -X, 2 +X, 3 -Y, 4 +Y
Only the first ``num_uavs`` action components are applied; the rest are
ignored.

Observation (fixed length, independent of num_users and num_uavs):
    per-UAV block  x MAX_UAVS : [x_n, y_n, z_n, load_frac, local_cov_frac]
    global                    : [coverage, connected, rss_n, sinr_n, users_n, uavs_n]
    user distribution         : [centroid_x_n, centroid_y_n, spread_x_n, spread_y_n]
    user occupancy grid 6x6   : normalised user counts (sum ~ 1)

RF feedback:
    mode="surrogate"  free-space path loss (LOS only, no buildings). Fast,
                      deterministic. NEVER labelled as Sionna.
    mode="sionna"     sionna_feedback_engine.evaluate_multi_uav (ray tracing).

Both modes return the SAME result dict shape, and both use best-server
association plus a co-channel SINR (serving power / (noise + sum of the
other UAVs' received powers)); assumption: all UAVs share one channel,
full buffer, no scheduling. For 1 UAV, SINR == SNR.

Reward (multi-objective; weights in ``DEFAULT_REWARD_WEIGHTS``):
    + w_cov  * covered_fraction
    + w_conn * connected_fraction
    + w_qual * normalised mean serving RSS
    + w_prog * change in connected_fraction this step
    + w_balance * load balance across UAVs        (even users-per-UAV)
    - w_overlap * UAV overlap penalty             (UAVs closer than min_sep)
    - w_move  * fraction of UAVs that moved
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE, _ROOT / "sionna"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scenarios  # noqa: E402

try:
    from sionna_feedback_engine import (  # noqa: E402
        evaluate_multi_uav,
        evaluate_connectivity,
        ConnectivityConfig,
    )
except Exception:  # pragma: no cover - Sionna optional at import time
    evaluate_multi_uav = None
    evaluate_connectivity = None
    ConnectivityConfig = None

MAX_UAVS = scenarios.MAX_UAVS
GRID = 6
OBS_LEN = 5 * MAX_UAVS + 6 + 4 + GRID * GRID   # 25 + 6 + 4 + 36 = 71

C_LIGHT = 299_792_458.0

DEFAULT_REWARD_WEIGHTS = {
    "w_cov": 0.35,
    "w_conn": 0.25,
    "w_qual": 0.10,
    "w_prog": 0.10,
    "w_balance": 0.10,
    "w_overlap": 0.15,
    "w_move": 0.05,
}

RSS_MIN_DBM, RSS_MAX_DBM = -95.0, -40.0
SINR_MIN_DB, SINR_MAX_DB = -10.0, 40.0


# ----------------------------------------------------------------------
# surrogate RF model (free-space, LOS only) - NOT Sionna
# ----------------------------------------------------------------------
def _fspl_db(dist_m: float, freq_hz: float) -> float:
    d = max(dist_m, 1.0)
    return 20.0 * math.log10(d) + 20.0 * math.log10(freq_hz) + 20.0 * math.log10(4.0 * math.pi / C_LIGHT)


def surrogate_multi_uav(
    uav_positions: list[list[float]],
    user_positions: list[list[float]],
    connectivity_config: Any,
    tx_power_dbm: float = 30.0,
    freq: float = 2.1e9,
    interference_model: str = "co_channel",
) -> dict:
    """Free-space-approximation counterpart of evaluate_multi_uav().

    Returns the identical dict shape. Buildings / reflections are NOT
    modelled - this is only for fast development and PPO training.
    """
    cfg = connectivity_config or (ConnectivityConfig() if ConnectivityConfig else None)
    noise_dbm = getattr(cfg, "noise_power_dbm", -94.0)
    noise_mw = 10 ** (noise_dbm / 10.0) if noise_dbm is not None else None
    pg_thresh = getattr(cfg, "path_gain_threshold_db", -100.0)

    uavs = np.asarray(uav_positions, dtype=float)
    users = np.asarray(user_positions, dtype=float)
    K, N = len(uavs), len(users)

    user_results = []
    best_rss_l, best_snr_l, best_sinr_l, best_pg_l = [], [], [], []
    connected_count = covered_count = 0
    users_per_uav = {f"uav_{k+1}": 0 for k in range(K)}

    for i in range(N):
        p_mw = np.zeros(K)
        per_uav = []
        for k in range(K):
            d = float(np.linalg.norm(users[i] - uavs[k]))
            fspl = _fspl_db(d, freq)
            pg_db = round(-fspl, 2)
            rss_db = round(tx_power_dbm - fspl, 2)
            rx_mw = 10 ** (rss_db / 10.0)
            p_mw[k] = rx_mw
            snr_db = round(float(10.0 * math.log10(rx_mw / noise_mw)), 2) if noise_mw else None
            per_uav.append({
                "uav_id": f"uav_{k+1}",
                "uav_position": [round(float(c), 2) for c in uavs[k]],
                "path_gain_db": pg_db,
                "rss_dbm": rss_db,
                "snr_db": snr_db,
                "num_paths": 1,
                "shortest_delay_sec": d / C_LIGHT,
            })

        serving = int(np.argmax(p_mw))
        serving_p = p_mw[serving]
        sinr_db = None
        if noise_mw is not None:
            interf = (float(np.sum(p_mw)) - serving_p) if interference_model == "co_channel" else 0.0
            sinr_db = round(float(10.0 * math.log10(serving_p / (noise_mw + interf))), 2)

        best = per_uav[serving]
        coverage = bool(best["path_gain_db"] >= pg_thresh)
        connected = bool(
            evaluate_connectivity(
                rss_dbm=best["rss_dbm"], sinr_db=sinr_db,
                path_gain_db=best["path_gain_db"], config=cfg,
            )
            if evaluate_connectivity
            else best["rss_dbm"] >= getattr(cfg, "rss_threshold_dbm", -95.0)
        )
        if coverage:
            covered_count += 1
        if connected:
            connected_count += 1
            users_per_uav[f"uav_{serving+1}"] += 1

        best_rss_l.append(best["rss_dbm"])
        best_pg_l.append(best["path_gain_db"])
        if best["snr_db"] is not None:
            best_snr_l.append(best["snr_db"])
        if sinr_db is not None:
            best_sinr_l.append(sinr_db)

        user_results.append({
            "user_id": f"user_{i+1}",
            "position": [round(float(c), 2) for c in users[i]],
            "per_uav": per_uav,
            "serving_uav": serving + 1,
            "serving_uav_id": f"uav_{serving+1}",
            "best_rss_dbm": best["rss_dbm"],
            "best_snr_db": best["snr_db"],
            "best_sinr_db": sinr_db,
            "best_path_gain_db": best["path_gain_db"],
            "num_paths": best["num_paths"],
            "shortest_delay_sec": best["shortest_delay_sec"],
            "coverage": coverage,
            "connected": connected,
        })

    mean = lambda xs: round(float(np.mean(xs)), 2) if xs else -200.0  # noqa: E731
    return {
        "uav_positions": [[round(float(c), 2) for c in q] for q in uavs],
        "scene_name": "surrogate:free_space",
        "num_uavs": K,
        "num_users": N,
        "interference_model": interference_model,
        "user_results": user_results,
        "aggregate": {
            "connected_users_count": connected_count,
            "connected_users_percentage": round(connected_count / N * 100.0, 2),
            "coverage_percentage": round(covered_count / N * 100.0, 2),
            "mean_best_path_gain_db": mean(best_pg_l),
            "mean_best_rss_dbm": mean(best_rss_l),
            "mean_best_snr_db": mean(best_snr_l),
            "mean_best_sinr_db": mean(best_sinr_l),
            "users_per_uav": users_per_uav,
        },
    }


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------
class MultiUAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_uavs: int = 3,
        num_users: int = 30,
        scenario: str = "blocked_urban",
        mode: str = "surrogate",
        seed: int = 42,
        max_steps: int = 12,
        step_size: float = 10.0,
        min_separation_m: float = 25.0,
        interference_model: str = "co_channel",
        connectivity_config: Optional[Any] = None,
        reward_weights: Optional[dict] = None,
        sampler: Optional[Callable[[int], dict]] = None,
    ):
        super().__init__()
        if mode not in ("surrogate", "sionna"):
            raise ValueError("mode must be 'surrogate' or 'sionna'")
        self.num_uavs = int(num_uavs)
        self.num_users = int(num_users)
        self.scenario = scenario
        self.mode = mode
        self.base_seed = int(seed)
        self.max_steps = int(max_steps)
        self.step_size = float(step_size)
        self.min_sep = float(min_separation_m)
        self.interference_model = interference_model
        self.cfg = connectivity_config or (ConnectivityConfig() if ConnectivityConfig else None)
        self.w = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.sampler = sampler
        self._episode = 0

        self.action_space = spaces.MultiDiscrete([5] * MAX_UAVS)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_LEN,), dtype=np.float32
        )

        self.uav_positions: np.ndarray | None = None
        self.user_positions: np.ndarray | None = None
        self.current_step = 0
        self.last_eval: dict | None = None
        self.initial_positions: np.ndarray | None = None

    # -- RF evaluation ------------------------------------------------
    def _evaluate(self) -> dict:
        uavs = self.uav_positions.tolist()
        users = self.user_positions.tolist()
        if self.mode == "sionna":
            if evaluate_multi_uav is None:
                raise RuntimeError("Sionna not available; use mode='surrogate'.")
            return evaluate_multi_uav(
                uavs, user_positions=users,
                scene_name=self._scene_xml(), connectivity_config=self.cfg,
                seed=self.base_seed, interference_model=self.interference_model,
            )
        return surrogate_multi_uav(
            uavs, users, connectivity_config=self.cfg,
            interference_model=self.interference_model,
        )

    def _scene_xml(self) -> str:
        return scenarios.get_scenario(self.scenario)["scene"]

    # -- observation ------------------------------------------------
    def _region(self):
        return scenarios.get_scenario(self.scenario)["uav_region"], \
               scenarios.get_scenario(self.scenario)["user_region"]

    def _obs(self) -> np.ndarray:
        uav_reg, user_reg = self._region()
        ux0, ux1 = uav_reg["x"]; uy0, uy1 = uav_reg["y"]; uz0, uz1 = uav_reg["z"]
        agg = self.last_eval["aggregate"]

        parts = []
        # per-UAV block (padded to MAX_UAVS)
        upu = agg["users_per_uav"]
        # local coverage per UAV: connected users it serves / users it serves
        serve_conn = {f"uav_{k+1}": [0, 0] for k in range(self.num_uavs)}
        for u in self.last_eval["user_results"]:
            key = u["serving_uav_id"]
            if key in serve_conn:
                serve_conn[key][1] += 1
                if u["connected"]:
                    serve_conn[key][0] += 1
        for k in range(MAX_UAVS):
            if k < self.num_uavs:
                x, y, z = self.uav_positions[k]
                load = upu.get(f"uav_{k+1}", 0) / max(self.num_users, 1)
                c, tot = serve_conn.get(f"uav_{k+1}", [0, 0])
                local_cov = c / tot if tot else 0.0
                parts += [
                    (x - ux0) / (ux1 - ux0),
                    (y - uy0) / (uy1 - uy0),
                    (z - uz0) / (uz1 - uz0),
                    load,
                    local_cov,
                ]
            else:
                parts += [0.0, 0.0, 0.0, 0.0, 0.0]

        # global
        rss_n = np.clip((agg["mean_best_rss_dbm"] - RSS_MIN_DBM) / (RSS_MAX_DBM - RSS_MIN_DBM), 0, 1)
        sinr_n = np.clip((agg["mean_best_sinr_db"] - SINR_MIN_DB) / (SINR_MAX_DB - SINR_MIN_DB), 0, 1)
        parts += [
            agg["coverage_percentage"] / 100.0,
            agg["connected_users_percentage"] / 100.0,
            float(rss_n),
            float(sinr_n),
            min(self.num_users / 60.0, 1.0),
            self.num_uavs / MAX_UAVS,
        ]

        # user distribution summary
        xs = (self.user_positions[:, 0] - user_reg["x"][0]) / (user_reg["x"][1] - user_reg["x"][0])
        ys = (self.user_positions[:, 1] - user_reg["y"][0]) / (user_reg["y"][1] - user_reg["y"][0])
        parts += [float(xs.mean()), float(ys.mean()), float(xs.std()), float(ys.std())]

        # occupancy grid
        gx = np.clip((xs * GRID).astype(int), 0, GRID - 1)
        gy = np.clip((ys * GRID).astype(int), 0, GRID - 1)
        grid = np.zeros((GRID, GRID))
        for a, b in zip(gy, gx):
            grid[a, b] += 1
        grid = grid.flatten() / max(self.num_users, 1)
        parts += grid.tolist()

        return np.asarray(parts, dtype=np.float32)

    # -- reward ---------------------------------------------------
    def _overlap_penalty(self) -> float:
        k = self.num_uavs
        if k < 2:
            return 0.0
        vals = []
        for i in range(k):
            for j in range(i + 1, k):
                d = float(np.linalg.norm(self.uav_positions[i] - self.uav_positions[j]))
                vals.append(max(0.0, 1.0 - d / self.min_sep))
        return float(np.mean(vals))

    def _load_balance(self) -> float:
        upu = self.last_eval["aggregate"]["users_per_uav"]
        loads = np.array([upu.get(f"uav_{k+1}", 0) for k in range(self.num_uavs)], dtype=float)
        mean_load = self.num_users / self.num_uavs
        if mean_load <= 0:
            return 0.0
        imbalance = loads.std() / (mean_load + 1e-9)
        return float(np.clip(1.0 - imbalance, 0.0, 1.0))

    def _reward(self, prev_conn_frac: float, moved_frac: float) -> tuple[float, dict]:
        agg = self.last_eval["aggregate"]
        cov = agg["coverage_percentage"] / 100.0
        conn = agg["connected_users_percentage"] / 100.0
        rss_n = float(np.clip((agg["mean_best_rss_dbm"] - RSS_MIN_DBM) / (RSS_MAX_DBM - RSS_MIN_DBM), 0, 1))
        d_conn = conn - prev_conn_frac
        overlap = self._overlap_penalty()
        balance = self._load_balance()
        w = self.w
        r = (
            w["w_cov"] * cov
            + w["w_conn"] * conn
            + w["w_qual"] * rss_n
            + w["w_prog"] * d_conn
            + w["w_balance"] * balance
            - w["w_overlap"] * overlap
            - w["w_move"] * moved_frac
        )
        return float(r), {
            "cov": cov, "conn": conn, "rss_n": rss_n, "d_conn": d_conn,
            "overlap": overlap, "balance": balance, "moved_frac": moved_frac,
        }

    # -- gym API ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.sampler is not None:
            options = {**self.sampler(self._episode), **options}
        self._episode += 1

        self.scenario = options.get("scenario", self.scenario)
        self.num_uavs = int(options.get("num_uavs", self.num_uavs))
        self.num_users = int(options.get("num_users", self.num_users))
        ep_seed = int(options.get("episode_seed", seed if seed is not None else self.base_seed))

        self.user_positions = scenarios.generate_users(self.scenario, self.num_users, ep_seed)
        self.uav_positions = scenarios.initial_uav_positions(self.num_uavs, self.scenario).astype(float)
        self.initial_positions = self.uav_positions.copy()
        self.current_step = 0
        self.last_eval = self._evaluate()

        obs = self._obs()
        info = {"scenario": self.scenario, "num_uavs": self.num_uavs,
                "num_users": self.num_users, "episode_seed": ep_seed,
                "eval": self.last_eval}
        return obs, info

    def _apply(self, actions):
        uav_reg = self._region()[0]
        moved = 0
        for k in range(self.num_uavs):
            a = int(actions[k])
            if a == 1:
                self.uav_positions[k, 0] -= self.step_size; moved += 1
            elif a == 2:
                self.uav_positions[k, 0] += self.step_size; moved += 1
            elif a == 3:
                self.uav_positions[k, 1] -= self.step_size; moved += 1
            elif a == 4:
                self.uav_positions[k, 1] += self.step_size; moved += 1
            # a == 0 -> stay
        self.uav_positions[:, 0] = np.clip(self.uav_positions[:, 0], uav_reg["x"][0], uav_reg["x"][1])
        self.uav_positions[:, 1] = np.clip(self.uav_positions[:, 1], uav_reg["y"][0], uav_reg["y"][1])
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], uav_reg["z"][0], uav_reg["z"][1])
        return moved / max(self.num_uavs, 1)

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1)
        prev_conn = self.last_eval["aggregate"]["connected_users_percentage"] / 100.0
        moved_frac = self._apply(actions)
        self.last_eval = self._evaluate()
        reward, breakdown = self._reward(prev_conn, moved_frac)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False
        info = {
            "eval": self.last_eval,
            "actions": [int(actions[k]) for k in range(self.num_uavs)],
            "uav_positions": self.uav_positions.tolist(),
            "reward_breakdown": breakdown,
        }
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        pass
