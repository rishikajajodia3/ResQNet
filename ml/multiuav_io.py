"""Shared data structures for the multi-UAV pipeline handoff and trajectory.

Design rule: **for num_uavs == 1 the output is byte-identical to the
existing 1-UAV pipeline** (ml/run_ppo.py). For num_uavs > 1 the same
files gain list-shaped fields (`uav_positions`, `actions`, ...) while
keeping the scalar mirrors (`position`, `action`) so older readers do not
break.

Both ml/run_ppo.py (1-UAV path) and the multi-UAV runner import these
builders so there is one schema definition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence


def _round_pos(p: Sequence[float]) -> list[float]:
    return [round(float(c), 2) for c in p]


# ----------------------------------------------------------------------
# uav_positions.json  (the NS-3 handoff file)
# ----------------------------------------------------------------------
def build_uav_positions(
    uav_positions: Sequence[Sequence[float]],
    user_positions: Sequence[Sequence[float]],
    associations: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Build the handoff dict.

    num_uavs == 1 and associations in (None, all-zeros) -> identical to the
    legacy file (no `associations` key).
    """
    uavs = [_round_pos(p) for p in uav_positions]
    users = [_round_pos(p) for p in user_positions]

    out: dict[str, Any] = {
        "num_uavs": len(uavs),
        "uav_positions": uavs,
        "num_users": len(users),
        "user_positions": users,
    }

    trivial = associations is None or (len(uavs) == 1 and all(a == 0 for a in associations))
    if not trivial:
        out["associations"] = [int(a) for a in associations]
    return out


def read_uav_positions(path: str | Path) -> dict[str, Any]:
    """Read a handoff file (old or new) into a normalised dict.

    Always returns `num_uavs`, `uav_positions` (list of [x,y,z]),
    `num_users`, `user_positions`, `associations` (list; nearest-UAV
    fallback filled in if absent).
    """
    data = json.loads(Path(path).read_text())
    uavs = [list(map(float, p)) for p in data["uav_positions"]]
    users = [list(map(float, p)) for p in data.get("user_positions", [])]
    assoc = data.get("associations")
    if assoc is None:
        assoc = nearest_uav_association(uavs, users)
    return {
        "num_uavs": int(data.get("num_uavs", len(uavs))),
        "uav_positions": uavs,
        "num_users": int(data.get("num_users", len(users))),
        "user_positions": users,
        "associations": [int(a) for a in assoc],
    }


def nearest_uav_association(
    uav_positions: Sequence[Sequence[float]],
    user_positions: Sequence[Sequence[float]],
) -> list[int]:
    """Fallback association: each user -> index of the nearest UAV (3-D)."""
    out = []
    for u in user_positions:
        best_i, best_d = 0, float("inf")
        for i, a in enumerate(uav_positions):
            d = sum((float(u[k]) - float(a[k])) ** 2 for k in range(3))
            if d < best_d:
                best_i, best_d = i, d
        out.append(best_i)
    return out


# ----------------------------------------------------------------------
# ppo_trajectory.json
# ----------------------------------------------------------------------
def build_trajectory_step(
    *,
    step: int,
    num_uavs: int,
    uav_positions: Sequence[Sequence[float]],
    actions: Sequence[Optional[int]],
    connected_users: int,
    coverage_percent: float,
    mean_path_gain_db: float,
    mean_rss_dbm: float,
    mean_sinr_db: float,
    reward: float,
    user_results: list[dict],
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """One trajectory step.

    num_uavs == 1  -> exactly the legacy step keys
        {step, action, position, connected_users, coverage_percent,
         mean_path_gain_db, mean_rss_dbm, mean_sinr_db, reward, user_results}

    num_uavs > 1   -> the same, plus `uav_positions` (list) and `actions`
        (list); `position`/`action` mirror index 0.
    """
    uavs = [_round_pos(p) for p in uav_positions]
    acts = [None if a is None else int(a) for a in actions]

    base = {
        "step": int(step),
        "action": acts[0],
        "position": uavs[0],
        "connected_users": int(connected_users),
        "coverage_percent": float(coverage_percent),
        "mean_path_gain_db": float(mean_path_gain_db),
        "mean_rss_dbm": float(mean_rss_dbm),
        "mean_sinr_db": float(mean_sinr_db),
        "reward": float(reward),
        "user_results": user_results,
    }
    if num_uavs > 1:
        base["uav_positions"] = uavs
        base["actions"] = acts
        if extra:
            base.update(extra)
    return base


def build_trajectory(
    *,
    scene: str,
    mode: str,
    num_uavs: int,
    num_users: int,
    num_steps: int,
    total_reward: float,
    initial_positions: Sequence[Sequence[float]],
    final_positions: Sequence[Sequence[float]],
    steps: list[dict],
    scenario: Optional[str] = None,
    interference_model: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Full trajectory document.

    num_uavs == 1 -> legacy top-level keys only.
    num_uavs > 1  -> adds `initial_positions`, `final_positions`, `scenario`,
        `interference_model`, `seed`; keeps scalar `initial_position` /
        `final_position` mirrors.
    """
    ini = [_round_pos(p) for p in initial_positions]
    fin = [_round_pos(p) for p in final_positions]

    doc: dict[str, Any] = {
        "scene": scene,
        "mode": mode,
        "num_uavs": int(num_uavs),
        "num_users": int(num_users),
        "num_steps": int(num_steps),
        "total_reward": float(total_reward),
        "initial_position": list(initial_positions[0]),
        "final_position": fin[0],
        "trajectory": steps,
    }
    if num_uavs > 1:
        doc["initial_positions"] = [list(p) for p in initial_positions]
        doc["final_positions"] = fin
        if scenario is not None:
            doc["scenario"] = scenario
        if interference_model is not None:
            doc["interference_model"] = interference_model
        if seed is not None:
            doc["seed"] = int(seed)
    return doc


# ----------------------------------------------------------------------
# consistency checks (used by tests and the orchestrator)
# ----------------------------------------------------------------------
def validate_consistency(
    handoff: dict,
    trajectory: dict,
    results: Optional[dict] = None,
    *,
    tol: float = 0.05,
) -> list[str]:
    """Return a list of human-readable problems (empty == consistent)."""
    problems: list[str] = []

    h_uavs = [list(map(float, p)) for p in handoff["uav_positions"]]
    h_users = [list(map(float, p)) for p in handoff["user_positions"]]

    # trajectory final positions
    t_final = trajectory.get("final_positions") or [trajectory["final_position"]]
    t_final = [list(map(float, p)) for p in t_final]
    if len(t_final) != len(h_uavs):
        problems.append(
            f"handoff has {len(h_uavs)} UAVs, trajectory final has {len(t_final)}"
        )
    else:
        for i, (a, b) in enumerate(zip(h_uavs, t_final)):
            if max(abs(x - y) for x, y in zip(a, b)) > tol:
                problems.append(f"UAV {i}: handoff {a} != trajectory final {b}")

    # user positions must match the trajectory's per-step user_results
    last_step = trajectory["trajectory"][-1]
    t_users = [list(map(float, u["position"])) for u in last_step["user_results"]]
    if len(t_users) != len(h_users):
        problems.append(
            f"handoff has {len(h_users)} users, trajectory step has {len(t_users)}"
        )
    else:
        for i, (a, b) in enumerate(zip(h_users, t_users)):
            if max(abs(x - y) for x, y in zip(a, b)) > tol:
                problems.append(f"user {i}: handoff {a} != trajectory {b}")

    # no duplicate UAV positions unless there is only one UAV
    if len(h_uavs) > 1:
        for i in range(len(h_uavs)):
            for j in range(i + 1, len(h_uavs)):
                if max(abs(x - y) for x, y in zip(h_uavs[i], h_uavs[j])) <= tol:
                    problems.append(f"UAVs {i} and {j} occupy the same position {h_uavs[i]}")

    # results file, if given
    if results is not None:
        r_uavs = results.get("uav_positions")
        if r_uavs is None and "uav_position" in results:
            r_uavs = [[results["uav_position"][k] for k in ("x", "y", "z")]]
        r_uavs = [list(map(float, p)) for p in (r_uavs or [])]
        if len(r_uavs) != len(h_uavs):
            problems.append(
                f"results has {len(r_uavs)} UAVs, handoff has {len(h_uavs)}"
            )
        else:
            for i, (a, b) in enumerate(zip(h_uavs, r_uavs)):
                if max(abs(x - y) for x, y in zip(a, b)) > tol:
                    problems.append(f"UAV {i}: handoff {a} != results {b}")

    return problems
