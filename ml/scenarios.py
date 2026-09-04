"""Configurable disaster scenarios and deterministic user / UAV generation.

A "scenario" is a set of spatial and propagation conditions (region size,
mesh, user spatial distribution). It is NOT a claim about a specific real
disaster.

Everything here is deterministic given a seed:

    generate_users("blocked_urban", num_users=30, seed=42)

returns the same 30 positions every time. This keeps experiments
reproducible.

Config lives in ``configs/scenarios.json`` and ``configs/training.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_FILE = PROJECT_ROOT / "configs" / "scenarios.json"
TRAINING_FILE = PROJECT_ROOT / "configs" / "training.json"

# Hard ceiling on UAV count the multi-UAV env / observation is sized for.
MAX_UAVS = 5

VALID_DISTRIBUTIONS = ("uniform", "clustered", "edge_heavy", "random")


# ----------------------------------------------------------------------
# config loading
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_scenarios(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else SCENARIOS_FILE
    data = json.loads(p.read_text())
    return data


@lru_cache(maxsize=1)
def load_training_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else TRAINING_FILE
    return json.loads(p.read_text())


def scenario_ids() -> list[str]:
    return list(load_scenarios()["scenarios"].keys())


def get_scenario(scenario_id: str) -> dict[str, Any]:
    scen = load_scenarios()["scenarios"]
    if scenario_id not in scen:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. Known: {list(scen.keys())}"
        )
    return scen[scenario_id]


def user_height() -> float:
    return float(load_scenarios().get("user_height_m", 1.5))


# ----------------------------------------------------------------------
# user generation
# ----------------------------------------------------------------------
def _clip_xy(xy: np.ndarray, region: dict) -> np.ndarray:
    xy = xy.copy()
    xy[:, 0] = np.clip(xy[:, 0], region["x"][0], region["x"][1])
    xy[:, 1] = np.clip(xy[:, 1], region["y"][0], region["y"][1])
    return xy


def _cluster_centers(rng: np.random.Generator, region: dict, k: int) -> np.ndarray:
    """k well-separated centres inside the region (seeded, reproducible)."""
    x0, x1 = region["x"]
    y0, y1 = region["y"]
    span = min(x1 - x0, y1 - y0)
    min_sep = 0.35 * span
    centers: list[np.ndarray] = []
    attempts = 0
    while len(centers) < k and attempts < 2000:
        attempts += 1
        c = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if all(np.linalg.norm(c - e) >= min_sep for e in centers):
            centers.append(c)
    # fall back: relax separation if the region is too small for k clusters
    while len(centers) < k:
        centers.append(
            np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        )
    return np.array(centers)


def generate_users(
    scenario_id: str,
    num_users: int,
    seed: int,
    distribution: str | None = None,
) -> np.ndarray:
    """Deterministic (num_users, 3) array of ground-user positions.

    Same (scenario_id, num_users, seed) -> identical output.
    """
    if num_users < 1:
        raise ValueError("num_users must be >= 1")

    scen = get_scenario(scenario_id)
    region = scen["user_region"]
    dist = distribution or scen.get("user_distribution", "uniform")
    if dist not in VALID_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{dist}'. {VALID_DISTRIBUTIONS}")

    rng = np.random.default_rng(seed)
    x0, x1 = region["x"]
    y0, y1 = region["y"]
    h = user_height()

    if dist in ("uniform", "random"):
        # 'random' is a seeded uniform draw; kept as a distinct label so a
        # scenario config can request it explicitly.
        xs = rng.uniform(x0, x1, num_users)
        ys = rng.uniform(y0, y1, num_users)
        xy = np.column_stack([xs, ys])

    elif dist == "clustered":
        k = int(scen.get("num_clusters", 3))
        std = float(scen.get("cluster_std_m", 8.0))
        centers = _cluster_centers(rng, region, k)
        # round-robin assignment so every cluster is populated
        assign = np.array([i % k for i in range(num_users)])
        rng.shuffle(assign)
        pts = centers[assign] + rng.normal(0.0, std, size=(num_users, 2))
        xy = _clip_xy(pts, region)

    elif dist == "edge_heavy":
        # each user is placed near one of the four boundaries, within an
        # inward band of 15 % of the region span
        band_x = 0.15 * (x1 - x0)
        band_y = 0.15 * (y1 - y0)
        edges = rng.integers(0, 4, num_users)
        xs = np.empty(num_users)
        ys = np.empty(num_users)
        for i, e in enumerate(edges):
            if e == 0:  # bottom
                xs[i] = rng.uniform(x0, x1)
                ys[i] = rng.uniform(y0, y0 + band_y)
            elif e == 1:  # top
                xs[i] = rng.uniform(x0, x1)
                ys[i] = rng.uniform(y1 - band_y, y1)
            elif e == 2:  # left
                xs[i] = rng.uniform(x0, x0 + band_x)
                ys[i] = rng.uniform(y0, y1)
            else:  # right
                xs[i] = rng.uniform(x1 - band_x, x1)
                ys[i] = rng.uniform(y0, y1)
        xy = np.column_stack([xs, ys])

    else:  # pragma: no cover - guarded above
        raise ValueError(dist)

    out = np.column_stack([xy, np.full(len(xy), h)])
    return np.round(out.astype(float), 2)


# ----------------------------------------------------------------------
# initial UAV placement
# ----------------------------------------------------------------------
def initial_uav_positions(num_uavs: int, scenario_id: str, seed: int | None = None) -> np.ndarray:
    """Deterministic (num_uavs, 3) spatially separated start positions.

    Purely geometric (no randomness) so UAVs never coincide for num_uavs>=2.
    ``seed`` is accepted for API symmetry but not used.
    """
    if not 1 <= num_uavs <= MAX_UAVS:
        raise ValueError(f"num_uavs must be in [1, {MAX_UAVS}], got {num_uavs}")

    region = get_scenario(scenario_id)["uav_region"]
    x0, x1 = region["x"]
    y0, y1 = region["y"]
    z0, z1 = region["z"]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    z = round((z0 + z1) / 2.0, 2)
    # offsets at 30 % of half-span from centre
    ox = 0.30 * (x1 - x0) / 2.0
    oy = 0.30 * (y1 - y0) / 2.0

    if num_uavs == 1:
        pts = [[cx, cy, z]]
    elif num_uavs == 2:
        pts = [[cx - ox, cy, z], [cx + ox, cy, z]]
    elif num_uavs == 3:
        pts = [[cx, cy + oy, z], [cx - ox, cy - oy, z], [cx + ox, cy - oy, z]]
    elif num_uavs == 4:
        pts = [
            [cx - ox, cy + oy, z],
            [cx + ox, cy + oy, z],
            [cx - ox, cy - oy, z],
            [cx + ox, cy - oy, z],
        ]
    else:  # 5
        pts = [
            [cx, cy, z],
            [cx - ox, cy + oy, z],
            [cx + ox, cy + oy, z],
            [cx - ox, cy - oy, z],
            [cx + ox, cy - oy, z],
        ]
    return np.round(np.array(pts, dtype=float), 2)


# ----------------------------------------------------------------------
# training-config helpers
# ----------------------------------------------------------------------
def train_scenarios() -> list[str]:
    return list(load_training_config()["train"]["scenarios"])


def eval_scenarios() -> list[str]:
    return list(load_training_config()["eval"]["scenarios"])


def assert_train_eval_disjoint() -> None:
    tr, ev = set(train_scenarios()), set(eval_scenarios())
    overlap = tr & ev
    if overlap:
        raise AssertionError(f"train / eval scenarios overlap: {overlap}")
    known = set(scenario_ids())
    unknown = (tr | ev) - known
    if unknown:
        raise AssertionError(f"training config references unknown scenarios: {unknown}")


if __name__ == "__main__":
    assert_train_eval_disjoint()
    print("scenarios:", scenario_ids())
    print("train:", train_scenarios(), "| eval (held out):", eval_scenarios())
    for sid in scenario_ids():
        u = generate_users(sid, 12, seed=42)
        p = initial_uav_positions(3, sid)
        print(f"  {sid:20s} users[0]={u[0].tolist()}  uav0={p[0].tolist()}")
