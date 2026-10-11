"""Placement optimisers compared against PPO under equal budgets.

Surrogate optimisers (objective = the calibrated state reward PPO is trained
on, ml/calibrated_model.py; budget counted in surrogate evaluations):

  greedy_joint   one-step lookahead over all 5^K joint 10 m moves, at most
                 12 steps (exactly PPO's action space and step budget)
  coordinate     pattern search, one UAV/axis at a time, steps 40 -> 20 -> 10 m
  evolution      elitist (1+lambda) evolution strategy, sigma 30 m -> 5 m

ns-3 in the loop (objective = MEASURED ns-3 packet delivery ratio):

  ns3_refine     starts from the best of a set of candidate placements, then
                 evaluates single-UAV neighbour moves (20 m, then 10 m) in
                 actual ns-3 runs and keeps the best measured placement.
                 Runs use a dedicated RngRun so that the final evaluation
                 (different RngRun values) is not biased by the selection.
                 One round per step size, i.e. at most two accepted moves.
  ns3_refine_iter
                 the same search, repeated per step size until a round brings
                 no measured gain (at most max_rounds rounds).

All optimisers keep altitude fixed and stay inside the scenario's UAV region.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np

import scenarios
from multiuav_env import MultiUAVEnv, MAX_UAVS

SEARCH_RNG_RUN = 101          # ns-3 RngRun used inside ns3_refine (never used for final scoring)
SEARCH_TRAFFIC_SECONDS = 3    # short runs preserve the 8 s ranking (measured)


class SurrogateObjective:
    """Calibrated state reward of a placement; counts evaluations."""

    def __init__(self, sid, users, num_uavs, model_path=None):
        self.env = MultiUAVEnv(num_uavs=num_uavs, num_users=len(users), scenario=sid,
                               mode="surrogate", reward_mode="calibrated",
                               calibrated_model_path=model_path)
        self.env.user_positions = np.asarray(users, dtype=float)
        reg = scenarios.get_scenario(sid)["uav_region"]
        self.lo = np.array([reg["x"][0], reg["y"][0]])
        self.hi = np.array([reg["x"][1], reg["y"][1]])
        self.evals = 0

    def clip(self, pos):
        pos = np.array(pos, dtype=float)
        pos[:, :2] = np.clip(pos[:, :2], self.lo, self.hi)
        return pos

    def __call__(self, pos) -> float:
        self.evals += 1
        self.env.uav_positions = self.clip(pos)
        self.env._refresh()
        return self.env.state_reward()


def greedy_joint(obj: SurrogateObjective, start, steps=12, step_m=10.0, budget=1500):
    pos, best = obj.clip(start), obj(start)
    k = len(pos)
    moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    move_cost = obj.env.w["w_move"]
    for _ in range(steps):
        cand_best, cand_pos = best, None
        for a in itertools.product(range(5), repeat=k):
            if not any(a) or obj.evals >= budget:
                continue
            p = pos.copy()
            for u, ai in enumerate(a):
                if ai:
                    p[u, :2] += step_m * np.array(moves[ai])
            r = obj(p) - move_cost * sum(1 for ai in a if ai) / k
            if r > cand_best + 1e-9:
                cand_best, cand_pos = r, obj.clip(p)
        if cand_pos is None:
            break
        pos, best = cand_pos, obj(cand_pos)
    return pos, best


def coordinate(obj: SurrogateObjective, start, steps_m=(40.0, 20.0, 10.0), budget=1500):
    pos, best = obj.clip(start), obj(start)
    for s in steps_m:
        improved = True
        while improved and obj.evals < budget:
            improved = False
            for u in range(len(pos)):
                for d in ((s, 0), (-s, 0), (0, s), (0, -s)):
                    if obj.evals >= budget:
                        break
                    p = pos.copy(); p[u, :2] += d
                    r = obj(p)
                    if r > best + 1e-9:
                        pos, best, improved = obj.clip(p), r, True
    return pos, best


def evolution(obj: SurrogateObjective, start, budget=1500, lam=24, sigma0=30.0, sigma1=5.0, seed=0):
    rng = np.random.default_rng(seed)
    pos, best = obj.clip(start), obj(start)
    gens = max(1, (budget - 1) // lam)
    for g in range(gens):
        sigma = sigma0 * (sigma1 / sigma0) ** (g / max(gens - 1, 1))
        for _ in range(lam):
            p = pos.copy(); p[:, :2] += rng.normal(0, sigma, size=(len(p), 2))
            r = obj(p)
            if r > best + 1e-9:
                pos, best = obj.clip(p), r
    return pos, best


SURROGATE_OPTIMISERS: dict[str, Callable] = {
    "greedy_joint": greedy_joint, "coordinate": coordinate, "evolution": evolution,
}


# ----------------------------------------------------------------------
# ns-3 in the loop
# ----------------------------------------------------------------------
def _neighbours(pos, step, lo, hi):
    out = []
    for u in range(len(pos)):
        for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
            p = np.array(pos, dtype=float); p[u, :2] = np.clip(p[u, :2] + d, lo, hi)
            if not np.allclose(p, pos):
                out.append(p)
    return out


def ns3_refine_batch(instances: Sequence[dict], measure: Callable[[list], list],
                     steps_m=(20.0, 10.0), min_gain=0.002, max_rounds=1) -> list[dict]:
    """Measured-feedback local search for many instances at once.

    instances: dicts with "candidates" (list of [K,3] start placements),
               "lo", "hi" (UAV region bounds).
    measure:   callable(list of (instance_idx, placement)) -> list of measured PDR,
               run in parallel by the caller (actual ns-3 simulations).
    max_rounds: neighbourhood rounds per step size. Each round re-measures the
               neighbours of the current best placement; an instance stops at a
               step size as soon as a round brings no measured gain. 1 (default)
               is the original single round per step size.
    Returns per instance: best placement, its measured search PDR, ns-3 runs used,
    and the history of accepted moves.
    """
    state = []
    jobs = [(i, np.asarray(c, dtype=float)) for i, inst in enumerate(instances) for c in inst["candidates"]]
    vals = measure(jobs)
    for i, inst in enumerate(instances):
        mine = [(v, p) for (j, p), v in zip(jobs, vals) if j == i]
        v, p = max(mine, key=lambda t: t[0])
        state.append({"pos": p, "pdr": v, "runs": len(mine), "history": [("start", float(v))]})
    for step in steps_m:
        active = list(range(len(instances)))
        for _ in range(max_rounds):
            jobs = []
            for i in active:
                jobs += [(i, p) for p in _neighbours(state[i]["pos"], step, instances[i]["lo"], instances[i]["hi"])]
            if not jobs:
                break
            vals = measure(jobs)
            improved = []
            for i in active:
                st = state[i]
                mine = [(v, p) for (j, p), v in zip(jobs, vals) if j == i]
                if not mine:
                    continue
                st["runs"] += len(mine)
                v, p = max(mine, key=lambda t: t[0])
                if v > st["pdr"] + min_gain:
                    st["pos"], st["pdr"] = p, v
                    st["history"].append((f"move {step:g} m", float(v)))
                    improved.append(i)
            active = improved
            if not active:
                break
    return state


def region_bounds(sid):
    reg = scenarios.get_scenario(sid)["uav_region"]
    return np.array([reg["x"][0], reg["y"][0]]), np.array([reg["x"][1], reg["y"][1]])


__all__ = ["SurrogateObjective", "SURROGATE_OPTIMISERS", "ns3_refine_batch", "region_bounds",
           "SEARCH_RNG_RUN", "SEARCH_TRAFFIC_SECONDS", "MAX_UAVS"]
