import sys
import json
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
SIONNA_DIR = PROJECT_ROOT / "sionna"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(SIONNA_DIR))


# ============================================================
# IMPORT ENVIRONMENT
# ============================================================

from dynamic_model import UAVDynamicEnv

# The ground-user layout evaluated by Sionna RT (see
# sionna/sionna_feedback_engine.py). It is written into the NS-3
# handoff file so that NS-3 simulates the exact same user positions.
from sionna_feedback_engine import DEFAULT_10_USERS


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = PROJECT_ROOT / "models" / "ppo_uav_dynamic.zip"

UAV_POSITION_PATH = (
    PROJECT_ROOT / "data" / "uav_positions.json"
)

TRAJECTORY_PATH = (
    PROJECT_ROOT / "data" / "ppo_trajectory.json"
)


# ============================================================
# GET REAL SIONNA METRICS
# ============================================================

def get_metrics(env):

    position = np.asarray(
        env.uav_position,
        dtype=float
    )

    coverage = float(
        env.current_coverage
        if env.current_coverage is not None
        else 0.0
    )

    connected_users = 0
    mean_rss = 0.0
    mean_path_gain = 0.0
    mean_sinr = 0.0
    user_results = []

    sionna_metrics = getattr(
        env,
        "current_sionna_metrics",
        None
    )

    if sionna_metrics is not None:

        aggregate = sionna_metrics.get(
            "aggregate",
            {}
        )

        # Preserve the live Sionna per-user link metrics for this step so
        # the dashboard can render per-user diagnostics. These come straight
        # from sionna_feedback_engine's evaluate_uav_position() output.
        for u in sionna_metrics.get("user_results", []):
            user_results.append({
                "user_id": u.get("user_id"),
                "position": u.get("position"),
                "rss_dbm": u.get("rss_dbm"),
                "sinr_db": u.get("sinr_db"),
                "path_gain_db": u.get("path_gain_db"),
                "num_paths": u.get("num_paths"),
                "shortest_delay_sec": u.get("shortest_delay_sec"),
                "coverage": u.get("coverage"),
                "connected": u.get("connected")
            })

        coverage = float(
            aggregate.get(
                "coverage_percentage",
                coverage
            )
        )

        connected_users = int(
            aggregate.get(
                "connected_users_count",
                0
            )
        )

        mean_rss = float(
            aggregate.get(
                "mean_rss_dbm",
                aggregate.get(
                    "mean_rs_dbm",
                    0.0
                )
            )
        )

        mean_path_gain = float(
            aggregate.get(
                "mean_path_gain_db",
                0.0
            )
        )

        # The Sionna feedback engine reports this quantity as "mean_sinr_db".
        # For the single-UAV scenario there is no co-channel interference
        # (I = 0), so SINR reduces to SNR = RSS - noise_power. We keep the
        # engine's key name to stay consistent with its output schema.
        mean_sinr = float(
            aggregate.get(
                "mean_sinr_db",
                0.0
            )
        )

    return {
        "position": position.tolist(),
        "coverage_percentage": coverage,
        "connected_users_count": connected_users,
        "mean_rss_dbm": mean_rss,
        "mean_path_gain_db": mean_path_gain,
        "mean_sinr_db": mean_sinr,
        "user_results": user_results
    }


# ============================================================
# RUN PPO
# ============================================================

def run_ppo_episode(
    use_sionna=True,
    max_steps=5
):

    mode_name = (
        "sionna"
        if use_sionna
        else "surrogate"
    )

    print()
    print("=" * 60)
    print("RESQNET PPO UAV OPTIMIZATION")
    print("=" * 60)

    print(f"Mode       : {mode_name}")
    print(f"Steps      : {max_steps}")
    print("Scene      : city_damaged.xml")

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    env = UAVDynamicEnv(
        use_sionna_feedback=use_sionna,
        scene_name="city_damaged.xml"
    )

    # ========================================================
    # LOAD PPO MODEL
    # ========================================================

    model = None

    if MODEL_PATH.exists():

        from stable_baselines3 import PPO

        print()
        print("Loading PPO model...")

        model = PPO.load(
            str(MODEL_PATH)
        )

        print(
            "PPO model loaded successfully."
        )

    else:

        print()
        print(
            "WARNING: PPO model not found."
        )

    # ========================================================
    # RESET
    # ========================================================

    observation, info = env.reset(
        seed=42
    )

    observation = np.asarray(
        observation
    )

    # ========================================================
    # INITIAL METRICS
    # ========================================================

    initial = get_metrics(env)

    print()
    print("Initial UAV state:")

    print(
        "Position:",
        initial["position"]
    )

    print(
        "Coverage:",
        initial["coverage_percentage"],
        "%"
    )

    print(
        "Connected users:",
        initial["connected_users_count"]
    )

    print(
        "Mean RSS:",
        initial["mean_rss_dbm"],
        "dBm"
    )

    print(
        "Mean path gain:",
        initial["mean_path_gain_db"],
        "dB"
    )

    print(
        "Mean SINR:",
        initial["mean_sinr_db"],
        "dB"
    )

    # ========================================================
    # STORE TRAJECTORY
    # ========================================================

    trajectory = []

    trajectory.append({
        "step": 0,
        "action": None,
        "position": initial["position"],
        "connected_users": initial[
            "connected_users_count"
        ],
        "coverage_percent": initial[
            "coverage_percentage"
        ],
        "mean_path_gain_db": initial[
            "mean_path_gain_db"
        ],
        "mean_rss_dbm": initial[
            "mean_rss_dbm"
        ],
        "mean_sinr_db": initial[
            "mean_sinr_db"
        ],
        "reward": 0.0,
        "user_results": initial[
            "user_results"
        ]
    })

    # ========================================================
    # PPO LOOP
    # ========================================================

    total_reward = 0.0

    for step in range(
        1,
        max_steps + 1
    ):

        # ----------------------------------------------------
        # SELECT ACTION
        # ----------------------------------------------------

        if model is not None:

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = int(
                np.asarray(action).item()
            )

        else:

            # Fallback:
            # Move UAV in -X direction.
            action = 1

        # ----------------------------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------------------------

        result = env.step(
            action
        )

        observation = np.asarray(
            result[0]
        )

        reward = float(
            result[1]
        )

        terminated = bool(
            result[2]
        )

        truncated = bool(
            result[3]
        )

        total_reward += reward

        # ----------------------------------------------------
        # GET REAL SIONNA RESULTS
        # ----------------------------------------------------

        metrics = get_metrics(env)

        # ----------------------------------------------------
        # SAVE STEP
        # ----------------------------------------------------

        step_data = {

            "step": step,

            "action": action,

            "position": metrics[
                "position"
            ],

            "connected_users": metrics[
                "connected_users_count"
            ],

            "coverage_percent": metrics[
                "coverage_percentage"
            ],

            "mean_path_gain_db": metrics[
                "mean_path_gain_db"
            ],

            "mean_rss_dbm": metrics[
                "mean_rss_dbm"
            ],

            "mean_sinr_db": metrics[
                "mean_sinr_db"
            ],

            "reward": reward,

            "user_results": metrics[
                "user_results"
            ]
        }

        trajectory.append(
            step_data
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(
            f"Step {step}"
        )

        print(
            "  Action:",
            action
        )

        print(
            "  UAV position:",
            metrics["position"]
        )

        print(
            "  Coverage:",
            metrics[
                "coverage_percentage"
            ],
            "%"
        )

        print(
            "  Connected:",
            metrics[
                "connected_users_count"
            ],
            "/ 10"
        )

        print(
            "  Mean RSS:",
            metrics[
                "mean_rss_dbm"
            ],
            "dBm"
        )

        print(
            "  Mean path gain:",
            metrics[
                "mean_path_gain_db"
            ],
            "dB"
        )

        print(
            "  Mean SINR:",
            metrics[
                "mean_sinr_db"
            ],
            "dB"
        )

        print(
            "  Reward:",
            round(
                reward,
                4
            )
        )

        if terminated or truncated:

            print()
            print(
                "Episode finished early."
            )

            break

    # ========================================================
    # FINAL POSITION
    # ========================================================

    final_position = [
        round(
            float(x),
            2
        )
        for x in trajectory[-1][
            "position"
        ]
    ]

    # ========================================================
    # SAVE UAV POSITION FOR NS-3
    # ========================================================

    # Hand off the same ground-user positions that Sionna RT evaluated,
    # so NS-3 simulates an identical scenario. Prefer the exact list from
    # the last Sionna evaluation; fall back to the canonical default.
    sionna_metrics = getattr(env, "current_sionna_metrics", None)

    if sionna_metrics is not None:
        user_positions = [
            [round(float(c), 2) for c in u["position"]]
            for u in sionna_metrics["user_results"]
        ]
    else:
        user_positions = [
            [round(float(c), 2) for c in u]
            for u in DEFAULT_10_USERS
        ]

    uav_data = {

        "num_uavs": 1,

        "uav_positions": [
            final_position
        ],

        "num_users": len(user_positions),

        "user_positions": user_positions
    }

    UAV_POSITION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        UAV_POSITION_PATH,
        "w"
    ) as f:

        json.dump(
            uav_data,
            f,
            indent=2
        )

    # ========================================================
    # SAVE COMPLETE PPO TRAJECTORY
    # ========================================================

    trajectory_data = {

        "scene":
            "city_damaged.xml",

        "mode":
            mode_name,

        "num_uavs":
            1,

        "num_users":
            10,

        "num_steps":
            len(trajectory) - 1,

        "total_reward":
            float(total_reward),

        "initial_position":
            trajectory[0]["position"],

        "final_position":
            final_position,

        "trajectory":
            trajectory
    }

    with open(
        TRAJECTORY_PATH,
        "w"
    ) as f:

        json.dump(
            trajectory_data,
            f,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PPO COMPLETE")
    print("=" * 60)

    print(
        "Initial position:",
        trajectory[0]["position"]
    )

    print(
        "Final position:",
        final_position
    )

    print(
        "Total reward:",
        round(
            total_reward,
            4
        )
    )

    print()
    print(
        "Saved UAV position:"
    )

    print(
        UAV_POSITION_PATH
    )

    print()
    print(
        "Saved PPO trajectory:"
    )

    print(
        TRAJECTORY_PATH
    )

    print("=" * 60)

    return trajectory


# ============================================================
# MULTI-UAV EPISODE  (num_uavs >= 1, scenario-driven)
# ============================================================
#
# The legacy single-UAV path above (run_ppo_episode / UAVDynamicEnv /
# models/ppo_uav_dynamic.zip) is left completely untouched. This function
# is only used when the caller passes multi-UAV / scenario arguments.

MULTIUAV_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_multiuav" / "model.zip"


def _step_user_results(eval_result):
    """Map evaluate_multi_uav()/surrogate user_results into the trajectory's
    per-user shape (keeps the legacy fields + adds serving_uav / per_uav)."""
    out = []
    for u in eval_result["user_results"]:
        out.append({
            "user_id": u["user_id"],
            "position": u["position"],
            "rss_dbm": u.get("best_rss_dbm"),
            "sinr_db": u.get("best_sinr_db"),
            "snr_db": u.get("best_snr_db"),
            "path_gain_db": u.get("best_path_gain_db"),
            "num_paths": u.get("num_paths"),
            "shortest_delay_sec": u.get("shortest_delay_sec"),
            "coverage": u.get("coverage"),
            "connected": u.get("connected"),
            "serving_uav": u.get("serving_uav"),
            "per_uav": u.get("per_uav"),
        })
    return out


def run_multiuav_episode(
    scenario="blocked_urban",
    num_uavs=3,
    num_users=30,
    mode="surrogate",
    max_steps=10,
    seed=42,
    interference_model="co_channel",
    out_uav_positions=None,
    out_trajectory=None,
):
    import multiuav_io as mio
    from multiuav_env import MultiUAVEnv, MAX_UAVS

    uav_out = Path(out_uav_positions) if out_uav_positions else UAV_POSITION_PATH
    traj_out = Path(out_trajectory) if out_trajectory else TRAJECTORY_PATH

    print()
    print("=" * 66)
    print("RESQNET MULTI-UAV PPO OPTIMIZATION")
    print("=" * 66)
    print(f"Scenario   : {scenario}")
    print(f"UAVs       : {num_uavs}   Users: {num_users}")
    print(f"Mode       : {mode}   Steps: {max_steps}")
    print(f"Interference: {interference_model} (co-channel SINR from ray-traced powers)")

    env = MultiUAVEnv(
        num_uavs=num_uavs, num_users=num_users, scenario=scenario,
        mode=mode, seed=seed, max_steps=max_steps,
        interference_model=interference_model,
    )
    obs, info = env.reset(seed=seed)

    model = None
    if MULTIUAV_MODEL_PATH.exists():
        from stable_baselines3 import PPO
        print(f"\nLoading multi-UAV PPO model: {MULTIUAV_MODEL_PATH}")
        model = PPO.load(str(MULTIUAV_MODEL_PATH))
    else:
        print("\nWARNING: multi-UAV PPO model not found "
              f"({MULTIUAV_MODEL_PATH}); using random actions.\n"
              "Train it with: python ml/train_multiuav.py")

    scene_xml = env._scene_xml()
    initial_positions = env.initial_positions.tolist()

    def _mk_step(step_idx, actions, ev):
        agg = ev["aggregate"]
        return mio.build_trajectory_step(
            step=step_idx,
            num_uavs=num_uavs,
            uav_positions=env.uav_positions.tolist(),
            actions=actions,
            connected_users=agg["connected_users_count"],
            coverage_percent=agg["coverage_percentage"],
            mean_path_gain_db=agg["mean_best_path_gain_db"],
            mean_rss_dbm=agg["mean_best_rss_dbm"],
            mean_sinr_db=agg["mean_best_sinr_db"],
            reward=0.0,
            user_results=_step_user_results(ev),
            extra={
                "mean_snr_db": agg["mean_best_snr_db"],
                "users_per_uav": agg["users_per_uav"],
            },
        )

    trajectory = [_mk_step(0, [None] * num_uavs, env.last_eval)]

    total_reward = 0.0
    for step in range(1, max_steps + 1):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action).ravel()
        else:
            action = env.np_random.integers(0, 5, size=MAX_UAVS)

        obs, reward, terminated, truncated, sinfo = env.step(action)
        total_reward += float(reward)

        rec = _mk_step(step, sinfo["actions"], sinfo["eval"])
        rec["reward"] = float(reward)
        trajectory.append(rec)

        agg = sinfo["eval"]["aggregate"]
        print(f"\nStep {step}  actions={sinfo['actions']}")
        print(f"  coverage {agg['coverage_percentage']}%  "
              f"connected {agg['connected_users_count']}/{num_users}  "
              f"mean SNR {agg['mean_best_snr_db']} dB  "
              f"mean SINR {agg['mean_best_sinr_db']} dB  "
              f"reward {reward:.4f}")
        print(f"  users_per_uav {agg['users_per_uav']}")
        if terminated or truncated:
            break

    final_positions = env.uav_positions.tolist()
    final_eval = env.last_eval
    associations = [u["serving_uav"] - 1 for u in final_eval["user_results"]]
    user_positions = [u["position"] for u in final_eval["user_results"]]

    uav_data = mio.build_uav_positions(final_positions, user_positions, associations)
    uav_out.parent.mkdir(parents=True, exist_ok=True)
    with open(uav_out, "w") as f:
        json.dump(uav_data, f, indent=2)

    traj_doc = mio.build_trajectory(
        scene=scene_xml, mode=mode, num_uavs=num_uavs, num_users=num_users,
        num_steps=len(trajectory) - 1, total_reward=total_reward,
        initial_positions=initial_positions, final_positions=final_positions,
        steps=trajectory, scenario=scenario,
        interference_model=interference_model, seed=seed,
    )
    with open(traj_out, "w") as f:
        json.dump(traj_doc, f, indent=2)

    problems = mio.validate_consistency(uav_data, traj_doc)
    print()
    print("=" * 66)
    print("MULTI-UAV PPO COMPLETE")
    print("=" * 66)
    print(f"Initial UAV positions: {initial_positions}")
    print(f"Final UAV positions  : {[[round(c, 2) for c in p] for p in final_positions]}")
    print(f"Total reward         : {round(total_reward, 4)}")
    print(f"Consistency check    : {'OK' if not problems else problems}")
    print(f"Saved: {uav_out}")
    print(f"Saved: {traj_out}")
    print("=" * 66)
    if problems:
        raise RuntimeError(f"handoff/trajectory consistency check failed: {problems}")
    return traj_doc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run PPO UAV optimization (single-UAV legacy path, or "
                    "multi-UAV / scenario path when those arguments are given)."
    )

    parser.add_argument("--mode", choices=["sionna", "surrogate"], default="sionna")
    parser.add_argument("--steps", type=int, default=5)

    # Multi-UAV / scenario arguments. If NONE of these are given the legacy
    # single-UAV path runs unchanged.
    parser.add_argument("--num-uavs", type=int, default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--interference-model", choices=["co_channel", "orthogonal"],
        default="co_channel",
    )

    args = parser.parse_args()

    multiuav_requested = (
        args.num_uavs is not None
        or args.num_users is not None
        or args.scenario is not None
    )

    if not multiuav_requested:
        # ---- unchanged legacy single-UAV path ----
        run_ppo_episode(
            use_sionna=(args.mode == "sionna"),
            max_steps=args.steps,
        )
    else:
        run_multiuav_episode(
            scenario=args.scenario or "blocked_urban",
            num_uavs=args.num_uavs if args.num_uavs is not None else 3,
            num_users=args.num_users if args.num_users is not None else 30,
            mode=args.mode,
            max_steps=args.steps,
            seed=args.seed,
            interference_model=args.interference_model,
        )
