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
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run PPO UAV optimization "
            "with Sionna RT"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "sionna",
            "surrogate"
        ],
        default="sionna"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5
    )

    args = parser.parse_args()

    run_ppo_episode(
        use_sionna=(
            args.mode == "sionna"
        ),
        max_steps=args.steps
    )
