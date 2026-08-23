import numpy as np
import pandas as pd
import gymnasium as gym

from gymnasium import spaces

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_FILE = (
    PROJECT_ROOT
    / "sionna"
    / "earthquake_trajectory_dataset.csv"
)


COVERAGE_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "dynamic_coverage_model.pkl"
)


class UAVDynamicEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(self):

        super().__init__()

        # ==================================================
        # LOAD EARTHQUAKE DATA
        # ==================================================

        self.df = pd.read_csv(
            DATA_FILE
        )

        if len(self.df) == 0:

            raise ValueError(
                "earthquake_trajectory_dataset.csv is empty."
            )

        # ==================================================
        # LOAD COVERAGE MODEL
        # ==================================================

        if not COVERAGE_MODEL_FILE.exists():

            raise FileNotFoundError(
                "\nDynamic coverage model not found:\n"
                f"{COVERAGE_MODEL_FILE}\n\n"
                "Run:\n"
                "python ml/train_dynamic_coverage.py\n"
                "first."
            )

        self.coverage_model = joblib.load(
            COVERAGE_MODEL_FILE
        )

        # ==================================================
        # ACTION SPACE
        # ==================================================

        # 0 = Stay
        # 1 = Move -X
        # 2 = Move +X
        # 3 = Move -Y
        # 4 = Move +Y

        self.action_space = spaces.Discrete(5)

        # ==================================================
        # MOVEMENT
        # ==================================================

        self.step_size = 10.0

        # ==================================================
        # OBSERVATION
        # ==================================================

        # [uav_x,
        #  uav_y,
        #  uav_z,
        #  coverage,
        #  mean_received_power,
        #  num_users]

        self.observation_space = spaces.Box(

            low=np.array(
                [
                    -100.0,
                    -100.0,
                    0.0,
                    0.0,
                    -200.0,
                    1.0
                ],
                dtype=np.float32
            ),

            high=np.array(
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    0.0,
                    1000.0
                ],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        # ==================================================
        # CREATE TRAJECTORIES
        # ==================================================

        self.trajectories = []

        grouped = self.df.groupby(
            [
                "state_id",
                "trajectory_type"
            ],
            sort=False
        )

        for _, group in grouped:

            group = (
                group
                .sort_values("t")
                .reset_index(drop=True)
            )

            self.trajectories.append(
                group
            )

        self.num_trajectories = len(
            self.trajectories
        )

        # ==================================================
        # CURRENT ENVIRONMENT STATE
        # ==================================================

        self.current_trajectory = None

        self.current_step = None

        self.uav_position = None

        self.current_coverage = None

        self.current_power = None

        self.num_users = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        # Pick random earthquake trajectory

        trajectory_index = (
            self.np_random.integers(
                0,
                self.num_trajectories
            )
        )

        self.current_trajectory = (
            self.trajectories[
                trajectory_index
            ]
        )

        # Start at t = 0

        self.current_step = 0

        row = (
            self.current_trajectory
            .iloc[self.current_step]
        )

        # ==================================================
        # UAV INITIAL POSITION
        # ==================================================

        self.uav_position = np.array(
            [
                float(row["uav_x"]),
                float(row["uav_y"]),
                float(row["uav_z"])
            ],
            dtype=np.float32
        )

        # ==================================================
        # NUMBER OF USERS
        # ==================================================

        self.num_users = int(
            row["num_users"]
        )

        # ==================================================
        # INITIAL RECEIVED POWER
        # ==================================================

        users = self.current_trajectory[
            self.current_trajectory["t"]
            == row["t"]
        ]

        valid_power = (
            users["received_power_dbm"]
            .replace(
                -200,
                np.nan
            )
            .dropna()
        )

        if len(valid_power) > 0:

            self.current_power = float(
                valid_power.mean()
            )

        else:

            self.current_power = -200.0

        # ==================================================
        # INITIAL COVERAGE
        # ==================================================

        self.current_coverage = float(
            row["coverage_percentage"]
        )

        observation = (
            self._make_observation()
        )

        info = {

            "state_id":
                int(row["state_id"]),

            "trajectory_type":
                row["trajectory_type"],

            "t":
                int(row["t"]),

            "coverage":
                self.current_coverage
        }

        return observation, info

    # ======================================================
    # OBSERVATION
    # ======================================================

    def _make_observation(self):

        return np.array(
            [
                self.uav_position[0],
                self.uav_position[1],
                self.uav_position[2],

                self.current_coverage,

                self.current_power,

                self.num_users
            ],
            dtype=np.float32
        )

    # ======================================================
    # ACTION → MOVEMENT
    # ======================================================

    def _apply_action(
        self,
        action
    ):

        new_position = (
            self.uav_position.copy()
        )

        if action == 0:

            # Stay
            pass

        elif action == 1:

            # Move left / -X

            new_position[0] -= (
                self.step_size
            )

        elif action == 2:

            # Move right / +X

            new_position[0] += (
                self.step_size
            )

        elif action == 3:

            # Move down / -Y

            new_position[1] -= (
                self.step_size
            )

        elif action == 4:

            # Move up / +Y

            new_position[1] += (
                self.step_size
            )

        else:

            raise ValueError(
                f"Invalid action: {action}"
            )

        return new_position

    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        action
    ):

        action = int(action)

        # ==================================================
        # CURRENT STATE
        # ==================================================

        old_position = (
            self.uav_position.copy()
        )

        old_coverage = (
            self.current_coverage
        )

        # ==================================================
        # APPLY PPO ACTION
        # ==================================================

        new_position = (
            self._apply_action(
                action
            )
        )

        # ==================================================
        # POSITION LIMITS
        # ==================================================

        new_position[0] = np.clip(
            new_position[0],
            -50.0,
            70.0
        )

        new_position[1] = np.clip(
            new_position[1],
            -50.0,
            70.0
        )

        new_position[2] = np.clip(
            new_position[2],
            10.0,
            50.0
        )

        self.uav_position = (
            new_position
        )

        # ==================================================
        # PREDICT COVERAGE
        # ==================================================

        prediction_input = pd.DataFrame(
            [
                {
                    "uav_x":
                        self.uav_position[0],

                    "uav_y":
                        self.uav_position[1],

                    "uav_z":
                        self.uav_position[2],

                    "num_users":
                        self.num_users,

                    "mean_received_power":
                        self.current_power
                }
            ]
        )

        predicted_coverage = float(
            self.coverage_model.predict(
                prediction_input
            )[0]
        )

        predicted_coverage = float(
            np.clip(
                predicted_coverage,
                0.0,
                100.0
            )
        )

        self.current_coverage = (
            predicted_coverage
        )

        # ==================================================
        # REWARD
        # ==================================================

        coverage_change = (
            self.current_coverage
            - old_coverage
        )

        # Small cost for movement

        if action == 0:

            movement_penalty = 0.0

        else:

            movement_penalty = 0.05

        reward = (
            coverage_change
            - movement_penalty
        )

        # ==================================================
        # ADVANCE EARTHQUAKE TIME
        # ==================================================

        self.current_step += 1

        terminated = (
            self.current_step
            >= len(
                self.current_trajectory
            ) - 1
        )

        truncated = False

        # ==================================================
        # INFO
        # ==================================================

        info = {

            "action":
                action,

            "old_position":
                old_position.copy(),

            "new_position":
                self.uav_position.copy(),

            "old_coverage":
                old_coverage,

            "new_coverage":
                self.current_coverage,

            "coverage_change":
                coverage_change,

            "reward":
                reward,

            "trajectory_type":
                self.current_trajectory.iloc[
                    self.current_step
                ]["trajectory_type"]
        }

        return (

            self._make_observation(),

            float(reward),

            terminated,

            truncated,

            info
        )

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        pass