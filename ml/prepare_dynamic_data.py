import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "sionna"
    / "earthquake_trajectory_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dynamic_transitions.csv"
)


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("EARTHQUAKE DYNAMIC DATA")
print("=" * 60)

print("Raw rows:", len(df))

print()
print("Unique trajectories:",
      df[["state_id", "trajectory_type"]]
      .drop_duplicates()
      .shape[0])

print(
    "Unique timesteps:",
    df["t"].nunique()
)


# =========================================================
# CLEAN NUMERIC COLUMNS
# =========================================================

numeric_columns = [
    "uav_x",
    "uav_y",
    "uav_z",
    "num_users",
    "coverage_percentage",
    "mean_path_gain_db",
    "user_x",
    "user_y",
    "received_power_dbm",
    "shortest_delay_sec"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# INVALID RADIO VALUES
# =========================================================

df["received_power_dbm"] = (
    df["received_power_dbm"]
    .fillna(-200)
)

df["shortest_delay_sec"] = (
    df["shortest_delay_sec"]
    .fillna(0)
)


# =========================================================
# ONE ROW PER UAV STATE
#
# A state has multiple users.
#
# Example:
#
# state_id = 1
# trajectory = toward_cluster
# t = 0
#
#        user 1
#        user 2
#        ...
#        user 22
#
# We aggregate those into ONE state.
# =========================================================

state_df = (

    df

    .groupby(
        [
            "state_id",
            "trajectory_type",
            "t",
            "uav_x",
            "uav_y",
            "uav_z",
            "num_users",
            "coverage_percentage",
            "mean_path_gain_db"
        ],

        as_index=False
    )

    .agg(

        mean_received_power=(
            "received_power_dbm",
            "mean"
        ),

        min_received_power=(
            "received_power_dbm",
            "min"
        ),

        max_received_power=(
            "received_power_dbm",
            "max"
        ),

        mean_delay=(
            "shortest_delay_sec",
            "mean"
        ),

        mean_user_x=(
            "user_x",
            "mean"
        ),

        mean_user_y=(
            "user_y",
            "mean"
        ),

        std_user_x=(
            "user_x",
            "std"
        ),

        std_user_y=(
            "user_y",
            "std"
        ),

        min_user_x=(
            "user_x",
            "min"
        ),

        max_user_x=(
            "user_x",
            "max"
        ),

        min_user_y=(
            "user_y",
            "min"
        ),

        max_user_y=(
            "user_y",
            "max"
        )
    )
)


state_df["std_user_x"] = (
    state_df["std_user_x"]
    .fillna(0)
)

state_df["std_user_y"] = (
    state_df["std_user_y"]
    .fillna(0)
)


# =========================================================
# SORT CORRECTLY
#
# VERY IMPORTANT:
#
# state_id + trajectory_type define a trajectory
# t defines the position inside that trajectory
# =========================================================

state_df = state_df.sort_values(
    [
        "state_id",
        "trajectory_type",
        "t"
    ]
).reset_index(drop=True)


# =========================================================
# CREATE NEXT STATE
# =========================================================

trajectory_group = [
    "state_id",
    "trajectory_type"
]


state_df["next_uav_x"] = (
    state_df
    .groupby(trajectory_group)["uav_x"]
    .shift(-1)
)

state_df["next_uav_y"] = (
    state_df
    .groupby(trajectory_group)["uav_y"]
    .shift(-1)
)

state_df["next_uav_z"] = (
    state_df
    .groupby(trajectory_group)["uav_z"]
    .shift(-1)
)

state_df["next_coverage"] = (
    state_df
    .groupby(trajectory_group)["coverage_percentage"]
    .shift(-1)
)

state_df["next_t"] = (
    state_df
    .groupby(trajectory_group)["t"]
    .shift(-1)
)


# =========================================================
# REMOVE FINAL TIMESTEP
#
# t=7 has no t=8
# therefore it cannot form a transition
# =========================================================

state_df = state_df.dropna(
    subset=[
        "next_uav_x",
        "next_uav_y",
        "next_uav_z",
        "next_coverage"
    ]
).copy()


# =========================================================
# MOVEMENT
# =========================================================

state_df["dx"] = (
    state_df["next_uav_x"]
    - state_df["uav_x"]
)

state_df["dy"] = (
    state_df["next_uav_y"]
    - state_df["uav_y"]
)

state_df["dz"] = (
    state_df["next_uav_z"]
    - state_df["uav_z"]
)


# =========================================================
# ACTION
#
# 0 = stay
# 1 = left
# 2 = right
# 3 = down
# 4 = up
# =========================================================

def determine_action(row):

    dx = row["dx"]
    dy = row["dy"]

    tolerance = 1e-4

    # No horizontal movement
    if (
        abs(dx) < tolerance
        and abs(dy) < tolerance
    ):
        return 0

    # Horizontal movement dominates
    if abs(dx) >= abs(dy):

        if dx < 0:
            return 1

        return 2

    # Vertical movement dominates
    else:

        if dy < 0:
            return 3

        return 4


state_df["action"] = (
    state_df.apply(
        determine_action,
        axis=1
    )
)


# =========================================================
# REWARD
# =========================================================

state_df["reward"] = (
    state_df["next_coverage"]
    - state_df["coverage_percentage"]
)


# =========================================================
# VERIFY TRANSITIONS
# =========================================================

print()
print("=" * 60)
print("DYNAMIC TRANSITIONS")
print("=" * 60)

print(
    "State-level rows:",
    len(state_df)
)

print(
    "Unique trajectories:",
    state_df[
        ["state_id", "trajectory_type"]
    ]
    .drop_duplicates()
    .shape[0]
)

print()

print("Action distribution:")

print(
    state_df["action"]
    .value_counts()
    .sort_index()
)

print()

print("Reward statistics:")

print(
    "Mean:",
    state_df["reward"].mean()
)

print(
    "Min:",
    state_df["reward"].min()
)

print(
    "Max:",
    state_df["reward"].max()
)


# =========================================================
# EXAMPLE TRANSITIONS
# =========================================================

print()
print("=" * 60)
print("SAMPLE TRANSITIONS")
print("=" * 60)

print(

    state_df[
        [
            "state_id",
            "trajectory_type",
            "t",
            "uav_x",
            "uav_y",
            "uav_z",
            "action",
            "coverage_percentage",
            "next_coverage",
            "reward"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

state_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("SAVED")
print("=" * 60)

print(OUTPUT_FILE)