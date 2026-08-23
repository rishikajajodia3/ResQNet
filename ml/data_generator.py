import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grid_sweep_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "static_ml_dataset.csv"
)


def create_static_dataset():

    print("=" * 60)
    print("CREATING STATIC ML DATASET")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print("Raw rows:", len(df))

    # ---------------------------------------------------------
    # Remove completely empty columns
    # ---------------------------------------------------------

    df = df.dropna(axis=1, how="all")

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------

    required = [
        "env_id",
        "run_id",
        "uav_x",
        "uav_y",
        "uav_z",
        "num_users",
        "coverage_percentage",
        "user_x",
        "user_y",
        "user_z",
        "received_power_dbm",
        "path_gain_linear",
        "shortest_delay_sec",
        "num_paths"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Replace invalid values
    # ---------------------------------------------------------

    df["received_power_dbm"] = (
        pd.to_numeric(
            df["received_power_dbm"],
            errors="coerce"
        )
    )

    df["path_gain_linear"] = (
        pd.to_numeric(
            df["path_gain_linear"],
            errors="coerce"
        )
    )

    df["shortest_delay_sec"] = (
        pd.to_numeric(
            df["shortest_delay_sec"],
            errors="coerce"
        )
    )

    df["num_paths"] = (
        pd.to_numeric(
            df["num_paths"],
            errors="coerce"
        )
    )

    # -200 represents no received signal
    df["received_power_dbm"] = (
        df["received_power_dbm"]
        .fillna(-200)
    )

    df["path_gain_linear"] = (
        df["path_gain_linear"]
        .fillna(0)
    )

    df["shortest_delay_sec"] = (
        df["shortest_delay_sec"]
        .fillna(0)
    )

    df["num_paths"] = (
        df["num_paths"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Aggregate each UAV configuration
    #
    # One configuration = env_id + run_id
    # ---------------------------------------------------------

    grouped = (
        df
        .groupby(
            [
                "env_id",
                "run_id",
                "uav_x",
                "uav_y",
                "uav_z",
                "num_users"
            ],
            as_index=False
        )
        .agg(

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
            ),

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

            mean_path_gain=(
                "path_gain_linear",
                "mean"
            ),

            mean_delay=(
                "shortest_delay_sec",
                "mean"
            ),

            mean_num_paths=(
                "num_paths",
                "mean"
            ),

            coverage_percentage=(
                "coverage_percentage",
                "first"
            )
        )
    )

    # std is NaN when only one user exists
    grouped["std_user_x"] = (
        grouped["std_user_x"]
        .fillna(0)
    )

    grouped["std_user_y"] = (
        grouped["std_user_y"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    grouped.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Configurations:", len(grouped))
    print("Features:", len(grouped.columns))

    print()
    print("Coverage:")
    print(
        "Min:",
        grouped["coverage_percentage"].min()
    )

    print(
        "Max:",
        grouped["coverage_percentage"].max()
    )

    print(
        "Mean:",
        grouped["coverage_percentage"].mean()
    )

    print()
    print("Columns:")
    print(list(grouped.columns))

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    create_static_dataset()