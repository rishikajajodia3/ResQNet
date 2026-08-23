import numpy as np
import pandas as pd

from pathlib import Path

from static_model import (
    load_model,
    FEATURE_COLUMNS
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_best_deployment(df):

    model = load_model()

    X = df[FEATURE_COLUMNS]

    predicted_coverage = model.predict(X)

    df = df.copy()

    df["predicted_coverage"] = predicted_coverage

    best_index = (
        df["predicted_coverage"]
        .idxmax()
    )

    best = df.loc[best_index]

    return best, df


def print_deployment(best):

    print()
    print("=" * 50)
    print("BEST UAV DEPLOYMENT")
    print("=" * 50)

    print()

    print(
        "Scenario:",
        int(best["scenario_id"])
    )

    print(
        "Number of UAVs:",
        int(best["num_uavs"])
    )

    print()

    print(
        "UAV 1:",
        (
            best["uav1_x"],
            best["uav1_y"],
            best["uav1_z"]
        )
    )

    if best["num_uavs"] >= 2:

        print(
            "UAV 2:",
            (
                best["uav2_x"],
                best["uav2_y"],
                best["uav2_z"]
            )
        )

    print()

    print(
        "Predicted coverage:",
        f"{best['predicted_coverage']:.2f}%"
    )

    print(
        "Actual coverage:",
        f"{best['coverage_percentage']:.2f}%"
    )


if __name__ == "__main__":

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ml_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    best, results = find_best_deployment(df)

    print_deployment(best)

    print()
    print("All candidate deployments:")
    print(
        results[
            [
                "scenario_id",
                "num_uavs",
                "coverage_percentage",
                "predicted_coverage"
            ]
        ].to_string(index=False)
    )