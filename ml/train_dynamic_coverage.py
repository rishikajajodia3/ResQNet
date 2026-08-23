import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "earthquake_trajectory_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "dynamic_coverage_model.pkl"
)


def main():

    print("=" * 60)
    print("TRAINING DYNAMIC COVERAGE MODEL")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    # ==================================================
    # AGGREGATE USER ROWS
    # ==================================================

    # One row = one UAV state / timestep

    state_df = (

        df.groupby(
            [
                "state_id",
                "trajectory_type",
                "t"
            ],
            as_index=False
        )
        .agg({

            "uav_x": "first",
            "uav_y": "first",
            "uav_z": "first",

            "num_users": "first",

            "coverage_percentage": "first",

            "received_power_dbm":
                lambda x:
                x.replace(
                    -200,
                    np.nan
                ).mean()
        })
    )

    state_df[
        "mean_received_power"
    ] = state_df[
        "received_power_dbm"
    ].fillna(-200)

    # ==================================================
    # FEATURES
    # ==================================================

    FEATURES = [

        "uav_x",
        "uav_y",
        "uav_z",
        "num_users",
        "mean_received_power"

    ]

    TARGET = "coverage_percentage"

    X = state_df[FEATURES]

    y = state_df[TARGET]

    # ==================================================
    # TRAIN TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )

    print()
    print("State samples:", len(state_df))
    print("Training:", len(X_train))
    print("Testing:", len(X_test))

    # ==================================================
    # RANDOM FOREST
    # ==================================================

    model = RandomForestRegressor(

        n_estimators=200,

        max_depth=8,

        min_samples_leaf=2,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ==================================================
    # EVALUATION
    # ==================================================

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("DYNAMIC COVERAGE MODEL")
    print("=" * 60)

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    # ==================================================
    # SAVE
    # ==================================================

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print()
    print("Model saved:")
    print(MODEL_FILE)


if __name__ == "__main__":

    main()