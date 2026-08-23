import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from static_model import (
    train_random_forest,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)


def main():

    print("=" * 60)
    print("RANDOM FOREST EVALUATION")
    print("=" * 60)

    df = pd.read_csv(
        DATASET_PATH
    )

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # Train only on training data
    model = train_random_forest(
        pd.concat(
            [
                X_train,
                y_train
            ],
            axis=1
        )
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print()
    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print()

    print(
        f"MAE:  {mae:.4f} percentage points"
    )

    print(
        f"RMSE: {rmse:.4f} percentage points"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    print()
    print("Sample predictions")
    print("-" * 60)

    for actual, predicted in zip(
        y_test.iloc[:10],
        predictions[:10]
    ):

        print(
            f"Actual: {actual:.2f}% "
            f"| Predicted: {predicted:.2f}%"
        )


if __name__ == "__main__":

    main()