import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from static_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    train_random_forest,
    save_model
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "static_ml_dataset.csv"
)


print("=" * 60)
print("STATIC UAV COVERAGE MODEL")
print("=" * 60)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("Total configurations:", len(df))


# ---------------------------------------------------------
# FEATURES / TARGET
# ---------------------------------------------------------

X = df[FEATURE_COLUMNS]

y = df[TARGET_COLUMN]


# ---------------------------------------------------------
# TRAIN / TEST
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

model = train_random_forest(
    X_train,
    y_train
)


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

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
print("RESULTS")
print("=" * 60)

print(
    f"MAE:  {mae:.4f} percentage points"
)

print(
    f"RMSE: {rmse:.4f} percentage points"
)

print(
    f"R²:   {r2:.4f}"
)


# ---------------------------------------------------------
# SAMPLE PREDICTIONS
# ---------------------------------------------------------

print()
print("Sample predictions")
print("-" * 60)

for actual, predicted in zip(
    y_test.iloc[:10],
    predictions[:10]
):

    print(
        f"Actual: {actual:.2f}% | "
        f"Predicted: {predicted:.2f}%"
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

save_model(model)