import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_coverage.pkl"
)


FEATURE_COLUMNS = [

    "uav_x",
    "uav_y",
    "uav_z",

    "num_users",

    "mean_user_x",
    "mean_user_y",

    "std_user_x",
    "std_user_y",

    "min_user_x",
    "max_user_x",

    "min_user_y",
    "max_user_y",

    "mean_received_power",
    "min_received_power",
    "max_received_power",

    "mean_path_gain",
    "mean_delay",
    "mean_num_paths"
]


TARGET_COLUMN = "coverage_percentage"


def train_random_forest(X, y):

    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1
    )

    model.fit(X, y)

    return model


def save_model(model):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print()
    print("Model saved:")
    print(MODEL_PATH)


def load_model():

    return joblib.load(
        MODEL_PATH
    )