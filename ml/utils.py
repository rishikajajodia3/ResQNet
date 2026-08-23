from pathlib import Path
import json


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_directories():
    """
    Make sure all required project directories exist.
    """

    directories = [
        DATA_DIR,
        PROCESSED_DIR,
        SYNTHETIC_DIR,
        MODELS_DIR,
        OUTPUTS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    """
    Save Python dictionary/list as JSON.
    """

    filepath = Path(filepath)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    """
    Load JSON file.
    """

    filepath = Path(filepath)

    with open(filepath, "r") as f:
        return json.load(f)