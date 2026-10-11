"""Central paths and configuration for the ResQNet dashboard backend.

This module contains no logic that touches PPO, Sionna or NS-3 - it only
records where the existing pipeline writes its real output files so the
API can read them.
"""

from __future__ import annotations

import os
from pathlib import Path

# dashboard/backend/config.py  ->  repo root is three parents up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- existing pipeline entry points (unchanged source) --------------------
RUN_PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_full_pipeline.py"
RUN_PPO_SCRIPT = PROJECT_ROOT / "ml" / "run_ppo.py"
NS3_SCRATCH_SOURCE = PROJECT_ROOT / "ns3" / "uav-demo.cc"
SIONNA_DIR = PROJECT_ROOT / "sionna"

# --- real pipeline output files -----------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
UAV_POSITIONS_FILE = DATA_DIR / "uav_positions.json"
PPO_TRAJECTORY_FILE = DATA_DIR / "ppo_trajectory.json"
FULL_PIPELINE_RESULTS_FILE = DATA_DIR / "full_pipeline_results.json"
EXPERIMENT_SUMMARY_FILE = DATA_DIR / "experiment_summary.json"
EXPERIMENT_METADATA_FILE = DATA_DIR / "experiment_metadata.json"
EXPERIMENT_TRAJECTORY_CSV = DATA_DIR / "experiment_challenging_trajectory.csv"

GRID_SWEEP_JSON = SIONNA_DIR / "grid_sweep_dataset.json"
EARTHQUAKE_JSON = SIONNA_DIR / "earthquake_trajectory_dataset.json"
EARTHQUAKE_STATS = SIONNA_DIR / "earthquake_stats_summary.json"

# --- multi-UAV / scenario system -------------------------------------
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCENARIOS_CONFIG = CONFIGS_DIR / "scenarios.json"
TRAINING_CONFIG = CONFIGS_DIR / "training.json"
MULTIUAV_MODEL_DIR = PROJECT_ROOT / "models" / "ppo_multiuav"
MULTIUAV_MODEL = MULTIUAV_MODEL_DIR / "model.zip"
MULTIUAV_TRAIN_LOG = MULTIUAV_MODEL_DIR / "train_log.csv"
MULTIUAV_TRAIN_CONFIG = MULTIUAV_MODEL_DIR / "train_config.json"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
EXPERIMENTS_INDEX = EXPERIMENTS_DIR / "index.json"

# --- dashboard-owned artefacts -----------------------------------------
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
ASSETS_DIR = DASHBOARD_DIR / "assets"
SCENE_META_FILE = ASSETS_DIR / "scene_meta.json"
GLB_DAMAGED = ASSETS_DIR / "city_damaged.glb"
GLB_BASELINE = ASSETS_DIR / "city_baseline.glb"
FRONTEND_DIST = DASHBOARD_DIR / "frontend" / "dist"

# Per-run snapshots created by dashboard-triggered runs. The live
# data/*.json files are never moved - these are copies.
RUNS_DIR = DATA_DIR / "runs"

# --- NS-3 installation -------------------------------------------------
# Resolved when a run is triggered: explicit env var wins, otherwise the
# verified local install is used.
DEFAULT_NS3_PATH = Path.home() / "ns-allinone-3.47" / "ns-3.47"


def resolve_ns3_path() -> Path:
    env = os.environ.get("NS3_PATH")
    return Path(env).expanduser() if env else DEFAULT_NS3_PATH


# --- server ----------------------------------------------------------
HOST = os.environ.get("RESQNET_DASH_HOST", "127.0.0.1")
PORT = int(os.environ.get("RESQNET_DASH_PORT", "8000"))

# Vite dev server origin, allowed for CORS during development.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
