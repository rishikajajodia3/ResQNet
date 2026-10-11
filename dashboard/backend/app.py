"""ResQNet dashboard API.

Thin read layer over the real pipeline output files plus a "Run" endpoint
that shells out to the existing ``scripts/run_full_pipeline.py`` orchestrator.

Run locally:

    cd ~/ResQNet
    source .venv/bin/activate
    python dashboard/backend/app.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Make sibling modules importable when run as `python dashboard/backend/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import runners  # noqa: E402
import schemas  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

app = FastAPI(title="ResQNet Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.DEV_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found. Run the pipeline first.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
        raise HTTPException(status_code=500, detail=f"{path.name} is not valid JSON: {exc}")


def _maybe_json(path: Path) -> Optional[Any]:
    return json.loads(path.read_text()) if path.exists() else None


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


# ----------------------------------------------------------------------
# health / meta
# ----------------------------------------------------------------------
@app.get("/api/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    ns3 = config.resolve_ns3_path()
    return schemas.HealthResponse(
        status="ok",
        data_files={
            "uav_positions.json": config.UAV_POSITIONS_FILE.exists(),
            "ppo_trajectory.json": config.PPO_TRAJECTORY_FILE.exists(),
            "full_pipeline_results.json": config.FULL_PIPELINE_RESULTS_FILE.exists(),
            "experiment_summary.json": config.EXPERIMENT_SUMMARY_FILE.exists(),
            "experiment_metadata.json": config.EXPERIMENT_METADATA_FILE.exists(),
        },
        scene_assets={
            "scene_meta.json": config.SCENE_META_FILE.exists(),
            "city_damaged.glb": config.GLB_DAMAGED.exists(),
            "city_baseline.glb": config.GLB_BASELINE.exists(),
        },
        ns3_path=str(ns3),
        ns3_path_exists=ns3.exists(),
    )


# ----------------------------------------------------------------------
# raw pass-through endpoints
# ----------------------------------------------------------------------
@app.get("/api/ppo/trajectory")
def ppo_trajectory() -> Any:
    """Real, unmodified contents of data/ppo_trajectory.json."""
    return _read_json(config.PPO_TRAJECTORY_FILE)


@app.get("/api/ns3/results")
def ns3_results() -> Any:
    """Real, unmodified contents of data/full_pipeline_results.json."""
    return _read_json(config.FULL_PIPELINE_RESULTS_FILE)


@app.get("/api/experiment")
def experiment() -> dict:
    """The scripted 'challenging' experiment: summary + metadata + per-step CSV.

    This is a separate, pre-recorded episode (different start position) from
    the live ppo_trajectory.json.
    """
    return {
        "summary": _maybe_json(config.EXPERIMENT_SUMMARY_FILE),
        "metadata": _maybe_json(config.EXPERIMENT_METADATA_FILE),
        "trajectory": _read_csv_rows(config.EXPERIMENT_TRAJECTORY_CSV),
    }


# ----------------------------------------------------------------------
# latest complete pipeline result
# ----------------------------------------------------------------------
@app.get("/api/run/latest")
def run_latest() -> dict:
    """The most recent complete pipeline result.

    Uses the newest dashboard-triggered snapshot if one exists, otherwise the
    live data/*.json files.
    """
    snap = runners.manager.latest_succeeded_snapshot()
    if snap is not None:
        source = snap.name
        uav = _maybe_json(snap / "uav_positions.json")
        ppo = _maybe_json(snap / "ppo_trajectory.json")
        ns3 = _maybe_json(snap / "full_pipeline_results.json")
        generated_at = None
        meta = _maybe_json(snap / "meta.json")
        if meta:
            generated_at = meta.get("finished_at")
    else:
        source = "live"
        uav = _maybe_json(config.UAV_POSITIONS_FILE)
        ppo = _maybe_json(config.PPO_TRAJECTORY_FILE)
        ns3 = _maybe_json(config.FULL_PIPELINE_RESULTS_FILE)
        generated_at = None

    if ppo is None:
        raise HTTPException(status_code=404, detail="No pipeline results available yet.")

    return {
        "source": source,
        "generated_at": generated_at,
        "uav_positions": uav,
        "ppo_trajectory": ppo,
        "ns3_results": ns3,
    }


# ----------------------------------------------------------------------
# datasets (read only if they contain usable numeric values)
# ----------------------------------------------------------------------
@app.get("/api/datasets/grid-sweep")
def grid_sweep() -> dict:
    data = _maybe_json(config.GRID_SWEEP_JSON)
    if not data:
        raise HTTPException(status_code=404, detail="grid_sweep_dataset.json not found.")
    points: list[dict] = []
    for env in data.get("environments", []):
        for run in env.get("runs", []):
            if not isinstance(run.get("coverage_percentage"), (int, float)):
                continue
            points.append(
                {
                    "env_id": env.get("env_id"),
                    "uav_x": run.get("uav_x"),
                    "uav_y": run.get("uav_y"),
                    "uav_z": run.get("uav_z"),
                    "coverage_percentage": run.get("coverage_percentage"),
                    "mean_path_gain_db": run.get("mean_path_gain_db"),
                    "no_coverage_percentage": run.get("no_coverage_percentage"),
                }
            )
    return {
        "schema_version": data.get("schema_version"),
        "notes": data.get("notes"),
        "provenance": "Sionna RT propagation sweep. Scene not recorded in the "
        "dataset; generated by sionna/sionna_pipeline.py which falls back to a "
        "placeholder street-canyon scene when run from the repo root. Treat as a "
        "propagation study, not the damaged-city model.",
        "num_points": len(points),
        "points": points,
    }


@app.get("/api/datasets/earthquake")
def earthquake() -> dict:
    data = _maybe_json(config.EARTHQUAKE_JSON)
    if not data:
        raise HTTPException(status_code=404, detail="earthquake_trajectory_dataset.json not found.")
    states: list[dict] = []
    for st in data.get("disaster_states", []):
        steps = [
            {
                "t": ts.get("t"),
                "uav_position": ts.get("uav_position"),
                "coverage_percentage": ts.get("coverage_percentage"),
                "mean_path_gain_db": ts.get("mean_path_gain_db"),
                "no_coverage_percentage": ts.get("no_coverage_percentage"),
            }
            for ts in st.get("timesteps", [])
        ]
        states.append(
            {
                "state_id": st.get("state_id"),
                "trajectory_type": st.get("trajectory_type"),
                "cluster_centers": st.get("cluster_centers"),
                "num_users": len(st.get("fixed_users", [])),
                "timesteps": steps,
            }
        )
    return {
        "schema_version": data.get("schema_version"),
        "notes": data.get("notes"),
        "provenance": "Generated on a placeholder scene (simple_street_canyon) per "
        "the dataset's own notes - NOT the city_damaged.xml mesh.",
        "stats": _maybe_json(config.EARTHQUAKE_STATS),
        "num_states": len(states),
        "states": states,
    }


# ----------------------------------------------------------------------
# 3D scene
# ----------------------------------------------------------------------
@app.get("/api/scene/meta")
def scene_meta() -> dict:
    meta = _maybe_json(config.SCENE_META_FILE)
    if not meta:
        raise HTTPException(
            status_code=404,
            detail="scene_meta.json not found. Run: python dashboard/build_assets.py",
        )
    # attach the live UAV / user coordinates so the frontend has one source
    uav = _maybe_json(config.UAV_POSITIONS_FILE)
    ppo = _maybe_json(config.PPO_TRAJECTORY_FILE)
    meta = dict(meta)
    meta["uav_positions_file"] = uav
    if ppo:
        steps = ppo.get("trajectory", [])
        num_uavs = ppo.get("num_uavs", 1)
        meta["num_uavs"] = num_uavs
        meta["scenario"] = ppo.get("scenario")
        meta["interference_model"] = ppo.get("interference_model")
        # single-UAV mirror (unchanged)
        meta["trajectory"] = [s["position"] for s in steps]
        meta["final_position"] = ppo.get("final_position")
        meta["initial_position"] = ppo.get("initial_position")
        # per-UAV trajectories: [uav_idx][step] -> [x,y,z]
        per_uav = []
        for k in range(num_uavs):
            per_uav.append([
                (s.get("uav_positions", [s["position"]])[k]
                 if k < len(s.get("uav_positions", [s["position"]]))
                 else s["position"])
                for s in steps
            ])
        meta["uav_trajectories"] = per_uav
        meta["initial_positions"] = ppo.get("initial_positions") or [ppo.get("initial_position")]
        meta["final_positions"] = ppo.get("final_positions") or [ppo.get("final_position")]
    return meta


@app.get("/api/scene/geometry")
def scene_geometry(variant: str = Query("damaged", pattern="^(damaged|baseline)$")) -> FileResponse:
    path = config.GLB_DAMAGED if variant == "damaged" else config.GLB_BASELINE
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} not found. Run: python dashboard/build_assets.py",
        )
    return FileResponse(path, media_type="model/gltf-binary", filename=path.name)


# ----------------------------------------------------------------------
# scenario / training / experiment metadata (multi-UAV)
# ----------------------------------------------------------------------
@app.get("/api/scenarios")
def scenarios() -> dict:
    cfg = _maybe_json(config.SCENARIOS_CONFIG)
    train_cfg = _maybe_json(config.TRAINING_CONFIG) or {}
    if not cfg:
        raise HTTPException(status_code=404, detail="configs/scenarios.json not found.")
    scen = cfg.get("scenarios", {})
    train = set(train_cfg.get("train", {}).get("scenarios", []))
    evl = set(train_cfg.get("eval", {}).get("scenarios", []))
    out = []
    for sid, s in scen.items():
        out.append({
            "id": sid,
            "name": s.get("name", sid),
            "scene": s.get("scene"),
            "user_distribution": s.get("user_distribution"),
            "purpose": s.get("purpose"),
            "note": s.get("note"),
            "split": "train" if sid in train else ("eval (held out)" if sid in evl else "unused"),
        })
    return {
        "scenarios": out,
        "user_counts": train_cfg.get("train", {}).get("user_counts", [10, 20, 30, 50]),
        "uav_counts": [1, 2, 3, 5],
        "interference_note": (
            "Multi-UAV runs model co-channel interference: SINR = serving power / "
            "(noise + sum of the other UAVs' ray-traced received powers). Assumption: "
            "all UAVs share one channel, full buffer, no scheduling. For 1 UAV, SINR = SNR."
        ),
    }


@app.get("/api/training/summary")
def training_summary() -> dict:
    if not config.MULTIUAV_MODEL.exists():
        raise HTTPException(
            status_code=404,
            detail="No multi-UAV model. Train it: python ml/train_multiuav.py",
        )
    rows = _read_csv_rows(config.MULTIUAV_TRAIN_LOG)
    for r in rows:
        for k in ("episode", "num_uavs", "num_users", "ep_len"):
            if r.get(k) not in (None, ""):
                r[k] = int(float(r[k]))
        for k in ("ep_reward", "coverage_pct", "connected_pct",
                  "mean_best_rss_dbm", "mean_best_sinr_db"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return {
        "config": _maybe_json(config.MULTIUAV_TRAIN_CONFIG),
        "episodes": rows,
        "num_episodes": len(rows),
        "rf_model": "surrogate / free-space (training). Sionna used only for validation.",
    }


@app.get("/api/experiments")
def experiments() -> dict:
    idx = _maybe_json(config.EXPERIMENTS_INDEX)
    return {"experiments": idx or [], "note": "All values are from real runs."}


@app.get("/api/experiment/{exp_id}")
def experiment_detail(exp_id: str) -> dict:
    safe = "".join(c for c in exp_id if c.isalnum() or c in "-_")
    d = config.EXPERIMENTS_DIR / safe
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {exp_id}")
    out: dict = {"experiment_id": safe}
    for name in ("config", "baselines", "ppo_trajectory", "full_pipeline_results"):
        out[name] = _maybe_json(d / f"{name}.json")
    return out


# ----------------------------------------------------------------------
# run control
# ----------------------------------------------------------------------
@app.post("/api/run", response_model=schemas.RunAccepted, status_code=202)
def start_run(req: schemas.RunRequest) -> schemas.RunAccepted:
    if req.steps < 1 or req.steps > 50:
        raise HTTPException(status_code=422, detail="steps must be between 1 and 50.")
    if req.num_uavs is not None and not (1 <= req.num_uavs <= 5):
        raise HTTPException(status_code=422, detail="num_uavs must be between 1 and 5.")
    if req.num_users is not None and not (1 <= req.num_users <= 200):
        raise HTTPException(status_code=422, detail="num_users must be between 1 and 200.")
    if req.scenario is not None:
        known = (_maybe_json(config.SCENARIOS_CONFIG) or {}).get("scenarios", {})
        if req.scenario not in known:
            raise HTTPException(status_code=422, detail=f"Unknown scenario: {req.scenario}")
    try:
        rec = runners.manager.start(
            mode=req.mode, steps=req.steps, scenario=req.scenario,
            num_uavs=req.num_uavs, num_users=req.num_users,
            interference_model=req.interference_model,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return schemas.RunAccepted(run_id=rec.run_id, status=rec.status)


@app.get("/api/run/{run_id}", response_model=schemas.RunInfo)
def run_status(run_id: str) -> schemas.RunInfo:
    rec = runners.manager.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return schemas.RunInfo(**rec.as_dict())


@app.get("/api/runs", response_model=list[schemas.RunInfo])
def list_runs() -> list[schemas.RunInfo]:
    return [schemas.RunInfo(**r) for r in runners.manager.list()]


# ----------------------------------------------------------------------
# static frontend (optional - only if built)
# ----------------------------------------------------------------------
if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIST), html=True), name="frontend")


def main() -> None:
    import uvicorn

    print(f"ResQNet dashboard API on http://{config.HOST}:{config.PORT}")
    print(f"  NS-3 path : {config.resolve_ns3_path()}")
    print(f"  frontend  : {'built' if config.FRONTEND_DIST.exists() else 'dev server (npm run dev)'}")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
