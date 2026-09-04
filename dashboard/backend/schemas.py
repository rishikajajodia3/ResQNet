"""Pydantic models for the shaped API responses.

Raw pass-through endpoints (``/api/ppo/trajectory``, ``/api/ns3/results``)
return the untouched file contents and are intentionally not modelled here.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    data_files: dict[str, bool]
    scene_assets: dict[str, bool]
    ns3_path: str
    ns3_path_exists: bool


class RunRequest(BaseModel):
    mode: Literal["sionna", "surrogate"] = "sionna"
    steps: int = 5
    # Multi-UAV / scenario. If all three are None the run uses the historical
    # single-UAV / 10-user pipeline unchanged.
    scenario: Optional[str] = None
    num_uavs: Optional[int] = None
    num_users: Optional[int] = None
    interference_model: Literal["co_channel", "orthogonal"] = "co_channel"


class RunInfo(BaseModel):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    mode: str
    steps: int
    scenario: Optional[str] = None
    num_uavs: Optional[int] = None
    num_users: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    log_tail: list[str] = []
    snapshot_dir: Optional[str] = None


class RunAccepted(BaseModel):
    run_id: str
    status: str


class LatestRun(BaseModel):
    source: str  # "live" or a run_id
    uav_positions: dict[str, Any]
    ppo_trajectory: dict[str, Any]
    ns3_results: Optional[dict[str, Any]] = None
    generated_at: Optional[str] = None
