"""Pydantic models for the shaped API responses.

Raw pass-through endpoints (``/api/ppo/trajectory``, ``/api/ns3/results``)
return the untouched file contents and are intentionally not modelled here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ObjectiveWeights(BaseModel):
    coverage: float = Field(ge=0)
    pdr: float = Field(ge=0)
    fairness: float = Field(ge=0)
    latency: float = Field(ge=0)


class ControlStartRequest(BaseModel):
    """Overrides for configs/closed_loop_control.json (all optional)."""
    channel_mode: Optional[Literal["obstruction", "distance"]] = None
    rf_mode: Optional[Literal["sionna", "surrogate"]] = None
    network: Optional[Literal["ns3", "mock"]] = None
    preset: Optional[Literal["balanced", "coverage_first", "latency_first"]] = None
    weights: Optional[ObjectiveWeights] = None
    max_steps: Optional[int] = Field(default=None, ge=1, le=40)
    initial_altitude_m: Optional[float] = Field(default=None, ge=15, le=60)
    traffic_seconds: Optional[float] = Field(default=None, ge=1, le=8)
    events: Optional[bool] = None   # False disables the scripted survivor events
    # 3D PPO policy (ml/ppo_controller.py); the default stays rule_based
    policy: Optional[Literal["rule_based", "ppo"]] = None
    ppo_checkpoint: Optional[str] = None   # models/ppo_multiuav_3d/<run>/<file>.zip only
    site: Optional[Literal["urban_core", "val_a"]] = None

    @field_validator("ppo_checkpoint")
    @classmethod
    def _checkpoint_in_models(cls, v: Optional[str]) -> Optional[str]:
        # SB3 checkpoints contain pickled objects: only load our own trained runs
        if v is None:
            return v
        root = (Path(__file__).resolve().parents[2] / "models" / "ppo_multiuav_3d").resolve()
        p = (root / v).resolve() if not Path(v).is_absolute() else Path(v).resolve()
        if root not in p.parents or p.suffix != ".zip" or not p.is_file():
            raise ValueError(f"ppo_checkpoint must be an existing .zip under {root}")
        return str(p)

    @model_validator(mode="after")
    def _ppo_needs_checkpoint(self):
        if self.policy == "ppo" and not self.ppo_checkpoint:
            raise ValueError('policy "ppo" needs ppo_checkpoint')
        return self

    def to_overrides(self) -> dict:
        o: dict[str, Any] = {}
        if self.policy is not None:
            o["policy"] = self.policy
        if self.ppo_checkpoint is not None:
            o["ppo"] = {"checkpoint": self.ppo_checkpoint}
        if self.site == "val_a":
            o["scenario_config"] = "configs/eval_val_a_3d.json"
        elif self.site == "urban_core":
            o["scenario_config"] = "configs/eval_urban_core.json"
        for k in ("channel_mode", "rf_mode", "network", "max_steps", "initial_altitude_m"):
            v = getattr(self, k)
            if v is not None:
                o[k] = v
        obj: dict[str, Any] = {}
        if self.preset is not None:
            obj["preset"] = self.preset
            obj["weights"] = None
        if self.weights is not None:
            obj["weights"] = self.weights.model_dump()
        if obj:
            o["objective"] = obj
        if self.traffic_seconds is not None:
            o["ns3"] = {"traffic_seconds": self.traffic_seconds}
        if self.events is False:
            o["events"] = []
        return o


class LatestRun(BaseModel):
    source: str  # "live" or a run_id
    uav_positions: dict[str, Any]
    ppo_trajectory: dict[str, Any]
    ns3_results: Optional[dict[str, Any]] = None
    generated_at: Optional[str] = None
