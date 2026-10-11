"""Background run manager for the ResQNet dashboard.

The "Run ResQNet" button calls :func:`RunManager.start`, which shells out to
the *existing* orchestrator ``scripts/run_full_pipeline.py``. No PPO / Sionna
/ NS-3 logic is reimplemented here - this module only:

  * makes sure the current ``ns3/uav-demo.cc`` is in the NS-3 scratch dir
    (a step the orchestrator itself does not perform),
  * runs the orchestrator in a background thread so HTTP stays responsive,
  * snapshots the produced ``data/*.json`` files into ``data/runs/<id>/``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


SNAPSHOT_FILES = [
    config.UAV_POSITIONS_FILE,
    config.PPO_TRAJECTORY_FILE,
    config.FULL_PIPELINE_RESULTS_FILE,
]


class RunRecord:
    def __init__(self, run_id: str, mode: str, steps: int,
                 scenario=None, num_uavs=None, num_users=None,
                 interference_model="co_channel"):
        self.run_id = run_id
        self.status = "queued"
        self.mode = mode
        self.steps = steps
        self.scenario = scenario
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.interference_model = interference_model
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.returncode: Optional[int] = None
        self.error: Optional[str] = None
        self.log: deque[str] = deque(maxlen=400)
        self.snapshot_dir: Optional[Path] = None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "steps": self.steps,
            "scenario": self.scenario,
            "num_uavs": self.num_uavs,
            "num_users": self.num_users,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
            "log_tail": list(self.log)[-60:],
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
        }


class RunManager:
    """Serialises pipeline runs (one at a time) and keeps their history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, RunRecord] = {}
        self._order: list[str] = []
        self._active: Optional[str] = None

    # -- queries -------------------------------------------------------
    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._records.get(run_id)

    def list(self) -> list[dict]:
        return [self._records[r].as_dict() for r in reversed(self._order)]

    def latest_succeeded_snapshot(self) -> Optional[Path]:
        for rid in reversed(self._order):
            rec = self._records[rid]
            if rec.status == "succeeded" and rec.snapshot_dir:
                return rec.snapshot_dir
        return None

    def is_busy(self) -> bool:
        return self._active is not None

    # -- start --------------------------------------------------------
    def start(self, mode: str, steps: int, scenario=None, num_uavs=None,
              num_users=None, interference_model="co_channel") -> RunRecord:
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    f"A pipeline run ({self._active}) is already in progress."
                )
            rid = _run_id()
            # guarantee uniqueness even for sub-second repeats
            suffix = 1
            base = rid
            while rid in self._records:
                suffix += 1
                rid = f"{base}_{suffix}"
            rec = RunRecord(rid, mode, steps, scenario, num_uavs, num_users,
                            interference_model)
            self._records[rid] = rec
            self._order.append(rid)
            self._active = rid

        thread = threading.Thread(
            target=self._execute, args=(rec,), name=f"resqnet-{rec.run_id}", daemon=True
        )
        thread.start()
        return rec

    # -- worker -----------------------------------------------------
    def _execute(self, rec: RunRecord) -> None:
        rec.status = "running"
        rec.started_at = _utc_now_iso()
        try:
            ns3_path = config.resolve_ns3_path()
            if not ns3_path.exists():
                raise FileNotFoundError(
                    f"NS-3 installation not found at {ns3_path}. "
                    "Set NS3_PATH to your ns-3 directory."
                )

            # Step the orchestrator does NOT do: sync the scenario file
            # into the NS-3 scratch directory.
            scratch_target = ns3_path / "scratch" / "uav-demo.cc"
            scratch_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config.NS3_SCRATCH_SOURCE, scratch_target)
            rec.log.append(f"[dashboard] synced ns3/uav-demo.cc -> {scratch_target}")

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(config.SIONNA_DIR), env.get("PYTHONPATH", "")]
            )
            env["NS3_PATH"] = str(ns3_path)

            cmd = [
                "python",
                str(config.RUN_PIPELINE_SCRIPT),
                "--mode",
                rec.mode,
                "--steps",
                str(rec.steps),
                "--ns3-path",
                str(ns3_path),
            ]
            if rec.scenario or rec.num_uavs or rec.num_users:
                cmd += [
                    "--scenario", str(rec.scenario or "blocked_urban"),
                    "--num-uavs", str(rec.num_uavs or 3),
                    "--num-users", str(rec.num_users or 30),
                    "--interference-model", rec.interference_model,
                ]
            rec.log.append("[dashboard] $ " + " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                cwd=str(config.PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                rec.log.append(line.rstrip("\n"))
            proc.wait()
            rec.returncode = proc.returncode

            if proc.returncode != 0:
                rec.status = "failed"
                rec.error = f"run_full_pipeline.py exited with code {proc.returncode}"
                return

            self._snapshot(rec)
            rec.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            rec.status = "failed"
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.log.append(f"[dashboard] ERROR {rec.error}")
        finally:
            rec.finished_at = _utc_now_iso()
            self._active = None

    def _snapshot(self, rec: RunRecord) -> None:
        snap = config.RUNS_DIR / rec.run_id
        snap.mkdir(parents=True, exist_ok=True)
        for src in SNAPSHOT_FILES:
            if src.exists():
                shutil.copy2(src, snap / src.name)
        (snap / "meta.json").write_text(
            _json_dump(
                {
                    "run_id": rec.run_id,
                    "mode": rec.mode,
                    "steps": rec.steps,
                    "started_at": rec.started_at,
                    "finished_at": _utc_now_iso(),
                    "returncode": rec.returncode,
                    "files": [f.name for f in SNAPSHOT_FILES if f.exists()],
                }
            )
        )
        rec.snapshot_dir = snap
        rec.log.append(f"[dashboard] snapshot -> {snap}")


def _json_dump(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2)


# Single shared instance used by the FastAPI app.
manager = RunManager()
