"""Closed-loop control sessions for the dashboard.

Runs ``ml/closed_loop_control.py`` as a subprocess (one session at a time) and
serves the state file that process writes after every phase of every control
interval. Pause / resume / stop go through ``control.json`` in the run
directory, which the controller reads between phases, so a running ns-3 or
Sionna call always finishes first. Reset stops the process (killing it if it
does not stop within a few seconds) and clears the session; the run directory
on disk is kept.

Run directories live in ``RESQNET_CONTROL_DIR`` (default
``~/.cache/resqnet_control``), outside the repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

CONTROLLER_SCRIPT = config.PROJECT_ROOT / "ml" / "closed_loop_control.py"
CONTROL_CONFIG = config.PROJECT_ROOT / "configs" / "closed_loop_control.json"
CONTROL_DIR = Path(os.environ.get("RESQNET_CONTROL_DIR", Path.home() / ".cache" / "resqnet_control")).expanduser()


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class ControlSession:
    def __init__(self, run_id: str, out: Path, proc: subprocess.Popen, overrides: dict):
        self.run_id, self.out, self.proc, self.overrides = run_id, out, proc, overrides
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.command = "run"


class ControlManager:
    def __init__(self, python: str = sys.executable, script: Path = CONTROLLER_SCRIPT,
                 config_path: Path = CONTROL_CONFIG, base_dir: Path = CONTROL_DIR):
        self.python, self.script, self.config_path, self.base_dir = python, script, config_path, base_dir
        self._lock = threading.Lock()
        self.session: Optional[ControlSession] = None

    # ---------------- commands -------------------------------------------
    def start(self, overrides: dict) -> dict:
        with self._lock:
            if self.session and self.session.proc.poll() is None:
                raise RuntimeError(f"control session {self.session.run_id} is still running; pause or reset it first")
            run_id = datetime.now(timezone.utc).strftime("ctl_%Y%m%dT%H%M%SZ")
            out = self.base_dir / run_id
            n = 1
            while out.exists():
                n += 1
                out = self.base_dir / f"{run_id}_{n}"
            out.mkdir(parents=True)
            _write_json(out / "control.json", {"command": "run"})
            with open(out / "stdout.log", "w") as log:   # the child keeps its own handle
                proc = subprocess.Popen(
                    [self.python, str(self.script), "--config", str(self.config_path),
                     "--override", json.dumps(overrides), "--out", str(out), "--run-id", out.name],
                    cwd=str(config.PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            self.session = ControlSession(out.name, out, proc, overrides)
            return {"run_id": out.name, "out_dir": str(out)}

    def _send(self, command: str) -> None:
        s = self.session
        if s is None:
            raise LookupError("no control session")
        if s.proc.poll() is not None:
            raise RuntimeError(f"control session {s.run_id} is not running (exit {s.proc.returncode})")
        _write_json(s.out / "control.json", {"command": command})
        s.command = command

    def pause(self) -> None:
        self._send("pause")

    def resume(self) -> None:
        self._send("run")

    def reset(self, timeout_s: float = 5.0) -> Optional[str]:
        with self._lock:
            s = self.session
            if s is None:
                return None
            if s.proc.poll() is None:
                _write_json(s.out / "control.json", {"command": "stop"})
                t0 = time.time()
                while s.proc.poll() is None and time.time() - t0 < timeout_s:
                    time.sleep(0.1)
                if s.proc.poll() is None:
                    s.proc.kill()
                    s.proc.wait(timeout=10)
            self.session = None
            return s.run_id

    # ---------------- state ----------------------------------------------
    def state(self) -> dict:
        s = self.session
        if s is None:
            return {"status": "idle", "session": None, "run_id": None}
        st = _read_json(s.out / "state.json") or {"status": "starting", "phase": "init", "version": 0,
                                                    "run_id": s.run_id}
        rc = s.proc.poll()
        alive = rc is None
        if not alive and st.get("status") in ("starting", "running", "paused"):
            st["status"] = "failed"
            st["error"] = st.get("error") or f"controller exited with code {rc}"
        if alive and s.command == "pause" and st.get("status") == "running":
            st["status"] = "pausing"   # pause takes effect at the next phase boundary
        tail = []
        try:
            tail = (s.out / "stdout.log").read_text().splitlines()[-40:]
        except OSError:
            pass
        st["session"] = {"run_id": s.run_id, "out_dir": str(s.out), "process_alive": alive, "returncode": rc,
                         "command": s.command, "started_at": s.started_at, "overrides": s.overrides,
                         "log_tail": tail}
        return st

    def runs(self) -> list[dict]:
        out = []
        if self.base_dir.exists():
            for d in sorted(self.base_dir.iterdir(), reverse=True):
                st = _read_json(d / "state.json") if d.is_dir() else None
                if st:
                    out.append({"run_id": d.name, "status": st.get("status"), "step": st.get("step"),
                                "channel_mode": st.get("channel_mode"),
                                "before": (st.get("before") or {}).get("metrics"),
                                "after": (st.get("after") or {}).get("metrics")})
        return out[:50]


manager = ControlManager()
