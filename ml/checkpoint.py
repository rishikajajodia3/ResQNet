"""Checkpoint / resume support for long ns-3 evaluations.

* atomic_write_json   - write via a temp file + os.replace, so an interrupted
                        write never leaves a truncated result behind;
* fingerprint         - the settings, code and model hashes that determine an
                        experiment's results; a resumed run must match exactly;
* CachedNs3Runner     - every ns-3 simulation is stored under a content key
                        (binary hash + exact handoff + arguments) as soon as it
                        finishes, so a rerun skips everything already done;
                        includes a memory guard and a cooperative stop flag
                        (SIGINT / SIGTERM / low memory) that lets running jobs
                        finish and be saved before exiting.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence


class StopRequested(RuntimeError):
    """Raised when the run is being stopped safely (signal or low memory)."""


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: Path):
    """Return the parsed file, or None if it is missing or corrupt."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(settings: dict, files: dict[str, Path]) -> dict:
    """Settings plus the SHA-256 of every file that can change the results."""
    return {"settings": settings, "files": {k: file_sha256(p) for k, p in sorted(files.items())}}


def check_or_write_config(out: Path, fp: dict) -> bool:
    """Write config.json for a new run, or verify it for a resumed one.
    Returns True when resuming. Raises SystemExit listing any mismatch."""
    cfg = out / "config.json"
    old = read_json(cfg)
    if old is None:
        if cfg.exists():
            raise SystemExit(f"{cfg} is unreadable - refusing to resume into it")
        atomic_write_json(cfg, fp)
        return False
    diffs = []
    for section in ("settings", "files"):
        a, b = old.get(section, {}), fp[section]
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append(f"{section}.{k}: saved={a.get(k)!r} now={b.get(k)!r}")
    if diffs:
        raise SystemExit("existing results in " + str(out) + " were produced with a different "
                         "configuration - refusing to combine them:\n  " + "\n  ".join(diffs))
    return True


def mem_available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("inf")


class CachedNs3Runner:
    """Parallel, cached, memory-guarded ns-3 job runner.

    run_fn(handoff, args) -> dict performs one simulation (normally
    evaluate_network.run_ns3 bound to the built binary).
    """

    def __init__(self, cache_dir: Path, binary_hash: str, run_fn: Callable, workers: int = 4,
                 min_free_mb: float = 2048.0, low_mem_wait_s: float = 180.0,
                 mem_fn: Callable[[], float] = mem_available_mb, log: Callable[[str], None] = print,
                 fallback_dirs: Sequence[Path] = ()):
        self.cache_dir = Path(cache_dir)
        # read-only caches of other experiments (same key scheme, so a hit is the
        # identical binary + handoff + arguments); hits are copied into cache_dir
        self.fallback_dirs = [Path(d) for d in fallback_dirs]
        self.binary_hash = binary_hash
        self.run_fn = run_fn
        self.workers = workers
        self.min_free_mb = min_free_mb
        self.low_mem_wait_s = low_mem_wait_s
        self.mem_fn = mem_fn
        self.log = log
        self.stop = threading.Event()
        self.stop_reason = ""
        self.stats = {"cached": 0, "simulated": 0, "low_mem_waits": 0,
                      "min_mem_available_mb": float("inf")}
        self._lock = threading.Lock()

    # -- keys / cache --------------------------------------------------
    def key(self, handoff: dict, args: Sequence[str]) -> str:
        blob = json.dumps({"binary": self.binary_hash, "handoff": handoff, "args": list(args)},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, handoff: dict, args: Sequence[str]):
        k = self.key(handoff, args)
        for d in (self.cache_dir, *self.fallback_dirs):
            rec = read_json(d / k[:2] / f"{k}.json")
            if rec and rec.get("key") == k and isinstance(rec.get("result"), dict):
                if d != self.cache_dir:
                    atomic_write_json(self._path(k), rec)
                return rec["result"]
        return None

    # -- safety -------------------------------------------------------
    def request_stop(self, reason: str) -> None:
        if not self.stop.is_set():
            self.stop_reason = reason
            self.log(f"[checkpoint] stop requested: {reason} - finishing running jobs, "
                     "no new jobs will start")
        self.stop.set()

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.request_stop(f"signal {signal.Signals(signum).name}")
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        if hasattr(signal, "SIGHUP"):      # controlling terminal / wsl.exe wrapper went away
            signal.signal(signal.SIGHUP, handler)

    def _wait_for_memory(self) -> None:
        waited = 0.0
        while True:
            free = self.mem_fn()
            with self._lock:
                self.stats["min_mem_available_mb"] = min(self.stats["min_mem_available_mb"], free)
            if free >= self.min_free_mb:
                return
            if waited == 0.0:
                with self._lock:
                    self.stats["low_mem_waits"] += 1
                self.log(f"[checkpoint] MemAvailable {free:.0f} MB < {self.min_free_mb:.0f} MB - waiting")
            if waited >= self.low_mem_wait_s or self.stop.is_set():
                self.request_stop(f"memory stayed below {self.min_free_mb:.0f} MB for {waited:.0f} s")
                raise StopRequested(self.stop_reason)
            time.sleep(min(5.0, self.low_mem_wait_s))
            waited += min(5.0, self.low_mem_wait_s)

    # -- execution -------------------------------------------------------
    def run_one(self, handoff: dict, args: Sequence[str]) -> dict:
        hit = self.cached(handoff, args)
        if hit is not None:
            with self._lock:
                self.stats["cached"] += 1
            return hit
        if self.stop.is_set():
            raise StopRequested(self.stop_reason)
        self._wait_for_memory()
        if self.stop.is_set():
            raise StopRequested(self.stop_reason)
        result = self.run_fn(handoff, tuple(args))
        k = self.key(handoff, args)
        atomic_write_json(self._path(k), {"key": k, "args": list(args), "result": result,
                                          "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        with self._lock:
            self.stats["simulated"] += 1
        return result

    def map(self, jobs: Iterable[tuple[dict, Sequence[str]]],
            on_done: Callable[[int, dict], None] | None = None) -> list[dict]:
        """Run (handoff, args) jobs; results in input order. Cached jobs are
        answered without simulating. On a stop request, pending jobs are
        cancelled, running ones finish (and are cached), then StopRequested."""
        jobs = list(jobs)
        results: list = [None] * len(jobs)
        todo = []
        for i, (h, a) in enumerate(jobs):
            hit = self.cached(h, a)
            if hit is not None:
                results[i] = hit
                with self._lock:
                    self.stats["cached"] += 1
                if on_done:
                    on_done(i, hit)
            else:
                todo.append(i)
        if not todo:
            return results
        if self.stop.is_set():
            raise StopRequested(self.stop_reason)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.run_one, *jobs[i]): i for i in todo}
            stopped = None
            for f in as_completed(futs):
                i = futs[f]
                if f.cancelled():          # cancelled after a stop request
                    continue
                try:
                    results[i] = f.result()
                    if on_done:
                        on_done(i, results[i])
                except StopRequested as e:
                    stopped = e
                    for g in futs:
                        g.cancel()
                except Exception:
                    self.request_stop("a simulation failed")
                    for g in futs:
                        g.cancel()
                    raise
            if any(results[i] is None for i in todo):
                raise StopRequested(self.stop_reason or str(stopped))
        return results
