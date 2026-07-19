"""Houdini-side client for the persistent project-Python XLB worker."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

import numpy as np

from .config import XlbConfig
from .core import AnalysisResult, load_cached_heightmap, prepare_heightmap
from .protocol import READY, RESPONSE

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _venv_python(root: Path) -> Path:
    return root / ".venv" / "Scripts" / "python.exe"


def _workspace_candidates(search_root: str | Path | None = None) -> tuple[Path, ...]:
    starts = [Path.cwd().resolve()]
    if search_root is not None:
        starts.insert(0, Path(search_root).resolve())
    candidates = [candidate for start in starts for candidate in (start, *start.parents)]
    candidates.extend(Path(__file__).resolve().parents)
    return tuple(dict.fromkeys(candidates))


def _default_workspace(search_root: str | Path | None = None) -> Path:
    for candidate in _workspace_candidates(search_root):
        if _venv_python(candidate).exists():
            return candidate
    return Path(search_root).resolve() if search_root is not None else Path.cwd().resolve()


def worker_environment(
    inherited: dict[str, str] | None = None,
    *,
    source_root: str | Path | None = None,
) -> dict[str, str]:
    """Build a Python-3.12 environment without leaking Houdini's Python 3.11 runtime."""
    environment = dict(os.environ if inherited is None else inherited)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONEXECUTABLE", None)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if source_root is None:
        candidate = PACKAGE_ROOT / "src"
        paths = [str(candidate.resolve())] if (candidate / "houdini_xlb").exists() else []
    else:
        paths = [str(Path(source_root).resolve())]
    extra = environment.pop("HOUDINI_XLB_PYTHONPATH", "")
    if extra:
        paths.append(extra)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def default_python_executable(search_root: str | Path | None = None) -> Path:
    configured = os.environ.get("HOUDINI_XLB_PYTHON")
    candidate = Path(configured) if configured else _venv_python(_default_workspace(search_root))
    if not candidate.exists():
        raise FileNotFoundError(f"project Python not found at {candidate}; set HOUDINI_XLB_PYTHON")
    return candidate.resolve()


class XlbWorkerClient:
    """One persistent GPU worker, safe to retain in a Houdini session."""

    def __init__(
        self,
        *,
        python_executable: str | Path | None = None,
        cache_dir: str | Path | None = None,
        log: str | Path | None = None,
        startup_timeout_s: float = 60.0,
        request_timeout_s: float = 3600.0,
        shutdown_timeout_s: float = 10.0,
    ):
        if min(startup_timeout_s, request_timeout_s, shutdown_timeout_s) <= 0:
            raise ValueError("worker timeouts must be positive")
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.python_executable = (
            Path(python_executable).resolve()
            if python_executable is not None
            else default_python_executable()
        )
        default_cache = os.environ.get("HOUDINI_XLB_CACHE")
        if default_cache is None:
            default_cache = _default_workspace() / "artifacts" / "houdini" / "cache" / "xlb"
        self.cache_dir = Path(cache_dir or default_cache).resolve()
        self.requests_dir = self.cache_dir / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._log_stream: TextIO | None = (
            Path(log).open("a", encoding="utf-8") if log is not None else None
        )
        environment = worker_environment()
        try:
            self.process = subprocess.Popen(
                [str(self.python_executable), "-m", "houdini_xlb.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_stream,
                text=True,
                bufsize=1,
                env=environment,
            )
        except BaseException:
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            raise
        self._reader = threading.Thread(
            target=self._read_stdout,
            name="houdini-xlb-stdout",
            daemon=True,
        )
        self._reader.start()
        try:
            self._wait_for_ready()
        except BaseException:
            self._terminate()
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            raise

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._stdout_queue.put(line)
        finally:
            self._stdout_queue.put(None)

    def _next_line(self, deadline: float, phase: str) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._terminate()
            raise TimeoutError(f"XLB worker {phase} timed out")
        try:
            line = self._stdout_queue.get(timeout=remaining)
        except queue.Empty as exc:
            self._terminate()
            raise TimeoutError(f"XLB worker {phase} timed out") from exc
        if line is None:
            raise RuntimeError(f"XLB worker closed during {phase} (exit={self.process.poll()})")
        return line

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            if self._next_line(deadline, "startup").strip() == READY:
                return

    def _terminate(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                pass

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        if self.process.poll() is not None:
            raise RuntimeError(f"XLB worker is not running (exit={self.process.returncode})")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        with self._lock:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + self.request_timeout_s
            while True:
                line = self._next_line(deadline, "request")
                if not line.startswith(RESPONSE):
                    continue
                response = json.loads(line[len(RESPONSE) :])
                if not response.get("ok"):
                    raise RuntimeError(
                        f"XLB worker failed: {response.get('error')}\n"
                        f"{response.get('traceback', '')}"
                    )
                return response

    def health(self) -> dict[str, object]:
        return self._request({"op": "health"})

    def analyze(
        self,
        heightmap: np.ndarray,
        config: XlbConfig | None = None,
    ) -> AnalysisResult:
        config = config or XlbConfig.profile("preview")
        heightmap = prepare_heightmap(heightmap)
        expected_shape = (config.grid_y, config.grid_x)
        if heightmap.shape != expected_shape:
            raise ValueError(
                f"heightmap shape {heightmap.shape} must equal XLB lattice y/x {expected_shape}"
            )
        request_path = self.requests_dir / f"{uuid.uuid4().hex}.npy"
        np.save(request_path, heightmap, allow_pickle=False)
        try:
            response = self._request(
                {
                    "op": "analyze",
                    "heightmap_path": str(request_path),
                    "cache_dir": str(self.cache_dir),
                    "config": config.to_dict(),
                }
            )
            cached = load_cached_heightmap(
                heightmap,
                config,
                cache_dir=self.cache_dir,
            )
            if cached is None or cached.cache_key != str(response["cache_key"]):
                raise RuntimeError("worker response cache failed integrity validation")
            return AnalysisResult(
                speed=cached.speed,
                cache_key=cached.cache_key,
                cache_hit=bool(response["cache_hit"]),
                elapsed_s=float(response["elapsed_s"]),
                config=config,
                cache_path=cached.cache_path,
            )
        finally:
            request_path.unlink(missing_ok=True)

    def analyze_async(
        self,
        heightmap: np.ndarray,
        config: XlbConfig | None = None,
    ) -> Future[AnalysisResult]:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="houdini-xlb")
        return self._executor.submit(self.analyze, np.asarray(heightmap).copy(), config)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self.process.poll() is None:
            acquired = self._lock.acquire(blocking=False)
            if acquired:
                try:
                    assert self.process.stdin is not None
                    self.process.stdin.write("shutdown\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=self.shutdown_timeout_s)
                except Exception:
                    self._terminate()
                finally:
                    self._lock.release()
            else:
                self._terminate()
        self._reader.join(timeout=self.shutdown_timeout_s)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def __enter__(self) -> XlbWorkerClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
