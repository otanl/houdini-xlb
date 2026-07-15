"""Houdini-side client for the persistent project-Python XLB worker."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

import numpy as np

from .config import XlbConfig
from .core import AnalysisResult, prepare_heightmap
from .protocol import READY, RESPONSE

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _venv_python(root: Path) -> Path:
    return root / ".venv" / "Scripts" / "python.exe"


def _workspace_candidates() -> tuple[Path, ...]:
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    return tuple(dict.fromkeys(candidates))


def _default_workspace() -> Path:
    for candidate in _workspace_candidates():
        if _venv_python(candidate).exists():
            return candidate
    return Path.cwd().resolve()


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


def default_python_executable() -> Path:
    configured = os.environ.get("HOUDINI_XLB_PYTHON")
    candidate = Path(configured) if configured else _venv_python(_default_workspace())
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
    ):
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
        self._log_stream: TextIO | None = (
            Path(log).open("a", encoding="utf-8") if log is not None else None
        )
        environment = worker_environment()
        self.process = subprocess.Popen(
            [str(self.python_executable), "-m", "houdini_xlb.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_stream,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.strip() == READY:
                break
        else:
            raise RuntimeError("XLB worker exited before announcing readiness")

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        if self.process.poll() is not None:
            raise RuntimeError(f"XLB worker is not running (exit={self.process.returncode})")
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        with self._lock:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
            for line in self.process.stdout:
                if not line.startswith(RESPONSE):
                    continue
                response = json.loads(line[len(RESPONSE) :])
                if not response.get("ok"):
                    raise RuntimeError(
                        f"XLB worker failed: {response.get('error')}\n"
                        f"{response.get('traceback', '')}"
                    )
                return response
        raise RuntimeError("XLB worker closed without a response")

    def health(self) -> dict[str, object]:
        return self._request({"op": "health"})

    def analyze(
        self,
        heightmap: np.ndarray,
        config: XlbConfig | None = None,
    ) -> AnalysisResult:
        config = config or XlbConfig.profile("preview")
        heightmap = prepare_heightmap(heightmap)
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
            cache_path = Path(str(response["cache_path"]))
            with np.load(cache_path, allow_pickle=False) as data:
                speed = np.asarray(data["speed"], dtype=np.float32)
            return AnalysisResult(
                speed=speed,
                cache_key=str(response["cache_key"]),
                cache_hit=bool(response["cache_hit"]),
                elapsed_s=float(response["elapsed_s"]),
                config=config,
                cache_path=cache_path,
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
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write("shutdown\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def __enter__(self) -> XlbWorkerClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
