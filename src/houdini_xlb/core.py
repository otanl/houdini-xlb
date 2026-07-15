"""Cached, Houdini-independent height-map analysis through XLB."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import XlbConfig

Solver = Callable[[np.ndarray, XlbConfig], np.ndarray]
CACHE_VERSION = 2


@dataclass(frozen=True)
class AnalysisResult:
    speed: np.ndarray
    cache_key: str
    cache_hit: bool
    elapsed_s: float
    config: XlbConfig
    cache_path: Path | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "cache_version": CACHE_VERSION,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "elapsed_s": self.elapsed_s,
            "shape": list(self.speed.shape),
            "config": self.config.to_dict(),
            "cache_path": str(self.cache_path) if self.cache_path else None,
        }


def prepare_heightmap(heightmap: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(heightmap, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("heightmap must be a two-dimensional field")
    if not np.isfinite(array).all():
        raise ValueError("heightmap contains non-finite values")
    tolerance = 1e-6
    if float(array.min()) < -tolerance or float(array.max()) > 1.0 + tolerance:
        raise ValueError("heightmap values must be normalized to [0, 1]")
    return np.clip(array, 0.0, 1.0)


def normalize_heights(height_m: np.ndarray, domain_height_m: float) -> np.ndarray:
    if domain_height_m <= 0:
        raise ValueError("domain_height_m must be positive")
    return prepare_heightmap(np.asarray(height_m, dtype=np.float32) / domain_height_m)


def analysis_key(heightmap: np.ndarray, config: XlbConfig) -> str:
    heightmap = prepare_heightmap(heightmap)
    digest = hashlib.sha256()
    digest.update(f"houdini-xlb-cache-v{CACHE_VERSION}".encode())
    digest.update(json.dumps(config.to_dict(), sort_keys=True).encode())
    digest.update(np.asarray(heightmap.shape, dtype=np.int64).tobytes())
    digest.update(heightmap.tobytes())
    return digest.hexdigest()


def _default_solver(heightmap: np.ndarray, config: XlbConfig) -> np.ndarray:
    from .backend import simulate_heightmap_xlb

    return simulate_heightmap_xlb(
        heightmap,
        grid_xyz=config.grid_xyz,
        wind=config.wind,
        reynolds=config.reynolds,
        steps=config.steps,
        reference_height=config.reference_height,
        pedestrian_z=config.pedestrian_z,
        precision=config.precision,
        average_window=config.average_window,
        average_every=config.average_every,
    )


def _read_cache(path: Path, config: XlbConfig, key: str) -> AnalysisResult:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        speed = np.asarray(data["speed"], dtype=np.float32)
    return AnalysisResult(
        speed=speed,
        cache_key=key,
        cache_hit=True,
        elapsed_s=time.perf_counter() - started,
        config=config,
        cache_path=path,
    )


def load_cached_heightmap(
    heightmap: np.ndarray,
    config: XlbConfig | None = None,
    *,
    cache_dir: str | Path,
) -> AnalysisResult | None:
    """Return an exact cached result without starting the XLB backend."""
    config = config or XlbConfig.profile("preview")
    heightmap = prepare_heightmap(heightmap)
    key = analysis_key(heightmap, config)
    cache_path = Path(cache_dir).resolve() / f"{key}.npz"
    if not cache_path.exists():
        return None
    return _read_cache(cache_path, config, key)


def _write_cache(path: Path, result: AnalysisResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(result.metadata(), sort_keys=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, speed=result.speed, metadata=np.asarray(metadata))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def analyze_heightmap(
    heightmap: np.ndarray,
    config: XlbConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
    solver: Solver | None = None,
) -> AnalysisResult:
    """Run or retrieve one deterministic XLB analysis."""
    config = config or XlbConfig.profile("preview")
    heightmap = prepare_heightmap(heightmap)
    key = analysis_key(heightmap, config)
    cache_path = Path(cache_dir).resolve() / f"{key}.npz" if cache_dir is not None else None
    if cache_path is not None:
        cached = load_cached_heightmap(heightmap, config, cache_dir=cache_dir)
        if cached is not None:
            return cached

    started = time.perf_counter()
    speed = np.asarray((solver or _default_solver)(heightmap, config), dtype=np.float32)
    if speed.shape != heightmap.shape:
        raise RuntimeError(
            f"XLB returned {speed.shape}, expected the input height-map shape {heightmap.shape}"
        )
    if not np.isfinite(speed).all():
        raise RuntimeError("XLB returned non-finite velocity values")
    result = AnalysisResult(
        speed=np.maximum(speed, 0.0),
        cache_key=key,
        cache_hit=False,
        elapsed_s=time.perf_counter() - started,
        config=config,
        cache_path=cache_path,
    )
    if cache_path is not None:
        _write_cache(cache_path, result)
    return result
