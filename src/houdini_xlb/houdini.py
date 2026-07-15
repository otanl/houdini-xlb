"""Thin helpers that are imported inside Houdini, never by the GPU worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .client import XlbWorkerClient
from .config import XlbConfig
from .core import AnalysisResult, normalize_heights
from .raster import rasterize_points


def geometry_heightmap(
    geometry,
    *,
    ny: int,
    nx: int,
    length_x: float,
    length_y: float,
    domain_height_m: float,
    class_attribute: str = "class",
) -> np.ndarray:
    """Convert connected Houdini geometry pieces into a normalized height map."""
    attribute = geometry.findPointAttrib(class_attribute)
    if attribute is None:
        raise ValueError(
            f"input geometry needs point attribute {class_attribute!r}; add a Connectivity SOP"
        )
    points = geometry.points()
    positions = np.asarray([point.position() for point in points], dtype=np.float64)
    classes = np.asarray([point.attribValue(attribute) for point in points])
    height_m = rasterize_points(positions, classes, ny, nx, length_x, length_y)
    return normalize_heights(height_m, domain_height_m)


def session_client(
    *,
    cache_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
) -> XlbWorkerClient:
    """Return one worker retained in hou.session for the life of the editor session."""
    import hou

    name = "_houdini_xlb_client"
    client = getattr(hou.session, name, None)
    if client is None or client.process.poll() is not None:
        client = XlbWorkerClient(
            cache_dir=cache_dir,
            python_executable=python_executable,
        )
        setattr(hou.session, name, client)
    return client


def analyze_geometry(
    geometry,
    *,
    ny: int = 96,
    nx: int = 96,
    length_x: float = 100.0,
    length_y: float = 100.0,
    domain_height_m: float = 40.0,
    profile: str = "preview",
    cache_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
) -> AnalysisResult:
    heightmap = geometry_heightmap(
        geometry,
        ny=ny,
        nx=nx,
        length_x=length_x,
        length_y=length_y,
        domain_height_m=domain_height_m,
    )
    return session_client(
        cache_dir=cache_dir,
        python_executable=python_executable,
    ).analyze(heightmap, XlbConfig.profile(profile))
