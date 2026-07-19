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
    cache_changed = (
        client is not None
        and cache_dir is not None
        and client.cache_dir != Path(cache_dir).resolve()
    )
    python_changed = (
        client is not None
        and python_executable is not None
        and client.python_executable != Path(python_executable).resolve()
    )
    if client is None or client.process.poll() is not None or cache_changed or python_changed:
        if client is not None:
            client.close()
        client = XlbWorkerClient(
            cache_dir=cache_dir,
            python_executable=python_executable,
        )
        setattr(hou.session, name, client)
    return client


def analyze_geometry(
    geometry,
    *,
    ny: int | None = None,
    nx: int | None = None,
    length_x: float = 100.0,
    length_y: float = 100.0,
    domain_height_m: float = 40.0,
    reference_height_m: float = 10.0,
    pedestrian_height_m: float = 1.5,
    profile: str = "preview",
    cache_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
) -> AnalysisResult:
    config = XlbConfig.profile(profile).with_domain(
        length_x_m=length_x,
        length_y_m=length_y,
        height_m=domain_height_m,
        reference_height_m=reference_height_m,
        pedestrian_height_m=pedestrian_height_m,
    )
    requested_shape = (config.grid_y if ny is None else ny, config.grid_x if nx is None else nx)
    if requested_shape != (config.grid_y, config.grid_x):
        raise ValueError(
            f"profile {profile!r} requires height-map shape "
            f"{(config.grid_y, config.grid_x)}; got {requested_shape}"
        )
    heightmap = geometry_heightmap(
        geometry,
        ny=config.grid_y,
        nx=config.grid_x,
        length_x=length_x,
        length_y=length_y,
        domain_height_m=domain_height_m,
    )
    return session_client(
        cache_dir=cache_dir,
        python_executable=python_executable,
    ).analyze(heightmap, config)
