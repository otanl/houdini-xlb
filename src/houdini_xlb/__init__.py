"""Interactive Houdini-to-XLB analysis using a persistent native-Windows GPU worker."""

from .client import XlbWorkerClient, default_python_executable, worker_environment
from .config import XlbConfig, profile_names
from .core import (
    AnalysisResult,
    analysis_key,
    analyze_heightmap,
    normalize_heights,
    prepare_heightmap,
)
from .houdini_sop import install_parameters, sop_code
from .raster import rasterize_points

__all__ = [
    "AnalysisResult",
    "XlbConfig",
    "XlbWorkerClient",
    "analysis_key",
    "analyze_heightmap",
    "default_python_executable",
    "normalize_heights",
    "prepare_heightmap",
    "profile_names",
    "rasterize_points",
    "install_parameters",
    "sop_code",
    "worker_environment",
]
