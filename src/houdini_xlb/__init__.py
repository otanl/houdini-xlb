"""Interactive Houdini-to-XLB analysis using a persistent native-Windows GPU worker."""

from .client import XlbWorkerClient, default_python_executable, worker_environment
from .config import XlbConfig, profile_names
from .core import (
    BACKEND_SIGNATURE,
    AnalysisResult,
    analysis_key,
    analyze_heightmap,
    load_cached_heightmap,
    normalize_heights,
    prepare_heightmap,
)
from .houdini_sop import install_parameters, sop_code
from .raster import rasterize_points
from .timeline import TimelineJob, TimelineScheduler
from .validation import AijCaseA, ValidationCriteria

__all__ = [
    "AnalysisResult",
    "AijCaseA",
    "BACKEND_SIGNATURE",
    "XlbConfig",
    "XlbWorkerClient",
    "TimelineJob",
    "TimelineScheduler",
    "ValidationCriteria",
    "analysis_key",
    "analyze_heightmap",
    "default_python_executable",
    "normalize_heights",
    "load_cached_heightmap",
    "prepare_heightmap",
    "profile_names",
    "rasterize_points",
    "install_parameters",
    "sop_code",
    "worker_environment",
]
