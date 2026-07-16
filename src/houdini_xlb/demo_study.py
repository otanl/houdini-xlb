"""Small, reproducible XLB-in-the-loop optimization used by the public demo.

The problem is intentionally finite and transparent: two fixed-volume buildings
each occupy one of four locations in assigned windward parcels.  All 16 layouts
are evaluated with XLB.  The objective minimizes plaza area outside an
illustrative speed band while a central ventilation route must retain at least
95 percent of its baseline mean speed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from math import hypot

import numpy as np

DOMAIN_X_M = 100.0
DOMAIN_Y_M = 100.0
DOMAIN_HEIGHT_M = 40.0
GRID_NX = 96
GRID_NY = 96

START_FRAME = 1
END_FRAME = 120
MILESTONE_FRAMES = (1, 31, 61, 91, 120)
MILESTONE_EVALUATIONS = (1, 3, 5, 11, 16)

COMFORT_MIN = 0.55
COMFORT_MAX = 1.20
MIN_VENT_RETENTION = 0.95
MIN_CLEARANCE_M = 4.0

# Bounds are (xmin, ymin, xmax, ymax) in world metres.
PLAZA_BOUNDS = (56.0, 42.0, 88.0, 58.0)
VENT_BOUNDS = (20.0, 47.0, 56.0, 53.0)

# Each movable building has two x and two y choices inside its assigned parcel.
LOWER_X_OPTIONS = (38.0, 50.0)
LOWER_Y_OPTIONS = (30.0, 38.0)
UPPER_X_OPTIONS = (38.0, 50.0)
UPPER_Y_OPTIONS = (62.0, 70.0)
BASE_DESIGN = (38.0, 30.0, 38.0, 70.0)
SEARCH_SEED = 2

Design = tuple[float, float, float, float]


@dataclass(frozen=True)
class Massing:
    """Axis-aligned building massing in world metres."""

    cx: float
    cy: float
    width: float
    depth: float
    height: float

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass(frozen=True)
class StudyMetrics:
    """Scalar measurements used by the illustrative optimization objective."""

    plaza_mean: float
    comfort_fraction: float
    vent_mean: float
    band_error: float


MOVABLE_TEMPLATES = (
    Massing(38.0, 30.0, 12.0, 18.0, 20.0),
    Massing(38.0, 70.0, 12.0, 18.0, 20.0),
)
FIXED_BUILDINGS = (
    Massing(72.0, 30.0, 16.0, 16.0, 14.0),
    Massing(72.0, 70.0, 16.0, 16.0, 14.0),
)


def candidate_designs(*, seed: int = SEARCH_SEED) -> tuple[Design, ...]:
    """Return the baseline followed by a deterministic shuffled exhaustive search."""

    candidates = [
        tuple(map(float, design))
        for design in product(
            LOWER_X_OPTIONS,
            LOWER_Y_OPTIONS,
            UPPER_X_OPTIONS,
            UPPER_Y_OPTIONS,
        )
    ]
    candidates.remove(BASE_DESIGN)
    random.Random(seed).shuffle(candidates)
    return (BASE_DESIGN, *candidates)


def study_buildings(design: Design) -> tuple[Massing, ...]:
    """Decode four design variables into two movable and two fixed buildings."""

    x0, y0, x1, y1 = map(float, design)
    movable = (
        replace(MOVABLE_TEMPLATES[0], cx=x0, cy=y0),
        replace(MOVABLE_TEMPLATES[1], cx=x1, cy=y1),
    )
    return movable + FIXED_BUILDINGS


def footprint_clearance(first: Massing, second: Massing) -> float:
    """Euclidean clearance between two axis-aligned rectangular footprints."""

    gap_x = abs(first.cx - second.cx) - (first.width + second.width) / 2.0
    gap_y = abs(first.cy - second.cy) - (first.depth + second.depth) / 2.0
    return hypot(max(gap_x, 0.0), max(gap_y, 0.0))


def minimum_clearance(buildings: Iterable[Massing]) -> float:
    """Return the smallest pairwise footprint clearance."""

    pairs = list(combinations(tuple(buildings), 2))
    if not pairs:
        return float("inf")
    return min(footprint_clearance(first, second) for first, second in pairs)


def validate_design(design: Design) -> float:
    """Validate parcel choices, fixed volumes and clearance for one layout."""

    x0, y0, x1, y1 = map(float, design)
    if x0 not in LOWER_X_OPTIONS or y0 not in LOWER_Y_OPTIONS:
        raise ValueError(f"lower building lies outside its discrete parcel options: {design}")
    if x1 not in UPPER_X_OPTIONS or y1 not in UPPER_Y_OPTIONS:
        raise ValueError(f"upper building lies outside its discrete parcel options: {design}")
    buildings = study_buildings(design)
    clearance = minimum_clearance(buildings)
    if clearance < MIN_CLEARANCE_M:
        raise ValueError(f"building clearance {clearance:.3f} m is below {MIN_CLEARANCE_M:.3f} m")
    expected_volumes = tuple(building.volume for building in MOVABLE_TEMPLATES)
    actual_volumes = tuple(building.volume for building in buildings[:2])
    if actual_volumes != expected_volumes:
        raise ValueError("movable building volume changed")
    return clearance


def heightmap_from_design(
    design: Design,
    *,
    ny: int = GRID_NY,
    nx: int = GRID_NX,
) -> np.ndarray:
    """Rasterize the axis-aligned demo buildings exactly on XLB grid centres."""

    validate_design(design)
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * DOMAIN_X_M / nx
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * DOMAIN_Y_M / ny
    xx, yy = np.meshgrid(xs, ys)
    heightmap = np.zeros((ny, nx), dtype=np.float32)
    for building in study_buildings(design):
        inside = (np.abs(xx - building.cx) <= building.width / 2.0) & (
            np.abs(yy - building.cy) <= building.depth / 2.0
        )
        heightmap[inside] = np.maximum(
            heightmap[inside],
            building.height / DOMAIN_HEIGHT_M,
        )
    return heightmap


def _zone_mask(
    bounds: tuple[float, float, float, float],
    *,
    ny: int,
    nx: int,
) -> np.ndarray:
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * DOMAIN_X_M / nx
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * DOMAIN_Y_M / ny
    xx, yy = np.meshgrid(xs, ys)
    xmin, ymin, xmax, ymax = bounds
    return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)


def metrics_from_speed(speed: np.ndarray, *, inlet_speed: float) -> StudyMetrics:
    """Measure plaza comfort-band coverage and central-route mean speed."""

    speed = np.asarray(speed, dtype=np.float64)
    if speed.ndim != 2 or inlet_speed <= 0:
        raise ValueError("speed must be a 2-D field and inlet_speed must be positive")
    ny, nx = speed.shape
    normalized = speed / inlet_speed
    plaza = normalized[_zone_mask(PLAZA_BOUNDS, ny=ny, nx=nx)]
    vent = normalized[_zone_mask(VENT_BOUNDS, ny=ny, nx=nx)]
    lower_error = np.maximum(COMFORT_MIN - plaza, 0.0)
    upper_error = np.maximum(plaza - COMFORT_MAX, 0.0)
    return StudyMetrics(
        plaza_mean=float(plaza.mean()),
        comfort_fraction=float(np.mean((plaza >= COMFORT_MIN) & (plaza <= COMFORT_MAX))),
        vent_mean=float(vent.mean()),
        band_error=float(np.mean(lower_error**2 + upper_error**2)),
    )


def mean_speed_in_bounds(
    positions: np.ndarray,
    speed: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> float:
    """Mean point speed inside a rectangular world-space measurement zone."""

    positions = np.asarray(positions, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must have shape (n, 2+)")
    if speed.ndim != 1 or len(speed) != len(positions):
        raise ValueError("speed must contain one scalar per position")
    xmin, ymin, xmax, ymax = bounds
    inside = (
        (positions[:, 0] >= xmin)
        & (positions[:, 0] <= xmax)
        & (positions[:, 1] >= ymin)
        & (positions[:, 1] <= ymax)
    )
    if not np.any(inside):
        raise ValueError(f"measurement zone {bounds} contains no points")
    return float(speed[inside].mean())


def _rank(
    design: Design,
    metrics: StudyMetrics,
    *,
    baseline_vent: float,
) -> tuple[float, ...]:
    retention = metrics.vent_mean / baseline_vent
    feasible = retention + 1e-12 >= MIN_VENT_RETENTION
    objective = 1.0 - metrics.comfort_fraction
    if feasible:
        return (0.0, objective, metrics.band_error, *design)
    return (
        1.0,
        MIN_VENT_RETENTION - retention,
        objective,
        metrics.band_error,
        *design,
    )


def optimize_study(
    evaluate: Callable[[Design], StudyMetrics],
    *,
    seed: int = SEARCH_SEED,
) -> dict[str, object]:
    """Exhaustively evaluate 16 layouts and return JSON-ready best-so-far history."""

    designs = candidate_designs(seed=seed)
    baseline_metrics = evaluate(BASE_DESIGN)
    baseline_vent = baseline_metrics.vent_mean
    evaluations: list[dict[str, object]] = []
    best_rank: tuple[float, ...] | None = None
    best_record: dict[str, object] | None = None

    for evaluation, design in enumerate(designs, start=1):
        validate_design(design)
        metrics = baseline_metrics if design == BASE_DESIGN else evaluate(design)
        retention = metrics.vent_mean / baseline_vent
        objective = 1.0 - metrics.comfort_fraction
        rank = _rank(design, metrics, baseline_vent=baseline_vent)
        feasible = retention + 1e-12 >= MIN_VENT_RETENTION
        candidate = {
            "evaluation": evaluation,
            "design": list(design),
            "metrics": asdict(metrics),
            "objective": objective,
            "vent_retention": retention,
            "feasible": feasible,
        }
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_record = candidate
        if best_record is None:
            raise RuntimeError("optimizer did not establish a best record")
        evaluations.append(
            {
                **candidate,
                "best_evaluation": best_record["evaluation"],
                "best_design": best_record["design"],
                "best_metrics": best_record["metrics"],
                "best_objective": best_record["objective"],
                "best_vent_retention": best_record["vent_retention"],
            }
        )

    milestones = []
    for frame, evaluation in zip(
        MILESTONE_FRAMES,
        MILESTONE_EVALUATIONS,
        strict=True,
    ):
        record = evaluations[evaluation - 1]
        milestones.append(
            {
                "frame": frame,
                "evaluation": evaluation,
                "best_evaluation": record["best_evaluation"],
                "design": record["best_design"],
                "metrics": record["best_metrics"],
                "objective": record["best_objective"],
                "vent_retention": record["best_vent_retention"],
            }
        )

    final = evaluations[-1]
    return {
        "schema_version": 1,
        "search": {
            "method": "exhaustive-discrete",
            "seed": seed,
            "candidate_count": len(designs),
            "design_variables": ["lower_x", "lower_y", "upper_x", "upper_y"],
        },
        "objective": {
            "name": "plaza_outside_comfort_band_fraction",
            "comfort_band_u_over_uin": [COMFORT_MIN, COMFORT_MAX],
            "ventilation_constraint": MIN_VENT_RETENTION,
        },
        "baseline": {
            "design": list(BASE_DESIGN),
            "metrics": asdict(baseline_metrics),
        },
        "evaluations": evaluations,
        "milestones": milestones,
        "result": {
            "design": final["best_design"],
            "metrics": final["best_metrics"],
            "objective": final["best_objective"],
            "vent_retention": final["best_vent_retention"],
            "best_evaluation": final["best_evaluation"],
        },
    }


def validate_optimization(data: dict[str, object]) -> None:
    """Reject stale or internally inconsistent optimization JSON."""

    if data.get("schema_version") != 1:
        raise ValueError("unsupported demo optimization schema")
    evaluations = data.get("evaluations")
    milestones = data.get("milestones")
    if not isinstance(evaluations, list) or len(evaluations) != 16:
        raise ValueError("demo optimization must contain all 16 evaluations")
    if not isinstance(milestones, list) or len(milestones) != len(MILESTONE_FRAMES):
        raise ValueError("demo optimization has the wrong milestone count")
    best_objectives = []
    for record in evaluations:
        design = tuple(map(float, record["design"]))
        validate_design(design)
        best_objectives.append(float(record["best_objective"]))
    if any(
        after > before + 1e-12
        for before, after in zip(best_objectives, best_objectives[1:], strict=False)
    ):
        raise ValueError("best-so-far objective is not monotone")
    result = data.get("result")
    if not isinstance(result, dict):
        raise ValueError("demo optimization result is missing")
    validate_design(tuple(map(float, result["design"])))
    if float(result["vent_retention"]) + 1e-12 < MIN_VENT_RETENTION:
        raise ValueError("demo optimization result violates the ventilation constraint")
