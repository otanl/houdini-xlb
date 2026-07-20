"""Reproducible validation utilities for isolated-building external flow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .config import XlbConfig

AIJ_CASE_A_DOI = "10.5281/zenodo.15050148"
AIJ_CASE_A_LICENSE = "CC BY 4.0"
AIJ_CASE_A_RECORD = "https://zenodo.org/records/15050148"
AIJ_CASE_A_FILES = {
    "AF_caseA.csv": "9f116e2ec5f6984c12c4076bbf242986",
    "RS-caseA.csv": "fc0fe3c1b6c36f20bca135ad1c37f83b",
    "readme_caseA.md": "1b0c46bb2fae8d3defd48571640aee4d",
}


@dataclass(frozen=True)
class AijCaseA:
    """Physical and computational geometry for the AIJ 1:1:2 isolated building."""

    building_width_m: float = 0.08
    building_depth_m: float = 0.08
    building_height_m: float = 0.16
    upstream_clearance_h: float = 2.0
    downstream_clearance_h: float = 10.0
    lateral_extent_h: float = 3.0
    domain_height_h: float = 5.0
    reynolds: float = 24_000.0

    @property
    def x_min_m(self) -> float:
        return -self.building_depth_m / 2 - self.upstream_clearance_h * self.building_height_m

    @property
    def x_max_m(self) -> float:
        return self.building_depth_m / 2 + self.downstream_clearance_h * self.building_height_m

    @property
    def y_min_m(self) -> float:
        return -self.lateral_extent_h * self.building_height_m

    @property
    def y_max_m(self) -> float:
        return self.lateral_extent_h * self.building_height_m

    @property
    def domain_xyz_m(self) -> tuple[float, float, float]:
        return (
            self.x_max_m - self.x_min_m,
            self.y_max_m - self.y_min_m,
            self.domain_height_h * self.building_height_m,
        )

    def coordinates(self, config: XlbConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx, dy, dz = config.cell_sizes_m
        return (
            self.x_min_m + np.arange(config.grid_x) * dx,
            self.y_min_m + np.arange(config.grid_y) * dy,
            np.arange(config.grid_z) * dz,
        )

    def heightmap(self, config: XlbConfig) -> np.ndarray:
        """Rasterize the benchmark building directly on the selected XLB lattice."""

        x, y, _ = self.coordinates(config)
        inside_x = (x >= -self.building_depth_m / 2) & (x < self.building_depth_m / 2)
        inside_y = (y >= -self.building_width_m / 2) & (y < self.building_width_m / 2)
        heightmap = np.zeros((config.grid_y, config.grid_x), dtype=np.float32)
        heightmap[np.ix_(inside_y, inside_x)] = self.building_height_m / config.domain_height_m
        return heightmap

    def config(
        self,
        cells_per_b: int,
        *,
        lattice_wind: float = 0.05,
        flow_throughs: float = 1.5,
        average_flow_throughs: float = 0.5,
        average_samples: int = 40,
        inlet_power_alpha: float = 0.16,
        precision: str = "FP32FP32",
    ) -> XlbConfig:
        """Create an isotropic grid while holding geometry and advective time fixed."""

        if cells_per_b < 2:
            raise ValueError("cells_per_b must be at least 2")
        if flow_throughs <= 0 or not 0 < average_flow_throughs <= flow_throughs:
            raise ValueError("averaging time must be positive and no longer than the run")
        if average_samples < 2:
            raise ValueError("average_samples must be at least 2")

        dx = self.building_width_m / cells_per_b
        domain_x, domain_y, domain_z = self.domain_xyz_m
        grid_x = round(domain_x / dx)
        grid_y = round(domain_y / dx)
        grid_z = round(domain_z / dx)
        advective_steps = grid_x / lattice_wind
        steps = math.ceil(flow_throughs * advective_steps)
        average_window = min(steps, math.ceil(average_flow_throughs * advective_steps))
        average_every = max(1, average_window // average_samples)
        return XlbConfig(
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            steps=steps,
            wind=lattice_wind,
            reynolds=self.reynolds,
            domain_length_x_m=domain_x,
            domain_length_y_m=domain_y,
            domain_height_m=domain_z,
            reference_height_m=self.building_height_m,
            pedestrian_height_m=0.01,
            precision=precision,
            average_window=average_window,
            average_every=average_every,
            inlet_profile="power_law",
            inlet_power_alpha=inlet_power_alpha,
            initial_condition="uniform_reference",
        )


@dataclass(frozen=True)
class AijCaseAReference:
    """Parsed mean-flow benchmark data, normalized by the measured speed at H."""

    approach_z_m: np.ndarray
    approach_u_m_s: np.ndarray
    points_xyz_m: np.ndarray
    velocity_m_s: np.ndarray
    reference_speed_m_s: float
    power_alpha: float

    @property
    def normalized_velocity(self) -> np.ndarray:
        return self.velocity_m_s / self.reference_speed_m_s

    @property
    def normalized_approach(self) -> np.ndarray:
        return self.approach_u_m_s / self.reference_speed_m_s


@dataclass(frozen=True)
class ValidationCriteria:
    """Provisional gates; passing them does not replace expert CFD validation."""

    grid_drift: float = 0.03
    time_window_drift: float = 0.03
    inlet_profile_relative_rmse: float = 0.10
    experimental_u_relative_l2: float = 0.15
    minimum_cells_per_b: int = 20


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324 - published integrity checksum


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed Zenodo URL
        return response.read()


def ensure_aij_case_a_reference(
    directory: str | Path,
    *,
    download: bool = True,
) -> Path:
    """Ensure the small CC-BY reference files exist and match Zenodo checksums."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, checksum in AIJ_CASE_A_FILES.items():
        path = destination / name
        if path.is_file() and _md5(path.read_bytes()) == checksum:
            continue
        if not download:
            raise FileNotFoundError(
                f"{path} is missing or has the wrong checksum; enable reference download"
            )
        data = _download(f"{AIJ_CASE_A_RECORD}/files/{name}?download=1")
        actual = _md5(data)
        if actual != checksum:
            raise RuntimeError(f"checksum mismatch for {name}: expected {checksum}, got {actual}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return destination


def _read_numeric_csv(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [[float(row[column]) for column in columns] for row in csv.DictReader(handle)]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns) or not np.isfinite(values).all():
        raise ValueError(f"invalid numeric data in {path}")
    return values


def fit_power_law_alpha(z_m: np.ndarray, u_m_s: np.ndarray, reference_height_m: float) -> float:
    """Least-squares exponent for U/U_H=(z/H)^alpha with the intercept fixed."""

    order = np.argsort(z_m)
    z = np.asarray(z_m, dtype=np.float64)[order]
    u = np.asarray(u_m_s, dtype=np.float64)[order]
    reference_speed = float(np.interp(reference_height_m, z, u))
    selected = (z > 0) & (u > 0)
    x = np.log(z[selected] / reference_height_m)
    y = np.log(u[selected] / reference_speed)
    denominator = float(x @ x)
    if denominator <= 0:
        raise ValueError("approach profile cannot determine a power-law exponent")
    return float((x @ y) / denominator)


def load_aij_case_a_reference(directory: str | Path) -> AijCaseAReference:
    case = AijCaseA()
    directory = Path(directory)
    approach = _read_numeric_csv(directory / "AF_caseA.csv", ("z(m)", "U(m/s)"))
    results = _read_numeric_csv(
        directory / "RS-caseA.csv",
        ("x(m)", "y(m)", "z(m)", "U(m/s)", "V(m/s)", "W(m/s)"),
    )
    reference_speed = float(np.interp(case.building_height_m, approach[:, 0], approach[:, 1]))
    return AijCaseAReference(
        approach_z_m=approach[:, 0],
        approach_u_m_s=approach[:, 1],
        points_xyz_m=results[:, :3],
        velocity_m_s=results[:, 3:],
        reference_speed_m_s=reference_speed,
        power_alpha=fit_power_law_alpha(
            approach[:, 0],
            approach[:, 1],
            case.building_height_m,
        ),
    )


def trilinear_sample_velocity(
    field: np.ndarray,
    points_xyz_m: np.ndarray,
    case: AijCaseA,
    config: XlbConfig,
) -> np.ndarray:
    """Sample a (component,z,y,x) velocity field at physical coordinates."""

    velocity = np.asarray(field, dtype=np.float64)
    expected = (3, config.grid_z, config.grid_y, config.grid_x)
    if velocity.shape != expected:
        raise ValueError(f"velocity field shape {velocity.shape} must be {expected}")
    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (n,3) array")

    dx, dy, dz = config.cell_sizes_m
    fractional = np.column_stack(
        (
            (points[:, 0] - case.x_min_m) / dx,
            (points[:, 1] - case.y_min_m) / dy,
            points[:, 2] / dz,
        )
    )
    upper = np.asarray((config.grid_x - 1, config.grid_y - 1, config.grid_z - 1))
    if np.any(fractional < 0) or np.any(fractional > upper):
        raise ValueError("one or more measurement points lie outside the lattice")

    lower = np.floor(fractional).astype(int)
    high = np.minimum(lower + 1, upper)
    weight = fractional - lower
    result = np.zeros((len(points), 3), dtype=np.float64)
    for x_high in (0, 1):
        ix = high[:, 0] if x_high else lower[:, 0]
        wx = weight[:, 0] if x_high else 1.0 - weight[:, 0]
        for y_high in (0, 1):
            iy = high[:, 1] if y_high else lower[:, 1]
            wy = weight[:, 1] if y_high else 1.0 - weight[:, 1]
            for z_high in (0, 1):
                iz = high[:, 2] if z_high else lower[:, 2]
                wz = weight[:, 2] if z_high else 1.0 - weight[:, 2]
                result += (wx * wy * wz)[:, None] * velocity[:, iz, iy, ix].T
    return result


def inlet_profile_metrics(
    reference: AijCaseAReference,
    exponent: float,
    case: AijCaseA,
) -> dict[str, float]:
    normalized = reference.normalized_approach
    predicted = (reference.approach_z_m / case.building_height_m) ** exponent
    relative_rmse = float(
        np.sqrt(np.mean((predicted - normalized) ** 2)) / np.sqrt(np.mean(normalized**2))
    )
    return {
        "power_alpha": float(exponent),
        "relative_rmse": relative_rmse,
        "max_abs_error": float(np.max(np.abs(predicted - normalized))),
    }


def simulated_approach_profile_metrics(
    field: np.ndarray,
    reference: AijCaseAReference,
    case: AijCaseA,
    config: XlbConfig,
) -> tuple[dict[str, float], np.ndarray]:
    """Compare the empty-domain profile at the future building centre with AIJ data."""

    points = np.column_stack(
        (
            np.zeros_like(reference.approach_z_m),
            np.zeros_like(reference.approach_z_m),
            reference.approach_z_m,
        )
    )
    sampled = trilinear_sample_velocity(field, points, case, config)
    predicted = sampled[:, 0] / config.wind
    measured = reference.normalized_approach
    difference = predicted - measured
    metrics = {
        "relative_rmse": float(np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(measured**2))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_ratio": float(np.mean(predicted) / np.mean(measured)),
    }
    return metrics, predicted


def prediction_metrics(
    field: np.ndarray,
    reference: AijCaseAReference,
    case: AijCaseA,
    config: XlbConfig,
) -> tuple[dict[str, float], np.ndarray]:
    predicted = trilinear_sample_velocity(field, reference.points_xyz_m, case, config)
    predicted_normalized = predicted / config.wind
    measured = reference.normalized_velocity
    difference = predicted_normalized - measured
    u_denominator = float(np.linalg.norm(measured[:, 0]))
    vector_denominator = float(np.linalg.norm(measured))
    predicted_u = predicted_normalized[:, 0]
    measured_u = measured[:, 0]
    correlation = (
        float(np.corrcoef(predicted_u, measured_u)[0, 1])
        if np.std(predicted_u) > 0 and np.std(measured_u) > 0
        else 0.0
    )
    metrics = {
        "u_relative_l2": float(np.linalg.norm(difference[:, 0]) / u_denominator),
        "vector_relative_l2": float(np.linalg.norm(difference) / vector_denominator),
        "u_rmse_u_h": float(np.sqrt(np.mean(difference[:, 0] ** 2))),
        "vector_rmse_u_h": float(np.sqrt(np.mean(difference**2))),
        "u_correlation": correlation,
        "mean_speed_ratio": float(
            np.mean(np.linalg.norm(predicted_normalized, axis=1))
            / np.mean(np.linalg.norm(measured, axis=1))
        ),
    }
    return metrics, predicted_normalized


def relative_prediction_drift(coarse: np.ndarray, fine: np.ndarray) -> float:
    coarse = np.asarray(coarse, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    if coarse.shape != fine.shape:
        raise ValueError("predictions must share a shape")
    denominator = float(np.linalg.norm(fine))
    if denominator == 0:
        raise ValueError("fine prediction norm is zero")
    return float(np.linalg.norm(coarse - fine) / denominator)


def validation_cache_key(
    case: AijCaseA,
    config: XlbConfig,
    backend_signature: str,
    *,
    geometry: str = "building",
    collision_model: str = "KBC",
) -> str:
    if geometry not in {"building", "empty"}:
        raise ValueError("validation geometry must be 'building' or 'empty'")
    if collision_model not in {"KBC", "SmagorinskyLESBGK"}:
        raise ValueError("unsupported validation collision model")
    payload = {
        "case": asdict(case),
        "config": config.to_dict(),
        "backend_signature": backend_signature,
        "field_contract": "component-z-y-x-fluid-masked-v2",
        "geometry": geometry,
        "collision_model": collision_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def report_status(
    *,
    cells_per_b: list[int],
    run_metrics: list[dict[str, float]],
    grid_drifts: list[float],
    time_window_drift: float | None,
    inlet_relative_rmse: float | None,
    criteria: ValidationCriteria,
) -> tuple[str, dict[str, bool | None]]:
    checks: dict[str, bool | None] = {
        "grid_convergence": (grid_drifts[-1] <= criteria.grid_drift if grid_drifts else None),
        "time_window_convergence": (
            time_window_drift <= criteria.time_window_drift
            if time_window_drift is not None
            else None
        ),
        "inlet_profile": (
            inlet_relative_rmse <= criteria.inlet_profile_relative_rmse
            if inlet_relative_rmse is not None
            else None
        ),
        "experimental_mean_u": (
            run_metrics[-1]["u_relative_l2"] <= criteria.experimental_u_relative_l2
            if run_metrics
            else None
        ),
        "reference_resolution": (
            max(cells_per_b) >= criteria.minimum_cells_per_b if cells_per_b else None
        ),
    }
    if not run_metrics:
        return "not_run", checks
    if any(value is None for value in checks.values()):
        return "incomplete", checks
    return ("provisional_pass" if all(checks.values()) else "provisional_fail"), checks


def reference_provenance() -> dict[str, object]:
    return {
        "name": "AIJ UWE Benchmark Dataset - Case A (112)",
        "doi": AIJ_CASE_A_DOI,
        "record": AIJ_CASE_A_RECORD,
        "license": AIJ_CASE_A_LICENSE,
        "files": dict(AIJ_CASE_A_FILES),
    }


def read_cached_velocity(path: Path, config: XlbConfig) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_config = json.loads(str(data["config_json"].item()))
            velocity = np.asarray(data["velocity"], dtype=np.float32)
        if stored_config != config.to_dict():
            return None
        expected = (3, config.grid_z, config.grid_y, config.grid_x)
        if velocity.shape != expected or not np.isfinite(velocity).all():
            return None
        return velocity
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def write_cached_velocity(path: Path, field: np.ndarray, config: XlbConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            _write_npz(handle, field, config)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npz(handle: BinaryIO, field: np.ndarray, config: XlbConfig) -> None:
    np.savez_compressed(
        handle,
        velocity=np.asarray(field, dtype=np.float32),
        config_json=np.asarray(json.dumps(config.to_dict(), sort_keys=True)),
    )
