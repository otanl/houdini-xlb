"""Command-line runner for the AIJ Case A isolated-building validation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .backend import simulate_velocity_field_xlb
from .core import BACKEND_SIGNATURE
from .validation import (
    AijCaseA,
    ValidationCriteria,
    ensure_aij_case_a_reference,
    inlet_profile_metrics,
    load_aij_case_a_reference,
    prediction_metrics,
    read_cached_velocity,
    reference_provenance,
    relative_prediction_drift,
    report_status,
    simulated_approach_profile_metrics,
    validation_cache_key,
    write_cached_velocity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run a grid/time-sensitivity comparison against the CC-BY AIJ Case A "
            "isolated-building wind-tunnel data."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run XLB. Without this flag only a deterministic cost/configuration plan is written.",
    )
    parser.add_argument(
        "--cells-per-b",
        default="8,12,16",
        help="Comma-separated uniform-grid resolutions across the 0.08 m building width.",
    )
    parser.add_argument("--lattice-wind", type=float, default=0.05)
    parser.add_argument(
        "--collision-model",
        choices=("KBC", "SmagorinskyLESBGK"),
        default="KBC",
    )
    parser.add_argument("--flow-throughs", type=float, default=1.5)
    parser.add_argument("--average-flow-throughs", type=float, default=0.5)
    parser.add_argument("--average-samples", type=int, default=40)
    parser.add_argument(
        "--inlet-power-alpha",
        type=float,
        help="Override the exponent fitted from the AIJ approach-flow measurements.",
    )
    parser.add_argument(
        "--time-check",
        action="store_true",
        help="On the finest grid, rerun longer with twice the averaging duration.",
    )
    parser.add_argument(
        "--skip-empty-domain-check",
        action="store_true",
        help="Skip the finest-grid empty-domain approach-flow run; status remains incomplete.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("artifacts/validation/aij_case_a/reference"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/validation/aij_case_a/cache"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/validation/aij_case_a_report.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/validation/aij_case_a_predictions.csv"),
    )
    parser.add_argument(
        "--inlet-profile-out",
        type=Path,
        default=Path("outputs/validation/aij_case_a_inlet_profile.csv"),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore validation field caches.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 unless all provisional gates pass.",
    )
    return parser


def _levels(value: str) -> list[int]:
    try:
        levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cells-per-b must contain integers") from exc
    if len(levels) < 2 or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError("provide at least two levels, each >= 2")
    if levels != sorted(set(levels)):
        raise argparse.ArgumentTypeError("cells-per-b levels must be unique and increasing")
    return levels


def _run_xlb(
    case: AijCaseA,
    config,
    cache_dir: Path,
    force: bool,
    *,
    geometry: str = "building",
    collision_model: str = "KBC",
):
    key = validation_cache_key(
        case,
        config,
        BACKEND_SIGNATURE,
        geometry=geometry,
        collision_model=collision_model,
    )
    cache_path = cache_dir / f"{key}.npz"
    if not force:
        cached = read_cached_velocity(cache_path, config)
        if cached is not None:
            return cached, True, 0.0, cache_path

    heightmap = (
        case.heightmap(config)
        if geometry == "building"
        else np.zeros((config.grid_y, config.grid_x), dtype=np.float32)
    )
    started = time.perf_counter()
    field = simulate_velocity_field_xlb(
        heightmap,
        grid_xyz=config.grid_xyz,
        wind=config.wind,
        reynolds=config.reynolds,
        steps=config.steps,
        precision=config.precision,
        average_window=config.average_window,
        average_every=config.average_every,
        reference_height_lattice=config.reference_height_lattice,
        max_speed_ratio=config.max_speed_ratio,
        inlet_profile=config.inlet_profile,
        inlet_power_alpha=config.inlet_power_alpha,
        initial_condition=config.initial_condition,
        collision_model=collision_model,
    )
    elapsed = time.perf_counter() - started
    write_cached_velocity(cache_path, field, config)
    return field, False, elapsed, cache_path


def _plan_record(case: AijCaseA, cells_per_b: int, config) -> dict[str, object]:
    cells = config.grid_x * config.grid_y * config.grid_z
    building_cells = int(np.count_nonzero(case.heightmap(config)))
    return {
        "cells_per_b": cells_per_b,
        "grid_xyz": list(config.grid_xyz),
        "lattice_cells": cells,
        "building_footprint_cells": building_cells,
        "steps": config.steps,
        "average_window": config.average_window,
        "average_every": config.average_every,
        "average_samples": 1 + (config.average_window - 1) // config.average_every,
        "flow_throughs": config.steps * config.wind / config.grid_x,
        "average_flow_throughs": config.average_window * config.wind / config.grid_x,
        "estimated_two_population_buffers_gib": 2 * 27 * cells * 4 / 1024**3,
        "cell_size_m": config.cell_sizes_m[0],
        "config": config.to_dict(),
    }


def _write_predictions(
    path: Path,
    points: np.ndarray,
    measured: np.ndarray,
    levels: list[int],
    predictions: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["point", "x_m", "y_m", "z_m", "measured_u", "measured_v", "measured_w"]
    for level in levels:
        header.extend((f"predicted_u_b{level}", f"predicted_v_b{level}", f"predicted_w_b{level}"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, (point, observed) in enumerate(zip(points, measured, strict=True), start=1):
            row: list[float | int] = [index, *point.tolist(), *observed.tolist()]
            for prediction in predictions:
                row.extend(prediction[index - 1].tolist())
            writer.writerow(row)


def _write_inlet_profile(
    path: Path,
    z_m: np.ndarray,
    measured: np.ndarray,
    predicted: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("z_m", "measured_u_over_u_h", "predicted_u_over_u_h"))
        writer.writerows(zip(z_m, measured, predicted, strict=True))


def _base_report(
    case,
    reference,
    criteria,
    levels,
    configs,
    inlet_metrics,
    collision_model,
):
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": reference_provenance(),
        "case": {
            **asdict(case),
            "coordinate_origin": "building footprint centre at ground level",
            "domain_bounds_m": {
                "x": [case.x_min_m, case.x_max_m],
                "y": [case.y_min_m, case.y_max_m],
                "z": [0.0, case.domain_xyz_m[2]],
            },
        },
        "reference": {
            "measurement_points": len(reference.points_xyz_m),
            "approach_profile_points": len(reference.approach_z_m),
            "reference_speed_at_building_height_m_s": reference.reference_speed_m_s,
            "fitted_power_alpha": reference.power_alpha,
        },
        "criteria": asdict(criteria),
        "collision_model": collision_model,
        "inlet_profile": {
            "prescribed_power_law": inlet_metrics,
            "empty_domain_at_building_centre": None,
        },
        "plan": [
            _plan_record(case, level, config) for level, config in zip(levels, configs, strict=True)
        ],
        "limitations": [
            "The inlet reproduces only a fitted mean power-law profile; measured turbulent "
            "fluctuations are not injected.",
            "The current XLB backend uses no-slip floor, top and side walls and an "
            "extrapolation outlet; boundary-condition sensitivity is still required.",
            "The building uses voxel/full-way bounce-back on a uniform Cartesian lattice.",
            "Only mean velocity is compared. AIJ turbulence statistics are retained for a "
            "later synthetic-inflow validation stage.",
            "A passing result is provisional and must not be presented as engineering "
            "validation until the listed limitations and an independent solver comparison "
            "are resolved.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        levels = _levels(args.cells_per_b)
    except argparse.ArgumentTypeError as exc:
        _parser().error(str(exc))

    case = AijCaseA()
    reference_dir = ensure_aij_case_a_reference(
        args.reference_dir,
        download=not args.no_download,
    )
    reference = load_aij_case_a_reference(reference_dir)
    exponent = reference.power_alpha if args.inlet_power_alpha is None else args.inlet_power_alpha
    configs = [
        case.config(
            level,
            lattice_wind=args.lattice_wind,
            flow_throughs=args.flow_throughs,
            average_flow_throughs=args.average_flow_throughs,
            average_samples=args.average_samples,
            inlet_power_alpha=exponent,
        )
        for level in levels
    ]
    criteria = ValidationCriteria()
    inlet_metrics = inlet_profile_metrics(reference, exponent, case)
    report = _base_report(
        case,
        reference,
        criteria,
        levels,
        configs,
        inlet_metrics,
        args.collision_model,
    )
    report["mode"] = "run" if args.run else "plan"

    run_metrics: list[dict[str, float]] = []
    predictions: list[np.ndarray] = []
    grid_records: list[dict[str, float | int]] = []
    time_record: dict[str, object] | None = None
    simulated_inlet_metrics: dict[str, float] | None = None
    execution_errors: list[dict[str, object]] = []

    if args.run:
        run_records = []
        for level, config in zip(levels, configs, strict=True):
            try:
                field, cache_hit, elapsed, cache_path = _run_xlb(
                    case,
                    config,
                    args.cache_dir,
                    args.force,
                    collision_model=args.collision_model,
                )
                metrics, predicted = prediction_metrics(field, reference, case, config)
            except Exception as exc:
                failure = {
                    "stage": "building_grid",
                    "cells_per_b": level,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                execution_errors.append(failure)
                run_records.append(failure)
                break
            predictions.append(predicted)
            run_metrics.append(metrics)
            run_records.append(
                {
                    "cells_per_b": level,
                    "metrics": metrics,
                    "cache_hit": cache_hit,
                    "elapsed_seconds": elapsed,
                    "cache_path": str(cache_path),
                }
            )
            if len(predictions) > 1:
                grid_records.append(
                    {
                        "coarse_cells_per_b": levels[len(predictions) - 2],
                        "fine_cells_per_b": level,
                        "u_relative_l2_drift": relative_prediction_drift(
                            predictions[-2][:, :1],
                            predictions[-1][:, :1],
                        ),
                        "vector_relative_l2_drift": relative_prediction_drift(
                            predictions[-2],
                            predictions[-1],
                        ),
                    }
                )

        completed_all_grids = len(predictions) == len(levels)
        if completed_all_grids and not args.skip_empty_domain_check:
            fine_config = configs[-1]
            try:
                empty_field, cache_hit, elapsed, cache_path = _run_xlb(
                    case,
                    fine_config,
                    args.cache_dir,
                    args.force,
                    geometry="empty",
                    collision_model=args.collision_model,
                )
                simulated_inlet_metrics, predicted_inlet = simulated_approach_profile_metrics(
                    empty_field,
                    reference,
                    case,
                    fine_config,
                )
                report["inlet_profile"]["empty_domain_at_building_centre"] = {
                    **simulated_inlet_metrics,
                    "cells_per_b": levels[-1],
                    "cache_hit": cache_hit,
                    "elapsed_seconds": elapsed,
                    "cache_path": str(cache_path),
                }
                _write_inlet_profile(
                    args.inlet_profile_out,
                    reference.approach_z_m,
                    reference.normalized_approach,
                    predicted_inlet,
                )
                report["inlet_profile_csv"] = str(args.inlet_profile_out)
            except Exception as exc:
                failure = {
                    "stage": "empty_domain_inlet",
                    "cells_per_b": levels[-1],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                execution_errors.append(failure)
                report["inlet_profile"]["empty_domain_at_building_centre"] = failure

        if completed_all_grids and args.time_check:
            fine_level = levels[-1]
            long_config = case.config(
                fine_level,
                lattice_wind=args.lattice_wind,
                flow_throughs=args.flow_throughs + args.average_flow_throughs,
                average_flow_throughs=min(
                    args.flow_throughs + args.average_flow_throughs,
                    2 * args.average_flow_throughs,
                ),
                average_samples=args.average_samples,
                inlet_power_alpha=exponent,
            )
            try:
                long_field, cache_hit, elapsed, cache_path = _run_xlb(
                    case,
                    long_config,
                    args.cache_dir,
                    args.force,
                    collision_model=args.collision_model,
                )
                long_metrics, long_prediction = prediction_metrics(
                    long_field,
                    reference,
                    case,
                    long_config,
                )
                time_record = {
                    "cells_per_b": fine_level,
                    "base_flow_throughs": args.flow_throughs,
                    "long_flow_throughs": (args.flow_throughs + args.average_flow_throughs),
                    "base_average_flow_throughs": args.average_flow_throughs,
                    "long_average_flow_throughs": 2 * args.average_flow_throughs,
                    "u_relative_l2_drift": relative_prediction_drift(
                        predictions[-1][:, :1],
                        long_prediction[:, :1],
                    ),
                    "vector_relative_l2_drift": relative_prediction_drift(
                        predictions[-1],
                        long_prediction,
                    ),
                    "long_run_metrics": long_metrics,
                    "cache_hit": cache_hit,
                    "elapsed_seconds": elapsed,
                    "cache_path": str(cache_path),
                }
            except Exception as exc:
                failure = {
                    "stage": "time_window",
                    "cells_per_b": fine_level,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                execution_errors.append(failure)
                time_record = failure

        if predictions:
            completed_levels = levels[: len(predictions)]
            _write_predictions(
                args.predictions,
                reference.points_xyz_m,
                reference.normalized_velocity,
                completed_levels,
                predictions,
            )
            report["predictions_csv"] = str(args.predictions)
        report["runs"] = run_records
        report["grid_convergence"] = grid_records
        report["time_convergence"] = time_record
        report["execution_errors"] = execution_errors

    grid_drifts = [float(record["u_relative_l2_drift"]) for record in grid_records]
    time_drift = (
        float(time_record["u_relative_l2_drift"])
        if time_record is not None and "u_relative_l2_drift" in time_record
        else None
    )
    status, checks = report_status(
        cells_per_b=levels[: len(run_metrics)] if args.run else [],
        run_metrics=run_metrics,
        grid_drifts=grid_drifts,
        time_window_drift=time_drift,
        inlet_relative_rmse=(
            simulated_inlet_metrics["relative_rmse"]
            if simulated_inlet_metrics is not None
            else None
        ),
        criteria=criteria,
    )
    if execution_errors:
        status = "failed"
    report["status"] = status
    report["checks"] = checks

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"AIJ Case A {report['mode']}: {status}")
    print(f"report: {args.out}")
    if args.run:
        if predictions:
            print(f"predictions: {args.predictions}")
        for failure in execution_errors:
            print(f"error[{failure['stage']}]: {failure['error_type']}: {failure['message']}")
    if execution_errors:
        return 1
    if args.strict and status != "provisional_pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
