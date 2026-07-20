from __future__ import annotations

import json

import numpy as np
import pytest

from houdini_xlb import AijCaseA, backend, validation_cli
from houdini_xlb.validation import (
    AijCaseAReference,
    ValidationCriteria,
    fit_power_law_alpha,
    read_cached_velocity,
    relative_prediction_drift,
    report_status,
    trilinear_sample_velocity,
    validation_cache_key,
    write_cached_velocity,
)


def test_case_grid_preserves_physical_geometry_and_advective_time():
    case = AijCaseA()
    coarse = case.config(
        4,
        lattice_wind=0.05,
        flow_throughs=1.5,
        average_flow_throughs=0.5,
    )
    fine = case.config(
        8,
        lattice_wind=0.05,
        flow_throughs=1.5,
        average_flow_throughs=0.5,
    )

    assert coarse.grid_xyz == (100, 48, 40)
    assert fine.grid_xyz == (200, 96, 80)
    assert coarse.cell_sizes_m == pytest.approx((0.02, 0.02, 0.02))
    assert fine.cell_sizes_m == pytest.approx((0.01, 0.01, 0.01))
    assert coarse.steps * coarse.wind / coarse.grid_x == pytest.approx(1.5)
    assert fine.steps * fine.wind / fine.grid_x == pytest.approx(1.5)
    assert coarse.average_window * coarse.wind / coarse.grid_x == pytest.approx(0.5)
    assert fine.average_window * fine.wind / fine.grid_x == pytest.approx(0.5)

    coarse_map = case.heightmap(coarse)
    fine_map = case.heightmap(fine)
    assert np.count_nonzero(coarse_map) == 4**2
    assert np.count_nonzero(fine_map) == 8**2
    assert coarse_map.max() == pytest.approx(0.2)
    assert fine_map.max() == pytest.approx(0.2)


def test_power_law_fit_recovers_known_exponent():
    height = 0.16
    z = np.asarray([0.01, 0.02, 0.04, 0.08, 0.16])
    u = 4.5 * (z / height) ** 0.22
    assert fit_power_law_alpha(z, u, height) == pytest.approx(0.22)


def test_trilinear_sampling_is_exact_for_affine_velocity_field():
    case = AijCaseA()
    config = case.config(2)
    x, y, z = case.coordinates(config)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    field = np.stack(
        (
            1.0 + 2.0 * xx + 3.0 * yy + 4.0 * zz,
            -0.5 * xx + yy,
            0.25 * yy - 2.0 * zz,
        )
    ).astype(np.float32)
    points = np.asarray(
        [
            [-0.03, 0.01, 0.07],
            [0.13, -0.11, 0.21],
            [0.255, -0.155, 0.275],
        ]
    )
    sampled = trilinear_sample_velocity(field, points, case, config)
    expected = np.column_stack(
        (
            1.0 + 2.0 * points[:, 0] + 3.0 * points[:, 1] + 4.0 * points[:, 2],
            -0.5 * points[:, 0] + points[:, 1],
            0.25 * points[:, 1] - 2.0 * points[:, 2],
        )
    )
    assert sampled == pytest.approx(expected, abs=2e-7)


def test_relative_prediction_drift_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share a shape"):
        relative_prediction_drift(np.ones((2, 3)), np.ones((3, 2)))


def test_velocity_cache_round_trip_and_corruption(tmp_path):
    case = AijCaseA()
    config = case.config(2)
    field = np.zeros((3, config.grid_z, config.grid_y, config.grid_x), dtype=np.float32)
    field[0] = 0.05
    path = tmp_path / "field.npz"

    write_cached_velocity(path, field, config)
    assert read_cached_velocity(path, config) == pytest.approx(field)

    path.write_bytes(b"not-an-npz")
    assert read_cached_velocity(path, config) is None
    assert validation_cache_key(case, config, "test", geometry="building") != (
        validation_cache_key(case, config, "test", geometry="empty")
    )
    assert validation_cache_key(
        case, config, "test", geometry="building", collision_model="KBC"
    ) != validation_cache_key(
        case,
        config,
        "test",
        geometry="building",
        collision_model="SmagorinskyLESBGK",
    )


def test_pedestrian_speed_slice_is_interpolated_from_public_velocity_field(monkeypatch):
    captured = {}

    def fake_velocity(heightmap, **kwargs):
        captured.update(kwargs)
        field = np.zeros((3, 8, 8, 8), dtype=np.float32)
        field[0, 1] = 1.0
        field[0, 2] = 3.0
        return field

    monkeypatch.setattr(backend, "simulate_velocity_field_xlb", fake_velocity)
    speed = backend.simulate_heightmap_xlb(
        np.zeros((8, 8), dtype=np.float32),
        grid_xyz=(8, 8, 8),
        wind=0.05,
        reynolds=8000,
        steps=10,
        pedestrian_z=1.25,
        precision="FP32FP32",
        average_window=0,
        average_every=1,
        reference_height_lattice=2.0,
        inlet_profile="power_law",
        inlet_power_alpha=0.2,
        collision_model="SmagorinskyLESBGK",
    )
    assert speed == pytest.approx(np.full((8, 8), 1.5))
    assert captured["inlet_profile"] == "power_law"
    assert captured["inlet_power_alpha"] == pytest.approx(0.2)
    assert captured["collision_model"] == "SmagorinskyLESBGK"


def test_nonfluid_macroscopic_artifacts_are_masked():
    field = np.ones((3, 6, 7, 8), dtype=np.float32)
    solid = (
        np.asarray([2]),
        np.asarray([3]),
        np.asarray([4]),
    )
    masked = backend._mask_nonfluid_velocity(field, solid)

    assert np.all(masked[:, 4, 3, 2] == 0)
    assert np.all(masked[:, (0, -1), :, :] == 0)
    assert np.all(masked[:, :, (0, -1), :] == 0)
    assert np.all(masked[:, 2, 2, 2] == 1)
    assert np.all(field == 1)


def test_report_requires_grid_time_profile_reference_and_resolution_gates():
    criteria = ValidationCriteria()
    metrics = [{"u_relative_l2": 0.1}]
    status, checks = report_status(
        cells_per_b=[10, 15, 20],
        run_metrics=metrics,
        grid_drifts=[0.02, 0.02],
        time_window_drift=0.02,
        inlet_relative_rmse=0.05,
        criteria=criteria,
    )
    assert status == "provisional_pass"
    assert all(checks.values())

    incomplete, incomplete_checks = report_status(
        cells_per_b=[8, 12, 16],
        run_metrics=metrics,
        grid_drifts=[0.02, 0.02],
        time_window_drift=None,
        inlet_relative_rmse=0.05,
        criteria=criteria,
    )
    assert incomplete == "incomplete"
    assert incomplete_checks["time_window_convergence"] is None
    assert incomplete_checks["reference_resolution"] is False

    failed, failed_checks = report_status(
        cells_per_b=[10, 15, 20],
        run_metrics=metrics,
        grid_drifts=[0.02, 0.04],
        time_window_drift=0.02,
        inlet_relative_rmse=0.05,
        criteria=criteria,
    )
    assert failed == "provisional_fail"
    assert failed_checks["grid_convergence"] is False

    missing_inlet, missing_inlet_checks = report_status(
        cells_per_b=[10, 15, 20],
        run_metrics=metrics,
        grid_drifts=[0.02, 0.02],
        time_window_drift=0.02,
        inlet_relative_rmse=None,
        criteria=criteria,
    )
    assert missing_inlet == "incomplete"
    assert missing_inlet_checks["inlet_profile"] is None


def test_cli_writes_failed_report_when_solver_is_unstable(monkeypatch, tmp_path):
    reference = AijCaseAReference(
        approach_z_m=np.asarray([0.08, 0.16]),
        approach_u_m_s=np.asarray([3.5, 4.5]),
        points_xyz_m=np.asarray([[-0.06, 0.0, 0.08], [0.06, 0.0, 0.08]]),
        velocity_m_s=np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        reference_speed_m_s=4.5,
        power_alpha=0.2,
    )
    monkeypatch.setattr(
        validation_cli,
        "ensure_aij_case_a_reference",
        lambda directory, download: tmp_path,
    )
    monkeypatch.setattr(
        validation_cli,
        "load_aij_case_a_reference",
        lambda directory: reference,
    )

    def unstable(*args, **kwargs):
        raise RuntimeError("non-finite field")

    monkeypatch.setattr(validation_cli, "_run_xlb", unstable)
    report_path = tmp_path / "failed.json"
    exit_code = validation_cli.main(
        [
            "--run",
            "--cells-per-b",
            "2,3",
            "--collision-model",
            "SmagorinskyLESBGK",
            "--out",
            str(report_path),
            "--no-download",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["collision_model"] == "SmagorinskyLESBGK"
    assert report["execution_errors"] == [
        {
            "stage": "building_grid",
            "cells_per_b": 2,
            "error_type": "RuntimeError",
            "message": "non-finite field",
        }
    ]
