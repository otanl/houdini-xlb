from __future__ import annotations

import ast
import io
import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from houdini_xlb import (
    XlbConfig,
    analysis_key,
    analyze_heightmap,
    default_python_executable,
    load_cached_heightmap,
    rasterize_points,
    sop_code,
    worker_environment,
)
from houdini_xlb import houdini as houdini_api
from houdini_xlb.backend import _solid_indices
from houdini_xlb.cli import _configured_profile, _parser
from houdini_xlb.demo_study import (
    BASE_DESIGN,
    FIXED_BUILDINGS,
    GRID_NX,
    GRID_NY,
    MILESTONE_EVALUATIONS,
    MILESTONE_FRAMES,
    MIN_CLEARANCE_M,
    MOVABLE_TEMPLATES,
    PLAZA_BOUNDS,
    Massing,
    candidate_designs,
    heightmap_from_design,
    mean_speed_in_bounds,
    minimum_clearance,
    study_buildings,
    validate_design,
    validate_optimization,
)
from houdini_xlb.protocol import RESPONSE
from houdini_xlb.timeline import TimelineJob, TimelineScheduler
from houdini_xlb.worker import serve


def test_package_does_not_depend_on_windcfd_or_mokumitsu():
    package = Path(__file__).parents[1] / "src" / "houdini_xlb"
    forbidden = {"windcfd", "mokumitsu"}
    imports = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    assert imports.isdisjoint(forbidden)


def test_demo_candidate_space_is_collision_free_and_fixed_volume():
    designs = candidate_designs()
    assert len(designs) == 16
    assert len(set(designs)) == 16
    assert designs[0] == BASE_DESIGN
    baseline_buildings = study_buildings(BASE_DESIGN)
    baseline_volumes = tuple(building.volume for building in baseline_buildings)
    assert tuple((building.cx, building.cy) for building in baseline_buildings[:2]) == (
        (BASE_DESIGN[0], BASE_DESIGN[1]),
        (BASE_DESIGN[2], BASE_DESIGN[3]),
    )
    assert tuple(
        (building.width, building.depth, building.height) for building in baseline_buildings[:2]
    ) == tuple((template.width, template.depth, template.height) for template in MOVABLE_TEMPLATES)
    assert baseline_buildings[2:] == FIXED_BUILDINGS

    for design in designs:
        buildings = study_buildings(design)
        assert validate_design(design) >= MIN_CLEARANCE_M
        assert minimum_clearance(buildings) >= MIN_CLEARANCE_M
        assert tuple(building.volume for building in buildings) == baseline_volumes

    heightmap = heightmap_from_design(BASE_DESIGN)
    assert heightmap.shape == (GRID_NY, GRID_NX)
    assert heightmap.max() == pytest.approx(0.5)


def test_tracked_demo_optimization_is_valid_and_reproducible():
    path = Path(__file__).parents[1] / "examples" / "houdini_xlb_demo_optimization.json"
    optimization = json.loads(path.read_text(encoding="utf-8"))
    validate_optimization(optimization)
    assert len(optimization["evaluations"]) == 16
    assert [item["frame"] for item in optimization["milestones"]] == list(MILESTONE_FRAMES)
    assert [item["evaluation"] for item in optimization["milestones"]] == list(
        MILESTONE_EVALUATIONS
    )
    assert optimization["result"]["best_evaluation"] == 6
    assert optimization["result"]["design"] == [50.0, 30.0, 50.0, 70.0]
    assert optimization["baseline"]["metrics"]["comfort_fraction"] == pytest.approx(0.3083333333)
    assert optimization["result"]["metrics"]["comfort_fraction"] == pytest.approx(1.0)
    assert optimization["result"]["vent_retention"] == pytest.approx(1.9020064007)


def test_demo_zone_metric_uses_world_space_bounds():
    positions = np.asarray(
        [
            [PLAZA_BOUNDS[0], PLAZA_BOUNDS[1], 0.0],
            [70.0, 50.0, 0.0],
            [10.0, 10.0, 0.0],
        ]
    )
    speed = np.asarray([1.0, 3.0, 100.0])
    assert mean_speed_in_bounds(positions, speed, PLAZA_BOUNDS) == pytest.approx(2.0)
    assert minimum_clearance(
        (
            Massing(0.0, 0.0, 2.0, 2.0, 1.0),
            Massing(5.0, 0.0, 2.0, 2.0, 1.0),
        )
    ) == pytest.approx(3.0)


def test_profiles_and_cache_key_are_explicit():
    draft = XlbConfig.profile("draft")
    preview = XlbConfig.profile("preview")
    heightmap = np.zeros((12, 12), dtype=np.float32)
    assert draft.steps < preview.steps
    assert analysis_key(heightmap, draft) != analysis_key(heightmap, preview)
    with pytest.raises(ValueError):
        XlbConfig.profile("realtime")


def test_cached_analysis_avoids_second_solver_call(tmp_path):
    calls = []

    def fake_solver(heightmap, config):
        calls.append(config.steps)
        return np.ones_like(heightmap) * 0.25

    heightmap = np.zeros((10, 10), dtype=np.float32)
    heightmap[3:6, 4:7] = 0.4
    config = XlbConfig.profile("draft")
    first = analyze_heightmap(
        heightmap,
        config,
        cache_dir=tmp_path,
        solver=fake_solver,
        solver_signature="unit-fake-v1",
    )
    second = analyze_heightmap(
        heightmap,
        config,
        cache_dir=tmp_path,
        solver=fake_solver,
        solver_signature="unit-fake-v1",
    )
    cached = load_cached_heightmap(
        heightmap,
        config,
        cache_dir=tmp_path,
        backend_signature="custom:unit-fake-v1",
    )
    assert calls == [config.steps]
    assert not first.cache_hit
    assert second.cache_hit
    assert cached is not None and cached.cache_hit
    np.testing.assert_allclose(first.speed, second.speed)


def test_physical_profile_resolves_real_pedestrian_height():
    config = XlbConfig.profile("preview")
    dx, dy, dz = config.cell_sizes_m
    assert dx == pytest.approx(dy)
    assert dz == pytest.approx(dx, rel=config.isotropy_tolerance)
    assert config.resolved_pedestrian_height_m == pytest.approx(
        config.pedestrian_height_m, abs=dz / 2
    )
    assert config.reference_height_lattice == pytest.approx(config.reference_height_m / dz)
    with pytest.raises(ValueError, match="nearly cubic"):
        XlbConfig(grid_z=64)


def test_legacy_config_migrates_to_a_cubic_physical_domain():
    config = XlbConfig.from_dict(
        {
            "grid_x": 128,
            "grid_y": 128,
            "grid_z": 48,
            "reference_height": 0.3,
            "pedestrian_z": 4,
        }
    )
    assert config.domain_height_m == pytest.approx(37.5)
    assert config.reference_height_m == pytest.approx(11.25)
    assert config.pedestrian_height_m == pytest.approx(3.125)
    current_wins = XlbConfig.from_dict(
        {
            **config.to_dict(),
            "reference_height": 0.9,
            "pedestrian_z": 20,
        }
    )
    assert current_wins == config


def test_corrupt_cache_is_ignored_and_replaced(tmp_path):
    calls = 0

    def fake_solver(heightmap, _config):
        nonlocal calls
        calls += 1
        return np.full_like(heightmap, 0.1)

    heightmap = np.zeros((8, 8), dtype=np.float32)
    config = XlbConfig.profile("draft")
    first = analyze_heightmap(
        heightmap,
        config,
        cache_dir=tmp_path,
        solver=fake_solver,
        solver_signature="unit-corrupt-v1",
    )
    assert first.cache_path is not None
    np.savez_compressed(
        first.cache_path,
        speed=np.asarray([[np.nan]], dtype=np.float32),
        metadata=np.asarray(json.dumps(first.metadata())),
    )

    second = analyze_heightmap(
        heightmap,
        config,
        cache_dir=tmp_path,
        solver=fake_solver,
        solver_signature="unit-corrupt-v1",
    )
    assert calls == 2
    assert not second.cache_hit
    assert np.isfinite(second.speed).all()


def test_unstable_solver_result_is_rejected_before_caching(tmp_path):
    config = XlbConfig.profile("draft")
    heightmap = np.zeros((8, 8), dtype=np.float32)

    def unstable(heightmap, _config):
        return np.full_like(heightmap, config.wind * (config.max_speed_ratio + 1))

    with pytest.raises(RuntimeError, match="numerically unstable"):
        analyze_heightmap(
            heightmap,
            config,
            cache_dir=tmp_path,
            solver=unstable,
            solver_signature="unit-unstable-v1",
        )
    assert not list(tmp_path.glob("*.npz"))


def test_default_xlb_contract_rejects_the_wrong_raster_shape(tmp_path):
    config = XlbConfig.profile("draft")
    with pytest.raises(ValueError, match="must equal XLB lattice"):
        analyze_heightmap(np.zeros((8, 8), dtype=np.float32), config, cache_dir=tmp_path)


def test_custom_solver_cache_cannot_poison_the_xlb_namespace(tmp_path):
    config = XlbConfig.profile("draft")
    heightmap = np.zeros((8, 8), dtype=np.float32)

    analyze_heightmap(
        heightmap,
        config,
        cache_dir=tmp_path,
        solver=lambda field, _config: np.full_like(field, 0.1),
        solver_signature="unit-isolated-v1",
    )

    assert (
        load_cached_heightmap(
            heightmap,
            config,
            cache_dir=tmp_path,
            backend_signature="custom:unit-isolated-v1",
        )
        is not None
    )
    with pytest.raises(ValueError, match="must equal XLB lattice"):
        load_cached_heightmap(heightmap, config, cache_dir=tmp_path)


def test_backend_requires_geometry_at_the_exact_lattice_resolution():
    heightmap = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="rasterize source geometry directly"):
        _solid_indices(heightmap, (64, 64, 26))

    x, y, z = _solid_indices(heightmap, (32, 32, 13))
    assert not len(x) and not len(y) and not len(z)


def test_analyze_geometry_uses_the_exact_selected_profile_grid(monkeypatch):
    class Point:
        def __init__(self, position):
            self._position = position

        def position(self):
            return self._position

        def attribValue(self, _attribute):
            return 0

    class Geometry:
        def __init__(self):
            self._points = [
                Point((x, y, z))
                for z in (0.0, 6.0)
                for x, y in ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0))
            ]

        def findPointAttrib(self, name):
            return name if name == "class" else None

        def points(self):
            return self._points

    captured = {}

    class Client:
        def analyze(self, heightmap, config):
            captured["heightmap"] = heightmap
            captured["config"] = config
            return "analysis-result"

    monkeypatch.setattr(houdini_api, "session_client", lambda **_kwargs: Client())
    assert houdini_api.analyze_geometry(Geometry(), profile="preview") == "analysis-result"
    config = XlbConfig.profile("preview")
    assert captured["heightmap"].shape == (config.grid_y, config.grid_x)
    assert captured["heightmap"].max() == pytest.approx(6.0 / config.domain_height_m)
    assert captured["config"] == config

    with pytest.raises(ValueError, match="requires height-map shape"):
        houdini_api.analyze_geometry(Geometry(), profile="preview", ny=96, nx=96)


def test_rasterize_connected_piece_in_world_coordinates():
    points = np.asarray(
        [
            (2.0, 2.0, 0.0),
            (4.0, 2.0, 0.0),
            (4.0, 4.0, 6.0),
            (2.0, 4.0, 6.0),
        ]
    )
    heightmap = rasterize_points(points, np.zeros(4), 10, 10, 10.0, 10.0)
    assert heightmap.max() == pytest.approx(6.0)
    assert np.count_nonzero(heightmap) == 4


def test_worker_protocol_with_injected_solver(tmp_path):
    heightmap_path = tmp_path / "heightmap.npy"
    np.save(heightmap_path, np.zeros((8, 8), dtype=np.float32))
    request = {
        "op": "analyze",
        "heightmap_path": str(heightmap_path),
        "cache_dir": str(tmp_path / "cache"),
        "config": XlbConfig.profile("draft").to_dict(),
    }
    input_stream = io.StringIO(json.dumps(request) + "\nshutdown\n")
    output_stream = io.StringIO()

    def fake_solver(heightmap, _config):
        return np.full_like(heightmap, 0.1)

    serve(
        input_stream,
        output_stream,
        solver=fake_solver,
        solver_signature="unit-worker-v1",
    )
    lines = output_stream.getvalue().splitlines()
    response = json.loads(
        next(line[len(RESPONSE) :] for line in lines if line.startswith(RESPONSE))
    )
    assert response["ok"]
    assert not response["cache_hit"]
    assert response["shape"] == [8, 8]


def test_cli_profile_overrides_are_explicit():
    args = _parser().parse_args(
        [
            "heightmap.npy",
            "--profile",
            "draft",
            "--steps",
            "420",
            "--grid-z",
            "48",
            "--domain-height-m",
            "50",
        ]
    )
    config = _configured_profile(args)
    assert config.steps == 420
    assert config.grid_z == 48
    assert config.domain_height_m == 50
    assert config.grid_x == XlbConfig.profile("draft").grid_x


def test_houdini_sop_template_resolves_portable_runtime_paths():
    code = sop_code(
        control_path="/obj/demo/xlb_solver",
        refresh_path="/obj/demo/xlb_result",
        merge_buildings=False,
        role="step",
    )
    assert "__PACKAGE_SRC__" not in code
    assert "__CACHE_DIR__" not in code
    assert "__PYTHON_EXE__" not in code
    assert "__CONTROL_PATH__" not in code
    assert "__REFRESH_PATH__" not in code
    assert "cook_solver_sop" in code
    assert 'control_path = r"/obj/demo/xlb_solver"' in code
    assert 'refresh_path=r"/obj/demo/xlb_result"' in code
    assert "merge_buildings=False" in code
    assert 'role="step"' in code
    assert "Run XLB" not in code
    assert 'hou.getenv("HIP")' in code
    assert "HOUDINI_XLB_SOURCE" in code
    assert "HOUDINI_XLB_CACHE" in code
    assert "HOUDINI_XLB_PYTHON" in code
    assert "default_python_executable(search_root=project_root)" in code


def test_demo_builder_uses_real_solver_sop_prev_frame_network():
    root = Path(__file__).parents[1]
    source = (root / "houdini" / "build_demo_hip.py").read_text(encoding="utf-8")
    timeline = (root / "src" / "houdini_xlb" / "timeline.py").read_text(encoding="utf-8")
    assert 'createNode("solver", "xlb_solver")' in source
    assert 'solver_network.node("Prev_Frame")' in source
    assert 'solver_network.node("Input_2")' in source
    assert 'solver_network.node("OUT").setFirstInput(step)' in source
    assert 'role="init"' in source
    assert 'role="step"' in source
    assert 'role="display"' in source
    assert "_constant_keys" in source
    assert "MILESTONE_FRAMES" in source
    assert "optimization_path" in source
    assert "FPS = 12.0" in source
    assert "hou.setFps(FPS)" in source
    assert "_houdini_xlb_display_state" not in timeline
    assert '("xlb_solver_state", 1)' in timeline


def _timeline_job(
    tmp_path: Path,
    signature: str,
    *,
    frame: int,
    kind: str = "auto",
) -> TimelineJob:
    return TimelineJob(
        node_path="/obj/test/xlb",
        signature=signature,
        heightmap=np.zeros((8, 8), dtype=np.float32),
        config=XlbConfig.profile("draft"),
        cache_dir=tmp_path,
        python_executable=tmp_path / "python.exe",
        frame=frame,
        kind=kind,
    )


def test_timeline_scheduler_debounces_and_keeps_only_latest(tmp_path):
    scheduler = TimelineScheduler()
    submitted = []
    completed = []
    futures: dict[str, Future] = {}

    def submit(job):
        submitted.append(job.signature)
        future = Future()
        futures[job.signature] = future
        return future

    def complete(job, _result, error):
        completed.append((job.signature, error))

    first = _timeline_job(tmp_path, "first", frame=1)
    scheduler.request(first, debounce_s=0.75, now=0.0)
    scheduler.tick(submit, complete, now=0.5)
    assert submitted == []
    scheduler.tick(submit, complete, now=0.8)
    assert submitted == ["first"]
    assert scheduler.status(first.node_path, first.signature) == "running"

    scheduler.request(_timeline_job(tmp_path, "middle", frame=2), now=0.9)
    latest = _timeline_job(tmp_path, "latest", frame=3)
    scheduler.request(latest, now=1.0)
    assert scheduler.status(latest.node_path, latest.signature) == "queued"
    futures["first"].set_result(None)
    scheduler.tick(submit, complete, now=1.0)
    assert submitted == ["first", "latest"]
    assert completed == [("first", None)]


def test_timeline_bake_deduplicates_and_cancel_stops_after_active(tmp_path):
    scheduler = TimelineScheduler()
    submitted = []
    future = Future()

    def submit(job):
        submitted.append(job.signature)
        return future

    jobs = [
        _timeline_job(tmp_path, "same", frame=1, kind="bake"),
        _timeline_job(tmp_path, "same", frame=2, kind="bake"),
        _timeline_job(tmp_path, "next", frame=3, kind="bake"),
    ]
    assert scheduler.enqueue_bake(jobs[0].node_path, jobs) == 2
    scheduler.tick(
        submit,
        lambda *_args: None,
        now=0.0,
        allow_submit=False,
    )
    assert submitted == []
    scheduler.tick(submit, lambda *_args: None, now=0.0)
    assert submitted == ["same"]
    scheduler.cancel_bake(jobs[0].node_path)
    future.set_result(None)
    scheduler.tick(submit, lambda *_args: None, now=0.1)
    assert submitted == ["same"]


def test_worker_environment_does_not_leak_houdini_python(tmp_path):
    environment = worker_environment(
        {
            "PYTHONHOME": r"G:\Houdini\python311",
            "PYTHONPATH": r"G:\Houdini\python311\Lib",
            "PYTHONEXECUTABLE": "hython.exe",
            "HOUDINI_XLB_PYTHONPATH": str(tmp_path / "extra"),
            "PATH": "kept",
        },
        source_root=tmp_path / "src",
    )
    assert "PYTHONHOME" not in environment
    assert "PYTHONEXECUTABLE" not in environment
    assert r"G:\Houdini" not in environment["PYTHONPATH"]
    assert str(tmp_path / "src") in environment["PYTHONPATH"]
    assert str(tmp_path / "extra") in environment["PYTHONPATH"]
    assert environment["PATH"] == "kept"


def test_default_python_searches_upward_from_the_hip_project(monkeypatch, tmp_path):
    monkeypatch.delenv("HOUDINI_XLB_PYTHON", raising=False)
    executable = tmp_path / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    nested = tmp_path / "examples" / "nested"
    nested.mkdir(parents=True)
    assert default_python_executable(search_root=nested) == executable.resolve()


def test_session_client_restarts_when_the_hip_cache_changes(monkeypatch, tmp_path):
    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

    class FakeClient:
        def __init__(self, *, cache_dir, python_executable):
            self.cache_dir = Path(cache_dir).resolve()
            self.python_executable = Path(python_executable).resolve()
            self.process = FakeProcess()
            self.closed = False

        def close(self):
            self.closed = True
            self.process.returncode = 0

    fake_hou = SimpleNamespace(session=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "hou", fake_hou)
    monkeypatch.setattr(houdini_api, "XlbWorkerClient", FakeClient)
    python = tmp_path / "python.exe"
    first = houdini_api.session_client(
        cache_dir=tmp_path / "cache-a",
        python_executable=python,
    )
    assert (
        houdini_api.session_client(
            cache_dir=tmp_path / "cache-a",
            python_executable=python,
        )
        is first
    )

    second = houdini_api.session_client(
        cache_dir=tmp_path / "cache-b",
        python_executable=python,
    )
    assert second is not first
    assert first.closed
