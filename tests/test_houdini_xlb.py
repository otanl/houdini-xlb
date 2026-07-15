from __future__ import annotations

import ast
import io
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from houdini_xlb import (
    XlbConfig,
    analysis_key,
    analyze_heightmap,
    load_cached_heightmap,
    rasterize_points,
    sop_code,
    worker_environment,
)
from houdini_xlb.cli import _configured_profile, _parser
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
    first = analyze_heightmap(heightmap, config, cache_dir=tmp_path, solver=fake_solver)
    second = analyze_heightmap(heightmap, config, cache_dir=tmp_path, solver=fake_solver)
    cached = load_cached_heightmap(heightmap, config, cache_dir=tmp_path)
    assert calls == [config.steps]
    assert not first.cache_hit
    assert second.cache_hit
    assert cached is not None and cached.cache_hit
    np.testing.assert_allclose(first.speed, second.speed)


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

    serve(input_stream, output_stream, solver=fake_solver)
    lines = output_stream.getvalue().splitlines()
    response = json.loads(
        next(line[len(RESPONSE) :] for line in lines if line.startswith(RESPONSE))
    )
    assert response["ok"]
    assert not response["cache_hit"]
    assert response["shape"] == [8, 8]


def test_cli_profile_overrides_are_explicit():
    args = _parser().parse_args(
        ["heightmap.npy", "--profile", "draft", "--steps", "420", "--grid-z", "44"]
    )
    config = _configured_profile(args)
    assert config.steps == 420
    assert config.grid_z == 44
    assert config.grid_x == XlbConfig.profile("draft").grid_x


def test_houdini_sop_template_has_no_unresolved_runtime_paths(tmp_path):
    code = sop_code(
        package_src=tmp_path / "src",
        cache_dir=tmp_path / "cache",
        python_executable=tmp_path / "python.exe",
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
    assert "END_FRAME = 120" in source
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
