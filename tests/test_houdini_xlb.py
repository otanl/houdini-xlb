from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest

from houdini_xlb import (
    XlbConfig,
    analysis_key,
    analyze_heightmap,
    rasterize_points,
    sop_code,
    worker_environment,
)
from houdini_xlb.cli import _configured_profile, _parser
from houdini_xlb.protocol import RESPONSE
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
    assert calls == [config.steps]
    assert not first.cache_hit
    assert second.cache_hit
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
    )
    assert "__PACKAGE_SRC__" not in code
    assert "__CACHE_DIR__" not in code
    assert "__PYTHON_EXE__" not in code
    assert "stale: geometry changed; press Run XLB" in code


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
