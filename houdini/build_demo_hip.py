"""Build a minimal timeline-driven Houdini/XLB scene."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from houdini_xlb.houdini_sop import install_parameters, sop_code  # noqa: E402

LENGTH_X = 100.0
LENGTH_Y = 100.0
NY = 96
NX = 96


def default_worker_python() -> Path:
    """Find the project venv in both standalone and monorepo checkouts."""
    configured = os.environ.get("HOUDINI_XLB_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT.parent.parent / ".venv" / "Scripts" / "python.exe",
    )
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[0],
    )


def _linear_keys(parm, values: tuple[tuple[int, float], ...]) -> None:
    for frame, value in values:
        key = hou.Keyframe()
        key.setFrame(frame)
        key.setValue(value)
        key.setExpression("linear()", hou.exprLanguage.Hscript)
        parm.setKeyframe(key)


def build_scene(
    name: str = "houdini_xlb",
    *,
    python_executable: Path | None = None,
    cache_dir: Path | None = None,
) -> hou.SopNode:
    """Create animated boxes and a Prev_Frame-driven XLB Solver SOP."""
    python_executable = (python_executable or default_worker_python()).resolve()
    cache_dir = (cache_dir or PROJECT_ROOT / "artifacts" / "cache" / "xlb").resolve()
    container = hou.node("/obj").createNode("geo", name, run_init_scripts=False)

    ground = container.createNode("grid", "ground")
    ground.parmTuple("size").set((LENGTH_X, LENGTH_Y))
    ground.parmTuple("t").set((LENGTH_X / 2, LENGTH_Y / 2, 0.0))
    ground.parm("orient").set("xy")
    ground.parm("rows").set(NY)
    ground.parm("cols").set(NX)

    initial = (
        (28.0, 42.0, 12.0, 18.0, 12.0),
        (48.0, 58.0, 16.0, 12.0, 18.0),
        (67.0, 38.0, 11.0, 20.0, 9.0),
    )
    boxes = []
    for index, (cx, cy, width, depth, height) in enumerate(initial):
        box = container.createNode("box", f"building{index}")
        box.parmTuple("size").set((width, depth, height))
        box.parmTuple("t").set((cx, cy, height / 2))
        box.parm("tz").setExpression("ch('sizez')/2")
        boxes.append(box)

    _linear_keys(
        boxes[0].parm("tx"),
        ((1, 24.0), (12, 31.0), (24, 38.0), (36, 28.0)),
    )
    _linear_keys(
        boxes[0].parm("sizez"),
        ((1, 10.0), (12, 15.0), (24, 22.0), (36, 13.0)),
    )
    _linear_keys(
        boxes[1].parm("ty"),
        ((1, 54.0), (12, 62.0), (24, 51.0), (36, 58.0)),
    )
    _linear_keys(
        boxes[2].parm("tx"),
        ((1, 69.0), (12, 63.0), (24, 71.0), (36, 66.0)),
    )

    merge = container.createNode("merge", "buildings")
    for index, box in enumerate(boxes):
        merge.setInput(index, box)
    connectivity = container.createNode("connectivity", "connected_buildings")
    connectivity.setFirstInput(merge)
    colour = container.createNode("color", "building_colour")
    colour.setFirstInput(connectivity)
    colour.parmTuple("color").set((0.15, 0.55, 0.95))

    init = container.createNode("python", "xlb_init")
    init.setInput(0, ground)
    init.setInput(1, colour)

    solver = container.createNode("solver", "xlb_solver")
    solver.setInput(0, init)
    solver.setInput(1, colour)
    solver.parm("startframe").set(1)
    solver.parm("cacheenabled").set(1)
    if solver.parm("cachemaxsize") is not None:
        solver.parm("cachemaxsize").set(512)

    result = container.createNode("python", "xlb_result")
    result.setInput(0, solver)
    result.setInput(1, colour)

    install_parameters(
        solver,
        package_src=PACKAGE_SRC,
        cache_dir=cache_dir,
        python_executable=python_executable,
        refresh_path=result.path(),
    )

    init.parm("python").set(
        sop_code(
            package_src=PACKAGE_SRC,
            cache_dir=cache_dir,
            python_executable=python_executable,
            control_path=solver.path(),
            refresh_path=result.path(),
            merge_buildings=False,
            role="init",
        )
    )
    solver_network = solver.node("d/s")
    step = solver_network.createNode("python", "xlb_step")
    step.setInput(0, solver_network.node("Prev_Frame"))
    step.setInput(1, solver_network.node("Input_2"))
    step.parm("python").set(
        sop_code(
            package_src=PACKAGE_SRC,
            cache_dir=cache_dir,
            python_executable=python_executable,
            control_path=solver.path(),
            refresh_path=result.path(),
            merge_buildings=False,
            role="step",
        )
    )
    solver_network.node("OUT").setFirstInput(step)
    solver_network.layoutChildren()

    result.parm("python").set(
        sop_code(
            package_src=PACKAGE_SRC,
            cache_dir=cache_dir,
            python_executable=python_executable,
            control_path=solver.path(),
            refresh_path=result.path(),
            merge_buildings=True,
            role="display",
        )
    )
    result.setDisplayFlag(True)
    result.setRenderFlag(True)

    note = container.createStickyNote()
    note.setText(
        "HOUDINI × XLB — TIMELINE DESIGN STUDY\n"
        "1. Select xlb_solver: Prev_Frame carries wind/state in the Simulation Cache.\n"
        "2. Scrub while paused: the current design auto-analyses after 0.75 s.\n"
        "3. Bake Range fills the SHA cache; playback launches no new XLB jobs.\n"
        "Frames are design alternatives, not physical CFD time."
    )
    note.setSize(hou.Vector2(5.0, 2.0))
    container.layoutChildren()
    hou.playbar.setFrameRange(1, 36)
    hou.playbar.setPlaybackRange(1, 36)
    hou.setFrame(1)
    return solver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "examples" / "houdini_xlb_demo.hip",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        help="external Python 3.12 executable (default: HOUDINI_XLB_PYTHON or nearest .venv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cache" / "xlb",
        help="XLB result cache",
    )
    parser.add_argument(
        "--run-xlb-smoke",
        action="store_true",
        help="also execute the draft profile through the external GPU worker",
    )
    args = parser.parse_args()
    output = args.out.resolve()
    python_executable = (args.python_executable or default_worker_python()).resolve()
    if not python_executable.exists():
        raise FileNotFoundError(
            f"external Python not found at {python_executable}; create .venv with Python 3.12 "
            "or pass --python-executable"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    solver = build_scene(
        python_executable=python_executable,
        cache_dir=args.cache_dir,
    )
    result = solver.parent().node("xlb_result")
    if result is None:
        raise RuntimeError("xlb_result node was not created")
    try:
        try:
            result.cook(force=True)
        except hou.OperationFailed:
            print("\n".join(result.errors()))
            raise
        status = result.geometry().attribValue("xlb_status")
        expected_initial = {
            "current",
            "not-baked: pause to analyze or use Bake Range",
        }
        if status not in expected_initial:
            raise RuntimeError(f"unexpected initial XLB SOP status: {status}")
        if args.run_xlb_smoke:
            solver.parm("profile").set(0)
            solver.parm("runxlb").pressButton()
            result.cook(force=True)
            status = result.geometry().attribValue("xlb_status")
            if status != "current":
                raise RuntimeError(f"XLB smoke result is not current: {status}")
            print(
                "XLB smoke current; "
                f"elapsed={result.geometry().attribValue('xlb_elapsed_s'):.3f}s "
                f"cache_hit={result.geometry().attribValue('xlb_cache_hit')}"
            )
        hou.hipFile.save(str(output))
        print(f"saved {output}; verified Solver status={status}")
    finally:
        client = getattr(hou.session, "_houdini_xlb_client", None)
        if client is not None:
            client.close()
            delattr(hou.session, "_houdini_xlb_client")


if __name__ == "__main__":
    main()
