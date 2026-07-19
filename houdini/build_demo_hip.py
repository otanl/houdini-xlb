"""Build the constrained XLB layout-optimization demo scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import hou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT_ROOT / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from houdini_xlb.config import profile_names  # noqa: E402
from houdini_xlb.demo_study import (  # noqa: E402
    END_FRAME,
    MILESTONE_FRAMES,
    PLAZA_BOUNDS,
    START_FRAME,
    VENT_BOUNDS,
    study_buildings,
    validate_design,
    validate_optimization,
)
from houdini_xlb.houdini_sop import install_parameters, sop_code  # noqa: E402

LENGTH_X = 100.0
LENGTH_Y = 100.0
NY = 96
NX = 96
FPS = 12.0
DEFAULT_OPTIMIZATION = PROJECT_ROOT / "examples" / "houdini_xlb_demo_optimization.json"


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


def load_optimization(path: Path | None = None) -> dict[str, object]:
    """Load and validate the XLB-generated best-so-far trajectory."""

    source = (path or DEFAULT_OPTIMIZATION).resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"optimization result not found at {source}; run scripts/optimize_demo.py"
        )
    data = json.loads(source.read_text(encoding="utf-8"))
    validate_optimization(data)
    return data


def _constant_keys(parm, values: tuple[tuple[int, float], ...]) -> None:
    for frame, value in values:
        key = hou.Keyframe()
        key.setFrame(frame)
        key.setValue(value)
        key.setExpression("constant()", hou.exprLanguage.Hscript)
        parm.setKeyframe(key)


def _outline_rectangle(container, name, bounds, colour):
    """Create a thin raised rectangle used only as a viewport measurement guide."""

    xmin, ymin, xmax, ymax = bounds
    thickness = 0.7
    height = 0.25
    segments = (
        ((xmax - xmin, thickness, height), ((xmin + xmax) / 2, ymin, 0.3)),
        ((xmax - xmin, thickness, height), ((xmin + xmax) / 2, ymax, 0.3)),
        ((thickness, ymax - ymin, height), (xmin, (ymin + ymax) / 2, 0.3)),
        ((thickness, ymax - ymin, height), (xmax, (ymin + ymax) / 2, 0.3)),
    )
    merge = container.createNode("merge", f"{name}_outline")
    for index, (size, position) in enumerate(segments):
        segment = container.createNode("box", f"{name}_edge{index}")
        segment.parmTuple("size").set(size)
        segment.parmTuple("t").set(position)
        merge.setInput(index, segment)
    colour_node = container.createNode("color", f"{name}_colour")
    colour_node.setFirstInput(merge)
    colour_node.parmTuple("color").set(colour)
    return colour_node


def build_scene(
    name: str = "houdini_xlb",
    *,
    optimization_path: Path | None = None,
) -> hou.SopNode:
    """Create a constrained best-so-far study and Prev_Frame-driven XLB Solver SOP."""
    optimization = load_optimization(optimization_path)
    milestones = optimization["milestones"]
    designs = [tuple(map(float, milestone["design"])) for milestone in milestones]
    minimum_clearance = min(validate_design(design) for design in designs)
    hou.setFps(FPS)
    container = hou.node("/obj").createNode("geo", name, run_init_scripts=False)

    ground = container.createNode("grid", "ground")
    ground.parmTuple("size").set((LENGTH_X, LENGTH_Y))
    ground.parmTuple("t").set((LENGTH_X / 2, LENGTH_Y / 2, 0.0))
    ground.parm("orient").set("xy")
    ground.parm("rows").set(NY)
    ground.parm("cols").set(NX)

    boxes = []
    for index, massing in enumerate(study_buildings(designs[0])):
        box = container.createNode("box", f"building{index}")
        box.parmTuple("size").set((massing.width, massing.depth, massing.height))
        box.parmTuple("t").set((massing.cx, massing.cy, massing.height / 2))
        box.parm("tz").setExpression("ch('sizez')/2")
        boxes.append(box)

    for building_index, design_indices in enumerate(((0, 1), (2, 3))):
        for parm_name, design_index in zip(
            ("tx", "ty"),
            design_indices,
            strict=True,
        ):
            _constant_keys(
                boxes[building_index].parm(parm_name),
                tuple(
                    (frame, design[design_index])
                    for frame, design in zip(MILESTONE_FRAMES, designs, strict=True)
                ),
            )

    movable_merge = container.createNode("merge", "movable_buildings")
    movable_merge.setInput(0, boxes[0])
    movable_merge.setInput(1, boxes[1])
    movable_colour = container.createNode("color", "movable_colour")
    movable_colour.setFirstInput(movable_merge)
    movable_colour.parmTuple("color").set((0.15, 0.55, 0.95))

    fixed_merge = container.createNode("merge", "fixed_buildings")
    fixed_merge.setInput(0, boxes[2])
    fixed_merge.setInput(1, boxes[3])
    fixed_colour = container.createNode("color", "fixed_colour")
    fixed_colour.setFirstInput(fixed_merge)
    fixed_colour.parmTuple("color").set((0.32, 0.39, 0.46))

    merge = container.createNode("merge", "buildings")
    merge.setInput(0, movable_colour)
    merge.setInput(1, fixed_colour)
    connectivity = container.createNode("connectivity", "connected_buildings")
    connectivity.setFirstInput(merge)
    building_geometry = connectivity

    init = container.createNode("python", "xlb_init")
    init.setInput(0, ground)
    init.setInput(1, building_geometry)

    solver = container.createNode("solver", "xlb_solver")
    solver.setInput(0, init)
    solver.setInput(1, building_geometry)
    solver.parm("startframe").set(1)
    solver.parm("cacheenabled").set(1)
    if solver.parm("cachemaxsize") is not None:
        solver.parm("cachemaxsize").set(512)

    result = container.createNode("python", "xlb_result")
    result.setInput(0, solver)
    result.setInput(1, building_geometry)

    plaza_guide = _outline_rectangle(
        container,
        "plaza",
        PLAZA_BOUNDS,
        (1.0, 0.72, 0.18),
    )
    vent_guide = _outline_rectangle(
        container,
        "vent_route",
        VENT_BOUNDS,
        (0.18, 0.88, 0.92),
    )
    display = container.createNode("merge", "study_display")
    display.setInput(0, result)
    display.setInput(1, plaza_guide)
    display.setInput(2, vent_guide)

    install_parameters(
        solver,
        refresh_path=display.path(),
    )
    profile = str(optimization["solver"]["profile"])
    solver.parm("profile").set(profile_names().index(profile))
    solver.parm("bakestart").set(START_FRAME)
    solver.parm("bakeend").set(END_FRAME)
    solver.parm("vmax").set(0.10)

    init.parm("python").set(
        sop_code(
            control_path=solver.path(),
            refresh_path=display.path(),
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
            control_path=solver.path(),
            refresh_path=display.path(),
            merge_buildings=False,
            role="step",
        )
    )
    solver_network.node("OUT").setFirstInput(step)
    solver_network.layoutChildren()

    result.parm("python").set(
        sop_code(
            control_path=solver.path(),
            refresh_path=display.path(),
            merge_buildings=True,
            role="display",
        )
    )
    display.setDisplayFlag(True)
    display.setRenderFlag(True)

    note = container.createStickyNote()
    note.setText(
        "HOUDINI + XLB — CONSTRAINED LAYOUT OPTIMIZATION\n"
        "Two blue blocks move in four variables; two gray blocks remain fixed.\n"
        "16 collision-free layouts are evaluated with XLB; the timeline shows only "
        "best-so-far milestones.\n"
        "Objective: maximize plaza cells in 0.55 <= U/Uin <= 1.20.\n"
        "Hard constraint: central ventilation route >= 95% of baseline; "
        f"minimum clearance = {minimum_clearance:.1f} m.\n"
        "Study CFD: 96x96x38, 2400 steps, result height 1.5 m.\n"
        "Yellow = plaza target; cyan = ventilation route.\n"
        "Select xlb_solver and use Bake Range to populate the SHA cache.\n"
        "Timeline frames are optimization milestones, not physical CFD time."
    )
    note.setSize(hou.Vector2(7.5, 3.7))
    container.layoutChildren()
    hou.playbar.setFrameRange(START_FRAME, END_FRAME)
    hou.playbar.setPlaybackRange(START_FRAME, END_FRAME)
    hou.setFrame(START_FRAME)
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
        "--optimization",
        type=Path,
        default=DEFAULT_OPTIMIZATION,
        help="optimization JSON produced by scripts/optimize_demo.py",
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
    os.environ["HOUDINI_XLB_PYTHON"] = str(python_executable)
    os.environ["HOUDINI_XLB_CACHE"] = str(args.cache_dir.resolve())
    solver = build_scene(
        optimization_path=args.optimization,
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
            original_profile = int(solver.evalParm("profile"))
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
            solver.parm("profile").set(original_profile)
        hou.hipFile.save(str(output))
        print(f"saved {output}; verified Solver status={status}")
    finally:
        client = getattr(hou.session, "_houdini_xlb_client", None)
        if client is not None:
            client.close()
            delattr(hou.session, "_houdini_xlb_client")


if __name__ == "__main__":
    main()
