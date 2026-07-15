"""Build a minimal Houdini scene with an explicit cached XLB confirmation button."""

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


def build_scene(
    name: str = "houdini_xlb",
    *,
    python_executable: Path | None = None,
    cache_dir: Path | None = None,
) -> hou.SopNode:
    """Create boxes → connectivity → explicit XLB Python SOP."""
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

    merge = container.createNode("merge", "buildings")
    for index, box in enumerate(boxes):
        merge.setInput(index, box)
    connectivity = container.createNode("connectivity", "connected_buildings")
    connectivity.setFirstInput(merge)
    colour = container.createNode("color", "building_colour")
    colour.setFirstInput(connectivity)
    colour.parmTuple("color").set((0.15, 0.55, 0.95))

    xlb = container.createNode("python", "xlb_confirmation")
    xlb.setInput(0, ground)
    xlb.setInput(1, colour)
    install_parameters(xlb)
    xlb.parm("python").set(
        sop_code(
            package_src=PACKAGE_SRC,
            cache_dir=cache_dir,
            python_executable=python_executable,
        )
    )
    xlb.setDisplayFlag(True)
    xlb.setRenderFlag(True)

    note = container.createStickyNote()
    note.setText(
        "HOUDINI × XLB\n"
        "1. Edit building Box SOPs.\n"
        "2. Select xlb_confirmation and press Run XLB.\n"
        "3. Geometry edits make the previous field grey/stale until the next run.\n"
        "draft/preview are interactive confirmations, not frame-rate CFD."
    )
    note.setSize(hou.Vector2(5.0, 2.0))
    container.layoutChildren()
    return xlb


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
    xlb = build_scene(
        python_executable=python_executable,
        cache_dir=args.cache_dir,
    )
    try:
        try:
            xlb.cook(force=True)
        except hou.OperationFailed:
            print("\n".join(xlb.errors()))
            raise
        status = xlb.geometry().attribValue("xlb_status")
        if status != "not-run: press Run XLB":
            raise RuntimeError(f"unexpected initial XLB SOP status: {status}")
        if args.run_xlb_smoke:
            xlb.parm("profile").set(0)
            xlb.parm("request").set(1)
            xlb.cook(force=True)
            status = xlb.geometry().attribValue("xlb_status")
            if status != "current":
                raise RuntimeError(f"XLB smoke result is not current: {status}")
            print(
                "XLB smoke current; "
                f"elapsed={xlb.geometry().attribValue('xlb_elapsed_s'):.3f}s "
                f"cache_hit={xlb.geometry().attribValue('xlb_cache_hit')}"
            )
            xlb.parm("request").set(0)
        hou.hipFile.save(str(output))
        print(f"saved {output}; verified status={status}; saved request=0")
    finally:
        client = getattr(hou.session, "_houdini_xlb_client", None)
        if client is not None:
            client.close()
            delattr(hou.session, "_houdini_xlb_client")


if __name__ == "__main__":
    main()
