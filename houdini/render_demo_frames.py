"""Render README demo frames from Houdini geometry and real XLB results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_demo_hip import build_scene, default_worker_python  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "readme-demo" / "frames",
    )
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cache" / "xlb",
    )
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--vmax", type=float, default=0.08)
    args = parser.parse_args()

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.glob("frame_*.png"):
        old_frame.unlink()

    python_executable = (args.python_executable or default_worker_python()).resolve()
    if not python_executable.exists():
        raise FileNotFoundError(f"external Python not found: {python_executable}")

    hou.hipFile.clear(suppress_save_prompt=True)
    solver = build_scene(
        python_executable=python_executable,
        cache_dir=args.cache_dir,
    )
    result = solver.parent().node("xlb_result")
    if result is None:
        raise RuntimeError("xlb_result node was not created")
    solver.parm("profile").set(0)
    solver.parm("vmax").set(args.vmax)

    camera = hou.node("/obj").createNode("cam", "readme_camera")
    camera.parmTuple("t").set((50.0, 50.0, 140.0))
    camera.parm("projection").set("ortho")
    camera.parm("orthowidth").set(112.0)

    renderer = hou.node("/out").createNode("opengl", "readme_render")
    renderer.parm("camera").set(camera.path())
    renderer.parm("vobjects").set(solver.parent().path())
    renderer.parm("tres").set(1)
    renderer.parm("res1").set(args.size)
    renderer.parm("res2").set(args.size)
    renderer.parm("usegeocolor").set(1)

    building = solver.parent().node("building0")
    timeline_frames = (1, 12, 24, 36)
    metadata: list[dict[str, object]] = []
    frame_index = 0

    def render(timeline_frame: int, design_index: int) -> None:
        nonlocal frame_index
        geometry = result.geometry()
        status = str(geometry.attribValue("xlb_status"))
        frame_path = output_dir / f"frame_{frame_index:02d}.png"
        renderer.parm("picture").set(str(frame_path))
        renderer.render(frame_range=(timeline_frame, timeline_frame))
        speed = geometry.pointFloatAttribValues("windspeed")
        metadata.append(
            {
                "file": frame_path.name,
                "status": status,
                "timeline_frame": timeline_frame,
                "design": design_index + 1,
                "design_count": len(timeline_frames),
                "x": float(building.evalParm("tx")),
                "height": float(building.evalParm("sizez")),
                "elapsed_s": float(geometry.attribValue("xlb_elapsed_s")),
                "cache_hit": int(geometry.attribValue("xlb_cache_hit")),
                "max_speed": max(speed, default=0.0),
            }
        )
        print(f"rendered {frame_path.name}: {status}")
        frame_index += 1

    try:
        for design_index, timeline_frame in enumerate(timeline_frames):
            hou.setFrame(timeline_frame)
            solver.parm("runxlb").pressButton()
            result.cook(force=True)
            if result.geometry().attribValue("xlb_status") != "current":
                raise RuntimeError("XLB result is not current")
            render(timeline_frame, design_index)
    finally:
        client = getattr(hou.session, "_houdini_xlb_client", None)
        if client is not None:
            client.close()
            delattr(hou.session, "_houdini_xlb_client")

    metadata_path = output_dir / "frames.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
