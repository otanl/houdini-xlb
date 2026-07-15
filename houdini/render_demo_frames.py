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
    xlb = build_scene(
        python_executable=python_executable,
        cache_dir=args.cache_dir,
    )
    xlb.parm("profile").set(0)
    xlb.parm("vmax").set(args.vmax)

    camera = hou.node("/obj").createNode("cam", "readme_camera")
    camera.parmTuple("t").set((50.0, 50.0, 140.0))
    camera.parm("projection").set("ortho")
    camera.parm("orthowidth").set(112.0)

    renderer = hou.node("/out").createNode("opengl", "readme_render")
    renderer.parm("camera").set(camera.path())
    renderer.parm("vobjects").set(xlb.parent().path())
    renderer.parm("tres").set(1)
    renderer.parm("res1").set(args.size)
    renderer.parm("res2").set(args.size)
    renderer.parm("usegeocolor").set(1)

    building = xlb.parent().node("building0")
    layouts = (
        {"x": 28.0, "y": 42.0, "height": 12.0},
        {"x": 34.0, "y": 40.0, "height": 16.0},
        {"x": 40.0, "y": 40.0, "height": 20.0},
        {"x": 46.0, "y": 42.0, "height": 14.0},
    )
    metadata: list[dict[str, object]] = []
    frame_index = 0

    def render(stage: str, design_index: int) -> None:
        nonlocal frame_index
        geometry = xlb.geometry()
        status = str(geometry.attribValue("xlb_status"))
        frame_path = output_dir / f"frame_{frame_index:02d}.png"
        renderer.parm("picture").set(str(frame_path))
        renderer.render(frame_range=(1, 1))
        speed = geometry.pointFloatAttribValues("windspeed")
        metadata.append(
            {
                "file": frame_path.name,
                "stage": stage,
                "status": status,
                "design": design_index + 1,
                "design_count": len(layouts),
                **layouts[design_index],
                "elapsed_s": float(geometry.attribValue("xlb_elapsed_s")),
                "cache_hit": int(geometry.attribValue("xlb_cache_hit")),
                "max_speed": max(speed, default=0.0),
            }
        )
        print(f"rendered {frame_path.name}: {status}")
        frame_index += 1

    try:
        for design_index, layout in enumerate(layouts):
            building.parm("tx").set(layout["x"])
            building.parm("ty").set(layout["y"])
            building.parm("sizez").set(layout["height"])

            if design_index > 0:
                xlb.cook(force=True)
                if not str(xlb.geometry().attribValue("xlb_status")).startswith("stale"):
                    raise RuntimeError("geometry edit did not mark the XLB result stale")
                render("stale", design_index)

            xlb.parm("request").set(xlb.evalParm("request") + 1)
            xlb.cook(force=True)
            if xlb.geometry().attribValue("xlb_status") != "current":
                raise RuntimeError("XLB result is not current")
            render("current", design_index)
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
