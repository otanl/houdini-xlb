"""Render the constrained-optimization README frames with real XLB results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hou
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_demo_hip import (  # noqa: E402
    DEFAULT_OPTIMIZATION,
    build_scene,
    default_worker_python,
    load_optimization,
)

from houdini_xlb import XlbConfig  # noqa: E402
from houdini_xlb.demo_study import (  # noqa: E402
    COMFORT_MAX,
    COMFORT_MIN,
    MIN_CLEARANCE_M,
    MIN_VENT_RETENTION,
    PLAZA_BOUNDS,
    VENT_BOUNDS,
    Massing,
    minimum_clearance,
)


def _region_mask(
    positions: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return (
        (positions[:, 0] >= x0)
        & (positions[:, 0] <= x1)
        & (positions[:, 1] >= y0)
        & (positions[:, 1] <= y1)
    )


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
    parser.add_argument(
        "--optimization",
        type=Path,
        default=DEFAULT_OPTIMIZATION,
        help="optimization JSON produced by scripts/optimize_demo.py",
    )
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--vmax", type=float, default=0.10)
    args = parser.parse_args()

    optimization = load_optimization(args.optimization)
    milestones = optimization["milestones"]
    baseline = optimization["baseline"]
    evaluation_count = len(optimization["evaluations"])
    final_best_evaluation = int(optimization["result"]["best_evaluation"])
    baseline_comfort = float(baseline["metrics"]["comfort_fraction"])
    baseline_vent = float(baseline["metrics"]["vent_mean"])

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
        optimization_path=args.optimization,
    )
    result = solver.parent().node("xlb_result")
    if result is None:
        raise RuntimeError("xlb_result node was not created")
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

    buildings = [solver.parent().node(f"building{index}") for index in range(4)]
    if any(building is None for building in buildings):
        raise RuntimeError("the four study buildings were not created")
    profile_name = str(optimization["solver"]["profile"])
    profile = XlbConfig.profile(profile_name)
    metadata: list[dict[str, object]] = []

    try:
        for frame_index, milestone in enumerate(milestones):
            timeline_frame = int(milestone["frame"])
            frame_path = output_dir / f"frame_{frame_index:02d}.png"
            hou.setFrame(timeline_frame)
            solver.parm("runxlb").pressButton()
            result.cook(force=True)
            geometry = result.geometry()
            status = str(geometry.attribValue("xlb_status"))
            if status != "current":
                raise RuntimeError(f"XLB result is not current at frame {timeline_frame}")

            renderer.parm("picture").set(str(frame_path))
            renderer.render(frame_range=(timeline_frame, timeline_frame))
            field_geometry = solver.geometry()
            speed = np.asarray(
                field_geometry.pointFloatAttribValues("windspeed"),
                dtype=np.float64,
            )
            positions = np.asarray(
                field_geometry.pointFloatAttribValues("P"),
                dtype=np.float64,
            ).reshape(-1, 3)
            normalized = speed / profile.wind
            plaza_values = normalized[_region_mask(positions, PLAZA_BOUNDS)]
            vent_values = normalized[_region_mask(positions, VENT_BOUNDS)]
            if plaza_values.size == 0 or vent_values.size == 0:
                raise RuntimeError("plaza or ventilation route has no field samples")
            comfort_fraction = float(
                np.mean((plaza_values >= COMFORT_MIN) & (plaza_values <= COMFORT_MAX))
            )
            plaza_ratio = float(plaza_values.mean())
            vent_ratio = float(vent_values.mean())
            vent_retention = vent_ratio / baseline_vent

            current_massings = tuple(
                Massing(
                    float(building.evalParm("tx")),
                    float(building.evalParm("ty")),
                    float(building.evalParm("sizex")),
                    float(building.evalParm("sizey")),
                    float(building.evalParm("sizez")),
                )
                for building in buildings
            )
            clearance = minimum_clearance(current_massings)
            if clearance < MIN_CLEARANCE_M:
                raise RuntimeError(
                    f"building clearance {clearance:.3f} m is below "
                    f"{MIN_CLEARANCE_M:.3f} m at frame {timeline_frame}"
                )
            if vent_retention + 1.0e-6 < MIN_VENT_RETENTION:
                raise RuntimeError(
                    f"vent retention {vent_retention:.4f} violates the hard constraint"
                )

            expected = milestone["metrics"]
            comparisons = {
                "comfort_fraction": comfort_fraction,
                "plaza_mean": plaza_ratio,
                "vent_mean": vent_ratio,
            }
            for name, measured in comparisons.items():
                target = float(expected[name])
                if not np.isclose(measured, target, rtol=2.0e-4, atol=2.0e-4):
                    raise RuntimeError(
                        f"{name} mismatch at frame {timeline_frame}: "
                        f"render={measured:.6f}, optimization={target:.6f}"
                    )

            if frame_index == 0:
                stage = "BASELINE"
            elif frame_index == len(milestones) - 1:
                stage = "GLOBAL BEST"
            else:
                stage = "BEST SO FAR"
            metadata.append(
                {
                    "file": frame_path.name,
                    "profile": profile_name,
                    "pedestrian_height_m": profile.resolved_pedestrian_height_m,
                    "status": status,
                    "timeline_frame": timeline_frame,
                    "evaluation": int(milestone["evaluation"]),
                    "evaluation_count": evaluation_count,
                    "best_evaluation": int(milestone["best_evaluation"]),
                    "final_best_evaluation": final_best_evaluation,
                    "stage": stage,
                    "design": [float(value) for value in milestone["design"]],
                    "clearance_m": clearance,
                    "comfort_fraction": comfort_fraction,
                    "comfort_change_pp": (comfort_fraction - baseline_comfort) * 100.0,
                    "objective": 1.0 - comfort_fraction,
                    "plaza_ratio": plaza_ratio,
                    "vent_ratio": vent_ratio,
                    "vent_retained_pct": vent_retention * 100.0,
                    "comfort_min": COMFORT_MIN,
                    "comfort_max": COMFORT_MAX,
                    "min_vent_retention_pct": MIN_VENT_RETENTION * 100.0,
                    "colour_vmax_ratio": args.vmax / profile.wind,
                    "elapsed_s": float(geometry.attribValue("xlb_elapsed_s")),
                    "cache_hit": int(geometry.attribValue("xlb_cache_hit")),
                    "max_speed": float(speed.max(initial=0.0)),
                }
            )
            print(
                f"rendered {frame_path.name}: eval {milestone['evaluation']}/"
                f"{evaluation_count}; comfort={comfort_fraction:.1%}; "
                f"vent={vent_retention:.1%}; clearance={clearance:.1f} m"
            )
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
