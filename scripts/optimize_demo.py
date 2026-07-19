"""Run the public 16-layout design optimization directly through XLB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from houdini_xlb import BACKEND_SIGNATURE, XlbConfig, analyze_heightmap, profile_names
from houdini_xlb.demo_study import (
    Design,
    heightmap_from_design,
    metrics_from_speed,
    optimize_study,
    validate_optimization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "examples" / "houdini_xlb_demo_optimization.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cache" / "xlb",
    )
    parser.add_argument("--profile", choices=profile_names(), default="study")
    args = parser.parse_args()

    config = XlbConfig.profile(args.profile)
    cache_dir = args.cache_dir.resolve()

    def evaluate(design: Design):
        result = analyze_heightmap(
            heightmap_from_design(
                design,
                ny=config.grid_y,
                nx=config.grid_x,
            ),
            config,
            cache_dir=cache_dir,
        )
        metrics = metrics_from_speed(result.speed, inlet_speed=config.wind)
        print(
            f"design={design} comfort={metrics.comfort_fraction:.3f} "
            f"plaza={metrics.plaza_mean:.3f} vent={metrics.vent_mean:.3f} "
            f"cache_hit={result.cache_hit}"
        )
        return metrics

    optimization = optimize_study(evaluate)
    optimization["solver"] = {
        "engine": "XLB",
        "backend_signature": BACKEND_SIGNATURE,
        "profile": args.profile,
        "config": config.to_dict(),
    }
    validate_optimization(optimization)

    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(optimization, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    baseline = optimization["baseline"]["metrics"]
    result = optimization["result"]
    final = result["metrics"]
    print(
        f"wrote {output}; best evaluation={result['best_evaluation']}/16; "
        f"comfort={baseline['comfort_fraction']:.3f}->{final['comfort_fraction']:.3f}; "
        f"vent_retained={result['vent_retention']:.3f}"
    )


if __name__ == "__main__":
    main()
