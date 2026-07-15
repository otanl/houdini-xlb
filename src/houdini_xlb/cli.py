"""Command-line height-map analysis for the same path the Houdini client uses."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import XlbConfig, profile_names
from .core import analyze_heightmap


def _load_heightmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as data:
        if "heightmap" in data:
            return data["heightmap"]
        return data[data.files[0]]


def _configured_profile(args: argparse.Namespace) -> XlbConfig:
    config = XlbConfig.profile(args.profile)
    overrides = {
        name: value
        for name, value in {
            "grid_x": args.grid_x,
            "grid_y": args.grid_y,
            "grid_z": args.grid_z,
            "steps": args.steps,
            "wind": args.wind,
            "reynolds": args.reynolds,
            "reference_height": args.reference_height,
            "pedestrian_z": args.pedestrian_z,
            "average_window": args.average_window,
            "average_every": args.average_every,
        }.items()
        if value is not None
    }
    return replace(config, **overrides)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cached XLB analysis for a normalized height map"
    )
    parser.add_argument("heightmap", type=Path)
    parser.add_argument("--profile", choices=profile_names(), default="preview")
    parser.add_argument("--grid-x", type=int)
    parser.add_argument("--grid-y", type=int)
    parser.add_argument("--grid-z", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--wind", type=float)
    parser.add_argument("--reynolds", type=float)
    parser.add_argument("--reference-height", type=float)
    parser.add_argument("--pedestrian-z", type=int)
    parser.add_argument("--average-window", type=int)
    parser.add_argument("--average-every", type=int)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/houdini/cache/xlb"))
    parser.add_argument("--out", type=Path, default=Path("outputs/houdini_xlb_result.npz"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    config = _configured_profile(args)
    result = analyze_heightmap(_load_heightmap(args.heightmap), config, cache_dir=args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        speed=result.speed,
        metadata=np.asarray(json.dumps(result.metadata(), sort_keys=True)),
    )
    print(
        f"shape={result.speed.shape} cache_hit={result.cache_hit} "
        f"elapsed={result.elapsed_s:.3f}s key={result.cache_key[:12]} saved={args.out}"
    )
    return 0


if __name__ == "__main__":
    main()
