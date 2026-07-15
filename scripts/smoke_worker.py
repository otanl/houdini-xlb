"""Run one small real-XLB request through the same persistent worker used by Houdini."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from houdini_xlb import XlbConfig, XlbWorkerClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/houdini/cache/xlb-smoke"))
    args = parser.parse_args()

    heightmap = np.zeros((48, 48), dtype=np.float32)
    heightmap[17:31, 20:28] = 0.35
    config = XlbConfig(
        grid_x=64,
        grid_y=64,
        grid_z=32,
        steps=120,
        average_window=40,
        average_every=10,
        pedestrian_z=3,
    )
    with XlbWorkerClient(cache_dir=args.cache) as client:
        health = client.health()
        first = client.analyze(heightmap, config)
        second = client.analyze(heightmap, config)

    if not np.isfinite(first.speed).all():
        raise RuntimeError("worker returned non-finite speed")
    if first.cache_hit or not second.cache_hit:
        raise RuntimeError("expected first request to run and second request to hit cache")
    print(
        f"ready={health['ok']} shape={first.speed.shape} "
        f"elapsed={first.elapsed_s:.3f}s second_cache_hit={second.cache_hit} "
        f"key={first.cache_key[:12]}"
    )


if __name__ == "__main__":
    main()
