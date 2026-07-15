"""Serializable XLB analysis settings shared by Houdini and the GPU worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class XlbConfig:
    """One height-map simulation profile.

    Height-map values are normalized domain heights in the closed interval [0, 1].
    The output is the pedestrian-level speed field at the input map resolution.
    """

    grid_x: int = 128
    grid_y: int = 128
    grid_z: int = 48
    steps: int = 600
    wind: float = 0.05
    reynolds: float = 8000.0
    reference_height: float = 0.3
    pedestrian_z: int = 4
    precision: str = "FP32FP32"
    average_window: int = 200
    average_every: int = 50

    def __post_init__(self) -> None:
        if min(self.grid_x, self.grid_y, self.grid_z) < 8:
            raise ValueError("XLB grid dimensions must each be at least 8")
        if self.steps <= 0 or self.average_window < 0 or self.average_every <= 0:
            raise ValueError("step and averaging settings must be positive")
        if self.average_window > self.steps:
            raise ValueError("average_window cannot exceed steps")
        if self.wind <= 0 or self.reynolds <= 0 or self.reference_height <= 0:
            raise ValueError("wind, Reynolds number and reference height must be positive")
        if not 0 <= self.pedestrian_z < self.grid_z:
            raise ValueError("pedestrian_z must lie inside the vertical lattice")

    @property
    def grid_xyz(self) -> tuple[int, int, int]:
        return self.grid_x, self.grid_y, self.grid_z

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> XlbConfig:
        return cls(**values)

    @classmethod
    def profile(cls, name: str) -> XlbConfig:
        try:
            return cls(**_PROFILES[name])
        except KeyError as exc:
            raise ValueError(f"unknown XLB profile {name!r}; choose {sorted(_PROFILES)}") from exc


_PROFILES: dict[str, dict[str, object]] = {
    "draft": {
        "grid_x": 96,
        "grid_y": 96,
        "grid_z": 40,
        "steps": 300,
        "average_window": 100,
        "average_every": 25,
    },
    "preview": {
        "grid_x": 128,
        "grid_y": 128,
        "grid_z": 48,
        "steps": 600,
        "average_window": 200,
        "average_every": 50,
    },
    "quality": {
        "grid_x": 256,
        "grid_y": 256,
        "grid_z": 64,
        "steps": 2500,
        "average_window": 800,
        "average_every": 100,
    },
}


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
