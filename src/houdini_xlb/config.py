"""Serializable XLB analysis settings shared by Houdini and the GPU worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class XlbConfig:
    """One physically-scaled height-map simulation profile.

    Height maps store fractions of the physical domain height. Lattice cells must
    remain nearly cubic so geometry and Reynolds scaling share one length scale.
    """

    grid_x: int = 128
    grid_y: int = 128
    grid_z: int = 51
    steps: int = 600
    wind: float = 0.05
    reynolds: float = 8000.0
    domain_length_x_m: float = 100.0
    domain_length_y_m: float = 100.0
    domain_height_m: float = 40.0
    reference_height_m: float = 10.0
    pedestrian_height_m: float = 1.5
    precision: str = "FP32FP32"
    average_window: int = 200
    average_every: int = 50
    max_speed_ratio: float = 8.0
    isotropy_tolerance: float = 0.08
    inlet_profile: str = "uniform"
    inlet_power_alpha: float = 0.16
    initial_condition: str = "rest"

    def __post_init__(self) -> None:
        if min(self.grid_x, self.grid_y, self.grid_z) < 8:
            raise ValueError("XLB grid dimensions must each be at least 8")
        if self.steps <= 0 or self.average_window < 0 or self.average_every <= 0:
            raise ValueError("step count must be positive and averaging settings non-negative")
        if self.average_window > self.steps:
            raise ValueError("average_window cannot exceed steps")
        if self.wind <= 0 or self.reynolds <= 0:
            raise ValueError("wind and Reynolds number must be positive")
        if min(self.domain_xyz_m) <= 0:
            raise ValueError("physical domain dimensions must be positive")
        if not 0 < self.reference_height_m < self.domain_height_m:
            raise ValueError("reference_height_m must lie inside the physical domain")
        if not 0 < self.pedestrian_height_m < self.domain_height_m:
            raise ValueError("pedestrian_height_m must lie inside the physical domain")
        if self.max_speed_ratio <= 1:
            raise ValueError("max_speed_ratio must be greater than one")
        if not 0 <= self.isotropy_tolerance < 1:
            raise ValueError("isotropy_tolerance must lie in [0, 1)")
        if self.inlet_profile not in {"uniform", "power_law"}:
            raise ValueError("inlet_profile must be 'uniform' or 'power_law'")
        if not 0 <= self.inlet_power_alpha <= 1:
            raise ValueError("inlet_power_alpha must lie in [0, 1]")
        if self.initial_condition not in {"rest", "uniform_reference"}:
            raise ValueError("initial_condition must be 'rest' or 'uniform_reference'")

        cell_sizes = self.cell_sizes_m
        anisotropy = max(cell_sizes) / min(cell_sizes) - 1.0
        if anisotropy > self.isotropy_tolerance:
            raise ValueError(
                "physical lattice cells must be nearly cubic; "
                f"cell sizes are {cell_sizes!r} m (anisotropy {anisotropy:.1%})"
            )

    @property
    def grid_xyz(self) -> tuple[int, int, int]:
        return self.grid_x, self.grid_y, self.grid_z

    @property
    def domain_xyz_m(self) -> tuple[float, float, float]:
        return self.domain_length_x_m, self.domain_length_y_m, self.domain_height_m

    @property
    def cell_sizes_m(self) -> tuple[float, float, float]:
        return (
            self.domain_length_x_m / self.grid_x,
            self.domain_length_y_m / self.grid_y,
            self.domain_height_m / self.grid_z,
        )

    @property
    def pedestrian_z(self) -> float:
        """Fractional lattice coordinate used to interpolate the result height."""

        z = self.pedestrian_height_m / self.cell_sizes_m[2]
        return min(max(1.0, z), self.grid_z - 2.0)

    @property
    def resolved_pedestrian_height_m(self) -> float:
        return self.pedestrian_z * self.cell_sizes_m[2]

    @property
    def reference_height_lattice(self) -> float:
        return self.reference_height_m / self.cell_sizes_m[2]

    def with_domain(
        self,
        *,
        length_x_m: float,
        length_y_m: float,
        height_m: float,
        reference_height_m: float | None = None,
        pedestrian_height_m: float | None = None,
    ) -> XlbConfig:
        """Return a config with y/z resolution derived from x and metre extents."""

        if min(length_x_m, length_y_m, height_m) <= 0:
            raise ValueError("physical domain dimensions must be positive")
        dx = length_x_m / self.grid_x
        return replace(
            self,
            grid_y=max(8, round(length_y_m / dx)),
            grid_z=max(8, round(height_m / dx)),
            domain_length_x_m=length_x_m,
            domain_length_y_m=length_y_m,
            domain_height_m=height_m,
            reference_height_m=(
                self.reference_height_m if reference_height_m is None else reference_height_m
            ),
            pedestrian_height_m=(
                self.pedestrian_height_m if pedestrian_height_m is None else pedestrian_height_m
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> XlbConfig:
        """Load current settings and migrate the pre-physical-coordinate format."""

        migrated = dict(values)
        grid_x = int(migrated.get("grid_x", 128))
        grid_y = int(migrated.get("grid_y", 128))
        grid_z = int(migrated.get("grid_z", 51))
        length_x_m = float(migrated.get("domain_length_x_m", 100.0))
        domain_height_m = float(migrated.get("domain_height_m", length_x_m * grid_z / grid_x))
        migrated.setdefault("domain_height_m", domain_height_m)
        migrated.setdefault("domain_length_y_m", length_x_m * grid_y / grid_x)
        if "reference_height_m" not in migrated and "reference_height" in migrated:
            migrated["reference_height_m"] = (
                float(migrated.pop("reference_height")) * domain_height_m
            )
        if "pedestrian_height_m" not in migrated and "pedestrian_z" in migrated:
            migrated["pedestrian_height_m"] = (
                float(migrated.pop("pedestrian_z")) * domain_height_m / grid_z
            )
        migrated.pop("reference_height", None)
        migrated.pop("pedestrian_z", None)
        return cls(**migrated)

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
        "grid_z": 38,
        "steps": 300,
        "average_window": 100,
        "average_every": 25,
    },
    "preview": {
        "grid_x": 128,
        "grid_y": 128,
        "grid_z": 51,
        "steps": 600,
        "average_window": 200,
        "average_every": 50,
    },
    "study": {
        "grid_x": 96,
        "grid_y": 96,
        "grid_z": 38,
        "steps": 2400,
        "average_window": 800,
        "average_every": 40,
    },
    "quality": {
        "grid_x": 256,
        "grid_y": 256,
        "grid_z": 102,
        "steps": 2500,
        "average_window": 800,
        "average_every": 100,
    },
}


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
