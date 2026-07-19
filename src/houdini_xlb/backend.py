"""Native-Windows XLB height-map backend owned by the Houdini bridge package."""

from __future__ import annotations

import numpy as np

_CONTEXTS: dict[tuple[tuple[int, int, int], float, str], dict[str, object]] = {}


def _context(grid_xyz: tuple[int, int, int], wind: float, precision: str):
    key = (tuple(grid_xyz), float(wind), precision)
    if key in _CONTEXTS:
        return _CONTEXTS[key]

    import xlb
    from xlb.compute_backend import ComputeBackend
    from xlb.grid import grid_factory
    from xlb.operator.macroscopic import Macroscopic
    from xlb.precision_policy import PrecisionPolicy

    policy = getattr(PrecisionPolicy, precision)
    backend = ComputeBackend.WARP
    velocity_set = xlb.velocity_set.D3Q27(
        precision_policy=policy,
        compute_backend=backend,
    )
    xlb.init(
        velocity_set=velocity_set,
        default_backend=backend,
        default_precision_policy=policy,
    )
    grid = grid_factory(grid_xyz, compute_backend=backend)
    box = grid.bounding_box_indices()
    box_without_edges = grid.bounding_box_indices(remove_edges=True)
    walls = [
        box["bottom"][index] + box["top"][index] + box["front"][index] + box["back"][index]
        for index in range(velocity_set.d)
    ]
    walls = np.unique(np.asarray(walls), axis=-1).tolist()
    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=policy,
        velocity_set=xlb.velocity_set.D3Q27(
            precision_policy=policy,
            compute_backend=ComputeBackend.JAX,
        ),
    )
    context = {
        "grid": grid,
        "walls": walls,
        "inlet": box_without_edges["left"],
        "outlet": box_without_edges["right"],
        "macro": macro,
    }
    _CONTEXTS[key] = context
    return context


def _solid_indices(heightmap: np.ndarray, grid_xyz: tuple[int, int, int]):
    grid_x, grid_y, grid_z = grid_xyz
    expected_shape = (grid_y, grid_x)
    if heightmap.shape != expected_shape:
        raise ValueError(
            f"heightmap shape {heightmap.shape} must equal lattice y/x {expected_shape}; "
            "rasterize source geometry directly at the XLB resolution"
        )
    lattice_map = np.asarray(heightmap, dtype=np.float64).clip(0.0, 1.0)
    height_cells = np.rint(lattice_map.T * grid_z).astype(int)
    height_cells[height_cells < 1] = 0
    x, y = np.where(height_cells > 0)
    heights = np.minimum(height_cells[x, y], grid_z - 1)
    solid_x = np.repeat(x, heights)
    solid_y = np.repeat(y, heights)
    columns = [np.arange(1, height + 1) for height in heights]
    solid_z = np.concatenate(columns) if columns else np.asarray([], dtype=int)
    return solid_x, solid_y, solid_z


def _run(
    context,
    solid,
    *,
    wind: float,
    omega: float,
    steps: int,
    output_shape: tuple[int, int],
    pedestrian_z: float,
    average_window: int,
    average_every: int,
) -> np.ndarray:
    import warp as wp
    from xlb.operator.boundary_condition import (
        ExtrapolationOutflowBC,
        FullwayBounceBackBC,
        RegularizedBC,
    )
    from xlb.operator.boundary_condition.boundary_condition_registry import (
        boundary_condition_registry,
    )
    from xlb.operator.stepper import IncompressibleNavierStokesStepper
    from xlb.utils import warp_array_to_jax

    boundary_condition_registry.next_id = 1
    boundary_condition_registry.id_to_bc.clear()
    boundary_condition_registry.bc_to_id.clear()

    boundary_conditions = [
        FullwayBounceBackBC(indices=context["walls"]),
        RegularizedBC(
            "velocity",
            prescribed_value=(wind, 0.0, 0.0),
            indices=context["inlet"],
        ),
        ExtrapolationOutflowBC(indices=context["outlet"]),
        FullwayBounceBackBC(indices=[solid[0].tolist(), solid[1].tolist(), solid[2].tolist()]),
    ]
    stepper = IncompressibleNavierStokesStepper(
        grid=context["grid"],
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )
    f0, f1, boundary_mask, missing_mask = stepper.prepare_fields()
    accumulator = None
    samples = 0
    average_start = max(0, steps - average_window)
    for step in range(steps):
        f0, f1 = stepper(f0, f1, boundary_mask, missing_mask, omega, step)
        f0, f1 = f1, f0
        if (
            average_window > 0
            and step >= average_start
            and (step - average_start) % average_every == 0
        ):
            populations = f0.numpy()
            accumulator = (
                populations.astype(np.float64) if accumulator is None else accumulator + populations
            )
            samples += 1
    wp.synchronize()

    if samples:
        import jax.numpy as jnp

        _, velocity = context["macro"](jnp.asarray((accumulator / samples).astype(np.float32)))
    else:
        _, velocity = context["macro"](warp_array_to_jax(f0))
    velocity = np.asarray(velocity)
    speed_3d = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
    z0 = int(np.floor(pedestrian_z))
    z1 = min(z0 + 1, speed_3d.shape[2] - 1)
    fraction = pedestrian_z - z0
    speed = ((1.0 - fraction) * speed_3d[:, :, z0] + fraction * speed_3d[:, :, z1]).T
    if speed.shape != output_shape:
        raise RuntimeError(f"XLB speed shape {speed.shape} does not match {output_shape}")
    return speed.astype(np.float32)


def simulate_heightmap_xlb(
    heightmap: np.ndarray,
    *,
    grid_xyz: tuple[int, int, int],
    wind: float,
    reynolds: float,
    steps: int,
    pedestrian_z: float,
    precision: str,
    average_window: int,
    average_every: int,
    reference_height_lattice: float | None = None,
    reference_height: float | None = None,
    max_speed_ratio: float = 8.0,
) -> np.ndarray:
    """Run one normalized urban height map and return pedestrian-level speed.

    reference_height is a compatibility input expressed as a fraction of the
    vertical lattice. New callers should pass reference_height_lattice after
    resolving their physical coordinate contract.
    """

    heightmap = np.asarray(heightmap, dtype=np.float32)
    if heightmap.ndim != 2 or min(heightmap.shape) < 2:
        raise ValueError("heightmap must be a two-dimensional array")
    if not np.isfinite(heightmap).all() or heightmap.min() < 0 or heightmap.max() > 1:
        raise ValueError("heightmap values must be finite and lie in [0, 1]")
    if not 1 <= pedestrian_z < grid_xyz[2] - 1:
        raise ValueError("pedestrian_z must select an interior lattice slice")
    if reference_height_lattice is None:
        if reference_height is None:
            raise ValueError("reference_height_lattice is required")
        reference_height_lattice = reference_height * grid_xyz[2]
    if reference_height_lattice <= 0 or max_speed_ratio <= 1:
        raise ValueError("reference height and max_speed_ratio must be positive")

    context = _context(grid_xyz, wind, precision)
    solid = _solid_indices(heightmap, grid_xyz)
    omega = 1.0 / (3.0 * wind * reference_height_lattice / reynolds + 0.5)
    speed = _run(
        context,
        solid,
        wind=wind,
        omega=omega,
        steps=steps,
        output_shape=heightmap.shape,
        pedestrian_z=pedestrian_z,
        average_window=average_window,
        average_every=average_every,
    )
    if not np.isfinite(speed).all() or np.any(speed < 0):
        raise RuntimeError("XLB produced a non-finite or negative speed field")
    peak = float(np.max(speed))
    limit = wind * max_speed_ratio
    if peak > limit:
        raise RuntimeError(
            f"XLB result is numerically unstable: peak {peak:.6g} exceeds {limit:.6g}"
        )
    return speed
