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
    from scipy.ndimage import zoom

    grid_x, grid_y, grid_z = grid_xyz
    lattice_map = zoom(
        np.asarray(heightmap, dtype=np.float64),
        (grid_y / heightmap.shape[0], grid_x / heightmap.shape[1]),
        order=1,
    ).clip(0.0, 1.0)
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
    pedestrian_z: int,
    average_window: int,
    average_every: int,
) -> np.ndarray:
    import warp as wp
    from scipy.ndimage import zoom
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
    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)[:, :, pedestrian_z].T
    output_y, output_x = output_shape
    return zoom(
        speed,
        (output_y / speed.shape[0], output_x / speed.shape[1]),
        order=1,
    ).astype(np.float32)


def simulate_heightmap_xlb(
    heightmap: np.ndarray,
    *,
    grid_xyz: tuple[int, int, int],
    wind: float,
    reynolds: float,
    steps: int,
    reference_height: float,
    pedestrian_z: int,
    precision: str,
    average_window: int,
    average_every: int,
) -> np.ndarray:
    """Run one normalized urban height map and return pedestrian-level speed."""
    heightmap = np.asarray(heightmap, dtype=np.float32)
    context = _context(grid_xyz, wind, precision)
    solid = _solid_indices(heightmap, grid_xyz)
    omega = 1.0 / (3.0 * wind * (reference_height * grid_xyz[2]) / reynolds + 0.5)
    return _run(
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
