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
        "grid_xyz": tuple(grid_xyz),
        "precision": precision,
        "velocity_set": velocity_set,
        "precision_policy": policy,
        "compute_backend": backend,
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


def _power_law_profile(
    *,
    wind: float,
    reference_height_lattice: float,
    exponent: float,
    precision: str,
):
    """Build a Warp inlet function normalized to wind at the reference height."""

    import warp as wp
    from xlb.precision_policy import PrecisionPolicy

    wp_dtype = getattr(PrecisionPolicy, precision).compute_precision.wp_dtype
    wind_value = wp_dtype(wind)
    reference_value = wp_dtype(reference_height_lattice)
    exponent_value = wp_dtype(exponent)
    minimum_z = wp_dtype(1.0)

    @wp.func
    def profile(index: wp.vec3i):
        z = wp.max(wp_dtype(index[2]), minimum_z)
        velocity = wind_value * wp.pow(z / reference_value, exponent_value)
        return wp.vec(velocity, length=1)

    return profile


def _mask_nonfluid_velocity(field: np.ndarray, solid) -> np.ndarray:
    """Zero macroscopic reconstruction artifacts inside bounce-back cells."""

    masked = np.asarray(field, dtype=np.float32).copy()
    solid_x, solid_y, solid_z = (np.asarray(axis, dtype=int) for axis in solid)
    if solid_x.size:
        masked[:, solid_z, solid_y, solid_x] = 0.0
    masked[:, (0, -1), :, :] = 0.0
    masked[:, :, (0, -1), :] = 0.0
    return masked


def _run(
    context,
    solid,
    *,
    wind: float,
    omega: float,
    steps: int,
    average_window: int,
    average_every: int,
    reference_height_lattice: float,
    inlet_profile: str,
    inlet_power_alpha: float,
    initial_condition: str,
    collision_model: str,
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

    inlet_boundary = (
        RegularizedBC(
            "velocity",
            prescribed_value=(wind, 0.0, 0.0),
            indices=context["inlet"],
        )
        if inlet_profile == "uniform"
        else RegularizedBC(
            "velocity",
            profile=_power_law_profile(
                wind=wind,
                reference_height_lattice=reference_height_lattice,
                exponent=inlet_power_alpha,
                precision=context["precision"],
            ),
            indices=context["inlet"],
        )
    )
    boundary_conditions = [
        FullwayBounceBackBC(indices=context["walls"]),
        inlet_boundary,
        ExtrapolationOutflowBC(indices=context["outlet"]),
        FullwayBounceBackBC(indices=[solid[0].tolist(), solid[1].tolist(), solid[2].tolist()]),
    ]
    stepper = IncompressibleNavierStokesStepper(
        grid=context["grid"],
        boundary_conditions=boundary_conditions,
        collision_type=collision_model,
    )
    initializer = None
    if initial_condition == "uniform_reference":
        from xlb.helper.initializers import CustomInitializer

        initializer = CustomInitializer(
            constant_velocity_vector=[wind, 0.0, 0.0],
            velocity_set=context["velocity_set"],
            precision_policy=context["precision_policy"],
            compute_backend=context["compute_backend"],
        )
    f0, f1, boundary_mask, missing_mask = stepper.prepare_fields(initializer=initializer)
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
    field = np.transpose(velocity, (0, 3, 2, 1))
    grid_x, grid_y, grid_z = context["grid_xyz"]
    expected_shape = (3, grid_z, grid_y, grid_x)
    if field.shape != expected_shape:
        raise RuntimeError(f"XLB velocity shape {field.shape} does not match {expected_shape}")
    return _mask_nonfluid_velocity(field, solid)


def simulate_velocity_field_xlb(
    heightmap: np.ndarray,
    *,
    grid_xyz: tuple[int, int, int],
    wind: float,
    reynolds: float,
    steps: int,
    precision: str,
    average_window: int,
    average_every: int,
    reference_height_lattice: float | None = None,
    reference_height: float | None = None,
    max_speed_ratio: float = 8.0,
    inlet_profile: str = "uniform",
    inlet_power_alpha: float = 0.16,
    initial_condition: str = "rest",
    collision_model: str = "KBC",
) -> np.ndarray:
    """Run one height map and return mean velocity as (component, z, y, x)."""

    heightmap = np.asarray(heightmap, dtype=np.float32)
    if heightmap.ndim != 2 or min(heightmap.shape) < 2:
        raise ValueError("heightmap must be a two-dimensional array")
    expected_map_shape = (grid_xyz[1], grid_xyz[0])
    if heightmap.shape != expected_map_shape:
        raise ValueError(
            f"heightmap shape {heightmap.shape} must equal lattice y/x {expected_map_shape}"
        )
    if not np.isfinite(heightmap).all() or heightmap.min() < 0 or heightmap.max() > 1:
        raise ValueError("heightmap values must be finite and lie in [0, 1]")
    if reference_height_lattice is None:
        if reference_height is None:
            raise ValueError("reference_height_lattice is required")
        reference_height_lattice = reference_height * grid_xyz[2]
    if reference_height_lattice <= 0 or max_speed_ratio <= 1:
        raise ValueError("reference height and max_speed_ratio must be positive")
    if inlet_profile not in {"uniform", "power_law"}:
        raise ValueError("inlet_profile must be 'uniform' or 'power_law'")
    if not 0 <= inlet_power_alpha <= 1:
        raise ValueError("inlet_power_alpha must lie in [0, 1]")
    if initial_condition not in {"rest", "uniform_reference"}:
        raise ValueError("initial_condition must be 'rest' or 'uniform_reference'")
    if collision_model not in {"KBC", "SmagorinskyLESBGK"}:
        raise ValueError("collision_model must be 'KBC' or 'SmagorinskyLESBGK'")

    context = _context(grid_xyz, wind, precision)
    solid = _solid_indices(heightmap, grid_xyz)
    omega = 1.0 / (3.0 * wind * reference_height_lattice / reynolds + 0.5)
    field = _run(
        context,
        solid,
        wind=wind,
        omega=omega,
        steps=steps,
        average_window=average_window,
        average_every=average_every,
        reference_height_lattice=reference_height_lattice,
        inlet_profile=inlet_profile,
        inlet_power_alpha=inlet_power_alpha,
        initial_condition=initial_condition,
        collision_model=collision_model,
    )
    if not np.isfinite(field).all():
        raise RuntimeError("XLB produced a non-finite velocity field")
    peak = float(np.sqrt(np.sum(field.astype(np.float64) ** 2, axis=0)).max())
    limit = wind * max_speed_ratio
    if peak > limit:
        raise RuntimeError(
            f"XLB result is numerically unstable: peak {peak:.6g} exceeds {limit:.6g}"
        )
    return field


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
    inlet_profile: str = "uniform",
    inlet_power_alpha: float = 0.16,
    initial_condition: str = "rest",
    collision_model: str = "KBC",
) -> np.ndarray:
    """Run one normalized urban height map and return pedestrian-level speed.

    reference_height is a compatibility input expressed as a fraction of the
    vertical lattice. New callers should pass reference_height_lattice after
    resolving their physical coordinate contract.
    """

    if not 1 <= pedestrian_z < grid_xyz[2] - 1:
        raise ValueError("pedestrian_z must select an interior lattice slice")
    field = simulate_velocity_field_xlb(
        heightmap,
        grid_xyz=grid_xyz,
        wind=wind,
        reynolds=reynolds,
        steps=steps,
        precision=precision,
        average_window=average_window,
        average_every=average_every,
        reference_height_lattice=reference_height_lattice,
        reference_height=reference_height,
        max_speed_ratio=max_speed_ratio,
        inlet_profile=inlet_profile,
        inlet_power_alpha=inlet_power_alpha,
        initial_condition=initial_condition,
        collision_model=collision_model,
    )
    speed_3d = np.sqrt(np.sum(field**2, axis=0))
    z0 = int(np.floor(pedestrian_z))
    z1 = min(z0 + 1, speed_3d.shape[0] - 1)
    fraction = pedestrian_z - z0
    speed = (1.0 - fraction) * speed_3d[z0] + fraction * speed_3d[z1]
    return speed.astype(np.float32)
