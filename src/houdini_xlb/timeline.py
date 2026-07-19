"""Solver-SOP state and asynchronous XLB scheduling for Houdini.

The Houdini timeline represents design alternatives, not physical CFD time.
The Solver SOP carries the previous displayed field and request metadata through
Prev_Frame.  XLB itself stays in a persistent external Python worker so a Solver
cook never blocks Houdini's UI.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import XlbConfig, profile_names
from .core import AnalysisResult, analysis_key, load_cached_heightmap
from .houdini import geometry_heightmap, session_client


@dataclass(frozen=True)
class TimelineJob:
    """One immutable geometry/configuration request."""

    node_path: str
    signature: str
    heightmap: np.ndarray = field(repr=False, compare=False)
    config: XlbConfig
    cache_dir: Path
    python_executable: Path
    frame: int
    kind: str = "auto"
    refresh_path: str | None = None


@dataclass
class _Pending:
    job: TimelineJob
    deadline: float


@dataclass
class _Active:
    job: TimelineJob
    future: Any


class TimelineScheduler:
    """Single-worker latest-only scheduler with an optional range-bake queue."""

    def __init__(self) -> None:
        self._desired: dict[str, _Pending] = {}
        self._bake: dict[str, deque[TimelineJob]] = {}
        self._active: _Active | None = None
        self._errors: dict[tuple[str, str], str] = {}
        self._bake_done: dict[str, int] = {}
        self._bake_total: dict[str, int] = {}
        self._cancelled: set[str] = set()
        self._collecting: set[str] = set()
        self._event_callback: Callable[[], None] | None = None
        self._event_registered = False

    @property
    def idle(self) -> bool:
        return self._active is None and not self._desired and not any(self._bake.values())

    def begin_collection(self, node_path: str) -> None:
        self._collecting.add(node_path)

    def end_collection(self, node_path: str) -> None:
        self._collecting.discard(node_path)

    def is_collecting(self, node_path: str) -> bool:
        return node_path in self._collecting

    def request(
        self,
        job: TimelineJob,
        *,
        debounce_s: float = 0.0,
        now: float | None = None,
    ) -> None:
        """Keep only the newest interactive request for a Solver SOP."""
        now = time.monotonic() if now is None else now
        active = self._active
        if (
            active is not None
            and active.job.node_path == job.node_path
            and active.job.signature == job.signature
        ):
            self._desired.pop(job.node_path, None)
            return

        queue = self._bake.get(job.node_path)
        if queue and any(item.signature == job.signature for item in queue):
            kept = deque(item for item in queue if item.signature != job.signature)
            removed = len(queue) - len(kept)
            self._bake[job.node_path] = kept
            self._bake_total[job.node_path] = max(
                self._bake_done.get(job.node_path, 0),
                self._bake_total.get(job.node_path, 0) - removed,
            )

        pending = self._desired.get(job.node_path)
        deadline = now + max(0.0, debounce_s)
        if pending is not None and pending.job.signature == job.signature:
            pending.deadline = min(pending.deadline, deadline)
            return

        self._cancelled.discard(job.node_path)
        self._errors.pop((job.node_path, job.signature), None)
        self._desired[job.node_path] = _Pending(job=job, deadline=deadline)

    def cancel_auto(self, node_path: str) -> None:
        self._desired.pop(node_path, None)

    def enqueue_bake(self, node_path: str, jobs: list[TimelineJob]) -> int:
        """Replace a Solver SOP's bake queue and deduplicate equal designs."""
        self.cancel_auto(node_path)
        existing: set[str] = set()
        if self._active is not None and self._active.job.node_path == node_path:
            existing.add(self._active.job.signature)

        unique: list[TimelineJob] = []
        for job in jobs:
            if job.signature not in existing:
                existing.add(job.signature)
                unique.append(job)

        self._bake[node_path] = deque(unique)
        active_count = int(
            self._active is not None
            and self._active.job.node_path == node_path
            and self._active.job.kind == "bake"
        )
        self._bake_done[node_path] = 0
        self._bake_total[node_path] = len(unique) + active_count
        self._cancelled.discard(node_path)
        return len(unique)

    def cancel_bake(self, node_path: str) -> None:
        self._bake.pop(node_path, None)
        self._cancelled.add(node_path)

    def status(self, node_path: str, signature: str) -> str:
        active = self._active
        if (
            active is not None
            and active.job.node_path == node_path
            and active.job.signature == signature
        ):
            return "running"
        pending = self._desired.get(node_path)
        if pending is not None and pending.job.signature == signature:
            return "queued"
        if any(job.signature == signature for job in self._bake.get(node_path, ())):
            return "bake-queued"
        if (node_path, signature) in self._errors:
            return "error"
        if node_path in self._cancelled:
            return "cancelled"
        return "not-baked"

    def error(self, node_path: str, signature: str) -> str:
        return self._errors.get((node_path, signature), "")

    def bake_progress(self, node_path: str) -> tuple[int, int]:
        return (
            self._bake_done.get(node_path, 0),
            self._bake_total.get(node_path, 0),
        )

    def tick(
        self,
        submit: Callable[[TimelineJob], Any],
        on_complete: Callable[[TimelineJob, AnalysisResult | None, BaseException | None], None],
        *,
        now: float | None = None,
        allow_submit: bool = True,
    ) -> None:
        """Poll once and, when allowed, submit at most one new XLB job."""
        now = time.monotonic() if now is None else now
        if self._active is not None:
            if not self._active.future.done():
                return
            active = self._active
            self._active = None
            result: AnalysisResult | None = None
            error: BaseException | None = None
            try:
                result = active.future.result()
                self._errors.pop((active.job.node_path, active.job.signature), None)
            except BaseException as exc:  # surfaced in xlb_status
                error = exc
                self._errors[(active.job.node_path, active.job.signature)] = str(exc)
            if active.job.kind == "bake":
                path = active.job.node_path
                self._bake_done[path] = self._bake_done.get(path, 0) + 1
            on_complete(active.job, result, error)

        if not allow_submit or self._active is not None:
            return

        selected: TimelineJob | None = None
        eligible = [pending for pending in self._desired.values() if pending.deadline <= now]
        if eligible:
            pending = min(eligible, key=lambda item: item.deadline)
            selected = pending.job
            self._desired.pop(selected.node_path, None)
        elif self._desired:
            # Do not start a long bake during an interactive debounce window.
            return
        else:
            for node_path in list(self._bake):
                queue = self._bake[node_path]
                if queue:
                    selected = queue.popleft()
                    break
                self._bake.pop(node_path, None)

        if selected is None:
            return
        try:
            self._active = _Active(job=selected, future=submit(selected))
        except BaseException as exc:
            self._errors[(selected.node_path, selected.signature)] = str(exc)
            on_complete(selected, None, exc)

    def ensure_houdini_callback(self) -> bool:
        """Register one GUI event-loop poller; do nothing in headless hython."""
        import hou

        if self._event_registered or not hou.isUIAvailable() or not hasattr(hou, "ui"):
            return self._event_registered

        def callback() -> None:
            _houdini_tick(self)

        self._event_callback = callback
        hou.ui.addEventLoopCallback(callback)
        self._event_registered = True
        return True

    def remove_houdini_callback_if_idle(self) -> None:
        if not self.idle or not self._event_registered or self._event_callback is None:
            return
        import hou

        if hasattr(hou, "ui"):
            hou.ui.removeEventLoopCallback(self._event_callback)
        self._event_registered = False
        self._event_callback = None


def session_scheduler() -> TimelineScheduler:
    """Return the transient worker queue retained for this Houdini session."""
    import hou

    name = "_houdini_xlb_timeline_scheduler"
    scheduler = getattr(hou.session, name, None)
    if scheduler is None:
        scheduler = TimelineScheduler()
        setattr(hou.session, name, scheduler)
    return scheduler


def _refresh_node(path: str | None) -> None:
    if not path:
        return
    import hou

    node = hou.node(path)
    if node is not None:
        node.cook(force=True)


def _resimulate_solver(path: str) -> None:
    import hou

    node = hou.node(path)
    if node is None:
        return
    reset = node.parm("resimulate")
    if reset is not None:
        reset.pressButton()


def _houdini_tick(scheduler: TimelineScheduler) -> None:
    """Poll the worker from Houdini's main UI thread."""
    import hou

    def submit(job: TimelineJob):
        return session_client(
            cache_dir=job.cache_dir,
            python_executable=job.python_executable,
        ).analyze_async(job.heightmap, job.config)

    def complete(
        job: TimelineJob,
        _result: AnalysisResult | None,
        _error: BaseException | None,
    ) -> None:
        done, total = scheduler.bake_progress(job.node_path)
        bake_finished = job.kind == "bake" and total > 0 and done >= total
        current_finished = job.frame == int(round(hou.frame())) and not hou.playbar.isPlaying()
        if bake_finished or current_finished:
            _resimulate_solver(job.node_path)
        _refresh_node(job.refresh_path or job.node_path)

    scheduler.tick(
        submit,
        complete,
        allow_submit=not bool(hou.playbar.isPlaying()),
    )
    scheduler.remove_houdini_callback_if_idle()


def _building_geometry(node):
    inputs = node.inputs()
    return inputs[1].geometry() if len(inputs) > 1 and inputs[1] is not None else None


def _analysis_input(node, control_node) -> tuple[np.ndarray, XlbConfig]:
    building_geometry = _building_geometry(node)
    length_x_m = float(control_node.evalParm("lengthx"))
    length_y_m = float(control_node.evalParm("lengthy"))
    domain_height_m = float(control_node.evalParm("domainheight"))
    profiles = profile_names()
    profile = XlbConfig.profile(profiles[int(control_node.evalParm("profile"))])
    config = profile.with_domain(
        length_x_m=length_x_m,
        length_y_m=length_y_m,
        height_m=domain_height_m,
        reference_height_m=float(control_node.evalParm("refheight")),
        pedestrian_height_m=float(control_node.evalParm("pedheight")),
    )
    if building_geometry is None or len(building_geometry.iterPoints()) == 0:
        heightmap = np.zeros((config.grid_y, config.grid_x), dtype=np.float32)
    else:
        heightmap = geometry_heightmap(
            building_geometry,
            ny=config.grid_y,
            nx=config.grid_x,
            length_x=length_x_m,
            length_y=length_y_m,
            domain_height_m=domain_height_m,
        )
    return heightmap, config


def _job_for_control(
    control_node,
    *,
    cache_dir: str | Path,
    python_executable: str | Path,
    refresh_path: str | None,
    kind: str = "auto",
) -> TimelineJob:
    import hou

    heightmap, config = _analysis_input(control_node, control_node)
    return TimelineJob(
        node_path=control_node.path(),
        signature=analysis_key(heightmap, config),
        heightmap=heightmap,
        config=config,
        cache_dir=Path(cache_dir).resolve(),
        python_executable=Path(python_executable).resolve(),
        frame=int(round(hou.frame())),
        kind=kind,
        refresh_path=refresh_path,
    )


def _global_attrib(geometry, name: str, default: object) -> None:
    import hou

    if geometry.findGlobalAttrib(name) is None:
        geometry.addAttrib(hou.attribType.Global, name, default)


def _previous_field(node, control_node, shape: tuple[int, int]) -> np.ndarray:
    geometry = node.geometry()
    attribute = geometry.findPointAttrib("windspeed")
    field = np.zeros(shape, dtype=np.float32)
    if attribute is None:
        return field
    values = np.asarray(
        geometry.pointFloatAttribValues("windspeed"),
        dtype=np.float32,
    )
    positions = np.asarray(
        geometry.pointFloatAttribValues("P"),
        dtype=np.float64,
    ).reshape(-1, 3)
    if len(values) != len(positions) or not len(values):
        return field
    ny, nx = shape
    length_x = float(control_node.evalParm("lengthx"))
    length_y = float(control_node.evalParm("lengthy"))
    ix = np.clip((positions[:, 0] / length_x * nx).astype(int), 0, nx - 1)
    iy = np.clip((positions[:, 1] / length_y * ny).astype(int), 0, ny - 1)
    np.maximum.at(field, (iy, ix), values)
    return field


def _paint_field(
    node,
    control_node,
    speed: np.ndarray,
    *,
    stale: bool,
    building_geometry,
    merge_buildings: bool,
) -> None:
    import hou

    geometry = node.geometry()
    length_x = float(control_node.evalParm("lengthx"))
    length_y = float(control_node.evalParm("lengthy"))
    ny, nx = speed.shape
    if geometry.findPointAttrib("Cd") is None:
        geometry.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
    if geometry.findPointAttrib("windspeed") is None:
        geometry.addAttrib(hou.attribType.Point, "windspeed", 0.0)

    positions = np.asarray(
        geometry.pointFloatAttribValues("P"),
        dtype=np.float64,
    ).reshape(-1, 3)
    if len(positions):
        ix = np.clip((positions[:, 0] / length_x * nx).astype(int), 0, nx - 1)
        iy = np.clip((positions[:, 1] / length_y * ny).astype(int), 0, ny - 1)
        point_speed = speed[iy, ix]
        vmax = float(control_node.evalParm("vmax"))
        if vmax <= 0:
            vmax = max(float(point_speed.max()), 1e-9)
        values = np.clip(point_speed / vmax, 0.0, 1.0)
        stops = np.asarray(
            [
                [0.02, 0.04, 0.16],
                [0.12, 0.32, 0.68],
                [0.18, 0.72, 0.72],
                [0.98, 0.78, 0.24],
                [0.78, 0.08, 0.08],
            ],
            dtype=np.float64,
        )
        locations = np.linspace(0.0, 1.0, len(stops))
        colours = np.stack(
            [np.interp(values, locations, stops[:, channel]) for channel in range(3)],
            axis=1,
        )
        if stale:
            colours *= np.asarray([0.62, 0.62, 0.62])
        geometry.setPointFloatAttribValues(
            "Cd",
            colours.astype(np.float32).ravel().tolist(),
        )
        geometry.setPointFloatAttribValues(
            "windspeed",
            point_speed.astype(np.float32).tolist(),
        )
    if merge_buildings and building_geometry is not None:
        geometry.merge(building_geometry)


def _write_state(
    node,
    *,
    control_path: str,
    role: str,
    status: str,
    job_state: str,
    signature: str,
    previous_key: str,
    stale: int,
    cache_hit: int,
    frame: int,
    done: int,
    total: int,
    elapsed_s: float,
) -> None:
    geometry = node.geometry()
    for name, default in (
        ("xlb_status", ""),
        ("xlb_job_state", ""),
        ("xlb_cache_key", ""),
        ("xlb_previous_cache_key", ""),
        ("xlb_state_role", ""),
        ("xlb_owner_path", ""),
    ):
        _global_attrib(geometry, name, default)
    for name, default in (
        ("xlb_solver_state", 1),
        ("xlb_stale", 1),
        ("xlb_cache_hit", 0),
        ("xlb_frame", 0),
        ("xlb_bake_done", 0),
        ("xlb_bake_total", 0),
    ):
        _global_attrib(geometry, name, default)
    _global_attrib(geometry, "xlb_elapsed_s", 0.0)
    geometry.setGlobalAttribValue("xlb_status", status)
    geometry.setGlobalAttribValue("xlb_job_state", job_state)
    geometry.setGlobalAttribValue("xlb_cache_key", signature)
    geometry.setGlobalAttribValue("xlb_previous_cache_key", previous_key)
    geometry.setGlobalAttribValue("xlb_state_role", role)
    geometry.setGlobalAttribValue("xlb_owner_path", control_path)
    geometry.setGlobalAttribValue("xlb_solver_state", 1)
    geometry.setGlobalAttribValue("xlb_stale", stale)
    geometry.setGlobalAttribValue("xlb_cache_hit", cache_hit)
    geometry.setGlobalAttribValue("xlb_frame", frame)
    geometry.setGlobalAttribValue("xlb_bake_done", done)
    geometry.setGlobalAttribValue("xlb_bake_total", total)
    geometry.setGlobalAttribValue("xlb_elapsed_s", elapsed_s)


def cook_solver_sop(
    node,
    *,
    control_node,
    refresh_path: str | None,
    cache_dir: str | Path,
    python_executable: str | Path,
    merge_buildings: bool = False,
    role: str = "solver",
) -> None:
    """Cook one Solver-SOP state or its downstream display layer."""
    import hou

    cache_dir = Path(cache_dir).resolve()
    python_executable = Path(python_executable).resolve()
    heightmap, config = _analysis_input(node, control_node)
    signature = analysis_key(heightmap, config)
    frame = int(round(hou.frame()))
    scheduler = session_scheduler()
    building_geometry = _building_geometry(node)
    geometry = node.geometry()
    old_key_attrib = geometry.findGlobalAttrib("xlb_cache_key")
    previous_key = str(geometry.attribValue(old_key_attrib)) if old_key_attrib is not None else ""
    previous = _previous_field(node, control_node, heightmap.shape)
    result = load_cached_heightmap(heightmap, config, cache_dir=cache_dir)
    gui_available = hou.isUIAvailable() and hasattr(hou, "ui")
    playing = bool(hou.playbar.isPlaying()) if gui_available else False

    if result is not None:
        scheduler.cancel_auto(control_node.path())
        speed = np.asarray(result.speed, dtype=np.float32)
        status = "current"
        job_state = "current"
        stale = 0
        cache_hit = int(result.cache_hit)
        elapsed_s = float(result.elapsed_s)
    else:
        job = TimelineJob(
            node_path=control_node.path(),
            signature=signature,
            heightmap=heightmap,
            config=config,
            cache_dir=cache_dir,
            python_executable=python_executable,
            frame=frame,
            kind="auto",
            refresh_path=refresh_path,
        )
        collecting = scheduler.is_collecting(control_node.path())
        if playing:
            scheduler.cancel_auto(control_node.path())
        elif (
            gui_available
            and not collecting
            and bool(control_node.evalParm("autoanalyze"))
            and scheduler.status(control_node.path(), signature) not in {"error", "bake-queued"}
        ):
            scheduler.request(
                job,
                debounce_s=float(control_node.evalParm("debounce")),
            )
            scheduler.ensure_houdini_callback()

        job_state = scheduler.status(control_node.path(), signature)
        if collecting:
            job_state = "collecting"
            status = "collecting bake range"
        elif playing:
            job_state = "not-baked"
            status = "not-baked: playback uses cached frames"
        elif job_state == "queued":
            status = f"queued: frame {frame}"
        elif job_state == "bake-queued":
            status = f"bake-queued: frame {frame}"
        elif job_state == "running":
            status = f"running: frame {frame}"
        elif job_state == "error":
            status = "error: " + scheduler.error(control_node.path(), signature)
        elif job_state == "cancelled":
            status = "bake cancelled"
        else:
            status = "not-baked: pause to analyze or use Bake Range"
        speed = previous
        stale = 1
        cache_hit = 0
        elapsed_s = 0.0

    done, total = scheduler.bake_progress(control_node.path())
    _write_state(
        node,
        control_path=control_node.path(),
        role=role,
        status=status,
        job_state=job_state,
        signature=signature,
        previous_key=previous_key,
        stale=stale,
        cache_hit=cache_hit,
        frame=frame,
        done=done,
        total=total,
        elapsed_s=elapsed_s,
    )
    _paint_field(
        node,
        control_node,
        speed,
        stale=bool(stale),
        building_geometry=building_geometry,
        merge_buildings=merge_buildings,
    )


def cook_timeline_sop(
    node,
    *,
    cache_dir: str | Path,
    python_executable: str | Path,
) -> None:
    """Backward-compatible single-Python-SOP entry point."""
    cook_solver_sop(
        node,
        control_node=node,
        refresh_path=node.path(),
        cache_dir=cache_dir,
        python_executable=python_executable,
        merge_buildings=True,
        role="legacy-display",
    )


def run_now(
    control_node,
    *,
    cache_dir: str | Path,
    python_executable: str | Path,
    refresh_path: str | None,
) -> AnalysisResult | None:
    """Queue the current Solver frame immediately, or run synchronously in hython."""
    import hou

    job = _job_for_control(
        control_node,
        cache_dir=cache_dir,
        python_executable=python_executable,
        refresh_path=refresh_path,
    )
    cached = load_cached_heightmap(
        job.heightmap,
        job.config,
        cache_dir=job.cache_dir,
    )
    if cached is not None:
        _resimulate_solver(control_node.path())
        _refresh_node(refresh_path)
        return cached
    if not hou.isUIAvailable() or not hasattr(hou, "ui"):
        result = session_client(
            cache_dir=job.cache_dir,
            python_executable=job.python_executable,
        ).analyze(job.heightmap, job.config)
        _resimulate_solver(control_node.path())
        _refresh_node(refresh_path)
        return result

    scheduler = session_scheduler()
    scheduler.request(job, debounce_s=0.0)
    scheduler.ensure_houdini_callback()
    _refresh_node(refresh_path)
    return None


def bake_range(
    control_node,
    *,
    cache_dir: str | Path,
    python_executable: str | Path,
    refresh_path: str | None,
) -> None:
    """Collect animated designs, then analyse unique cache misses sequentially."""
    import hou

    start = int(control_node.evalParm("bakestart"))
    end = int(control_node.evalParm("bakeend"))
    step = max(1, int(control_node.evalParm("bakestep")))
    direction = 1 if end >= start else -1
    frames = range(start, end + direction, step * direction)
    scheduler = session_scheduler()
    scheduler.cancel_auto(control_node.path())
    scheduler.begin_collection(control_node.path())
    original_frame = hou.frame()
    jobs: list[TimelineJob] = []
    seen: set[str] = set()
    try:
        for frame in frames:
            hou.setFrame(frame)
            job = _job_for_control(
                control_node,
                cache_dir=cache_dir,
                python_executable=python_executable,
                refresh_path=refresh_path,
                kind="bake",
            )
            if job.signature in seen:
                continue
            seen.add(job.signature)
            if (
                load_cached_heightmap(
                    job.heightmap,
                    job.config,
                    cache_dir=job.cache_dir,
                )
                is None
            ):
                jobs.append(job)
    finally:
        hou.setFrame(original_frame)
        scheduler.end_collection(control_node.path())

    scheduler.enqueue_bake(control_node.path(), jobs)
    if jobs:
        scheduler.ensure_houdini_callback()
    else:
        _resimulate_solver(control_node.path())
    _refresh_node(refresh_path)


def cancel_bake(
    control_node,
    *,
    cache_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    refresh_path: str | None = None,
) -> None:
    """Stop submitting range jobs after the active XLB call."""
    scheduler = session_scheduler()
    scheduler.cancel_bake(control_node.path())
    _refresh_node(refresh_path)


def refresh_display(
    control_node,
    *,
    cache_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    refresh_path: str | None = None,
) -> None:
    """Recook only the display layer after a colour or auto-analysis change."""
    _refresh_node(refresh_path)


def refresh_solver(
    control_node,
    *,
    cache_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    refresh_path: str | None = None,
) -> None:
    """Reset the Solver simulation when analysis settings change."""
    _resimulate_solver(control_node.path())
    _refresh_node(refresh_path)
