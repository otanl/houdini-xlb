"""Python-SOP code and controls for the Houdini XLB Solver SOP."""

from __future__ import annotations

from pathlib import Path

_SOP_TEMPLATE = r"""
import sys

import hou

package_src = r"__PACKAGE_SRC__"
if package_src not in sys.path:
    sys.path.insert(0, package_src)

from houdini_xlb.timeline import cook_solver_sop

control_path = r"__CONTROL_PATH__"
control_node = hou.node(control_path) if control_path else hou.pwd()
if control_node is None:
    raise RuntimeError("Houdini XLB control node is missing: " + control_path)

cook_solver_sop(
    hou.pwd(),
    control_node=control_node,
    refresh_path=r"__REFRESH_PATH__",
    cache_dir=r"__CACHE_DIR__",
    python_executable=r"__PYTHON_EXE__",
    merge_buildings=__MERGE_BUILDINGS__,
    role="__STATE_ROLE__",
)
"""


def sop_code(
    *,
    package_src: str | Path,
    cache_dir: str | Path,
    python_executable: str | Path,
    control_path: str | None = None,
    refresh_path: str | None = None,
    merge_buildings: bool = True,
    role: str = "display",
) -> str:
    """Render a Python SOP body for init, step, or display."""
    replacements = {
        "__PACKAGE_SRC__": Path(package_src).resolve().as_posix(),
        "__CACHE_DIR__": Path(cache_dir).resolve().as_posix(),
        "__PYTHON_EXE__": Path(python_executable).resolve().as_posix(),
        "__CONTROL_PATH__": control_path or "",
        "__REFRESH_PATH__": refresh_path or "",
        "__MERGE_BUILDINGS__": repr(bool(merge_buildings)),
        "__STATE_ROLE__": role,
    }
    code = _SOP_TEMPLATE
    for token, value in replacements.items():
        code = code.replace(token, value)
    return code


def _module_callback(
    function: str,
    *,
    package_src: str | Path,
    cache_dir: str | Path,
    python_executable: str | Path,
    refresh_path: str,
) -> str:
    package = Path(package_src).resolve().as_posix()
    cache = Path(cache_dir).resolve().as_posix()
    python = Path(python_executable).resolve().as_posix()
    lines = (
        "import sys",
        f"package_src = r'{package}'",
        "if package_src not in sys.path:",
        "    sys.path.insert(0, package_src)",
        f"from houdini_xlb.timeline import {function}",
        (
            f"{function}(kwargs['node'], "
            f"cache_dir=r'{cache}', "
            f"python_executable=r'{python}', "
            f"refresh_path=r'{refresh_path}')"
        ),
    )
    return chr(10).join(lines)


def _set_callback(template, code: str) -> None:
    import hou

    template.setScriptCallback(code)
    template.setScriptCallbackLanguage(hou.scriptLanguage.Python)


def install_parameters(
    solver,
    *,
    package_src: str | Path,
    cache_dir: str | Path,
    python_executable: str | Path,
    refresh_path: str,
) -> None:
    """Install automatic analysis, range bake, and display controls on a Solver SOP."""
    import hou

    callback = {
        name: _module_callback(
            name,
            package_src=package_src,
            cache_dir=cache_dir,
            python_executable=python_executable,
            refresh_path=refresh_path,
        )
        for name in (
            "run_now",
            "bake_range",
            "cancel_bake",
            "refresh_display",
            "refresh_solver",
        )
    }

    group = solver.parmTemplateGroup()
    folder = hou.FolderParmTemplate("xlb", "XLB Solver")

    auto = hou.ToggleParmTemplate(
        "autoanalyze",
        "Auto Analyze on Pause",
        default_value=True,
    )
    auto.setHelp(
        "When playback is stopped, analyse the current uncached Solver frame after the debounce."
    )
    _set_callback(auto, callback["refresh_display"])
    folder.addParmTemplate(auto)
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "debounce",
            "Pause Debounce [s]",
            1,
            default_value=(0.75,),
            min=0.0,
            max=10.0,
        )
    )

    run = hou.ButtonParmTemplate("runxlb", "Run Now (Current Frame)")
    _set_callback(run, callback["run_now"])
    folder.addParmTemplate(run)

    bake = hou.ButtonParmTemplate("bakerange", "Bake Range")
    _set_callback(bake, callback["bake_range"])
    folder.addParmTemplate(bake)

    cancel = hou.ButtonParmTemplate("cancelbake", "Cancel Bake")
    _set_callback(cancel, callback["cancel_bake"])
    folder.addParmTemplate(cancel)

    folder.addParmTemplate(hou.IntParmTemplate("bakestart", "Bake Start", 1, default_value=(1,)))
    folder.addParmTemplate(hou.IntParmTemplate("bakeend", "Bake End", 1, default_value=(36,)))
    folder.addParmTemplate(
        hou.IntParmTemplate(
            "bakestep",
            "Bake Step",
            1,
            default_value=(1,),
            min=1,
        )
    )

    profile = hou.MenuParmTemplate(
        "profile",
        "Analysis Profile",
        ("draft", "preview", "quality"),
        ("draft", "preview", "quality"),
        default_value=0,
    )
    _set_callback(profile, callback["refresh_solver"])
    folder.addParmTemplate(profile)

    ny = hou.IntParmTemplate(
        "ny",
        "Height-map Rows",
        1,
        default_value=(96,),
        min=16,
        max=1024,
    )
    nx = hou.IntParmTemplate(
        "nx",
        "Height-map Cols",
        1,
        default_value=(96,),
        min=16,
        max=1024,
    )
    length_x = hou.FloatParmTemplate(
        "lengthx",
        "Domain X [m]",
        1,
        default_value=(100.0,),
        min=1.0,
    )
    length_y = hou.FloatParmTemplate(
        "lengthy",
        "Domain Y [m]",
        1,
        default_value=(100.0,),
        min=1.0,
    )
    domain_height = hou.FloatParmTemplate(
        "domainheight",
        "Domain Height [m]",
        1,
        default_value=(40.0,),
        min=1.0,
    )
    for template in (ny, nx, length_x, length_y, domain_height):
        _set_callback(template, callback["refresh_solver"])
        folder.addParmTemplate(template)

    vmax = hou.FloatParmTemplate(
        "vmax",
        "Colour Vmax (0=Auto)",
        1,
        default_value=(0.0,),
        min=0.0,
    )
    _set_callback(vmax, callback["refresh_display"])
    folder.addParmTemplate(vmax)

    group.append(folder)
    solver.setParmTemplateGroup(group)
