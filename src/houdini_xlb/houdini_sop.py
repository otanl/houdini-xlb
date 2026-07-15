"""Python-SOP and parameter support for timeline-aware XLB analysis."""

from __future__ import annotations

from pathlib import Path

_SOP_TEMPLATE = r"""
import sys

import hou

package_src = r"__PACKAGE_SRC__"
if package_src not in sys.path:
    sys.path.insert(0, package_src)

from houdini_xlb.timeline import cook_timeline_sop

cook_timeline_sop(
    hou.pwd(),
    cache_dir=r"__CACHE_DIR__",
    python_executable=r"__PYTHON_EXE__",
)
"""


def sop_code(
    *,
    package_src: str | Path,
    cache_dir: str | Path,
    python_executable: str | Path,
) -> str:
    """Render the small Python SOP body with explicit runtime paths."""
    replacements = {
        "__PACKAGE_SRC__": Path(package_src).resolve().as_posix(),
        "__CACHE_DIR__": Path(cache_dir).resolve().as_posix(),
        "__PYTHON_EXE__": Path(python_executable).resolve().as_posix(),
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
) -> str:
    package = Path(package_src).resolve().as_posix()
    cache = Path(cache_dir).resolve().as_posix()
    python = Path(python_executable).resolve().as_posix()
    arguments = (
        f", cache_dir=r'{cache}', python_executable=r'{python}'" if function == "bake_range" else ""
    )
    return (
        "import sys\n"
        f"package_src = r'{package}'\n"
        "if package_src not in sys.path:\n"
        "    sys.path.insert(0, package_src)\n"
        f"from houdini_xlb.timeline import {function}\n"
        f"{function}(kwargs['node']{arguments})"
    )


def install_parameters(
    sop,
    *,
    package_src: str | Path,
    cache_dir: str | Path,
    python_executable: str | Path,
) -> None:
    """Install automatic analysis, range-bake, and display controls."""
    import hou

    group = sop.parmTemplateGroup()
    folder = hou.FolderParmTemplate("xlb", "XLB Timeline")

    auto = hou.ToggleParmTemplate(
        "autoanalyze",
        "Auto Analyze on Pause",
        default_value=True,
    )
    auto.setHelp("When playback is stopped, analyse the current uncached frame after the debounce.")
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
    run.setScriptCallback(
        "node = kwargs['node']\n"
        "node.parm('request').set(node.evalParm('request') + 1)\n"
        "node.cook(force=True)"
    )
    run.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(run)

    bake = hou.ButtonParmTemplate("bakerange", "Bake Range")
    bake.setScriptCallback(
        _module_callback(
            "bake_range",
            package_src=package_src,
            cache_dir=cache_dir,
            python_executable=python_executable,
        )
    )
    bake.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(bake)

    cancel = hou.ButtonParmTemplate("cancelbake", "Cancel Bake")
    cancel.setScriptCallback(
        _module_callback(
            "cancel_bake",
            package_src=package_src,
            cache_dir=cache_dir,
            python_executable=python_executable,
        )
    )
    cancel.setScriptCallbackLanguage(hou.scriptLanguage.Python)
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

    request = hou.IntParmTemplate("request", "request", 1, default_value=(0,))
    request.hide(True)
    folder.addParmTemplate(request)
    folder.addParmTemplate(
        hou.MenuParmTemplate(
            "profile",
            "Analysis Profile",
            ("draft", "preview", "quality"),
            ("draft", "preview", "quality"),
            default_value=0,
        )
    )
    folder.addParmTemplate(
        hou.IntParmTemplate(
            "ny",
            "Height-map Rows",
            1,
            default_value=(96,),
            min=16,
            max=1024,
        )
    )
    folder.addParmTemplate(
        hou.IntParmTemplate(
            "nx",
            "Height-map Cols",
            1,
            default_value=(96,),
            min=16,
            max=1024,
        )
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "lengthx",
            "Domain X [m]",
            1,
            default_value=(100.0,),
            min=1.0,
        )
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "lengthy",
            "Domain Y [m]",
            1,
            default_value=(100.0,),
            min=1.0,
        )
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "domainheight",
            "Domain Height [m]",
            1,
            default_value=(40.0,),
            min=1.0,
        )
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "vmax",
            "Colour Vmax (0=Auto)",
            1,
            default_value=(0.0,),
            min=0.0,
        )
    )
    group.append(folder)
    sop.setParmTemplateGroup(group)
