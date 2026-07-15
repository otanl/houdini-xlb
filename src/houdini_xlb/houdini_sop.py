"""Python-SOP support for an explicit, cached XLB confirmation step in Houdini."""

from __future__ import annotations

from pathlib import Path

_SOP_TEMPLATE = r"""
import hashlib
import sys

import hou
import numpy as np

package_src = r"__PACKAGE_SRC__"
if package_src not in sys.path:
    sys.path.insert(0, package_src)

from houdini_xlb.config import XlbConfig
from houdini_xlb.houdini import geometry_heightmap, session_client

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
bgeo = inputs[1].geometry() if len(inputs) > 1 and inputs[1] is not None else None
ny = int(node.evalParm("ny"))
nx = int(node.evalParm("nx"))
length_x = float(node.evalParm("lengthx"))
length_y = float(node.evalParm("lengthy"))
domain_height = float(node.evalParm("domainheight"))
profile = ("draft", "preview", "quality")[int(node.evalParm("profile"))]
request = int(node.evalParm("request"))

if bgeo is None or len(bgeo.iterPoints()) == 0:
    heightmap = np.zeros((ny, nx), dtype=np.float32)
else:
    heightmap = geometry_heightmap(
        bgeo,
        ny=ny,
        nx=nx,
        length_x=length_x,
        length_y=length_y,
        domain_height_m=domain_height,
    )

signature = hashlib.sha256()
signature.update(heightmap.tobytes())
signature.update(profile.encode("utf-8"))
signature = signature.hexdigest()

session = hou.session
if not hasattr(session, "_houdini_xlb_sop_state"):
    session._houdini_xlb_sop_state = {}
states = session._houdini_xlb_sop_state
state = states.get(node.path())

if request > 0 and (state is None or request != state["request"]):
    result = session_client(
        cache_dir=r"__CACHE_DIR__",
        python_executable=r"__PYTHON_EXE__",
    ).analyze(heightmap, XlbConfig.profile(profile))
    state = {
        "request": request,
        "signature": signature,
        "speed": result.speed,
        "cache_hit": result.cache_hit,
        "elapsed_s": result.elapsed_s,
        "cache_key": result.cache_key,
    }
    states[node.path()] = state

if state is None:
    speed = np.zeros((ny, nx), dtype=np.float32)
    status = "not-run: press Run XLB"
    stale = 1
    cache_hit = 0
    elapsed_s = 0.0
    cache_key = ""
else:
    speed = np.asarray(state["speed"], dtype=np.float32)
    stale = int(state["signature"] != signature)
    cache_hit = int(state["cache_hit"])
    elapsed_s = float(state["elapsed_s"])
    cache_key = str(state["cache_key"])
    status = "stale: geometry changed; press Run XLB" if stale else "current"

for name, default in (
    ("xlb_status", ""),
    ("xlb_cache_key", ""),
):
    if geo.findGlobalAttrib(name) is None:
        geo.addAttrib(hou.attribType.Global, name, default)
for name, default in (
    ("xlb_stale", 1),
    ("xlb_cache_hit", 0),
):
    if geo.findGlobalAttrib(name) is None:
        geo.addAttrib(hou.attribType.Global, name, default)
if geo.findGlobalAttrib("xlb_elapsed_s") is None:
    geo.addAttrib(hou.attribType.Global, "xlb_elapsed_s", 0.0)
geo.setGlobalAttribValue("xlb_status", status)
geo.setGlobalAttribValue("xlb_cache_key", cache_key)
geo.setGlobalAttribValue("xlb_stale", stale)
geo.setGlobalAttribValue("xlb_cache_hit", cache_hit)
geo.setGlobalAttribValue("xlb_elapsed_s", elapsed_s)

if geo.findPointAttrib("Cd") is None:
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
if geo.findPointAttrib("windspeed") is None:
    geo.addAttrib(hou.attribType.Point, "windspeed", 0.0)

positions = np.asarray(geo.pointFloatAttribValues("P"), dtype=np.float64).reshape(-1, 3)
ix = np.clip((positions[:, 0] / length_x * nx).astype(int), 0, nx - 1)
iy = np.clip((positions[:, 1] / length_y * ny).astype(int), 0, ny - 1)
point_speed = speed[iy, ix]
vmax = float(node.evalParm("vmax"))
if vmax <= 0:
    vmax = max(float(point_speed.max()), 1e-9)
t = np.clip(point_speed / vmax, 0.0, 1.0)
stops = np.asarray(
    [[0.02, 0.04, 0.16], [0.12, 0.32, 0.68], [0.18, 0.72, 0.72],
     [0.98, 0.78, 0.24], [0.78, 0.08, 0.08]],
    dtype=np.float64,
)
locations = np.linspace(0.0, 1.0, len(stops))
colours = np.stack(
    [np.interp(t, locations, stops[:, channel]) for channel in range(3)],
    axis=1,
)
if stale:
    colours *= np.asarray([0.62, 0.62, 0.62])
geo.setPointFloatAttribValues("Cd", colours.astype(np.float32).ravel().tolist())
geo.setPointFloatAttribValues("windspeed", point_speed.astype(np.float32).tolist())

if bgeo is not None:
    geo.merge(bgeo)
"""


def sop_code(
    *,
    package_src: str | Path,
    cache_dir: str | Path,
    python_executable: str | Path,
) -> str:
    """Render the self-contained Python SOP body with explicit runtime paths."""
    replacements = {
        "__PACKAGE_SRC__": Path(package_src).resolve().as_posix(),
        "__CACHE_DIR__": Path(cache_dir).resolve().as_posix(),
        "__PYTHON_EXE__": Path(python_executable).resolve().as_posix(),
    }
    code = _SOP_TEMPLATE
    for token, value in replacements.items():
        code = code.replace(token, value)
    return code


def install_parameters(sop) -> None:
    """Install Run/profile/domain/display controls on a Houdini Python SOP."""
    import hou

    group = sop.parmTemplateGroup()
    folder = hou.FolderParmTemplate("xlb", "XLB confirmation")

    run = hou.ButtonParmTemplate("runxlb", "Run XLB")
    run.setScriptCallback(
        "node = kwargs['node']\n"
        "node.parm('request').set(node.evalParm('request') + 1)\n"
        "node.cook(force=True)"
    )
    run.setScriptCallbackLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(run)

    request = hou.IntParmTemplate("request", "request", 1, default_value=(0,))
    request.hide(True)
    folder.addParmTemplate(request)
    folder.addParmTemplate(
        hou.MenuParmTemplate(
            "profile",
            "profile",
            ("draft", "preview", "quality"),
            ("draft", "preview", "quality"),
            default_value=1,
        )
    )
    folder.addParmTemplate(
        hou.IntParmTemplate("ny", "height-map rows", 1, default_value=(96,), min=16, max=1024)
    )
    folder.addParmTemplate(
        hou.IntParmTemplate("nx", "height-map cols", 1, default_value=(96,), min=16, max=1024)
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate("lengthx", "domain X [m]", 1, default_value=(100.0,), min=1.0)
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate("lengthy", "domain Y [m]", 1, default_value=(100.0,), min=1.0)
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "domainheight",
            "domain height [m]",
            1,
            default_value=(40.0,),
            min=1.0,
        )
    )
    folder.addParmTemplate(
        hou.FloatParmTemplate(
            "vmax",
            "colour vmax (0=auto)",
            1,
            default_value=(0.0,),
            min=0.0,
        )
    )
    group.append(folder)
    sop.setParmTemplateGroup(group)
