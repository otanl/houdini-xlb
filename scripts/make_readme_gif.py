"""Generate the README optimization GIF through Houdini and real XLB."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit('Install demo dependencies with: uv pip install -e ".[demo]"') from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZATION = PROJECT_ROOT / "examples" / "houdini_xlb_demo_optimization.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "cache" / "xlb"


def find_hython(configured: str | None) -> Path:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    if os.environ.get("HYTHON"):
        candidates.append(Path(os.environ["HYTHON"]))
    if found := shutil.which("hython"):
        candidates.append(Path(found))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates.extend(sorted(program_files.glob("Side Effects Software/Houdini */bin/hython.exe")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("hython.exe not found; pass --hython")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def compose_frame(field_path: Path, item: dict[str, object]) -> Image.Image:
    field = Image.open(field_path).convert("RGB")
    panel_width = 320
    canvas = Image.new("RGB", (field.width + panel_width, field.height), "#11161c")
    canvas.paste(field, (0, 0))
    draw = ImageDraw.Draw(canvas)
    panel_x = field.width
    draw.line((panel_x, 0, panel_x, field.height), fill="#39434d", width=2)

    title = font(22, bold=True)
    metric = font(22, bold=True)
    heading = font(17, bold=True)
    body = font(16)
    small = font(13)
    tiny = font(12)
    draw.text((panel_x + 20, 20), "XLB LAYOUT OPTIMIZATION", font=title, fill="#f3f6f8")
    draw.text(
        (panel_x + 20, 54),
        "4 VARIABLES · 16 REAL XLB CASES",
        font=small,
        fill="#9ba7b3",
    )

    stage = str(item["stage"])
    badge_fill = {
        "BASELINE": "#52616f",
        "BEST SO FAR": "#a76d17",
        "GLOBAL BEST": "#17864b",
    }[stage]
    badge_width = {"BASELINE": 116, "BEST SO FAR": 142, "GLOBAL BEST": 142}[stage]
    draw.rounded_rectangle(
        (panel_x + 20, 82, panel_x + 20 + badge_width, 114),
        radius=9,
        fill=badge_fill,
    )
    draw.text((panel_x + 35, 89), stage, font=small, fill="white")

    draw.rectangle((panel_x + 20, 132, panel_x + 34, 146), fill="#3978d4")
    draw.text((panel_x + 42, 130), "MOVABLE", font=tiny, fill="#dce3e8")
    draw.rectangle((panel_x + 151, 132, panel_x + 165, 146), fill="#7b8792")
    draw.text((panel_x + 173, 130), "FIXED", font=tiny, fill="#dce3e8")
    draw.line((panel_x + 20, 166, panel_x + 45, 166), fill="#ffb82e", width=5)
    draw.text((panel_x + 54, 155), "PLAZA", font=tiny, fill="#dce3e8")
    draw.line((panel_x + 151, 166, panel_x + 176, 166), fill="#2ee0eb", width=5)
    draw.text((panel_x + 185, 155), "VENT ROUTE", font=tiny, fill="#dce3e8")

    draw.line((panel_x + 20, 188, panel_x + panel_width - 20, 188), fill="#35404a", width=1)
    draw.text(
        (panel_x + 20, 202),
        f"EVALUATED  {int(item['evaluation']):02d} / {int(item['evaluation_count']):02d}",
        font=heading,
        fill="#f3f6f8",
    )
    draw.text(
        (panel_x + 20, 230),
        f"best found @ evaluation {int(item['best_evaluation'])}",
        font=small,
        fill="#bbc5ce",
    )
    design = [float(value) for value in item["design"]]
    draw.text(
        (panel_x + 20, 251),
        f"centers  L({design[0]:.0f},{design[1]:.0f})  U({design[2]:.0f},{design[3]:.0f})",
        font=tiny,
        fill="#84919b",
    )
    timing = "cache hit" if item["cache_hit"] else f"{item['elapsed_s']:.1f} s GPU"
    draw.text((panel_x + 20, 271), f"XLB draft · {timing}", font=tiny, fill="#68d99a")

    draw.text((panel_x + 20, 300), "PLAZA IN COMFORT BAND", font=small, fill="#ffb82e")
    comfort = float(item["comfort_fraction"]) * 100.0
    draw.text(
        (panel_x + 20, 320),
        f"{comfort:.1f}%",
        font=metric,
        fill="#f3f6f8",
    )
    change = float(item["comfort_change_pp"])
    change_text = "baseline" if abs(change) < 0.05 else f"{change:+.1f} percentage points"
    draw.text(
        (panel_x + 20, 352),
        change_text,
        font=body,
        fill="#68d99a" if change >= -0.05 else "#ef767a",
    )
    draw.text(
        (panel_x + 20, 377),
        f"band  {item['comfort_min']:.2f} <= U/Uin <= {item['comfort_max']:.2f}",
        font=tiny,
        fill="#84919b",
    )
    draw.text(
        (panel_x + 20, 398),
        f"outside band objective  {float(item['objective']):.1%}",
        font=tiny,
        fill="#bbc5ce",
    )

    draw.text((panel_x + 20, 428), "CENTRAL VENT ROUTE", font=small, fill="#2ee0eb")
    retained = float(item["vent_retained_pct"])
    draw.text(
        (panel_x + 20, 448),
        f"{retained:.1f}% retained",
        font=metric,
        fill="#f3f6f8",
    )
    threshold = float(item["min_vent_retention_pct"])
    draw.text(
        (panel_x + 20, 480),
        f"hard constraint  >= {threshold:.0f}% of baseline",
        font=small,
        fill="#68d99a" if retained + 1.0e-6 >= threshold else "#ef767a",
    )

    feasible = retained + 1.0e-6 >= threshold
    draw.rounded_rectangle(
        (panel_x + 20, 507, panel_x + panel_width - 20, 539),
        radius=8,
        outline="#4f9f76" if feasible else "#a94b50",
        width=2,
    )
    draw.text(
        (panel_x + 33, 514),
        f"FEASIBLE · NO OVERLAP · {item['clearance_m']:.1f} m CLEAR",
        font=tiny,
        fill="#b8c9d3",
    )

    draw.text((panel_x + 20, 553), "PEDESTRIAN SPEED  U / Uin", font=tiny, fill="#9ba7b3")
    bar_x0 = panel_x + 20
    bar_x1 = panel_x + panel_width - 20
    bar_y0 = 572
    stops = (
        (5, 10, 41),
        (31, 82, 173),
        (46, 184, 184),
        (250, 199, 61),
        (199, 20, 20),
    )
    for x in range(bar_x0, bar_x1):
        value = (x - bar_x0) / max(bar_x1 - bar_x0 - 1, 1) * (len(stops) - 1)
        index = min(int(value), len(stops) - 2)
        blend = value - index
        colour = tuple(
            round(stops[index][channel] * (1.0 - blend) + stops[index + 1][channel] * blend)
            for channel in range(3)
        )
        draw.line((x, bar_y0, x, bar_y0 + 10), fill=colour)
    vmax = float(item["colour_vmax_ratio"])
    draw.text((bar_x0, bar_y0 + 12), "0", font=tiny, fill="#84919b")
    draw.text(
        ((bar_x0 + bar_x1) // 2 - 7, bar_y0 + 12),
        f"{vmax / 2:.0f}",
        font=tiny,
        fill="#84919b",
    )
    draw.text((bar_x1 - 12, bar_y0 + 12), f"{vmax:.0f}", font=tiny, fill="#84919b")

    arrow_y = field.height - 27
    draw.text((panel_x + 20, arrow_y - 15), "WIND +X", font=tiny, fill="#9ba7b3")
    draw.line(
        (panel_x + 88, arrow_y - 7, panel_x + panel_width - 36, arrow_y - 7),
        fill="#dbe4eb",
        width=3,
    )
    draw.polygon(
        (
            (panel_x + panel_width - 36, arrow_y - 13),
            (panel_x + panel_width - 22, arrow_y - 7),
            (panel_x + panel_width - 36, arrow_y - 1),
        ),
        fill="#dbe4eb",
    )
    draw.text(
        (panel_x + 20, field.height - 14),
        "timeline = best-so-far milestones",
        font=tiny,
        fill="#6f7c87",
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hython")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "assets" / "houdini_xlb_demo.gif",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "readme-demo" / "frames",
    )
    parser.add_argument(
        "--optimization",
        type=Path,
        default=DEFAULT_OPTIMIZATION,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    optimization = args.optimization.resolve()
    cache_dir = args.cache_dir.resolve()
    if not args.skip_optimize:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "optimize_demo.py"),
            "--out",
            str(optimization),
            "--cache-dir",
            str(cache_dir),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    work_dir = args.work_dir.resolve()
    metadata_path = work_dir / "frames.json"
    if not args.skip_render:
        hython = find_hython(args.hython)
        command = [
            str(hython),
            str(PROJECT_ROOT / "houdini" / "render_demo_frames.py"),
            "--out-dir",
            str(work_dir),
            "--python-executable",
            sys.executable,
            "--optimization",
            str(optimization),
            "--cache-dir",
            str(cache_dir),
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = [compose_frame(work_dir / item["file"], item) for item in metadata]
    durations = [1600, 1800, 1800, 2200, 2600]
    if len(frames) != len(durations):
        raise RuntimeError(f"expected {len(durations)} milestones, got {len(frames)}")

    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
