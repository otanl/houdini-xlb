"""Generate the README GIF through Houdini OpenGL and real XLB simulations."""

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
    panel_width = 292
    canvas = Image.new("RGB", (field.width + panel_width, field.height), "#11161c")
    canvas.paste(field, (0, 0))
    draw = ImageDraw.Draw(canvas)
    panel_x = field.width
    draw.line((panel_x, 0, panel_x, field.height), fill="#39434d", width=2)

    title = font(27, bold=True)
    heading = font(18, bold=True)
    body = font(17)
    small = font(14)
    draw.text((panel_x + 24, 28), "SOLVER SOP + XLB", font=title, fill="#f3f6f8")
    draw.text((panel_x + 24, 68), "PREV_FRAME WIND STUDY", font=small, fill="#9ba7b3")

    badge = "BAKED"
    badge_fill = "#17864b"
    draw.rounded_rectangle(
        (panel_x + 24, 106, panel_x + 146, 140),
        radius=9,
        fill=badge_fill,
    )
    draw.text((panel_x + 40, 113), badge, font=small, fill="white")

    steps = (("●", "AUTO ON PAUSE"), ("▶", "CACHE PLAYBACK"))
    y = 180
    for symbol, label in steps:
        color = "#34d17b"
        draw.ellipse((panel_x + 24, y, panel_x + 56, y + 32), fill=color)
        draw.text((panel_x + 34, y + 6), symbol, font=small, fill="white")
        draw.text((panel_x + 70, y + 5), label, font=heading, fill="#eef2f5")
        y += 58

    draw.line((panel_x + 24, 307, panel_x + panel_width - 24, 307), fill="#35404a", width=1)
    draw.text(
        (panel_x + 24, 332),
        f"FRAME {item['timeline_frame']:02d}  ·  DESIGN {item['design']} / {item['design_count']}",
        font=heading,
        fill="#f3f6f8",
    )
    draw.text((panel_x + 24, 374), f"building x   {item['x']:.0f} m", font=body, fill="#bbc5ce")
    draw.text(
        (panel_x + 24, 405),
        f"height       {item['height']:.0f} m",
        font=body,
        fill="#bbc5ce",
    )
    timing = "cache hit" if item["cache_hit"] else f"{item['elapsed_s']:.1f} s GPU"
    draw.text((panel_x + 24, 449), f"XLB draft · {timing}", font=small, fill="#68d99a")

    arrow_y = field.height - 70
    draw.text((panel_x + 24, arrow_y - 23), "WIND  +X", font=small, fill="#9ba7b3")
    draw.line(
        (panel_x + 24, arrow_y + 16, panel_x + panel_width - 38, arrow_y + 16),
        fill="#dbe4eb",
        width=4,
    )
    draw.polygon(
        (
            (panel_x + panel_width - 38, arrow_y + 7),
            (panel_x + panel_width - 20, arrow_y + 16),
            (panel_x + panel_width - 38, arrow_y + 25),
        ),
        fill="#dbe4eb",
    )
    draw.text(
        (panel_x + 24, field.height - 28),
        "Prev_Frame · GPU LBM cache",
        font=small,
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
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

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
        ]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = [compose_frame(work_dir / item["file"], item) for item in metadata]
    durations = [1200 for _item in metadata]
    durations[-1] = 1800

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
