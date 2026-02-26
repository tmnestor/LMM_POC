"""Export Mermaid diagrams from threads_processes_and_thread_safety.md to high-res PNGs.

Uses mmdc (Mermaid CLI) for rendering, then ImageMagick for 3840x2160 landscape framing.

Usage:
    python docs/export_thread_slides.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

console = Console()

SLIDE_W = 3840
SLIDE_H = 2160
PADDING = 200

SRC_DIR = Path(__file__).parent
MD_FILE = SRC_DIR / "threads_processes_and_thread_safety.md"
OUT_DIR = SRC_DIR / "png_slides"
CONFIG = Path("/Users/tod/Desktop/FlashAttention2_Diagrams/mermaid-slides.json")

MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)

SLIDE_NAMES = [
    "01_memory_model_comparison",
    "02_gil_switching_sequence",
    "03_gil_ownership_4gpu_gantt",
    "04_three_thread_hazards",
    "05_cuda_streams_parallel",
    "06_safe_vs_unsafe_architecture",
    "07_lazy_module_race_sequence",
    "08_data_parallel_architecture",
    "09_two_phase_execution",
    "10_bank_statement_multiturn",
    "11_gil_vs_nogil_comparison",
    "12_decision_framework",
]


def extract_blocks(md_path: Path) -> list[tuple[str, str]]:
    """Extract mermaid blocks from markdown with named slugs."""
    text = md_path.read_text()
    blocks = []
    for i, match in enumerate(MERMAID_BLOCK.finditer(text)):
        source = match.group(1).strip()
        name = SLIDE_NAMES[i] if i < len(SLIDE_NAMES) else f"{i + 1:02d}_diagram"
        blocks.append((name, source))
    return blocks


def render_to_png(
    name: str, mmd_source: str, out_dir: Path, config: Path | None
) -> Path | None:
    """Render a single mermaid diagram to high-res PNG via mmdc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_png = out_dir / f"{name}_raw.png"

    with tempfile.NamedTemporaryFile(suffix=".mmd", mode="w", delete=False) as f:
        f.write(mmd_source)
        mmd_path = Path(f.name)

    try:
        cmd = [
            "mmdc",
            "-i",
            str(mmd_path),
            "-o",
            str(raw_png),
            "-s",
            "4",
            "-b",
            "white",
            "-w",
            "2400",
        ]
        if config and config.exists():
            cmd.extend(["-c", str(config)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            console.print(f"  [red]mmdc error[/red]: {result.stderr.strip()}")
            return None
    finally:
        mmd_path.unlink(missing_ok=True)

    if not raw_png.exists():
        console.print(f"  [red]No output[/red] for {name}")
        return None

    return raw_png


def frame_landscape(raw_png: Path, final_png: Path) -> bool:
    """Place rendered diagram onto a 3840x2160 landscape canvas using ImageMagick."""
    max_w = SLIDE_W - (PADDING * 2)
    max_h = SLIDE_H - (PADDING * 2)

    cmd = [
        "magick",
        str(raw_png),
        "-resize",
        f"{max_w}x{max_h}",
        "-gravity",
        "center",
        "-background",
        "white",
        "-extent",
        f"{SLIDE_W}x{SLIDE_H}",
        str(final_png),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    raw_png.unlink(missing_ok=True)

    if result.returncode != 0:
        console.print(f"  [red]magick error[/red]: {result.stderr.strip()}")
        return False
    return True


def main() -> None:
    blocks = extract_blocks(MD_FILE)
    console.print(f"Found [bold]{len(blocks)}[/bold] diagrams in {MD_FILE.name}\n")

    config = CONFIG if CONFIG.exists() else None
    if config:
        console.print(f"Using config: {config}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    for name, source in blocks:
        console.print(f"  Rendering [cyan]{name}[/cyan]...")
        raw = render_to_png(name, source, OUT_DIR, config)
        if raw is None:
            continue

        final = OUT_DIR / f"{name}.png"
        if frame_landscape(raw, final):
            size = final.stat().st_size / 1024
            dims = subprocess.run(
                ["magick", "identify", "-format", "%wx%h", str(final)],
                capture_output=True,
                text=True,
            ).stdout.strip()
            console.print(f"    [green]OK[/green]  {dims}  ({size:.0f} KB)")
            success += 1

    console.print(
        f"\n[bold green]{success}/{len(blocks)}[/bold green] slides exported to {OUT_DIR}/"
    )


if __name__ == "__main__":
    main()
