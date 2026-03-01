"""Render fenced code blocks from markdown as Dracula-themed terminal SVGs.

Extracts ```python, ```text, etc. blocks (skipping ```mermaid) and renders
them as retro terminal SVGs with Dracula syntax highlighting.

Pipeline: pygmentize → SVG → add terminal chrome → rsvg-convert → PDF → Inkscape → plain SVG

Usage:
    python3 render_code_blocks.py cop_presentation.md
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Dracula palette
BG = "#282a36"
TITLE_BAR_BG = "#44475a"
FG = "#f8f8f2"
RED = "#ff5555"
YELLOW = "#f1fa8c"
GREEN = "#50fa7b"

# Layout constants
FONT_SIZE = 14
LINE_HEIGHT = 19
CHAR_WIDTH = 8.4
PADDING_X = 20
PADDING_Y = 16
TITLE_BAR_HEIGHT = 32
DOT_RADIUS = 6
DOT_Y = TITLE_BAR_HEIGHT // 2
DOT_GAP = 20
CORNER_RADIUS = 10


def extract_code_blocks(md_path: Path) -> list[dict]:
    """Extract all non-mermaid fenced code blocks with context."""
    content = md_path.read_text()
    blocks = []
    for m in re.finditer(r"```(\w+)\n(.*?)\n```", content, re.DOTALL):
        lang = m.group(1)
        if lang == "mermaid":
            continue
        code = m.group(2)
        before = content[: m.start()]
        headings = re.findall(r"^##\s+(.+)$", before, re.MULTILINE)
        heading = headings[-1] if headings else "unknown"
        blocks.append({"lang": lang, "code": code, "heading": heading})
    return blocks


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:50]


def pygmentize_to_svg(code: str, lang: str) -> str:
    """Run pygmentize CLI to get SVG with Dracula-highlighted <tspan>s."""
    # Map markdown language hints to Pygments lexer names
    lexer_map = {"text": "text", "python": "python"}
    lexer = lexer_map.get(lang, lang)

    result = subprocess.run(
        ["pygmentize", "-f", "svg", "-l", lexer, "-O", "style=dracula"],
        input=code,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        # Fallback: render as plain text
        result = subprocess.run(
            ["pygmentize", "-f", "svg", "-l", "text", "-O", "style=dracula"],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
    return result.stdout


def parse_pygmentize_svg(svg_text: str) -> list[str]:
    """Extract the <text> elements from pygmentize SVG output."""
    # Remove the XML declaration and DOCTYPE for easier parsing
    svg_text = re.sub(r"<\?xml[^>]+\?>", "", svg_text)
    svg_text = re.sub(r"<!DOCTYPE[^>]+>", "", svg_text)

    root = ET.fromstring(svg_text)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    text_elements = []
    for g in root.findall(".//svg:g", ns):
        for text_el in g.findall("svg:text", ns):
            text_elements.append(ET.tostring(text_el, encoding="unicode"))

    # Fallback: try without namespace
    if not text_elements:
        for text_el in root.iter("text"):
            text_elements.append(ET.tostring(text_el, encoding="unicode"))

    return text_elements


def build_terminal_svg(
    text_elements: list[str], code: str, lang: str, title: str
) -> str:
    """Wrap pygmentize text elements in a terminal chrome SVG."""
    lines = code.split("\n")
    num_lines = len(lines)
    max_cols = max(len(line) for line in lines) if lines else 0

    # Calculate dimensions
    code_width = max_cols * CHAR_WIDTH + 2 * PADDING_X
    code_height = num_lines * LINE_HEIGHT + 2 * PADDING_Y
    total_width = max(code_width, 300)  # minimum width
    total_height = TITLE_BAR_HEIGHT + code_height

    # Offset for code area
    code_y_offset = TITLE_BAR_HEIGHT + PADDING_Y

    # Rewrite text element positions
    repositioned = []
    for i, te in enumerate(text_elements):
        # Update x position
        te = re.sub(r'x="[^"]*"', f'x="{PADDING_X}"', te)
        # Update y position
        new_y = code_y_offset + i * LINE_HEIGHT + FONT_SIZE
        te = re.sub(r'y="[^"]*"', f'y="{new_y}"', te)
        repositioned.append(te)

    # File extension for title
    ext_map = {"python": ".py", "text": ".txt", "yaml": ".yaml", "json": ".json"}
    ext = ext_map.get(lang, f".{lang}")

    svg = f"""<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{total_width}" height="{total_height}"
     viewBox="0 0 {total_width} {total_height}">

  <!-- Background -->
  <rect width="{total_width}" height="{total_height}"
        rx="{CORNER_RADIUS}" ry="{CORNER_RADIUS}"
        fill="{BG}" />

  <!-- Title bar -->
  <rect y="0" width="{total_width}" height="{TITLE_BAR_HEIGHT}"
        rx="{CORNER_RADIUS}" ry="{CORNER_RADIUS}"
        fill="{TITLE_BAR_BG}" />
  <!-- Square off bottom corners of title bar -->
  <rect y="{TITLE_BAR_HEIGHT - CORNER_RADIUS}"
        width="{total_width}" height="{CORNER_RADIUS}"
        fill="{TITLE_BAR_BG}" />

  <!-- Title text -->
  <text x="{PADDING_X}" y="{DOT_Y + 5}"
        font-family="monospace" font-size="{FONT_SIZE}px"
        fill="{FG}">{title}{ext}</text>

  <!-- Code -->
  <g font-family="monospace" font-size="{FONT_SIZE}px">
    {"".join(repositioned)}
  </g>
</svg>"""
    return svg


def render_to_portable_svg(raw_svg: str, output_path: Path) -> bool:
    """Run the rsvg-convert → PDF → Inkscape pipeline for PowerPoint compat."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        svg_file = tmp / "raw.svg"
        pdf_file = tmp / "tmp.pdf"

        svg_file.write_text(raw_svg)

        # rsvg-convert → PDF
        r = subprocess.run(
            ["rsvg-convert", "-f", "pdf", "-o", str(pdf_file), str(svg_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            print(f"  rsvg-convert failed: {r.stderr[:200]}")
            return False

        # Inkscape → plain SVG
        r = subprocess.run(
            [
                "inkscape",
                str(pdf_file),
                "--export-type=svg",
                "--export-plain-svg",
                f"--export-filename={output_path}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            print(f"  inkscape failed: {r.stderr[:200]}")
            return False

    return True


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 render_code_blocks.py <markdown_file>")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    if not md_path.is_absolute():
        md_path = Path.cwd() / md_path

    out_dir = md_path.parent / "diagrams"
    out_dir.mkdir(exist_ok=True)

    blocks = extract_code_blocks(md_path)
    print(f"Found {len(blocks)} code blocks")

    for i, block in enumerate(blocks, 1):
        lang = block["lang"]
        code = block["code"]
        heading = block["heading"]
        slug = slugify(heading)
        name = f"code-{i:02d}-{slug}"
        out_svg = out_dir / f"{name}.svg"

        # Get syntax-highlighted SVG from pygmentize
        pyg_svg = pygmentize_to_svg(code, lang)
        text_elements = parse_pygmentize_svg(pyg_svg)

        if not text_elements:
            print(f"  SKIP {name}: no text elements from pygmentize")
            continue

        # Build terminal chrome SVG
        title = slug.replace("-", "_")[:20]
        raw_svg = build_terminal_svg(text_elements, code, lang, title)

        # Write directly — Pygments SVG already uses inline fill attrs,
        # no CSS to bake. The rsvg→PDF→Inkscape pipeline drops comment text.
        out_svg.write_text(raw_svg)
        size = out_svg.stat().st_size
        print(f"  {name}.svg  ({size:,} bytes)  [{lang}]")


if __name__ == "__main__":
    main()
