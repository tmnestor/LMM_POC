"""Extract Mermaid diagrams from a Markdown file and export as SVGs.

Uses mermaid-py (mermaid.ink API) — no local Node.js or Puppeteer required.

Usage:
    python docs/export_diagrams.py [markdown_file] [output_dir]

Defaults:
    markdown_file = docs/internvl3_extraction_pipeline.md
    output_dir    = docs/diagrams/
"""

import re
import sys
from pathlib import Path

from mermaid import Mermaid
from rich.console import Console

console = Console()

MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


def extract_mermaid_blocks(md_path: Path) -> list[tuple[str, str]]:
    """Return list of (slug, mermaid_source) from fenced mermaid blocks.

    The slug is derived from the first meaningful line of the diagram
    (e.g. 'flowchart_LR' or 'sequenceDiagram').
    """
    text = md_path.read_text()
    blocks: list[tuple[str, str]] = []

    for i, match in enumerate(MERMAID_BLOCK.finditer(text), start=1):
        source = match.group(1).strip()
        first_line = source.split("\n")[0].strip()
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", first_line).strip("_").lower()
        name = f"{i:02d}_{slug}" if slug else f"{i:02d}_diagram"
        blocks.append((name, source))

    return blocks


def export_diagrams(md_path: Path, out_dir: Path) -> None:
    """Extract and export all Mermaid diagrams to SVG."""
    blocks = extract_mermaid_blocks(md_path)
    if not blocks:
        console.print("[yellow]No mermaid blocks found.[/yellow]")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Found [bold]{len(blocks)}[/bold] diagrams in {md_path.name}\n")

    for name, source in blocks:
        svg_path = out_dir / f"{name}.svg"
        try:
            diagram = Mermaid(source)
            diagram.to_svg(svg_path)
            console.print(f"  [green]OK[/green]  {svg_path.name}")
        except Exception as e:
            console.print(f"  [red]FAIL[/red]  {name}: {e}")

    console.print(f"\nExported to {out_dir}/")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    md_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else project_root / "docs" / "internvl3_extraction_pipeline.md"
    )
    output_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else project_root / "docs" / "diagrams"
    )

    export_diagrams(md_file, output_dir)
