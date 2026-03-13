# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **presentation project** — a single Markdown document (`cop_presentation.md`) with embedded Mermaid diagrams, covering agentic document extraction with vision-language models (InternVL3.5-8B). There is no application code; the deliverable is the presentation content and its rendered diagram SVGs.

## Rendering Diagrams

Use the `/render-svg` slash command to render all Mermaid blocks from the markdown into portable SVGs:

```
/render-svg cop_presentation.md
```

**Pipeline**: `mmdc → SVG → thicken strokes → rsvg-convert → PDF → Inkscape → plain SVG`

Output goes to `diagrams/` as `NN-heading-slug.svg`. The SVGs are CSS-free (all styling baked into vector paths) so they import cleanly into PowerPoint/Keynote.

**Prerequisites**: `mmdc` (mermaid-cli), `rsvg-convert` (librsvg), `inkscape`

## Mermaid Authoring Constraints

The rendering pipeline disables `htmlLabels` (required for PowerPoint compatibility), which imposes strict syntax rules:

- **No HTML tags** in node labels — `<b>`, `<i>` etc. require `foreignObject` which is disabled
- **Use `<br/>` for line breaks**, never `\n` (renders as literal text)
- **Keep subgraph titles short** (1-2 words) — long titles overlap bounding rectangles
- **Use Unicode** `①②③` instead of `"1. "` in edge labels (avoids markdown list parsing)
- **Use `≈`** instead of `~` (tilde triggers strikethrough)
- Spaces may be stripped in rendered output (e.g., "Git tag push" → "Gittagpush") — this is a known cosmetic limitation of the PDF round-trip

## Mermaid Config

`mermaid-svg.json` in project root controls theme, spacing, and stroke styling. Key setting: `"htmlLabels": false` at both top level and under `flowchart` — this must never be changed to `true`.

## File Structure

- `cop_presentation.md` — the full presentation (18 Mermaid blocks across sections on data parallelism, GIL/threading, KFP deployment, FlashAttention2, and 4 agentic patterns)
- `diagrams/` — rendered SVGs, numbered by order of appearance in the markdown
- `mermaid-svg.json` — Mermaid CLI config for rendering
