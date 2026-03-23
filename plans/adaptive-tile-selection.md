# Adaptive Per-Image Tile Selection Based on Image Quality

## Context

InternVL3 splits document images into 448x448 tiles before inference. Currently `max_tiles` is a fixed value for all images. Testing shows accuracy peaks at ~18 tiles then declines due to semantic fragmentation. However, poor quality images (blurry, low-res) benefit more from higher tile counts (magnification effect), while sharp images do fine with fewer. We need adaptive per-image tile selection based on lightweight quality assessment.

## Activation Mechanism

`min_tiles` parameter controls adaptive mode:
- `min_tiles = None` (default) → adaptive OFF, uses fixed `max_tiles` (current behavior, fully backward compatible)
- `min_tiles = N` where N < max_tiles → adaptive ON, quality assessment selects tiles in `[N, max_tiles]`

No new boolean flags needed — presence of `min_tiles` is the signal.

## Quality Metrics (PIL + numpy only — no OpenCV)

Resolution (`width * height`) is intentionally excluded — a full-page scan has the same pixel count regardless of text legibility. The two metrics that matter for document readability:

| Metric | Method | Reference Value | Weight |
|--------|--------|-----------------|--------|
| Sharpness | Laplacian variance via PIL `ImageFilter.Kernel` + numpy | 500.0 | 0.7 |
| Contrast | `ImageStat.Stat(gray).stddev[0]` | 80.0 | 0.3 |

Each metric normalized to [0, 1], combined into composite score.
Mapping: `tiles = max_tiles - round(composite * (max_tiles - min_tiles))`
Low quality (score ~0) → max_tiles; High quality (score ~1) → min_tiles.

Overhead: ~2ms per image (negligible vs 2-10s inference).

## Changes

### 1. `models/internvl3_image_preprocessor.py` — core change

- Add `ImageQualityMetrics` dataclass (sharpness_score, contrast_score, composite_score, recommended_tiles, width, height)
- Add `min_tiles: int | None = None` to `__init__`, set `self.adaptive_enabled = min_tiles is not None and min_tiles < max_tiles`
- Add `assess_image_quality(image: Image.Image) -> ImageQualityMetrics` method
- Add `_resolve_tile_count(image: Image.Image, max_num: int | None) -> int` helper:
  - Explicit `max_num` override → return it (notebooks, direct callers)
  - Adaptive OFF → return `self.max_tiles`
  - Adaptive ON → assess quality, return recommended tiles
- Modify `load_image`: open image first, then call `_resolve_tile_count(image, max_num)` before `dynamic_preprocess`
- Modify `load_image_from_pil`: same pattern with `_resolve_tile_count`
- Update per-image print to show adaptive info: `filename: N tiles (quality: 0.58)`

### 2. `common/pipeline_config.py`

- Add field: `min_tiles: int | None = None` after `max_tiles` (line 93)
- Add YAML loading: `flat_config["min_tiles"] = raw_config["model"].get("min_tiles")` (after line 168)
- Add env mapping: `f"{ENV_PREFIX}MIN_TILES": ("min_tiles", int)` (after line 226)

### 3. `cli.py`

- Add CLI option after `max_tiles` (line 752):
  ```python
  min_tiles: int | None = typer.Option(
      None, "--min-tiles",
      help="Min tiles for adaptive quality-based tiling. Enables adaptive mode.",
  ),
  ```
- Add to `arg_mapping` (line 825): `"min_tiles": min_tiles`
- Update config table (after line 451): show `Tiles` as `6..18 (adaptive)` or `18 (fixed)`

### 4. `models/document_aware_internvl3_processor.py`

- Add `min_tiles: int | None = None` param to `__init__` (after `max_tiles`, line 56)
- Pass to preprocessor (line 79): `InternVL3ImagePreprocessor(max_tiles=max_tiles, debug=debug, min_tiles=min_tiles)`

### 5. `models/registry.py`

- In `_internvl3_processor_creator` (line 300): add `min_tiles=config.min_tiles`

### 6. `config/run_config.yml`

- Add commented-out option after `max_tiles` (line 12):
  ```yaml
  # min_tiles: 6         # Set to enable adaptive quality-based tiling (range: min_tiles..max_tiles)
  ```

## Usage

```bash
# Fixed tiles (current behavior, unchanged)
python cli.py --max-tiles 18

# Adaptive tiles: quality assessment selects between 6 and 18
python cli.py --max-tiles 18 --min-tiles 6
```

## Verification (on L40 sandbox)

1. Without `--min-tiles`: confirm output is identical to current behavior
2. With `--min-tiles 6 --max-tiles 18`: confirm per-image tile counts vary based on image quality
3. Check that blurry/low-res images get more tiles, sharp images get fewer
4. Compare F1 scores: adaptive vs fixed at 18 tiles
