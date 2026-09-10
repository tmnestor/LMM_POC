# tests/config/test_run_config_surface.py
"""Characterization test: pin every value the config loaders surface.

Golden-file pattern: first run writes the golden JSON and skips; later runs
compare. A config refactor that moves a key without changing what the code
reads keeps this byte-identical; one that quietly drops a value breaks it.

The golden is regenerated intentionally ONLY when the surface itself is meant
to change -- as in the strip to the standalone screen, which removed the trust,
linking, extraction-ordering and bank-header-cache sections along with their
accessors. Regenerating it to make a red test go green defeats the point.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from common.app_config import AppConfig

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "run_config.yml"
GOLDEN = Path(__file__).parent / "run_config_surface_golden.json"
# The baseline the golden is measured against, regardless of which model
# bootstrap.model.type currently selects. See _local_config().
_CANONICAL_MODEL = "internvl3-vllm"


def _local_config(tmp_path: Path) -> Path:
    """Copy run_config.yml, normalised to the CANONICAL baseline.

    Two normalisations, both of operator *selections* rather than config
    structure:

    * ``bootstrap.model.type`` is reset to the InternVL3.5-8B baseline, and its
      path to a real local dir so ``AppConfig.load``'s path-exists check passes;
    * the input/output/log paths are repointed under ``tmp_path``. The real
      ones name a generated corpus that need not exist on a dev machine, and
      ``AppConfig.load`` checks the input dir exists -- so this test would pass
      or fail on whether the last corpus generation happened to be lying around.

    ``_scrub`` puts the tmp paths back to fixed placeholders afterwards, so the
    golden stays byte-identical across runs and machines while still pinning
    everything a config refactor could break. A golden that breaks on every
    dataset switch is a golden nobody reads.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()

    raw = yaml.safe_load(CONFIG.read_text())
    raw["bootstrap"]["model"]["type"] = _CANONICAL_MODEL
    raw["bootstrap"]["model"]["path"] = str(model_dir)
    raw["inference"]["tiling"]["pre_tiling"]["enabled"] = True
    raw["pipeline"]["information_extraction"]["input"]["dir"] = str(data_dir)
    raw["pipeline"]["information_extraction"]["input"]["ground_truth"] = str(
        data_dir / "quality_ground_truth.jsonl"
    )
    raw["pipeline"]["information_extraction"]["output"]["dir"] = str(data_dir / "output")
    raw["bootstrap"]["logging"]["log_dir"] = str(data_dir / "output" / "logs")
    cfg = tmp_path / "run_config.yml"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    return cfg


def _scrub(surface: dict, tmp_path: Path) -> dict:
    """Replace the per-run tmp paths with fixed placeholders."""
    text = json.dumps(surface, sort_keys=True, default=str)
    text = text.replace(str(tmp_path / "corpus"), "<CORPUS>")
    text = text.replace(str(tmp_path / "model"), "<MODEL>")
    text = text.replace(str(tmp_path), "<TMP>")
    return json.loads(text)


def _resolve_yaml_defaults(cfg: Path) -> dict:
    """Capture scripts/resolve_yaml_defaults.py stdout as a dict."""
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "resolve_yaml_defaults.py"), str(cfg)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    d = {}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        d[k] = v.strip().strip("'\"")
    return d


def _surface(tmp_path: Path) -> dict:
    """Every value the config system surfaces, as a JSON-able dict."""
    cfg_path = _local_config(tmp_path)
    cfg = AppConfig.load({}, config_path=cfg_path)
    pc = cfg.pipeline
    model = "internvl3-vllm"

    surface: dict = {}

    # --- PipelineConfig flat fields (from pipeline_config.py) ---
    for field in (
        "model_type",
        "max_tiles",
        "min_tiles",
        "flash_attn",
        "enforce_eager",
        "dtype",
        "max_new_tokens",
        "max_images",
        "document_types",
        "batch_size",
        "bank_v2",
        "balance_correction",
        "verbose",
        "debug",
        "num_gpus",
        "data_parallel_size",
        "device_map",
        "trace_raw_prompts",
        "pre_tiling_enabled",
        "pre_tiling_image_size",
        "pre_tiling_use_thumbnail",
    ):
        surface[f"pc.{field}"] = getattr(pc, field, "<<MISSING>>")

    # --- AppConfig accessors ---
    surface["classification_fallback_type"] = cfg.classification_fallback_type
    surface["vllm_config"] = cfg.get_vllm_config(model)
    surface["generation_config"] = cfg.get_generation_config(model)
    surface["max_image_budget_tiles"] = cfg.max_image_budget_tiles()
    for dt in ("default", "bank_statement", "invoice", "receipt"):
        surface[f"image_budget.{dt}"] = cfg.get_image_budget(dt)
    for name in (
        "classify",
        "quality_screen",
        "detection",
        "fallback_base",
        "fallback_per_field",
        "bank_statement_floor",
    ):
        surface[f"token_budget.{name}"] = cfg.get_token_budget(name)
    surface["batch.default_internvl3"] = cfg.get_batch_size_for_model(model)

    # The screen's own config: the one section on this branch that a run
    # actually depends on end to end.
    surface["quality_screen_config"] = cfg.quality_screen_config

    # --- resolve_yaml_defaults.py (entrypoint resolver) ---
    surface["resolve_defaults"] = _resolve_yaml_defaults(cfg_path)

    return surface


def test_run_config_surface_unchanged(tmp_path):
    got = _scrub(_surface(tmp_path), tmp_path)
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True))
        pytest.skip("golden written — re-run to compare")
    assert got == json.loads(GOLDEN.read_text())


def test_the_stripped_sections_have_no_accessors_left(tmp_path):
    """The removed config sections must be gone from BOTH sides.

    A property surviving its YAML section is the silent-fallback shape: it
    returns the constructor's empty default and reads as configured.
    """
    cfg = AppConfig.load({}, config_path=_local_config(tmp_path))

    for name in (
        "extraction_order",
        "secondary_sort",
        "extraction_skip_labels",
        "bank_header_cache_config",
    ):
        assert not hasattr(cfg, name), f"{name} outlived its config section"

    raw = yaml.safe_load(CONFIG.read_text())
    for section in ("trust", "linking", "extraction", "bank_header_cache"):
        assert section not in raw["pipeline"], f"pipeline.{section} is back"
