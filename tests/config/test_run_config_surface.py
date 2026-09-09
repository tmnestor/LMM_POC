# tests/config/test_run_config_surface.py
"""Characterization test: pin every value the config loaders surface.

Golden-file pattern: first run writes the golden JSON and skips; later runs
compare. The golden is regenerated intentionally ONLY in the prune task; every
move task must keep it byte-identical.
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
# Likewise for the DATASET. Evaluation runs repoint
# pipeline.information_extraction.input at whichever corpus is being measured
# (e.g. the sandbox-only /home/jovyan/nfs_share/... sets), which both breaks
# AppConfig.load's path-exists check locally and rewrites two golden values.
# Both are operator selections, not config structure — normalise them for the
# same reason the model is normalised.
_CANONICAL_DATA_DIR = "../evaluation_data/synthetic_transaction_linking"
_CANONICAL_GROUND_TRUTH = "../evaluation_data/synthetic_transaction_linking/ground_truth_extraction.csv"
# Output and log dirs move with the dataset, so normalise them for the same
# reason. Unlike the input dir these are never existence-checked, so they break
# only the golden comparison, not AppConfig.load.
_CANONICAL_OUTPUT_DIR = "../evaluation_data/output"
_CANONICAL_LOG_DIR = "../evaluation_data/output/logs"


def _local_config(tmp_path: Path) -> Path:
    """Copy run_config.yml, normalised to the CANONICAL baseline model.

    Two normalisations, both of operator *selections* rather than config
    structure:

    * the model path is repointed at a real local dir so ``AppConfig.load``'s
      path-exists check passes;
    * ``bootstrap.model.type`` / ``path`` and ``pre_tiling.enabled`` are reset to
      the InternVL3.5-8B baseline;
    * ``information_extraction.input.dir`` / ``.ground_truth`` are reset to the
      canonical local corpus — evaluation runs point these at sandbox-only
      absolute paths, which do not exist on a dev machine.

    The second one matters: this is a characterization test of the config's
    *structure and derived values*, not of whichever model the working checkout
    happens to point at. Evaluation runs switch ``bootstrap.model.type`` between
    InternVL and the two Gemma 4 models (and must flip ``pre_tiling`` with them),
    so without normalising, the golden would break on every legitimate switch and
    the test would get ignored. Normalising keeps the golden byte-identical while
    still pinning everything a config refactor could break.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    raw = yaml.safe_load(CONFIG.read_text())
    raw["bootstrap"]["model"]["type"] = _CANONICAL_MODEL
    raw["bootstrap"]["model"]["path"] = str(model_dir)
    raw["inference"]["tiling"]["pre_tiling"]["enabled"] = True
    raw["pipeline"]["information_extraction"]["input"]["dir"] = _CANONICAL_DATA_DIR
    raw["pipeline"]["information_extraction"]["input"]["ground_truth"] = _CANONICAL_GROUND_TRUTH
    raw["pipeline"]["information_extraction"]["output"]["dir"] = _CANONICAL_OUTPUT_DIR
    raw["bootstrap"]["logging"]["log_dir"] = _CANONICAL_LOG_DIR
    cfg = tmp_path / "run_config.yml"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    return cfg


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
    surface["extraction_order"] = list(cfg.extraction_order)
    surface["secondary_sort"] = cfg.secondary_sort
    surface["extraction_skip_labels"] = list(cfg.extraction_skip_labels)
    surface["vllm_config"] = cfg.get_vllm_config(model)
    surface["generation_config"] = cfg.get_generation_config(model)
    surface["max_image_budget_tiles"] = cfg.max_image_budget_tiles()
    for dt in ("default", "bank_statement", "invoice", "receipt"):
        surface[f"image_budget.{dt}"] = cfg.get_image_budget(dt)
    for name in (
        "classify",
        "detection",
        "extract_bank",
        "transaction_match",
        "fallback_base",
        "fallback_per_field",
        "bank_statement_floor",
        "trust_classify",
    ):
        surface[f"token_budget.{name}"] = cfg.get_token_budget(name)
    surface["bank_header_cache_config"] = cfg.bank_header_cache_config
    surface["batch.default_internvl3"] = cfg.get_batch_size_for_model(model)

    # --- resolve_yaml_defaults.py (entrypoint resolver) ---
    # Exclude the two model-path vars (one is tmp-patched, one is the real path).
    rd = _resolve_yaml_defaults(cfg_path)
    rd.pop("YAML_MODEL_PATH", None)
    surface["resolve_defaults"] = rd

    # --- per-stage validators / direct readers ---
    from stages.transaction_link import _load_linking_config

    surface["linking_config"] = _load_linking_config(cfg_path)

    raw = yaml.safe_load(CONFIG.read_text())  # raw of the REAL file (paths stable)
    pipeline = raw.get("pipeline", {})
    surface["trust.subdirectories"] = pipeline.get("trust", {}).get("subdirectories")
    surface["trust_classify_budget"] = pipeline.get("token_budgets", {}).get("trust_classify")

    return surface


def test_run_config_surface_unchanged(tmp_path):
    surface = _surface(tmp_path)
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(surface, indent=2, sort_keys=True, default=str))
        pytest.skip("golden written — re-run to compare")
    expected = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(surface, sort_keys=True, default=str))
    assert got == expected
