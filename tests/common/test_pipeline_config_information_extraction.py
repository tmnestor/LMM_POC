"""Tests for the io.* -> pipeline.information_extraction.* retirement.

The classic classify->extract->clean->evaluate pipeline's paths moved out of the
special top-level `io:` block into `pipeline.information_extraction:` (symmetric
with pipeline.trust / pipeline.linking). load_yaml_config must read the new
location, and a leftover top-level `io:` must fail fast (no silent ignore).
"""

from pathlib import Path

import pytest

from common.pipeline_config import load_yaml_config

# Present sections must be COMPLETE (see test_pipeline_config_required_keys.py),
# so the shared model block lists every bootstrap.model key.
_MODEL = (
    "bootstrap:\n"
    "  model:\n"
    "    type: internvl3-vllm\n"
    "    path: /models/x\n"
    "    max_tiles: 18\n"
    "    min_tiles: null\n"
    "    flash_attn: true\n"
    "    enforce_eager: true\n"
    "    dtype: bfloat16\n"
    "    chat_template: none\n"
)


def _write(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "run_config.yml"
    cfg.write_text(_MODEL + body)
    return cfg


def test_loads_input_output_from_information_extraction(tmp_path):
    cfg = _write(
        tmp_path,
        "pipeline:\n"
        "  information_extraction:\n"
        "    input:\n"
        "      dir: ../data\n"
        "      ground_truth: ../data/gt.csv\n"
        "      max_images: 5\n"
        "      document_types: [INVOICE, RECEIPT]\n"
        "    output:\n"
        "      dir: ../out\n"
        "      skip_visualizations: true\n"
        "      skip_reports: false\n",
    )
    flat, _raw = load_yaml_config(cfg)
    assert flat["data_dir"] == "../data"
    assert flat["ground_truth"] == "../data/gt.csv"
    assert flat["max_images"] == 5
    assert flat["document_types"] == ["INVOICE", "RECEIPT"]
    assert flat["output_dir"] == "../out"
    assert flat["skip_visualizations"] is True
    assert flat["skip_reports"] is False


def test_legacy_io_block_fails_fast(tmp_path, assert_diagnostic_error):
    cfg = _write(
        tmp_path,
        "io:\n  input:\n    dir: ../data\n  output:\n    dir: ../out\n",
    )
    with pytest.raises(ValueError) as exc:
        load_yaml_config(cfg)
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert "io" in msg
    assert "pipeline.information_extraction" in msg


def test_no_io_and_no_information_extraction_is_not_an_error(tmp_path):
    # Absence of both is fine here (downstream required-field checks handle it);
    # only a LEFTOVER top-level io: is the error.
    cfg = _write(
        tmp_path,
        "pipeline:\n"
        "  processing:\n"
        "    batch_size: null\n"
        "    bank_v2: true\n"
        "    balance_correction: false\n"
        "    verbose: false\n"
        "    debug: false\n",
    )
    flat, _raw = load_yaml_config(cfg)
    assert "output_dir" not in flat  # nothing surfaced, no crash
