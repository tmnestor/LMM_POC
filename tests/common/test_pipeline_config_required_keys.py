"""Fail-fast tests: a PRESENT config section must be COMPLETE.

load_yaml_config used to read every section with .get() and strip Nones,
silently funneling missing keys into PipelineConfig dataclass defaults
(e.g. a typo'd bootstrap.model.max_tiles ran with 11 instead of YAML's 18).
Wholly-absent sections stay legal (CLI-driven modes supply paths via cli_args);
a present section with a missing key is now a diagnostic error. Explicit null
remains the documented way to keep a default visible (batch_size: null = auto).
"""

from pathlib import Path

import pytest

from common.pipeline_config import load_yaml_config

_FULL_MODEL = (
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

_FULL_PROCESSING = "pipeline:\n  processing:\n    verbose: false\n    debug: false\n"


def _write(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "run_config.yml"
    cfg.write_text(body)
    return cfg


@pytest.mark.parametrize(
    "omit", ["type", "path", "max_tiles", "min_tiles", "flash_attn", "enforce_eager", "dtype"]
)
def test_missing_model_key_is_diagnostic(tmp_path, assert_diagnostic_error, omit):
    body = "\n".join(ln for ln in _FULL_MODEL.splitlines() if not ln.strip().startswith(f"{omit}:"))
    with pytest.raises(ValueError) as exc:
        load_yaml_config(_write(tmp_path, body + "\n"))
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert omit in msg


def test_missing_processing_key_is_diagnostic(tmp_path, assert_diagnostic_error):
    body = _FULL_MODEL + "\n".join(ln for ln in _FULL_PROCESSING.splitlines() if "debug" not in ln)
    with pytest.raises(ValueError) as exc:
        load_yaml_config(_write(tmp_path, body + "\n"))
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert "debug" in msg


def test_missing_input_key_is_diagnostic(tmp_path, assert_diagnostic_error):
    body = _FULL_MODEL + (
        "pipeline:\n"
        "  information_extraction:\n"
        "    input:\n"
        "      dir: ../data\n"
        "      ground_truth: null\n"
        "      document_types: null\n"  # max_images omitted
    )
    with pytest.raises(ValueError) as exc:
        load_yaml_config(_write(tmp_path, body))
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert "max_images" in msg


def test_missing_gpus_key_is_diagnostic(tmp_path, assert_diagnostic_error):
    body = _FULL_MODEL + "  gpus:\n    num_gpus: 0\n"  # data_parallel_size omitted
    with pytest.raises(ValueError) as exc:
        load_yaml_config(_write(tmp_path, body))
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert "data_parallel_size" in msg


def test_missing_max_new_tokens_is_diagnostic(tmp_path, assert_diagnostic_error):
    body = _FULL_MODEL + "inference:\n  some_other_key: 1\n"
    with pytest.raises(ValueError) as exc:
        load_yaml_config(_write(tmp_path, body))
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert "max_new_tokens" in msg


def test_complete_sections_with_explicit_nulls_load(tmp_path):
    body = _FULL_MODEL + _FULL_PROCESSING
    flat, _raw = load_yaml_config(_write(tmp_path, body))
    assert flat["max_tiles"] == 18
    assert flat["verbose"] is False
    # Explicit nulls are filtered (null = documented default behavior).
    assert "min_tiles" not in flat


def test_wholly_absent_sections_stay_legal(tmp_path):
    flat, _raw = load_yaml_config(_write(tmp_path, _FULL_MODEL))
    assert flat["model_type"] == "internvl3-vllm"
    assert "verbose" not in flat


def test_repo_run_config_passes(tmp_path):
    repo_cfg = Path(__file__).parents[2] / "config" / "run_config.yml"
    flat, _raw = load_yaml_config(repo_cfg)
    assert flat["max_tiles"] == 18  # the value the dataclass default used to shadow
