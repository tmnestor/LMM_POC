"""Tests for scripts/resolve_yaml_defaults.py.

The entrypoint evals this script's stdout, so every documented YAML_* key must
be emitted in BOTH the config-present and config-missing branches (so that
`set -o nounset` reads with `${YAML_*:-}` always have a value to fall back on).
"""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "resolve_yaml_defaults.py"

# Every key the entrypoint relies on. YAML_LOG_DIR was added when the
# pre-conda regex resolver (resolve_log_dirs.py) was folded into this one.
EXPECTED_KEYS = {
    "YAML_MODEL_TYPE",
    "YAML_MODEL_PATH",
    "YAML_DATA_DIR",
    "YAML_GROUND_TRUTH",
    "YAML_OUTPUT_DIR",
    "YAML_LOG_DIR",
    "YAML_TRUST_DATA_DIR",
    "YAML_TRUST_QUADS",
    "YAML_TRUST_QUADS_INCOMPLETE",
    "YAML_TRUST_GROUND_TRUTH",
    "YAML_TRUST_CLASSIFICATION_GT",
    "YAML_TRUST_CLASSIFICATIONS",
    "YAML_TRUST_RAW_EXTRACTIONS",
    "YAML_TRUST_COMPLIANCE_RESULTS",
    "YAML_TRUST_OUTPUT_DIR",
    "YAML_TRUST_EVALUATION_DIR",
    "YAML_TRUST_LOG_DIR",
    "YAML_LINKING_DATA_DIR",
    "YAML_LINKING_OUTPUT",
    "YAML_LINKING_GROUND_TRUTH",
    "YAML_LINKING_EVALUATION_DIR",
    "YAML_LINKING_LOG_DIR",
}


def _run(arg: str) -> dict[str, str]:
    """Run the resolver and parse its `KEY=value` lines the way bash eval would."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), arg],
        capture_output=True,
        text=True,
        check=True,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition("=")
        # shlex.split reverses shlex.quote() used by the script's _emit().
        out[key] = shlex.split(value)[0] if value else ""
    return out


def test_emits_all_keys_when_config_present(tmp_path: Path) -> None:
    cfg = tmp_path / "run_config.yml"
    cfg.write_text(
        "bootstrap:\n"
        "  model:\n"
        "    type: internvl3-vllm\n"
        "    path: /models/InternVL3_5-8B\n"
        "  logging:\n"
        "    log_dir: ../out/logs\n"
        "pipeline:\n"
        "  information_extraction:\n"
        "    input:\n"
        "      dir: ../data\n"
        "      ground_truth: ../data/gt.csv\n"
        "    output:\n"
        "      dir: ../out\n"
        "  trust:\n"
        "    data_dir: ../trust_docs\n"
        "    log_dir: ../trust_docs/logs\n"
        "  linking:\n"
        "    data_dir: ../link_docs\n"
        "    log_dir: ../link_docs/logs\n"
    )
    result = _run(str(cfg))
    assert EXPECTED_KEYS.issubset(result.keys())
    assert result["YAML_MODEL_TYPE"] == "internvl3-vllm"
    assert result["YAML_LOG_DIR"] == "../out/logs"
    # Classic IO vars now come from pipeline.information_extraction.*, but keep
    # their UNPREFIXED emitted names (entrypoint.sh contract unchanged).
    assert result["YAML_DATA_DIR"] == "../data"
    assert result["YAML_GROUND_TRUTH"] == "../data/gt.csv"
    assert result["YAML_OUTPUT_DIR"] == "../out"
    assert result["YAML_TRUST_DATA_DIR"] == "../trust_docs"
    assert result["YAML_TRUST_LOG_DIR"] == "../trust_docs/logs"
    assert result["YAML_LINKING_DATA_DIR"] == "../link_docs"
    assert result["YAML_LINKING_LOG_DIR"] == "../link_docs/logs"


def test_legacy_top_level_io_is_ignored_by_resolver(tmp_path: Path) -> None:
    # The resolver no longer reads top-level io.* (retired -> moved under
    # pipeline.information_extraction). A leftover io: block yields empty IO vars
    # rather than feeding them; the Python load_yaml_config guard is what fails
    # fast on the leftover block.
    cfg = tmp_path / "run_config.yml"
    cfg.write_text(
        "bootstrap:\n  model:\n    type: internvl3-vllm\n"
        "io:\n  input:\n    dir: ../data\n  output:\n    dir: ../out\n"
    )
    result = _run(str(cfg))
    assert result["YAML_DATA_DIR"] == ""
    assert result["YAML_OUTPUT_DIR"] == ""


def test_log_dir_empty_when_logging_section_absent(tmp_path: Path) -> None:
    cfg = tmp_path / "run_config.yml"
    cfg.write_text("bootstrap:\n  model:\n    type: internvl3-vllm\n")
    result = _run(str(cfg))
    # Missing section -> empty string, never an unset/missing key.
    assert "YAML_LOG_DIR" in result
    assert result["YAML_LOG_DIR"] == ""


def test_emits_all_keys_when_config_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yml"
    result = _run(str(missing))
    # The file-missing branch must still emit every key (all empty) so the
    # entrypoint's `${YAML_*:-}` reads never trip `set -o nounset`.
    assert EXPECTED_KEYS.issubset(result.keys())
    assert all(result[k] == "" for k in EXPECTED_KEYS)


def test_usage_error_on_wrong_argc() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
