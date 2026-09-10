"""Tests for scripts/resolve_yaml_defaults.py.

The entrypoint evals this script's stdout, so the set of emitted keys IS the
contract between the two files. Both directions of a mismatch are silent:

  * emitted but never read -- dead weight, and the next person deleting a
    config section has to work out by hand whether anything wanted it;
  * read but never emitted -- `${YAML_FOO:-}` resolves to the empty string
    rather than failing, and the run starts against whatever the CLI default
    happens to be, on the wrong dataset, reporting plausible numbers.

So rather than restate the key list here (a third copy, free to drift from the
other two), these tests read it out of entrypoint.sh and require exact
agreement. That is what caught the sixteen YAML_TRUST_*/YAML_LINKING_* keys
left behind when those pipelines were removed.
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "resolve_yaml_defaults.py"
ENTRYPOINT = ROOT / "entrypoint.sh"


def _keys_read_by_entrypoint() -> set[str]:
    """Every YAML_* name entrypoint.sh actually dereferences."""
    text = ENTRYPOINT.read_text()
    # Only ${YAML_FOO...} expansions count. A bare mention in a comment is not
    # a read, and the prose in this file's header names several.
    return set(re.findall(r"\$\{(YAML_[A-Z0-9_]+)", text))


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


@pytest.fixture
def populated_config(tmp_path: Path) -> Path:
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
        "      ground_truth: ../data/labels.jsonl\n"
        "    output:\n"
        "      dir: ../out\n"
    )
    return cfg


def test_emitted_keys_match_what_the_entrypoint_reads(populated_config: Path) -> None:
    """The contract, in both directions."""
    emitted = set(_run(str(populated_config)))
    read = _keys_read_by_entrypoint()

    assert emitted - read == set(), "emitted but never read by entrypoint.sh"
    assert read - emitted == set(), "read by entrypoint.sh but never emitted"


def test_the_config_missing_branch_emits_the_same_keys(populated_config: Path, tmp_path: Path) -> None:
    """A missing file must not emit FEWER keys.

    The two branches are the classic drift point -- a key added to one and
    forgotten in the other only breaks on a box with no config file, which is
    the local-dev case and so the last one anybody runs.
    """
    present = set(_run(str(populated_config)))
    absent = _run(str(tmp_path / "does_not_exist.yml"))

    assert set(absent) == present
    assert all(value == "" for value in absent.values())


def test_values_come_from_the_declared_sections(populated_config: Path) -> None:
    result = _run(str(populated_config))

    assert result["YAML_MODEL_TYPE"] == "internvl3-vllm"
    assert result["YAML_MODEL_PATH"] == "/models/InternVL3_5-8B"
    assert result["YAML_LOG_DIR"] == "../out/logs"
    # The IO vars keep their UNPREFIXED emitted names while reading from
    # pipeline.information_extraction.* -- PROD run_config files are edited
    # against these names, so renaming them is a coordinated change.
    assert result["YAML_DATA_DIR"] == "../data"
    assert result["YAML_GROUND_TRUTH"] == "../data/labels.jsonl"
    assert result["YAML_OUTPUT_DIR"] == "../out"


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


def test_a_missing_section_yields_an_empty_value_not_a_missing_key(tmp_path: Path) -> None:
    cfg = tmp_path / "run_config.yml"
    cfg.write_text("bootstrap:\n  model:\n    type: internvl3-vllm\n")
    result = _run(str(cfg))

    assert "YAML_LOG_DIR" in result
    assert result["YAML_LOG_DIR"] == ""


def test_the_shipped_config_resolves_every_key(tmp_path: Path) -> None:
    """Not just well-formed -- actually populated.

    An empty YAML_DATA_DIR is a valid emission and a broken run.
    """
    result = _run(str(ROOT / "config" / "run_config.yml"))

    empty = sorted(key for key, value in result.items() if not value)
    assert not empty, f"config/run_config.yml leaves these unresolved: {empty}"


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
