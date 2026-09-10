"""The four run paths must still describe one run.

tests/ is gitignored — local-only.

run_config.yml writes every path out in full, so any single line reads as what
it is. The cost is that three of the four can be updated and one left behind,
and two of those mismatches never announce themselves:

  * a stale `ground_truth` names an older corpus. Filenames repeat across
    generated sets, so the crossed pair matches on every row and the report is
    an ordinary-looking number rather than an error.
  * a `log_dir` outside `output.dir` fails at the first write under KFP, where
    the output directory is the only writable path in the pod -- and that write
    happens after the model has loaded.

The check refuses those. It derives nothing, so the YAML remains the only
statement of where anything is.
"""

from pathlib import Path

import pytest
import yaml

from common.pipeline_config import _validate_path_consistency

CONFIG_FILE = Path("config/run_config.yml")


def config(*, data_dir, ground_truth, output_dir, log_dir):
    """A raw config carrying only the four paths under test."""
    return {
        "bootstrap": {"logging": {"log_dir": log_dir}},
        "pipeline": {
            "information_extraction": {
                "input": {"dir": data_dir, "ground_truth": ground_truth},
                "output": {"dir": output_dir},
            }
        },
    }


CONSISTENT = config(
    data_dir="../evaluation_data/quality_20260909",
    ground_truth="../evaluation_data/quality_20260909/quality_ground_truth.jsonl",
    output_dir="../evaluation_data/quality_20260909/output",
    log_dir="../evaluation_data/quality_20260909/output/logs",
)


def test_the_shipped_config_is_consistent():
    """The one that matters: run_config.yml as committed."""
    raw = yaml.safe_load(CONFIG_FILE.read_text())

    _validate_path_consistency(raw, CONFIG_FILE)


def test_a_consistent_set_passes():
    _validate_path_consistency(CONSISTENT, CONFIG_FILE)


def test_a_stale_ground_truth_from_another_corpus_is_refused(assert_diagnostic_error):
    """The silent one. Both corpora exist, both files are real, and every
    filename matches -- so nothing downstream can tell."""
    broken = config(
        data_dir="../evaluation_data/quality_20260909",
        ground_truth="../evaluation_data/quality_20260812/quality_ground_truth.jsonl",
        output_dir="../evaluation_data/quality_20260909/output",
        log_dir="../evaluation_data/quality_20260909/output/logs",
    )

    with pytest.raises(ValueError) as exc_info:
        _validate_path_consistency(broken, CONFIG_FILE)

    message = str(exc_info.value)
    assert_diagnostic_error(message)
    assert "ground_truth" in message


def test_a_log_dir_outside_the_output_dir_is_refused(assert_diagnostic_error):
    broken = config(
        data_dir="../evaluation_data/quality_20260909",
        ground_truth="../evaluation_data/quality_20260909/quality_ground_truth.jsonl",
        output_dir="../evaluation_data/quality_20260909/output",
        log_dir="/var/log/lmm",
    )

    with pytest.raises(ValueError) as exc_info:
        _validate_path_consistency(broken, CONFIG_FILE)

    message = str(exc_info.value)
    assert_diagnostic_error(message)
    assert "log_dir" in message


def test_the_diagnostic_shows_a_corrected_path():
    """A message that only says 'wrong' makes the reader work it out. This one
    has to hand them the line to paste."""
    broken = config(
        data_dir="../evaluation_data/quality_20260909",
        ground_truth="../evaluation_data/old/quality_ground_truth.jsonl",
        output_dir="../evaluation_data/quality_20260909/output",
        log_dir="../evaluation_data/quality_20260909/output/logs",
    )

    with pytest.raises(ValueError) as exc_info:
        _validate_path_consistency(broken, CONFIG_FILE)

    assert "../evaluation_data/quality_20260909/quality_ground_truth.jsonl" in str(exc_info.value)


def test_a_deeper_nesting_is_allowed():
    """Under, not immediately under. A run that groups its logs deeper is fine
    -- the rule is about the writable volume, not about depth."""
    nested = config(
        data_dir="../evaluation_data/quality_20260909",
        ground_truth="../evaluation_data/quality_20260909/labels/quality_ground_truth.jsonl",
        output_dir="../evaluation_data/quality_20260909/output",
        log_dir="../evaluation_data/quality_20260909/output/run7/logs",
    )

    _validate_path_consistency(nested, CONFIG_FILE)


def test_a_sibling_directory_sharing_a_prefix_is_not_inside():
    """String-prefix matching would accept `.../output_old` as inside
    `.../output`. Path semantics, not text."""
    broken = config(
        data_dir="../evaluation_data/quality_20260909",
        ground_truth="../evaluation_data/quality_20260909/quality_ground_truth.jsonl",
        output_dir="../evaluation_data/quality_20260909/output",
        log_dir="../evaluation_data/quality_20260909/output_old/logs",
    )

    with pytest.raises(ValueError):
        _validate_path_consistency(broken, CONFIG_FILE)


@pytest.mark.parametrize(
    "absent",
    ["data_dir", "ground_truth", "output_dir", "log_dir"],
)
def test_an_absent_path_is_left_to_its_own_validator(absent):
    """This function has one job. A missing key is reported by the code that
    requires it, with its own message -- reporting it twice, differently, sends
    the reader to the wrong place."""
    paths = {
        "data_dir": "../evaluation_data/quality_20260909",
        "ground_truth": "../evaluation_data/quality_20260909/quality_ground_truth.jsonl",
        "output_dir": "../evaluation_data/quality_20260909/output",
        "log_dir": "../evaluation_data/quality_20260909/output/logs",
    }
    paths[absent] = None

    _validate_path_consistency(config(**paths), CONFIG_FILE)


def test_an_empty_config_does_not_raise():
    _validate_path_consistency({}, CONFIG_FILE)
