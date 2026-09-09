"""The quality-screen config section is validated at startup, not on first use.

Every key here is required with no default. Each one silently defaulted would
produce a run that completes and measures the wrong thing -- the wrong prompt
variant, records written where the scorer will not look, or a condition mapping
that drops a whole severity from the report without saying so.

These test the validator directly rather than through `AppConfig.load`, because
a full load also validates the model path, which points at the GPU box and does
not exist on a development machine.
"""

import copy
from pathlib import Path

import pytest
import yaml

from common.app_config import AppConfig, ConfigError

CONFIG_FILE = Path("config/run_config.yml")
REQUIRED = ["prompt_file", "variant", "output_name", "condition_to_level"]


@pytest.fixture
def raw_config():
    return yaml.safe_load(CONFIG_FILE.read_text())


def test_the_shipped_config_declares_every_required_key(raw_config):
    screen = AppConfig._validate_quality_screen(raw_config, str(CONFIG_FILE))

    assert set(REQUIRED) <= set(screen)
    assert screen["variant"], "a run must name the prompt variant it used"
    assert Path(screen["prompt_file"]).exists(), "prompt_file must point at a real file"


def test_the_declared_variant_exists_in_the_declared_prompt_file(raw_config):
    """Guards the two halves of the config drifting apart: a variant name that
    no longer exists would fail at inference time, after the model has loaded."""
    from common.quality_screen_parser import load_screen_vocabulary

    screen = AppConfig._validate_quality_screen(raw_config, str(CONFIG_FILE))
    vocabulary = load_screen_vocabulary(Path(screen["prompt_file"]), variant=screen["variant"])

    assert vocabulary.criteria, "the declared variant must carry criteria"


def test_condition_mapping_covers_the_generated_corpus(raw_config):
    """The corpus and the prompt name one ladder differently. A condition with
    no level would be dropped from the report rather than scored."""
    corpus = Path("../evaluation_data/quality_20260909/quality_ground_truth.jsonl")
    if not corpus.exists():
        pytest.skip(f"corpus not generated at {corpus}")

    import json

    conditions = {json.loads(line)["condition"] for line in corpus.read_text().splitlines() if line.strip()}
    screen = AppConfig._validate_quality_screen(raw_config, str(CONFIG_FILE))

    unmapped = conditions - set(screen["condition_to_level"])
    assert not unmapped, f"corpus conditions with no declared level: {sorted(unmapped)}"


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_missing_key_fails_with_a_four_element_diagnostic(assert_diagnostic_error, raw_config, missing):
    broken = copy.deepcopy(raw_config)
    del broken["pipeline"]["quality_screen"][missing]

    with pytest.raises(ConfigError) as exc_info:
        AppConfig._validate_quality_screen(broken, str(CONFIG_FILE))

    message = str(exc_info.value)
    assert_diagnostic_error(message)
    assert missing in message


def test_a_missing_section_fails_with_a_four_element_diagnostic(assert_diagnostic_error, raw_config):
    broken = copy.deepcopy(raw_config)
    del broken["pipeline"]["quality_screen"]

    with pytest.raises(ConfigError) as exc_info:
        AppConfig._validate_quality_screen(broken, str(CONFIG_FILE))

    assert_diagnostic_error(str(exc_info.value))


def test_an_empty_condition_mapping_is_rejected(assert_diagnostic_error, raw_config):
    """Empty is not the same as absent, and would otherwise score nothing while
    reporting success."""
    broken = copy.deepcopy(raw_config)
    broken["pipeline"]["quality_screen"]["condition_to_level"] = {}

    with pytest.raises(ConfigError) as exc_info:
        AppConfig._validate_quality_screen(broken, str(CONFIG_FILE))

    assert_diagnostic_error(str(exc_info.value))


def test_the_token_budget_for_the_screen_is_declared(raw_config):
    """A truncated final slot reads as a missing answer and costs the whole
    record, so the budget is not something to leave to a default."""
    budgets = raw_config["pipeline"]["token_budgets"]

    assert "quality_screen" in budgets
    assert budgets["quality_screen"] > 0
