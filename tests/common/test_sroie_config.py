"""Tests for reading the pipeline.sroie configuration block."""

from pathlib import Path

import pytest

from common.sroie.config import SroieSettings

_CONFIG_PATH = Path("/repo/config/run_config.yml")


def _raw(**overrides: object) -> dict:
    block = {
        "data_dir": "/data/sroie/test",
        "output_dir": "/data/sroie/output_internvl3",
        "max_new_tokens": 256,
        "batch_size": 8,
    }
    block.update(overrides)
    return {"pipeline": {"sroie": block}}


def test_reads_every_setting_from_yaml() -> None:
    settings = SroieSettings.from_raw(_raw(), config_path=_CONFIG_PATH)

    assert settings.data_dir == Path("/data/sroie/test")
    assert settings.output_dir == Path("/data/sroie/output_internvl3")
    assert settings.max_new_tokens == 256
    assert settings.batch_size == 8


def test_tile_budget_is_not_a_setting_here() -> None:
    """Tiling has one home: inference.tiling.budgets.receipt. A second
    tile knob in this block would read as operator intent while changing
    nothing — pre-tiling is off, and Gemma 4 cannot enable it at all."""
    assert not hasattr(SroieSettings.from_raw(_raw(), config_path=_CONFIG_PATH), "max_tiles")


@pytest.mark.parametrize("missing", ["data_dir", "output_dir", "max_new_tokens", "batch_size"])
def test_every_key_is_required(missing: str) -> None:
    """No key falls back to a Python default: reading the YAML alone must
    answer what the run is configured to do."""
    raw = _raw()
    del raw["pipeline"]["sroie"][missing]

    with pytest.raises(ValueError) as excinfo:
        SroieSettings.from_raw(raw, config_path=_CONFIG_PATH)

    message = str(excinfo.value)
    assert missing in message  # what is wrong
    assert str(_CONFIG_PATH) in message  # where to fix it
    assert "pipeline.sroie" in message  # the dotted key path
    assert "How to fix" in message  # how to recover


def test_missing_block_names_the_whole_section() -> None:
    with pytest.raises(ValueError) as excinfo:
        SroieSettings.from_raw({"pipeline": {}}, config_path=_CONFIG_PATH)

    message = str(excinfo.value)
    assert "pipeline.sroie" in message
    assert str(_CONFIG_PATH) in message


def test_non_integer_token_budget_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        SroieSettings.from_raw(_raw(max_new_tokens="lots"), config_path=_CONFIG_PATH)

    assert "max_new_tokens" in str(excinfo.value)


def test_zero_token_budget_is_rejected() -> None:
    """A zero token budget yields an empty reply for every receipt and a
    plausible-looking zero score."""
    with pytest.raises(ValueError) as excinfo:
        SroieSettings.from_raw(_raw(max_new_tokens=0), config_path=_CONFIG_PATH)

    assert "max_new_tokens" in str(excinfo.value)
