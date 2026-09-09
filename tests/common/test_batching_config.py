"""Tests for reading the pipeline.batching block."""

from pathlib import Path

import pytest

from common.extraction_batching import read_batch_sizes

_CONFIG = Path("/repo/config/run_config.yml")


def _raw(**overrides) -> dict:
    block = {"receipt": 4, "invoice": 2, "bank_statement": 2, "default": 2}
    block.update(overrides)
    return {"pipeline": {"batching": block}}


def test_reads_every_size() -> None:
    sizes = read_batch_sizes(_raw(), config_path=_CONFIG)

    assert sizes["receipt"] == 4
    assert sizes["default"] == 2


def test_a_missing_block_is_a_diagnostic_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        read_batch_sizes({"pipeline": {}}, config_path=_CONFIG)

    message = str(excinfo.value)
    assert "pipeline.batching" in message
    assert str(_CONFIG) in message
    assert "How to fix" in message


def test_a_missing_default_is_an_error() -> None:
    """Every unlisted document type falls back to default, so its absence
    would surface as a KeyError deep in the extraction loop."""
    raw = _raw()
    del raw["pipeline"]["batching"]["default"]

    with pytest.raises(ValueError) as excinfo:
        read_batch_sizes(raw, config_path=_CONFIG)

    assert "default" in str(excinfo.value)


def test_a_zero_size_is_rejected() -> None:
    """Zero would silently produce no batches and extract nothing."""
    with pytest.raises(ValueError) as excinfo:
        read_batch_sizes(_raw(receipt=0), config_path=_CONFIG)

    assert "receipt" in str(excinfo.value)


def test_a_non_integer_size_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        read_batch_sizes(_raw(receipt="four"), config_path=_CONFIG)

    assert "receipt" in str(excinfo.value)
