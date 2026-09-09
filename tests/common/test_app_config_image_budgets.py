"""Tests for image_budgets validation (min_tiles / max_tiles).

tests/ is gitignored — local-only. Locks the fail-fast contract for the new
``min_tiles`` floor added for dense-bank adaptive tiling
(plans/2026-06-04-adaptive-tiling-dense-bank.md). Every validation error must
carry all four diagnostic elements (What / Where / Expected / How to fix).
"""

import pytest

from common.app_config import AppConfig, ConfigError

_CFG = "/tmp/run_config.yml"


def _budgets(**doc_types: dict) -> dict:
    return {
        "inference": {"tiling": {"budgets": {"default": {"min_tiles": 1, "max_tiles": 18}, **doc_types}}}
    }


def test_valid_budget_round_trips() -> None:
    raw = _budgets(bank_statement={"min_tiles": 12, "max_tiles": 18})
    result = AppConfig._validate_image_budgets(raw, _CFG)
    assert result["bank_statement"] == {"min_tiles": 12, "max_tiles": 18}


def test_missing_min_tiles_is_diagnostic(assert_diagnostic_error) -> None:
    raw = _budgets(bank_statement={"max_tiles": 18})  # no min_tiles
    with pytest.raises(ConfigError) as exc:
        AppConfig._validate_image_budgets(raw, _CFG)
    assert_diagnostic_error(str(exc.value))


def test_min_greater_than_max_is_diagnostic(assert_diagnostic_error) -> None:
    raw = _budgets(bank_statement={"min_tiles": 20, "max_tiles": 18})
    with pytest.raises(ConfigError) as exc:
        AppConfig._validate_image_budgets(raw, _CFG)
    msg = str(exc.value)
    assert_diagnostic_error(msg)
    assert "min_tiles" in msg and "max_tiles" in msg


def test_non_positive_min_tiles_is_diagnostic(assert_diagnostic_error) -> None:
    raw = _budgets(bank_statement={"min_tiles": 0, "max_tiles": 18})
    with pytest.raises(ConfigError) as exc:
        AppConfig._validate_image_budgets(raw, _CFG)
    assert_diagnostic_error(str(exc.value))
