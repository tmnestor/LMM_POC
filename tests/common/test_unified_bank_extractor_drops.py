"""Drop-visibility tests for UnifiedBankExtractor (no GPU; stubbed generate_fn).

The schema fallback and the per-strategy array assembly silently discarded
rows (truncate-to-shortest, zero-all-on-empty, missing-field filtering).
These tests lock the new WARNING-level counters that make that loss visible.
"""

import logging

from common.bank_types import ColumnMapping
from common.unified_bank_extractor import UnifiedBankExtractor, _warn_dropped_rows


def _ube(response):
    return UnifiedBankExtractor(generate_fn=lambda *a, **k: response, verbose=False)


def test_warn_dropped_rows_helper(caplog):
    with caplog.at_level(logging.WARNING, logger="common.unified_bank_extractor"):
        _warn_dropped_rows("balance_description", kept=3, total=5)
    assert any("dropped 2/5" in r.message for r in caplog.records)


def test_warn_dropped_rows_silent_when_nothing_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger="common.unified_bank_extractor"):
        _warn_dropped_rows("balance_description", kept=5, total=5)
    assert not caplog.records


def test_schema_fallback_warns_on_array_length_mismatch(caplog):
    response = (
        "STATEMENT_DATE_RANGE: NOT_FOUND\n"
        "TRANSACTION_DATES: 01/01/2024 | 02/01/2024\n"
        "LINE_ITEM_DESCRIPTIONS: STORE A | STORE B\n"
        "TRANSACTION_AMOUNTS_PAID: $5.00\n"
    )
    with caplog.at_level(logging.WARNING, logger="common.unified_bank_extractor"):
        result = _ube(response)._extract_schema_fallback("img.png", [], ColumnMapping())
    assert len(result.transaction_dates) == 1
    assert any("mismatch" in r.message for r in caplog.records)


def test_schema_fallback_warns_when_all_rows_zeroed(caplog):
    response = (
        "STATEMENT_DATE_RANGE: NOT_FOUND\n"
        "TRANSACTION_DATES: 01/01/2024 | 02/01/2024\n"
        "LINE_ITEM_DESCRIPTIONS: STORE A | STORE B\n"
        "TRANSACTION_AMOUNTS_PAID: NOT_FOUND\n"
    )
    with caplog.at_level(logging.WARNING, logger="common.unified_bank_extractor"):
        result = _ube(response)._extract_schema_fallback("img.png", [], ColumnMapping())
    assert result.transaction_dates == []
    assert any("discarding" in r.message.lower() for r in caplog.records)
