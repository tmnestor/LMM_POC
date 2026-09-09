"""Tests for bank_post_process: date-range formats + dropped-row visibility."""

import logging

from common.bank_post_process import (
    _align_amount_arrays,
    _align_balance_arrays,
    _compute_date_range,
)

# ---------------------------------------------------------------------------
# _compute_date_range must parse what the row parser emits
# ---------------------------------------------------------------------------


def test_date_range_no_year():
    assert _compute_date_range(["25 May", "20 May"]) == "20 May - 25 May"


def test_date_range_two_digit_year():
    assert _compute_date_range(["06 Aug 24", "01 Aug 24"]) == "01 Aug 24 - 06 Aug 24"


def test_date_range_existing_format_still_parses():
    assert _compute_date_range(["04 Sep 2025", "01 Sep 2025"]) == "01 Sep 2025 - 04 Sep 2025"


# ---------------------------------------------------------------------------
# Aligned-array assembly: dropped rows must be visible, not silent
# ---------------------------------------------------------------------------


def test_align_balance_arrays_warns_on_dropped_rows(caplog):
    rows = [
        {"Date": "01/01/2024", "Desc": "KEEP", "Debit": "$5.00"},
        {"Date": "02/01/2024", "Desc": "LOST", "Debit": ""},
    ]
    with caplog.at_level(logging.WARNING, logger="common.bank_post_process"):
        dates, descs, amounts, balances = _align_balance_arrays(rows, "Date", "Desc", "Debit", None)
    assert len(dates) == 1
    assert any("dropped 1/2" in r.message for r in caplog.records)


def test_align_balance_arrays_silent_when_complete(caplog):
    rows = [{"Date": "01/01/2024", "Desc": "K", "Debit": "$5.00"}]
    with caplog.at_level(logging.WARNING, logger="common.bank_post_process"):
        _align_balance_arrays(rows, "Date", "Desc", "Debit", None)
    assert not caplog.records


def test_align_amount_arrays_warns_on_dropped_rows(caplog):
    rows = [
        {"Date": "01/01/2024", "Desc": "KEEP", "Amount": "-5.00"},
        {"Date": "", "Desc": "LOST", "Amount": "-9.00"},
    ]
    with caplog.at_level(logging.WARNING, logger="common.bank_post_process"):
        dates, descs, amounts, balances = _align_amount_arrays(rows, "Date", "Desc", "Amount", None)
    assert len(dates) == 1
    assert any("dropped 1/2" in r.message for r in caplog.records)
