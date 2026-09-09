"""Tests for bank_corrector: parser-emitted date formats + drop visibility.

The row parser (UnifiedBankExtractor date patterns 1-6) emits no-year dates
("20 May"), 2-digit-year dates ("06 Aug 24") and 2-digit slash years
("03/05/25"). The corrector's date handling must parse everything the parser
produces — otherwise chronology detection silently fails and BalanceCorrector
computes sign-inverted deltas on reverse-chronological statements.
"""

import logging

from common.bank_corrector import BalanceCorrector, TransactionFilter


def _rows(dates):
    return [{"Date": d, "Description": "X", "Debit": "$5.00"} for d in dates]


# ---------------------------------------------------------------------------
# Date formats the row parser emits
# ---------------------------------------------------------------------------


def test_chronological_no_year_dates():
    ok, reason = BalanceCorrector.is_chronological_order(_rows(["20 May", "25 May"]), "Date")
    assert ok is True, reason


def test_reverse_chronological_two_digit_year():
    ok, reason = BalanceCorrector.is_chronological_order(_rows(["06 Aug 24", "01 Aug 24"]), "Date")
    assert ok is False
    assert "Reverse" in reason


def test_chronological_two_digit_slash_year():
    ok, reason = BalanceCorrector.is_chronological_order(_rows(["03/05/25", "07/05/25"]), "Date")
    assert ok is True, reason


def test_chronological_existing_formats_still_parse():
    ok, reason = BalanceCorrector.is_chronological_order(_rows(["04 Sep 2025", "09 Sep 2025"]), "Date")
    assert ok is True, reason


def test_sort_by_date_no_year():
    rows = _rows(["25 May", "20 May", "22 May"])
    out = BalanceCorrector.sort_by_date(rows, "Date")
    assert [r["Date"] for r in out] == ["20 May", "22 May", "25 May"]


def test_sort_by_date_two_digit_year():
    rows = _rows(["06 Aug 24", "01 Aug 24"])
    out = BalanceCorrector.sort_by_date(rows, "Date")
    assert [r["Date"] for r in out] == ["01 Aug 24", "06 Aug 24"]


# ---------------------------------------------------------------------------
# filter_debits: unparseable amounts must be visible, not silently deleted
# ---------------------------------------------------------------------------


def test_parse_amount_or_none():
    assert TransactionFilter.parse_amount_or_none("$1,234.56") == 1234.56
    assert TransactionFilter.parse_amount_or_none("-5.00") == -5.00
    assert TransactionFilter.parse_amount_or_none("garbled##") is None
    assert TransactionFilter.parse_amount_or_none("") is None
    # parse_amount keeps its legacy zero-on-failure contract
    assert TransactionFilter.parse_amount("garbled##") == 0.0


def test_filter_debits_warns_on_unparseable_amounts(caplog):
    rows = [
        {"Debit": "$5.00", "Description": "KEEP"},
        {"Debit": "garbled##", "Description": "LOST"},
        {"Debit": "", "Description": "EMPTY-IS-CREDIT-ROW"},
        {"Debit": "NOT_FOUND", "Description": "NF-IS-FINE"},
    ]
    with caplog.at_level(logging.WARNING, logger="common.bank_corrector"):
        kept = TransactionFilter.filter_debits(rows, "Debit")
    assert len(kept) == 1
    assert any("unparseable" in r.message for r in caplog.records)


def test_filter_debits_silent_when_all_parse(caplog):
    rows = [
        {"Debit": "$5.00", "Description": "KEEP"},
        {"Debit": "", "Description": "CREDIT-ROW"},
        {"Debit": "$0.00", "Description": "ZERO-IS-NOT-DATA-LOSS"},
    ]
    with caplog.at_level(logging.WARNING, logger="common.bank_corrector"):
        kept = TransactionFilter.filter_debits(rows, "Debit")
    assert len(kept) == 1
    assert not caplog.records
