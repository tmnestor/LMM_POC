"""Tests for SROIE field normalisation.

Every date and total pattern asserted here was enumerated from the real
ICDAR 2019 SROIE test split (347 records), not invented.
"""

from datetime import date
from decimal import Decimal

import pytest

from common.sroie.normalise import (
    SroieNormalisationError,
    normalise_date,
    normalise_text_lenient,
    normalise_text_strict,
    normalise_total,
)


def test_parses_dd_mm_yyyy_the_dominant_pattern() -> None:
    """187 of 347 test-split records use DD/MM/YYYY."""
    assert normalise_date("15/01/2019") == date(2019, 1, 15)


def test_two_digit_year_expands_to_the_2000s() -> None:
    """34 records use DD/MM/YY. The old scorer compared these as raw
    strings, marking correct extractions wrong."""
    assert normalise_date("03/03/18") == date(2018, 3, 3)


def test_separator_does_not_change_the_date() -> None:
    """Slash, hyphen and dot all appear in the test split."""
    assert normalise_date("22-03-2018") == date(2018, 3, 22)
    assert normalise_date("20-11-17") == date(2017, 11, 20)
    assert normalise_date("23.03.18") == date(2018, 3, 23)


def test_parses_alphabetic_month_names() -> None:
    """43 records name the month, with space, hyphen or slash separators."""
    assert normalise_date("07 MAR 2018") == date(2018, 3, 7)
    assert normalise_date("28-FEB-2018") == date(2018, 2, 28)
    assert normalise_date("27/MAR/2018") == date(2018, 3, 27)
    assert normalise_date("22 MAR 18") == date(2018, 3, 22)


def test_leading_four_digit_year_is_read_as_iso_order() -> None:
    """A 4-digit first field can only be a year, so the order is Y-M-D."""
    assert normalise_date("2016-07-31") == date(2016, 7, 31)
    assert normalise_date("2018/03/27") == date(2018, 3, 27)


def test_month_day_order_when_the_second_field_exceeds_twelve() -> None:
    """'4/22/2018' can only be MM/DD — 22 is not a month."""
    assert normalise_date("4/22/2018") == date(2018, 4, 22)


def test_ambiguous_order_defaults_to_day_first() -> None:
    """Malaysian receipts write DD/MM, so an all-under-13 date is read
    day-first. This is an assumption the corpus cannot settle."""
    assert normalise_date("06/03/2018") == date(2018, 3, 6)


def test_unparseable_date_raises_a_diagnostic_error() -> None:
    """The old scorer fell back to raw string comparison here, silently
    turning correct extractions into failures. Fail loudly instead."""
    with pytest.raises(SroieNormalisationError) as excinfo:
        normalise_date("sometime last tuesday", source="X51005447850")

    message = str(excinfo.value)
    assert "sometime last tuesday" in message  # what is wrong
    assert "X51005447850" in message  # where it came from
    assert "15/01/2019" in message  # what it should look like
    assert "common/sroie/normalise.py" in message  # how to recover


def test_ignores_surrounding_brackets() -> None:
    """One train record writes '(06/12/2016)'."""
    assert normalise_date("(06/12/2016)") == date(2016, 12, 6)


def test_parses_a_leading_month_name() -> None:
    """'OCT 3, 2016' puts the month first and separates with a comma."""
    assert normalise_date("OCT 3, 2016") == date(2016, 10, 3)


def test_parses_eight_digit_dates_in_both_orderings() -> None:
    """The train split writes both. A leading 19xx/20xx can only be a year,
    so the ordering is recoverable without guessing."""
    assert normalise_date("20180304") == date(2018, 3, 4)
    assert normalise_date("25032018") == date(2018, 3, 25)


def test_incomplete_date_raises_rather_than_guessing() -> None:
    """'15/01' has no year. Inventing one would fabricate a match."""
    with pytest.raises(SroieNormalisationError):
        normalise_date("15/01")


def test_impossible_calendar_date_raises() -> None:
    """31 February is not a date, however well-formed it looks."""
    with pytest.raises(SroieNormalisationError):
        normalise_date("31/02/2018")


def test_total_keeps_two_decimal_places() -> None:
    """153 of 347 totals are already plain NN.NN."""
    assert normalise_total("193.00") == Decimal("193.00")


def test_total_strips_currency_markers() -> None:
    """46 test-split totals carry a currency prefix, spaced or not."""
    assert normalise_total("$8.20") == Decimal("8.20")
    assert normalise_total("RM7.42") == Decimal("7.42")
    assert normalise_total("RM 10.60") == Decimal("10.60")


def test_total_strips_repeated_currency_markers() -> None:
    """Models emit 'RM $8.20' as readily as 'RM8.20'."""
    assert normalise_total("RM $8.20") == Decimal("8.20")
    assert normalise_total("RM$8.20") == Decimal("8.20")


def test_total_pads_a_single_decimal_place() -> None:
    """'108.0' and '108.00' are the same amount."""
    assert normalise_total("108.0") == Decimal("108.00")


def test_total_keeps_a_negative_sign() -> None:
    """Two test-split totals are negative; dropping the sign would make a
    refund compare equal to a charge."""
    assert normalise_total("-9.99") == Decimal("-9.99")


def test_total_handles_thousands_separators() -> None:
    """Models emit '1,234.50' for larger totals."""
    assert normalise_total("1,234.50") == Decimal("1234.50")


def test_unparseable_total_raises_a_diagnostic_error() -> None:
    with pytest.raises(SroieNormalisationError) as excinfo:
        normalise_total("about ten ringgit", source="X51005447850")

    message = str(excinfo.value)
    assert "about ten ringgit" in message
    assert "X51005447850" in message


def test_strict_text_ignores_case_and_whitespace_only() -> None:
    """The official SROIE protocol is exact match; case and run-length
    whitespace are the only free variation."""
    assert normalise_text_strict("ENW  HARDWARE   CENTRE") == normalise_text_strict("enw hardware centre")


def test_strict_text_still_separates_on_punctuation() -> None:
    """Strict must NOT forgive punctuation, or it is not strict."""
    assert normalise_text_strict("MR. D. I. Y.") != normalise_text_strict("MR. D.I.Y.")


def test_lenient_text_forgives_punctuation_and_spacing() -> None:
    """105 of 250 address failures in the previous run differed only by a
    comma or a space, e.g. '27, JALAN' vs '27,JALAN'."""
    gt = "27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU, JOHOR."
    predicted = "27,JALAN DEDAP 13, TAMAN JOHOR JAYA, 81100 JOHOR BAHRU,JOHOR."
    assert normalise_text_lenient(gt) == normalise_text_lenient(predicted)


def test_lenient_text_does_not_collapse_different_addresses() -> None:
    """Leniency must not go so far that distinct values compare equal."""
    assert normalise_text_lenient("27 JALAN DEDAP 13") != normalise_text_lenient("28 JALAN DEDAP 13")
