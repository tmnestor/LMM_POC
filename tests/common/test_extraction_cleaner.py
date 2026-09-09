"""Tests for ExtractionCleaner thousands-separator handling in monetary list fields.

tests/ is gitignored — local-only. Locks in the 2026-06-06 fix: the list-field
cleaner split values on EVERY comma to convert comma-separated lists to
pipe-separated, which also split the thousands separator inside monetary values
(``$8,026.87`` -> ``$8 | 026.87``). That destroyed every transaction amount
>= $1,000 and misaligned the amount array against dates/descriptions, surfacing
as linking false negatives (PROD synthetic_clean: 7 of 10 FN were >= $1,000).
Proven via raw_extractions.jsonl (value present in raw_response, absent in
cleaned_extractions). Both comma-splits (clean_field_value pre-process and
_clean_list_field) must ignore commas that sit inside a number.
"""

from common.extraction_cleaner import clean_field_value

_FIELD = "TRANSACTION_AMOUNTS_PAID"


def _amounts(result: str) -> list[float]:
    """Parse a pipe-delimited cleaned amount string to floats (strip $ and commas)."""
    return [float(item.strip().replace("$", "").replace(",", "")) for item in result.split("|")]


class TestThousandsSeparator:
    def test_single_thousands_amount_stays_whole(self) -> None:
        # Was split into "$8 | 026.87" (two junk values) by the comma split.
        result = clean_field_value(_FIELD, "$8,026.87")
        assert _amounts(result) == [8026.87]

    def test_comma_list_with_thousands_amount_splits_correctly(self) -> None:
        # The list commas are delimiters; the thousands comma is not.
        result = clean_field_value(_FIELD, "$127.35, $8,026.87, $48.50")
        assert _amounts(result) == [127.35, 8026.87, 48.50]

    def test_pipe_list_with_thousands_amount_unchanged(self) -> None:
        # Already pipe-delimited input must round-trip without corruption.
        result = clean_field_value(_FIELD, "$127.35 | $8,026.87 | $48.50")
        assert _amounts(result) == [127.35, 8026.87, 48.50]

    def test_multiple_thousands_amounts_comma_list(self) -> None:
        result = clean_field_value(_FIELD, "$1,234.56, $7,890.12")
        assert _amounts(result) == [1234.56, 7890.12]

    def test_millions_amount_with_two_internal_commas(self) -> None:
        # ,(?!\\d) must leave every in-number comma intact, not just the first.
        result = clean_field_value(_FIELD, "$1,000,000.00")
        assert _amounts(result) == [1000000.00]
