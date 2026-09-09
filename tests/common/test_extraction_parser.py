"""Tests for hybrid_parse_response thousands-separator handling in list fields.

tests/ is gitignored — local-only. Locks in the 2026-06-06 fix: the parser's
list-field handling (``extraction_parser.py`` ~line 611) converted comma-separated
lists to pipe-separated by splitting on EVERY comma, shredding the thousands
separator inside monetary values (``$8,026.87`` -> ``$8 | 026.87``). This is the
OPERATIVE path in the clean stage (ResponseHandler.handle -> hybrid_parse_response),
upstream of ExtractionCleaner, so it corrupted bank amounts >= $1,000 before the
cleaner ever saw them — the real cause of the linking false negatives.
"""

from common.extraction_parser import hybrid_parse_response

_FIELD = "TRANSACTION_AMOUNTS_PAID"


def _amounts(value: str) -> list[float]:
    return [float(item.strip().replace("$", "").replace(",", "")) for item in value.split("|")]


def _parse(raw: str) -> str:
    return hybrid_parse_response(f"{_FIELD}: {raw}", [_FIELD])[_FIELD]


class TestParserThousandsSeparator:
    def test_single_thousands_amount_stays_whole(self) -> None:
        assert _amounts(_parse("$8,026.87")) == [8026.87]

    def test_comma_list_with_thousands_amount_splits_correctly(self) -> None:
        assert _amounts(_parse("$127.35, $8,026.87, $48.50")) == [127.35, 8026.87, 48.50]

    def test_pipe_list_with_thousands_amount_unchanged(self) -> None:
        assert _amounts(_parse("$127.35 | $8,026.87 | $48.50")) == [127.35, 8026.87, 48.50]

    def test_multiple_thousands_amounts_comma_list(self) -> None:
        assert _amounts(_parse("$1,234.56, $7,890.12")) == [1234.56, 7890.12]
