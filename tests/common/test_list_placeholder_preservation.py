"""Cleaning a list column must not shift it against its siblings.

tests/ is gitignored -- local-only.

Found 2026-08-13. Every list field in this schema is one column of an
index-aligned group: LINE_ITEM_DESCRIPTIONS / QUANTITIES / PRICES /
TOTAL_PRICES are parallel, as are the TRANSACTION_* and JOURNEY_* columns.
Position n names the same row in each.

The cleaner preserved NOT_FOUND placeholders only for fields whose names begin
"TRANSACTION_AMOUNTS", plus ACCOUNT_BALANCE -- the right property recognised on
the wrong grounds. For every other list it dropped them, shortening the column
and shifting everything after it.

Measured on the run of 2026-08-13, Gemma returned exactly the right answer:

    raw     NOT_FOUND | $111.69 | NOT_FOUND | NOT_FOUND | $12.87
    ground  NOT_FOUND | $111.69 | NOT_FOUND | NOT_FOUND | $12.87
    cleaned $111.69 | $12.87            <- what the scorer actually saw

A perfect extraction was scored as two misplaced values. The model was right
and the pipeline broke it.
"""

import pytest

from common.extraction_cleaner import ExtractionCleaner
from common.field_schema import get_field_schema

ALIGNED_LINE_ITEM_FIELDS = [
    "LINE_ITEM_DESCRIPTIONS",
    "LINE_ITEM_QUANTITIES",
    "LINE_ITEM_PRICES",
    "LINE_ITEM_TOTAL_PRICES",
]


@pytest.fixture
def cleaner() -> ExtractionCleaner:
    return ExtractionCleaner()


class TestPlaceholdersSurviveCleaning:
    def test_the_measured_case(self, cleaner: ExtractionCleaner) -> None:
        raw = "NOT_FOUND | $111.69 | NOT_FOUND | NOT_FOUND | $12.87"
        assert cleaner.clean_field_value("LINE_ITEM_PRICES", raw).count("|") == 4

    def test_every_position_is_kept(self, cleaner: ExtractionCleaner) -> None:
        raw = "NOT_FOUND | $111.69 | NOT_FOUND | NOT_FOUND | $12.87"
        out = [v.strip() for v in cleaner.clean_field_value("LINE_ITEM_PRICES", raw).split("|")]
        assert len(out) == 5
        assert out[0] == "NOT_FOUND"
        assert out[2] == out[3] == "NOT_FOUND"

    @pytest.mark.parametrize("field", ALIGNED_LINE_ITEM_FIELDS)
    def test_length_is_preserved_for_each_parallel_column(
        self, cleaner: ExtractionCleaner, field: str
    ) -> None:
        raw = "A | NOT_FOUND | C"
        cleaned = cleaner.clean_field_value(field, raw)
        assert len([v for v in cleaned.split("|")]) == 3, f"{field} lost a position"

    def test_columns_stay_aligned_with_each_other(self, cleaner: ExtractionCleaner) -> None:
        """The property that actually matters: same length across siblings."""
        row = {
            "LINE_ITEM_DESCRIPTIONS": "Item A | Item B | Item C",
            "LINE_ITEM_QUANTITIES": "3 | 1 | 2",
            "LINE_ITEM_PRICES": "NOT_FOUND | $111.69 | NOT_FOUND",
            "LINE_ITEM_TOTAL_PRICES": "$9.72 | $111.69 | $25.38",
        }
        lengths = {
            field: len(cleaner.clean_field_value(field, value).split("|")) for field, value in row.items()
        }
        assert len(set(lengths.values())) == 1, f"columns diverged: {lengths}"


class TestBankRegisterStillWorks:
    """The behaviour that was already correct must not regress."""

    @pytest.mark.parametrize("field", ["TRANSACTION_AMOUNTS_PAID", "ACCOUNT_BALANCE"])
    def test_placeholders_still_preserved(self, cleaner: ExtractionCleaner, field: str) -> None:
        raw = "NOT_FOUND | $42805.81 | $587.22 | NOT_FOUND"
        out = [v.strip() for v in cleaner.clean_field_value(field, raw).split("|")]
        assert len(out) == 4
        assert out[0] == "NOT_FOUND" and out[-1] == "NOT_FOUND"


class TestUnchangedCases:
    def test_a_wholly_absent_field_stays_scalar(self, cleaner: ExtractionCleaner) -> None:
        assert cleaner.clean_field_value("LINE_ITEM_PRICES", "NOT_FOUND") == "NOT_FOUND"

    def test_a_full_list_is_untouched(self, cleaner: ExtractionCleaner) -> None:
        out = cleaner.clean_field_value("LINE_ITEM_PRICES", "$9.72 | $111.69")
        assert [v.strip() for v in out.split("|")] == ["$9.72", "$111.69"]

    def test_the_schema_types_all_four_as_lists(self) -> None:
        """Guard the premise: these are list fields, so this cleaner path runs."""
        schema = get_field_schema()
        for field in ALIGNED_LINE_ITEM_FIELDS:
            assert field in schema.list_fields


class TestAllPlaceholderListCollapses:
    """A column of nothing but placeholders is written as the scalar.

    The ground-truth projection collapses it that way
    (Synthetic_Doc_Generation, _blank_unprinted_unit_prices), so a model
    emitting one placeholder per line scored zero against a ground truth of
    "NOT_FOUND" -- on 10 of 55 receipts, a field where both sides agreed there
    was nothing to find. Re-cleaning the collected run with this rule lifts
    LINE_ITEM_PRICES from 0.485 to 0.705.
    """

    @pytest.mark.parametrize(
        "raw",
        ["NOT_FOUND | NOT_FOUND", "NOT_FOUND | NOT_FOUND | NOT_FOUND | NOT_FOUND"],
    )
    def test_collapses_to_the_scalar(self, cleaner: ExtractionCleaner, raw: str) -> None:
        assert cleaner.clean_field_value("LINE_ITEM_PRICES", raw) == "NOT_FOUND"

    def test_a_single_real_value_prevents_collapse(self, cleaner: ExtractionCleaner) -> None:
        out = cleaner.clean_field_value("LINE_ITEM_PRICES", "NOT_FOUND | $12.87 | NOT_FOUND")
        assert [v.strip() for v in out.split("|")] == ["NOT_FOUND", "$12.87", "NOT_FOUND"]

    def test_the_bank_register_collapses_too(self, cleaner: ExtractionCleaner) -> None:
        """An all-credit statement has no debits, which the scalar says exactly."""
        assert cleaner.clean_field_value("TRANSACTION_AMOUNTS_PAID", "NOT_FOUND | NOT_FOUND") == (
            "NOT_FOUND"
        )
