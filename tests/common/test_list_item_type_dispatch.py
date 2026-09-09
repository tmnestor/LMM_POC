"""A list field must be compared by its TYPE, not by where it happens to live.

tests/ is gitignored -- local-only.

Found 2026-08-13. Every list path asked only "is this a transaction_list
field?" and, if not, fell through to text matching. That is a structural
question standing in for a semantic one, and it silently mis-scored monetary
list fields the bank register does not contain:

    LINE_ITEM_TOTAL_PRICES  extracted "9.72 | 111.69"
                            truth     "$9.72 | $111.69"   ->  f1 0.000

The values are correct; only the currency symbol differs. As text, Jaccard
overlap is 0 and similarity is 0.80, so both miss. InternVL lost the whole
field on all 55 receipts while Gemma, which emits "$", scored 0.987 -- which
reads as a capability gap and is a formatting difference.

config/field_definitions.yaml already declared both fields under
field_types.monetary. The scorer just never consulted it for list items.
"""

import pytest

from common.evaluation_metrics import _list_item_matches, calculate_field_accuracy_with_method
from common.field_schema import get_field_schema

MONEY = "LINE_ITEM_TOTAL_PRICES"
UNIT = "LINE_ITEM_PRICES"
TEXT = "LINE_ITEM_DESCRIPTIONS"


class TestSchemaDeclaresTheseMonetary:
    """Guard the premise: the fix reads the schema, so the schema must say so."""

    @pytest.mark.parametrize("field", [MONEY, UNIT])
    def test_declared_monetary(self, field: str) -> None:
        assert field in get_field_schema().monetary_fields


class TestCurrencySymbolNoLongerCosts:
    @pytest.mark.parametrize(
        ("extracted", "truth"),
        [
            ("9.72", "$9.72"),  # the measured InternVL case
            ("$9.72", "9.72"),
            ("1234.56", "$1,234.56"),  # thousands separator too
            ("$9.72", "$9.72"),
        ],
    )
    def test_same_amount_matches_however_written(self, extracted: str, truth: str) -> None:
        assert _list_item_matches(extracted, truth, MONEY)

    def test_the_whole_field_now_scores(self) -> None:
        result = calculate_field_accuracy_with_method(
            "9.72 | 111.69 | 12.99", "$9.72 | $111.69 | $12.99", MONEY, method="order_aware_f1"
        )
        assert result["f1_score"] == 1.0

    def test_a_different_amount_still_fails(self) -> None:
        assert not _list_item_matches("9.72", "$97.20", MONEY)

    def test_not_found_is_not_an_amount(self) -> None:
        assert not _list_item_matches("9.72", "NOT_FOUND", MONEY)


class TestOtherTypesAreUnaffected:
    def test_text_fields_still_use_fuzzy_matching(self) -> None:
        # LINE_ITEM_DESCRIPTIONS is a transaction_list field, so it keeps the
        # register's own comparator -- the union matcher.
        assert _list_item_matches("2x USB Hub 4-port", "USB Hub 4-port", TEXT)

    def test_unrelated_text_still_fails(self) -> None:
        assert not _list_item_matches("Coffee Beans 1kg", "Drill Bit Set", TEXT)

    def test_a_plain_text_list_field_is_untouched(self) -> None:
        """A field the schema types as neither monetary nor date keeps text rules."""
        schema = get_field_schema()
        assert "PAYER_NAME" not in schema.monetary_fields
        assert _list_item_matches("Robin Wood", "Robin Wood", "PAYER_NAME")


class TestPromptAgreesWithTheAnswerKey:
    """The prompt and the ground truth must not contradict each other.

    They did: the answer key keeps the printed amount where quantity is 1 (there
    it IS the unit price), while the prompt said to write NOT_FOUND for "any line
    showing a single amount" -- which is every line. Both models obeyed the
    prompt, so LINE_ITEM_PRICES scored an identical 0.273 for both at every
    degradation tier: the metric had stopped depending on the model at all.
    """

    @staticmethod
    def receipt_prompt() -> str:
        import re
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[2]
        data = yaml.safe_load((root / "prompts" / "internvl3_prompts.yaml").read_text())
        return re.sub(r"\s+", " ", data["prompts"]["receipt"]["prompt"])

    def test_quantity_one_is_told_to_report_the_amount(self) -> None:
        assert "Quantity 1 means the single printed amount is also the unit price" in (
            self.receipt_prompt()
        )

    def test_quantity_above_one_is_told_to_write_not_found(self) -> None:
        body = self.receipt_prompt()
        assert "Quantity 2 or more" in body
        assert "write NOT_FOUND for that line" in body

    def test_the_blanket_rule_is_gone(self) -> None:
        """The exact wording that contradicted the answer key."""
        assert "NOT_FOUND for any line showing a single amount" not in self.receipt_prompt()

    def test_still_told_not_to_divide(self) -> None:
        assert "Never divide a line total to produce a unit price" in self.receipt_prompt()
