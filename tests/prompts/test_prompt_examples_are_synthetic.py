"""Prompt examples must never contain a real value from the corpus.

tests/ is gitignored -- local-only.

Found 2026-08-12 while adding a LINE_ITEM_QUANTITIES format rule. Four prompt
files illustrated STATEMENT_DATE_RANGE with the literal
`01/01/2024 - 31/01/2024`, which is the true answer for 2 of the 55 evaluation
bank statements. For those documents the prompt handed the model the answer, so
the field's near-perfect score (InternVL 0.991, Gemma 0.945) was measuring the
prompt as much as the model.

A literal example is only ever a convenience -- a format placeholder carries the
same information and cannot leak. These tests enforce that.
"""

import re
from pathlib import Path

import pytest
import yaml

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_FILES = sorted(PROMPT_DIR.rglob("*.yaml"))

# A date the model is asked to find is by definition a candidate answer, so a
# literal one in an example is always a leak risk. Placeholders like
# "DD/MM/YYYY" carry the format without carrying a value.
LITERAL_DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

# The specific string that leaked, kept as its own case so the regression is
# named rather than only implied by the general rule.
LEAKED_RANGE = "01/01/2024 - 31/01/2024"


def test_prompt_files_exist() -> None:
    """Guard the premise: a glob that silently matches nothing tests nothing."""
    assert len(PROMPT_FILES) >= 5


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_no_literal_dates_in_prompts(path: Path) -> None:
    found = LITERAL_DATE.findall(path.read_text())
    assert not found, (
        f"{path} contains literal date(s) {found}. A date example must be a "
        f'format placeholder such as "DD/MM/YYYY", never a value that could be '
        f"a document's true answer."
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_the_known_leak_stays_fixed(path: Path) -> None:
    assert LEAKED_RANGE not in path.read_text()


class TestQuantityFormatRule:
    """The receipt prompt must say how to write a quantity.

    Gemma emitted "3x | 1x | 2x" against ground truth "3 | 1 | 2" -- every value
    correct, the field scored 0.000 on 29 of 55 receipts -- because the prompt
    specified no format for this field at all.
    """

    @staticmethod
    def receipt_prompt() -> str:
        data = yaml.safe_load((PROMPT_DIR / "internvl3_prompts.yaml").read_text())
        return data["prompts"]["receipt"]["prompt"]

    def test_quantities_are_asked_for_as_digits(self) -> None:
        body = self.receipt_prompt().lower()
        assert "digits only" in body

    def test_the_x_suffix_is_called_out(self) -> None:
        # The instruction has to name the failure mode; "one number per item"
        # alone did not stop it, since "3x" is one number per item.
        assert '"3x"' in self.receipt_prompt()

    def test_quantity_and_name_are_separated(self) -> None:
        body = self.receipt_prompt()
        assert "LINE_ITEM_QUANTITIES" in body
        assert "no leading count" in body

    def test_the_example_is_a_placeholder_not_a_real_item(self) -> None:
        """The rule that started this: no corpus values in examples."""
        assert "<item name>" in self.receipt_prompt()


class TestPriceFieldRules:
    """The receipt prompt must say which price field takes the printed amount.

    Moved here 2026-08-13 from tests/scripts/, whose subject
    (scripts/patch_receipt_unit_prices.py) was deleted once the generator began
    producing correct ground truth. The prompt half of that work is still live
    and still needs guarding.

    A receipt line prints ONE amount and that amount is the line total; the
    per-unit price is nowhere on the page. Both models mishandled this: InternVL
    left LINE_ITEM_TOTAL_PRICES empty despite having read the number, and Gemma
    multiplied an amount that was already a line total.
    """

    @staticmethod
    def receipt_prompt() -> str:
        """Prompt text with runs of whitespace collapsed.

        The rules wrap across lines in the YAML, so asserting on exact layout
        would test the line breaks rather than the instruction.
        """
        import re

        data = yaml.safe_load((PROMPT_DIR / "internvl3_prompts.yaml").read_text())
        return re.sub(r"\s+", " ", data["prompts"]["receipt"]["prompt"])

    def test_the_printed_amount_is_named_a_line_total(self) -> None:
        assert "LINE_ITEM_TOTAL_PRICES" in self.receipt_prompt()

    def test_model_is_told_not_to_multiply(self) -> None:
        assert "never multiply it by the quantity" in self.receipt_prompt()

    def test_model_is_told_not_to_divide(self) -> None:
        assert "Never divide a line total to produce a unit price" in self.receipt_prompt()

    def test_the_unit_price_rule_keys_off_quantity(self) -> None:
        """It must agree with the answer key, which keeps quantity-1 amounts.

        An earlier wording said to write NOT_FOUND for "any line showing a
        single amount" — every line — while ground truth keeps the amount where
        quantity is 1, because there the printed figure IS the unit price. Both
        models obeyed the prompt, so the field scored an identical 0.273 for
        both at every tier: it had stopped measuring the model.
        """
        body = self.receipt_prompt()
        assert "Quantity 1 means the single printed amount is also the unit price" in body
        assert "Quantity 2 or more" in body
        assert "write NOT_FOUND for that line" in body
        assert "NOT_FOUND for any line showing a single amount" not in body
