"""Credit rows must be dropped from BOTH sides before bank-statement scoring.

tests/ is gitignored — local-only.

Locks in the 2026-08-11 finding. Bank ground truth carries EVERY row, blanking
the credit rows with ``NOT_FOUND`` in ``TRANSACTION_AMOUNTS_PAID``. The model is
only asked for the paid column, so it emits the debit rows and nothing else.
Under the default ``order_aware_f1`` scorer the two lists are then offset by the
number of credit rows and every position mismatches, so a substantially CORRECT
extraction scored 0.000 (tp=0, fp=0, fn=len(gt) — the "nothing extracted"
branch). Real example: CASE004 median_f1 0.000 with 23 of 28 rows correct.

``_filter_debit_transactions`` was meant to reconcile this but needs
``ACCOUNT_BALANCE`` / ``TRANSACTION_AMOUNTS_RECEIVED``, which this prompt's
5-field bank set does not request — so it logs "No balance data available" and
returns unchanged.

The fix drops NOT_FOUND-paid rows from each side independently, which keeps the
strict row correspondence that order-aware scoring depends on rather than
abandoning it for position-agnostic matching.
"""

from common.extraction_evaluator import _drop_credit_rows

BANK = "BANK_STATEMENT"


def _gt() -> dict[str, str]:
    """Ground truth shaped like the real thing: rows 0 and 3 are credits."""
    return {
        "DOCUMENT_TYPE": BANK,
        "STATEMENT_DATE_RANGE": "02/12/2023 - 31/12/2023",
        "LINE_ITEM_DESCRIPTIONS": "Salary PAYROLL | BPAY VERRALL | ATM WITHDRAWAL | Transfer In",
        "TRANSACTION_DATES": "02/12/2023 | 03/12/2023 | 05/12/2023 | 06/12/2023",
        "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $42805.81 | $587.22 | NOT_FOUND",
    }


class TestDropsCreditRows:
    def test_credit_rows_removed_from_every_parallel_field(self) -> None:
        out = _drop_credit_rows(_gt(), BANK)
        assert out["TRANSACTION_AMOUNTS_PAID"] == "$42805.81 | $587.22"
        assert out["LINE_ITEM_DESCRIPTIONS"] == "BPAY VERRALL | ATM WITHDRAWAL"
        assert out["TRANSACTION_DATES"] == "03/12/2023 | 05/12/2023"

    def test_scalar_fields_untouched(self) -> None:
        out = _drop_credit_rows(_gt(), BANK)
        assert out["STATEMENT_DATE_RANGE"] == "02/12/2023 - 31/12/2023"
        assert out["DOCUMENT_TYPE"] == BANK

    def test_input_is_not_mutated(self) -> None:
        gt = _gt()
        _drop_credit_rows(gt, BANK)
        assert gt["TRANSACTION_AMOUNTS_PAID"].startswith("NOT_FOUND")

    def test_symmetric_on_the_extracted_side(self) -> None:
        # A model that DOES emit NOT_FOUND placeholders must be filtered the same
        # way, or the fix would re-introduce the very offset it removes.
        ext = {
            "DOCUMENT_TYPE": BANK,
            "LINE_ITEM_DESCRIPTIONS": "Salary | BPAY VERRALL | ATM WITHDRAWAL",
            "TRANSACTION_DATES": "02/12/2023 | 03/12/2023 | 05/12/2023",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $42805.81 | $587.22",
        }
        out = _drop_credit_rows(ext, BANK)
        assert out["TRANSACTION_AMOUNTS_PAID"] == "$42805.81 | $587.22"
        assert out["LINE_ITEM_DESCRIPTIONS"] == "BPAY VERRALL | ATM WITHDRAWAL"


class TestLeavesEverythingElseAlone:
    def test_no_credit_rows_returns_unchanged(self) -> None:
        gt = {
            "DOCUMENT_TYPE": BANK,
            "LINE_ITEM_DESCRIPTIONS": "A | B",
            "TRANSACTION_DATES": "01/01/2024 | 02/01/2024",
            "TRANSACTION_AMOUNTS_PAID": "$1.00 | $2.00",
        }
        assert _drop_credit_rows(gt, BANK) == gt

    def test_non_bank_doc_type_untouched(self) -> None:
        inv = {
            "DOCUMENT_TYPE": "INVOICE",
            "LINE_ITEM_DESCRIPTIONS": "Consulting | Travel",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $5.00",
        }
        assert _drop_credit_rows(inv, "INVOICE") == inv

    def test_all_rows_credit_returns_unchanged(self) -> None:
        # Blanking every row would turn a real comparison into an empty one and
        # silently score 0 — worse than the bug being fixed.
        gt = {
            "DOCUMENT_TYPE": BANK,
            "LINE_ITEM_DESCRIPTIONS": "A | B",
            "TRANSACTION_DATES": "01/01/2024 | 02/01/2024",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | NOT_FOUND",
        }
        assert _drop_credit_rows(gt, BANK) == gt

    def test_missing_paid_field_returns_unchanged(self) -> None:
        gt = {"DOCUMENT_TYPE": BANK, "LINE_ITEM_DESCRIPTIONS": "A | B"}
        assert _drop_credit_rows(gt, BANK) == gt

    def test_mismatched_field_length_is_skipped_not_corrupted(self) -> None:
        # A parallel field of the wrong length cannot be index-filtered safely;
        # leave it whole rather than silently dropping the wrong rows.
        gt = {
            "DOCUMENT_TYPE": BANK,
            "LINE_ITEM_DESCRIPTIONS": "A | B | C",  # 3 items vs 2 amounts
            "TRANSACTION_DATES": "01/01/2024 | 02/01/2024",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND | $2.00",
        }
        out = _drop_credit_rows(gt, BANK)
        assert out["LINE_ITEM_DESCRIPTIONS"] == "A | B | C"
        assert out["TRANSACTION_DATES"] == "02/01/2024"

    def test_single_value_fields_are_not_split(self) -> None:
        gt = {
            "DOCUMENT_TYPE": BANK,
            "LINE_ITEM_DESCRIPTIONS": "Only one",
            "TRANSACTION_DATES": "01/01/2024",
            "TRANSACTION_AMOUNTS_PAID": "$1.00",
        }
        assert _drop_credit_rows(gt, BANK) == gt
