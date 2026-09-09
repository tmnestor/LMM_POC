"""Tests for ColumnMatcher merged debit/credit header splitting.

tests/ is gitignored — local-only. Locks in the 2026-06-04 westpac fix: header
detection merges the two adjacent "Debits" + "Credits (-)" columns into one
"Debits/Credits (-)" header, which made debit_col == credit_col and collapsed the
debit-credit extraction. The matcher now splits it back into two columns.
(Confirmed via raw_prompt_trace.jsonl + the actual westpac statement image.)
"""

from common.unified_bank_extractor import ColumnMatcher

_PATTERNS = {
    "date": {"keywords": ["date", "transaction date"]},
    "description": {"keywords": ["description", "details", "particulars"]},
    "debit": {"keywords": ["debit", "withdrawal", "dr"]},
    "credit": {"keywords": ["credit", "deposit", "cr"]},
    "amount": {"keywords": ["amount"]},
    "balance": {"keywords": ["balance"]},
}


def _match(headers: list[str]):
    return ColumnMatcher(_PATTERNS).match(headers)


class TestMergedHeaderSplit:
    def test_westpac_merged_header_is_split(self) -> None:
        m = _match(["Date of Transaction", "Description", "Debits/Credits (-)"])
        assert m.debit == "Debits"
        assert m.credit == "Credits (-)"

    def test_merged_header_no_slash_is_split(self) -> None:
        # Header detection is inconsistent: sometimes it drops the slash.
        m = _match(["Date of Transaction", "Description", "DebitsCredits (-)"])
        assert m.debit == "Debits"
        assert m.credit == "Credits (-)"

    def test_order_robust_credits_first(self) -> None:
        m = _match(["Date", "Description", "Credits/Debits"])
        assert m.debit == "Debits"
        assert m.credit == "Credits"


class TestNoSpuriousSplit:
    def test_separate_columns_unchanged(self) -> None:
        m = _match(["Date", "Description", "Debits", "Credits (-)"])
        assert m.debit == "Debits"
        assert m.credit == "Credits (-)"

    def test_withdrawal_deposit_unchanged(self) -> None:
        m = _match(["Date", "Description", "Withdrawal", "Deposit", "Balance"])
        assert m.debit == "Withdrawal"
        assert m.credit == "Deposit"
        assert m.balance == "Balance"

    def test_single_amount_column_not_split(self) -> None:
        m = _match(["Date", "Description", "Amount", "Balance"])
        assert m.amount == "Amount"
        assert m.debit is None and m.credit is None
