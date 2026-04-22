"""GPU-free tests for turn parsers."""

import pytest

from common.extraction_types import WorkflowState
from common.turn_parsers import (
    FieldValueParser,
    HeaderListParser,
    ParseError,
    ReceiptListParser,
    TransactionMatchParser,
    _parse_amount,
    dedup_by_field,
    enforce_amount_gate,
)

CTX = WorkflowState()


# ---------------------------------------------------------------------------
# HeaderListParser
# ---------------------------------------------------------------------------


class TestHeaderListParser:
    parser = HeaderListParser()

    def test_numbered_list(self) -> None:
        response = "1. Date\n2. Description\n3. Debit\n4. Credit\n5. Balance"
        result = self.parser.parse(response, CTX)
        assert result["headers"] == [
            "Date",
            "Description",
            "Debit",
            "Credit",
            "Balance",
        ]
        assert result["column_mapping"]["date"] == "Date"
        assert result["column_mapping"]["description"] == "Description"
        assert result["column_mapping"]["debit"] == "Debit"
        assert result["column_mapping"]["credit"] == "Credit"
        assert result["column_mapping"]["balance"] == "Balance"

    def test_comma_separated(self) -> None:
        response = "Date, Description, Debit, Credit, Balance"
        result = self.parser.parse(response, CTX)
        assert len(result["headers"]) == 5
        assert result["column_mapping"]["date"] == "Date"

    def test_pipe_separated(self) -> None:
        response = "Date | Description | Amount | Balance"
        result = self.parser.parse(response, CTX)
        assert len(result["headers"]) == 4
        assert result["column_mapping"]["amount"] == "Amount"

    def test_with_markdown_bold(self) -> None:
        response = "1. **Date**\n2. **Description**\n3. **Debit**"
        result = self.parser.parse(response, CTX)
        assert result["headers"] == ["Date", "Description", "Debit"]

    def test_empty_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("", CTX)

    def test_skips_labels_and_long_lines(self) -> None:
        response = (
            "Headers:\n"
            "1. Date\n"
            "2. Description\n"
            "This is a very long line that should be skipped entirely from the output"
        )
        result = self.parser.parse(response, CTX)
        assert result["headers"] == ["Date", "Description"]


# ---------------------------------------------------------------------------
# ReceiptListParser
# ---------------------------------------------------------------------------


class TestReceiptListParser:
    parser = ReceiptListParser()

    def test_single_receipt(self) -> None:
        response = (
            "--- RECEIPT 1 ---\nSTORE: Corner Shop\nDATE: 15/03/2025\nTOTAL: $42.50\n"
        )
        result = self.parser.parse(response, CTX)
        assert result["receipt_count"] == 1
        assert result["receipts"][0]["STORE"] == "Corner Shop"
        assert result["receipts"][0]["DATE"] == "15/03/2025"
        assert result["receipts"][0]["TOTAL"] == "$42.50"
        assert "Purchase 1:" in result["formatted_text"]

    def test_multiple_receipts(self) -> None:
        response = (
            "--- RECEIPT 1 ---\n"
            "STORE: Shop A\n"
            "DATE: 01/01/2025\n"
            "TOTAL: $10.00\n"
            "\n"
            "--- RECEIPT 2 ---\n"
            "STORE: Shop B\n"
            "DATE: 02/01/2025\n"
            "TOTAL: $20.00\n"
        )
        result = self.parser.parse(response, CTX)
        assert result["receipt_count"] == 2
        assert result["receipts"][1]["STORE"] == "Shop B"

    def test_no_receipts_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("No receipts found.", CTX)

    def test_ignores_extra_fields(self) -> None:
        response = (
            "--- RECEIPT 1 ---\n"
            "STORE: Test Store\n"
            "DATE: 01/01/2025\n"
            "TOTAL: $5.00\n"
            "PAYMENT_METHOD: Visa\n"
        )
        result = self.parser.parse(response, CTX)
        assert "PAYMENT_METHOD" not in result["receipts"][0]


# ---------------------------------------------------------------------------
# TransactionMatchParser
# ---------------------------------------------------------------------------


class TestTransactionMatchParser:
    parser = TransactionMatchParser()

    def test_single_match(self) -> None:
        response = (
            "--- RECEIPT 1 ---\n"
            "MATCHED_TRANSACTION: FOUND\n"
            "TRANSACTION_DATE: 15/03/2025\n"
            "TRANSACTION_AMOUNT: $42.50\n"
            "TRANSACTION_DESCRIPTION: CORNER SHOP PTY\n"
            "RECEIPT_STORE: Corner Shop\n"
            "RECEIPT_TOTAL: $42.50\n"
            "CONFIDENCE: HIGH\n"
            "REASONING: Amount matches exactly\n"
        )
        result = self.parser.parse(response, CTX)
        assert result["match_count"] == 1
        m = result["matches"][0]
        assert m["MATCHED_TRANSACTION"] == "FOUND"
        assert m["TRANSACTION_AMOUNT"] == "$42.50"

    def test_not_found(self) -> None:
        response = (
            "--- RECEIPT 1 ---\n"
            "MATCHED_TRANSACTION: NOT_FOUND\n"
            "TRANSACTION_DATE: NOT_FOUND\n"
            "TRANSACTION_AMOUNT: NOT_FOUND\n"
            "TRANSACTION_DESCRIPTION: NOT_FOUND\n"
            "RECEIPT_STORE: Corner Shop\n"
            "RECEIPT_TOTAL: $42.50\n"
            "CONFIDENCE: LOW\n"
            "REASONING: No matching debit found\n"
        )
        result = self.parser.parse(response, CTX)
        assert result["matches"][0]["MATCHED_TRANSACTION"] == "NOT_FOUND"

    def test_multiple_matches(self) -> None:
        response = (
            "--- RECEIPT 1 ---\n"
            "MATCHED_TRANSACTION: FOUND\n"
            "TRANSACTION_DATE: 01/01/2025\n"
            "TRANSACTION_AMOUNT: $10.00\n"
            "TRANSACTION_DESCRIPTION: SHOP A\n"
            "RECEIPT_STORE: Shop A\n"
            "RECEIPT_TOTAL: $10.00\n"
            "CONFIDENCE: HIGH\n"
            "REASONING: Exact match\n"
            "\n"
            "--- RECEIPT 2 ---\n"
            "MATCHED_TRANSACTION: FOUND\n"
            "TRANSACTION_DATE: 02/01/2025\n"
            "TRANSACTION_AMOUNT: $20.00\n"
            "TRANSACTION_DESCRIPTION: SHOP B\n"
            "RECEIPT_STORE: Shop B\n"
            "RECEIPT_TOTAL: $20.00\n"
            "CONFIDENCE: HIGH\n"
            "REASONING: Exact match\n"
        )
        result = self.parser.parse(response, CTX)
        assert result["match_count"] == 2

    def test_empty_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("No matches.", CTX)


# ---------------------------------------------------------------------------
# FieldValueParser
# ---------------------------------------------------------------------------


class TestFieldValueParser:
    parser = FieldValueParser()

    def test_basic_fields(self) -> None:
        response = (
            "SUPPLIER_NAME: Acme Corp\n"
            "TOTAL_AMOUNT: $1,234.56\n"
            "INVOICE_DATE: 15/03/2025\n"
        )
        result = self.parser.parse(response, CTX)
        assert result["SUPPLIER_NAME"] == "Acme Corp"
        assert result["TOTAL_AMOUNT"] == "$1,234.56"

    def test_empty_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("", CTX)

    def test_no_valid_fields_raises(self) -> None:
        with pytest.raises(ParseError):
            self.parser.parse("Just some text\nwithout colons", CTX)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestParseAmount:
    def test_basic(self) -> None:
        assert _parse_amount("$42.50") == pytest.approx(42.50)

    def test_negative(self) -> None:
        assert _parse_amount("-$42.50") == pytest.approx(-42.50)

    def test_parentheses(self) -> None:
        assert _parse_amount("($42.50)") == pytest.approx(-42.50)

    def test_comma_separator(self) -> None:
        assert _parse_amount("$1,234.56") == pytest.approx(1234.56)

    def test_none_on_invalid(self) -> None:
        assert _parse_amount("NOT_FOUND") is None
        assert _parse_amount("") is None
        assert _parse_amount("abc") is None


class TestEnforceAmountGate:
    def test_matching_amounts_pass(self) -> None:
        receipts = [{"TOTAL": "$42.50"}]
        matches = [{"MATCHED_TRANSACTION": "FOUND", "TRANSACTION_AMOUNT": "$42.50"}]
        result = enforce_amount_gate(receipts, matches)
        assert result[0]["MATCHED_TRANSACTION"] == "FOUND"

    def test_mismatched_amounts_override(self) -> None:
        receipts = [{"TOTAL": "$42.50"}]
        matches = [{"MATCHED_TRANSACTION": "FOUND", "TRANSACTION_AMOUNT": "$99.99"}]
        result = enforce_amount_gate(receipts, matches)
        assert result[0]["MATCHED_TRANSACTION"] == "NOT_FOUND"
        assert "Amount gate" in result[0]["REASONING"]

    def test_not_found_unchanged(self) -> None:
        receipts = [{"TOTAL": "$42.50"}]
        matches = [
            {"MATCHED_TRANSACTION": "NOT_FOUND", "TRANSACTION_AMOUNT": "NOT_FOUND"}
        ]
        result = enforce_amount_gate(receipts, matches)
        assert result[0]["MATCHED_TRANSACTION"] == "NOT_FOUND"


class TestDedupByField:
    def test_removes_duplicates(self) -> None:
        records = [
            {"RECEIPT_STORE": "Shop A", "TOTAL": "$10"},
            {"RECEIPT_STORE": "Shop A", "TOTAL": "$10"},
            {"RECEIPT_STORE": "Shop B", "TOTAL": "$20"},
        ]
        result = dedup_by_field(records, "RECEIPT_STORE")
        assert len(result) == 2

    def test_keeps_records_without_field(self) -> None:
        records = [
            {"TOTAL": "$10"},
            {"RECEIPT_STORE": "Shop A", "TOTAL": "$20"},
        ]
        result = dedup_by_field(records, "RECEIPT_STORE")
        assert len(result) == 2
