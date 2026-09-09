"""Tests for ClassificationParser CoT/markdown-drift resilience.

tests/ is gitignored — local-only. These lock in the fix for the 2026-06-04
misclassification where 4 bank statements (CASE007/017/039/047) were tagged
RECEIPT because the model answered the COLUMNS/PAID/ROWS detection prompt with
chain-of-thought + markdown (heading label, value on a later line inside a code
fence), which the inline-only COLUMNS regex could not parse.

The responses below are the real raw_response payloads captured from
classifications.jsonl on the 2xL4 dev run.
"""

from common.turn_parsers import ClassificationParser

# --- Real CoT-drift responses that USED to misclassify as RECEIPT -------------

CASE007_ANZ = """Let's analyze the document image step by step to answer each question.

### 1. COLUMNS
The document shows a transaction table with the following column headers:
- Date
- Transaction Description
- Debits
- Credits
- Balance

So, the exact column headers separated by " | " are:
```
Date | Transaction Description | Debits | Credits | Balance
```

### 2. PAID
To determine if there is evidence that payment was completed, we need to look for:
- Payment method
- "PAID" stamp
- Amount tendered
- Change given
- Receipt number
- EFTPOS/card details

Upon reviewing the document, there are several transactions involving EFTPOS, which indicates"""

CASE017_CBA = """Let's analyze the document image step by step to answer each question.

### 1. COLUMNS
The document contains a transaction table with the following column headers:
- Date
- Description
- Withdrawal
- Deposit
- Balance

So, the exact column headers separated by " | " are:
```
Date | Description | Withdrawal | Deposit | Balance
```

### 2. PAID
To determine if there is evidence that payment was completed, we need to look for:
- Payment method
- "PAID" stamp"""

# --- Terse, format-compliant control (already worked) -------------------------

CASE009_TERSE = (
    "1. COLUMNS: Date | Description | Withdrawal | Deposit | Balance\n\n2. PAID: NO\n\n3. ROWS: 37"
)


def _doc_type(raw: str) -> str | None:
    result = ClassificationParser()._parse_enriched(raw)
    return result["DOCUMENT_TYPE"] if result else None


class TestCotDriftRecovery:
    """CoT/markdown drift must still resolve bank statements correctly."""

    def test_case007_cot_heading_and_codefence_is_bank(self) -> None:
        assert _doc_type(CASE007_ANZ) == "BANK_STATEMENT"

    def test_case017_cot_withdrawal_deposit_is_bank(self) -> None:
        assert _doc_type(CASE017_CBA) == "BANK_STATEMENT"

    def test_terse_control_still_bank(self) -> None:
        assert _doc_type(CASE009_TERSE) == "BANK_STATEMENT"

    def test_markdown_bold_inline_is_bank(self) -> None:
        raw = "**COLUMNS:** Date | Description | Debit | Credit | Balance\n**PAID:** NO\n**ROWS:** 12"
        assert _doc_type(raw) == "BANK_STATEMENT"

    def test_heading_with_inline_value_is_bank(self) -> None:
        raw = "### 1. COLUMNS: Date | Description | Debit | Credit | Balance\n### 2. PAID: NO"
        assert _doc_type(raw) == "BANK_STATEMENT"

    def test_recovered_mapping_includes_bank_columns(self) -> None:
        result = ClassificationParser()._parse_enriched(CASE007_ANZ)
        assert result is not None
        mapping = result["column_mapping"]
        assert mapping["debit"] and mapping["credit"] and mapping["balance"]


class TestNoFalsePromotion:
    """The debit/credit/balance guard must keep non-bank docs out of BANK_STATEMENT."""

    def test_columns_none_with_payment_is_receipt(self) -> None:
        raw = "1. COLUMNS: NONE\n2. PAID: YES\n3. ROWS: 1"
        assert _doc_type(raw) == "RECEIPT"

    def test_columns_none_without_payment_is_invoice(self) -> None:
        # classification_evidence default is INVOICE: a doc with no columns and
        # no payment evidence matches no rule -> hard-defaults to INVOICE.
        raw = "1. COLUMNS: NONE\n2. PAID: NO\n3. ROWS: 4"
        assert _doc_type(raw) == "INVOICE"

    def test_non_bank_line_item_table_is_not_bank(self) -> None:
        # A receipt's line-item table (Qty/Price/Total) has a pipe table but no
        # debit/credit/balance columns — must NOT be promoted to BANK_STATEMENT.
        raw = (
            "### 1. COLUMNS\n"
            "The receipt line items are:\n"
            "- Qty\n- Description\n- Price\n- Total\n\n"
            "Qty | Description | Price | Total\n\n"
            "2. PAID: YES\n"
        )
        assert _doc_type(raw) == "RECEIPT"


class TestExpandedDocumentTypes:
    """YAML-driven evidence rules detect logbooks and unpaid invoices."""

    def test_logbook_columns_classify_logbook(self) -> None:
        raw = "1. COLUMNS: Date | Odometer | Distance | Purpose\n2. PAID: NO\n3. ROWS: 8"
        assert _doc_type(raw) == "LOGBOOK"

    def test_unpaid_itemised_table_is_invoice(self) -> None:
        raw = "1. COLUMNS: Description | Quantity | Unit Price | GST\n2. PAID: NO\n3. ROWS: 5"
        assert _doc_type(raw) == "INVOICE"

    def test_paid_itemised_table_stays_receipt(self) -> None:
        # Payment evidence wins over invoice columns (RECEIPT before INVOICE).
        raw = "1. COLUMNS: Description | Quantity | Unit Price | GST\n2. PAID: YES\n3. ROWS: 5"
        assert _doc_type(raw) == "RECEIPT"

    def test_logbook_columns_win_over_payment_order(self) -> None:
        # Logbook columns are checked before the paid->RECEIPT rule.
        raw = "1. COLUMNS: Date | Odometer | Purpose\n2. PAID: YES\n3. ROWS: 3"
        assert _doc_type(raw) == "LOGBOOK"


class TestNonEnrichedFallsToLegacy:
    """A response with no COLUMNS label must defer to legacy parsing (return None)."""

    def test_no_columns_label_returns_none(self) -> None:
        assert ClassificationParser()._parse_enriched("DOCUMENT_TYPE: INVOICE") is None


# --- Reasoning-model (InternVL3.5 thinking mode) <think> handling --------------
# Real raw_response shape captured from the thinking-mode re-run: the model
# reasons inside <think> and runs out of tokens before emitting the answer.

THINK_TRUNCATED_RECEIPT = (
    "<think>\nOkay, let's tackle these questions one by one. First, the columns. "
    "The document lists items with descriptions and prices, but there's no table "
    "with headers like Date, Description, Debit, etc. So the answer should be NONE.\n\n"
    "Next, PAID. The receipt doesn't mention a payment method"
)


class TestThinkingModeStripping:
    """<think> blocks must be stripped; reasoning prose must never classify."""

    def test_truncated_think_returns_none_not_bank(self) -> None:
        # The whole response is an unterminated <think> that merely *mentions*
        # "Debit" in prose. It must NOT be promoted to BANK_STATEMENT — earlier
        # whole-response harvesting did exactly that. Stripped -> no answer -> None.
        assert ClassificationParser()._parse_enriched(THINK_TRUNCATED_RECEIPT) is None

    def test_strip_think_removes_closed_block(self) -> None:
        stripped = ClassificationParser._strip_think("<think>reasoning here</think>\n1. COLUMNS: NONE")
        assert "reasoning" not in stripped
        assert "COLUMNS: NONE" in stripped

    def test_completed_think_bank_answer_is_bank(self) -> None:
        raw = (
            "<think>Let me look at the table and decide.</think>\n"
            "1. COLUMNS: Date | Description | Debit | Credit | Balance\n2. PAID: NO\n3. ROWS: 20"
        )
        assert _doc_type(raw) == "BANK_STATEMENT"

    def test_completed_think_none_answer_is_receipt(self) -> None:
        raw = "<think>It's a receipt, no transaction table.</think>\n1. COLUMNS: NONE\n2. PAID: YES\n3. ROWS: 3"
        assert _doc_type(raw) == "RECEIPT"

    def test_bank_columns_inside_think_are_ignored_when_answer_is_none(self) -> None:
        # Even when the <think> block contains a full pipe header line, only the
        # post-think answer counts — here NONE -> not a bank statement.
        raw = (
            "<think>The headers could be Date | Description | Debit | Credit | Balance.</think>\n"
            "1. COLUMNS: NONE\n2. PAID: YES\n3. ROWS: 2"
        )
        assert _doc_type(raw) == "RECEIPT"

    def test_completed_think_combined_debit_credit_header_is_bank(self) -> None:
        # Westpac-style combined "Debits/Credits (-)" column still maps to bank.
        raw = (
            "<think>Reasoning about the statement.</think>\n"
            "1. COLUMNS: Date of Transaction | Description | Debits/Credits (-)\n2. PAID: NO\n3. ROWS: 15"
        )
        assert _doc_type(raw) == "BANK_STATEMENT"


# --- Positional (label-free) answers ------------------------------------------
# Real raw_response payloads captured from the Gemma 4 12B W4A16 2xL4 run on
# 2026-08-11. The detection prompt NUMBERS its four questions but never tells the
# model to echo the label back; InternVL3.5 volunteers "COLUMNS:" etc., Gemma
# answers positionally. Every one of these previously fell through to the legacy
# keyword path and landed on the UNIVERSAL fallback, discarding correct evidence.


class TestPositionalAnswers:
    """Label-free numbered answers must parse from position."""

    def test_positional_bank_statement(self) -> None:
        raw = "1. Date | Description | Withdrawal | Deposit | Balance\n2. YES\n3. 33\n4. NO"
        assert _doc_type(raw) == "BANK_STATEMENT"

    def test_positional_westpac_combined_column_is_bank(self) -> None:
        raw = "1. Date of Transaction | Description | Debits | Credits (-)\n2. YES\n3. 38\n4. NO"
        assert _doc_type(raw) == "BANK_STATEMENT"

    def test_positional_unpaid_line_item_table_is_invoice(self) -> None:
        raw = "1. Description | Qty | Unit Price | Total\n2. NO\n3. 6\n4. NO"
        assert _doc_type(raw) == "INVOICE"

    def test_positional_none_columns_paid_is_receipt(self) -> None:
        raw = "1. NONE\n2. YES\n3. 2\n4. NO"
        assert _doc_type(raw) == "RECEIPT"

    def test_positional_row_count_is_captured(self) -> None:
        raw = "1. Date | Particulars | Debits | Credits | Balance\n2. YES\n3. 21\n4. NO"
        result = ClassificationParser()._parse_enriched(raw)
        assert result is not None
        assert result["row_count"] == 21
        assert result["payment_evidence"] is True
        assert result["travel_evidence"] is False

    def test_positional_travel_yes_is_travel(self) -> None:
        raw = "1. NONE\n2. YES\n3. 1\n4. YES"
        assert _doc_type(raw) == "TRAVEL"

    def test_labelled_form_still_wins(self) -> None:
        # The labelled path must be unchanged — positional is a fallback only.
        raw = "1. COLUMNS: NONE\n2. PAID: YES\n3. ROWS: 3\n4. TRAVEL: NO"
        assert _doc_type(raw) == "RECEIPT"


class TestPositionalFalsePositiveGuards:
    """The positional path must not fire on prose that merely looks numbered."""

    def test_missing_fourth_answer_does_not_parse(self) -> None:
        # Partial numbering is not the agreed shape — defer to legacy.
        assert ClassificationParser()._parse_enriched("1. NONE\n2. YES\n3. 2") is None

    def test_non_yesno_second_answer_does_not_parse(self) -> None:
        raw = "1. Some heading\n2. Maybe, it is unclear\n3. Several\n4. Possibly"
        assert ClassificationParser()._parse_enriched(raw) is None

    def test_non_numeric_row_count_does_not_parse(self) -> None:
        raw = "1. NONE\n2. YES\n3. several\n4. NO"
        assert ClassificationParser()._parse_enriched(raw) is None

    def test_numbered_prose_does_not_parse(self) -> None:
        raw = (
            "1. First, I looked at the document carefully.\n"
            "2. Then I considered the layout of the page.\n"
            "3. After that I reviewed the totals.\n"
            "4. Finally I reached my conclusion."
        )
        assert ClassificationParser()._parse_enriched(raw) is None

    def test_think_only_response_still_returns_none(self) -> None:
        # Regression guard: stripping <think> leaves nothing, so nothing to parse.
        assert ClassificationParser()._parse_enriched(THINK_TRUNCATED_RECEIPT) is None
