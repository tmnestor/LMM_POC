"""GPU-free tests for the robust probe-based classification graph workflow.

Tests the full robust_extract.yaml workflow with mock generate_fn,
covering probe scoring, type selection, state cleanup, and bank subgraph
routing.
"""

from pathlib import Path
from unittest.mock import MagicMock

import yaml
from PIL import Image

from common.bank_post_process import _normalize_doc_type, run_select_best_type
from common.extraction_types import GenerateResult, NodeResult, WorkflowState
from common.graph_executor import GraphExecutor
from common.turn_parsers import build_parser_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "workflows"
    / "robust_extract.yaml"
)


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as f:
        return yaml.safe_load(f)


def _make_test_image(tmp_path: Path, name: str = "test.png") -> str:
    img = Image.new("RGB", (10, 10), color="white")
    path = tmp_path / name
    img.save(path)
    return str(path)


def _mock_generate_fn(responses: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.side_effect = [GenerateResult(text=r) for r in responses]
    return mock


# ---------------------------------------------------------------------------
# Canned model responses
# ---------------------------------------------------------------------------

# Receipt probe: ~10 fields extracted, model says RECEIPT
PROBE_RECEIPT = """\
DOCUMENT_TYPE: RECEIPT
BUSINESS_ABN: 12 345 678 901
SUPPLIER_NAME: Corner Store
BUSINESS_ADDRESS: 123 Main St
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: 15/03/2025
LINE_ITEM_DESCRIPTIONS: Coffee | Sandwich
LINE_ITEM_QUANTITIES: 1 | 1
LINE_ITEM_PRICES: $4.50 | $8.00
LINE_ITEM_TOTAL_PRICES: $4.50 | $8.00
IS_GST_INCLUDED: true
GST_AMOUNT: $1.14
TOTAL_AMOUNT: $12.50"""

# Invoice probe: ~10 fields extracted, model says INVOICE
PROBE_INVOICE = """\
DOCUMENT_TYPE: INVOICE
BUSINESS_ABN: 98 765 432 109
SUPPLIER_NAME: Acme Services
BUSINESS_ADDRESS: 456 Business Ave
PAYER_NAME: John Smith
PAYER_ADDRESS: 789 Client Rd
INVOICE_DATE: 01/04/2025
LINE_ITEM_DESCRIPTIONS: Consulting | Report
LINE_ITEM_QUANTITIES: 2 | 1
LINE_ITEM_PRICES: $150.00 | $200.00
LINE_ITEM_TOTAL_PRICES: $300.00 | $200.00
IS_GST_INCLUDED: true
GST_AMOUNT: $50.00
TOTAL_AMOUNT: $550.00"""

# Bank statement probe: very few fields extracted (bank image is a table, not receipt)
PROBE_BANK_DOC = """\
DOCUMENT_TYPE: BANK_STATEMENT
BUSINESS_ABN: NOT_FOUND
SUPPLIER_NAME: NOT_FOUND
BUSINESS_ADDRESS: NOT_FOUND
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: NOT_FOUND
LINE_ITEM_DESCRIPTIONS: NOT_FOUND
LINE_ITEM_QUANTITIES: NOT_FOUND
LINE_ITEM_PRICES: NOT_FOUND
LINE_ITEM_TOTAL_PRICES: NOT_FOUND
IS_GST_INCLUDED: NOT_FOUND
GST_AMOUNT: NOT_FOUND
TOTAL_AMOUNT: NOT_FOUND"""

# Probe with DOCUMENT_TYPE=NOT_FOUND (ambiguous doc)
PROBE_AMBIGUOUS = """\
DOCUMENT_TYPE: NOT_FOUND
BUSINESS_ABN: 12 345 678 901
SUPPLIER_NAME: Some Business
BUSINESS_ADDRESS: 42 Elm St
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: 10/01/2025
LINE_ITEM_DESCRIPTIONS: Widget
LINE_ITEM_QUANTITIES: 1
LINE_ITEM_PRICES: $99.00
LINE_ITEM_TOTAL_PRICES: $99.00
IS_GST_INCLUDED: NOT_FOUND
GST_AMOUNT: NOT_FOUND
TOTAL_AMOUNT: $99.00"""

# Bank headers: typical 5-column layout (high bank score)
HEADERS_BALANCE = """\
1. Date
2. Description
3. Debit
4. Credit
5. Balance"""

# Bank headers: non-bank garbage (will trigger parse error)
HEADERS_GARBAGE = "This is just a receipt, no table headers."

# Balance extraction response
BALANCE_RESPONSE = """\
1. **03/01/2025**
   - Description: Grocery Store Purchase
   - Debit: $45.50
   - Credit: NOT_FOUND
   - Balance: $1,954.50

2. **05/01/2025**
   - Description: Salary Deposit
   - Debit: NOT_FOUND
   - Credit: $3,000.00
   - Balance: $4,954.50"""


# ---------------------------------------------------------------------------
# select_best_type validator unit tests
# ---------------------------------------------------------------------------


class TestNormalizeDocType:
    """Test _normalize_doc_type() covers model output variants."""

    def test_canonical_passthrough(self) -> None:
        assert _normalize_doc_type("RECEIPT") == "RECEIPT"
        assert _normalize_doc_type("INVOICE") == "INVOICE"
        assert _normalize_doc_type("BANK_STATEMENT") == "BANK_STATEMENT"

    def test_tax_invoice_variants(self) -> None:
        assert _normalize_doc_type("TAX INVOICE") == "INVOICE"
        assert _normalize_doc_type("Tax Invoice") == "INVOICE"
        assert _normalize_doc_type("tax invoice") == "INVOICE"

    def test_not_found_defaults_to_receipt(self) -> None:
        assert _normalize_doc_type("NOT_FOUND") == "RECEIPT"

    def test_unknown_defaults_to_receipt(self) -> None:
        assert _normalize_doc_type("TRAVEL_EXPENSE") == "RECEIPT"

    def test_bank_statement_aliases(self) -> None:
        assert _normalize_doc_type("credit card statement") == "BANK_STATEMENT"
        assert _normalize_doc_type("Bank Statement") == "BANK_STATEMENT"


class TestSelectBestType:
    """Test run_select_best_type() directly with synthetic state."""

    def _make_state(
        self,
        doc_parsed: dict,
        bank_parsed: dict | None = None,
    ) -> WorkflowState:
        state = WorkflowState()
        state.node_results["probe_document"] = NodeResult(
            key="probe_document",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed=doc_parsed,
            elapsed=0.0,
            attempt=1,
            edge_taken="ok",
        )
        if bank_parsed is not None:
            state.node_results["probe_bank_headers"] = NodeResult(
                key="probe_bank_headers",
                image_ref="primary",
                prompt_sent="",
                raw_response="",
                parsed=bank_parsed,
                elapsed=0.0,
                attempt=1,
                edge_taken="ok",
            )
        return state

    def test_receipt_wins_high_doc_score(self) -> None:
        """Receipt with many extracted fields beats empty bank probe."""
        doc = {
            "DOCUMENT_TYPE": "RECEIPT",
            "SUPPLIER_NAME": "Corner Store",
            "TOTAL_AMOUNT": "$12.50",
            "INVOICE_DATE": "15/03/2025",
            "BUSINESS_ABN": "12 345 678 901",
            "LINE_ITEM_DESCRIPTIONS": "Coffee",
            "GST_AMOUNT": "$1.14",
            "IS_GST_INCLUDED": "true",
        }
        bank = {"error": "ParseError", "raw": "garbage"}
        state = self._make_state(doc, bank)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "RECEIPT"
        # Bank probe should be removed
        assert "probe_bank_headers" not in state.node_results
        # Document probe preserved
        assert "probe_document" in state.node_results

    def test_invoice_wins(self) -> None:
        """Invoice type from doc probe, no bank columns."""
        doc = {
            "DOCUMENT_TYPE": "INVOICE",
            "SUPPLIER_NAME": "Acme",
            "TOTAL_AMOUNT": "$550.00",
            "INVOICE_DATE": "01/04/2025",
            "PAYER_NAME": "Client",
            "LINE_ITEM_DESCRIPTIONS": "Consulting",
            "GST_AMOUNT": "$50.00",
        }
        bank = {"column_mapping": {"date": None, "description": None}}
        state = self._make_state(doc, bank)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "INVOICE"

    def test_bank_wins_high_bank_score_low_doc(self) -> None:
        """Bank wins when >=3 columns and doc score < 6."""
        doc = {
            "DOCUMENT_TYPE": "BANK_STATEMENT",
            "SUPPLIER_NAME": "NOT_FOUND",
            "TOTAL_AMOUNT": "NOT_FOUND",
        }
        bank = {
            "column_mapping": {
                "date": "Date",
                "description": "Description",
                "debit": "Debit",
                "credit": "Credit",
                "balance": "Balance",
            },
            "headers": ["Date", "Description", "Debit", "Credit", "Balance"],
        }
        state = self._make_state(doc, bank)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "BANK_STATEMENT"
        # Document probe removed, bank renamed to detect_headers
        assert "probe_document" not in state.node_results
        assert "probe_bank_headers" not in state.node_results
        assert "detect_headers" in state.node_results

    def test_doc_type_not_found_falls_back_to_receipt(self) -> None:
        """When doc probe says NOT_FOUND and bank score is low, default to RECEIPT."""
        doc = {
            "DOCUMENT_TYPE": "NOT_FOUND",
            "SUPPLIER_NAME": "Some Shop",
            "TOTAL_AMOUNT": "$99.00",
            "INVOICE_DATE": "10/01/2025",
            "BUSINESS_ABN": "12 345 678 901",
            "LINE_ITEM_DESCRIPTIONS": "Widget",
            "GST_AMOUNT": "NOT_FOUND",
        }
        state = self._make_state(doc, None)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "RECEIPT"

    def test_payment_date_overrides_invoice_to_receipt(self) -> None:
        """PAYMENT_DATE present -> receipt wins even if model says INVOICE."""
        doc = {
            "DOCUMENT_TYPE": "INVOICE",
            "SUPPLIER_NAME": "Store",
            "TOTAL_AMOUNT": "$50.00",
            "INVOICE_DATE": "01/01/2025",
            "PAYMENT_DATE": "01/01/2025",
            "LINE_ITEM_DESCRIPTIONS": "Item",
            "GST_AMOUNT": "$5.00",
            "IS_GST_INCLUDED": "true",
        }
        state = self._make_state(doc, None)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "RECEIPT"

    def test_bank_parse_error_handled(self) -> None:
        """Bank probe with parse error (error dict) -> bank_score = 0."""
        doc = {
            "DOCUMENT_TYPE": "RECEIPT",
            "SUPPLIER_NAME": "Store",
            "TOTAL_AMOUNT": "$10.00",
            "INVOICE_DATE": "01/01/2025",
            "LINE_ITEM_DESCRIPTIONS": "Item",
            "GST_AMOUNT": "$1.00",
            "IS_GST_INCLUDED": "true",
        }
        bank = {"error": "Could not parse headers", "raw": "no table here"}
        state = self._make_state(doc, bank)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "RECEIPT"


# ---------------------------------------------------------------------------
# Full workflow: receipt path
# ---------------------------------------------------------------------------


class TestReceiptPath:
    """probe_document -> probe_bank_headers (fail) -> select_best_type -> receipt -> done."""

    def test_receipt_full_path(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([PROBE_RECEIPT, HEADERS_GARBAGE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "RECEIPT"
        assert session.final_fields["SUPPLIER_NAME"] == "Corner Store"
        assert session.final_fields["TOTAL_AMOUNT"] == "$12.50"
        assert session.trace.total_model_calls == 2
        nodes = session.trace.nodes_visited
        assert "probe_document" in nodes
        assert "probe_bank_headers" in nodes
        assert "select_best_type" in nodes
        assert "route_best_type" in nodes

    def test_receipt_fields_in_final_output(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([PROBE_RECEIPT, HEADERS_GARBAGE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.final_fields["DOCUMENT_TYPE"] == "RECEIPT"
        assert "BUSINESS_ABN" in session.final_fields
        assert session.final_fields["INVOICE_DATE"] == "15/03/2025"


# ---------------------------------------------------------------------------
# Full workflow: invoice path
# ---------------------------------------------------------------------------


class TestInvoicePath:
    """probe_document -> probe_bank_headers (fail) -> select_best_type -> invoice -> done."""

    def test_invoice_full_path(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([PROBE_INVOICE, HEADERS_GARBAGE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "INVOICE"
        assert session.final_fields["SUPPLIER_NAME"] == "Acme Services"
        assert session.final_fields["TOTAL_AMOUNT"] == "$550.00"


# ---------------------------------------------------------------------------
# Full workflow: bank statement path
# ---------------------------------------------------------------------------


class TestBankPath:
    """probe_document -> probe_bank_headers -> select_best_type -> bank subgraph."""

    def test_bank_full_path(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([PROBE_BANK_DOC, HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "BANK_STATEMENT"
        nodes = session.trace.nodes_visited
        assert "select_best_type" in nodes
        assert "select_bank_strategy" in nodes
        assert "extract_balance" in nodes
        assert "bank_post_process" in nodes
        # 3 model calls: probe_document + probe_bank_headers + extract_balance
        assert session.trace.total_model_calls == 3

    def test_bank_produces_pipe_delimited_fields(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([PROBE_BANK_DOC, HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.final_fields["DOCUMENT_TYPE"] == "BANK_STATEMENT"
        assert "TRANSACTION_DATES" in session.final_fields
        assert "LINE_ITEM_DESCRIPTIONS" in session.final_fields
        assert "TRANSACTION_AMOUNTS_PAID" in session.final_fields

    def test_bank_probe_fields_not_in_final(self, tmp_path: Path) -> None:
        """Document probe fields should NOT leak into bank final output."""
        gen_fn = _mock_generate_fn([PROBE_BANK_DOC, HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # These are receipt/invoice fields that should NOT appear in bank output
        assert "SUPPLIER_NAME" not in session.final_fields
        assert "TOTAL_AMOUNT" not in session.final_fields
        assert "GST_AMOUNT" not in session.final_fields


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_ambiguous_doc_type_falls_back_to_receipt(self, tmp_path: Path) -> None:
        """DOCUMENT_TYPE=NOT_FOUND from probe -> falls back to RECEIPT."""
        gen_fn = _mock_generate_fn([PROBE_AMBIGUOUS, HEADERS_GARBAGE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "RECEIPT"
        assert session.final_fields["SUPPLIER_NAME"] == "Some Business"

    def test_serialization_to_record(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([PROBE_RECEIPT, HEADERS_GARBAGE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        record = session.to_record()
        assert record["document_type"] == "RECEIPT"
        assert "SUPPLIER_NAME: Corner Store" in record["raw_response"]
        assert record["error"] is None
