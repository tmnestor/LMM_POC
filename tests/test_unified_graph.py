"""GPU-free tests for the unified classify + extract graph workflow.

Tests the full unified_extract.yaml workflow with mock generate_fn,
covering classification routing, receipt/invoice/bank extraction paths,
and fallback behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock

import yaml
from PIL import Image

from common.extraction_types import GenerateResult, WorkflowState
from common.graph_executor import GraphExecutor
from common.turn_parsers import (
    ClassificationParser,
    _match_document_type,
    build_parser_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "workflows"
    / "unified_extract.yaml"
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

CLASSIFY_RECEIPT = "RECEIPT"
CLASSIFY_INVOICE = "INVOICE"
CLASSIFY_BANK = "BANK_STATEMENT"
# Must not contain any type_mapping or fallback_keyword matches
CLASSIFY_GARBAGE = "xyzzy plugh"

RECEIPT_RESPONSE = """\
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

INVOICE_RESPONSE = """\
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

# Bank responses (reuse patterns from test_bank_graph.py)
HEADERS_BALANCE = """\
1. Date
2. Description
3. Debit
4. Credit
5. Balance"""

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
# ClassificationParser unit tests
# ---------------------------------------------------------------------------


class TestClassificationParser:
    """Test ClassificationParser and _match_document_type()."""

    def test_parses_receipt(self) -> None:
        parser = ClassificationParser()
        state = WorkflowState()
        result = parser.parse("RECEIPT", state)
        assert result["DOCUMENT_TYPE"] == "RECEIPT"

    def test_parses_invoice(self) -> None:
        parser = ClassificationParser()
        state = WorkflowState()
        result = parser.parse("invoice", state)
        assert result["DOCUMENT_TYPE"] == "INVOICE"

    def test_parses_bank_statement(self) -> None:
        parser = ClassificationParser()
        state = WorkflowState()
        result = parser.parse("BANK_STATEMENT", state)
        assert result["DOCUMENT_TYPE"] == "BANK_STATEMENT"

    def test_parses_bank_statement_variant(self) -> None:
        parser = ClassificationParser()
        state = WorkflowState()
        result = parser.parse("This is a credit card statement", state)
        assert result["DOCUMENT_TYPE"] == "BANK_STATEMENT"

    def test_preserves_raw_response(self) -> None:
        parser = ClassificationParser()
        state = WorkflowState()
        result = parser.parse("RECEIPT", state)
        assert result["_raw_classification"] == "RECEIPT"


class TestMatchDocumentType:
    """Test _match_document_type() standalone function."""

    def test_type_mapping_match(self) -> None:
        mappings = {"invoice": "INVOICE", "receipt": "RECEIPT"}
        assert _match_document_type("invoice", mappings, {}, "RECEIPT") == "INVOICE"

    def test_type_mapping_case_insensitive(self) -> None:
        mappings = {"invoice": "INVOICE"}
        assert _match_document_type("INVOICE", mappings, {}, "RECEIPT") == "INVOICE"

    def test_fallback_keyword_match(self) -> None:
        mappings: dict[str, str] = {}
        keywords = {"BANK_STATEMENT": ["bank", "statement"]}
        result = _match_document_type(
            "this is a bank document", mappings, keywords, "RECEIPT"
        )
        assert result == "BANK_STATEMENT"

    def test_ultimate_fallback(self) -> None:
        result = _match_document_type("xyzzy", {}, {}, "RECEIPT")
        assert result == "RECEIPT"


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestDocumentTypeRouting:
    """Test that the router dispatches correctly by document type."""

    def test_routes_receipt(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_RECEIPT, RECEIPT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "route_by_type"]
        assert router_edge[0][1] == "is_receipt"

    def test_routes_invoice(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_INVOICE, INVOICE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "route_by_type"]
        assert router_edge[0][1] == "is_invoice"

    def test_routes_bank_statement(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_BANK, HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "route_by_type"]
        assert router_edge[0][1] == "is_bank_statement"

    def test_garbage_classification_falls_back_to_receipt(self, tmp_path: Path) -> None:
        """Garbage text -> parser fallback_type RECEIPT -> is_receipt edge."""
        gen_fn = _mock_generate_fn([CLASSIFY_GARBAGE, RECEIPT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # Parser's fallback_type resolves garbage to RECEIPT, so router
        # takes is_receipt edge (not default)
        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "route_by_type"]
        assert router_edge[0][1] == "is_receipt"
        assert session.document_type == "RECEIPT"

    def test_unknown_type_routes_to_default(self, tmp_path: Path) -> None:
        """A type with no matching is_* edge falls to default."""
        # TRAVEL_EXPENSE has no is_travel_expense edge in the workflow
        gen_fn = _mock_generate_fn(["travel expense", RECEIPT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "route_by_type"]
        assert router_edge[0][1] == "default"
        # Falls back to extract_receipt via default edge
        assert "extract_receipt" in session.trace.nodes_visited


# ---------------------------------------------------------------------------
# End-to-end extraction path tests
# ---------------------------------------------------------------------------


class TestReceiptPath:
    """Test classify -> extract_receipt -> done."""

    def test_extracts_receipt_fields(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_RECEIPT, RECEIPT_RESPONSE])
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
        assert "classify_document" in session.trace.nodes_visited
        assert "extract_receipt" in session.trace.nodes_visited
        assert session.trace.total_model_calls == 2

    def test_document_type_overrides_input(self, tmp_path: Path) -> None:
        """document_type='UNKNOWN' is replaced by classified type."""
        gen_fn = _mock_generate_fn([CLASSIFY_RECEIPT, RECEIPT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "RECEIPT"
        assert session.final_fields["DOCUMENT_TYPE"] == "RECEIPT"


class TestInvoicePath:
    """Test classify -> extract_invoice -> done."""

    def test_extracts_invoice_fields(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_INVOICE, INVOICE_RESPONSE])
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
        assert "extract_invoice" in session.trace.nodes_visited


class TestBankPath:
    """Test classify -> bank subgraph -> done."""

    def test_bank_goes_through_full_subgraph(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_BANK, HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "BANK_STATEMENT"
        nodes = session.trace.nodes_visited
        assert "classify_document" in nodes
        assert "detect_headers" in nodes
        assert "select_bank_strategy" in nodes
        assert "extract_balance" in nodes
        assert "bank_post_process" in nodes
        assert session.trace.total_model_calls == 3

    def test_bank_produces_pipe_delimited_fields(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_BANK, HEADERS_BALANCE, BALANCE_RESPONSE])
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


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test session.to_record() produces valid output."""

    def test_receipt_to_record(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_RECEIPT, RECEIPT_RESPONSE])
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
        assert record["processing_time"] >= 0

    def test_bank_to_record(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([CLASSIFY_BANK, HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        record = session.to_record()
        assert record["document_type"] == "BANK_STATEMENT"
        assert "TRANSACTION_DATES" in record["raw_response"]
