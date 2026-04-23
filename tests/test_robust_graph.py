"""GPU-free tests for the robust probe-based classification graph workflow.

Tests the full robust_extract.yaml workflow with mock generate_fn,
covering probe scoring, type selection, state cleanup, bank subgraph
routing, and the new travel/logbook extraction paths.
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

# Travel probe: few receipt fields, model says TRAVEL
PROBE_TRAVEL = """\
DOCUMENT_TYPE: TRAVEL
BUSINESS_ABN: NOT_FOUND
SUPPLIER_NAME: Qantas Airways
BUSINESS_ADDRESS: NOT_FOUND
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: 10/02/2026
LINE_ITEM_DESCRIPTIONS: NOT_FOUND
LINE_ITEM_QUANTITIES: NOT_FOUND
LINE_ITEM_PRICES: NOT_FOUND
LINE_ITEM_TOTAL_PRICES: NOT_FOUND
IS_GST_INCLUDED: NOT_FOUND
GST_AMOUNT: $28.54
TOTAL_AMOUNT: $314.00"""

# Logbook probe: very few receipt fields, model says LOGBOOK
PROBE_LOGBOOK = """\
DOCUMENT_TYPE: LOGBOOK
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

# Travel extraction response (from extract_travel node)
EXTRACT_TRAVEL_RESPONSE = """\
DOCUMENT_TYPE: TRAVEL
PASSENGER_NAME: Martin/Olivia
TRAVEL_MODE: plane
TRAVEL_ROUTE: Sydney | Hobart
TRAVEL_DATES: 16 Feb 2026 | 18 Feb 2026
INVOICE_DATE: 10 Feb 2026
GST_AMOUNT: $28.54
TOTAL_AMOUNT: $314.00
SUPPLIER_NAME: Qantas Airways"""

# Logbook extraction response (from extract_logbook node)
EXTRACT_LOGBOOK_RESPONSE = """\
DOCUMENT_TYPE: LOGBOOK
VEHICLE_MAKE: Toyota
VEHICLE_MODEL: Camry
VEHICLE_REGISTRATION: FD17PQ
ENGINE_CAPACITY: 2.5L
LOGBOOK_PERIOD_START: 01 Jan 2025
LOGBOOK_PERIOD_END: 25 Mar 2025
ODOMETER_START: 17811
ODOMETER_END: 20800
TOTAL_KILOMETERS: 2989
BUSINESS_KILOMETERS: 2328
BUSINESS_USE_PERCENTAGE: 78%
JOURNEY_DATES: 02 Jan 2025 | 05 Jan 2025 | 08 Jan 2025
JOURNEY_DISTANCES: 8 | 94 | 82
JOURNEY_PURPOSES: Project site | Warehouse pickup | Delivery"""


# ---------------------------------------------------------------------------
# select_best_type validator unit tests
# ---------------------------------------------------------------------------


class TestNormalizeDocType:
    """Test _normalize_doc_type() covers model output variants."""

    def test_canonical_passthrough(self) -> None:
        assert _normalize_doc_type("RECEIPT") == "RECEIPT"
        assert _normalize_doc_type("INVOICE") == "INVOICE"
        assert _normalize_doc_type("BANK_STATEMENT") == "BANK_STATEMENT"
        assert _normalize_doc_type("TRAVEL") == "TRAVEL"
        assert _normalize_doc_type("LOGBOOK") == "LOGBOOK"

    def test_tax_invoice_variants(self) -> None:
        assert _normalize_doc_type("TAX INVOICE") == "INVOICE"
        assert _normalize_doc_type("Tax Invoice") == "INVOICE"
        assert _normalize_doc_type("tax invoice") == "INVOICE"

    def test_not_found_defaults_to_receipt(self) -> None:
        assert _normalize_doc_type("NOT_FOUND") == "RECEIPT"

    def test_unknown_defaults_to_receipt(self) -> None:
        assert _normalize_doc_type("SOME_RANDOM_TYPE") == "RECEIPT"

    def test_bank_statement_aliases(self) -> None:
        assert _normalize_doc_type("credit card statement") == "BANK_STATEMENT"
        assert _normalize_doc_type("Bank Statement") == "BANK_STATEMENT"

    def test_travel_aliases(self) -> None:
        assert _normalize_doc_type("itinerary") == "TRAVEL"
        assert _normalize_doc_type("boarding pass") == "TRAVEL"
        assert _normalize_doc_type("flight ticket") == "TRAVEL"
        assert _normalize_doc_type("airline ticket") == "TRAVEL"
        assert _normalize_doc_type("e-ticket") == "TRAVEL"
        assert _normalize_doc_type("travel expense") == "TRAVEL"

    def test_logbook_aliases(self) -> None:
        assert _normalize_doc_type("vehicle logbook") == "LOGBOOK"
        assert _normalize_doc_type("vehicle_logbook") == "LOGBOOK"
        assert _normalize_doc_type("mileage log") == "LOGBOOK"
        assert _normalize_doc_type("motor vehicle logbook") == "LOGBOOK"


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

    def test_travel_type_passthrough(self) -> None:
        """Model says TRAVEL -> normaliser returns TRAVEL."""
        doc = {
            "DOCUMENT_TYPE": "TRAVEL",
            "SUPPLIER_NAME": "Qantas",
            "TOTAL_AMOUNT": "$314.00",
            "GST_AMOUNT": "$28.54",
            "INVOICE_DATE": "10/02/2026",
        }
        state = self._make_state(doc, None)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "TRAVEL"

    def test_logbook_type_passthrough(self) -> None:
        """Model says LOGBOOK -> normaliser returns LOGBOOK."""
        doc = {
            "DOCUMENT_TYPE": "LOGBOOK",
            "SUPPLIER_NAME": "NOT_FOUND",
            "TOTAL_AMOUNT": "NOT_FOUND",
        }
        state = self._make_state(doc, None)

        ok, parsed = run_select_best_type(state)
        assert ok is True
        assert parsed["DOCUMENT_TYPE"] == "LOGBOOK"


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
# Full workflow: travel path
# ---------------------------------------------------------------------------


class TestTravelPath:
    """probe_document -> probe_bank_headers (fail) -> select_best_type -> travel -> extract_travel -> done."""

    def test_travel_full_path(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn(
            [PROBE_TRAVEL, HEADERS_GARBAGE, EXTRACT_TRAVEL_RESPONSE]
        )
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "TRAVEL"
        assert session.trace.total_model_calls == 3
        nodes = session.trace.nodes_visited
        assert "probe_document" in nodes
        assert "select_best_type" in nodes
        assert "route_best_type" in nodes
        assert "extract_travel" in nodes

    def test_travel_fields_in_final_output(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn(
            [PROBE_TRAVEL, HEADERS_GARBAGE, EXTRACT_TRAVEL_RESPONSE]
        )
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.final_fields["DOCUMENT_TYPE"] == "TRAVEL"
        assert session.final_fields["PASSENGER_NAME"] == "Martin/Olivia"
        assert session.final_fields["TRAVEL_ROUTE"] == "Sydney | Hobart"
        assert session.final_fields["SUPPLIER_NAME"] == "Qantas Airways"
        assert session.final_fields["TOTAL_AMOUNT"] == "$314.00"

    def test_travel_probe_fields_replaced_by_extraction(self, tmp_path: Path) -> None:
        """extract_travel should provide the authoritative fields, not probe_document."""
        gen_fn = _mock_generate_fn(
            [PROBE_TRAVEL, HEADERS_GARBAGE, EXTRACT_TRAVEL_RESPONSE]
        )
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # extract_travel should have PASSENGER_NAME (not from probe)
        assert "PASSENGER_NAME" in session.final_fields
        assert session.final_fields["TRAVEL_MODE"] == "plane"


# ---------------------------------------------------------------------------
# Full workflow: logbook path
# ---------------------------------------------------------------------------


class TestLogbookPath:
    """probe_document -> probe_bank_headers (fail) -> select_best_type -> logbook -> extract_logbook -> done."""

    def test_logbook_full_path(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn(
            [PROBE_LOGBOOK, HEADERS_GARBAGE, EXTRACT_LOGBOOK_RESPONSE]
        )
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.document_type == "LOGBOOK"
        assert session.trace.total_model_calls == 3
        nodes = session.trace.nodes_visited
        assert "probe_document" in nodes
        assert "select_best_type" in nodes
        assert "route_best_type" in nodes
        assert "extract_logbook" in nodes

    def test_logbook_fields_in_final_output(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn(
            [PROBE_LOGBOOK, HEADERS_GARBAGE, EXTRACT_LOGBOOK_RESPONSE]
        )
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert session.final_fields["DOCUMENT_TYPE"] == "LOGBOOK"
        assert session.final_fields["VEHICLE_MAKE"] == "Toyota"
        assert session.final_fields["VEHICLE_MODEL"] == "Camry"
        assert session.final_fields["VEHICLE_REGISTRATION"] == "FD17PQ"
        assert session.final_fields["ENGINE_CAPACITY"] == "2.5L"
        assert session.final_fields["BUSINESS_USE_PERCENTAGE"] == "78%"
        assert "JOURNEY_DATES" in session.final_fields
        assert "JOURNEY_DISTANCES" in session.final_fields
        assert "JOURNEY_PURPOSES" in session.final_fields

    def test_logbook_probe_fields_not_leaked(self, tmp_path: Path) -> None:
        """Receipt fields from probe_document should NOT appear in logbook output."""
        gen_fn = _mock_generate_fn(
            [PROBE_LOGBOOK, HEADERS_GARBAGE, EXTRACT_LOGBOOK_RESPONSE]
        )
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="UNKNOWN",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # Probe had LINE_ITEM_DESCRIPTIONS: NOT_FOUND etc. -- these are from the
        # probe template and should still be in final_fields (they're legitimate
        # parsed fields). But the logbook-specific fields should dominate.
        assert session.final_fields["VEHICLE_MAKE"] == "Toyota"


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
