"""GPU-free tests for bank statement graph workflow.

Tests the full bank_extract.yaml workflow with mock generate_fn,
covering all four strategy paths (balance, amount, debit/credit,
schema fallback) and post-processing.
"""

from pathlib import Path
from unittest.mock import MagicMock

import yaml
from PIL import Image

from common.extraction_types import GenerateResult, WorkflowState
from common.graph_executor import GraphExecutor
from common.turn_parsers import build_parser_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "workflows"
    / "bank_extract.yaml"
)


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as f:
        return yaml.safe_load(f)


def _make_test_image(tmp_path: Path, name: str = "bank.png") -> str:
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

HEADERS_BALANCE = """\
1. Date
2. Description
3. Debit
4. Credit
5. Balance"""

HEADERS_AMOUNT = """\
1. Date
2. Description
3. Amount
4. Balance"""

HEADERS_DEBIT_CREDIT = """\
1. Date
2. Transaction Details
3. Withdrawals
4. Deposits"""

HEADERS_GARBAGE = """\
Monday
Tuesday
ATM Location
Reference Number"""

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
   - Balance: $4,954.50

3. **10/01/2025**
   - Description: Electricity Bill Payment
   - Debit: $120.00
   - Credit: NOT_FOUND
   - Balance: $4,834.50"""

AMOUNT_RESPONSE = """\
1. **03/01/2025**
   - Description: Grocery Store Purchase
   - Amount: -$45.50
   - Balance: $1,954.50

2. **05/01/2025**
   - Description: Salary Deposit
   - Amount: $3,000.00
   - Balance: $4,954.50

3. **10/01/2025**
   - Description: Electricity Bill Payment
   - Amount: -$120.00
   - Balance: $4,834.50"""

DEBIT_CREDIT_RESPONSE = """\
1. **03/01/2025**
   - Transaction Details: Grocery Store Purchase
   - Withdrawals: $45.50
   - Deposits: NOT_FOUND

2. **05/01/2025**
   - Transaction Details: Salary Deposit
   - Withdrawals: NOT_FOUND
   - Deposits: $3,000.00

3. **10/01/2025**
   - Transaction Details: Electricity Bill Payment
   - Withdrawals: $120.00
   - Deposits: NOT_FOUND"""

SCHEMA_FALLBACK_RESPONSE = """\
DOCUMENT_TYPE: BANK_STATEMENT
STATEMENT_DATE_RANGE: 01/01/2025 - 31/01/2025
TRANSACTION_DATES: 03/01/2025 | 10/01/2025
LINE_ITEM_DESCRIPTIONS: Grocery Store Purchase | Electricity Bill Payment
TRANSACTION_AMOUNTS_PAID: $45.50 | $120.00"""


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestRouterSelection:
    """Test that the router selects the correct strategy based on columns."""

    def test_routes_to_balance_when_balance_and_debit(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        assert "select_strategy" in session.trace.nodes_visited
        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "select_strategy"]
        assert router_edge[0][1] == "has_balance_debit"

    def test_routes_to_amount_when_amount_and_balance(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_AMOUNT, AMOUNT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "select_strategy"]
        assert router_edge[0][1] == "has_amount"

    def test_routes_to_debit_credit_when_no_balance(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_DEBIT_CREDIT, DEBIT_CREDIT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "select_strategy"]
        assert router_edge[0][1] == "has_debit_or_credit"

    def test_routes_to_schema_fallback_when_garbage_headers(
        self, tmp_path: Path
    ) -> None:
        gen_fn = _mock_generate_fn([HEADERS_GARBAGE, SCHEMA_FALLBACK_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        edges = session.trace.edges_taken
        router_edge = [e for e in edges if e[0] == "select_strategy"]
        assert router_edge[0][1] == "default"


# ---------------------------------------------------------------------------
# Strategy path tests
# ---------------------------------------------------------------------------


class TestBalanceStrategy:
    """Test balance-description strategy end-to-end."""

    def test_extracts_debit_transactions(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        fields = session.final_fields
        assert fields["DOCUMENT_TYPE"] == "BANK_STATEMENT"
        assert "Grocery Store Purchase" in fields["LINE_ITEM_DESCRIPTIONS"]
        assert "Electricity Bill Payment" in fields["LINE_ITEM_DESCRIPTIONS"]
        # Salary should be filtered out (credit, not debit)
        assert "Salary" not in fields["LINE_ITEM_DESCRIPTIONS"]

    def test_pipe_delimited_amounts(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        amounts = session.final_fields["TRANSACTION_AMOUNTS_PAID"]
        assert "$45.50" in amounts
        assert "$120.00" in amounts
        assert " | " in amounts

    def test_date_range_computed(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        date_range = session.final_fields["STATEMENT_DATE_RANGE"]
        assert "03/01/2025" in date_range
        assert "10/01/2025" in date_range

    def test_model_calls_count(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # detect_headers + extract_balance = 2 model calls
        assert session.trace.total_model_calls == 2


class TestAmountStrategy:
    """Test amount-description strategy end-to-end."""

    def test_filters_negative_amounts_only(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_AMOUNT, AMOUNT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        fields = session.final_fields
        assert "Grocery Store Purchase" in fields["LINE_ITEM_DESCRIPTIONS"]
        assert "Electricity Bill Payment" in fields["LINE_ITEM_DESCRIPTIONS"]
        assert "Salary" not in fields["LINE_ITEM_DESCRIPTIONS"]

    def test_preserves_negative_sign(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_AMOUNT, AMOUNT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        amounts = session.final_fields["TRANSACTION_AMOUNTS_PAID"]
        assert "-$45.50" in amounts
        assert "-$120.00" in amounts


class TestDebitCreditStrategy:
    """Test debit/credit strategy (no balance column)."""

    def test_extracts_debits(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_DEBIT_CREDIT, DEBIT_CREDIT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        fields = session.final_fields
        assert "Grocery Store Purchase" in fields["LINE_ITEM_DESCRIPTIONS"]
        assert "Salary" not in fields["LINE_ITEM_DESCRIPTIONS"]

    def test_balances_not_found(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_DEBIT_CREDIT, DEBIT_CREDIT_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # No balance column in this strategy
        assert "NOT_FOUND" in session.final_fields["ACCOUNT_BALANCE"]


class TestSchemaFallback:
    """Test schema-based fallback (garbage headers)."""

    def test_parses_field_value_format(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_GARBAGE, SCHEMA_FALLBACK_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        fields = session.final_fields
        assert fields["DOCUMENT_TYPE"] == "BANK_STATEMENT"
        assert "Grocery Store Purchase" in fields.get("LINE_ITEM_DESCRIPTIONS", "")
        assert "01/01/2025 - 31/01/2025" in fields.get("STATEMENT_DATE_RANGE", "")

    def test_bypasses_post_process(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_GARBAGE, SCHEMA_FALLBACK_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        # Schema fallback goes directly to done, no bank_post_process
        assert "bank_post_process" not in session.trace.nodes_visited


# ---------------------------------------------------------------------------
# Post-processing unit tests
# ---------------------------------------------------------------------------


class TestBankPostProcess:
    """Unit tests for run_bank_post_process() directly."""

    def test_balance_strategy_post_process(self) -> None:
        from common.bank_post_process import run_bank_post_process
        from common.extraction_types import NodeResult

        state = WorkflowState()
        state.node_results["detect_headers"] = NodeResult(
            key="detect_headers",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed={
                "headers": ["Date", "Description", "Debit", "Credit", "Balance"],
                "column_mapping": {
                    "date": "Date",
                    "description": "Description",
                    "debit": "Debit",
                    "credit": "Credit",
                    "balance": "Balance",
                    "amount": None,
                },
            },
            elapsed=0.0,
            attempt=1,
            edge_taken="ok",
        )
        state.node_results["extract_balance"] = NodeResult(
            key="extract_balance",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed={
                "rows": [
                    {
                        "Date": "01/01/2025",
                        "Description": "Opening Balance",
                        "Debit": "NOT_FOUND",
                        "Credit": "NOT_FOUND",
                        "Balance": "$2,000.00",
                    },
                    {
                        "Date": "03/01/2025",
                        "Description": "Grocery Purchase",
                        "Debit": "$50.00",
                        "Credit": "NOT_FOUND",
                        "Balance": "$1,950.00",
                    },
                    {
                        "Date": "05/01/2025",
                        "Description": "Salary",
                        "Debit": "NOT_FOUND",
                        "Credit": "$3,000.00",
                        "Balance": "$4,950.00",
                    },
                ],
                "row_count": 3,
                "date_col": "Date",
                "desc_col": "Description",
                "debit_col": "Debit",
                "credit_col": "Credit",
                "balance_col": "Balance",
            },
            elapsed=0.5,
            attempt=1,
            edge_taken="ok",
        )

        ok, fields = run_bank_post_process(state)

        assert ok is True
        assert fields["DOCUMENT_TYPE"] == "BANK_STATEMENT"
        assert (
            "Grocery Purchase" in fields["TRANSACTION_AMOUNTS_PAID"]
            or "50.00" in fields["TRANSACTION_AMOUNTS_PAID"]
        )
        assert "Salary" not in fields.get("LINE_ITEM_DESCRIPTIONS", "")

    def test_amount_strategy_post_process(self) -> None:
        from common.bank_post_process import run_bank_post_process
        from common.extraction_types import NodeResult

        state = WorkflowState()
        state.node_results["detect_headers"] = NodeResult(
            key="detect_headers",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed={
                "headers": ["Date", "Description", "Amount", "Balance"],
                "column_mapping": {
                    "date": "Date",
                    "description": "Description",
                    "debit": None,
                    "credit": None,
                    "balance": "Balance",
                    "amount": "Amount",
                },
            },
            elapsed=0.0,
            attempt=1,
            edge_taken="ok",
        )
        state.node_results["extract_amount"] = NodeResult(
            key="extract_amount",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed={
                "rows": [
                    {
                        "Date": "03/01/2025",
                        "Description": "Grocery Purchase",
                        "Amount": "-$50.00",
                        "Balance": "$1,950.00",
                    },
                    {
                        "Date": "05/01/2025",
                        "Description": "Salary",
                        "Amount": "$3,000.00",
                        "Balance": "$4,950.00",
                    },
                ],
                "row_count": 2,
                "date_col": "Date",
                "desc_col": "Description",
                "amount_col": "Amount",
                "balance_col": "Balance",
            },
            elapsed=0.5,
            attempt=1,
            edge_taken="ok",
        )

        ok, fields = run_bank_post_process(state)

        assert ok is True
        assert "Grocery Purchase" in fields["LINE_ITEM_DESCRIPTIONS"]
        assert "Salary" not in fields["LINE_ITEM_DESCRIPTIONS"]
        assert "-$50.00" in fields["TRANSACTION_AMOUNTS_PAID"]

    def test_no_extraction_node_returns_minimal(self) -> None:
        from common.bank_post_process import run_bank_post_process

        state = WorkflowState()
        ok, fields = run_bank_post_process(state)

        assert ok is True
        assert fields["DOCUMENT_TYPE"] == "BANK_STATEMENT"


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test that graph bank sessions serialize correctly."""

    def test_to_record_format(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        record = session.to_record()
        assert record["document_type"] == "BANK_STATEMENT"
        assert record["error"] is None
        assert "DOCUMENT_TYPE: BANK_STATEMENT" in record["raw_response"]
        assert "nodes" in record
        assert "trace" in record

    def test_raw_response_has_pipe_fields(self, tmp_path: Path) -> None:
        gen_fn = _mock_generate_fn([HEADERS_BALANCE, BALANCE_RESPONSE])
        definition = _load_workflow()
        executor = GraphExecutor(gen_fn, build_parser_registry())

        session = executor.run(
            document_type="BANK_STATEMENT",
            definition=definition,
            image_path=_make_test_image(tmp_path),
        )

        raw = session.raw_response
        assert "TRANSACTION_DATES:" in raw
        assert "LINE_ITEM_DESCRIPTIONS:" in raw
        assert "TRANSACTION_AMOUNTS_PAID:" in raw
        assert " | " in raw


# ---------------------------------------------------------------------------
# Inject defaults tests
# ---------------------------------------------------------------------------


class TestInjectDefaults:
    """Test pipe-default syntax in inject resolution."""

    def test_inject_uses_default_when_none(self, tmp_path: Path) -> None:
        """When column_mapping returns None, inject default is used."""
        from common.extraction_types import NodeResult

        state = WorkflowState()
        state.node_results["detect_headers"] = NodeResult(
            key="detect_headers",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed={
                "column_mapping": {
                    "description": None,
                    "balance": "Balance",
                },
            },
            elapsed=0.0,
            attempt=1,
            edge_taken="ok",
        )

        template = "Extract {desc_col} and {balance_col}"
        injections = {
            "desc_col": "detect_headers.column_mapping.description|Description",
            "balance_col": "detect_headers.column_mapping.balance|Balance",
        }

        result = GraphExecutor._resolve_inject(template, injections, state)

        assert result == "Extract Description and Balance"

    def test_inject_uses_actual_when_not_none(self, tmp_path: Path) -> None:
        from common.extraction_types import NodeResult

        state = WorkflowState()
        state.node_results["detect_headers"] = NodeResult(
            key="detect_headers",
            image_ref="primary",
            prompt_sent="",
            raw_response="",
            parsed={
                "column_mapping": {
                    "description": "Particulars",
                    "balance": "Running Balance",
                },
            },
            elapsed=0.0,
            attempt=1,
            edge_taken="ok",
        )

        template = "Extract {desc_col} and {balance_col}"
        injections = {
            "desc_col": "detect_headers.column_mapping.description|Description",
            "balance_col": "detect_headers.column_mapping.balance|Balance",
        }

        result = GraphExecutor._resolve_inject(template, injections, state)

        assert result == "Extract Particulars and Running Balance"
