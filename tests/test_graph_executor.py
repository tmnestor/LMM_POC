"""GPU-free tests for GraphExecutor with mock generate_fn."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from common.extraction_types import GenerateResult
from common.graph_executor import GraphExecutor
from common.turn_parsers import build_parser_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_image(tmp_path: Path, name: str = "test.png") -> str:
    """Create a tiny test image and return its path."""
    img = Image.new("RGB", (10, 10), color="white")
    path = tmp_path / name
    img.save(path)
    return str(path)


def _mock_generate_fn(
    responses: list[str],
) -> MagicMock:
    """Create a mock generate_fn that returns canned responses in order."""
    mock = MagicMock()
    mock.side_effect = [GenerateResult(text=r) for r in responses]
    return mock


# ---------------------------------------------------------------------------
# Single-node graph (invoice-like)
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_extract_fields(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)
        gen_fn = _mock_generate_fn(
            [
                "SUPPLIER_NAME: Acme Corp\nTOTAL_AMOUNT: $100.00",
            ]
        )

        definition = {
            "nodes": {
                "extract": {
                    "template": "Extract fields from this invoice.",
                    "max_tokens": 1024,
                    "parser": "field_value",
                    "edges": {"ok": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="INVOICE",
            definition=definition,
            image_path=img,
        )

        assert session.document_type == "INVOICE"
        assert session.final_fields["SUPPLIER_NAME"] == "Acme Corp"
        assert session.final_fields["TOTAL_AMOUNT"] == "$100.00"
        assert session.trace.total_model_calls == 1
        assert len(session.trace.nodes_visited) == 1

    def test_raw_response_format(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)
        gen_fn = _mock_generate_fn(
            [
                "SUPPLIER_NAME: Test\nTOTAL_AMOUNT: $50",
            ]
        )

        definition = {
            "nodes": {
                "extract": {
                    "template": "Extract.",
                    "parser": "field_value",
                    "edges": {"ok": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="INVOICE",
            definition=definition,
            image_path=img,
        )

        # raw_response should be FIELD: value lines
        assert "SUPPLIER_NAME: Test" in session.raw_response
        assert "TOTAL_AMOUNT: $50" in session.raw_response


# ---------------------------------------------------------------------------
# Multi-node graph with inject
# ---------------------------------------------------------------------------


class TestMultiNode:
    def test_receipt_to_match_workflow(self, tmp_path: Path) -> None:
        receipt_img = _make_test_image(tmp_path, "receipt.png")
        bank_img = _make_test_image(tmp_path, "bank.png")

        gen_fn = _mock_generate_fn(
            [
                # extract_receipts
                (
                    "--- RECEIPT 1 ---\n"
                    "STORE: Corner Shop\n"
                    "DATE: 15/03/2025\n"
                    "TOTAL: $42.50\n"
                ),
                # detect_headers
                "1. Date\n2. Description\n3. Debit\n4. Credit\n5. Balance",
                # match_to_statement
                (
                    "--- RECEIPT 1 ---\n"
                    "MATCHED_TRANSACTION: FOUND\n"
                    "TRANSACTION_DATE: 15/03/2025\n"
                    "TRANSACTION_AMOUNT: $42.50\n"
                    "TRANSACTION_DESCRIPTION: CORNER SHOP PTY\n"
                    "RECEIPT_STORE: Corner Shop\n"
                    "RECEIPT_TOTAL: $42.50\n"
                    "CONFIDENCE: HIGH\n"
                    "REASONING: Amount matches exactly\n"
                ),
            ]
        )

        definition = {
            "inputs": [
                {"name": "receipt", "type": "image"},
                {"name": "bank_statement", "type": "image"},
            ],
            "nodes": {
                "extract_receipts": {
                    "image": "receipt",
                    "template": "Extract receipts.",
                    "max_tokens": 500,
                    "parser": "receipt_list",
                    "edges": {
                        "ok": "detect_headers",
                        "parse_failed": "extract_receipts",
                    },
                },
                "detect_headers": {
                    "image": "bank_statement",
                    "template": "List column headers.",
                    "max_tokens": 200,
                    "parser": "header_list",
                    "edges": {
                        "ok": "match_to_statement",
                        "parse_failed": "detect_headers",
                    },
                },
                "match_to_statement": {
                    "image": "bank_statement",
                    "template": "Match purchases:\n{purchases_text}\nDebit: {debit_column}",
                    "inject": {
                        "purchases_text": "extract_receipts.formatted_text",
                        "debit_column": "detect_headers.column_mapping.debit",
                    },
                    "max_tokens": 2000,
                    "parser": "transaction_match",
                    "edges": {"ok": "validate_amounts"},
                },
                "validate_amounts": {
                    "type": "validator",
                    "check": "amount_gate",
                    "edges": {"ok": "done", "failed": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TRANSACTION_LINK",
            definition=definition,
            images={
                "receipt": receipt_img,
                "bank_statement": bank_img,
            },
            image_name="pair_001",
            extra_fields={"BANK_STATEMENT_FILE": "bank.png"},
        )

        assert session.document_type == "TRANSACTION_LINK"
        assert session.image_name == "pair_001"
        assert session.trace.total_model_calls == 3
        assert session.final_fields["BANK_STATEMENT_FILE"] == "bank.png"
        assert session.final_fields["RECEIPT_DATE"] == "15/03/2025"
        assert session.final_fields["RECEIPT_DESCRIPTION"] == "Corner Shop"
        assert session.final_fields["BANK_TRANSACTION_DEBIT"] == "$42.50"
        assert session.input_images is not None

    def test_inject_resolves_dot_paths(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)

        gen_fn = _mock_generate_fn(
            [
                "1. Date\n2. Description\n3. Debit",
                "SUPPLIER_NAME: Resolved\nTOTAL_AMOUNT: $10",
            ]
        )

        definition = {
            "nodes": {
                "detect": {
                    "template": "Detect headers.",
                    "parser": "header_list",
                    "edges": {"ok": "extract"},
                },
                "extract": {
                    "template": "Debit column: {debit_col}",
                    "inject": {"debit_col": "detect.column_mapping.debit"},
                    "parser": "field_value",
                    "edges": {"ok": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TEST",
            definition=definition,
            image_path=img,
        )

        # Verify inject resolved — the prompt should contain "Debit"
        extract_result = session.node_results[1]
        assert "Debit" in extract_result.prompt_sent


# ---------------------------------------------------------------------------
# Self-Refine retry
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retry_on_parse_failure(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)

        gen_fn = _mock_generate_fn(
            [
                "garbage response",  # First attempt fails parsing
                (
                    "--- RECEIPT 1 ---\n"
                    "STORE: Fixed Shop\n"
                    "DATE: 01/01/2025\n"
                    "TOTAL: $10.00\n"
                ),
            ]
        )

        definition = {
            "nodes": {
                "extract": {
                    "template": "Extract receipts.",
                    "parser": "receipt_list",
                    "max_retries": 1,
                    "reflection": "Parse failed: {error}\nTry again.",
                    "edges": {"ok": "done", "parse_failed": "extract"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TEST",
            definition=definition,
            image_path=img,
        )

        assert session.trace.total_model_calls == 2
        assert session.trace.retries == {"extract": 1}
        # Second attempt should have reflection appended
        retry_result = session.node_results[1]
        assert retry_result.attempt == 2
        assert "Parse failed" in retry_result.prompt_sent

    def test_exhausted_retries_proceed_with_ok(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)

        gen_fn = _mock_generate_fn(
            [
                "garbage 1",  # First attempt
                "garbage 2",  # Retry
            ]
        )

        definition = {
            "nodes": {
                "extract": {
                    "template": "Extract receipts.",
                    "parser": "receipt_list",
                    "max_retries": 1,
                    "reflection": "Try again: {error}",
                    "edges": {"ok": "done", "parse_failed": "extract"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TEST",
            definition=definition,
            image_path=img,
        )

        # After exhausting retries, should proceed with "ok" edge
        assert session.trace.total_model_calls == 2
        last_edge = session.trace.edges_taken[-1]
        assert last_edge[1] == "ok"  # edge taken
        assert last_edge[2] == "done"  # destination


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_exceeds_max_steps(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)

        # Always returns garbage -> always parse_failed -> infinite loop
        gen_fn = _mock_generate_fn(["garbage"] * 25)

        definition = {
            "nodes": {
                "extract": {
                    "template": "Extract.",
                    "parser": "receipt_list",
                    "max_retries": 100,  # very high
                    "reflection": "Retry: {error}",
                    "edges": {"ok": "done", "parse_failed": "extract"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry(), max_graph_steps=5)
        with pytest.raises(RuntimeError, match="exceeded 5 steps"):
            executor.run(
                document_type="TEST",
                definition=definition,
                image_path=img,
            )


# ---------------------------------------------------------------------------
# Validator node
# ---------------------------------------------------------------------------


class TestValidator:
    def test_amount_gate_pass(self, tmp_path: Path) -> None:
        receipt_img = _make_test_image(tmp_path, "receipt.png")
        bank_img = _make_test_image(tmp_path, "bank.png")

        gen_fn = _mock_generate_fn(
            [
                "--- RECEIPT 1 ---\nSTORE: Shop\nDATE: 01/01/2025\nTOTAL: $50.00",
                "1. Date\n2. Description\n3. Debit",
                (
                    "--- RECEIPT 1 ---\n"
                    "MATCHED_TRANSACTION: FOUND\n"
                    "TRANSACTION_DATE: 01/01/2025\n"
                    "TRANSACTION_AMOUNT: $50.00\n"
                    "TRANSACTION_DESCRIPTION: SHOP\n"
                    "RECEIPT_STORE: Shop\n"
                    "RECEIPT_TOTAL: $50.00\n"
                    "CONFIDENCE: HIGH\n"
                    "REASONING: Match\n"
                ),
            ]
        )

        definition = {
            "inputs": [{"name": "receipt"}, {"name": "bank_statement"}],
            "nodes": {
                "extract_receipts": {
                    "image": "receipt",
                    "template": "Extract.",
                    "parser": "receipt_list",
                    "edges": {"ok": "detect_headers"},
                },
                "detect_headers": {
                    "image": "bank_statement",
                    "template": "Headers.",
                    "parser": "header_list",
                    "edges": {"ok": "match_to_statement"},
                },
                "match_to_statement": {
                    "image": "bank_statement",
                    "template": "Match {purchases_text}.",
                    "inject": {"purchases_text": "extract_receipts.formatted_text"},
                    "parser": "transaction_match",
                    "edges": {"ok": "validate_amounts"},
                },
                "validate_amounts": {
                    "type": "validator",
                    "check": "amount_gate",
                    "edges": {"ok": "done", "failed": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TRANSACTION_LINK",
            definition=definition,
            images={"receipt": receipt_img, "bank_statement": bank_img},
        )

        # Amounts match -> validator takes "ok" edge
        validator_result = [
            r for r in session.node_results if r.key == "validate_amounts"
        ][0]
        assert validator_result.edge_taken == "ok"
        assert validator_result.parsed["overridden_count"] == 0

    def test_amount_gate_fail(self, tmp_path: Path) -> None:
        receipt_img = _make_test_image(tmp_path, "receipt.png")
        bank_img = _make_test_image(tmp_path, "bank.png")

        gen_fn = _mock_generate_fn(
            [
                "--- RECEIPT 1 ---\nSTORE: Shop\nDATE: 01/01/2025\nTOTAL: $50.00",
                "1. Date\n2. Description\n3. Debit",
                (
                    "--- RECEIPT 1 ---\n"
                    "MATCHED_TRANSACTION: FOUND\n"
                    "TRANSACTION_DATE: 01/01/2025\n"
                    "TRANSACTION_AMOUNT: $99.99\n"  # Mismatch!
                    "TRANSACTION_DESCRIPTION: SHOP\n"
                    "RECEIPT_STORE: Shop\n"
                    "RECEIPT_TOTAL: $50.00\n"
                    "CONFIDENCE: HIGH\n"
                    "REASONING: Match\n"
                ),
            ]
        )

        definition = {
            "inputs": [{"name": "receipt"}, {"name": "bank_statement"}],
            "nodes": {
                "extract_receipts": {
                    "image": "receipt",
                    "template": "Extract.",
                    "parser": "receipt_list",
                    "edges": {"ok": "detect_headers"},
                },
                "detect_headers": {
                    "image": "bank_statement",
                    "template": "Headers.",
                    "parser": "header_list",
                    "edges": {"ok": "match_to_statement"},
                },
                "match_to_statement": {
                    "image": "bank_statement",
                    "template": "Match {purchases_text}.",
                    "inject": {"purchases_text": "extract_receipts.formatted_text"},
                    "parser": "transaction_match",
                    "edges": {"ok": "validate_amounts"},
                },
                "validate_amounts": {
                    "type": "validator",
                    "check": "amount_gate",
                    "edges": {"ok": "done", "failed": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TRANSACTION_LINK",
            definition=definition,
            images={"receipt": receipt_img, "bank_statement": bank_img},
        )

        validator_result = [
            r for r in session.node_results if r.key == "validate_amounts"
        ][0]
        assert validator_result.edge_taken == "failed"
        assert validator_result.parsed["overridden_count"] == 1


# ---------------------------------------------------------------------------
# to_record serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_record_structure(self, tmp_path: Path) -> None:
        img = _make_test_image(tmp_path)
        gen_fn = _mock_generate_fn(["FIELD_A: value_a\nFIELD_B: value_b"])

        definition = {
            "nodes": {
                "extract": {
                    "template": "Extract.",
                    "parser": "field_value",
                    "edges": {"ok": "done"},
                },
            },
        }

        executor = GraphExecutor(gen_fn, build_parser_registry())
        session = executor.run(
            document_type="TEST",
            definition=definition,
            image_path=img,
        )
        record = session.to_record()

        assert record["document_type"] == "TEST"
        assert record["error"] is None
        assert len(record["nodes"]) == 1
        assert record["nodes"][0]["key"] == "extract"
        assert record["trace"]["total_model_calls"] == 1
        assert "raw_response" in record


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_images_kwarg_for_cross_image(self) -> None:
        gen_fn = _mock_generate_fn([])
        definition = {
            "inputs": [{"name": "receipt"}],
            "nodes": {"n": {"template": "T", "edges": {"ok": "done"}}},
        }
        executor = GraphExecutor(gen_fn, build_parser_registry())
        with pytest.raises(ValueError, match="images"):
            executor.run(document_type="T", definition=definition)

    def test_missing_image_path_for_single_image(self) -> None:
        gen_fn = _mock_generate_fn([])
        definition = {
            "nodes": {"n": {"template": "T", "edges": {"ok": "done"}}},
        }
        executor = GraphExecutor(gen_fn, build_parser_registry())
        with pytest.raises(ValueError, match="image_path"):
            executor.run(document_type="T", definition=definition)

    def test_unrecognized_definition(self) -> None:
        gen_fn = _mock_generate_fn([])
        executor = GraphExecutor(gen_fn, build_parser_registry())
        with pytest.raises(ValueError, match="Unrecognized"):
            executor.run(document_type="T", definition={"bad": "shape"})
