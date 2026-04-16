"""GPU-free tests for ResponseHandler using stub ports."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from common.response_handler import FieldSchemaPort, ResponseHandler

# ---------------------------------------------------------------------------
# Stubs: each satisfies one Protocol structurally
# ---------------------------------------------------------------------------


class StubParser:
    """Assigns each line of the response to a field, in order."""

    def parse(self, raw_response: str, expected_fields: list[str]) -> dict[str, str]:
        lines = raw_response.strip().split("\n") if raw_response.strip() else []
        result = {f: "NOT_FOUND" for f in expected_fields}
        for i, field_name in enumerate(expected_fields):
            if i < len(lines):
                result[field_name] = lines[i]
        return result


class IdentityCleaner:
    """Returns value unchanged."""

    def clean(self, field_name: str, raw_value: str) -> str:
        return raw_value


class NoOpValidator:
    """Returns dict unchanged."""

    def validate(self, cleaned_data: dict[str, str]) -> dict[str, str]:
        return cleaned_data


@dataclass(frozen=True)
class StubSchema:
    """Minimal schema for tests. Satisfies FieldSchemaPort structurally."""

    extraction_fields: tuple[str, ...] = ("TOTAL_AMOUNT", "INVOICE_DATE")
    field_types: dict[str, str] = field(
        default_factory=lambda: {"TOTAL_AMOUNT": "monetary", "INVOICE_DATE": "date"}
    )

    def resolve_doc_type(self, raw_type: str) -> str:
        return raw_type.lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(
    parser: Any = None,
    cleaner: Any = None,
    validator: Any = None,
    schema: Any = None,
) -> ResponseHandler:
    return ResponseHandler(
        schema=schema if schema is not None else StubSchema(),
        parser=parser if parser is not None else StubParser(),
        cleaner=cleaner if cleaner is not None else IdentityCleaner(),
        validator=validator if validator is not None else NoOpValidator(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandlePipeline:
    def test_runs_parse_clean_validate_in_sequence(self) -> None:
        handler = _make_handler()
        result = handler.handle(
            raw_response="$42.00\n15/03/2026",
            expected_fields=["TOTAL_AMOUNT", "INVOICE_DATE"],
        )
        assert result == {
            "TOTAL_AMOUNT": "$42.00",
            "INVOICE_DATE": "15/03/2026",
        }

    def test_fills_not_found_for_missing_fields(self) -> None:
        handler = _make_handler()
        result = handler.handle(
            raw_response="$42.00",
            expected_fields=["TOTAL_AMOUNT", "INVOICE_DATE", "GST_AMOUNT"],
        )
        assert result["TOTAL_AMOUNT"] == "$42.00"
        assert result["INVOICE_DATE"] == "NOT_FOUND"
        assert result["GST_AMOUNT"] == "NOT_FOUND"

    def test_empty_response_returns_all_not_found(self) -> None:
        handler = _make_handler()
        result = handler.handle(
            raw_response="",
            expected_fields=["TOTAL_AMOUNT", "INVOICE_DATE"],
        )
        assert all(v == "NOT_FOUND" for v in result.values())

    def test_cleaner_is_applied_per_field(self) -> None:
        class UpperCleaner:
            def clean(self, field_name: str, raw_value: str) -> str:
                return raw_value.upper() if raw_value != "NOT_FOUND" else raw_value

        handler = _make_handler(cleaner=UpperCleaner())
        result = handler.handle(
            raw_response="hello",
            expected_fields=["TOTAL_AMOUNT"],
        )
        assert result["TOTAL_AMOUNT"] == "HELLO"

    def test_validator_sees_cleaned_data(self) -> None:
        class RecordingValidator:
            def __init__(self) -> None:
                self.last_input: dict[str, str] | None = None

            def validate(self, cleaned_data: dict[str, str]) -> dict[str, str]:
                self.last_input = cleaned_data.copy()
                return cleaned_data

        class PrefixCleaner:
            def clean(self, field_name: str, raw_value: str) -> str:
                return f"CLEANED_{raw_value}"

        validator = RecordingValidator()
        handler = _make_handler(cleaner=PrefixCleaner(), validator=validator)
        handler.handle(
            raw_response="$42.00",
            expected_fields=["TOTAL_AMOUNT"],
        )
        assert validator.last_input == {"TOTAL_AMOUNT": "CLEANED_$42.00"}


class TestHandleBatch:
    def test_rejects_mismatched_lengths(self) -> None:
        handler = _make_handler()
        with pytest.raises(ValueError, match="length mismatch"):
            handler.handle_batch(
                raw_responses=["response1", "response2"],
                field_lists=[["TOTAL_AMOUNT"]],
            )

    def test_batch_processes_each_response(self) -> None:
        handler = _make_handler()
        results = handler.handle_batch(
            raw_responses=["$10.00", "$20.00"],
            field_lists=[["TOTAL_AMOUNT"], ["TOTAL_AMOUNT"]],
        )
        assert len(results) == 2
        assert results[0]["TOTAL_AMOUNT"] == "$10.00"
        assert results[1]["TOTAL_AMOUNT"] == "$20.00"


class TestSchemaProperty:
    def test_schema_exposed(self) -> None:
        schema = StubSchema()
        handler = _make_handler(schema=schema)
        assert handler.schema is schema

    def test_real_field_schema_satisfies_protocol(self) -> None:
        from common.field_schema import FieldSchema, get_field_schema

        schema = get_field_schema()
        assert isinstance(schema, FieldSchemaPort)
        assert isinstance(schema, FieldSchema)
