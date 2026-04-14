"""Ports & adapters for the response handling boundary.

Ports (Protocol classes):
    FieldSchemaPort    -- "what fields exist and their types"
    ResponseParser     -- "how to convert raw text to dict"
    FieldCleaner       -- "how to normalize individual values"
    BusinessValidator  -- "how to check cross-field business rules"

Coordinator:
    ResponseHandler    -- composes all four ports into parse -> clean -> validate

Factory:
    create_response_handler()  -- wires default adapters for production use
"""

from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Port A: Field schema (what fields exist and their types)
# ---------------------------------------------------------------------------


@runtime_checkable
class FieldSchemaPort(Protocol):
    """Read-only view of field metadata needed by the response handler.

    The real FieldSchema satisfies this Protocol structurally -- no adapter
    class needed.
    """

    @property
    def extraction_fields(self) -> tuple[str, ...]:
        """Universal superset of all extraction field names."""
        ...

    @property
    def field_types(self) -> dict[str, str]:
        """Map each field name to its type tag."""
        ...

    def resolve_doc_type(self, raw_type: str) -> str:
        """Resolve a document type alias to its canonical lowercase form."""
        ...


# ---------------------------------------------------------------------------
# Port B: Parsing strategy (how to convert raw text to dict)
# ---------------------------------------------------------------------------


@runtime_checkable
class ResponseParser(Protocol):
    """Convert raw model text to a raw field dict."""

    def parse(
        self,
        raw_response: str,
        expected_fields: list[str],
    ) -> dict[str, str]:
        """Parse raw model output into {field_name: raw_value}."""
        ...


# ---------------------------------------------------------------------------
# Port C: Per-field cleaning (how to normalize individual values)
# ---------------------------------------------------------------------------


@runtime_checkable
class FieldCleaner(Protocol):
    """Normalize a single field value based on its name and type."""

    def clean(self, field_name: str, raw_value: str) -> str:
        """Clean and normalize a single extracted value."""
        ...


# ---------------------------------------------------------------------------
# Port D: Cross-field business validation
# ---------------------------------------------------------------------------


@runtime_checkable
class BusinessValidator(Protocol):
    """Validate and optionally correct an entire extraction dict."""

    def validate(self, cleaned_data: dict[str, str]) -> dict[str, str]:
        """Apply cross-field business rules to cleaned extraction data."""
        ...


# ---------------------------------------------------------------------------
# Coordinator: ResponseHandler
# ---------------------------------------------------------------------------


class ResponseHandler:
    """Compose parse -> clean -> validate into a single ``handle()`` call.

    This is the ONLY class the orchestrator needs to import for response
    handling. All internal complexity is hidden behind the four ports.
    """

    def __init__(
        self,
        *,
        schema: FieldSchemaPort,
        parser: ResponseParser,
        cleaner: FieldCleaner,
        validator: BusinessValidator,
    ) -> None:
        self._schema = schema
        self._parser = parser
        self._cleaner = cleaner
        self._validator = validator

    @property
    def schema(self) -> FieldSchemaPort:
        """Expose the injected schema for callers that need field metadata."""
        return self._schema

    def handle(
        self,
        raw_response: str,
        expected_fields: list[str],
    ) -> dict[str, str]:
        """Run the full parse -> clean -> validate pipeline.

        Steps:
            1. ``parser.parse(raw_response, expected_fields)``
            2. For each field: ``cleaner.clean(field_name, value)``
            3. ``validator.validate(cleaned_dict)``
        """
        parsed = self._parser.parse(raw_response, expected_fields)

        cleaned = {
            field_name: self._cleaner.clean(field_name, value)
            for field_name, value in parsed.items()
        }

        return self._validator.validate(cleaned)

    def handle_batch(
        self,
        raw_responses: list[str],
        field_lists: list[list[str]],
    ) -> list[dict[str, str]]:
        """Handle multiple responses (one per image in a batch).

        Raises:
            ValueError: If ``len(raw_responses) != len(field_lists)``.
        """
        if len(raw_responses) != len(field_lists):
            msg = (
                f"raw_responses/field_lists length mismatch: "
                f"{len(raw_responses)} vs {len(field_lists)}"
            )
            raise ValueError(msg)

        return [
            self.handle(response, fields)
            for response, fields in zip(raw_responses, field_lists, strict=True)
        ]


# ---------------------------------------------------------------------------
# Default adapters wrapping existing implementations
# ---------------------------------------------------------------------------


class HybridParserAdapter:
    """Adapter: wraps hybrid_parse_response() as a ResponseParser."""

    def parse(self, raw_response: str, expected_fields: list[str]) -> dict[str, str]:
        from common.extraction_parser import hybrid_parse_response

        return hybrid_parse_response(raw_response, expected_fields)


class ExtractionCleanerAdapter:
    """Adapter: wraps ExtractionCleaner.clean_field_value() as a FieldCleaner."""

    def __init__(self, *, debug: bool = False) -> None:
        from common.extraction_cleaner import ExtractionCleaner

        self._cleaner = ExtractionCleaner(debug=debug)

    def clean(self, field_name: str, raw_value: str) -> str:
        return self._cleaner.clean_field_value(field_name, raw_value)


class BusinessKnowledgeAdapter:
    """Adapter: wraps ExtractionCleaner._apply_business_knowledge()."""

    def __init__(self, *, debug: bool = False) -> None:
        from common.extraction_cleaner import ExtractionCleaner

        self._cleaner = ExtractionCleaner(debug=debug)

    def validate(self, cleaned_data: dict[str, str]) -> dict[str, str]:
        return self._cleaner._apply_business_knowledge(cleaned_data)


# ---------------------------------------------------------------------------
# Factory: wires default adapters for production use
# ---------------------------------------------------------------------------


def create_response_handler(
    *,
    schema: FieldSchemaPort | None = None,
    debug: bool = False,
) -> ResponseHandler:
    """Create a ResponseHandler wired with production-default adapters.

    Args:
        schema: Explicit FieldSchema. If None, falls back to the
            module-level singleton via ``get_field_schema()``.
        debug: Propagated to cleaner/validator for diagnostic output.
    """
    if schema is None:
        from common.field_schema import get_field_schema

        schema = get_field_schema()

    return ResponseHandler(
        schema=schema,
        parser=HybridParserAdapter(),
        cleaner=ExtractionCleanerAdapter(debug=debug),
        validator=BusinessKnowledgeAdapter(debug=debug),
    )
