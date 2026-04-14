"""Unified field schema -- single YAML read, frozen, module-level cache.

Replaces:
  - SimpleFieldLoader          (common/field_definitions_loader.py)
  - FieldSchema                (common/app_config.py)
  - load_document_field_definitions()  (common/batch_processor.py)

One import, zero args:
    from common.field_schema import get_field_schema
    fields = get_field_schema()
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


class FieldSchemaError(Exception):
    """Raised when field_definitions.yaml is missing, malformed, or invalid."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


# The canonical set of fields excluded from extraction AND evaluation.
# Declared once, used everywhere -- no more inconsistency.
_VALIDATION_ONLY: frozenset[str] = frozenset(
    {
        "TRANSACTION_AMOUNTS_RECEIVED",
        "ACCOUNT_BALANCE",
    }
)

# Evaluation equivalence classes: canonical types that should be treated as
# identical for DOCUMENT_TYPE scoring.  "invoice" and "receipt" share the same
# extraction fields, so a model predicting one when the ground truth is the
# other should not be penalized.
# This is scoring *policy*, not field metadata -- lives in code, not YAML.
_EVAL_EQUIVALENCE: dict[str, str] = {
    "invoice": "invoice_receipt",
    "receipt": "invoice_receipt",
    "bank_statement": "bank_statement",
    "travel_expense": "travel_expense",
}


@dataclass(frozen=True)
class FieldSchema:
    """Immutable field schema loaded once from config/field_definitions.yaml.

    Provides three access patterns, ranked by frequency:

    1. Typed field categories (evaluation_metrics.py -- 7 call sites):
       ``fields.monetary_fields``, ``fields.boolean_fields``, etc.

    2. Per-doc-type field lists (cli.py, orchestrator.py, document_pipeline.py):
       ``fields.get_extraction_fields("bank_statement")``

    3. Raw universal field list (extraction_parser.py):
       ``fields.extraction_fields``
    """

    # -- Core: universal extraction fields (validation-only excluded) --
    extraction_fields: tuple[str, ...]
    field_count: int

    # -- Typed categories (what evaluation_metrics cares about) --
    monetary_fields: frozenset[str]
    date_fields: frozenset[str]
    list_fields: frozenset[str]
    boolean_fields: frozenset[str]
    calculated_fields: frozenset[str]
    transaction_list_fields: frozenset[str]
    text_fields: frozenset[str]
    phone_fields: frozenset[str]
    numeric_id_fields: frozenset[str]

    # -- Per-field type lookup (field_name -> type string) --
    field_types: dict[str, str]

    # -- Per-doc-type field lists (validation-only excluded) --
    _doc_type_fields: dict[str, tuple[str, ...]] = field(repr=False)

    # -- Doc-type alias map (lowercased alias -> canonical type) --
    _alias_map: dict[str, str] = field(repr=False)

    # -- Validation-only fields (constant, stored for introspection) --
    validation_only_fields: frozenset[str] = _VALIDATION_ONLY

    # -- Per-doc-type min_tokens (for generation config) --
    min_tokens_by_type: dict[str, int] = field(default_factory=dict, repr=False)

    # -- Critical fields for evaluation thresholds --
    critical_fields: tuple[str, ...] = ()

    # -- Supported document types --
    supported_document_types: tuple[str, ...] = ()

    # ==================================================================
    # CONSTRUCTION
    # ==================================================================

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | None = None,
    ) -> "FieldSchema":
        """Load from field_definitions.yaml with fail-fast validation.

        Args:
            config_path: Explicit path. Defaults to
                ``<project_root>/config/field_definitions.yaml``.

        Raises:
            FieldSchemaError: If the file is missing, has invalid YAML,
                or is missing required sections/fields.
        """
        resolved = config_path or (
            Path(__file__).parent.parent / "config" / "field_definitions.yaml"
        )

        # 1. File existence
        if not resolved.exists():
            raise FieldSchemaError(
                f"Field definitions file not found: {resolved.absolute()}\n"
                f"This file is REQUIRED. Ensure config/field_definitions.yaml "
                f"exists in the project root.\n"
                f"Required structure:\n"
                f"  document_fields:\n"
                f"    invoice:\n"
                f"      fields: [DOCUMENT_TYPE, BUSINESS_ABN, ...]",
                path=resolved,
            ) from None

        # 2. Parse YAML
        try:
            with resolved.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise FieldSchemaError(
                f"Invalid YAML in {resolved.absolute()}: {exc}",
                path=resolved,
            ) from exc

        if not isinstance(raw, dict):
            raise FieldSchemaError(
                f"Expected top-level mapping in {resolved.absolute()}, "
                f"got {type(raw).__name__}",
                path=resolved,
            ) from None

        # 3. Validate required sections
        doc_fields_raw = raw.get("document_fields")
        if not doc_fields_raw or not isinstance(doc_fields_raw, dict):
            raise FieldSchemaError(
                f"Missing or empty 'document_fields' section in "
                f"{resolved.absolute()}\n"
                f"Required structure:\n"
                f"document_fields:\n"
                f"  invoice:\n"
                f"    fields: [FIELD_A, FIELD_B, ...]",
                path=resolved,
            ) from None

        # 4. Build per-doc-type field lists (validation-only excluded)
        doc_type_fields: dict[str, tuple[str, ...]] = {}
        min_tokens_by_type: dict[str, int] = {}

        for doc_type, type_cfg in doc_fields_raw.items():
            if not isinstance(type_cfg, dict) or "fields" not in type_cfg:
                raise FieldSchemaError(
                    f"Document type '{doc_type}' missing 'fields' list in "
                    f"{resolved.absolute()}",
                    path=resolved,
                ) from None
            if not type_cfg["fields"]:
                raise FieldSchemaError(
                    f"Empty 'fields' list for '{doc_type}' in {resolved.absolute()}",
                    path=resolved,
                ) from None

            doc_type_fields[doc_type] = tuple(
                f for f in type_cfg["fields"] if f not in _VALIDATION_ONLY
            )

            if "min_tokens" in type_cfg:
                min_tokens_by_type[doc_type] = int(type_cfg["min_tokens"])

        # 5. Universal extraction fields
        if "universal" not in doc_type_fields:
            raise FieldSchemaError(
                f"Missing 'universal' document type in document_fields "
                f"section of {resolved.absolute()}. "
                f"This type is required as the superset of all fields.",
                path=resolved,
            ) from None

        extraction_fields = doc_type_fields["universal"]

        # 6. Field type classifications
        eval_cfg = raw.get("evaluation", {})
        ft_raw = eval_cfg.get("field_types", {})

        monetary = frozenset(ft_raw.get("monetary", []))
        date = frozenset(ft_raw.get("date", []))
        list_ = frozenset(ft_raw.get("list", []))
        boolean = frozenset(ft_raw.get("boolean", []))
        calculated = frozenset(ft_raw.get("calculated", []))
        transaction_list = frozenset(ft_raw.get("transaction_list", []))

        # Build per-field type lookup
        field_type_map: dict[str, str] = {}
        for fname in extraction_fields:
            if fname in monetary:
                field_type_map[fname] = "monetary"
            elif fname in date:
                field_type_map[fname] = "date"
            elif fname in list_:
                field_type_map[fname] = "list"
            elif fname in boolean:
                field_type_map[fname] = "boolean"
            elif fname in calculated:
                field_type_map[fname] = "calculated"
            elif fname in transaction_list:
                field_type_map[fname] = "transaction_list"
            else:
                field_type_map[fname] = "text"

        # 7. Document type aliases
        alias_map: dict[str, str] = {}
        aliases_raw = raw.get("document_type_aliases", {})
        for canonical, alias_list in aliases_raw.items():
            alias_map[canonical.lower()] = canonical
            if isinstance(alias_list, list):
                for alias in alias_list:
                    alias_map[alias.lower()] = canonical

        # 8. Other evaluation metadata
        critical = tuple(eval_cfg.get("critical_fields", []))
        supported = tuple(raw.get("supported_document_types", []))

        return cls(
            extraction_fields=extraction_fields,
            field_count=len(extraction_fields),
            monetary_fields=monetary,
            date_fields=date,
            list_fields=list_,
            boolean_fields=boolean,
            calculated_fields=calculated,
            transaction_list_fields=transaction_list,
            text_fields=frozenset(
                f for f in extraction_fields if field_type_map.get(f) == "text"
            ),
            phone_fields=frozenset(),
            numeric_id_fields=frozenset(),
            field_types=field_type_map,
            _doc_type_fields=doc_type_fields,
            _alias_map=alias_map,
            min_tokens_by_type=min_tokens_by_type,
            critical_fields=critical,
            supported_document_types=supported,
        )

    # ==================================================================
    # QUERY METHODS
    # ==================================================================

    def is_evaluation_field(self, field_name: str) -> bool:
        """True if the field should be included in evaluation metrics."""
        return field_name not in self.validation_only_fields

    def filter_evaluation_fields(self, fields: list[str]) -> list[str]:
        """Filter a list to exclude validation-only fields."""
        return [f for f in fields if f not in self.validation_only_fields]

    def resolve_doc_type(self, raw_type: str) -> str:
        """Resolve a document type alias to its canonical form.

        ``"tax invoice"`` -> ``"invoice"``,
        ``"statement"``   -> ``"bank_statement"``.

        Returns the input unchanged if no alias matches.
        """
        return self._alias_map.get(raw_type.lower(), raw_type.lower())

    def eval_doc_type_class(self, raw_type: str) -> str:
        """Return the evaluation equivalence class for a document type.

        Chains: resolve alias -> map to equivalence class.
        ``"invoice"`` and ``"receipt"`` both return ``"invoice_receipt"``.
        Types with no explicit equivalence class return their canonical form.

        Use this when comparing predicted vs ground-truth DOCUMENT_TYPE for
        accuracy scoring.
        """
        canonical = self.resolve_doc_type(raw_type)
        return _EVAL_EQUIVALENCE.get(canonical, canonical)

    def get_extraction_fields(self, document_type: str) -> list[str]:
        """Per-doc-type field list with validation-only fields excluded.

        Resolves aliases automatically:
        ``get_extraction_fields("tax invoice")`` returns invoice fields.

        Falls back to universal fields for unknown types.
        """
        canonical = self.resolve_doc_type(document_type)
        fields = self._doc_type_fields.get(canonical)
        if fields is not None:
            return list(fields)
        return list(self.extraction_fields)

    def get_all_doc_type_fields(self) -> dict[str, list[str]]:
        """All doc types -> field lists. Drop-in for load_document_field_definitions().

        Returns a new dict each call (callers can mutate freely).
        """
        return {dt: list(fields) for dt, fields in self._doc_type_fields.items()}

    def get_document_type_fields(self, document_type: str) -> list[str]:
        """Get fields for a doc type, filtered for evaluation.

        Backward-compatible with the old FieldSchema.get_document_type_fields().
        """
        return self.filter_evaluation_fields(self.get_extraction_fields(document_type))


# ==================================================================
# MODULE-LEVEL SINGLETON
# ==================================================================


@lru_cache(maxsize=1)
def get_field_schema() -> FieldSchema:
    """Return the singleton FieldSchema, loading from YAML on first call.

    This is THE canonical way to get field definitions. One import, zero args::

        from common.field_schema import get_field_schema
        fields = get_field_schema()

    The FieldSchema is frozen (immutable) and cached forever.
    Thread-safe via lru_cache's built-in lock.
    """
    return FieldSchema.from_yaml()
