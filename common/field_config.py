"""Field evaluation helpers and field-type accessors.

Provides evaluation-field filtering and field-type lookups used by
evaluation_metrics.py, simple_model_evaluator.py, and batch_processor.py.
Schema loading is handled by ``SimpleFieldLoader``; mutable globals have
been eliminated.
"""

from __future__ import annotations

from functools import lru_cache

# Fields EXTRACTED but EXCLUDED from evaluation metrics.
# These are used for mathematical validation/correction but don't count toward accuracy.
VALIDATION_ONLY_FIELDS = [
    "TRANSACTION_AMOUNTS_RECEIVED",
    "ACCOUNT_BALANCE",
]


@lru_cache(maxsize=1)
def _load_field_types() -> dict[str, list[str]]:
    """Load field type classifications from field_definitions.yaml (cached)."""
    from .field_definitions_loader import SimpleFieldLoader

    return SimpleFieldLoader().get_field_types()


@lru_cache(maxsize=1)
def _load_extraction_fields() -> list[str]:
    """Load universal extraction fields (cached)."""
    from .field_definitions_loader import SimpleFieldLoader

    fields = SimpleFieldLoader().get_document_fields("universal")
    return [f for f in fields if f != "TRANSACTION_AMOUNTS_RECEIVED"]


# -- Field-type accessors (used by evaluation_metrics.py) ------------------


def get_monetary_fields() -> list:
    """Get monetary fields."""
    return _load_field_types().get("monetary", [])


def get_date_fields() -> list:
    """Get date fields."""
    return _load_field_types().get("date", [])


def get_list_fields() -> list:
    """Get list fields."""
    return _load_field_types().get("list", [])


def get_boolean_fields() -> list:
    """Get boolean fields."""
    return _load_field_types().get("boolean", [])


def get_calculated_fields() -> list:
    """Get calculated fields."""
    return _load_field_types().get("calculated", [])


def get_transaction_list_fields() -> list:
    """Get transaction list fields."""
    return _load_field_types().get("transaction_list", [])


def get_phone_fields() -> list:
    """Get phone fields."""
    return []


def get_all_field_types() -> dict[str, str]:
    """Get field type mapping (field_name -> 'text')."""
    return {f: "text" for f in _load_extraction_fields()}


def is_evaluation_field(field_name: str) -> bool:
    """Check if a field should be included in evaluation metrics."""
    return field_name not in VALIDATION_ONLY_FIELDS


def filter_evaluation_fields(fields: list) -> list:
    """Filter a list of fields to exclude validation-only fields."""
    return [field for field in fields if is_evaluation_field(field)]


def get_document_type_fields(document_type: str) -> list:
    """Get fields specific to document type, filtered for evaluation.

    Args:
        document_type: Document type ('invoice', 'receipt', 'bank_statement')

    Returns:
        Fields specific to the document type, excluding validation-only fields.
    """
    from .field_definitions_loader import SimpleFieldLoader

    loader = SimpleFieldLoader()

    doc_type_mapping = {
        "invoice": "invoice",
        "tax_invoice": "invoice",
        "bill": "invoice",
        "receipt": "receipt",
        "purchase_receipt": "receipt",
        "bank_statement": "bank_statement",
        "statement": "bank_statement",
        "transaction_link": "transaction_link",
    }

    mapped_type = doc_type_mapping.get(document_type.lower(), document_type.lower())

    try:
        field_names = loader.get_document_fields(mapped_type)
    except Exception:
        # Fallback: load universal fields via the loader
        field_names = loader.get_document_fields("universal")

    return filter_evaluation_fields(field_names)
