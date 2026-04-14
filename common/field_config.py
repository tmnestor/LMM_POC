"""Field evaluation helpers.

Provides evaluation-field filtering used by evaluation_metrics.py and
simple_model_evaluator.py.  All schema-loading globals have been absorbed
into ``common.app_config.FieldSchema``.
"""

# Fields EXTRACTED but EXCLUDED from evaluation metrics.
# These are used for mathematical validation/correction but don't count toward accuracy.
VALIDATION_ONLY_FIELDS = [
    "TRANSACTION_AMOUNTS_RECEIVED",
    "ACCOUNT_BALANCE",
]


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
