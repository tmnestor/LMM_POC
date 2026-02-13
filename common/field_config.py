"""
Field schema loading, public field accessors, and evaluation field filtering.

Extracted from the former common/config.py so that field-related logic lives
in its own module.  Consumers: evaluation_metrics.py, extraction_parser.py,
simple_model_evaluator.py.
"""

# ============================================================================
# DYNAMIC SCHEMA-BASED FIELD DISCOVERY
# ============================================================================

# Document-aware schema system - deferred initialization to avoid module-level import
_config = None


def _get_config():
    """
    Get schema configuration with deferred initialization.

    SIMPLIFIED: Now uses field_definitions_loader instead of complex unified_schema.
    """
    global _config
    if _config is None:
        # Use simplified field definitions loader
        from .field_definitions_loader import SimpleFieldLoader

        loader = SimpleFieldLoader()

        # Create simple config object with simplified fields
        class SimpleConfig:
            def __init__(self, loader):
                self.field_loader = loader

                # Get universal fields for EXTRACTION (includes ACCOUNT_BALANCE for math enhancement)
                # This includes invoice (14) + bank statement fields (STATEMENT_DATE_RANGE, TRANSACTION_DATES, TRANSACTION_AMOUNTS_PAID, ACCOUNT_BALANCE)
                self.extraction_fields = loader.get_document_fields("universal")
                # Only exclude TRANSACTION_AMOUNTS_RECEIVED from extraction
                # ACCOUNT_BALANCE IS extracted (for balance correction) but NOT evaluated (see VALIDATION_ONLY_FIELDS)
                _exclude_from_extraction = ["TRANSACTION_AMOUNTS_RECEIVED"]
                self.extraction_fields = [
                    f
                    for f in self.extraction_fields
                    if f not in _exclude_from_extraction
                ]
                self.field_count = len(self.extraction_fields)

                # Load field type classifications from YAML
                field_types_from_yaml = loader.get_field_types()

                # Simplified field types - all text for simplicity
                self.field_types = {field: "text" for field in self.extraction_fields}

                # Load field classifications from YAML config
                self.phone_fields = []
                self.list_fields = field_types_from_yaml.get("list", [])
                self.monetary_fields = field_types_from_yaml.get("monetary", [])
                self.numeric_id_fields = []
                self.date_fields = field_types_from_yaml.get("date", [])
                self.text_fields = field_types_from_yaml.get(
                    "text", self.extraction_fields
                )
                self.boolean_fields = field_types_from_yaml.get("boolean", [])
                self.calculated_fields = field_types_from_yaml.get("calculated", [])
                self.transaction_list_fields = field_types_from_yaml.get(
                    "transaction_list", []
                )

        _config = SimpleConfig(loader)
    return _config


# Module-level field variables (populated by _ensure_fields_loaded)
EXTRACTION_FIELDS = []
FIELD_COUNT = None
FIELD_TYPES = None
PHONE_FIELDS = None
LIST_FIELDS = None
MONETARY_FIELDS = None
NUMERIC_ID_FIELDS = None
DATE_FIELDS = None
TEXT_FIELDS = None
BOOLEAN_FIELDS = None
CALCULATED_FIELDS = None
TRANSACTION_LIST_FIELDS = None


def _ensure_fields_loaded():
    """Ensure field data is loaded from schema."""
    global EXTRACTION_FIELDS, FIELD_COUNT, FIELD_TYPES
    global \
        PHONE_FIELDS, \
        LIST_FIELDS, \
        MONETARY_FIELDS, \
        NUMERIC_ID_FIELDS, \
        DATE_FIELDS, \
        TEXT_FIELDS
    global BOOLEAN_FIELDS, CALCULATED_FIELDS, TRANSACTION_LIST_FIELDS

    if not EXTRACTION_FIELDS or BOOLEAN_FIELDS is None:
        config = _get_config()
        EXTRACTION_FIELDS = config.extraction_fields
        FIELD_COUNT = config.field_count
        FIELD_TYPES = config.field_types
        PHONE_FIELDS = config.phone_fields
        LIST_FIELDS = config.list_fields
        MONETARY_FIELDS = config.monetary_fields
        NUMERIC_ID_FIELDS = config.numeric_id_fields
        DATE_FIELDS = config.date_fields
        TEXT_FIELDS = config.text_fields
        BOOLEAN_FIELDS = config.boolean_fields
        CALCULATED_FIELDS = config.calculated_fields
        TRANSACTION_LIST_FIELDS = config.transaction_list_fields


# Initialize fields on module import
_ensure_fields_loaded()


def _ensure_initialized():
    """Ensure module-level variables are initialized."""
    _ensure_fields_loaded()


# ============================================================================
# PUBLIC FIELD ACCESSORS (used by evaluation_metrics.py)
# ============================================================================


def get_phone_fields():
    """Get phone fields."""
    _ensure_initialized()
    return PHONE_FIELDS


def get_list_fields():
    """Get list fields."""
    _ensure_initialized()
    return LIST_FIELDS


def get_monetary_fields():
    """Get monetary fields."""
    _ensure_initialized()
    return MONETARY_FIELDS


def get_all_field_types():
    """Get all field types."""
    _ensure_initialized()
    return FIELD_TYPES


def get_field_types():
    """Get field types (alias for get_all_field_types)."""
    return get_all_field_types()


def get_boolean_fields():
    """Get boolean fields."""
    _ensure_initialized()
    return BOOLEAN_FIELDS


def get_calculated_fields():
    """Get calculated fields."""
    _ensure_initialized()
    return CALCULATED_FIELDS


def get_transaction_list_fields():
    """Get transaction list fields."""
    _ensure_initialized()
    return TRANSACTION_LIST_FIELDS


def get_document_type_fields(document_type: str) -> list:
    """
    Get fields specific to document type for intelligent field filtering.

    This enables the document-aware approach where:
    - Invoice documents: 14 fields
    - Receipt documents: 14 fields
    - Bank statement documents: 5 fields (evaluation only, excludes validation-only fields)

    Args:
        document_type (str): Document type ('invoice', 'receipt', 'bank_statement')

    Returns:
        List[str]: Fields specific to the document type

    Raises:
        ValueError: If document type not supported
    """
    from .field_definitions_loader import SimpleFieldLoader

    loader = SimpleFieldLoader()

    # Map common document type variations
    doc_type_mapping = {
        "invoice": "invoice",
        "tax_invoice": "invoice",
        "bill": "invoice",
        "receipt": "receipt",
        "purchase_receipt": "receipt",
        "bank_statement": "bank_statement",
        "statement": "bank_statement",
    }

    mapped_type = doc_type_mapping.get(document_type.lower(), document_type.lower())

    try:
        field_names = loader.get_document_fields(mapped_type)
    except Exception as e:
        # Fallback to full field list if document-specific filtering fails
        print(f"Warning: Field filtering failed for '{document_type}': {e}")
        print("Falling back to full field list (filtered for evaluation)")
        return filter_evaluation_fields(EXTRACTION_FIELDS)

    # CRITICAL: Filter out validation-only fields from EVALUATION
    # ACCOUNT_BALANCE is extracted but excluded from accuracy metrics
    return filter_evaluation_fields(field_names)


# ============================================================================
# EVALUATION VS VALIDATION FIELD SEPARATION
# ============================================================================

# Fields EXTRACTED but EXCLUDED from evaluation metrics
# These are used for mathematical validation/correction but don't count toward accuracy
VALIDATION_ONLY_FIELDS = [
    "TRANSACTION_AMOUNTS_RECEIVED",
    "ACCOUNT_BALANCE",
]


def is_evaluation_field(field_name: str) -> bool:
    """
    Check if a field should be included in evaluation metrics.

    Args:
        field_name (str): Field name to check

    Returns:
        bool: True if field should be evaluated, False if validation-only
    """
    return field_name not in VALIDATION_ONLY_FIELDS


def filter_evaluation_fields(fields: list) -> list:
    """
    Filter a list of fields to exclude validation-only fields.

    Args:
        fields (list): List of field names

    Returns:
        list: Filtered list excluding validation-only fields
    """
    return [field for field in fields if is_evaluation_field(field)]
