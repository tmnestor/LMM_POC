"""
Shared configuration for document extraction models.

Configuration values for InternVL3 and Llama vision-language models.
Uses YAML-first field discovery and environment variables for deployment.
"""


# ============================================================================
# MODEL PATH DEFAULTS (fallback when CLI/YAML don't provide a path)
# ============================================================================

INTERNVL3_MODEL_PATH = "/efs/shared/PTM/InternVL3-8B"
LLAMA_MODEL_PATH = "/efs/shared/PTM/Llama-3.2-11B-Vision-Instruct"


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
# IMAGE PROCESSING CONSTANTS
# ============================================================================

# ImageNet normalization constants (for vision transformers)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default image size for processing
DEFAULT_IMAGE_SIZE = 448


# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Default batch sizes per model (Balanced for typical VRAM)
DEFAULT_BATCH_SIZES = {
    "internvl3": 4,
    "internvl3-2b": 4,
    "internvl3-8b": 4,
}

# Maximum batch sizes per model (Aggressive for 64GB+ VRAM)
MAX_BATCH_SIZES = {
    "internvl3": 8,
    "internvl3-2b": 8,
    "internvl3-8b": 16,
}

# Conservative batch sizes per model (Safe for limited memory)
CONSERVATIVE_BATCH_SIZES = {
    "internvl3": 1,
    "internvl3-2b": 2,
    "internvl3-8b": 1,
}

# Minimum batch size (always 1 for single image processing)
MIN_BATCH_SIZE = 1

# Automatic batch size detection settings
AUTO_BATCH_SIZE_ENABLED = True
BATCH_SIZE_MEMORY_SAFETY_MARGIN = 0.8

# Memory management settings
CLEAR_GPU_CACHE_AFTER_BATCH = True
BATCH_PROCESSING_TIMEOUT_SECONDS = 300

# Batch size optimization strategies
BATCH_SIZE_STRATEGIES = {
    "conservative": "Use minimum safe batch sizes for stability",
    "balanced": "Use default batch sizes for typical hardware",
    "aggressive": "Use maximum batch sizes for high-end hardware",
}

# Current strategy (can be changed for different deployment scenarios)
CURRENT_BATCH_STRATEGY = "balanced"

# GPU memory thresholds for automatic batch size selection
GPU_MEMORY_THRESHOLDS = {
    "low": 8,
    "medium": 16,
    "high": 24,
    "very_high": 64,
}

# Automatic fallback settings
ENABLE_BATCH_SIZE_FALLBACK = True
BATCH_SIZE_FALLBACK_STEPS = [8, 4, 2, 1]


def apply_yaml_overrides(raw_config: dict) -> None:
    """Override module-level constants from run_config.yml.

    Called once from CLI startup. YAML values replace hardcoded defaults.
    Only keys present in the YAML are overridden — missing keys keep
    their Python-level defaults.
    """
    global DEFAULT_BATCH_SIZES, MAX_BATCH_SIZES, CONSERVATIVE_BATCH_SIZES
    global MIN_BATCH_SIZE, CURRENT_BATCH_STRATEGY, AUTO_BATCH_SIZE_ENABLED
    global BATCH_SIZE_MEMORY_SAFETY_MARGIN, CLEAR_GPU_CACHE_AFTER_BATCH
    global BATCH_PROCESSING_TIMEOUT_SECONDS, ENABLE_BATCH_SIZE_FALLBACK
    global BATCH_SIZE_FALLBACK_STEPS, GPU_MEMORY_THRESHOLDS
    global INTERNVL3_GENERATION_CONFIG, GENERATION_CONFIGS
    global INTERNVL3_TOKEN_LIMITS

    # --- Batch processing ---
    batch = raw_config.get("batch", {})
    if batch:
        if "default_sizes" in batch:
            DEFAULT_BATCH_SIZES.update(batch["default_sizes"])
        if "max_sizes" in batch:
            MAX_BATCH_SIZES.update(batch["max_sizes"])
        if "conservative_sizes" in batch:
            CONSERVATIVE_BATCH_SIZES.update(batch["conservative_sizes"])
        if "min_size" in batch:
            MIN_BATCH_SIZE = batch["min_size"]
        if "strategy" in batch:
            CURRENT_BATCH_STRATEGY = batch["strategy"]
        if "auto_detect" in batch:
            AUTO_BATCH_SIZE_ENABLED = batch["auto_detect"]
        if "memory_safety_margin" in batch:
            BATCH_SIZE_MEMORY_SAFETY_MARGIN = batch["memory_safety_margin"]
        if "clear_cache_after_batch" in batch:
            CLEAR_GPU_CACHE_AFTER_BATCH = batch["clear_cache_after_batch"]
        if "timeout_seconds" in batch:
            BATCH_PROCESSING_TIMEOUT_SECONDS = batch["timeout_seconds"]
        if "fallback_enabled" in batch:
            ENABLE_BATCH_SIZE_FALLBACK = batch["fallback_enabled"]
        if "fallback_steps" in batch:
            BATCH_SIZE_FALLBACK_STEPS = batch["fallback_steps"]

    # --- Generation parameters ---
    gen = raw_config.get("generation", {})
    if gen:
        if "max_new_tokens_base" in gen:
            INTERNVL3_GENERATION_CONFIG["max_new_tokens_base"] = gen[
                "max_new_tokens_base"
            ]
        if "max_new_tokens_per_field" in gen:
            INTERNVL3_GENERATION_CONFIG["max_new_tokens_per_field"] = gen[
                "max_new_tokens_per_field"
            ]
        if "do_sample" in gen:
            INTERNVL3_GENERATION_CONFIG["do_sample"] = gen["do_sample"]
            if "internvl3" in GENERATION_CONFIGS:
                GENERATION_CONFIGS["internvl3"]["do_sample"] = gen["do_sample"]
        if "use_cache" in gen:
            INTERNVL3_GENERATION_CONFIG["use_cache"] = gen["use_cache"]
        if "num_beams" in gen:
            if "internvl3" in GENERATION_CONFIGS:
                GENERATION_CONFIGS["internvl3"]["num_beams"] = gen["num_beams"]
        if "repetition_penalty" in gen:
            if "internvl3" in GENERATION_CONFIGS:
                GENERATION_CONFIGS["internvl3"]["repetition_penalty"] = gen[
                    "repetition_penalty"
                ]

        token_limits = gen.get("token_limits", {})
        if token_limits:
            for size_key, value in token_limits.items():
                INTERNVL3_TOKEN_LIMITS[str(size_key)] = value

    # --- GPU memory thresholds ---
    gpu = raw_config.get("gpu", {})
    if gpu:
        thresholds = gpu.get("memory_thresholds", {})
        if thresholds:
            GPU_MEMORY_THRESHOLDS.update(thresholds)


def get_model_name_with_size(
    base_model_name: str, model_path: str = None, is_8b_model: bool = None
) -> str:
    """
    Generate size-aware model name for batch size configuration lookup.

    Args:
        base_model_name (str): Base model name ('internvl3', etc.)
        model_path (str): Path to model (used for size detection if is_8b_model not provided)
        is_8b_model (bool): Whether model is 8B variant (overrides path detection)

    Returns:
        str: Size-aware model name ('internvl3-2b', 'internvl3-8b', or original name)
    """
    base_name = base_model_name.lower()

    # Only modify internvl3 models - other models use original names
    if base_name != "internvl3":
        return base_name

    # Determine if this is an 8B model
    if is_8b_model is None and model_path:
        is_8b_model = "8B" in str(model_path)

    # Return size-specific model name for InternVL3
    if is_8b_model:
        return "internvl3-8b"
    else:
        return "internvl3-2b"


def get_batch_size_for_model(model_name: str, strategy: str = None) -> int:
    """
    Get recommended batch size for a model based on strategy.

    Args:
        model_name (str): Model name ('internvl3', 'internvl3-2b', 'internvl3-8b')
        strategy (str): Batching strategy ('conservative', 'balanced', 'aggressive')

    Returns:
        int: Recommended batch size
    """
    strategy = strategy or CURRENT_BATCH_STRATEGY
    model_name = model_name.lower()

    if strategy == "conservative":
        return CONSERVATIVE_BATCH_SIZES.get(model_name, MIN_BATCH_SIZE)
    elif strategy == "aggressive":
        return MAX_BATCH_SIZES.get(model_name, MIN_BATCH_SIZE)
    else:  # balanced
        return DEFAULT_BATCH_SIZES.get(model_name, MIN_BATCH_SIZE)


def get_auto_batch_size(model_name: str, available_memory_gb: float = None) -> int:
    """
    Automatically determine batch size based on available GPU memory.

    Args:
        model_name (str): Model name ('internvl3', 'internvl3-2b', 'internvl3-8b')
        available_memory_gb (float): Available GPU memory in GB

    Returns:
        int: Recommended batch size based on available memory
    """
    if not AUTO_BATCH_SIZE_ENABLED or available_memory_gb is None:
        return get_batch_size_for_model(model_name, CURRENT_BATCH_STRATEGY)

    # Determine memory tier
    if available_memory_gb >= GPU_MEMORY_THRESHOLDS["very_high"]:
        strategy = "aggressive"
    elif available_memory_gb >= GPU_MEMORY_THRESHOLDS["high"]:
        strategy = "aggressive"
    elif available_memory_gb >= GPU_MEMORY_THRESHOLDS["medium"]:
        strategy = "balanced"
    else:
        strategy = "conservative"

    return get_batch_size_for_model(model_name, strategy)


# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================

# InternVL3 generation configuration
INTERNVL3_GENERATION_CONFIG = {
    "max_new_tokens_base": 2000,
    "max_new_tokens_per_field": 50,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": None,  # Set dynamically from tokenizer
}

# Llama generation configuration
LLAMA_GENERATION_CONFIG = {
    "max_new_tokens_base": 400,
    "max_new_tokens_per_field": 50,
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 0.95,
    "use_cache": True,
}

# Per-model generation parameters (for YAML overrides)
GENERATION_CONFIGS = {
    "internvl3": {
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    },
}

# Token limits for different model sizes
INTERNVL3_TOKEN_LIMITS = {
    "2b": None,  # Use get_max_new_tokens() calculation
    "8b": 800,
}

# Module-level cache for per-type min_tokens from field_definitions.yaml
_MIN_TOKENS_CACHE: dict[str, int] | None = None


def _get_min_tokens_for_type(document_type: str) -> int | None:
    """Look up min_tokens for a document type from field_definitions.yaml.

    Returns None if not specified for the given type.
    """
    global _MIN_TOKENS_CACHE  # noqa: PLW0603
    if _MIN_TOKENS_CACHE is None:
        from pathlib import Path

        import yaml

        _MIN_TOKENS_CACHE = {}
        yaml_path = Path(__file__).parent.parent / "config" / "field_definitions.yaml"
        try:
            with yaml_path.open() as f:
                data = yaml.safe_load(f)
            for doc_type, type_config in data.get("document_fields", {}).items():
                if isinstance(type_config, dict) and "min_tokens" in type_config:
                    _MIN_TOKENS_CACHE[doc_type] = type_config["min_tokens"]
        except Exception:
            pass

    return _MIN_TOKENS_CACHE.get(document_type)


def get_max_new_tokens(field_count: int = None, document_type: str = None) -> int:
    """
    Calculate max_new_tokens based on field count and document complexity.

    Args:
        field_count (int): Number of extraction fields (uses FIELD_COUNT if None)
        document_type (str): Document type ('bank_statement', 'invoice', 'receipt', etc.)

    Returns:
        int: Calculated max_new_tokens value
    """
    field_count = field_count or FIELD_COUNT or 15

    config = INTERNVL3_GENERATION_CONFIG

    base_tokens = max(
        config["max_new_tokens_base"], field_count * config["max_new_tokens_per_field"]
    )

    # Apply per-type min_tokens from field_definitions.yaml if present
    if document_type:
        min_tokens = _get_min_tokens_for_type(document_type)
        if min_tokens:
            return max(base_tokens, min_tokens)

    return base_tokens


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
