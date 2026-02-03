"""
Shared configuration for InternVL3.5-8B vision model evaluation.

This module contains all configuration values and constants for
the InternVL3.5-8B vision-language model.

NOTE: Uses YAML-first field discovery for single source of truth.
NOTE: Supports environment variables for flexible deployment configuration.
"""


# ============================================================================
# MODEL CONFIGURATIONS - PRIMARY HIERARCHY
# ============================================================================

# Available model variants
AVAILABLE_MODELS = {
    "internvl3": ["InternVL3-2B", "InternVL3-8B"],
}

# Current model selection (CHANGE THIS TO SWITCH MODELS)
CURRENT_INTERNVL3_MODEL = "InternVL3-8B"  # Options: "InternVL3-2B", "InternVL3-8B"

# V4 Schema Configuration - Enable by default
V4_SCHEMA_ENABLED = True

# ============================================================================
# DEPLOYMENT CONFIGURATIONS
# ============================================================================

# Base paths for different deployment scenarios
BASE_PATHS = {"AISandbox": "/home/jovyan/nfs_share", "efs": "/efs/shared"}

# Current deployment (change this to switch environments)
CURRENT_DEPLOYMENT = "efs"

# ============================================================================
# DYNAMIC PATH GENERATION - MODEL-DRIVEN INTERPOLATION
# ============================================================================

# Dynamic path generation using model + deployment interpolation
BASE_PATH = BASE_PATHS[CURRENT_DEPLOYMENT]

# Model directory structure varies by deployment
MODELS_BASE = (
    f"{BASE_PATH}/models" if CURRENT_DEPLOYMENT == "AISandbox" else f"{BASE_PATH}/PTM"
)

# Model path based on current model selection
INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"

# Data paths based on deployment
if CURRENT_DEPLOYMENT == "AISandbox":
    DATA_BASE = f"{BASE_PATH}/tod/LMM_POC"
    DATA_DIR = f"{DATA_BASE}/evaluation_data"
    GROUND_TRUTH_PATH = f"{DATA_DIR}/ground_truth.csv"
    OUTPUT_DIR = f"{BASE_PATH}/tod/output"
else:  # EFS deployment
    DATA_BASE = f"{BASE_PATH}/PoC_data"
    DATA_DIR = f"{DATA_BASE}/evaluation_data"
    GROUND_TRUTH_PATH = f"{DATA_DIR}/ground_truth.csv"
    OUTPUT_DIR = f"{DATA_BASE}/output"


def switch_model(model_name: str):
    """
    Switch to a different InternVL3 model variant.

    Args:
        model_name (str): Specific model name from AVAILABLE_MODELS['internvl3']
    """
    global CURRENT_INTERNVL3_MODEL, INTERNVL3_MODEL_PATH

    if model_name not in AVAILABLE_MODELS["internvl3"]:
        raise ValueError(
            f"Invalid model: {model_name}. Valid options: {AVAILABLE_MODELS['internvl3']}"
        )

    CURRENT_INTERNVL3_MODEL = model_name
    INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
    print(f"✅ Switched to {model_name}")
    print(f"   Path: {INTERNVL3_MODEL_PATH}")


def switch_deployment(deployment: str):
    """
    Switch to a different deployment environment.

    Args:
        deployment (str): Deployment type ('AISandbox' or 'efs')
    """
    global CURRENT_DEPLOYMENT, BASE_PATH, MODELS_BASE
    global INTERNVL3_MODEL_PATH
    global DATA_BASE, DATA_DIR, GROUND_TRUTH_PATH, OUTPUT_DIR

    if deployment not in BASE_PATHS:
        raise ValueError(
            f"Invalid deployment: {deployment}. Valid options: {list(BASE_PATHS.keys())}"
        )

    CURRENT_DEPLOYMENT = deployment
    BASE_PATH = BASE_PATHS[CURRENT_DEPLOYMENT]

    # Update model path using current model selection
    MODELS_BASE = (
        f"{BASE_PATH}/models"
        if CURRENT_DEPLOYMENT == "AISandbox"
        else f"{BASE_PATH}/PTM"
    )

    INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"

    # Update data paths
    if CURRENT_DEPLOYMENT == "AISandbox":
        DATA_BASE = f"{BASE_PATH}/tod/LMM_POC"
        DATA_DIR = f"{DATA_BASE}/evaluation_data"
        GROUND_TRUTH_PATH = f"{DATA_DIR}/ground_truth.csv"
        OUTPUT_DIR = f"{BASE_PATH}/tod/output"
    else:  # EFS deployment
        DATA_BASE = f"{BASE_PATH}/PoC_data"
        DATA_DIR = f"{DATA_BASE}/evaluation_data"
        GROUND_TRUTH_PATH = f"{DATA_DIR}/ground_truth.csv"
        OUTPUT_DIR = f"{DATA_BASE}/output"

    print(f"✅ Switched to {deployment} deployment")
    print(f"   Models: {MODELS_BASE}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")


def show_current_config():
    """Display current model and deployment configuration."""
    print("🔧 Current Configuration:")
    print(f"   Deployment: {CURRENT_DEPLOYMENT}")
    print()
    print("📁 Paths:")
    print(f"   Base Path: {BASE_PATH}")
    print(f"   Models: {MODELS_BASE}")
    print(f"   Data Dir: {DATA_DIR}")
    print(f"   Output Dir: {OUTPUT_DIR}")
    print(f"   Ground Truth: {GROUND_TRUTH_PATH}")
    print()
    print("🤖 Model:")
    print(f"   InternVL3: {CURRENT_INTERNVL3_MODEL}")
    print(f"   Path: {INTERNVL3_MODEL_PATH}")


# ============================================================================
# DYNAMIC SCHEMA-BASED FIELD DISCOVERY
# ============================================================================

# ============================================================================
# FIELD METADATA - OPTIONAL OVERRIDES ONLY
# ============================================================================

# Field definitions now managed by schema - no hardcoded definitions needed!
# All field metadata is generated dynamically from field_schema.yaml

# ============================================================================
# FIELD DISCOVERY - YAML ORDER IS THE TRUTH
# ============================================================================

# The YAML files define the field order. We use it as-is. No reordering logic needed.

# ============================================================================
# DERIVED CONFIGURATIONS - AUTO-GENERATED FROM FIELD_DEFINITIONS
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
                self.extraction_fields = [f for f in self.extraction_fields if f not in _exclude_from_extraction]
                self.field_count = len(self.extraction_fields)
                self.active_field_count = len(self.extraction_fields)

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
                self.text_fields = field_types_from_yaml.get("text", self.extraction_fields)
                self.boolean_fields = field_types_from_yaml.get("boolean", [])
                self.calculated_fields = field_types_from_yaml.get("calculated", [])
                self.transaction_list_fields = field_types_from_yaml.get("transaction_list", [])

        _config = SimpleConfig(loader)
    return _config


def get_document_schema():
    """Get document field loader."""
    return _get_config().field_loader


# Schema loader and fields - deferred access
def _get_extraction_fields():
    return _get_config().extraction_fields


def _get_field_count():
    return _get_config().field_count


def _get_field_types():
    return _get_config().field_types


def _get_phone_fields():
    return _get_config().phone_fields


def _get_list_fields():
    return _get_config().list_fields


def _get_monetary_fields():
    return _get_config().monetary_fields


def _get_numeric_id_fields():
    return _get_config().numeric_id_fields


def _get_date_fields():
    return _get_config().date_fields


def _get_text_fields():
    return _get_config().text_fields


def _get_boolean_fields():
    return _get_config().boolean_fields


def _get_calculated_fields():
    return _get_config().calculated_fields


def _get_transaction_list_fields():
    return _get_config().transaction_list_fields


# Module-level access via function calls (no module-level initialization)
EXTRACTION_FIELDS = []  # Will be set on first access

# Initialize module-level variables (will be populated by _ensure_fields_loaded)
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
        # Use simplified schema
        config = _get_config()
        EXTRACTION_FIELDS = config.extraction_fields
        FIELD_COUNT = config.field_count
        # Use the field_types dict that's already available
        FIELD_TYPES = config.field_types

        # Initialize all field type lists
        PHONE_FIELDS = config.phone_fields
        LIST_FIELDS = config.list_fields
        MONETARY_FIELDS = config.monetary_fields
        NUMERIC_ID_FIELDS = config.numeric_id_fields
        DATE_FIELDS = config.date_fields
        TEXT_FIELDS = config.text_fields

        # Initialize new v4 field types
        BOOLEAN_FIELDS = config.boolean_fields
        CALCULATED_FIELDS = config.calculated_fields
        TRANSACTION_LIST_FIELDS = config.transaction_list_fields


# Initialize fields on module import for backward compatibility
_ensure_fields_loaded()


def _ensure_initialized():
    """Ensure module-level variables are initialized."""
    _ensure_fields_loaded()  # Use the new initialization function


def get_document_schema_loader():
    """Get document schema loader (alias for compatibility)."""
    return _get_config().schema_loader


# All fields are required for extraction (must attempt to extract and return value or NOT_FOUND)


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


def get_extraction_fields():
    """Get extraction fields."""
    _ensure_initialized()
    return EXTRACTION_FIELDS


def get_field_count():
    """Get field count."""
    _ensure_initialized()
    return FIELD_COUNT


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

# ============================================================================
# IMAGE PROCESSING CONSTANTS
# ============================================================================

# ImageNet normalization constants (for vision transformers)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default image size for processing
DEFAULT_IMAGE_SIZE = 448

# ============================================================================
# EVALUATION METRICS THRESHOLDS
# ============================================================================

# Accuracy thresholds for deployment readiness
DEPLOYMENT_READY_THRESHOLD = 0.9  # 90% accuracy for production
PILOT_READY_THRESHOLD = 0.8  # 80% accuracy for pilot testing
NEEDS_OPTIMIZATION_THRESHOLD = 0.7  # Below 70% needs major improvements

# Field-specific accuracy thresholds
EXCELLENT_FIELD_THRESHOLD = 0.9  # Fields with ≥90% accuracy
GOOD_FIELD_THRESHOLD = 0.8  # Fields with ≥80% accuracy
POOR_FIELD_THRESHOLD = 0.5  # Fields with <50% accuracy

# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================

# Output file patterns
EXTRACTION_OUTPUT_PATTERN = "{model}_batch_extraction_{timestamp}.csv"
METADATA_OUTPUT_PATTERN = "{model}_extraction_metadata_{timestamp}.csv"
EVALUATION_OUTPUT_PATTERN = "{model}_evaluation_results_{timestamp}.json"
EXECUTIVE_SUMMARY_PATTERN = "{model}_executive_summary_{timestamp}.md"
DEPLOYMENT_CHECKLIST_PATTERN = "{model}_deployment_checklist_{timestamp}.md"

# ============================================================================
# SUPPORTED IMAGE FORMATS
# ============================================================================

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]


# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Default batch sizes per model (Balanced for 16GB VRAM)
DEFAULT_BATCH_SIZES = {
    "internvl3": 4,  # InternVL3 generic fallback (backward compatibility)
    "internvl3-2b": 4,  # InternVL3-2B is memory efficient, can handle larger batches
    "internvl3-8b": 1,  # InternVL3-8B with quantization needs conservative batching
}

# ============================================================================
# MODEL-SPECIFIC GENERATION CONFIGURATION
# ============================================================================

# Token limits for different model sizes with quantization
INTERNVL3_TOKEN_LIMITS = {
    "2b": None,  # Use get_max_new_tokens() calculation
    "8b": 800,  # Enough for all 25 fields with buffer after 8-bit quantization
}

# Generation parameters for InternVL3
GENERATION_CONFIGS = {
    "internvl3": {
        "do_sample": False,  # CRITICAL: Must be False for deterministic output (greedy decoding)
        # When do_sample=False, temperature/top_k/top_p are ignored and cause warnings
        # So we don't set them - greedy decoding automatically selects highest probability token
        "num_beams": 1,  # No beam search - single path only
        "repetition_penalty": 1.0,  # No repetition penalty
        # Note: seed is set at system level in _set_random_seeds(), not in generation config
    },
}

# Maximum batch sizes per model (Aggressive for 24GB+ VRAM)
MAX_BATCH_SIZES = {
    "internvl3": 8,  # InternVL3 generic fallback (backward compatibility)
    "internvl3-2b": 8,  # InternVL3-2B can handle large batches on high-end GPUs
    "internvl3-8b": 2,  # InternVL3-8B maximum safe batch size even on powerful GPUs
}

# Conservative batch sizes per model (Safe for limited memory situations)
CONSERVATIVE_BATCH_SIZES = {
    "internvl3": 1,  # InternVL3 generic fallback (backward compatibility)
    "internvl3-2b": 2,  # InternVL3-2B can safely handle 2 even in conservative mode
    "internvl3-8b": 1,  # InternVL3-8B must stay at 1 for safety
}

# Minimum batch size (always 1 for single image processing)
MIN_BATCH_SIZE = 1

# Automatic batch size detection settings
AUTO_BATCH_SIZE_ENABLED = True
BATCH_SIZE_MEMORY_SAFETY_MARGIN = 0.8  # Use 80% of available memory for batch sizing

# Memory management settings
CLEAR_GPU_CACHE_AFTER_BATCH = True
BATCH_PROCESSING_TIMEOUT_SECONDS = 300  # 5 minutes per batch maximum

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
    "low": 8,  # GB - Use conservative batching
    "medium": 16,  # GB - Use default batching
    "high": 24,  # GB - Use aggressive batching
}

# Automatic fallback settings
ENABLE_BATCH_SIZE_FALLBACK = True
BATCH_SIZE_FALLBACK_STEPS = [8, 4, 2, 1]  # Try these batch sizes if OOM occurs

# ============================================================================
# TILE CONFIGURATION - For OCR Quality Optimization
# ============================================================================

# InternVL3 tile counts - higher = better OCR but more memory
# V100 testing shows OOM issues above 12-14 tiles during generation phase
INTERNVL3_MAX_TILES_8B = (
    14  # Optimized for V100: Balance between OCR quality and memory
)
INTERNVL3_MAX_TILES_2B = 18  # 2B model can use more tiles with lower memory footprint


def get_model_name_with_size(
    base_model_name: str, model_path: str = None, is_8b_model: bool = None
) -> str:
    """
    Generate size-aware model name for batch size configuration lookup.

    Args:
        base_model_name (str): Base model name ('internvl3', 'llama', etc.)
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
        model_name (str): Model name ('llama', 'internvl3', 'internvl3-2b', 'internvl3-8b')
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
        model_name (str): Model name ('llama', 'internvl3', 'internvl3-2b', 'internvl3-8b')
        available_memory_gb (float): Available GPU memory in GB

    Returns:
        int: Recommended batch size based on available memory
    """
    if not AUTO_BATCH_SIZE_ENABLED or available_memory_gb is None:
        return get_batch_size_for_model(model_name, CURRENT_BATCH_STRATEGY)

    # Determine memory tier
    if available_memory_gb >= GPU_MEMORY_THRESHOLDS["high"]:
        strategy = "aggressive"
    elif available_memory_gb >= GPU_MEMORY_THRESHOLDS["medium"]:
        strategy = "balanced"
    else:
        strategy = "conservative"

    return get_batch_size_for_model(model_name, strategy)


# ============================================================================
# GENERATION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# InternVL3 generation configuration
INTERNVL3_GENERATION_CONFIG = {
    "max_new_tokens_base": 2000,  # Increased for complex bank statements (4 V100 setup)
    "max_new_tokens_per_field": 50,  # Additional tokens per extraction field
    "temperature": 0.0,  # Deterministic sampling for consistent results
    "do_sample": False,  # Deterministic for consistent field extraction
    "use_cache": True,  # CRITICAL parameter - required for extraction quality
    "pad_token_id": None,  # Set dynamically from tokenizer
}


# Helper function to calculate dynamic max_new_tokens
def get_max_new_tokens(field_count: int = None, document_type: str = None) -> int:
    """
    Calculate max_new_tokens based on field count and document complexity.

    Args:
        field_count (int): Number of extraction fields (uses FIELD_COUNT if None)
        document_type (str): Document type ('bank_statement', 'invoice', 'receipt', etc.)

    Returns:
        int: Calculated max_new_tokens value
    """
    field_count = (
        field_count or FIELD_COUNT or 15
    )  # Default to 15 for universal extraction

    config = INTERNVL3_GENERATION_CONFIG

    base_tokens = max(
        config["max_new_tokens_base"], field_count * config["max_new_tokens_per_field"]
    )

    # Special handling for complex documents that may output JSON with many transactions
    if document_type == "bank_statement":
        # Bank statements can have many transactions, need significantly more tokens for JSON format
        return max(base_tokens, 1500)  # Ensure at least 1500 tokens for complex bank statements

    return base_tokens


# ============================================================================
# V4 SCHEMA INTEGRATION FUNCTIONS
# ============================================================================


def get_v4_field_list() -> list:
    """
    Get all unique fields from v4 schema (49 total fields).

    This is the main function for V4 schema integration that returns
    all 49 unique fields across all document types.

    Returns:
        List[str]: All 49 unique field names from V4 schema
    """
    _ensure_initialized()
    return EXTRACTION_FIELDS


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
        print(f"⚠️ Document-specific field filtering failed for '{document_type}': {e}")
        print("🔄 Falling back to full field list (filtered for evaluation)")
        return filter_evaluation_fields(get_v4_field_list())

    # CRITICAL: Filter out validation-only fields from EVALUATION
    # ACCOUNT_BALANCE is extracted but excluded from accuracy metrics
    return filter_evaluation_fields(field_names)


def get_v4_field_count() -> int:
    """
    Get the total V4 schema field count (49).

    Returns:
        int: Total number of fields in V4 schema
    """
    return len(get_v4_field_list())


def is_v4_schema_enabled() -> bool:
    """
    Check if V4 schema is currently enabled.

    Returns:
        bool: True if V4 schema is enabled (configurable via V4_SCHEMA_ENABLED)
    """
    return V4_SCHEMA_ENABLED


def get_v4_new_fields() -> list:
    """
    Get fields that were added in V4 schema (not present in V3).

    Returns:
        List[str]: Fields added in V4 schema
    """
    v4_new_fields = [
        # Enhanced business details
        "SUPPLIER_EMAIL",
        "PAYER_ABN",
        # Document references
        "INVOICE_NUMBER",
        "RECEIPT_NUMBER",
        # Enhanced line items
        "LINE_ITEM_TOTAL_PRICES",
        "LINE_ITEM_GST_AMOUNTS",
        "LINE_ITEM_DISCOUNT_AMOUNTS",
        # Enhanced monetary
        "TOTAL_DISCOUNT_AMOUNT",
        "IS_GST_INCLUDED",
        # Payment status (new category)
        "TOTAL_AMOUNT_PAID",
        "BALANCE_OF_PAYMENT",
        "TOTAL_AMOUNT_PAYABLE",
        # Transaction details (new category)
        "TRANSACTION_DATES",
        "TRANSACTION_DESCRIPTIONS",
        "TRANSACTION_AMOUNTS_PAID",
        "TRANSACTION_AMOUNTS_RECEIVED",
        "TRANSACTION_BALANCES",
        "CREDIT_CARD_DUE_DATE",
    ]

    # Return only fields that exist in current schema
    all_fields = get_v4_field_list()
    return [field for field in v4_new_fields if field in all_fields]


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Enable/disable visualization generation
VISUALIZATION_ENABLED = True

# Chart output settings
CHART_OUTPUT_FORMAT = "png"  # png, svg, pdf
CHART_DPI = 300  # High DPI for publication quality reports
CHART_STYLE = "professional"  # professional, minimal, academic

# Figure size settings (width, height in inches)
# High DPI + smaller physical size = high quality but manageable file size
# For reports: 300 DPI with 8-10 inch width provides excellent print quality
CHART_SIZES = {
    "field_accuracy": (10, 6),  # Field accuracy bar chart - compact but readable
    "performance_dashboard": (10, 8),  # 2x2 performance dashboard - balanced layout
    "field_category": (10, 5),  # Field category analysis - wide but not tall
    "document_quality": (8, 5),  # Document quality distribution - compact
    "comparison_heatmap": (
        12,
        8,
    ),  # Multi-model comparison - slightly larger for complexity
    "classification_metrics": (
        12,
        8,
    ),  # Classification metrics dashboard - comprehensive layout
}

# Professional color scheme for business reports
VIZ_COLORS = {
    "primary": "#2E86AB",  # Professional blue
    "secondary": "#A23B72",  # Deep purple
    "success": "#F18F01",  # Warm orange
    "warning": "#C73E1D",  # Alert red
    "info": "#4ECDC4",  # Teal accent
    "text": "#2C3E50",  # Dark text
    "background": "#F8F9FA",  # Light background
}

# Chart quality thresholds for color coding
VIZ_QUALITY_THRESHOLDS = {
    "excellent": 0.9,  # 90%+ accuracy = green
    "good": 0.8,  # 80-90% accuracy = yellow
    "poor": 0.6,  # <60% accuracy = red
}

# Visualization output file patterns
VIZ_OUTPUT_PATTERNS = {
    "field_accuracy": "{model}_field_accuracy_bar_{timestamp}.png",
    "performance_dashboard": "{model}_performance_dashboard_{timestamp}.png",
    "document_quality": "{model}_document_quality_{timestamp}.png",
    "field_category": "field_category_analysis_{timestamp}.png",
    "comparison_heatmap": "comparison_field_heatmap_{timestamp}.png",
    "classification_metrics": "{model}_classification_metrics_{timestamp}.png",
    "html_summary": "visualization_summary_{timestamp}.html",
}


# ============================================================================
# GROUPED EXTRACTION CONFIGURATION
# ============================================================================

# Field group definitions for grouped extraction strategy
# DOCUMENT AWARE REDUCTION: Drastically reduced field groups for performance
FIELD_GROUPS = {
    "regulatory_financial": {
        # OLD_COUNT: 6 fields
        # NEW_COUNT: 3 fields (50% reduction)
        "fields": [
            "BUSINESS_ABN",  # SUBSET: Essential business identifier
            "TOTAL_AMOUNT",  # SUBSET: Essential financial total
            # SUPER_SET: "ACCOUNT_OPENING_BALANCE",    # Removed from boss's reduced schema
            # SUPER_SET: "ACCOUNT_CLOSING_BALANCE",    # Removed from boss's reduced schema
            # SUPER_SET: "SUBTOTAL_AMOUNT",            # Removed from boss's reduced schema
            "GST_AMOUNT",  # SUBSET: Essential tax information
        ],
        "expertise_frame": "Extract business ID and financial amounts.",
        "cognitive_context": "BUSINESS_ABN is 11 digits. TOTAL_AMOUNT is final amount due. GST_AMOUNT is tax.",
        "focus_instruction": "Find ABN (11 digits) and essential dollar amounts. Check decimal places carefully.",
    },
    "entity_contacts": {
        # OLD_COUNT: 8 fields
        # NEW_COUNT: 4 fields (50% reduction)
        "fields": [
            "SUPPLIER_NAME",  # SUBSET: Essential supplier info
            "BUSINESS_ADDRESS",  # SUBSET: Essential supplier location
            # SUPER_SET: "BUSINESS_PHONE",             # Removed from boss's reduced schema
            # SUPER_SET: "SUPPLIER_WEBSITE",           # Removed from boss's reduced schema
            "PAYER_NAME",  # SUBSET: Essential payer info
            "PAYER_ADDRESS",  # SUBSET: Essential payer location
            # SUPER_SET: "PAYER_PHONE",                # Removed from boss's reduced schema
            # SUPER_SET: "PAYER_EMAIL"                 # Removed from boss's reduced schema
        ],
        "expertise_frame": "Extract essential contact information for supplier and customer.",
        "cognitive_context": "SUPPLIER_NAME, BUSINESS_ADDRESS are supplier details. PAYER_NAME, PAYER_ADDRESS are customer details.",
        "focus_instruction": "Extract essential contact details. Focus on names and addresses. Australian postcodes are 4 digits.",
    },
    "transaction_details": {
        # OLD_COUNT: 3 fields
        # NEW_COUNT: 2 fields (33% reduction)
        "fields": [
            "LINE_ITEM_DESCRIPTIONS",  # SUBSET: Essential line item data
            # SUPER_SET: "LINE_ITEM_QUANTITIES",       # Removed from boss's reduced schema
            # SUPER_SET: "LINE_ITEM_PRICES",           # Removed from boss's reduced schema
            "LINE_ITEM_TOTAL_PRICES",  # SUBSET: Essential line item totals
        ],
        "expertise_frame": "Extract essential line item information.",
        "cognitive_context": "DESCRIPTIONS: Every product/service name. TOTAL_PRICES: Final price for each line item.",
        "focus_instruction": "Extract line item descriptions and total prices only. Use PIPE-SEPARATED format.",
    },
    "temporal_data": {
        # OLD_COUNT: 3 fields
        # NEW_COUNT: 2 fields (33% reduction)
        "fields": [
            "INVOICE_DATE",  # SUBSET: Essential invoice temporal data
            # SUPER_SET: "DUE_DATE",                   # Removed from boss's reduced schema
            "STATEMENT_DATE_RANGE",  # SUBSET: Essential bank statement temporal data
        ],
        "expertise_frame": "Extract essential document dates.",
        "cognitive_context": "INVOICE_DATE is issue date. STATEMENT_DATE_RANGE is for bank statements only.",
        "focus_instruction": "Find essential dates. Convert to consistent DD/MM/YYYY format where possible.",
    },
    # SUPER_SET: Entire banking_payment group removed due to boss's field reduction
    # "banking_payment": {
    #     "fields": ["BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_HOLDER"],
    #     "expertise_frame": "Extract banking information.",
    #     "cognitive_context": "BANK_NAME is financial institution. BSB_NUMBER is 6 digits. BANK_ACCOUNT_NUMBER varies. ACCOUNT_HOLDER is account name. Typically on bank statements only.",
    #     "focus_instruction": "Extract banking details if present. BSB is 6 digits, different from 11-digit ABN."
    # },
    "document_metadata": {
        # OLD_COUNT: 3 fields
        # NEW_COUNT: 1 field (67% reduction)
        "fields": [
            "DOCUMENT_TYPE"  # SUBSET: Essential document identification
            # SUPER_SET: "RECEIPT_NUMBER",             # Removed from boss's reduced schema
            # SUPER_SET: "STORE_LOCATION"              # Removed from boss's reduced schema
        ],
        "expertise_frame": "Extract essential document identifiers.",
        "cognitive_context": "DOCUMENT_TYPE: invoice, receipt, or statement.",
        "focus_instruction": "Extract document type only.",
    },
    # NEW: Bank statement transaction group for boss's reduced schema
    "bank_transactions": {
        # NEW_COUNT: 3 fields (specialized for bank statements including calculated fields)
        "fields": [
            "TRANSACTION_DATES",  # SUBSET: Essential transaction dates
            "TRANSACTION_AMOUNTS_PAID",  # SUBSET: Essential transaction amounts
            "TRANSACTION_AMOUNTS_RECEIVED",  # SUBSET: Calculated transaction amounts received
        ],
        "expertise_frame": "Extract bank statement transaction data including calculated amounts.",
        "cognitive_context": "TRANSACTION_DATES are when transactions occurred. TRANSACTION_AMOUNTS_PAID and TRANSACTION_AMOUNTS_RECEIVED are debit/credit amounts.",
        "focus_instruction": "Extract transaction dates and amounts from bank statements. Include both paid and received amounts.",
    },
}

# Grouping strategies configuration
GROUPING_STRATEGIES = {
    "detailed_grouped": FIELD_GROUPS,
    "6_groups": FIELD_GROUPS,  # Alias for backward compatibility
    "8_groups": FIELD_GROUPS,  # Alias for backward compatibility
}

# Group validation rules
GROUP_VALIDATION_RULES = {
    "min_fields_per_group": 1,
    "max_fields_per_group": 20,
    "required_groups": ["regulatory_financial", "entity_contacts"],
    "optional_groups": [
        "transaction_details",
        "temporal_data",
        "banking_payment",
        "document_metadata",
    ],
}

# ============================================================================
# EVALUATION VS VALIDATION FIELD SEPARATION
# ============================================================================

# Fields EXTRACTED but EXCLUDED from evaluation metrics
# These are used for mathematical validation/correction but don't count toward accuracy
VALIDATION_ONLY_FIELDS = [
    "TRANSACTION_AMOUNTS_RECEIVED",  # Used for mathematical transaction calculation
    "ACCOUNT_BALANCE",  # Extracted for balance correction, but not evaluated
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

def get_validation_only_fields() -> list:
    """
    Get list of fields used only for validation (not evaluation).

    Returns:
        list: List of validation-only field names
    """
    return VALIDATION_ONLY_FIELDS.copy()
