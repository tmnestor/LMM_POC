"""
Shared configuration for vision model evaluation.

This module contains all configuration values and constants shared between
different vision models (InternVL3, Llama, etc.).
"""

# ============================================================================
# MODEL CONFIGURATIONS - PRIMARY HIERARCHY
# ============================================================================

# Available model variants
AVAILABLE_MODELS = {
    "internvl3": ["InternVL3-2B", "InternVL3-8B"],
    "llama": ["Llama-3.2-11B-Vision-Instruct", "Llama-3.2-11B-Vision"],
}

# Current model selection (CHANGE THESE TO SWITCH MODELS)
CURRENT_INTERNVL3_MODEL = "InternVL3-8B"  # Options: "InternVL3-2B", "InternVL3-8B"
CURRENT_LLAMA_MODEL = "Llama-3.2-11B-Vision-Instruct"  # Options: "Llama-3.2-11B-Vision-Instruct", "Llama-3.2-11B-Vision"
CURRENT_LLAMA_DIRECT_MODEL = "Llama-3.2-11B-Vision"  # Base model for direct prompting

# ============================================================================
# DEPLOYMENT CONFIGURATIONS
# ============================================================================

# Base paths for different deployment scenarios
BASE_PATHS = {"AISandbox": "/home/jovyan/nfs_share", "efs": "/efs/shared"}

# Current deployment (change this to switch environments)
CURRENT_DEPLOYMENT = "AISandbox"

# ============================================================================
# DYNAMIC PATH GENERATION - MODEL-DRIVEN INTERPOLATION
# ============================================================================

# Dynamic path generation using model + deployment interpolation
BASE_PATH = BASE_PATHS[CURRENT_DEPLOYMENT]

# Model directory structure varies by deployment
MODELS_BASE = (
    f"{BASE_PATH}/models" if CURRENT_DEPLOYMENT == "AISandbox" else f"{BASE_PATH}/PTM"
)

# Model paths driven by current model selection
INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
LLAMA_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"
LLAMA_DIRECT_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_DIRECT_MODEL}"

# Data paths with interpolation
if CURRENT_DEPLOYMENT == "AISandbox":
    DATA_BASE = f"{BASE_PATH}/tod/LMM_POC"
    DATA_DIR = f"{DATA_BASE}/evaluation_data"
    GROUND_TRUTH_PATH = f"{DATA_DIR}/evaluation_ground_truth.csv"
    OUTPUT_DIR = f"{BASE_PATH}/tod/output"
else:  # EFS deployment
    DATA_BASE = f"{BASE_PATH}/PoC_data"
    DATA_DIR = f"{DATA_BASE}/evaluation_data"
    GROUND_TRUTH_PATH = f"{DATA_DIR}/evaluation_ground_truth.csv"
    OUTPUT_DIR = f"{DATA_BASE}/output"


def switch_model(model_type: str, model_name: str):
    """
    Switch to a different model variant.

    Args:
        model_type (str): Model type ('internvl3' or 'llama')
        model_name (str): Specific model name from AVAILABLE_MODELS
    """
    global CURRENT_INTERNVL3_MODEL, CURRENT_LLAMA_MODEL
    global INTERNVL3_MODEL_PATH, LLAMA_MODEL_PATH

    if model_type not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model type: {model_type}. Valid options: {list(AVAILABLE_MODELS.keys())}"
        )

    if model_name not in AVAILABLE_MODELS[model_type]:
        raise ValueError(
            f"Invalid {model_type} model: {model_name}. Valid options: {AVAILABLE_MODELS[model_type]}"
        )

    if model_type == "internvl3":
        CURRENT_INTERNVL3_MODEL = model_name
        INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
        print(f"✅ Switched to {model_name}")
        print(f"   Path: {INTERNVL3_MODEL_PATH}")
    elif model_type == "llama":
        CURRENT_LLAMA_MODEL = model_name
        LLAMA_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"
        print(f"✅ Switched to {model_name}")
        print(f"   Path: {LLAMA_MODEL_PATH}")


def switch_deployment(deployment: str):
    """
    Switch to a different deployment environment.

    Args:
        deployment (str): Deployment type ('AISandbox' or 'efs')
    """
    global CURRENT_DEPLOYMENT, BASE_PATH, MODELS_BASE
    global INTERNVL3_MODEL_PATH, LLAMA_MODEL_PATH
    global DATA_BASE, DATA_DIR, GROUND_TRUTH_PATH, OUTPUT_DIR

    if deployment not in BASE_PATHS:
        raise ValueError(
            f"Invalid deployment: {deployment}. Valid options: {list(BASE_PATHS.keys())}"
        )

    CURRENT_DEPLOYMENT = deployment
    BASE_PATH = BASE_PATHS[CURRENT_DEPLOYMENT]

    # Update model paths using current model selections
    MODELS_BASE = (
        f"{BASE_PATH}/models"
        if CURRENT_DEPLOYMENT == "AISandbox"
        else f"{BASE_PATH}/PTM"
    )
    INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
    LLAMA_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"

    # Update data paths
    if CURRENT_DEPLOYMENT == "AISandbox":
        DATA_BASE = f"{BASE_PATH}/tod/LMM_POC"
        DATA_DIR = f"{DATA_BASE}/evaluation_data"
        GROUND_TRUTH_PATH = f"{DATA_DIR}/evaluation_ground_truth.csv"
        OUTPUT_DIR = f"{BASE_PATH}/tod/output"
    else:  # EFS deployment
        DATA_BASE = f"{BASE_PATH}/PoC_data"
        DATA_DIR = f"{DATA_BASE}/evaluation_data"
        GROUND_TRUTH_PATH = f"{DATA_BASE}/evaluation_ground_truth.csv"
        OUTPUT_DIR = f"{DATA_BASE}/output"

    print(f"✅ Switched to {deployment} deployment")
    print(f"   Models: {MODELS_BASE}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")


def show_current_config():
    """Display current model and deployment configuration."""
    print("🔧 Current Configuration:")
    print(f"   Deployment: {CURRENT_DEPLOYMENT}")
    print(f"   InternVL3 Model: {CURRENT_INTERNVL3_MODEL}")
    print(f"   Llama Model: {CURRENT_LLAMA_MODEL}")
    print(f"   Models Base: {MODELS_BASE}")
    print(f"   Data Dir: {DATA_DIR}")
    print(f"   Output Dir: {OUTPUT_DIR}")
    print(f"   InternVL3 Path: {INTERNVL3_MODEL_PATH}")
    print(f"   Llama Path: {LLAMA_MODEL_PATH}")


# ============================================================================
# FIELD DEFINITIONS - SINGLE SOURCE OF TRUTH
# ============================================================================

# Comprehensive field definitions with all specifications in one place
# This replaces scattered field configurations across the codebase
FIELD_DEFINITIONS = {
    "ABN": {
        "type": "numeric_id",
        "instruction": "[11-digit Australian Business Number or N/A]",
        "evaluation_logic": "exact_numeric_match",
        "description": "Australian Business Number for tax identification",
        "required": True,
    },
    "ACCOUNT_HOLDER": {
        "type": "text",
        "instruction": "[account holder name or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Name of bank account holder",
        "required": False,
    },
    "BANK_ACCOUNT_NUMBER": {
        "type": "numeric_id",
        "instruction": "[account number from bank statements only or N/A]",
        "evaluation_logic": "exact_numeric_match",
        "description": "Bank account number from statements",
        "required": False,
    },
    "BANK_NAME": {
        "type": "text",
        "instruction": "[bank name from bank statements only or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Name of banking institution",
        "required": False,
    },
    "BSB_NUMBER": {
        "type": "numeric_id",
        "instruction": "[6-digit BSB from bank statements only or N/A]",
        "evaluation_logic": "exact_numeric_match",
        "description": "Bank State Branch routing number",
        "required": False,
    },
    "BUSINESS_ADDRESS": {
        "type": "text",
        "instruction": "[business address or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Physical address of business",
        "required": False,
    },
    "BUSINESS_PHONE": {
        "type": "text",
        "instruction": "[business phone number or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Business contact phone number",
        "required": False,
    },
    "CLOSING_BALANCE": {
        "type": "monetary",
        "instruction": "[closing balance amount in dollars or N/A]",
        "evaluation_logic": "monetary_with_tolerance",
        "description": "Final balance on statement",
        "required": False,
    },
    "DESCRIPTIONS": {
        "type": "list",
        "instruction": "[list of transaction descriptions or N/A]",
        "evaluation_logic": "list_overlap_match",
        "description": "Transaction or item descriptions",
        "required": False,
    },
    "DOCUMENT_TYPE": {
        "type": "text",
        "instruction": "[document type (invoice/receipt/statement) or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Type of business document",
        "required": False,
    },
    "DUE_DATE": {
        "type": "date",
        "instruction": "[payment due date or N/A]",
        "evaluation_logic": "flexible_date_match",
        "description": "Payment deadline",
        "required": False,
    },
    "GST": {
        "type": "monetary",
        "instruction": "[GST amount in dollars or N/A]",
        "evaluation_logic": "monetary_with_tolerance",
        "description": "Goods and Services Tax amount",
        "required": False,
    },
    "INVOICE_DATE": {
        "type": "date",
        "instruction": "[invoice date or N/A]",
        "evaluation_logic": "flexible_date_match",
        "description": "Date invoice was issued",
        "required": False,
    },
    "OPENING_BALANCE": {
        "type": "monetary",
        "instruction": "[opening balance amount in dollars or N/A]",
        "evaluation_logic": "monetary_with_tolerance",
        "description": "Starting balance on statement",
        "required": False,
    },
    "PAYER_ADDRESS": {
        "type": "text",
        "instruction": "[payer address or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Address of person/entity making payment",
        "required": False,
    },
    "PAYER_EMAIL": {
        "type": "text",
        "instruction": "[payer email address or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Email address of payer",
        "required": False,
    },
    "PAYER_NAME": {
        "type": "text",
        "instruction": "[payer name or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Name of person/entity making payment",
        "required": False,
    },
    "PAYER_PHONE": {
        "type": "text",
        "instruction": "[payer phone number or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Phone number of payer",
        "required": False,
    },
    "PRICES": {
        "type": "list",
        "instruction": "[individual prices in dollars or N/A]",
        "evaluation_logic": "list_overlap_match",
        "description": "List of individual item prices",
        "required": False,
    },
    "QUANTITIES": {
        "type": "list",
        "instruction": "[list of quantities or N/A]",
        "evaluation_logic": "list_overlap_match",
        "description": "Quantities of items purchased",
        "required": False,
    },
    "STATEMENT_PERIOD": {
        "type": "date",
        "instruction": "[statement period or N/A]",
        "evaluation_logic": "flexible_date_match",
        "description": "Time period covered by statement",
        "required": False,
    },
    "SUBTOTAL": {
        "type": "monetary",
        "instruction": "[subtotal amount in dollars or N/A]",
        "evaluation_logic": "monetary_with_tolerance",
        "description": "Subtotal before taxes and fees",
        "required": False,
    },
    "SUPPLIER": {
        "type": "text",
        "instruction": "[supplier name or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Name of goods/services provider",
        "required": False,
    },
    "SUPPLIER_WEBSITE": {
        "type": "text",
        "instruction": "[supplier website or N/A]",
        "evaluation_logic": "fuzzy_text_match",
        "description": "Website URL of supplier",
        "required": False,
    },
    "TOTAL": {
        "type": "monetary",
        "instruction": "[total amount in dollars or N/A]",
        "evaluation_logic": "monetary_with_tolerance",
        "description": "Final total amount including all charges",
        "required": True,
    },
}

# ============================================================================
# DERIVED CONFIGURATIONS - AUTO-GENERATED FROM FIELD_DEFINITIONS
# ============================================================================

# Primary configurations derived from FIELD_DEFINITIONS
EXTRACTION_FIELDS = list(FIELD_DEFINITIONS.keys())
FIELD_COUNT = len(EXTRACTION_FIELDS)
FIELD_INSTRUCTIONS = {k: v["instruction"] for k, v in FIELD_DEFINITIONS.items()}
FIELD_TYPES = {k: v["type"] for k, v in FIELD_DEFINITIONS.items()}
FIELD_DESCRIPTIONS = {k: v["description"] for k, v in FIELD_DEFINITIONS.items()}

# Field type groupings for evaluation logic
NUMERIC_ID_FIELDS = [
    k for k, v in FIELD_DEFINITIONS.items() if v["type"] == "numeric_id"
]
MONETARY_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v["type"] == "monetary"]
DATE_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v["type"] == "date"]
LIST_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v["type"] == "list"]
TEXT_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v["type"] == "text"]

# Required vs optional field groupings
REQUIRED_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v.get("required", False)]
OPTIONAL_FIELDS = [
    k for k, v in FIELD_DEFINITIONS.items() if not v.get("required", False)
]

# ============================================================================
# FIELD DEFINITION VALIDATION
# ============================================================================


def validate_field_definitions():
    """
    Validate that all field definitions are complete and consistent.

    Raises:
        ValueError: If any field definition is incomplete or invalid
    """
    required_keys = [
        "type",
        "instruction",
        "evaluation_logic",
        "description",
        "required",
    ]
    valid_types = ["numeric_id", "monetary", "date", "list", "text"]
    valid_evaluation_logic = [
        "exact_numeric_match",
        "monetary_with_tolerance",
        "flexible_date_match",
        "list_overlap_match",
        "fuzzy_text_match",
    ]

    for field_name, definition in FIELD_DEFINITIONS.items():
        # Check required keys
        for key in required_keys:
            if key not in definition:
                raise ValueError(f"Field '{field_name}' missing required key: '{key}'")

        # Validate field type
        if definition["type"] not in valid_types:
            raise ValueError(
                f"Field '{field_name}' has invalid type: '{definition['type']}'. "
                f"Valid types: {valid_types}"
            )

        # Validate evaluation logic
        if definition["evaluation_logic"] not in valid_evaluation_logic:
            raise ValueError(
                f"Field '{field_name}' has invalid evaluation_logic: '{definition['evaluation_logic']}'. "
                f"Valid options: {valid_evaluation_logic}"
            )

        # Check instruction format
        instruction = definition["instruction"]
        if not (instruction.startswith("[") and instruction.endswith("]")):
            raise ValueError(
                f"Field '{field_name}' instruction must be in format '[instruction or N/A]'"
            )

        # Ensure instruction mentions N/A
        if "N/A" not in instruction:
            raise ValueError(
                f"Field '{field_name}' instruction must mention 'N/A' option"
            )


# Run validation on import to catch configuration errors early
validate_field_definitions()

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
    "llama": 1,  # Llama-3.2-11B with optimized 8-bit quantization on 16GB VRAM
    "internvl3": 4,  # InternVL3 generic fallback (backward compatibility)
    "internvl3-2b": 4,  # InternVL3-2B is memory efficient, can handle larger batches
    "internvl3-8b": 1,  # InternVL3-8B with quantization needs conservative batching
}

# Maximum batch sizes per model (Aggressive for 24GB+ VRAM)
MAX_BATCH_SIZES = {
    "llama": 3,  # Higher end for powerful GPUs
    "internvl3": 8,  # InternVL3 generic fallback (backward compatibility)
    "internvl3-2b": 8,  # InternVL3-2B can handle large batches on high-end GPUs
    "internvl3-8b": 2,  # InternVL3-8B maximum safe batch size even on powerful GPUs
}

# Conservative batch sizes per model (Safe for limited memory situations)
CONSERVATIVE_BATCH_SIZES = {
    "llama": 1,  # Llama always uses 1 for conservative approach
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

# Llama-3.2-11B-Vision generation configuration
LLAMA_GENERATION_CONFIG = {
    "max_new_tokens_base": 600,  # Reduced from 800 to save memory
    "max_new_tokens_per_field": 30,  # Reduced from 40 - still adequate for extraction
    "temperature": 0.1,  # Near-deterministic sampling
    "do_sample": True,  # Enable sampling for controlled randomness
    "top_p": 0.95,  # Nucleus sampling parameter
    "use_cache": True,  # CRITICAL: Required for extraction quality (proven by testing)
}

# InternVL3 generation configuration
INTERNVL3_GENERATION_CONFIG = {
    "max_new_tokens_base": 1000,  # Base tokens for generation
    "max_new_tokens_per_field": 50,  # Additional tokens per extraction field
    "do_sample": False,  # Deterministic for consistent field extraction
    "pad_token_id": None,  # Set dynamically from tokenizer
}


# Helper function to calculate dynamic max_new_tokens
def get_max_new_tokens(model_name: str, field_count: int = None) -> int:
    """
    Calculate max_new_tokens based on model and field count.

    Args:
        model_name (str): Model name ('llama', 'internvl3', 'internvl3-2b', 'internvl3-8b')
        field_count (int): Number of extraction fields (uses FIELD_COUNT if None)

    Returns:
        int: Calculated max_new_tokens value
    """
    field_count = field_count or FIELD_COUNT

    model_name_lower = model_name.lower()

    if model_name_lower == "llama":
        config = LLAMA_GENERATION_CONFIG
    elif model_name_lower.startswith("internvl3"):
        # Handle all InternVL3 variants (internvl3, internvl3-2b, internvl3-8b)
        config = INTERNVL3_GENERATION_CONFIG
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return max(
        config["max_new_tokens_base"], field_count * config["max_new_tokens_per_field"]
    )


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
