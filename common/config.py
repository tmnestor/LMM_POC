"""
Shared configuration for vision model evaluation.

This module contains all configuration values and constants shared between
different vision models (InternVL3, Llama, etc.).

NOTE: Uses YAML-first field discovery for single source of truth.
NOTE: Supports environment variables for flexible deployment configuration.
"""

import os

from .schema_loader import get_global_schema

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

# ============================================================================
# ENVIRONMENT VARIABLE SUPPORT
# ============================================================================


def get_env_or_default(env_var: str, default_value: str) -> str:
    """
    Get environment variable value or use default.

    Args:
        env_var (str): Environment variable name
        default_value (str): Default value if environment variable not set

    Returns:
        str: Environment variable value or default
    """
    value = os.getenv(env_var)
    if value:
        print(f"🌍 Using environment variable {env_var}: {value}")
        return value
    return default_value


# Environment variables for flexible deployment
ENV_LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")
ENV_INTERNVL3_MODEL_PATH = os.getenv("INTERNVL3_MODEL_PATH")
ENV_GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_PATH")
ENV_OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# ============================================================================
# ENVIRONMENT PROFILES
# ============================================================================

# Define complete environment profiles for different deployment scenarios
ENVIRONMENT_PROFILES = {
    "development": {
        "description": "Local development environment",
        "base_path": "/Users/tod/Desktop/LMM_POC",
        "models_dir": "models",
        "data_dir": "evaluation_data",
        "output_dir": "output",
        "batch_strategy": "conservative",
        "enable_debug": True,
    },
    "testing": {
        "description": "H200 GPU testing environment",
        "base_path": "/home/jovyan/nfs_share",
        "models_dir": "models",
        "data_dir": "tod/LMM_POC/evaluation_data",
        "output_dir": "tod/output",
        "batch_strategy": "aggressive",
        "enable_debug": True,
    },
    "production": {
        "description": "V100 production deployment",
        "base_path": "/efs/shared",
        "models_dir": "PTM",
        "data_dir": "PoC_data/evaluation_data",
        "output_dir": "PoC_data/output",
        "batch_strategy": "balanced",
        "enable_debug": False,
    },
    "aisandbox": {
        "description": "AISandbox deployment (legacy)",
        "base_path": "/home/jovyan/nfs_share",
        "models_dir": "models",
        "data_dir": "tod/LMM_POC/evaluation_data",
        "output_dir": "tod/output",
        "batch_strategy": "balanced",
        "enable_debug": False,
    },
    "efs": {
        "description": "EFS deployment (legacy)",
        "base_path": "/efs/shared",
        "models_dir": "PTM",
        "data_dir": "PoC_data/evaluation_data",
        "output_dir": "PoC_data/output",
        "batch_strategy": "balanced",
        "enable_debug": False,
    },
}


def get_current_environment():
    """
    Determine current environment from environment variable or deployment setting.

    Priority:
    1. LMM_ENVIRONMENT environment variable
    2. Map CURRENT_DEPLOYMENT to environment profile
    3. Default to "development"

    Returns:
        str: Current environment profile name
    """
    # Check environment variable first
    env_profile = os.getenv("LMM_ENVIRONMENT")
    if env_profile and env_profile in ENVIRONMENT_PROFILES:
        print(f"🌍 Using environment profile from LMM_ENVIRONMENT: {env_profile}")
        return env_profile

    # Map legacy deployment names to profiles
    deployment_mapping = {"AISandbox": "aisandbox", "efs": "efs"}

    if hasattr(globals().get("CURRENT_DEPLOYMENT"), "value"):
        mapped_env = deployment_mapping.get(CURRENT_DEPLOYMENT)
        if mapped_env:
            print(
                f"🔄 Mapped deployment '{CURRENT_DEPLOYMENT}' to environment '{mapped_env}'"
            )
            return mapped_env

    # Default fallback
    print("🏠 Using default development environment")
    return "development"


def get_environment_config(environment: str = None) -> dict:
    """
    Get configuration for specified environment.

    Args:
        environment (str): Environment name, or None to auto-detect

    Returns:
        dict: Environment configuration
    """
    if environment is None:
        environment = get_current_environment()

    if environment not in ENVIRONMENT_PROFILES:
        raise ValueError(
            f"Unknown environment: {environment}. "
            f"Available: {list(ENVIRONMENT_PROFILES.keys())}"
        )

    return ENVIRONMENT_PROFILES[environment]


def get_environment_path(path_type: str, environment: str = None) -> str:
    """
    Get path for specific type in current or specified environment.

    Args:
        path_type (str): Type of path ('models', 'data', 'output', 'base')
        environment (str): Environment name, or None to auto-detect

    Returns:
        str: Resolved path for the environment
    """
    config = get_environment_config(environment)
    base = config["base_path"]

    if path_type == "base":
        return base
    elif path_type == "models":
        return f"{base}/{config['models_dir']}"
    elif path_type == "data":
        return f"{base}/{config['data_dir']}"
    elif path_type == "output":
        return f"{base}/{config['output_dir']}"
    else:
        raise ValueError(f"Unknown path type: {path_type}")


def switch_environment(environment: str):
    """
    Switch to a different environment profile.

    Args:
        environment (str): Environment name from ENVIRONMENT_PROFILES
    """
    if environment not in ENVIRONMENT_PROFILES:
        raise ValueError(
            f"Unknown environment: {environment}. "
            f"Available: {list(ENVIRONMENT_PROFILES.keys())}"
        )

    global CURRENT_DEPLOYMENT, BASE_PATH, MODELS_BASE
    global INTERNVL3_MODEL_PATH, LLAMA_MODEL_PATH
    global DATA_BASE, DATA_DIR, GROUND_TRUTH_PATH, OUTPUT_DIR

    config = ENVIRONMENT_PROFILES[environment]

    # Update paths based on environment config
    BASE_PATH = config["base_path"]
    MODELS_BASE = f"{BASE_PATH}/{config['models_dir']}"
    DATA_DIR = f"{BASE_PATH}/{config['data_dir']}"
    OUTPUT_DIR = f"{BASE_PATH}/{config['output_dir']}"

    # Update model paths if not overridden by environment variables
    if not ENV_INTERNVL3_MODEL_PATH:
        INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
    if not ENV_LLAMA_MODEL_PATH:
        LLAMA_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"
    if not ENV_GROUND_TRUTH_PATH:
        GROUND_TRUTH_PATH = f"{DATA_DIR}/ground_truth.csv"
    if not ENV_OUTPUT_DIR:
        OUTPUT_DIR = f"{BASE_PATH}/{config['output_dir']}"

    print(f"✅ Switched to environment: {environment}")
    print(f"   Description: {config['description']}")
    print(f"   Base Path: {BASE_PATH}")
    print(f"   Models: {MODELS_BASE}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Batch Strategy: {config['batch_strategy']}")

    # Update current deployment for legacy compatibility
    legacy_mapping = {"aisandbox": "AISandbox", "efs": "efs"}
    CURRENT_DEPLOYMENT = legacy_mapping.get(environment, environment)


# ============================================================================
# DEPLOYMENT CONFIGURATIONS (Legacy Support)
# ============================================================================

# Base paths for different deployment scenarios (DEPRECATED - use ENVIRONMENT_PROFILES)
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

# Model paths driven by environment variables or current model selection
INTERNVL3_MODEL_PATH = get_env_or_default(
    "INTERNVL3_MODEL_PATH", f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
)
LLAMA_MODEL_PATH = get_env_or_default(
    "LLAMA_MODEL_PATH", f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"
)

# Data paths with environment variable support or deployment-based fallback
if CURRENT_DEPLOYMENT == "AISandbox":
    DATA_BASE = f"{BASE_PATH}/tod/LMM_POC"
    DATA_DIR = f"{DATA_BASE}/evaluation_data"
    _default_ground_truth = f"{DATA_DIR}/ground_truth.csv"
    _default_output_dir = f"{BASE_PATH}/tod/output"
else:  # EFS deployment
    DATA_BASE = f"{BASE_PATH}/PoC_data"
    DATA_DIR = f"{DATA_BASE}/evaluation_data"
    _default_ground_truth = f"{DATA_DIR}/ground_truth.csv"
    _default_output_dir = f"{DATA_BASE}/output"

# Apply environment variable overrides
GROUND_TRUTH_PATH = get_env_or_default("GROUND_TRUTH_PATH", _default_ground_truth)
OUTPUT_DIR = get_env_or_default("OUTPUT_DIR", _default_output_dir)


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
        if not ENV_INTERNVL3_MODEL_PATH:  # Only update if no env var override
            INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
        print(f"✅ Switched to {model_name}")
        print(f"   Path: {INTERNVL3_MODEL_PATH}")
        if ENV_INTERNVL3_MODEL_PATH:
            print("   (Environment override active: INTERNVL3_MODEL_PATH)")
    elif model_type == "llama":
        CURRENT_LLAMA_MODEL = model_name
        if not ENV_LLAMA_MODEL_PATH:  # Only update if no env var override
            LLAMA_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"
        print(f"✅ Switched to {model_name}")
        print(f"   Path: {LLAMA_MODEL_PATH}")
        if ENV_LLAMA_MODEL_PATH:
            print("   (Environment override active: LLAMA_MODEL_PATH)")


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

    # Update model paths using current model selections (respecting env vars)
    MODELS_BASE = (
        f"{BASE_PATH}/models"
        if CURRENT_DEPLOYMENT == "AISandbox"
        else f"{BASE_PATH}/PTM"
    )

    # Only update paths if not overridden by environment variables
    if not ENV_INTERNVL3_MODEL_PATH:
        INTERNVL3_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_INTERNVL3_MODEL}"
    if not ENV_LLAMA_MODEL_PATH:
        LLAMA_MODEL_PATH = f"{MODELS_BASE}/{CURRENT_LLAMA_MODEL}"

    # Update data paths (respecting env vars)
    if CURRENT_DEPLOYMENT == "AISandbox":
        DATA_BASE = f"{BASE_PATH}/tod/LMM_POC"
        DATA_DIR = f"{DATA_BASE}/evaluation_data"
        _default_ground_truth = f"{DATA_DIR}/ground_truth.csv"
        _default_output_dir = f"{BASE_PATH}/tod/output"
    else:  # EFS deployment
        DATA_BASE = f"{BASE_PATH}/PoC_data"
        DATA_DIR = f"{DATA_BASE}/evaluation_data"
        _default_ground_truth = f"{DATA_DIR}/ground_truth.csv"
        _default_output_dir = f"{DATA_BASE}/output"

    # Only update paths if not overridden by environment variables
    if not ENV_GROUND_TRUTH_PATH:
        GROUND_TRUTH_PATH = _default_ground_truth
    if not ENV_OUTPUT_DIR:
        OUTPUT_DIR = _default_output_dir

    print(f"✅ Switched to {deployment} deployment")
    print(f"   Models: {MODELS_BASE}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")


def show_current_config():
    """Display current model and deployment configuration."""
    try:
        current_env = get_current_environment()
        env_config = get_environment_config(current_env)

        print("🔧 Current Configuration:")
        print(f"   Environment Profile: {current_env}")
        print(f"   Description: {env_config['description']}")
        print(f"   Batch Strategy: {env_config['batch_strategy']}")
        print(f"   Debug Mode: {env_config['enable_debug']}")
        print(f"   Legacy Deployment: {CURRENT_DEPLOYMENT}")
        print()
        print("📁 Paths:")
        print(f"   Base Path: {BASE_PATH}")
        print(f"   Models: {MODELS_BASE}")
        print(f"   Data Dir: {DATA_DIR}")
        print(f"   Output Dir: {OUTPUT_DIR}")
        print(f"   Ground Truth: {GROUND_TRUTH_PATH}")
        print()
        print("🤖 Models:")
        print(f"   InternVL3: {CURRENT_INTERNVL3_MODEL}")
        print(f"   Llama: {CURRENT_LLAMA_MODEL}")
        print(f"   InternVL3 Path: {INTERNVL3_MODEL_PATH}")
        print(f"   Llama Path: {LLAMA_MODEL_PATH}")

        # Show active environment variable overrides
        env_overrides = []
        if ENV_INTERNVL3_MODEL_PATH:
            env_overrides.append("INTERNVL3_MODEL_PATH")
        if ENV_LLAMA_MODEL_PATH:
            env_overrides.append("LLAMA_MODEL_PATH")
        if ENV_GROUND_TRUTH_PATH:
            env_overrides.append("GROUND_TRUTH_PATH")
        if ENV_OUTPUT_DIR:
            env_overrides.append("OUTPUT_DIR")
        if os.getenv("LMM_ENVIRONMENT"):
            env_overrides.append("LMM_ENVIRONMENT")

        if env_overrides:
            print()
            print("🌍 Environment Variable Overrides:")
            for var in env_overrides:
                print(f"   {var}: {os.getenv(var)}")
        else:
            print()
            print("🌍 No environment variable overrides active")

    except Exception as e:
        print(f"⚠️ Error showing configuration: {e}")
        print("🔧 Basic Configuration:")
        print(f"   Legacy Deployment: {CURRENT_DEPLOYMENT}")
        print(f"   InternVL3 Model: {CURRENT_INTERNVL3_MODEL}")
        print(f"   Llama Model: {CURRENT_LLAMA_MODEL}")


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

# Schema defines the fields and their order. We use it directly.
_schema = get_global_schema()
EXTRACTION_FIELDS = _schema.field_names
FIELD_COUNT = _schema.total_fields
# Schema info available via FIELD_COUNT and EXTRACTION_FIELDS variables
# Print statement removed to support both unified and document-aware modes

# FIELD_INSTRUCTIONS removed - using YAML-first field discovery for single source of truth

# Field metadata now provided by schema loader - no need for inference functions

# Generate field metadata and type groupings dynamically from schema
FIELD_TYPES = _schema.generate_field_types_mapping()
FIELD_DESCRIPTIONS = _schema.generate_field_descriptions_mapping()
FIELD_INSTRUCTIONS = _schema.generate_prompt_instructions()

# Field type groupings for evaluation logic (using schema)
NUMERIC_ID_FIELDS = _schema.get_fields_by_type("numeric_id")
MONETARY_FIELDS = _schema.get_fields_by_type("monetary")
DATE_FIELDS = _schema.get_fields_by_type("date")
LIST_FIELDS = _schema.get_fields_by_type("list")
TEXT_FIELDS = _schema.get_fields_by_type("text")

# All fields are required for extraction (must attempt to extract and return value or NOT_FOUND)

# ============================================================================
# GROUPED EXTRACTION CONFIGURATION
# ============================================================================

# Extraction modes for different strategies
EXTRACTION_MODES = ["single_pass", "field_grouped", "detailed_grouped", "adaptive"]
DEFAULT_EXTRACTION_MODE = "detailed_grouped"  # Production default

# Field groups for grouped extraction strategy
# Based on research showing improved accuracy with focused field extraction

# Generate detailed grouped strategy dynamically from schema
_detailed_strategy = _schema.get_grouping_strategy("detailed_grouped")
FIELD_GROUPS_DETAILED = _detailed_strategy["group_configs"]

# Generate cognitive grouped strategy dynamically from schema
_cognitive_strategy = _schema.get_grouping_strategy("field_grouped")
FIELD_GROUPS_COGNITIVE = _cognitive_strategy["group_configs"]

# Grouping strategy selection with semantic names
GROUPING_STRATEGIES = {
    "detailed_grouped": FIELD_GROUPS_DETAILED,  # 8-group detailed extraction
    "field_grouped": FIELD_GROUPS_COGNITIVE,  # 6-group cognitive optimization
}

# Default grouping strategy (using semantic name)
DEFAULT_GROUPING_STRATEGY = "detailed_grouped"  # 8-group detailed extraction
FIELD_GROUPS = FIELD_GROUPS_DETAILED  # Backward compatibility

# Group processing order (semantic order from schema)
GROUP_PROCESSING_ORDER = list(FIELD_GROUPS.keys())

# Adaptive mode thresholds
ADAPTIVE_MODE_CONFIG = {
    "simple_document_threshold": 10,  # Use single-pass for documents with <10 visible fields
    "complex_document_threshold": 20,  # Use grouped for documents with >20 visible fields
    "confidence_threshold": 0.85,  # Minimum confidence to skip grouped extraction
}

# Group-specific prompt templates
GROUP_PROMPT_TEMPLATES = {
    "precise": "Extract ONLY these critical business identifiers. Be extremely precise.",
    "numerical": "Extract ONLY these monetary amounts. Focus on numerical values and currency.",
    "temporal": "Extract ONLY these date fields. Look for date patterns and time periods.",
    "descriptive": "Extract ONLY these text fields. Capture complete information.",
    "mixed": "Extract ONLY these banking details. Include both text and numbers.",
    "list": "Extract ONLY these item details. Capture all items as lists.",
    "classification": "Identify ONLY the document type from the image.",
}

# Validation rules per group
# Generate validation rules dynamically from schema
GROUP_VALIDATION_RULES = {}
for group_name in _schema.schema.get("validation_rules", {}):
    GROUP_VALIDATION_RULES[group_name] = _schema.get_validation_rules(group_name)

# ============================================================================
# GROUPED EXTRACTION HELPER FUNCTIONS
# ============================================================================


def get_fields_for_group(group_name):
    """
    Get list of fields for a specific extraction group.

    Args:
        group_name (str): Name of the extraction group

    Returns:
        list: List of field names in the group
    """
    if group_name not in FIELD_GROUPS:
        raise ValueError(f"Unknown extraction group: {group_name}")
    return FIELD_GROUPS[group_name]["fields"]


def get_group_for_field(field_name):
    """
    Find which group a field belongs to.

    Args:
        field_name (str): Name of the field

    Returns:
        str: Group name containing the field, or None if not found
    """
    for group_name, group_config in FIELD_GROUPS.items():
        if field_name in group_config["fields"]:
            return group_name
    return None


def get_extraction_groups_summary():
    """
    Get a summary of all extraction groups.

    Returns:
        dict: Summary of groups with field counts and priorities
    """
    summary = {}
    for group_name, config in FIELD_GROUPS.items():
        summary[group_name] = {
            "name": config["name"],
            "field_count": len(config["fields"]),
            "fields": config["fields"],
        }
    return summary


# ============================================================================
# FIELD DEFINITION VALIDATION
# ============================================================================


# Field validation now handled by schema loader on initialization

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

# ============================================================================
# MODEL-SPECIFIC GENERATION CONFIGURATION
# ============================================================================

# Token limits for different model sizes with quantization
INTERNVL3_TOKEN_LIMITS = {
    "2b": None,  # Use get_max_new_tokens() calculation
    "8b": 800,  # Enough for all 25 fields with buffer after 8-bit quantization
}

# Generation parameters for different models
GENERATION_CONFIGS = {
    "internvl3": {
        "do_sample": False,  # CRITICAL: Must be False for deterministic output (greedy decoding)
        # When do_sample=False, temperature/top_k/top_p are ignored and cause warnings
        # So we don't set them - greedy decoding automatically selects highest probability token
        "num_beams": 1,  # No beam search - single path only
        "repetition_penalty": 1.0,  # No repetition penalty
        # Note: seed is set at system level in _set_random_seeds(), not in generation config
    },
    "llama": {
        "do_sample": False,  # Greedy decoding for determinism
        # No temperature/top_k/top_p to avoid warnings with do_sample=False
        "num_beams": 1,
        "repetition_penalty": 1.0,
        # Note: seed is set at system level, not in generation config
    },
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
    "temperature": 0.0,  # Deterministic sampling for consistent results
    "do_sample": False,  # Disable sampling for full determinism
    "top_p": 0.95,  # Nucleus sampling parameter (inactive with do_sample=False)
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
