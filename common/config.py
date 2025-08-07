"""
Shared configuration for vision model evaluation.

This module contains all configuration values and constants shared between
different vision models (InternVL3, Llama, etc.).
"""


# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

# Primary paths (local development)
DATA_DIR = "/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data"
GROUND_TRUTH_PATH = "/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data/evaluation_ground_truth.csv"
OUTPUT_DIR = "/home/jovyan/nfs_share/tod/output"

# Alternative paths (EFS deployment)
# DATA_DIR = "/efs/share/PoC_data/evaluation_data"
# GROUND_TRUTH_PATH = "/efs/share/PoC_data/evaluation_ground_truth.csv"
# OUTPUT_DIR = "/efs/share/PoC_data/output"

# ============================================================================
# FIELD DEFINITIONS - SINGLE SOURCE OF TRUTH
# ============================================================================

# Comprehensive field definitions with all specifications in one place
# This replaces scattered field configurations across the codebase
FIELD_DEFINITIONS = {
    'ABN': {
        'type': 'numeric_id',
        'instruction': '[11-digit Australian Business Number or N/A]',
        'evaluation_logic': 'exact_numeric_match',
        'description': 'Australian Business Number for tax identification',
        'required': True
    },
    'ACCOUNT_HOLDER': {
        'type': 'text',
        'instruction': '[account holder name or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Name of bank account holder',
        'required': False
    },
    'BANK_ACCOUNT_NUMBER': {
        'type': 'numeric_id',
        'instruction': '[account number from bank statements only or N/A]',
        'evaluation_logic': 'exact_numeric_match',
        'description': 'Bank account number from statements',
        'required': False
    },
    'BANK_NAME': {
        'type': 'text',
        'instruction': '[bank name from bank statements only or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Name of banking institution',
        'required': False
    },
    'BSB_NUMBER': {
        'type': 'numeric_id',
        'instruction': '[6-digit BSB from bank statements only or N/A]',
        'evaluation_logic': 'exact_numeric_match',
        'description': 'Bank State Branch routing number',
        'required': False
    },
    'BUSINESS_ADDRESS': {
        'type': 'text',
        'instruction': '[business address or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Physical address of business',
        'required': False
    },
    'BUSINESS_PHONE': {
        'type': 'text',
        'instruction': '[business phone number or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Business contact phone number',
        'required': False
    },
    'CLOSING_BALANCE': {
        'type': 'monetary',
        'instruction': '[closing balance amount in dollars or N/A]',
        'evaluation_logic': 'monetary_with_tolerance',
        'description': 'Final balance on statement',
        'required': False
    },
    'DESCRIPTIONS': {
        'type': 'list',
        'instruction': '[list of transaction descriptions or N/A]',
        'evaluation_logic': 'list_overlap_match',
        'description': 'Transaction or item descriptions',
        'required': False
    },
    'DOCUMENT_TYPE': {
        'type': 'text',
        'instruction': '[document type (invoice/receipt/statement) or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Type of business document',
        'required': False
    },
    'DUE_DATE': {
        'type': 'date',
        'instruction': '[payment due date or N/A]',
        'evaluation_logic': 'flexible_date_match',
        'description': 'Payment deadline',
        'required': False
    },
    'GST': {
        'type': 'monetary',
        'instruction': '[GST amount in dollars or N/A]',
        'evaluation_logic': 'monetary_with_tolerance',
        'description': 'Goods and Services Tax amount',
        'required': False
    },
    'INVOICE_DATE': {
        'type': 'date',
        'instruction': '[invoice date or N/A]',
        'evaluation_logic': 'flexible_date_match',
        'description': 'Date invoice was issued',
        'required': False
    },
    'OPENING_BALANCE': {
        'type': 'monetary',
        'instruction': '[opening balance amount in dollars or N/A]',
        'evaluation_logic': 'monetary_with_tolerance',
        'description': 'Starting balance on statement',
        'required': False
    },
    'PAYER_ADDRESS': {
        'type': 'text',
        'instruction': '[payer address or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Address of person/entity making payment',
        'required': False
    },
    'PAYER_EMAIL': {
        'type': 'text',
        'instruction': '[payer email address or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Email address of payer',
        'required': False
    },
    'PAYER_NAME': {
        'type': 'text',
        'instruction': '[payer name or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Name of person/entity making payment',
        'required': False
    },
    'PAYER_PHONE': {
        'type': 'text',
        'instruction': '[payer phone number or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Phone number of payer',
        'required': False
    },
    'PRICES': {
        'type': 'list',
        'instruction': '[individual prices in dollars or N/A]',
        'evaluation_logic': 'list_overlap_match',
        'description': 'List of individual item prices',
        'required': False
    },
    'QUANTITIES': {
        'type': 'list',
        'instruction': '[list of quantities or N/A]',
        'evaluation_logic': 'list_overlap_match',
        'description': 'Quantities of items purchased',
        'required': False
    },
    'STATEMENT_PERIOD': {
        'type': 'date',
        'instruction': '[statement period or N/A]',
        'evaluation_logic': 'flexible_date_match',
        'description': 'Time period covered by statement',
        'required': False
    },
    'SUBTOTAL': {
        'type': 'monetary',
        'instruction': '[subtotal amount in dollars or N/A]',
        'evaluation_logic': 'monetary_with_tolerance',
        'description': 'Subtotal before taxes and fees',
        'required': False
    },
    'SUPPLIER': {
        'type': 'text',
        'instruction': '[supplier name or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Name of goods/services provider',
        'required': False
    },
    'SUPPLIER_WEBSITE': {
        'type': 'text',
        'instruction': '[supplier website or N/A]',
        'evaluation_logic': 'fuzzy_text_match',
        'description': 'Website URL of supplier',
        'required': False
    },
    'TOTAL': {
        'type': 'monetary',
        'instruction': '[total amount in dollars or N/A]',
        'evaluation_logic': 'monetary_with_tolerance',
        'description': 'Final total amount including all charges',
        'required': True
    }
}

# ============================================================================
# DERIVED CONFIGURATIONS - AUTO-GENERATED FROM FIELD_DEFINITIONS
# ============================================================================

# Primary configurations derived from FIELD_DEFINITIONS
EXTRACTION_FIELDS = list(FIELD_DEFINITIONS.keys())
FIELD_COUNT = len(EXTRACTION_FIELDS)
FIELD_INSTRUCTIONS = {k: v['instruction'] for k, v in FIELD_DEFINITIONS.items()}
FIELD_TYPES = {k: v['type'] for k, v in FIELD_DEFINITIONS.items()}
FIELD_DESCRIPTIONS = {k: v['description'] for k, v in FIELD_DEFINITIONS.items()}

# Field type groupings for evaluation logic
NUMERIC_ID_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'numeric_id']
MONETARY_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'monetary']
DATE_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'date']
LIST_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'list']
TEXT_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v['type'] == 'text']

# Required vs optional field groupings
REQUIRED_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if v.get('required', False)]
OPTIONAL_FIELDS = [k for k, v in FIELD_DEFINITIONS.items() if not v.get('required', False)]

# ============================================================================
# FIELD DEFINITION VALIDATION
# ============================================================================

def validate_field_definitions():
    """
    Validate that all field definitions are complete and consistent.
    
    Raises:
        ValueError: If any field definition is incomplete or invalid
    """
    required_keys = ['type', 'instruction', 'evaluation_logic', 'description', 'required']
    valid_types = ['numeric_id', 'monetary', 'date', 'list', 'text']
    valid_evaluation_logic = [
        'exact_numeric_match', 'monetary_with_tolerance', 'flexible_date_match',
        'list_overlap_match', 'fuzzy_text_match'
    ]
    
    for field_name, definition in FIELD_DEFINITIONS.items():
        # Check required keys
        for key in required_keys:
            if key not in definition:
                raise ValueError(f"Field '{field_name}' missing required key: '{key}'")
        
        # Validate field type
        if definition['type'] not in valid_types:
            raise ValueError(f"Field '{field_name}' has invalid type: '{definition['type']}'. "
                           f"Valid types: {valid_types}")
        
        # Validate evaluation logic
        if definition['evaluation_logic'] not in valid_evaluation_logic:
            raise ValueError(f"Field '{field_name}' has invalid evaluation_logic: '{definition['evaluation_logic']}'. "
                           f"Valid options: {valid_evaluation_logic}")
        
        # Check instruction format
        instruction = definition['instruction']
        if not (instruction.startswith('[') and instruction.endswith(']')):
            raise ValueError(f"Field '{field_name}' instruction must be in format '[instruction or N/A]'")
        
        # Ensure instruction mentions N/A
        if 'N/A' not in instruction:
            raise ValueError(f"Field '{field_name}' instruction must mention 'N/A' option")

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
PILOT_READY_THRESHOLD = 0.8        # 80% accuracy for pilot testing
NEEDS_OPTIMIZATION_THRESHOLD = 0.7  # Below 70% needs major improvements

# Field-specific accuracy thresholds
EXCELLENT_FIELD_THRESHOLD = 0.9    # Fields with ≥90% accuracy
GOOD_FIELD_THRESHOLD = 0.8         # Fields with ≥80% accuracy
POOR_FIELD_THRESHOLD = 0.5         # Fields with <50% accuracy

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

IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Model paths
INTERNVL3_MODEL_PATH = "/home/jovyan/nfs_share/models/InternVL3-2B"
LLAMA_MODEL_PATH = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"

# Alternative model paths (EFS deployment)
# INTERNVL3_MODEL_PATH = "/efs/share/PTM/InternVL3-2B"
# LLAMA_MODEL_PATH = "/efs/share/PTM/Llama-3.2-11B-Vision-Instruct"

# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Default batch sizes per model (Conservative for 16GB VRAM)
DEFAULT_BATCH_SIZES = {
    'llama': 1,        # Llama-3.2-11B with 8-bit quantization on 16GB VRAM
    'internvl3': 4,    # InternVL3 is more memory efficient
}

# Maximum batch sizes per model (Aggressive for 24GB+ VRAM)
MAX_BATCH_SIZES = {
    'llama': 3,        # Higher end for powerful GPUs
    'internvl3': 8,    # InternVL3 can handle larger batches
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
    'conservative': 'Use minimum safe batch sizes for stability',
    'balanced': 'Use default batch sizes for typical hardware',
    'aggressive': 'Use maximum batch sizes for high-end hardware'
}

# Current strategy (can be changed for different deployment scenarios)
CURRENT_BATCH_STRATEGY = 'balanced'

# GPU memory thresholds for automatic batch size selection
GPU_MEMORY_THRESHOLDS = {
    'low': 8,      # GB - Use conservative batching
    'medium': 16,  # GB - Use default batching  
    'high': 24,    # GB - Use aggressive batching
}

# Automatic fallback settings
ENABLE_BATCH_SIZE_FALLBACK = True
BATCH_SIZE_FALLBACK_STEPS = [8, 4, 2, 1]  # Try these batch sizes if OOM occurs

def get_batch_size_for_model(model_name: str, strategy: str = None) -> int:
    """
    Get recommended batch size for a model based on strategy.
    
    Args:
        model_name (str): Model name ('llama' or 'internvl3')
        strategy (str): Batching strategy ('conservative', 'balanced', 'aggressive')
        
    Returns:
        int: Recommended batch size
    """
    strategy = strategy or CURRENT_BATCH_STRATEGY
    model_name = model_name.lower()
    
    if strategy == 'conservative':
        return MIN_BATCH_SIZE
    elif strategy == 'aggressive':
        return MAX_BATCH_SIZES.get(model_name, MIN_BATCH_SIZE)
    else:  # balanced
        return DEFAULT_BATCH_SIZES.get(model_name, MIN_BATCH_SIZE)

def get_auto_batch_size(model_name: str, available_memory_gb: float = None) -> int:
    """
    Automatically determine batch size based on available GPU memory.
    
    Args:
        model_name (str): Model name ('llama' or 'internvl3')
        available_memory_gb (float): Available GPU memory in GB
        
    Returns:
        int: Recommended batch size based on available memory
    """
    if not AUTO_BATCH_SIZE_ENABLED or available_memory_gb is None:
        return get_batch_size_for_model(model_name, CURRENT_BATCH_STRATEGY)
    
    # Determine memory tier
    if available_memory_gb >= GPU_MEMORY_THRESHOLDS['high']:
        strategy = 'aggressive'
    elif available_memory_gb >= GPU_MEMORY_THRESHOLDS['medium']:
        strategy = 'balanced'
    else:
        strategy = 'conservative'
    
    return get_batch_size_for_model(model_name, strategy)