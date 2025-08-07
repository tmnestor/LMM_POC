"""
Shared configuration for vision model evaluation.

This module contains all configuration values and constants shared between
different vision models (InternVL3, Llama, etc.).
"""


# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

# Primary paths (local development)
DATA_DIR = "/home/jovyan/nfs_share/tod/evaluation_data"
GROUND_TRUTH_PATH = "/home/jovyan/nfs_share/tod/unified_vision_processor_minimal/evaluation_ground_truth.csv"
OUTPUT_DIR = "/home/jovyan/nfs_share/tod/output"

# Alternative paths (EFS deployment)
# DATA_DIR = "/efs/share/PoC_data/evaluation_data"
# GROUND_TRUTH_PATH = "/efs/share/PoC_data/evaluation_ground_truth.csv"
# OUTPUT_DIR = "/efs/share/PoC_data/output"

# ============================================================================
# EXTRACTION FIELDS
# ============================================================================

# Production extraction fields - update this list as needed
# Keep in alphabetical order for consistent column ordering
EXTRACTION_FIELDS = [
    'ABN',
    'ACCOUNT_HOLDER',
    'BANK_ACCOUNT_NUMBER',
    'BANK_NAME',
    'BSB_NUMBER',
    'BUSINESS_ADDRESS',
    'BUSINESS_PHONE',
    'CLOSING_BALANCE',
    'DESCRIPTIONS',
    'DOCUMENT_TYPE',
    'DUE_DATE',
    'GST',
    'INVOICE_DATE',
    'OPENING_BALANCE',
    'PAYER_ADDRESS',
    'PAYER_EMAIL',
    'PAYER_NAME',
    'PAYER_PHONE',
    'PRICES',
    'QUANTITIES',
    'STATEMENT_PERIOD',
    'SUBTOTAL',
    'SUPPLIER',
    'SUPPLIER_WEBSITE',
    'TOTAL'
    # Add new fields here in alphabetical order:
    # 'NEW_FIELD_1',
    # 'NEW_FIELD_2',
    # etc.
]

# Field count - automatically calculated
FIELD_COUNT = len(EXTRACTION_FIELDS)

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