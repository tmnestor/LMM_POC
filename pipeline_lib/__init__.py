"""
pipeline_lib - Lightweight library for vision-language model pipelines

Clean, minimal implementation extracted from common/ with no technical debt.
Self-contained modules with no cross-dependencies.

Modules:
    parser: hybrid_parse_response for VLM output parsing
    cleaner: ExtractionCleaner for field value normalization
    evaluator: Ground truth loading and accuracy calculation
    stages: Pipeline stage functions for extraction workflow
    model_diagnostics: GPU diagnostics for Llama and InternVL3 models
    utils: Generic utility functions for pipeline notebooks

Total size: ~1,800 lines (vs 3,313 lines in common/)
"""

from .cleaner import ExtractionCleaner, sanitize_for_rich
from .evaluator import calculate_field_accuracy, load_ground_truth
from .model_diagnostics import show_model_diagnostics
from .parser import hybrid_parse_response
from .stages import stage_3_parsing, stage_4_cleaning, stage_5_evaluation
from .utils import show_pipeline_memory

__version__ = "1.0.0"

__all__ = [
    # Parser
    "hybrid_parse_response",
    # Cleaner
    "ExtractionCleaner",
    "sanitize_for_rich",
    # Evaluator
    "load_ground_truth",
    "calculate_field_accuracy",
    # Stages
    "stage_3_parsing",
    "stage_4_cleaning",
    "stage_5_evaluation",
    # Diagnostics
    "show_model_diagnostics",
    # Utils
    "show_pipeline_memory",
]
