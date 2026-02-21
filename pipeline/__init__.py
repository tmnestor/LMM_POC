"""Composable pipeline stages for document extraction.

Each stage reads the output of the preceding stage:
    classify  -> CSV  (image_path, document_type, confidence, ...)
    extract   -> JSON (nested extracted_data per image)
    evaluate  -> CSV summary + JSON field-level detail
"""

from .classify import classify_images, read_classification_csv, write_classification_csv
from .evaluate import evaluate_extractions, write_evaluation_csv, write_evaluation_json
from .extract import extract_documents, read_extraction_json, write_extraction_json
from .io_schemas import (
    ClassificationOutput,
    ClassificationRow,
    EvaluationOutput,
    ExtractionOutput,
    ExtractionRecord,
    FieldEvaluation,
    ImageEvaluation,
)

__all__ = [
    # Schemas
    "ClassificationRow",
    "ClassificationOutput",
    "ExtractionRecord",
    "ExtractionOutput",
    "FieldEvaluation",
    "ImageEvaluation",
    "EvaluationOutput",
    # Classify stage
    "classify_images",
    "write_classification_csv",
    "read_classification_csv",
    # Extract stage
    "extract_documents",
    "write_extraction_json",
    "read_extraction_json",
    # Evaluate stage
    "evaluate_extractions",
    "write_evaluation_csv",
    "write_evaluation_json",
]
