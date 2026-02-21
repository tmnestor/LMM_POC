"""I/O schemas for pipeline stage boundaries.

Defines the data contracts between classify -> extract -> evaluate stages.
Each stage reads the previous stage's output format and produces its own.

These dataclasses are the single source of truth for what flows between stages.
The CSV/JSON serialization functions in each stage module handle conversion.
"""

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Classification stage
# ---------------------------------------------------------------------------


@dataclass
class ClassificationRow:
    """Single classification result. Maps 1:1 to a CSV row."""

    image_path: str
    document_type: str  # Uppercase canonical: INVOICE, RECEIPT, BANK_STATEMENT
    confidence: float
    raw_response: str
    prompt_used: str
    error: str = ""
    timestamp: str = ""


@dataclass
class ClassificationOutput:
    """Full classification stage output."""

    rows: list[ClassificationRow]
    model_type: str
    timestamp: str


# ---------------------------------------------------------------------------
# Extraction stage
# ---------------------------------------------------------------------------


@dataclass
class ExtractionRecord:
    """Single extraction result. One per image."""

    image_path: str
    image_name: str
    document_type: str
    extracted_data: dict[str, str]
    processing_time: float
    prompt_used: str
    field_count: int
    fields_found: int
    timestamp: str
    raw_response: str = ""
    error: str = ""
    skip_math_enhancement: bool = False


@dataclass
class ExtractionOutput:
    """Full extraction stage output."""

    records: list[ExtractionRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation stage
# ---------------------------------------------------------------------------


@dataclass
class FieldEvaluation:
    """Per-field evaluation detail."""

    field_name: str
    f1_score: float
    precision: float
    recall: float
    extracted_value: str
    ground_truth_value: str


@dataclass
class ImageEvaluation:
    """Per-image evaluation result."""

    image_name: str
    image_path: str
    document_type: str
    overall_f1: float
    median_f1: float
    precision: float
    recall: float
    total_fields: int
    correct_fields: int
    fields_extracted: int
    field_evaluations: list[FieldEvaluation] = field(default_factory=list)


@dataclass
class EvaluationOutput:
    """Full evaluation stage output."""

    image_evaluations: list[ImageEvaluation]
    summary: dict[str, Any] = field(default_factory=dict)
