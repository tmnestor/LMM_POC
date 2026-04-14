"""Typed dataclasses for batch processing pipeline phases.

These replace the anonymous dicts that currently flow between detection,
extraction, and evaluation phases in batch_processor.py.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Output of document type detection for a single image."""

    image_path: str
    image_name: str
    document_type: str
    classification_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    """Output of field extraction for a single image."""

    image_path: str
    image_name: str
    document_type: str
    extracted_data: dict[str, str]
    processing_time: float
    prompt_used: str
    skip_math_enhancement: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchStats:
    """Throughput metadata from a batch run."""

    configured_batch_size: int
    avg_detection_batch: float
    avg_extraction_batch: float
    num_detection_calls: int
    num_extraction_calls: int

    def to_dict(self) -> dict[str, float]:
        """Convert to the dict shape expected by cli.py and multi_gpu.py."""
        return {
            "configured_batch_size": float(self.configured_batch_size),
            "avg_detection_batch": self.avg_detection_batch,
            "avg_extraction_batch": self.avg_extraction_batch,
            "num_detection_calls": float(self.num_detection_calls),
            "num_extraction_calls": float(self.num_extraction_calls),
        }


@dataclass(frozen=True)
class ImageResult:
    """Complete result for a single image through the full pipeline."""

    image_name: str
    image_path: str
    document_type: str
    extraction_result: dict[str, Any]
    evaluation: dict[str, Any]
    processing_time: float
    prompt_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible dict for BatchAnalytics/BatchReporter/BatchVisualizer."""
        d: dict[str, Any] = {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "document_type": self.document_type,
            "extraction_result": self.extraction_result,
            "evaluation": self.evaluation,
            "processing_time": self.processing_time,
            "prompt_used": self.prompt_used,
            "timestamp": self.timestamp,
        }
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass(frozen=True)
class BatchResult:
    """Aggregate output from a full batch processing run."""

    results: list[ImageResult]
    processing_times: list[float]
    document_types_found: dict[str, int]
    stats: BatchStats

    def results_as_dicts(self) -> list[dict[str, Any]]:
        """Produce the list[dict] shape downstream consumers expect."""
        return [r.to_dict() for r in self.results]

    def as_tuple(self) -> tuple[list[dict], list[float], dict[str, int]]:
        """Drop-in replacement for the current 3-tuple return from process_batch()."""
        return (
            self.results_as_dicts(),
            self.processing_times,
            self.document_types_found,
        )
