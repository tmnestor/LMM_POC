"""
Batch Processing Module for Document-Aware Extraction

Handles batch processing of images through document detection and extraction pipeline.
Delegates to composable pipeline stages (pipeline.classify, pipeline.extract, pipeline.evaluate).
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

logger = logging.getLogger(__name__)


def load_document_field_definitions() -> dict[str, list[str]]:
    """
    Load document-aware field definitions from field_definitions.yaml.

    CRITICAL: This function will raise an exception if YAML loading fails.
    NO FALLBACKS - fail fast with clear diagnostics.

    Returns:
        Dictionary mapping document types (lowercase) to field lists

    Raises:
        FileNotFoundError: If field_definitions.yaml does not exist
        ValueError: If YAML structure is invalid or missing required fields
    """
    import yaml

    field_def_path = Path(__file__).parent.parent / "config" / "field_definitions.yaml"

    # Check file exists first for clear error message
    if not field_def_path.exists():
        raise FileNotFoundError(
            f"❌ FATAL: Field definitions file not found\n"
            f"Expected location: {field_def_path.absolute()}\n"
            f"This file is REQUIRED for document-aware field filtering.\n"
            f"Ensure config/field_definitions.yaml exists in the project root."
        )

    try:
        with field_def_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"❌ FATAL: Invalid YAML syntax in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Error: {e}\n"
            f"Fix the YAML syntax errors before proceeding."
        ) from e
    except Exception as e:
        raise ValueError(
            f"❌ FATAL: Could not read field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Error: {e}"
        ) from e

    # Validate structure
    if "document_fields" not in config:
        raise ValueError(
            f"❌ FATAL: Missing 'document_fields' section in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Required structure:\n"
            f"document_fields:\n"
            f"  invoice:\n"
            f"    fields: [list of fields]\n"
            f"  receipt:\n"
            f"    fields: [list of fields]\n"
            f"  bank_statement:\n"
            f"    fields: [list of fields]"
        )

    doc_fields = config["document_fields"]

    # Validate all document types defined in YAML (excluding 'universal')
    result = {}
    for doc_type, type_config in doc_fields.items():
        if doc_type == "universal":
            continue
        if "fields" not in type_config:
            raise ValueError(
                f"❌ FATAL: Missing 'fields' list for '{doc_type}' in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must have a 'fields' list."
            )
        if not type_config["fields"]:
            raise ValueError(
                f"❌ FATAL: Empty 'fields' list for '{doc_type}' in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must have at least one field defined."
            )
        result[doc_type] = type_config["fields"]

    if not result:
        raise ValueError(
            f"❌ FATAL: No document types defined in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Expected: document_fields section with at least one document type."
        )

    return result


class BatchDocumentProcessor:
    """Handles batch processing of documents with extraction and evaluation.

    Delegates to composable pipeline stages:
        pipeline.classify.classify_images()
        pipeline.extract.extract_documents()
        pipeline.evaluate.evaluate_extractions()

    Returns legacy format for backward compatibility with analytics/reporting.
    """

    def __init__(
        self,
        model,
        prompt_config: dict,
        ground_truth_csv: str | None,
        console: Console | None = None,
        enable_math_enhancement: bool = True,
        bank_adapter=None,
        field_definitions: dict[str, list[str]] | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize batch processor for document extraction.

        Args:
            model: Model handler implementing detect_and_classify_document() and process_document_aware()
            prompt_config: Dictionary with prompt file paths and keys
            ground_truth_csv: Path to ground truth CSV file
            console: Rich console for output
            enable_math_enhancement: Whether to apply mathematical enhancement for bank statements
            bank_adapter: Optional BankStatementAdapter for sophisticated bank extraction
            field_definitions: Pre-loaded field definitions dict. If None, loads from YAML.
            batch_size: Images per batch (None = auto-detect from VRAM, 1 = sequential)
        """
        self.model_handler = model
        self.prompt_config = prompt_config
        self.ground_truth_csv = ground_truth_csv
        self.console = console or Console()
        self.enable_math_enhancement = enable_math_enhancement
        self.bank_adapter = bank_adapter
        self.batch_size = batch_size

        # Initialize file-based trace logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trace_file = f"batch_processor_trace_{timestamp}.log"

        # Cache field definitions and evaluation method (loaded once, not per-image)
        self.doc_type_fields = field_definitions or load_document_field_definitions()
        self.evaluation_method = os.environ.get("EVALUATION_METHOD", "order_aware_f1")

        # Batch stats — populated after process_batch() completes
        self.batch_stats: dict[str, float] = {}

    def _resolve_batch_size(self) -> int:
        """Resolve effective batch size from config or auto-detection."""
        if self.batch_size is not None:
            return max(1, self.batch_size)
        return getattr(self.model_handler, "batch_size", 1)

    def process_batch(
        self, image_paths: list[str], verbose: bool = True, progress_interval: int = 5
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """Process a batch of images through the composable pipeline.

        Delegates to pipeline stages: classify -> extract -> evaluate.
        Returns legacy format for backward compatibility with analytics/reporting.

        Args:
            image_paths: List of image file paths
            verbose: Whether to show progress updates
            progress_interval: How often to show detailed progress

        Returns:
            Tuple of (batch_results, processing_times, document_types_found)
        """
        from pipeline.classify import classify_images
        from pipeline.evaluate import evaluate_extractions
        from pipeline.extract import extract_documents

        effective_batch_size = self._resolve_batch_size()

        logger.info("Starting Batch Processing")
        logger.info("Batch size: %d", effective_batch_size)
        self.console.rule("[bold green]Batch Extraction[/bold green]")

        # Stage 1: Classification
        classifications = classify_images(
            processor=self.model_handler,
            image_paths=image_paths,
            batch_size=effective_batch_size,
            verbose=verbose,
            console=self.console,
        )

        # Stage 2: Extraction
        extractions = extract_documents(
            processor=self.model_handler,
            classifications=classifications,
            bank_adapter=self.bank_adapter,
            batch_size=effective_batch_size,
            verbose=verbose,
            console=self.console,
        )

        # Stage 3: Evaluation (if ground truth provided)
        evaluation_output = None
        if self.ground_truth_csv:
            evaluation_output = evaluate_extractions(
                extractions=extractions,
                ground_truth_csv=Path(self.ground_truth_csv),
                field_definitions=self.doc_type_fields,
                evaluation_method=self.evaluation_method,
                enable_math_enhancement=self.enable_math_enhancement,
                verbose=verbose,
            )

        self.console.rule("[bold green]Batch Processing Complete[/bold green]")

        # Populate batch stats
        self.batch_stats = {
            "avg_detection_batch": float(effective_batch_size),
            "avg_extraction_batch": float(effective_batch_size),
            "num_detection_calls": max(
                1, len(image_paths) // max(1, effective_batch_size)
            ),
            "num_extraction_calls": max(
                1, len(image_paths) // max(1, effective_batch_size)
            ),
            "configured_batch_size": effective_batch_size,
        }

        return self._to_legacy_format(
            classifications,
            extractions,
            evaluation_output,
        )

    def _to_legacy_format(
        self,
        classifications,
        extractions,
        evaluation_output,
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """Convert pipeline stage outputs to legacy return format.

        Preserves backward compatibility with analytics, reporting, and
        visualization modules that expect the original process_batch() output.
        """
        batch_results: list[dict] = []
        processing_times: list[float] = []
        document_types_found: dict[str, int] = {}

        # Build evaluation lookup by image_name if available
        eval_by_name: dict[str, Any] = {}
        if evaluation_output is not None:
            for ie in evaluation_output.image_evaluations:
                eval_by_name[ie.image_name] = {
                    "overall_accuracy": ie.overall_f1,
                    "median_f1": ie.median_f1,
                    "overall_precision": ie.precision,
                    "overall_recall": ie.recall,
                    "total_fields": ie.total_fields,
                    "correct_fields": ie.correct_fields,
                    "missing_fields": ie.total_fields - ie.correct_fields,
                    "incorrect_fields": ie.total_fields - ie.correct_fields,
                    "fields_extracted": ie.fields_extracted,
                    "fields_matched": ie.correct_fields,
                    "field_scores": {
                        fe.field_name: {
                            "f1_score": fe.f1_score,
                            "precision": fe.precision,
                            "recall": fe.recall,
                        }
                        for fe in ie.field_evaluations
                    },
                    "overall_metrics": {
                        "overall_accuracy": ie.overall_f1,
                        "median_f1": ie.median_f1,
                        "overall_precision": ie.precision,
                        "overall_recall": ie.recall,
                        "meets_threshold": ie.median_f1 >= 0.8,
                        "document_type_threshold": 0.8,
                    },
                }

        for i, record in enumerate(extractions.records):
            doc_type = classifications.rows[i].document_type
            document_types_found[doc_type] = document_types_found.get(doc_type, 0) + 1
            processing_times.append(record.processing_time)

            if record.error:
                batch_results.append(
                    {
                        "image_name": record.image_name,
                        "image_path": record.image_path,
                        "error": record.error,
                        "processing_time": record.processing_time,
                    }
                )
                continue

            evaluation = eval_by_name.get(
                record.image_name,
                {
                    "error": f"No ground truth for {record.image_name}",
                    "overall_accuracy": 0,
                },
            )

            batch_results.append(
                {
                    "image_name": record.image_name,
                    "image_path": record.image_path,
                    "document_type": doc_type,
                    "extraction_result": {
                        "extracted_data": record.extracted_data,
                        "document_type": doc_type,
                        "image_file": record.image_name,
                        "processing_time": record.processing_time,
                    },
                    "evaluation": evaluation,
                    "processing_time": record.processing_time,
                    "prompt_used": record.prompt_used,
                    "timestamp": record.timestamp,
                }
            )

        return batch_results, processing_times, document_types_found


def print_accuracy_by_document_type(
    batch_results: list[dict],
    console: Console | None = None,
) -> dict:
    """Print accuracy summary grouped dynamically by document type.

    Groups results by the actual document types found in batch_results
    rather than using a hardcoded type list.
    """
    if console is None:
        console = Console()

    # Group results dynamically by document type
    doc_type_results: dict[str, list[dict]] = {}

    for result in batch_results:
        if "error" in result:
            continue

        doc_type = result.get("document_type", "UNKNOWN").upper()
        evaluation = result.get("evaluation", {})

        if not evaluation or "overall_accuracy" not in evaluation:
            continue

        doc_type_results.setdefault(doc_type, []).append(result)

    # Calculate and display metrics for each document type
    console.rule("[bold cyan]Accuracy by Document Type[/bold cyan]")

    summary = {}

    for doc_type_key, results in sorted(doc_type_results.items()):
        if not results:
            continue

        # Extract metrics
        mean_f1_scores = []
        median_f1_scores = []

        for r in results:
            eval_data = r.get("evaluation", {})
            mean_f1_scores.append(eval_data.get("overall_accuracy", 0))
            median_f1_scores.append(eval_data.get("median_f1", 0))

        # Calculate aggregates
        n_docs = len(results)
        avg_mean_f1 = sum(mean_f1_scores) / n_docs if n_docs else 0
        avg_median_f1 = sum(median_f1_scores) / n_docs if n_docs else 0

        # Calculate median of medians (most robust)
        sorted_medians = sorted(median_f1_scores)
        mid = len(sorted_medians) // 2
        if len(sorted_medians) % 2 == 0 and len(sorted_medians) > 0:
            median_of_medians = (sorted_medians[mid - 1] + sorted_medians[mid]) / 2
        elif len(sorted_medians) > 0:
            median_of_medians = sorted_medians[mid]
        else:
            median_of_medians = 0

        display_name = doc_type_key.replace("_", " ").title()

        logger.info("%s", display_name)
        logger.info("  Documents: %d", n_docs)
        logger.info(
            "  Median F1 (avg): %.1f%% - typical field performance",
            avg_median_f1 * 100,
        )
        logger.info("  Mean F1 (avg): %.1f%%", avg_mean_f1 * 100)
        logger.info(
            "  Median of Medians: %.1f%% - most robust", median_of_medians * 100
        )

        summary[doc_type_key] = {
            "count": n_docs,
            "avg_mean_f1": avg_mean_f1,
            "avg_median_f1": avg_median_f1,
            "median_of_medians": median_of_medians,
        }

    # Overall summary (weighted by document count)
    total_docs = sum(s["count"] for s in summary.values())
    if total_docs > 0:
        weighted_median = (
            sum(s["avg_median_f1"] * s["count"] for s in summary.values()) / total_docs
        )
        weighted_mean = (
            sum(s["avg_mean_f1"] * s["count"] for s in summary.values()) / total_docs
        )

        logger.info("Overall (weighted by document count)")
        logger.info("  Total Documents: %d", total_docs)
        logger.info("  Weighted Median F1: %.1f%%", weighted_median * 100)
        logger.info("  Weighted Mean F1: %.1f%%", weighted_mean * 100)

        summary["overall"] = {
            "count": total_docs,
            "weighted_median_f1": weighted_median,
            "weighted_mean_f1": weighted_mean,
        }

    console.rule()

    return summary
