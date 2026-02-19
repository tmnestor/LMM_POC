"""
Batch Processing Module for Document-Aware Extraction

Handles batch processing of images through document detection and extraction pipeline.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .evaluation_metrics import (
    calculate_correlation_aware_f1,
    calculate_field_accuracy_with_method,
    load_ground_truth,
)

# Import Rich content sanitization to prevent recursion errors and ExtractionCleaner
from .simple_model_evaluator import SimpleModelEvaluator

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
    """Handles batch processing of documents with extraction and evaluation."""

    def __init__(
        self,
        model,
        prompt_config: dict,
        ground_truth_csv: str,
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
        # Store model handler
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

        # Use SimpleModelEvaluator for model comparison
        self.model_evaluator = SimpleModelEvaluator()
        self.ground_truth_data = None

        # Cache field definitions and evaluation method (loaded once, not per-image)
        self.doc_type_fields = field_definitions or load_document_field_definitions()
        self.evaluation_method = os.environ.get("EVALUATION_METHOD", "order_aware_f1")

        # Batch stats — populated after process_batch() completes
        self.batch_stats: dict[str, float] = {}

    def _trace_log(self, message: str):
        """Log message to both console and file"""
        logger.debug("%s", message)
        with Path(self._trace_file).open("a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def _resolve_batch_size(self) -> int:
        """Resolve effective batch size from config or auto-detection."""
        if self.batch_size is not None:
            return max(1, self.batch_size)

        # Auto-detect from model's configured batch size
        return getattr(self.model_handler, "batch_size", 1)

    def _lookup_ground_truth(self, image_path: str) -> dict:
        """Look up ground truth for an image with fuzzy matching."""
        if not self.ground_truth_data:
            return {}

        image_name = Path(image_path).name

        # Try exact match first
        ground_truth = self.ground_truth_data.get(image_name, {})
        if ground_truth:
            return ground_truth

        # Try without extension
        image_stem = Path(image_path).stem
        for gt_key in self.ground_truth_data:
            if Path(gt_key).stem == image_stem or gt_key == image_stem:
                return self.ground_truth_data[gt_key]

        return {}

    def process_batch(
        self, image_paths: list[str], verbose: bool = True, progress_interval: int = 5
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """
        Process a batch of images through the extraction pipeline.

        Uses a two-phase pipeline when batch_size > 1:
          Phase 1: Batched document detection (batch_chat)
          Phase 2: Batched extraction for standard docs, sequential for bank statements
          Phase 3: Evaluation (unchanged, CPU-only)

        Falls back to sequential processing when batch_size == 1.

        Args:
            image_paths: List of image file paths
            verbose: Whether to show progress updates
            progress_interval: How often to show detailed progress

        Returns:
            Tuple of (batch_results, processing_times, document_types_found)
        """
        # Load ground truth data once for the batch
        try:
            self.ground_truth_data = load_ground_truth(
                self.ground_truth_csv, verbose=verbose
            )
            logger.info(
                "Loaded ground truth for %d images", len(self.ground_truth_data)
            )
            sample_keys = list(self.ground_truth_data.keys())[:3]
            logger.debug("Sample GT keys: %s", sample_keys)
        except Exception as e:
            logger.error("Error loading ground truth: %s", e)
            self.ground_truth_data = {}

        effective_batch_size = self._resolve_batch_size()

        logger.info("Starting Batch Processing")
        logger.info("Batch size: %d", effective_batch_size)
        self.console.rule("[bold green]Batch Extraction[/bold green]")

        # Route to batched or sequential processing
        from models.protocol import BatchCapableProcessor

        if effective_batch_size > 1 and isinstance(
            self.model_handler, BatchCapableProcessor
        ):
            return self._process_batch_two_phase(
                image_paths, effective_batch_size, verbose, progress_interval
            )
        return self._process_batch_sequential(image_paths, verbose, progress_interval)

    def _process_batch_two_phase(
        self,
        image_paths: list[str],
        batch_size: int,
        verbose: bool,
        progress_interval: int,
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """Two-phase batched processing using model.batch_chat().

        Phase 1: Batch detection — classify all images
        Phase 2: Batch extraction — standard docs batched, bank statements sequential
        Phase 3: Evaluation — per-image, CPU-only
        """
        total_images = len(image_paths)
        batch_results: list[dict] = [None] * total_images  # type: ignore[list-item]
        processing_times: list[float] = [0.0] * total_images
        document_types_found: dict[str, int] = {}

        # Progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[current]}[/cyan]"),
            console=self.console,
            transient=False,
        )
        progress_task = progress.add_task(
            "Processing images", total=total_images, current=""
        )

        # Track actual batch sizes for reporting
        detection_batch_sizes: list[int] = []
        extraction_batch_sizes: list[int] = []

        # ================================================================
        # PHASE 1: BATCHED DETECTION
        # ================================================================
        logger.info("Phase 1: Batched Document Detection")

        detection_start = time.time()
        all_classification_infos: list[dict] = []

        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_paths = image_paths[batch_start:batch_end]
            detection_batch_sizes.append(len(batch_paths))

            logger.debug(
                "Detecting batch [%d-%d] / %d",
                batch_start + 1,
                batch_end,
                total_images,
            )

            batch_classifications = self.model_handler.batch_detect_documents(
                batch_paths, verbose=verbose
            )
            all_classification_infos.extend(batch_classifications)

        # Count document types
        for info in all_classification_infos:
            doc_type = info["document_type"]
            document_types_found[doc_type] = document_types_found.get(doc_type, 0) + 1

        detection_time = time.time() - detection_start
        logger.info(
            "Detection complete: %.1fs for %d images", detection_time, total_images
        )
        for doc_type, count in sorted(document_types_found.items()):
            logger.info("  %s: %d", doc_type, count)

        # ================================================================
        # PHASE 2: EXTRACTION (batched for standard, sequential for bank)
        # ================================================================
        logger.info("Phase 2: Document Extraction")

        extraction_start = time.time()

        # Partition images into bank vs standard, preserving original indices
        bank_indices = []
        standard_indices = []
        for i, info in enumerate(all_classification_infos):
            if info["document_type"].upper() == "BANK_STATEMENT":
                bank_indices.append(i)
            else:
                standard_indices.append(i)

        logger.debug(
            "Standard documents: %d, Bank statements: %d",
            len(standard_indices),
            len(bank_indices),
        )

        # --- Standard documents: batched extraction ---
        if standard_indices:
            for batch_start_idx in range(0, len(standard_indices), batch_size):
                batch_end_idx = min(batch_start_idx + batch_size, len(standard_indices))
                batch_orig_indices = standard_indices[batch_start_idx:batch_end_idx]

                batch_paths = [image_paths[i] for i in batch_orig_indices]
                batch_class_infos = [
                    all_classification_infos[i] for i in batch_orig_indices
                ]
                extraction_batch_sizes.append(len(batch_paths))

                logger.debug(
                    "Extracting standard batch [%d-%d] / %d",
                    batch_start_idx + 1,
                    batch_end_idx,
                    len(standard_indices),
                )

                batch_extract_start = time.time()
                extraction_results = self.model_handler.batch_extract_documents(
                    batch_paths, batch_class_infos, verbose=verbose
                )
                batch_extract_time = time.time() - batch_extract_start

                # Distribute time evenly across batch
                per_image_time = (
                    batch_extract_time / len(batch_orig_indices)
                    if batch_orig_indices
                    else 0
                )

                for j, orig_idx in enumerate(batch_orig_indices):
                    image_path = image_paths[orig_idx]
                    image_name = Path(image_path).name
                    document_type = all_classification_infos[orig_idx]["document_type"]
                    extraction_result = extraction_results[j]

                    # Format extraction result for evaluation
                    formatted_result = {
                        "extracted_data": extraction_result.get("extracted_data", {}),
                        "document_type": document_type,
                        "image_file": image_name,
                        "processing_time": per_image_time,
                    }

                    # Evaluate
                    ground_truth = self._lookup_ground_truth(image_path)
                    if ground_truth:
                        evaluation = self._evaluate_extraction(
                            formatted_result,
                            ground_truth,
                            document_type,
                            image_name,
                            verbose,
                        )
                    else:
                        evaluation = {
                            "error": f"No ground truth for {image_name}",
                            "overall_accuracy": 0,
                        }

                    processing_times[orig_idx] = per_image_time
                    batch_results[orig_idx] = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "document_type": document_type,
                        "extraction_result": formatted_result,
                        "evaluation": evaluation,
                        "processing_time": per_image_time,
                        "prompt_used": f"batch_{document_type.lower()}",
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Update progress
                    progress.update(progress_task, advance=1, current=image_name)
                    self.console.print(progress.get_renderable())

                    if evaluation and "median_f1" in evaluation:
                        median_f1_pct = evaluation.get("median_f1", 0) * 100
                        mean_f1_pct = evaluation.get("overall_accuracy", 0) * 100
                        logger.debug(
                            "%s: Median %.1f%% | Mean %.1f%% | %.1fs",
                            image_name,
                            median_f1_pct,
                            mean_f1_pct,
                            per_image_time,
                        )

        # --- Bank statements: sequential extraction (multi-turn required) ---
        for orig_idx in bank_indices:
            extraction_batch_sizes.append(1)
            image_path = image_paths[orig_idx]
            image_name = Path(image_path).name

            logger.info("BANK STATEMENT (sequential): %s", image_name)

            bank_start = time.time()

            try:
                document_type, extraction_result, prompt_name = self._process_image(
                    image_path, verbose
                )

                bank_time = time.time() - bank_start

                # Evaluate
                ground_truth = self._lookup_ground_truth(image_path)
                if ground_truth:
                    evaluation = self._evaluate_extraction(
                        extraction_result,
                        ground_truth,
                        document_type,
                        image_name,
                        verbose,
                    )
                else:
                    evaluation = {
                        "error": f"No ground truth for {image_name}",
                        "overall_accuracy": 0,
                    }

                processing_times[orig_idx] = bank_time
                batch_results[orig_idx] = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "document_type": document_type,
                    "extraction_result": extraction_result,
                    "evaluation": evaluation,
                    "processing_time": bank_time,
                    "prompt_used": prompt_name,
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error("Error processing bank statement %s: %s", image_name, e)
                bank_time = time.time() - bank_start
                processing_times[orig_idx] = bank_time
                batch_results[orig_idx] = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "error": str(e),
                    "processing_time": bank_time,
                }

            progress.update(progress_task, advance=1, current=image_name)
            self.console.print(progress.get_renderable())

            if batch_results[orig_idx].get("evaluation"):
                eval_data = batch_results[orig_idx]["evaluation"]
                if "median_f1" in eval_data:
                    median_f1_pct = eval_data.get("median_f1", 0) * 100
                    mean_f1_pct = eval_data.get("overall_accuracy", 0) * 100
                    logger.debug(
                        "%s: Median %.1f%% | Mean %.1f%% | %.1fs",
                        image_name,
                        median_f1_pct,
                        mean_f1_pct,
                        bank_time,
                    )

        extraction_time = time.time() - extraction_start
        logger.info("Extraction complete: %.1fs", extraction_time)

        # Print final progress
        progress.update(progress_task, current="done")
        self.console.print(progress.get_renderable())

        self.console.rule("[bold green]Batch Processing Complete[/bold green]")

        # Store batch stats for summary reporting
        avg_detect = (
            sum(detection_batch_sizes) / len(detection_batch_sizes)
            if detection_batch_sizes
            else 1.0
        )
        avg_extract = (
            sum(extraction_batch_sizes) / len(extraction_batch_sizes)
            if extraction_batch_sizes
            else 1.0
        )
        self.batch_stats = {
            "avg_detection_batch": avg_detect,
            "avg_extraction_batch": avg_extract,
            "num_detection_calls": len(detection_batch_sizes),
            "num_extraction_calls": len(extraction_batch_sizes),
            "configured_batch_size": batch_size,
        }

        return batch_results, processing_times, document_types_found

    def _process_batch_sequential(
        self,
        image_paths: list[str],
        verbose: bool,
        progress_interval: int,
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """Sequential processing (batch_size=1). Original behavior."""
        start_time = time.time()

        batch_results = []
        processing_times = []
        document_types_found = {}

        logger.info("Starting Batch Processing")
        self.console.rule("[bold green]Batch Extraction[/bold green]")

        total_images = len(image_paths)
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[current]}[/cyan]"),
            console=self.console,
            transient=False,
        )

        progress_task = progress.add_task(
            "Processing images", total=total_images, current=""
        )

        for idx, image_path in enumerate(image_paths, 1):
            image_name = Path(image_path).name

            progress.update(progress_task, current=image_name)
            self.console.print(progress.get_renderable())

            logger.debug("Processing [%d/%d]: %s", idx, total_images, image_name)

            try:
                img_start_time = time.time()

                logger.debug(
                    "TRACE: Processing image %d/%d: %s",
                    idx,
                    total_images,
                    image_name,
                )

                document_type, extraction_result, prompt_name = self._process_image(
                    image_path, verbose
                )
                document_types_found[document_type] = (
                    document_types_found.get(document_type, 0) + 1
                )

                logger.debug(
                    "TRACE: Processing complete for %s, doc_type=%s",
                    image_name,
                    document_type,
                )

                image_name = Path(image_path).name

                ground_truth = self._lookup_ground_truth(image_path)

                if ground_truth:
                    evaluation = self._evaluate_extraction(
                        extraction_result,
                        ground_truth,
                        document_type,
                        image_name,
                        verbose,
                    )
                else:
                    evaluation = {
                        "error": f"No ground truth for {image_name}",
                        "overall_accuracy": 0,
                    }

                processing_time = time.time() - img_start_time
                processing_times.append(processing_time)

                result = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "document_type": document_type,
                    "extraction_result": extraction_result,
                    "evaluation": evaluation,
                    "processing_time": processing_time,
                    "prompt_used": prompt_name,
                    "timestamp": datetime.now().isoformat(),
                }
                batch_results.append(result)

                progress.update(progress_task, advance=1)

                if evaluation and "median_f1" in evaluation:
                    median_f1_pct = evaluation.get("median_f1", 0) * 100
                    mean_f1_pct = evaluation.get("overall_accuracy", 0) * 100
                    logger.debug(
                        "%s: Median %.1f%% | Mean %.1f%% | %.1fs",
                        image_name,
                        median_f1_pct,
                        mean_f1_pct,
                        processing_time,
                    )
                elif evaluation:
                    mean_f1_pct = evaluation.get("overall_accuracy", 0) * 100
                    logger.debug(
                        "%s: F1 %.1f%% | %.1fs",
                        image_name,
                        mean_f1_pct,
                        processing_time,
                    )

                if idx % progress_interval == 0 or idx == total_images:
                    accuracy = (
                        evaluation.get("overall_accuracy", 0) * 100 if evaluation else 0
                    )
                    logger.debug(
                        "[%d/%d] %s: %s - Accuracy: %.1f%% - Time: %.2fs",
                        idx,
                        total_images,
                        image_name,
                        document_type,
                        accuracy,
                        processing_time,
                    )

            except Exception as e:
                logger.error("Error processing %s: %s", image_name, e)

                progress.update(progress_task, advance=1)

                batch_results.append(
                    {
                        "image_name": image_name,
                        "image_path": image_path,
                        "error": str(e),
                        "processing_time": time.time() - img_start_time
                        if "img_start_time" in locals()
                        else 0,
                    }
                )

        progress.update(progress_task, current="done")
        self.console.print(progress.get_renderable())

        self.console.rule("[bold green]Batch Processing Complete[/bold green]")

        # Sequential = batch size 1
        self.batch_stats = {
            "avg_detection_batch": 1.0,
            "avg_extraction_batch": 1.0,
            "num_detection_calls": len(image_paths),
            "num_extraction_calls": len(image_paths),
            "configured_batch_size": 1,
        }

        return batch_results, processing_times, document_types_found

    def _evaluate_extraction(
        self,
        extraction_result: dict,
        ground_truth: dict,
        document_type: str,
        image_name: str,
        verbose: bool,
    ) -> dict:
        """
        Evaluate extraction results against ground truth.

        Handles mathematical enhancement for bank statements, field filtering,
        F1 scoring (standard or correlation), and metric aggregation.

        Args:
            extraction_result: Raw extraction result with 'extracted_data' key
            ground_truth: Ground truth dictionary for this image
            document_type: Detected document type (e.g. 'BANK_STATEMENT', 'INVOICE')
            image_name: Image filename for logging
            verbose: Whether to show detailed output

        Returns:
            Evaluation dictionary with accuracy, F1 scores, and field-level metrics
        """
        extracted_data = extraction_result.get("extracted_data", {})

        # Apply mathematical enhancement for bank statements
        # Skip if already handled by UnifiedBankExtractor (V2 sophisticated extraction)
        mathematical_analysis = None
        skip_math = extraction_result.get("skip_math_enhancement", False)
        if (
            document_type.upper() == "BANK_STATEMENT"
            and self.enable_math_enhancement
            and not skip_math
        ):
            from .bank_statement_calculator import (
                enhance_bank_statement_extraction,
            )

            logger.debug("Applying mathematical enhancement for bank statement")

            enhanced_result = enhance_bank_statement_extraction(
                extracted_data, verbose=verbose
            )

            extracted_data = {
                k: v
                for k, v in enhanced_result.items()
                if k != "_mathematical_analysis"
            }

            mathematical_analysis = enhanced_result.get("_mathematical_analysis", {})

            logger.debug("Filtering to debit-only transactions for evaluation")

            extracted_data = self._filter_debit_transactions(extracted_data, verbose)
        elif skip_math:
            logger.debug(
                "Skipping batch_processor math enhancement (handled by UnifiedBankExtractor)"
            )

        found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
        logger.debug("Extracted %d fields from %s", len(found_fields), image_name)

        if (
            document_type.upper() == "BANK_STATEMENT"
            and mathematical_analysis is not None
        ):
            if mathematical_analysis.get("calculation_success"):
                logger.debug(
                    "Mathematical analysis: %d transactions calculated",
                    mathematical_analysis.get("transaction_count", 0),
                )
            else:
                logger.debug("Mathematical analysis failed")

        # Filter ground truth to document-specific fields for accurate evaluation
        document_type_lower_eval = document_type.lower()
        evaluation_fields = self.doc_type_fields.get(
            document_type_lower_eval, self.doc_type_fields["invoice"]
        )

        filtered_ground_truth = {
            field: ground_truth[field]
            for field in evaluation_fields
            if field in ground_truth
        }

        if document_type.upper() == "BANK_STATEMENT":
            logger.debug(
                "Evaluating using mathematically corrected values (not raw VLM output)"
            )

        evaluation_result = self.model_evaluator.evaluate_extraction(
            extracted_data, filtered_ground_truth, image_name
        )

        fields_extracted = len(
            [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
        )

        # Build field-level F1 scores
        field_scores = {}
        total_f1_score = 0.0
        total_precision = 0.0
        total_recall = 0.0

        if self.evaluation_method in ["correlation", "correlation_aware_f1"]:
            correlation_result = calculate_correlation_aware_f1(
                extracted_data,
                filtered_ground_truth,
                document_type,
                debug=False,
            )

            for field in filtered_ground_truth:
                field_scores[field] = correlation_result
                total_f1_score += correlation_result["f1_score"]
                total_precision += correlation_result["precision"]
                total_recall += correlation_result["recall"]
        else:
            for field in filtered_ground_truth:
                extracted_val = extracted_data.get(field, "NOT_FOUND")
                ground_val = filtered_ground_truth.get(field, "NOT_FOUND")

                is_debug = field == "IS_GST_INCLUDED" and verbose
                if is_debug:
                    logger.debug("BEFORE EVALUATION:")
                    logger.debug(
                        "  extracted_val = '%s' (type: %s)",
                        extracted_val,
                        type(extracted_val).__name__,
                    )
                    logger.debug(
                        "  ground_val = '%s' (type: %s)",
                        ground_val,
                        type(ground_val).__name__,
                    )
                    logger.debug("  Are they equal? %s", extracted_val == ground_val)

                f1_metrics = calculate_field_accuracy_with_method(
                    extracted_val,
                    ground_val,
                    field,
                    method=self.evaluation_method,
                    debug=is_debug,
                )

                if is_debug:
                    logger.debug("AFTER EVALUATION (%s):", field)
                    logger.debug(
                        "  Field '%s' f1_score = %s", field, f1_metrics["f1_score"]
                    )

                field_scores[field] = f1_metrics
                total_f1_score += f1_metrics["f1_score"]
                total_precision += f1_metrics["precision"]
                total_recall += f1_metrics["recall"]

        # Calculate overall metrics from F1 scores
        num_fields = len(field_scores)
        overall_accuracy = total_f1_score / num_fields if num_fields else 0.0
        overall_precision = total_precision / num_fields if num_fields else 0.0
        overall_recall = total_recall / num_fields if num_fields else 0.0

        # Calculate median F1 (more robust to outliers than mean)
        f1_values = [s["f1_score"] for s in field_scores.values()]
        if f1_values:
            sorted_f1 = sorted(f1_values)
            mid = len(sorted_f1) // 2
            if len(sorted_f1) % 2 == 0:
                median_f1 = (sorted_f1[mid - 1] + sorted_f1[mid]) / 2
            else:
                median_f1 = sorted_f1[mid]
        else:
            median_f1 = 0.0

        perfect_matches = sum(
            1 for score in field_scores.values() if score["f1_score"] == 1.0
        )

        evaluation = {
            "overall_accuracy": overall_accuracy,
            "median_f1": median_f1,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "total_fields": len(field_scores),
            "correct_fields": perfect_matches,
            "missing_fields": evaluation_result.missing_fields,
            "incorrect_fields": evaluation_result.incorrect_fields,
            "fields_extracted": fields_extracted,
            "fields_matched": perfect_matches,
            "field_scores": field_scores,
            "overall_metrics": {
                "overall_accuracy": overall_accuracy,
                "median_f1": median_f1,
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "meets_threshold": median_f1 >= 0.8,
                "document_type_threshold": 0.8,
            },
        }

        logger.debug(
            "Median F1: %.1f%% | Mean F1: %.1f%% for %s",
            median_f1 * 100,
            overall_accuracy * 100,
            image_name,
        )
        logger.debug(
            "Precision: %.1f%% | Recall: %.1f%%",
            overall_precision * 100,
            overall_recall * 100,
        )
        logger.debug("has_field_scores=True, field_count=%d", len(field_scores))

        return evaluation

    def _filter_debit_transactions(
        self, extracted_data: dict, verbose: bool = False
    ) -> dict:
        """
        Filter bank statement data to keep only debit transactions using pandas.

        This removes credit transactions from all transaction arrays to match ground truth
        which only contains debit transactions.
        """
        if extracted_data.get("DOCUMENT_TYPE") != "BANK_STATEMENT":
            return extracted_data  # Only filter bank statements

        try:
            import pandas as pd

            # Get transaction arrays
            descriptions = extracted_data.get("LINE_ITEM_DESCRIPTIONS", "")
            dates = extracted_data.get("TRANSACTION_DATES", "")
            paid = extracted_data.get("TRANSACTION_AMOUNTS_PAID", "")
            received = extracted_data.get("TRANSACTION_AMOUNTS_RECEIVED", "")
            balances = extracted_data.get("ACCOUNT_BALANCE", "")

            # Check for missing required fields
            if any(
                field == "" or field == "NOT_FOUND"
                for field in [descriptions, dates, paid]
            ):
                logger.warning("Missing transaction data - skipping debit filtering")
                return extracted_data

            # Check if balances are all NOT_FOUND (e.g., from DEBIT_CREDIT_DESCRIPTION strategy)
            # In this case, balances is something like "NOT_FOUND | NOT_FOUND | ..."
            balance_values = (
                [b.strip() for b in balances.split(" | ")] if balances else []
            )
            all_balances_missing = all(
                b == "NOT_FOUND" or b == "" for b in balance_values
            )

            if balances == "" or balances == "NOT_FOUND" or all_balances_missing:
                logger.warning("No balance data available - skipping debit filtering")
                return extracted_data

            # Split arrays
            desc_list = descriptions.split(" | ")
            date_list = dates.split(" | ")
            paid_list = paid.split(" | ")
            balance_list = balances.split(" | ")
            received_list = (
                received.split(" | ") if received and received != "NOT_FOUND" else None
            )

            # DEBUG: Show array lengths before DataFrame creation
            logger.debug(
                "Array lengths: desc=%d, date=%d, paid=%d, balance=%d",
                len(desc_list),
                len(date_list),
                len(paid_list),
                len(balance_list),
            )
            if received_list:
                logger.debug("received=%d", len(received_list))

            # Verify arrays have same length
            lengths = [
                len(desc_list),
                len(date_list),
                len(paid_list),
                len(balance_list),
            ]
            if len(set(lengths)) > 1:
                logger.warning(
                    "Array length mismatch: %s - skipping debit filtering", lengths
                )
                return extracted_data

            # Create DataFrame from transaction data
            transactions_df = pd.DataFrame(
                {
                    "description": desc_list,
                    "date": date_list,
                    "paid": paid_list,
                    "received": received_list,
                    "balance": balance_list,
                }
            )

            logger.debug("Pre-filter: %d transactions", len(transactions_df))

            # Filter to keep only debit transactions (where paid != 'NOT_FOUND')
            debit_df = transactions_df[transactions_df["paid"] != "NOT_FOUND"].copy()

            logger.debug(
                "Debit transactions found: %d/%d",
                len(debit_df),
                len(transactions_df),
            )

            # Convert back to pipe-separated strings
            filtered_data = extracted_data.copy()
            filtered_data["LINE_ITEM_DESCRIPTIONS"] = " | ".join(
                debit_df["description"].tolist()
            )
            filtered_data["TRANSACTION_DATES"] = " | ".join(debit_df["date"].tolist())
            filtered_data["TRANSACTION_AMOUNTS_PAID"] = " | ".join(
                debit_df["paid"].tolist()
            )
            filtered_data["TRANSACTION_AMOUNTS_RECEIVED"] = (
                "NOT_FOUND"  # No credits in debit-only
            )
            filtered_data["ACCOUNT_BALANCE"] = " | ".join(debit_df["balance"].tolist())

            logger.debug("Pandas filtered to %d debit transactions", len(debit_df))

            return filtered_data

        except Exception as e:
            logger.error("Pandas filtering failed: %s", e)
            logger.warning("Falling back to original data")
            return extracted_data

    def _process_image(self, image_path: str, verbose: bool) -> tuple[str, dict, str]:
        """
        Process single image using model handler.

        Routes bank statements to BankStatementAdapter when available,
        otherwise uses standard document-aware extraction for all types.

        Args:
            image_path: Path to image
            verbose: Whether to show verbose output

        Returns:
            Tuple of (document_type, extraction_result, prompt_name)
        """

        logger.debug("DOCUMENT TYPE DETECTION")

        # Step 1: Detect and classify document
        classification_info = self.model_handler.detect_and_classify_document(
            image_path, verbose=verbose
        )
        document_type = classification_info["document_type"]

        logger.debug("Detected Document Type: %s", document_type)

        # Step 2: Route bank statements to adapter when available
        if document_type.upper() == "BANK_STATEMENT" and self.bank_adapter is not None:
            logger.debug("BANK STATEMENT: Routing to BankStatementAdapter")

            try:
                schema_fields, metadata = self.bank_adapter.extract_bank_statement(
                    image_path
                )

                extraction_result = {
                    "extracted_data": schema_fields,
                    "raw_response": metadata.get("raw_responses", {}).get("turn1", ""),
                    "field_list": list(schema_fields.keys()),
                    "metadata": metadata,
                }

                strategy = metadata.get("strategy_used", "unknown")
                prompt_name = f"unified_bank_{strategy}"

                logger.debug("Strategy: %s", strategy)
                tx_count = (
                    len(schema_fields.get("TRANSACTION_DATES", "").split("|"))
                    if schema_fields.get("TRANSACTION_DATES") != "NOT_FOUND"
                    else 0
                )
                logger.debug("Transactions extracted: %d", tx_count)

                return document_type, extraction_result, prompt_name

            except Exception as e:
                logger.warning("BankStatementAdapter failed: %s", e)
                logger.warning("Falling back to standard extraction...")
                # Fall through to standard extraction

        # Step 3: Standard document-aware extraction (invoices, receipts, or bank fallback)
        logger.debug("DOCUMENT-AWARE EXTRACTION (%s)", document_type.upper())

        extraction_result = self.model_handler.process_document_aware(
            image_path, classification_info, verbose=verbose
        )

        # Extract the actual extracted_data for evaluation
        extracted_data = extraction_result.get("extracted_data", {})

        # Create extraction_result in the format expected by batch processor
        formatted_result = {
            "extracted_data": extracted_data,
            "document_type": document_type,
            "image_file": Path(image_path).name,
            "processing_time": extraction_result.get("processing_time", 0),
        }

        prompt_name = f"{document_type.lower()}"

        return document_type, formatted_result, prompt_name


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
