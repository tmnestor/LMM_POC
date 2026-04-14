"""
Batch Processing Module for Document-Aware Extraction

Handles batch processing of images through document detection and extraction pipeline.
Delegates evaluation to ExtractionEvaluator for clean separation of concerns.
"""

import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .batch_types import (
    BatchResult,
    BatchStats,
    DetectionResult,
    ExtractionOutput,
    ImageResult,
)
from .extraction_evaluator import ExtractionEvaluator
from .field_config import filter_evaluation_fields

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
            f"FATAL: Field definitions file not found\n"
            f"Expected location: {field_def_path.absolute()}\n"
            f"This file is REQUIRED for document-aware field filtering.\n"
            f"Ensure config/field_definitions.yaml exists in the project root."
        ) from None

    try:
        with field_def_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"FATAL: Invalid YAML syntax in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Error: {e}\n"
            f"Fix the YAML syntax errors before proceeding."
        ) from e
    except Exception as e:
        raise ValueError(
            f"FATAL: Could not read field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Error: {e}"
        ) from e

    # Validate structure
    if "document_fields" not in config:
        raise ValueError(
            f"FATAL: Missing 'document_fields' section in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Required structure:\n"
            f"document_fields:\n"
            f"  invoice:\n"
            f"    fields: [list of fields]\n"
            f"  receipt:\n"
            f"    fields: [list of fields]\n"
            f"  bank_statement:\n"
            f"    fields: [list of fields]"
        ) from None

    doc_fields = config["document_fields"]

    # Validate all document types defined in YAML (excluding 'universal')
    result = {}
    for doc_type, type_config in doc_fields.items():
        if "fields" not in type_config:
            raise ValueError(
                f"FATAL: Missing 'fields' list for '{doc_type}' in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must have a 'fields' list."
            ) from None
        if not type_config["fields"]:
            raise ValueError(
                f"FATAL: Empty 'fields' list for '{doc_type}' in field_definitions.yaml\n"
                f"File: {field_def_path.absolute()}\n"
                f"Each document type must have at least one field defined."
            ) from None
        result[doc_type] = filter_evaluation_fields(type_config["fields"])

    if not result:
        raise ValueError(
            f"FATAL: No document types defined in field_definitions.yaml\n"
            f"File: {field_def_path.absolute()}\n"
            f"Expected: document_fields section with at least one document type."
        ) from None

    return result


class BatchDocumentProcessor:
    """Orchestrates document detection, extraction, and evaluation.

    Detection and extraction are handled internally (batch or sequential based
    on model capability). Evaluation is delegated to an injected ExtractionEvaluator.
    """

    def __init__(
        self,
        model,
        evaluator: ExtractionEvaluator,
        *,
        bank_adapter=None,
        batch_size: int | None = None,
        console: Console | None = None,
    ):
        """Initialize batch processor.

        Args:
            model: Model handler implementing DocumentProcessor protocol.
            evaluator: Pre-configured evaluator for scoring extractions.
            bank_adapter: Optional BankStatementAdapter for multi-turn bank extraction.
            batch_size: Images per batch (None = auto-detect, 1 = sequential).
            console: Rich console for output.
        """
        self.model_handler = model
        self.evaluator = evaluator
        self.bank_adapter = bank_adapter
        self.batch_size = batch_size
        self.console = console or Console()

        # Batch stats -- populated after process_batch() completes
        self.batch_stats: dict[str, float] = {}

    def _resolve_batch_size(self) -> int:
        """Resolve effective batch size from config or auto-detection."""
        if self.batch_size is not None:
            return max(1, self.batch_size)
        return getattr(self.model_handler, "batch_size", 1)

    def process_batch(
        self,
        image_paths: list[str],
        verbose: bool = True,
        progress_interval: int = 5,
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """Process a batch of images through detect -> extract -> evaluate.

        Backward-compatible entry point returning the legacy 3-tuple.

        Args:
            image_paths: List of image file paths.
            verbose: Whether to show progress updates.
            progress_interval: How often to show detailed progress (unused, kept for compat).

        Returns:
            Tuple of (batch_results, processing_times, document_types_found)
        """
        result = self._run_pipeline(image_paths, verbose)
        self.batch_stats = result.stats.to_dict()
        return result.as_tuple()

    def run(
        self,
        image_paths: list[str],
        verbose: bool = True,
    ) -> BatchResult:
        """Process a batch of images, returning typed BatchResult.

        Args:
            image_paths: List of image file paths.
            verbose: Whether to show progress updates.

        Returns:
            BatchResult with typed per-image results, timing, and stats.
        """
        result = self._run_pipeline(image_paths, verbose)
        self.batch_stats = result.stats.to_dict()
        return result

    def _run_pipeline(
        self,
        image_paths: list[str],
        verbose: bool,
    ) -> BatchResult:
        """Unified pipeline: detect -> extract -> evaluate.

        Replaces the former _process_batch_two_phase and _process_batch_sequential
        with a single code path.
        """
        effective_batch_size = self._resolve_batch_size()

        logger.info("Starting Batch Processing")
        logger.info("Batch size: %d", effective_batch_size)
        self.console.rule("[bold green]Batch Extraction[/bold green]")

        # Phase 1: Detection
        detections, detection_batch_sizes = self._detect_all(
            image_paths, effective_batch_size, verbose
        )

        # Count document types
        document_types_found: dict[str, int] = {}
        for det in detections:
            document_types_found[det.document_type] = (
                document_types_found.get(det.document_type, 0) + 1
            )
        for doc_type, count in sorted(document_types_found.items()):
            logger.info("  %s: %d", doc_type, count)

        # Phase 2: Extraction
        extractions, extraction_batch_sizes = self._extract_all(
            image_paths, detections, effective_batch_size, verbose
        )

        self.console.rule("[bold green]Batch Processing Complete[/bold green]")

        # Phase 3: Evaluation (per-image, CPU-only)
        image_results: list[ImageResult] = []
        processing_times: list[float] = []

        for ext in extractions:
            evaluation = self.evaluator.evaluate(ext)

            extraction_result_dict = {
                "extracted_data": ext.extracted_data,
                "document_type": ext.document_type,
                "image_file": ext.image_name,
                "processing_time": ext.processing_time,
            }
            if ext.skip_math_enhancement:
                extraction_result_dict["skip_math_enhancement"] = True

            image_results.append(
                ImageResult(
                    image_name=ext.image_name,
                    image_path=ext.image_path,
                    document_type=ext.document_type,
                    extraction_result=extraction_result_dict,
                    evaluation=evaluation,
                    processing_time=ext.processing_time,
                    prompt_used=ext.prompt_used,
                    error=ext.error,
                )
            )
            processing_times.append(ext.processing_time)

            if evaluation and "median_f1" in evaluation:
                median_f1_pct = evaluation.get("median_f1", 0) * 100
                mean_f1_pct = evaluation.get("overall_accuracy", 0) * 100
                logger.debug(
                    "%s: Median %.1f%% | Mean %.1f%% | %.1fs",
                    ext.image_name,
                    median_f1_pct,
                    mean_f1_pct,
                    ext.processing_time,
                )

        # Compute batch stats
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
        stats = BatchStats(
            configured_batch_size=effective_batch_size,
            avg_detection_batch=avg_detect,
            avg_extraction_batch=avg_extract,
            num_detection_calls=len(detection_batch_sizes),
            num_extraction_calls=len(extraction_batch_sizes),
        )

        return BatchResult(
            results=image_results,
            processing_times=processing_times,
            document_types_found=document_types_found,
            stats=stats,
        )

    def _detect_all(
        self,
        image_paths: list[str],
        batch_size: int,
        verbose: bool,
    ) -> tuple[list[DetectionResult], list[int]]:
        """Detect document types for all images.

        Uses batch_detect_documents when model supports it, else sequential.

        Returns:
            Tuple of (detections, batch_sizes_used).
        """
        from models.protocol import BatchCapableProcessor

        total_images = len(image_paths)
        detection_batch_sizes: list[int] = []

        logger.info("Phase 1: Document Detection")
        detection_start = time.time()

        if batch_size > 1 and isinstance(self.model_handler, BatchCapableProcessor):
            # Batched detection
            all_classification_infos = []
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
        else:
            # Sequential detection
            all_classification_infos = []
            for image_path in image_paths:
                detection_batch_sizes.append(1)
                classification_info = self.model_handler.detect_and_classify_document(
                    image_path, verbose=verbose
                )
                all_classification_infos.append(classification_info)

        detection_time = time.time() - detection_start
        logger.info(
            "Detection complete: %.1fs for %d images", detection_time, total_images
        )

        # Convert to typed DetectionResult
        detections = [
            DetectionResult(
                image_path=image_paths[i],
                image_name=Path(image_paths[i]).name,
                document_type=info["document_type"],
                classification_info=dict(info),
            )
            for i, info in enumerate(all_classification_infos)
        ]

        return detections, detection_batch_sizes

    def _extract_all(
        self,
        image_paths: list[str],
        detections: list[DetectionResult],
        batch_size: int,
        verbose: bool,
    ) -> tuple[list[ExtractionOutput], list[int]]:
        """Extract fields from all images.

        Standard documents are batched when model supports it.
        Bank statements always use sequential multi-turn extraction.

        Returns:
            Tuple of (extractions in original order, batch_sizes_used).
        """
        from models.protocol import BatchCapableProcessor

        total_images = len(image_paths)
        extractions: list[ExtractionOutput | None] = [None] * total_images
        extraction_batch_sizes: list[int] = []

        logger.info("Phase 2: Document Extraction")
        extraction_start = time.time()

        # Set up progress bar
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

        # Partition into bank vs standard, preserving indices
        bank_indices = []
        standard_indices = []
        for i, det in enumerate(detections):
            if det.document_type.upper() == "BANK_STATEMENT":
                bank_indices.append(i)
            else:
                standard_indices.append(i)

        logger.debug(
            "Standard documents: %d, Bank statements: %d",
            len(standard_indices),
            len(bank_indices),
        )

        is_batch_capable = batch_size > 1 and isinstance(
            self.model_handler, BatchCapableProcessor
        )

        # --- Standard documents ---
        if standard_indices and is_batch_capable:
            # Batched extraction
            for batch_start_idx in range(0, len(standard_indices), batch_size):
                batch_end_idx = min(batch_start_idx + batch_size, len(standard_indices))
                batch_orig_indices = standard_indices[batch_start_idx:batch_end_idx]

                batch_paths = [image_paths[i] for i in batch_orig_indices]
                batch_class_infos = [
                    detections[i].classification_info for i in batch_orig_indices
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

                per_image_time = (
                    batch_extract_time / len(batch_orig_indices)
                    if batch_orig_indices
                    else 0
                )

                for j, orig_idx in enumerate(batch_orig_indices):
                    det = detections[orig_idx]
                    extraction_result = extraction_results[j]

                    extractions[orig_idx] = ExtractionOutput(
                        image_path=det.image_path,
                        image_name=det.image_name,
                        document_type=det.document_type,
                        extracted_data=extraction_result.get("extracted_data", {}),
                        processing_time=per_image_time,
                        prompt_used=f"batch_{det.document_type.lower()}",
                    )

                    progress.update(progress_task, advance=1, current=det.image_name)
                    self.console.print(progress.get_renderable())

        elif standard_indices:
            # Sequential extraction for standard docs
            for orig_idx in standard_indices:
                extraction_batch_sizes.append(1)
                det = detections[orig_idx]
                image_path = det.image_path

                # Print tile info
                if hasattr(self.model_handler, "image_preprocessor"):
                    tile_info = self.model_handler.image_preprocessor.get_tile_info(
                        image_path
                    )
                    self.console.print(f"[dim]{tile_info}[/dim]")

                img_start = time.time()
                try:
                    doc_type, extraction_result, prompt_name = self._process_image(
                        image_path, verbose
                    )
                    img_time = time.time() - img_start

                    extractions[orig_idx] = ExtractionOutput(
                        image_path=image_path,
                        image_name=det.image_name,
                        document_type=doc_type,
                        extracted_data=extraction_result.get("extracted_data", {}),
                        processing_time=img_time,
                        prompt_used=prompt_name,
                        skip_math_enhancement=extraction_result.get(
                            "skip_math_enhancement", False
                        ),
                    )
                except Exception as e:
                    logger.error("Error processing %s: %s", det.image_name, e)
                    img_time = time.time() - img_start
                    extractions[orig_idx] = ExtractionOutput(
                        image_path=image_path,
                        image_name=det.image_name,
                        document_type=det.document_type,
                        extracted_data={},
                        processing_time=img_time,
                        prompt_used="error",
                        error=str(e),
                    )

                progress.update(progress_task, advance=1, current=det.image_name)
                self.console.print(progress.get_renderable())

        # --- Bank statements: always sequential (multi-turn required) ---
        for orig_idx in bank_indices:
            extraction_batch_sizes.append(1)
            det = detections[orig_idx]
            image_path = det.image_path

            logger.info("BANK STATEMENT (sequential): %s", det.image_name)

            if hasattr(self.model_handler, "image_preprocessor"):
                tile_info = self.model_handler.image_preprocessor.get_tile_info(
                    image_path
                )
                self.console.print(f"[dim]{tile_info}[/dim]")

            bank_start = time.time()

            try:
                doc_type, extraction_result, prompt_name = self._process_image(
                    image_path, verbose
                )
                bank_time = time.time() - bank_start

                extractions[orig_idx] = ExtractionOutput(
                    image_path=image_path,
                    image_name=det.image_name,
                    document_type=doc_type,
                    extracted_data=extraction_result.get("extracted_data", {}),
                    processing_time=bank_time,
                    prompt_used=prompt_name,
                    skip_math_enhancement=extraction_result.get(
                        "skip_math_enhancement", False
                    ),
                )
            except Exception as e:
                logger.error(
                    "Error processing bank statement %s: %s", det.image_name, e
                )
                bank_time = time.time() - bank_start
                extractions[orig_idx] = ExtractionOutput(
                    image_path=image_path,
                    image_name=det.image_name,
                    document_type=det.document_type,
                    extracted_data={},
                    processing_time=bank_time,
                    prompt_used="error",
                    error=str(e),
                )

            progress.update(progress_task, advance=1, current=det.image_name)
            self.console.print(progress.get_renderable())

        extraction_time = time.time() - extraction_start
        logger.info("Extraction complete: %.1fs", extraction_time)

        # Print final progress
        progress.update(progress_task, current="done")
        self.console.print(progress.get_renderable())

        # Cast away None (all slots should be filled)
        return [e for e in extractions if e is not None], extraction_batch_sizes

    def _process_image(self, image_path: str, verbose: bool) -> tuple[str, dict, str]:
        """Process single image using model handler.

        Routes bank statements to BankStatementAdapter when available,
        otherwise uses standard document-aware extraction for all types.

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

        # Step 3: Standard document-aware extraction
        logger.debug("DOCUMENT-AWARE EXTRACTION (%s)", document_type.upper())

        extraction_result = self.model_handler.process_document_aware(
            image_path, classification_info, verbose=verbose
        )

        extracted_data = extraction_result.get("extracted_data", {})

        formatted_result = {
            "extracted_data": extracted_data,
            "document_type": document_type,
            "image_file": Path(image_path).name,
            "processing_time": extraction_result.get("processing_time", 0),
        }

        prompt_name = f"{document_type.lower()}"

        return document_type, formatted_result, prompt_name


def create_batch_pipeline(
    model,
    prompt_config: dict,
    ground_truth_csv: str | None,
    *,
    console: Console | None = None,
    enable_math_enhancement: bool = True,
    bank_adapter=None,
    field_definitions: dict[str, list[str]] | None = None,
    batch_size: int | None = None,
) -> BatchDocumentProcessor:
    """Create a BatchDocumentProcessor with the same parameters as the old constructor.

    This factory builds the ExtractionEvaluator internally, keeping the call site
    in cli.py minimal.

    Args:
        model: Model handler implementing DocumentProcessor protocol.
        prompt_config: Dictionary with prompt file paths (unused, kept for compat).
        ground_truth_csv: Path to ground truth CSV file. None = inference-only.
        console: Rich console for output.
        enable_math_enhancement: Whether to apply math enhancement for bank statements.
        bank_adapter: Optional BankStatementAdapter for multi-turn bank extraction.
        field_definitions: Pre-loaded field definitions. If None, loads from YAML.
        batch_size: Images per batch (None = auto-detect, 1 = sequential).

    Returns:
        Configured BatchDocumentProcessor ready for process_batch().
    """
    doc_type_fields = field_definitions or load_document_field_definitions()

    evaluator = ExtractionEvaluator(
        ground_truth_csv=ground_truth_csv,
        field_definitions=doc_type_fields,
        enable_math_enhancement=enable_math_enhancement,
    )

    return BatchDocumentProcessor(
        model=model,
        evaluator=evaluator,
        bank_adapter=bank_adapter,
        batch_size=batch_size,
        console=console,
    )


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
