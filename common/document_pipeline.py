"""Unified document extraction pipeline.

Replaces BatchDocumentProcessor with a single routing authority for
batch-vs-sequential and bank-vs-standard decisions. All routing
logic lives in _extract_all() -- no duplicated isinstance checks.

Closes #3.
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
from .field_schema import get_field_schema

logger = logging.getLogger(__name__)


def _release_gpu_memory() -> None:
    """Release fragmented GPU memory between inference calls.

    Wraps gpu_memory.release_memory with a guard for CPU-only environments.
    """
    try:
        from .gpu_memory import release_memory

        release_memory(threshold_gb=1.0)
    except ImportError:
        pass


def _log_gpu_memory(phase: str) -> None:
    """Log per-GPU allocated/reserved memory at a pipeline phase boundary.

    Used to diagnose whether GPU memory grows across phase boundaries
    (monolithic run_batch_inference with raw_response persistence) vs.
    stays bounded (streaming / staged pipeline).

    No-op on CPU-only environments.
    """
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    gib = 1024**3
    parts: list[str] = []
    for idx in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(idx) / gib
        reserved = torch.cuda.memory_reserved(idx) / gib
        peak = torch.cuda.max_memory_allocated(idx) / gib
        parts.append(
            f"cuda:{idx} alloc={alloc:.2f}GiB "
            f"reserved={reserved:.2f}GiB peak={peak:.2f}GiB"
        )
    logger.info("[gpu-mem][%s] %s", phase, " | ".join(parts))


class DocumentPipeline:
    """Single owner of the detect -> extract -> evaluate pipeline.

    Absorbs routing logic from both BatchDocumentProcessor and
    DocumentOrchestrator's batch methods. The orchestrator retains
    single-image detection/extraction/generation; this class owns
    the multi-image flow.

    All routing decisions happen in one place:
    - Batch vs sequential: checked once at construction via
      ``orchestrator.supports_batch``, stored as ``_can_batch``.
    - Bank vs standard: partitioned in ``_extract_all()``.
    """

    def __init__(
        self,
        orchestrator,
        evaluator: ExtractionEvaluator,
        *,
        bank_adapter=None,
        batch_size: int | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            orchestrator: DocumentOrchestrator instance.
            evaluator: Pre-configured evaluator for scoring extractions.
            bank_adapter: Optional UnifiedBankExtractor for multi-turn bank extraction.
            batch_size: Images per batch (None = auto-detect, 1 = sequential).
            console: Rich console for output.
        """
        self._orchestrator = orchestrator
        self._evaluator = evaluator
        self._bank_adapter = bank_adapter
        self._console = console or Console()

        # Resolve batch capability once -- no isinstance checks later
        self._can_batch: bool = getattr(orchestrator, "supports_batch", False)
        self._batch_size = self._resolve_batch_size(batch_size)

        # Stats from most recent run
        self.batch_stats: dict[str, float] = {}

    def _resolve_batch_size(self, batch_size: int | None) -> int:
        """Resolve effective batch size."""
        if batch_size is not None:
            return max(1, batch_size)
        return getattr(self._orchestrator, "batch_size", 1)

    # -- Public API -----------------------------------------------------------

    def process_batch(
        self,
        image_paths: list[str],
        verbose: bool = True,
        progress_interval: int = 5,
    ) -> tuple[list[dict], list[float], dict[str, int]]:
        """Backward-compatible 3-tuple entry point.

        Args:
            image_paths: List of image file paths.
            verbose: Whether to show progress updates.
            progress_interval: Unused, kept for backward compatibility.

        Returns:
            Tuple of (batch_results, processing_times, document_types_found).
        """
        result = self._run_pipeline(image_paths, verbose)
        self.batch_stats = result.stats.to_dict()
        return result.as_tuple()

    def run(
        self,
        image_paths: list[str],
        verbose: bool = True,
    ) -> BatchResult:
        """Process images through detect -> extract -> evaluate.

        Returns typed BatchResult.
        """
        result = self._run_pipeline(image_paths, verbose)
        self.batch_stats = result.stats.to_dict()
        return result

    # -- Pipeline phases ------------------------------------------------------

    def _run_pipeline(
        self,
        image_paths: list[str],
        verbose: bool,
    ) -> BatchResult:
        """Unified pipeline: detect -> extract -> evaluate."""
        logger.info("Starting Batch Processing")
        logger.info("Batch size: %d", self._batch_size)
        self._console.rule("[bold green]Batch Extraction[/bold green]")

        _log_gpu_memory("pipeline-start")

        # Phase 1: Detection
        detections, detection_batch_sizes = self._detect_all(image_paths, verbose)

        _log_gpu_memory("post-detect")

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
            image_paths, detections, verbose
        )

        _log_gpu_memory("post-extract")

        self._console.rule("[bold green]Batch Processing Complete[/bold green]")

        # Phase 3: Evaluation (per-image, CPU-only)
        image_results: list[ImageResult] = []
        processing_times: list[float] = []

        for ext in extractions:
            evaluation = self._evaluator.evaluate(ext)

            extraction_result_dict = {
                "extracted_data": ext.extracted_data,
                "document_type": ext.document_type,
                "image_file": ext.image_name,
                "processing_time": ext.processing_time,
            }

            image_results.append(
                ImageResult(
                    image_name=ext.image_name,
                    image_path=ext.image_path,
                    document_type=ext.document_type,
                    extraction_result=extraction_result_dict,
                    evaluation=evaluation,
                    processing_time=ext.processing_time,
                    prompt_used=ext.prompt_used,
                    raw_response=ext.raw_response,
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
            configured_batch_size=self._batch_size,
            avg_detection_batch=avg_detect,
            avg_extraction_batch=avg_extract,
            num_detection_calls=len(detection_batch_sizes),
            num_extraction_calls=len(extraction_batch_sizes),
        )

        _log_gpu_memory("pipeline-end")

        return BatchResult(
            results=image_results,
            processing_times=processing_times,
            document_types_found=document_types_found,
            stats=stats,
        )

    def _detect_all(
        self,
        image_paths: list[str],
        verbose: bool,
    ) -> tuple[list[DetectionResult], list[int]]:
        """Detect document types for all images.

        Uses batch detection when orchestrator supports it, else sequential.
        """
        total_images = len(image_paths)
        detection_batch_sizes: list[int] = []

        logger.info("Phase 1: Document Detection")
        detection_start = time.time()

        if self._batch_size > 1 and self._can_batch:
            # Batched detection
            all_classification_infos = []
            for batch_start in range(0, total_images, self._batch_size):
                batch_end = min(batch_start + self._batch_size, total_images)
                batch_paths = image_paths[batch_start:batch_end]
                detection_batch_sizes.append(len(batch_paths))

                logger.debug(
                    "Detecting batch [%d-%d] / %d",
                    batch_start + 1,
                    batch_end,
                    total_images,
                )

                batch_classifications = self._orchestrator.detect_batch(
                    batch_paths, verbose=verbose
                )
                all_classification_infos.extend(batch_classifications)
        else:
            # Sequential detection
            all_classification_infos = []
            for image_path in image_paths:
                detection_batch_sizes.append(1)
                classification_info = self._orchestrator.detect_and_classify_document(
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
        verbose: bool,
    ) -> tuple[list[ExtractionOutput], list[int]]:
        """Extract fields from all images.

        Single routing authority for both decisions:
        - Bank vs standard: partition by document_type
        - Batch vs sequential: use _can_batch flag (set once at construction)
        """
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
            console=self._console,
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

        # --- Standard documents ---
        if standard_indices and self._batch_size > 1 and self._can_batch:
            # Batched extraction
            for batch_start_idx in range(0, len(standard_indices), self._batch_size):
                batch_end_idx = min(
                    batch_start_idx + self._batch_size, len(standard_indices)
                )
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
                extraction_results = self._orchestrator.extract_batch(
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
                        raw_response=extraction_result.get("raw_response", ""),
                    )

                    progress.update(progress_task, advance=1, current=det.image_name)
                    self._console.print(progress.get_renderable())

                # Free batch activations before next batch or bank phase
                _release_gpu_memory()

        elif standard_indices:
            # Sequential extraction for standard docs
            for orig_idx in standard_indices:
                extraction_batch_sizes.append(1)
                det = detections[orig_idx]

                self._print_tile_info(det.image_path)

                img_start = time.time()
                try:
                    extraction_result = self._orchestrator.process_document_aware(
                        det.image_path,
                        det.classification_info,
                        verbose=verbose,
                    )
                    img_time = time.time() - img_start

                    extractions[orig_idx] = ExtractionOutput(
                        image_path=det.image_path,
                        image_name=det.image_name,
                        document_type=det.document_type,
                        extracted_data=extraction_result.get("extracted_data", {}),
                        processing_time=img_time,
                        prompt_used=det.document_type.lower(),
                        raw_response=extraction_result.get("raw_response", ""),
                    )
                except Exception as e:
                    logger.error("Error processing %s: %s", det.image_name, e)
                    img_time = time.time() - img_start
                    extractions[orig_idx] = ExtractionOutput(
                        image_path=det.image_path,
                        image_name=det.image_name,
                        document_type=det.document_type,
                        extracted_data={},
                        processing_time=img_time,
                        prompt_used="error",
                        error=str(e),
                    )

                progress.update(progress_task, advance=1, current=det.image_name)
                self._console.print(progress.get_renderable())

        # --- Bank statements: always sequential ---
        for orig_idx in bank_indices:
            extraction_batch_sizes.append(1)
            det = detections[orig_idx]

            # GPU cleanup before each bank image (was handled by
            # process_single_image pre-refactor, now we must do it here)
            _release_gpu_memory()

            logger.info("BANK STATEMENT (sequential): %s", det.image_name)
            self._print_tile_info(det.image_path)

            bank_start = time.time()
            try:
                ext_output = self._extract_bank(det, verbose)
                bank_time = time.time() - bank_start

                extractions[orig_idx] = ExtractionOutput(
                    image_path=det.image_path,
                    image_name=det.image_name,
                    document_type=ext_output.document_type,
                    extracted_data=ext_output.extracted_data,
                    processing_time=bank_time,
                    prompt_used=ext_output.prompt_used,
                    raw_response=ext_output.raw_response,
                )
            except Exception as e:
                logger.error(
                    "Error processing bank statement %s: %s",
                    det.image_name,
                    e,
                )
                bank_time = time.time() - bank_start
                extractions[orig_idx] = ExtractionOutput(
                    image_path=det.image_path,
                    image_name=det.image_name,
                    document_type=det.document_type,
                    extracted_data={},
                    processing_time=bank_time,
                    prompt_used="error",
                    error=str(e),
                )

            progress.update(progress_task, advance=1, current=det.image_name)
            self._console.print(progress.get_renderable())

        extraction_time = time.time() - extraction_start
        logger.info("Extraction complete: %.1fs", extraction_time)

        # Print final progress
        progress.update(progress_task, current="done")
        self._console.print(progress.get_renderable())

        # Cast away None (all slots should be filled)
        return [e for e in extractions if e is not None], extraction_batch_sizes

    # -- Bank extraction (no re-detection) ------------------------------------

    def _extract_bank(
        self, detection: DetectionResult, verbose: bool
    ) -> ExtractionOutput:
        """Extract bank statement fields.

        Uses bank_adapter when available, falls back to orchestrator's
        standard process_document_aware (which does vision-based structure
        classification internally).

        Unlike the old _process_image(), detection has already happened --
        no redundant re-detection.
        """
        if self._bank_adapter is not None:
            logger.debug("BANK STATEMENT: Routing to UnifiedBankExtractor")
            bank_failed = False
            try:
                schema_fields, metadata = self._bank_adapter.extract_bank_statement(
                    detection.image_path
                )

                strategy = metadata.get("strategy_used", "unknown")
                prompt_name = f"unified_bank_{strategy}"

                logger.debug("Strategy: %s", strategy)
                tx_count = (
                    len(schema_fields.get("TRANSACTION_DATES", "").split("|"))
                    if schema_fields.get("TRANSACTION_DATES") != "NOT_FOUND"
                    else 0
                )
                logger.debug("Transactions extracted: %d", tx_count)

                return ExtractionOutput(
                    image_path=detection.image_path,
                    image_name=detection.image_name,
                    document_type=detection.document_type,
                    extracted_data=schema_fields,
                    processing_time=0,  # caller measures wall time
                    prompt_used=prompt_name,
                )
            except Exception as e:
                logger.warning("UnifiedBankExtractor failed: %s", e)
                logger.warning("Falling back to standard extraction...")
                bank_failed = True

            # GPU cleanup OUTSIDE except block -- traceback refs are
            # released so gc.collect() + empty_cache() can actually
            # reclaim the tensors from the failed forward pass.
            if bank_failed:
                _release_gpu_memory()

        # Fallback: standard extraction (orchestrator handles bank structure
        # classification internally in process_document_aware)
        extraction_result = self._orchestrator.process_document_aware(
            detection.image_path,
            detection.classification_info,
            verbose=verbose,
        )

        return ExtractionOutput(
            image_path=detection.image_path,
            image_name=detection.image_name,
            document_type=detection.document_type,
            extracted_data=extraction_result.get("extracted_data", {}),
            processing_time=0,  # caller measures wall time
            prompt_used=detection.document_type.lower(),
            raw_response=extraction_result.get("raw_response", ""),
        )

    # -- Helpers --------------------------------------------------------------

    def _print_tile_info(self, image_path: str) -> None:
        """Print tile info if the orchestrator has an image_preprocessor."""
        if hasattr(self._orchestrator, "image_preprocessor"):
            tile_info = self._orchestrator.image_preprocessor.get_tile_info(image_path)
            self._console.print(f"[dim]{tile_info}[/dim]")


def create_document_pipeline(
    orchestrator,
    *,
    ground_truth_csv: str | None,
    bank_adapter=None,
    field_definitions: dict[str, list[str]] | None = None,
    batch_size: int | None = None,
    enable_math_enhancement: bool = False,
    console: Console | None = None,
) -> DocumentPipeline:
    """Build a fully configured pipeline.

    Constructs ExtractionEvaluator internally from ground_truth_csv
    and field_definitions. The caller never touches the evaluator.

    Args:
        orchestrator: DocumentOrchestrator instance.
        ground_truth_csv: Path to ground truth CSV. None = inference-only.
        bank_adapter: Optional UnifiedBankExtractor.
        field_definitions: Pre-loaded field definitions. If None, loads from YAML.
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        enable_math_enhancement: Apply bank statement balance calculations.
        console: Rich console for output.

    Returns:
        Configured DocumentPipeline ready for process_batch() or run().
    """
    doc_type_fields = field_definitions or get_field_schema().get_all_doc_type_fields()

    evaluator = ExtractionEvaluator(
        ground_truth_csv=ground_truth_csv,
        field_definitions=doc_type_fields,
        enable_math_enhancement=enable_math_enhancement,
    )

    return DocumentPipeline(
        orchestrator,
        evaluator,
        bank_adapter=bank_adapter,
        batch_size=batch_size,
        console=console,
    )
