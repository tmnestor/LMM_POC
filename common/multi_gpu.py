"""Multi-GPU parallel processing orchestrator.

Partitions images across N GPUs, loads an independent model per GPU,
processes image subsets in parallel using ThreadPoolExecutor, and merges
results in original order.

ThreadPoolExecutor is used (not multiprocessing) because PyTorch releases
the GIL during CUDA kernel execution, giving true GPU parallelism without
serialization overhead.

Pipeline integration:
    Each GPU worker runs classify_images() + extract_documents() in parallel.
    After merge, evaluate_extractions() runs once (CPU-only).
"""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console

from pipeline.io_schemas import (
    ClassificationOutput,
    EvaluationOutput,
    ExtractionOutput,
)

console = Console()

# Serialize model loading to avoid transformers lazy-import race conditions.
# Processing (GPU inference) runs fully parallel after loading.
_model_load_lock = threading.Lock()


class MultiGPUOrchestrator:
    """Orchestrate parallel document processing across multiple GPUs.

    Three-phase approach:
      1. Load models sequentially (avoids import race + shows unified GPU status)
      2. Classify + extract image chunks in parallel (GIL released during CUDA)
      3. Evaluate once on merged results (CPU-only, no model needed)
    """

    def __init__(self, config, num_gpus: int) -> None:
        self.config = config
        self.num_gpus = num_gpus

    def _load_gpu_stacks(
        self,
        num_chunks: int,
        prompt_config: dict[str, Any],
        universal_fields: list[str],
        field_definitions: dict[str, list[str]],
    ) -> list[tuple]:
        """Load one model per GPU. Returns list of (gpu_config, model_ctx, processor)."""
        from cli import create_processor, load_model
        from models.registry import _print_gpu_status

        console.print("\n[bold]Loading models on all GPUs...[/bold]")
        gpu_stacks: list[tuple] = []
        for gpu_id in range(num_chunks):
            gpu_config = replace(self.config, device_map=f"cuda:{gpu_id}")
            gpu_config._multi_gpu = True  # suppress per-loader GPU status

            model_ctx = load_model(gpu_config)
            model, tokenizer = model_ctx.__enter__()
            processor = create_processor(
                model,
                tokenizer,
                gpu_config,
                prompt_config,
                universal_fields,
                field_definitions,
            )
            gpu_stacks.append((gpu_config, model_ctx, processor))

        _print_gpu_status(console)
        return gpu_stacks

    @staticmethod
    def _cleanup_gpu_stacks(gpu_stacks: list[tuple]) -> None:
        """Exit all model context managers to free GPU memory."""
        for _gpu_config, model_ctx, _processor in gpu_stacks:
            model_ctx.__exit__(None, None, None)

    def run(
        self,
        images: list[Path],
        prompt_config: dict[str, Any],
        universal_fields: list[str],
        field_definitions: dict[str, list[str]],
    ) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
        """Process images in parallel across GPUs.

        Returns merged (batch_results, processing_times, document_types_found, batch_stats).
        """
        chunks = self._partition_images(images)
        actual_gpus = len(chunks)

        console.print(
            f"\n[bold cyan]Multi-GPU: distributing {len(images)} images "
            f"across {actual_gpus} GPUs[/bold cyan]"
        )
        for gpu_id, chunk in enumerate(chunks):
            console.print(
                f"  [dim]GPU {gpu_id}: {len(chunk)} images "
                f"({chunk[0].name} .. {chunk[-1].name})[/dim]"
            )

        # Phase 1: Load all models sequentially
        gpu_stacks = self._load_gpu_stacks(
            actual_gpus, prompt_config, universal_fields, field_definitions
        )

        # Phase 2: Classify + extract image chunks in parallel
        console.print("\n[bold]Processing images in parallel...[/bold]")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
            futures = {
                executor.submit(
                    self._process_chunk,
                    gpu_stacks[gpu_id],
                    chunks[gpu_id],
                ): gpu_id
                for gpu_id in range(actual_gpus)
            }

            gpu_results: list[tuple | None] = [None] * actual_gpus
            for future in as_completed(futures):
                gpu_id = futures[future]
                gpu_results[gpu_id] = future.result()
                console.print(f"  [green]GPU {gpu_id} finished[/green]")

        # Clean up: exit all model context managers (free GPU memory)
        self._cleanup_gpu_stacks(gpu_stacks)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Multi-GPU processing complete: "
            f"{elapsed:.1f}s total[/bold green]"
        )

        # Phase 3: Merge results + evaluate once (CPU-only)
        merged_classifications, merged_extractions = self._merge_stage_results(
            gpu_results
        )

        evaluation_output: EvaluationOutput | None = None
        if self.config.ground_truth:
            from pipeline.evaluate import evaluate_extractions

            console.print("\n[bold]Evaluating merged results (CPU)...[/bold]")
            evaluation_output = evaluate_extractions(
                extractions=merged_extractions,
                ground_truth_csv=Path(str(self.config.ground_truth)),
                field_definitions=field_definitions,
                evaluation_method=os.environ.get("EVALUATION_METHOD", "order_aware_f1"),
                enable_math_enhancement=False,
                verbose=self.config.verbose,
            )

        # Compute batch stats
        batch_size = self.config.batch_size or 1
        batch_stats: dict[str, float] = {
            "avg_detection_batch": float(batch_size),
            "avg_extraction_batch": float(batch_size),
            "num_detection_calls": max(1, len(images) // max(1, batch_size)),
            "num_extraction_calls": max(1, len(images) // max(1, batch_size)),
            "configured_batch_size": batch_size,
        }

        return self._to_legacy_format(
            merged_classifications,
            merged_extractions,
            evaluation_output,
            batch_stats,
        )

    def run_classify(
        self,
        images: list[Path],
        prompt_config: dict[str, Any],
        universal_fields: list[str],
        field_definitions: dict[str, list[str]],
    ) -> ClassificationOutput:
        """Classify images in parallel across GPUs. Returns merged ClassificationOutput."""
        chunks = self._partition_images(images)
        actual_gpus = len(chunks)

        console.print(
            f"\n[bold cyan]Multi-GPU classify: distributing {len(images)} images "
            f"across {actual_gpus} GPUs[/bold cyan]"
        )
        for gpu_id, chunk in enumerate(chunks):
            console.print(
                f"  [dim]GPU {gpu_id}: {len(chunk)} images "
                f"({chunk[0].name} .. {chunk[-1].name})[/dim]"
            )

        gpu_stacks = self._load_gpu_stacks(
            actual_gpus, prompt_config, universal_fields, field_definitions
        )

        console.print("\n[bold]Classifying images in parallel...[/bold]")
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
                futures = {
                    executor.submit(
                        self._classify_chunk,
                        gpu_stacks[gpu_id],
                        chunks[gpu_id],
                    ): gpu_id
                    for gpu_id in range(actual_gpus)
                }

                gpu_results: list[ClassificationOutput | None] = [None] * actual_gpus
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    gpu_results[gpu_id] = future.result()
                    console.print(f"  [green]GPU {gpu_id} finished[/green]")
        finally:
            self._cleanup_gpu_stacks(gpu_stacks)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Multi-GPU classify complete: {elapsed:.1f}s total[/bold green]"
        )

        return self._merge_classifications(gpu_results)

    def run_extract(
        self,
        classification_output: ClassificationOutput,
        prompt_config: dict[str, Any],
        universal_fields: list[str],
        field_definitions: dict[str, list[str]],
    ) -> ExtractionOutput:
        """Extract documents in parallel across GPUs. Returns merged ExtractionOutput."""
        sub_classifications = self._partition_classifications(classification_output)
        actual_gpus = len(sub_classifications)

        console.print(
            f"\n[bold cyan]Multi-GPU extract: distributing "
            f"{len(classification_output.rows)} documents "
            f"across {actual_gpus} GPUs[/bold cyan]"
        )
        for gpu_id, sub in enumerate(sub_classifications):
            console.print(f"  [dim]GPU {gpu_id}: {len(sub.rows)} documents[/dim]")

        gpu_stacks = self._load_gpu_stacks(
            actual_gpus, prompt_config, universal_fields, field_definitions
        )

        console.print("\n[bold]Extracting documents in parallel...[/bold]")
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
                futures = {
                    executor.submit(
                        self._extract_chunk,
                        gpu_stacks[gpu_id],
                        sub_classifications[gpu_id],
                    ): gpu_id
                    for gpu_id in range(actual_gpus)
                }

                gpu_results: list[ExtractionOutput | None] = [None] * actual_gpus
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    gpu_results[gpu_id] = future.result()
                    console.print(f"  [green]GPU {gpu_id} finished[/green]")
        finally:
            self._cleanup_gpu_stacks(gpu_stacks)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Multi-GPU extract complete: {elapsed:.1f}s total[/bold green]"
        )

        return self._merge_extractions(gpu_results)

    @staticmethod
    def _classify_chunk(
        gpu_stack: tuple,
        images: list[Path],
    ) -> ClassificationOutput:
        """Classify an image chunk on a single GPU."""
        from pipeline.classify import classify_images

        gpu_config, _model_ctx, processor = gpu_stack
        return classify_images(
            processor=processor,
            image_paths=images,
            batch_size=gpu_config.batch_size or 1,
            verbose=gpu_config.verbose,
        )

    @staticmethod
    def _extract_chunk(
        gpu_stack: tuple,
        classifications_chunk: ClassificationOutput,
    ) -> ExtractionOutput:
        """Extract documents from a classification chunk on a single GPU."""
        from common.bank_statement_adapter import BankStatementAdapter
        from pipeline.extract import extract_documents

        gpu_config, _model_ctx, processor = gpu_stack

        bank_adapter = None
        if gpu_config.bank_v2 and getattr(processor, "supports_multi_turn", True):
            bank_adapter = BankStatementAdapter(
                generate_fn=processor.generate,
                verbose=gpu_config.verbose,
                use_balance_correction=gpu_config.balance_correction,
            )

        return extract_documents(
            processor=processor,
            classifications=classifications_chunk,
            bank_adapter=bank_adapter,
            batch_size=gpu_config.batch_size or 1,
            verbose=gpu_config.verbose,
        )

    @staticmethod
    def _process_chunk(
        gpu_stack: tuple,
        images: list[Path],
    ) -> tuple[ClassificationOutput, ExtractionOutput]:
        """Process an image chunk: classify + extract (GPU-bound).

        Evaluation is deferred to the orchestrator (CPU-only, runs once).
        """
        from common.bank_statement_adapter import BankStatementAdapter
        from pipeline.classify import classify_images
        from pipeline.extract import extract_documents

        gpu_config, _model_ctx, processor = gpu_stack

        classification_output = classify_images(
            processor=processor,
            image_paths=images,
            batch_size=gpu_config.batch_size or 1,
            verbose=gpu_config.verbose,
        )

        # Create bank adapter when V2 bank extraction is enabled
        bank_adapter = None
        if gpu_config.bank_v2 and getattr(processor, "supports_multi_turn", True):
            bank_adapter = BankStatementAdapter(
                generate_fn=processor.generate,
                verbose=gpu_config.verbose,
                use_balance_correction=gpu_config.balance_correction,
            )

        extraction_output = extract_documents(
            processor=processor,
            classifications=classification_output,
            bank_adapter=bank_adapter,
            batch_size=gpu_config.batch_size or 1,
            verbose=gpu_config.verbose,
        )

        return classification_output, extraction_output

    def _partition_images(self, images: list[Path]) -> list[list[Path]]:
        """Split images into num_gpus contiguous chunks.

        If there are fewer images than GPUs, returns fewer chunks
        (one image per chunk).
        """
        n = min(self.num_gpus, len(images))
        chunk_size = math.ceil(len(images) / n)
        return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]

    def _partition_classifications(
        self, classification_output: ClassificationOutput
    ) -> list[ClassificationOutput]:
        """Split classification rows into num_gpus contiguous chunks."""
        n = min(self.num_gpus, len(classification_output.rows))
        chunk_size = math.ceil(len(classification_output.rows) / n)
        return [
            ClassificationOutput(
                rows=classification_output.rows[i : i + chunk_size],
                model_type=classification_output.model_type,
                timestamp=classification_output.timestamp,
            )
            for i in range(0, len(classification_output.rows), chunk_size)
        ]

    @staticmethod
    def _merge_classifications(
        gpu_results: list[ClassificationOutput | None],
    ) -> ClassificationOutput:
        """Merge classification results from all GPUs in order."""
        all_rows = []
        model_type = "unknown"
        timestamp = ""

        for result in gpu_results:
            if result is None:
                continue
            all_rows.extend(result.rows)
            if result.model_type != "unknown":
                model_type = result.model_type
            if result.timestamp:
                timestamp = result.timestamp

        return ClassificationOutput(
            rows=all_rows,
            model_type=model_type,
            timestamp=timestamp,
        )

    @staticmethod
    def _merge_extractions(
        gpu_results: list[ExtractionOutput | None],
    ) -> ExtractionOutput:
        """Merge extraction results from all GPUs in order."""
        all_records = []
        metadata: dict[str, Any] = {}

        for result in gpu_results:
            if result is None:
                continue
            all_records.extend(result.records)
            if result.metadata:
                metadata.update(result.metadata)

        return ExtractionOutput(records=all_records, metadata=metadata)

    @staticmethod
    def _merge_stage_results(
        gpu_results: list[tuple[ClassificationOutput, ExtractionOutput] | None],
    ) -> tuple[ClassificationOutput, ExtractionOutput]:
        """Merge classification + extraction results from all GPUs.

        Concatenates rows/records in GPU order (which preserves the original
        image order since chunks are contiguous).
        """
        all_classification_rows = []
        all_extraction_records = []
        model_type = "unknown"
        timestamp = ""
        metadata: dict[str, Any] = {}

        for result in gpu_results:
            if result is None:
                continue
            classifications, extractions = result
            all_classification_rows.extend(classifications.rows)
            all_extraction_records.extend(extractions.records)
            if classifications.model_type != "unknown":
                model_type = classifications.model_type
            if classifications.timestamp:
                timestamp = classifications.timestamp
            if extractions.metadata:
                metadata.update(extractions.metadata)

        merged_classifications = ClassificationOutput(
            rows=all_classification_rows,
            model_type=model_type,
            timestamp=timestamp,
        )
        merged_extractions = ExtractionOutput(
            records=all_extraction_records,
            metadata=metadata,
        )
        return merged_classifications, merged_extractions

    @staticmethod
    def _to_legacy_format(
        classifications: ClassificationOutput,
        extractions: ExtractionOutput,
        evaluation_output: EvaluationOutput | None,
        batch_stats: dict[str, float],
    ) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
        """Convert pipeline stage outputs to legacy return format.

        Preserves backward compatibility with analytics, reporting, and
        visualization modules that expect (batch_results, processing_times,
        document_types_found, batch_stats).
        """
        batch_results: list[dict] = []
        processing_times: list[float] = []
        document_types_found: dict[str, int] = {}

        # Build evaluation lookup by image_name
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

        return batch_results, processing_times, document_types_found, batch_stats
