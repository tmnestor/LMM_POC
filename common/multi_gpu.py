"""Multi-GPU dynamic dispatch orchestrator.

Loads an independent model per GPU, then processes images via a shared
work queue — each GPU pulls the next image when it finishes the current one.
Results are stored in a pre-allocated array indexed by original image order.

ThreadPoolExecutor is used (not multiprocessing) because PyTorch releases
the GIL during CUDA kernel execution, giving true GPU parallelism without
serialization overhead.
"""

import queue
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Serialize model loading to avoid transformers lazy-import race conditions.
# Processing (GPU inference) runs fully parallel after loading.
_model_load_lock = threading.Lock()


class MultiGPUOrchestrator:
    """Orchestrate parallel document processing across multiple GPUs.

    Two-phase approach:
      1. Load models sequentially (avoids import race + shows unified GPU status)
      2. Workers pull images from a shared queue (optimal load balancing)
    """

    def __init__(self, config, num_gpus: int) -> None:
        self.config = config
        self.num_gpus = num_gpus

    def run(
        self,
        images: list[Path],
        prompt_config: dict[str, Any],
        universal_fields: list[str],
        field_definitions: dict[str, list[str]],
    ) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
        """Process images in parallel across GPUs using dynamic dispatch.

        Returns (batch_results, processing_times, document_types_found, batch_stats).
        """
        import torch

        from cli import create_processor, load_model
        from models.registry import _print_gpu_status

        actual_gpus = min(self.num_gpus, len(images))

        console.print(
            f"\n[bold cyan]Multi-GPU: {len(images)} images, "
            f"{actual_gpus} GPUs (dynamic dispatch)[/bold cyan]"
        )

        # Phase 1: Load models on all GPUs sequentially
        console.print("\n[bold]Loading models on all GPUs...[/bold]")
        gpu_stacks: list[tuple] = []
        for gpu_id in range(actual_gpus):
            gpu_config = replace(self.config, device_map=f"cuda:{gpu_id}")
            gpu_config._multi_gpu = True
            torch.cuda.set_device(gpu_id)
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

        # Phase 2: Dynamic dispatch via shared work queue
        work_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
        for idx, img in enumerate(images):
            work_queue.put((idx, img))

        # Pre-allocate result slots (one per image, indexed by original order)
        total_images = len(images)
        results: list[dict | None] = [None] * total_images
        timings: list[float] = [0.0] * total_images
        doc_types_found: dict[str, int] = {}
        doc_types_lock = threading.Lock()
        gpu_counts: list[int] = [0] * actual_gpus
        completed = [0]  # mutable counter for thread-safe increment

        # Create per-GPU bank adapters
        bank_adapters = [
            self._create_bank_adapter(processor) for _, _, processor in gpu_stacks
        ]

        # Ground truth — loaded once, shared read-only across workers
        ground_truth_data = self._load_ground_truth()

        console.print("\n[bold]Processing images (dynamic dispatch)...[/bold]")
        start_time = time.time()

        threads: list[threading.Thread] = []
        for gpu_id in range(actual_gpus):
            _, _, processor = gpu_stacks[gpu_id]
            t = threading.Thread(
                target=self._worker_loop,
                args=(
                    gpu_id,
                    processor,
                    bank_adapters[gpu_id],
                    work_queue,
                    results,
                    timings,
                    doc_types_found,
                    doc_types_lock,
                    gpu_counts,
                    completed,
                    total_images,
                    field_definitions,
                    ground_truth_data,
                ),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        elapsed = time.time() - start_time

        # Print per-GPU summary
        for gpu_id in range(actual_gpus):
            console.print(f"  [green]GPU {gpu_id}: {gpu_counts[gpu_id]} images[/green]")

        console.print(
            f"\n[bold green]Multi-GPU processing complete: "
            f"{elapsed:.1f}s total[/bold green]"
        )

        # Clean up: exit all model context managers
        for _gpu_config, model_ctx, _processor in gpu_stacks:
            model_ctx.__exit__(None, None, None)

        # Collect results in original order
        batch_results = [r for r in results if r is not None]
        batch_stats: dict[str, float] = {
            "avg_detection_batch": 1.0,
            "avg_extraction_batch": 1.0,
            "configured_batch_size": 1.0,
        }

        self.inference_elapsed = elapsed
        return batch_results, list(timings), doc_types_found, batch_stats

    @staticmethod
    def _worker_loop(
        gpu_id: int,
        processor: Any,
        bank_adapter: Any,
        work_queue: queue.Queue,
        results: list[dict | None],
        timings: list[float],
        doc_types_found: dict[str, int],
        doc_types_lock: threading.Lock,
        gpu_counts: list[int],
        completed: list[int],
        total_images: int,
        field_definitions: dict[str, list[str]],
        ground_truth_data: dict[str, dict],
    ) -> None:
        """Pull images from shared queue until empty. Detect, extract, evaluate."""
        import torch

        from common.simple_model_evaluator import SimpleModelEvaluator

        torch.cuda.set_device(gpu_id)
        evaluator = SimpleModelEvaluator()

        # Build doc_type -> fields mapping for evaluation filtering
        from common.batch_processor import load_document_field_definitions

        doc_type_fields = field_definitions or load_document_field_definitions()

        while True:
            try:
                idx, image_path = work_queue.get_nowait()
            except queue.Empty:
                break

            img_start = time.time()
            image_name = image_path.name
            image_str = str(image_path)

            try:
                # Detect
                classification = processor.detect_and_classify_document(
                    image_str, verbose=False
                )
                doc_type = classification["document_type"]

                with doc_types_lock:
                    doc_types_found[doc_type] = doc_types_found.get(doc_type, 0) + 1

                # Extract (route bank statements to adapter)
                if doc_type.upper() == "BANK_STATEMENT" and bank_adapter is not None:
                    schema_fields, metadata = bank_adapter.extract_bank_statement(
                        image_str
                    )
                    extraction_result = {
                        "extracted_data": schema_fields,
                        "raw_response": metadata.get("raw_responses", {}).get(
                            "turn1", ""
                        ),
                        "field_list": list(schema_fields.keys()),
                        "metadata": metadata,
                        "skip_math_enhancement": True,
                    }
                    prompt_name = (
                        f"unified_bank_{metadata.get('strategy_used', 'unknown')}"
                    )
                else:
                    extraction_result = processor.process_document_aware(
                        image_str, classification, verbose=False
                    )
                    prompt_name = doc_type.lower()

                # Evaluate against ground truth
                gt = _lookup_ground_truth(ground_truth_data, image_str)
                if gt:
                    evaluation = _evaluate_extraction(
                        evaluator,
                        extraction_result,
                        gt,
                        doc_type,
                        image_name,
                        doc_type_fields,
                    )
                else:
                    evaluation = {
                        "error": f"No ground truth for {image_name}",
                        "overall_accuracy": 0,
                    }

                processing_time = time.time() - img_start

                results[idx] = {
                    "image_name": image_name,
                    "image_path": image_str,
                    "document_type": doc_type,
                    "extraction_result": extraction_result,
                    "evaluation": evaluation,
                    "processing_time": processing_time,
                    "prompt_used": prompt_name,
                    "timestamp": datetime.now().isoformat(),
                }
                timings[idx] = processing_time

            except Exception as e:
                processing_time = time.time() - img_start
                results[idx] = {
                    "image_name": image_name,
                    "image_path": image_str,
                    "error": str(e),
                    "processing_time": processing_time,
                }
                timings[idx] = processing_time

            gpu_counts[gpu_id] += 1
            with doc_types_lock:
                completed[0] += 1
                n = completed[0]
            elapsed = time.time() - img_start
            console.print(
                f"  [{n}/{total_images}] GPU {gpu_id}: {image_name} ({elapsed:.1f}s)"
            )
            work_queue.task_done()

    def _create_bank_adapter(self, processor: Any) -> Any:
        """Create a BankStatementAdapter for a processor, if bank_v2 is enabled."""
        if not self.config.bank_v2:
            return None
        if not getattr(processor, "supports_multi_turn", True):
            return None

        from common.bank_statement_adapter import BankStatementAdapter

        return BankStatementAdapter(
            generate_fn=processor.generate,
            verbose=self.config.verbose,
            use_balance_correction=self.config.balance_correction,
        )

    def _load_ground_truth(self) -> dict[str, dict]:
        """Load ground truth data once for the entire batch."""
        if not self.config.ground_truth:
            return {}
        try:
            from common.evaluation_metrics import load_ground_truth

            return load_ground_truth(str(self.config.ground_truth), verbose=False)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load ground truth: {e}[/yellow]")
            return {}


def _lookup_ground_truth(ground_truth_data: dict[str, dict], image_path: str) -> dict:
    """Look up ground truth for an image with fuzzy matching.

    Mirrors BatchDocumentProcessor._lookup_ground_truth but as a standalone
    function for use by worker threads.
    """
    if not ground_truth_data:
        return {}

    image_name = Path(image_path).name
    ground_truth = ground_truth_data.get(image_name, {})
    if ground_truth:
        return ground_truth

    image_stem = Path(image_path).stem
    for gt_key in ground_truth_data:
        if Path(gt_key).stem == image_stem or gt_key == image_stem:
            return ground_truth_data[gt_key]

    return {}


def _evaluate_extraction(
    evaluator: Any,
    extraction_result: dict,
    ground_truth: dict,
    document_type: str,
    image_name: str,
    doc_type_fields: dict[str, list[str]],
) -> dict:
    """Evaluate extraction results against ground truth.

    Mirrors the core evaluation logic from BatchDocumentProcessor._evaluate_extraction
    without math enhancement (handled by UnifiedBankExtractor) or verbose output.
    """
    import os

    from common.evaluation_metrics import calculate_field_accuracy_with_method

    extracted_data = extraction_result.get("extracted_data", {})

    # Filter ground truth to document-specific fields
    evaluation_fields = doc_type_fields.get(
        document_type.lower(), doc_type_fields.get("invoice", [])
    )
    filtered_gt = {f: ground_truth[f] for f in evaluation_fields if f in ground_truth}

    eval_result = evaluator.evaluate_extraction(extracted_data, filtered_gt, image_name)

    fields_extracted = len([k for k, v in extracted_data.items() if v != "NOT_FOUND"])

    # Field-level accuracy
    evaluation_method = os.environ.get("EVALUATION_METHOD", "order_aware_f1")
    field_scores: dict[str, dict] = {}
    total_f1 = 0.0
    for field_name in evaluation_fields:
        if field_name in filtered_gt:
            pred = extracted_data.get(field_name, "NOT_FOUND")
            gt_val = filtered_gt[field_name]
            f1_metrics = calculate_field_accuracy_with_method(
                pred, gt_val, field_name, method=evaluation_method
            )
            field_scores[field_name] = f1_metrics
            total_f1 += f1_metrics["f1_score"]

    num_fields = len(field_scores)
    mean_f1 = total_f1 / num_fields if num_fields else 0.0

    f1_values = [s["f1_score"] for s in field_scores.values()]
    if f1_values:
        sorted_f1 = sorted(f1_values)
        mid = len(sorted_f1) // 2
        median_f1 = (
            (sorted_f1[mid - 1] + sorted_f1[mid]) / 2
            if len(sorted_f1) % 2 == 0
            else sorted_f1[mid]
        )
    else:
        median_f1 = 0.0

    perfect_matches = sum(1 for s in field_scores.values() if s["f1_score"] == 1.0)

    return {
        "overall_accuracy": mean_f1,
        "median_f1": median_f1,
        "field_scores": field_scores,
        "fields_extracted": fields_extracted,
        "correct_fields": perfect_matches,
        "fields_matched": perfect_matches,
        "total_fields": num_fields,
        "missing_fields": eval_result.missing_fields,
        "incorrect_fields": eval_result.incorrect_fields,
        "evaluation_method": evaluation_method,
    }
