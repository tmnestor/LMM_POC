"""Multi-GPU parallel processing orchestrator.

Partitions images across N GPUs, loads an independent model per GPU,
processes image subsets in parallel using ThreadPoolExecutor, and merges
results in original order.

ThreadPoolExecutor is used (not multiprocessing) because PyTorch releases
the GIL during CUDA kernel execution, giving true GPU parallelism without
serialization overhead.
"""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
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
      2. Process image chunks in parallel (GIL released during CUDA kernels)
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
        """Process images in parallel across GPUs.

        Returns merged (batch_results, processing_times, document_types_found, batch_stats).
        """
        from cli import create_processor, load_model
        from models.registry import _print_gpu_status

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
        console.print("\n[bold]Loading models on all GPUs...[/bold]")
        gpu_stacks: list[tuple] = []
        for gpu_id in range(actual_gpus):
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

        # Print unified GPU status after all models are loaded
        _print_gpu_status(console)

        # Phase 2: Process image chunks in parallel
        console.print("\n[bold]Processing images in parallel...[/bold]")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
            futures = {
                executor.submit(
                    self._process_chunk,
                    gpu_stacks[gpu_id],
                    chunks[gpu_id],
                    prompt_config,
                    field_definitions,
                ): gpu_id
                for gpu_id in range(actual_gpus)
            }

            gpu_results: list[tuple | None] = [None] * actual_gpus
            for future in as_completed(futures):
                gpu_id = futures[future]
                gpu_results[gpu_id] = future.result()
                console.print(f"  [green]GPU {gpu_id} finished[/green]")

        # Clean up: exit all model context managers
        for _gpu_config, model_ctx, _processor in gpu_stacks:
            model_ctx.__exit__(None, None, None)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Multi-GPU processing complete: "
            f"{elapsed:.1f}s total[/bold green]"
        )

        return self._merge_results(gpu_results)

    @staticmethod
    def _process_chunk(
        gpu_stack: tuple,
        images: list[Path],
        prompt_config: dict[str, Any],
        field_definitions: dict[str, list[str]],
    ) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
        """Process an image chunk using a pre-loaded model/processor stack."""
        from cli import run_batch_processing

        gpu_config, _model_ctx, processor = gpu_stack
        return run_batch_processing(
            gpu_config, processor, prompt_config, images, field_definitions
        )

    def _partition_images(self, images: list[Path]) -> list[list[Path]]:
        """Split images into num_gpus contiguous chunks.

        If there are fewer images than GPUs, returns fewer chunks
        (one image per chunk).
        """
        n = min(self.num_gpus, len(images))
        chunk_size = math.ceil(len(images) / n)
        return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]

    @staticmethod
    def _merge_results(
        gpu_results: list[tuple | None],
    ) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
        """Merge results from all GPUs in original image order.

        Concatenates batch_results and processing_times, merges
        document_types_found dicts (summing counts), averages batch_stats.
        """
        all_results: list[dict] = []
        all_times: list[float] = []
        merged_doc_types: dict[str, int] = {}
        all_batch_stats: list[dict[str, float]] = []

        for result in gpu_results:
            if result is None:
                continue
            batch_results, processing_times, doc_types, batch_stats = result

            all_results.extend(batch_results)
            all_times.extend(processing_times)

            for doc_type, count in doc_types.items():
                merged_doc_types[doc_type] = merged_doc_types.get(doc_type, 0) + count

            if batch_stats:
                all_batch_stats.append(batch_stats)

        # Average batch_stats across GPUs
        averaged_stats: dict[str, float] = {}
        if all_batch_stats:
            all_keys = set()
            for stats in all_batch_stats:
                all_keys.update(stats.keys())
            for key in all_keys:
                values = [s[key] for s in all_batch_stats if key in s]
                averaged_stats[key] = sum(values) / len(values) if values else 0.0

        return all_results, all_times, merged_doc_types, averaged_stats
