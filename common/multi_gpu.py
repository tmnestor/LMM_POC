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

    Each GPU loads its own model + processor + bank adapter stack
    and processes a contiguous partition of images.
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

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=actual_gpus) as executor:
            futures = {
                executor.submit(
                    self._process_on_gpu,
                    gpu_id,
                    chunk,
                    prompt_config,
                    universal_fields,
                    field_definitions,
                ): gpu_id
                for gpu_id, chunk in enumerate(chunks)
            }

            # Collect results in GPU order
            gpu_results: list[tuple | None] = [None] * actual_gpus
            for future in as_completed(futures):
                gpu_id = futures[future]
                gpu_results[gpu_id] = future.result()
                console.print(f"  [green]GPU {gpu_id} finished[/green]")

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Multi-GPU processing complete: "
            f"{elapsed:.1f}s total[/bold green]"
        )

        return self._merge_results(gpu_results)

    def _process_on_gpu(
        self,
        gpu_id: int,
        images: list[Path],
        prompt_config: dict[str, Any],
        universal_fields: list[str],
        field_definitions: dict[str, list[str]],
    ) -> tuple[list[dict], list[float], dict[str, int], dict[str, float]]:
        """Load model on a specific GPU and process its image chunk.

        Each call creates an independent model/processor/adapter stack
        pinned to cuda:{gpu_id}. Model loading is serialized via
        _model_load_lock to avoid transformers import race conditions;
        GPU inference runs fully parallel after loading completes.
        """
        from cli import create_processor, load_model, run_batch_processing

        gpu_config = replace(
            self.config,
            device_map=f"cuda:{gpu_id}",
        )

        console.print(f"  [dim]GPU {gpu_id}: loading model on cuda:{gpu_id}...[/dim]")

        # load_model is a context manager — we need it alive during processing,
        # but only the loading phase itself needs the lock.
        model_ctx = load_model(gpu_config)

        with _model_load_lock:
            model, tokenizer = model_ctx.__enter__()
            processor = create_processor(
                model,
                tokenizer,
                gpu_config,
                prompt_config,
                universal_fields,
                field_definitions,
            )

        # Processing runs in parallel (GIL released during CUDA kernels)
        try:
            batch_results, processing_times, document_types_found, batch_stats = (
                run_batch_processing(
                    gpu_config,
                    processor,
                    prompt_config,
                    images,
                    field_definitions,
                )
            )
        finally:
            model_ctx.__exit__(None, None, None)

        return batch_results, processing_times, document_types_found, batch_stats

    def _partition_images(self, images: list[Path]) -> list[list[Path]]:
        """Split images into num_gpus contiguous chunks.

        If there are fewer images than GPUs, returns fewer chunks
        (one image per chunk).
        """
        n = min(self.num_gpus, len(images))
        chunk_size = math.ceil(len(images) / n)
        chunks = [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]
        return chunks

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
