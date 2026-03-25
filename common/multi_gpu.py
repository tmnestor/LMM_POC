"""Multi-GPU parallel processing orchestrator.

Partitions images across N GPUs, loads an independent model per GPU,
processes image subsets in parallel using ThreadPoolExecutor, and merges
results in original order.

ThreadPoolExecutor is used (not multiprocessing) because PyTorch releases
the GIL during CUDA kernel execution, giving true GPU parallelism without
serialization overhead.
"""

import math
import random
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

    def __init__(
        self,
        config,
        num_gpus: int,
        *,
        shuffle: bool = False,
        type_aware: bool = False,
    ) -> None:
        self.config = config
        self.num_gpus = num_gpus
        self.shuffle = shuffle
        self.type_aware = type_aware

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

        actual_gpus = min(self.num_gpus, len(images))
        chunk_classifications: list[list[dict]] | None = None

        if self.type_aware:
            # --- Type-aware flow ---
            # Load GPU 0 first, run Phase 0 detection, then partition by type
            console.print(
                "\n[bold]Loading model on GPU 0 for Phase 0 detection...[/bold]"
            )
            gpu0_config = replace(self.config, device_map="cuda:0")
            gpu0_config._multi_gpu = True

            import torch

            torch.cuda.set_device(0)
            model_ctx_0 = load_model(gpu0_config)
            model_0, tokenizer_0 = model_ctx_0.__enter__()
            processor_0 = create_processor(
                model_0,
                tokenizer_0,
                gpu0_config,
                prompt_config,
                universal_fields,
                field_definitions,
            )

            # Phase 0: Detect all images on GPU 0
            all_classifications = self._run_phase0_detection(processor_0, images)

            # Type-aware partition
            chunks, chunk_classifications = self._type_aware_partition(
                images, all_classifications, actual_gpus
            )
            actual_gpus = len(chunks)

            console.print(
                f"\n[bold cyan]Multi-GPU: distributing {len(images)} images "
                f"across {actual_gpus} GPUs (type-aware)[/bold cyan]"
            )
            for gpu_id, chunk in enumerate(chunks):
                bank_count = sum(
                    1
                    for c in chunk_classifications[gpu_id]
                    if c["document_type"].upper() == "BANK_STATEMENT"
                )
                console.print(
                    f"  [dim]GPU {gpu_id}: {len(chunk)} images "
                    f"({bank_count} bank, {len(chunk) - bank_count} standard)[/dim]"
                )

            # Load remaining GPUs
            gpu_stacks: list[tuple] = [(gpu0_config, model_ctx_0, processor_0)]
            if actual_gpus > 1:
                console.print("\n[bold]Loading models on remaining GPUs...[/bold]")
                for gpu_id in range(1, actual_gpus):
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
        else:
            # --- Standard flow ---
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

            console.print("\n[bold]Loading models on all GPUs...[/bold]")
            gpu_stacks = []
            for gpu_id in range(actual_gpus):
                gpu_config = replace(self.config, device_map=f"cuda:{gpu_id}")
                gpu_config._multi_gpu = True

                import torch

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

        # Print unified GPU status after all models are loaded
        _print_gpu_status(console)

        # Process image chunks in parallel (throughput timing starts here)
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
                gpu_elapsed = time.time() - start_time
                console.print(
                    f"  [green]GPU {gpu_id} finished: "
                    f"{gpu_elapsed:.1f}s ({len(chunks[gpu_id])} images)[/green]"
                )

        # Clean up: exit all model context managers
        for _gpu_config, model_ctx, _processor in gpu_stacks:
            model_ctx.__exit__(None, None, None)

        elapsed = time.time() - start_time
        console.print(
            f"\n[bold green]Multi-GPU processing complete: "
            f"{elapsed:.1f}s total[/bold green]"
        )

        results = self._merge_results(gpu_results)
        # Attach inference-only time (excludes model loading + Phase 0) for throughput
        self.inference_elapsed = elapsed
        return results

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

    @staticmethod
    def _run_phase0_detection(processor: Any, images: list[Path]) -> list[Any]:
        """Detect document types for all images on a single GPU.

        Runs before partitioning so we can distribute bank statements
        evenly across GPUs.
        """
        console.print(
            f"\n[bold]Phase 0: Detecting {len(images)} images on GPU 0...[/bold]"
        )
        start = time.time()
        image_strs = [str(img) for img in images]

        from models.protocol import BatchCapableProcessor

        if isinstance(processor, BatchCapableProcessor):
            classifications = processor.batch_detect_documents(
                image_strs, verbose=False
            )
        else:
            classifications = [
                processor.detect_and_classify_document(p, verbose=False)
                for p in image_strs
            ]

        elapsed = time.time() - start
        # Count types for summary
        type_counts: dict[str, int] = {}
        for c in classifications:
            dt = c["document_type"].upper()
            type_counts[dt] = type_counts.get(dt, 0) + 1

        console.print(f"  Phase 0 detection: {elapsed:.1f}s for {len(images)} images")
        for dt, count in sorted(type_counts.items()):
            console.print(f"    {dt}: {count}")

        return classifications

    def _type_aware_partition(
        self,
        images: list[Path],
        classifications: list[dict],
        num_gpus: int,
    ) -> tuple[list[list[Path]], list[list[dict]]]:
        """Partition images so each GPU gets an equal share of bank statements.

        Bank statements are dealt round-robin, then standard docs fill the
        remaining slots with a contiguous split.
        """
        bank_indices = [
            i
            for i, c in enumerate(classifications)
            if c["document_type"].upper() == "BANK_STATEMENT"
        ]
        standard_indices = [i for i in range(len(images)) if i not in set(bank_indices)]

        n = min(num_gpus, len(images))
        buckets: list[list[int]] = [[] for _ in range(n)]

        # Round-robin bank statements across GPUs
        for i, idx in enumerate(bank_indices):
            buckets[i % n].append(idx)

        # Fill with standard docs (contiguous split)
        chunk_size = math.ceil(len(standard_indices) / n) if standard_indices else 0
        for gpu_id in range(n):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(standard_indices))
            buckets[gpu_id].extend(standard_indices[start:end])

        chunks = [[images[i] for i in bucket] for bucket in buckets]
        chunk_cls = [[classifications[i] for i in bucket] for bucket in buckets]
        return chunks, chunk_cls

    @staticmethod
    def _shuffle_images(images: list[Path]) -> list[Path]:
        """Shuffle images for balanced document-type distribution across GPUs.

        Without shuffling, bank statements cluster by filename and one GPU
        gets most of the slow multi-turn extractions. Uses a fixed seed for
        deterministic reproducibility.
        """
        shuffled = list(images)
        random.seed(42)
        random.shuffle(shuffled)
        return shuffled

    def _partition_images(self, images: list[Path]) -> list[list[Path]]:
        """Split images into num_gpus contiguous chunks.

        If there are fewer images than GPUs, returns fewer chunks
        (one image per chunk).
        """
        if self.shuffle:
            images = self._shuffle_images(images)
            print(f"  Shuffled {len(images)} images for balanced GPU distribution")

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
            all_keys: set[str] = set()
            for stats in all_batch_stats:
                all_keys.update(stats.keys())
            for key in all_keys:
                values = [s[key] for s in all_batch_stats if key in s]
                averaged_stats[key] = sum(values) / len(values) if values else 0.0

        return all_results, all_times, merged_doc_types, averaged_stats
