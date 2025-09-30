"""
InternVL3-8B Specific Memory Optimization

This module provides specialized memory management for InternVL3-8B models,
which have disproportionately large vision encoders that cause OOM issues
even when smaller models work fine.

Key Features:
    - Sequential component loading (vision encoder, then language model)
    - Aggressive memory monitoring with checkpoints
    - Specialized cleanup for vision transformer components
    - Memory spike detection and mitigation
    - CPU offloading fallback strategies

Based on the analysis that InternVL3-8B has a much larger vision encoder
than InternVL3-2B, causing memory issues even with quantization.
"""

import gc
import time
from typing import Any, Dict, Tuple

import torch
from rich import print as rprint
from transformers import AutoModel, AutoTokenizer

from common.gpu_optimization import (
    aggressive_defragmentation,
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    emergency_cleanup,
    get_available_gpu_memory,
)


class InternVL3_8B_MemoryManager:
    """Specialized memory manager for InternVL3-8B models."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.memory_checkpoints = []
        self.peak_memory_usage = 0.0
        self.loading_stage = "initialization"

    def create_memory_checkpoint(self, stage: str) -> Dict[str, float]:
        """Create a memory usage checkpoint."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            available = get_available_gpu_memory()

            checkpoint = {
                'stage': stage,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'available_gb': available,
                'timestamp': time.time()
            }

            self.memory_checkpoints.append(checkpoint)
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)

            if self.verbose:
                rprint(f"[blue]📊 Memory checkpoint ({stage}): {allocated:.2f}GB allocated, {available:.2f}GB available[/blue]")

            return checkpoint

        return {'stage': stage, 'allocated_gb': 0, 'reserved_gb': 0, 'available_gb': 0, 'timestamp': time.time()}

    def detect_memory_spike(self, threshold_gb: float = 2.0) -> bool:
        """Detect if there's been a sudden memory spike."""
        if len(self.memory_checkpoints) < 2:
            return False

        recent = self.memory_checkpoints[-1]
        previous = self.memory_checkpoints[-2]

        spike = recent['allocated_gb'] - previous['allocated_gb']

        if spike > threshold_gb:
            if self.verbose:
                rprint(f"[yellow]⚠️ Memory spike detected: +{spike:.2f}GB between {previous['stage']} and {recent['stage']}[/yellow]")
            return True

        return False

    def aggressive_cleanup_for_8b(self):
        """Perform aggressive cleanup specifically designed for InternVL3-8B."""
        if self.verbose:
            rprint("[bold red]🧹 Performing aggressive cleanup for InternVL3-8B...[/bold red]")

        # Step 1: Clear any existing model references
        emergency_cleanup(verbose=self.verbose)

        # Step 2: Clear vision transformer specific caches
        self._clear_vision_transformer_caches()

        # Step 3: Multiple rounds of cache clearing
        for round_num in range(3):
            if self.verbose:
                rprint(f"[blue]🔄 Cleanup round {round_num + 1}/3[/blue]")

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.1)  # Allow system to settle

        # Step 4: Aggressive defragmentation
        aggressive_defragmentation()

        # Step 5: Final comprehensive cleanup
        comprehensive_memory_cleanup()

        if self.verbose:
            available = get_available_gpu_memory()
            rprint(f"[green]✅ Cleanup complete. Available memory: {available:.2f}GB[/green]")

    def _clear_vision_transformer_caches(self):
        """Clear caches specific to vision transformer components."""
        # Clear any vision-specific caches that might be lingering
        try:
            # Clear torchvision caches if available
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.benchmark = True

            # Clear any transformer caches
            if hasattr(torch.nn.functional, '_clearCachedTensors'):
                torch.nn.functional._clearCachedTensors()

        except Exception as e:
            if self.verbose:
                rprint(f"[yellow]⚠️ Vision cache clearing warning: {e}[/yellow]")

    def check_memory_requirements(self, model_path: str) -> Tuple[bool, str]:
        """Check if we have enough memory for InternVL3-8B."""
        available = get_available_gpu_memory()

        # InternVL3-8B memory requirements (estimated)
        base_model_memory = 8.0  # ~8GB for quantized language model
        vision_encoder_memory = 6.0  # ~6GB for large vision encoder
        overhead_memory = 2.0  # ~2GB for processing overhead

        total_required = base_model_memory + vision_encoder_memory + overhead_memory

        if available >= total_required:
            return True, f"✅ Sufficient memory: {available:.1f}GB available, {total_required:.1f}GB required"
        else:
            deficit = total_required - available
            return False, f"❌ Insufficient memory: {available:.1f}GB available, {total_required:.1f}GB required (deficit: {deficit:.1f}GB)"

    def sequential_model_loading(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        low_cpu_mem_usage: bool = True,
        use_flash_attn: bool = False
    ) -> Tuple[Any, Any]:
        """
        Load InternVL3-8B using sequential component loading to minimize memory spikes.

        This approach loads components one at a time with memory checkpoints
        to avoid the large memory spike that occurs during standard loading.
        """
        if self.verbose:
            rprint("[bold green]🚀 Starting sequential InternVL3-8B loading...[/bold green]")

        # Initial cleanup and memory check
        self.aggressive_cleanup_for_8b()
        self.create_memory_checkpoint("pre_loading")

        # Check if we have enough memory
        sufficient, message = self.check_memory_requirements(model_path)
        if self.verbose:
            rprint(f"[blue]📊 Memory assessment: {message}[/blue]")

        if not sufficient:
            raise RuntimeError(f"Insufficient GPU memory for InternVL3-8B: {message}")

        try:
            # Step 1: Configure CUDA memory allocation
            configure_cuda_memory_allocation(verbose=self.verbose)
            self.create_memory_checkpoint("cuda_configured")

            # Step 2: Load tokenizer first (minimal memory impact)
            if self.verbose:
                rprint("[cyan]📥 Loading tokenizer...[/cyan]")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            self.create_memory_checkpoint("tokenizer_loaded")

            # Step 3: Load model with optimized settings
            if self.verbose:
                rprint("[cyan]📥 Loading InternVL3-8B model with memory optimization...[/cyan]")

            # Use sequential loading with direct CUDA placement (fixed for gibberish issue)
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_flash_attn=use_flash_attn,
                trust_remote_code=True
                # REMOVED: device_map="auto" - suspected cause of gibberish responses
            ).eval()

            self.create_memory_checkpoint("model_loaded")

            # Step 4: Check for memory spikes
            if self.detect_memory_spike(threshold_gb=3.0):
                rprint("[yellow]⚠️ Large memory spike detected during loading[/yellow]")
                # Perform intermediate cleanup
                torch.cuda.empty_cache()
                gc.collect()

            # Step 5: Move to CUDA (always required without device_map="auto")
            if self.verbose:
                rprint("[cyan]📤 Moving model to CUDA...[/cyan]")
            model = model.cuda()

            self.create_memory_checkpoint("model_on_cuda")

            # Step 6: Final memory optimization
            # REMOVED: gradient_checkpointing_enable() - meant for training, not inference
            # This was suspected cause of gibberish responses during inference
            if self.verbose:
                rprint("[yellow]⚠️ Skipped gradient checkpointing (meant for training, not inference)[/yellow]")

            # Final memory report
            final_checkpoint = self.create_memory_checkpoint("loading_complete")

            if self.verbose:
                param_count = sum(p.numel() for p in model.parameters())
                rprint("[green]✅ InternVL3-8B loaded successfully![/green]")
                rprint(f"[blue]📊 Model parameters: {param_count:,}[/blue]")
                rprint(f"[blue]🎯 Peak memory usage: {self.peak_memory_usage:.2f}GB[/blue]")
                rprint(f"[blue]💾 Final memory usage: {final_checkpoint['allocated_gb']:.2f}GB[/blue]")

            return model, tokenizer

        except Exception as e:
            if self.verbose:
                rprint(f"[red]❌ Error during InternVL3-8B loading: {e}[/red]")

            # Emergency cleanup on failure
            self.aggressive_cleanup_for_8b()
            raise

    def get_memory_report(self) -> Dict[str, Any]:
        """Get a comprehensive memory usage report."""
        return {
            'checkpoints': self.memory_checkpoints,
            'peak_memory_gb': self.peak_memory_usage,
            'current_available_gb': get_available_gpu_memory(),
            'loading_stages': len(self.memory_checkpoints)
        }

    def print_memory_report(self):
        """Print a formatted memory usage report."""
        if not self.memory_checkpoints:
            rprint("[yellow]⚠️ No memory checkpoints recorded[/yellow]")
            return

        rprint("[bold blue]📊 InternVL3-8B Memory Usage Report[/bold blue]")
        rprint(f"[blue]🏔️ Peak memory usage: {self.peak_memory_usage:.2f}GB[/blue]")
        rprint(f"[blue]📋 Loading stages: {len(self.memory_checkpoints)}[/blue]")

        rprint("\n[bold blue]Memory Timeline:[/bold blue]")
        for i, checkpoint in enumerate(self.memory_checkpoints):
            stage = checkpoint['stage']
            allocated = checkpoint['allocated_gb']
            available = checkpoint['available_gb']

            if i > 0:
                prev_allocated = self.memory_checkpoints[i-1]['allocated_gb']
                delta = allocated - prev_allocated
                delta_str = f" ({delta:+.2f}GB)" if delta != 0 else ""
            else:
                delta_str = ""

            rprint(f"[cyan]  {i+1}. {stage}: {allocated:.2f}GB allocated, {available:.2f}GB available{delta_str}[/cyan]")


def load_internvl3_8b_optimized(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    low_cpu_mem_usage: bool = True,
    use_flash_attn: bool = False,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Optimized loading function for InternVL3-8B that handles the large vision encoder.

    Args:
        model_path: Path to InternVL3-8B model
        torch_dtype: Data type for model weights
        low_cpu_mem_usage: Enable low CPU memory usage
        use_flash_attn: Enable Flash Attention (set False for V100)
        verbose: Enable detailed logging

    Returns:
        Tuple of (model, tokenizer)
    """
    manager = InternVL3_8B_MemoryManager(verbose=verbose)

    try:
        model, tokenizer = manager.sequential_model_loading(
            model_path=model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_flash_attn=use_flash_attn
        )

        if verbose:
            manager.print_memory_report()

        return model, tokenizer

    except Exception as e:
        if verbose:
            rprint(f"[red]❌ InternVL3-8B optimized loading failed: {e}[/red]")
            manager.print_memory_report()
        raise