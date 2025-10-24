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
import math
import time
from typing import Any, Dict, Tuple

import torch
from rich import print as rprint
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
                "stage": stage,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "available_gb": available,
                "timestamp": time.time(),
            }

            self.memory_checkpoints.append(checkpoint)
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)

            if self.verbose:
                rprint(
                    f"[blue]📊 Memory checkpoint ({stage}): {allocated:.2f}GB allocated, {available:.2f}GB available[/blue]"
                )

            return checkpoint

        return {
            "stage": stage,
            "allocated_gb": 0,
            "reserved_gb": 0,
            "available_gb": 0,
            "timestamp": time.time(),
        }

    def detect_memory_spike(self, threshold_gb: float = 2.0) -> bool:
        """Detect if there's been a sudden memory spike."""
        if len(self.memory_checkpoints) < 2:
            return False

        recent = self.memory_checkpoints[-1]
        previous = self.memory_checkpoints[-2]

        spike = recent["allocated_gb"] - previous["allocated_gb"]

        if spike > threshold_gb:
            if self.verbose:
                rprint(
                    f"[yellow]⚠️ Memory spike detected: +{spike:.2f}GB between {previous['stage']} and {recent['stage']}[/yellow]"
                )
            return True

        return False

    def aggressive_cleanup_for_8b(self):
        """Perform aggressive cleanup specifically designed for InternVL3-8B."""
        if self.verbose:
            rprint(
                "[bold red]🧹 Performing aggressive cleanup for InternVL3-8B...[/bold red]"
            )

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
            rprint(
                f"[green]✅ Cleanup complete. Available memory: {available:.2f}GB[/green]"
            )

    def _clear_vision_transformer_caches(self):
        """Clear caches specific to vision transformer components."""
        # Clear any vision-specific caches that might be lingering
        try:
            # Clear torchvision caches if available
            if hasattr(torch.backends.cudnn, "benchmark"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.benchmark = True

            # Clear any transformer caches
            if hasattr(torch.nn.functional, "_clearCachedTensors"):
                torch.nn.functional._clearCachedTensors()

        except Exception as e:
            if self.verbose:
                rprint(f"[yellow]⚠️ Vision cache clearing warning: {e}[/yellow]")

    def create_official_device_map(self, model_path: str) -> Dict[str, int]:
        """
        Create official InternVL3 multi-GPU device map following documentation.

        Based on: https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html
        """
        device_map = {}
        world_size = torch.cuda.device_count()

        if world_size < 2:
            if self.verbose:
                rprint(
                    "[yellow]⚠️ Single GPU detected - using standard device_map='auto'[/yellow]"
                )
            return "auto"

        if self.verbose:
            rprint(
                f"[cyan]🔧 Creating official multi-GPU device map for {world_size} GPUs[/cyan]"
            )

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            num_layers = config.llm_config.num_hidden_layers

            # Official strategy: First GPU treated as half a GPU due to ViT
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

            # Distribute language model layers
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for _ in range(num_layer):  # Using _ for unused loop variable
                    if layer_cnt < num_layers:
                        device_map[f"language_model.model.layers.{layer_cnt}"] = i
                        layer_cnt += 1

            # Critical components on GPU 0 (official requirement)
            device_map["vision_model"] = 0
            device_map["mlp1"] = 0
            device_map["language_model.model.tok_embeddings"] = 0
            device_map["language_model.model.embed_tokens"] = 0
            device_map["language_model.output"] = 0
            device_map["language_model.model.norm"] = 0
            device_map["language_model.model.rotary_emb"] = 0
            device_map["language_model.lm_head"] = 0
            device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

            if self.verbose:
                rprint(
                    f"[green]✅ Official device map created: {num_layers} layers across {world_size} GPUs[/green]"
                )
                for device_id in range(world_size):
                    layer_count = sum(1 for v in device_map.values() if v == device_id)
                    rprint(f"[cyan]  GPU {device_id}: {layer_count} components[/cyan]")

            return device_map

        except Exception as e:
            if self.verbose:
                rprint(f"[yellow]⚠️ Error creating official device map: {e}[/yellow]")
                rprint("[yellow]💡 Falling back to device_map='auto'[/yellow]")
            return "auto"

    def check_memory_requirements(self, model_path: str) -> Tuple[bool, str]:
        """Check if we have enough memory for InternVL3-8B."""
        available = get_available_gpu_memory()

        world_size = torch.cuda.device_count()

        if world_size >= 2:
            # Multi-GPU setup - use official requirements from docs
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory / 1e9
                for i in range(world_size)
            )
            if self.verbose:
                rprint(
                    f"[cyan]🔍 Multi-GPU setup detected: {world_size} GPUs, {total_memory:.1f}GB total[/cyan]"
                )

            # Removed strict GPU count requirement - allow any multi-GPU configuration
            # Previously required 3+ GPUs or 90GB+ total, now accepts 2+ GPUs
            return (
                True,
                f"✅ Multi-GPU detected: {world_size} GPUs, {total_memory:.1f}GB total",
            )
        else:
            # Single GPU - original estimation
            base_model_memory = 8.0  # ~8GB for quantized language model
            vision_encoder_memory = 6.0  # ~6GB for large vision encoder
            overhead_memory = 2.0  # ~2GB for processing overhead

            total_required = base_model_memory + vision_encoder_memory + overhead_memory

            if available >= total_required:
                return (
                    True,
                    f"✅ Single GPU sufficient: {available:.1f}GB available, {total_required:.1f}GB required",
                )
            else:
                deficit = total_required - available
                return (
                    False,
                    f"❌ Single GPU insufficient: {available:.1f}GB available, {total_required:.1f}GB required (deficit: {deficit:.1f}GB)",
                )

    def sequential_model_loading(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,  # V100-compatible (changed from bfloat16)
        low_cpu_mem_usage: bool = True,
        use_flash_attn: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Load InternVL3-8B using sequential component loading to minimize memory spikes.

        This approach loads components one at a time with memory checkpoints
        to avoid the large memory spike that occurs during standard loading.
        """
        if self.verbose:
            rprint(
                "[bold green]🚀 Starting sequential InternVL3-8B loading...[/bold green]"
            )

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
                model_path, trust_remote_code=True, use_fast=False
            )
            self.create_memory_checkpoint("tokenizer_loaded")

            # Step 3: Load model with optimized settings
            if self.verbose:
                rprint(
                    "[cyan]📥 Loading InternVL3-8B model with memory optimization...[/cyan]"
                )

            # Create official device map for multi-GPU or fallback for single GPU
            device_map = self.create_official_device_map(model_path)

            if self.verbose:
                if isinstance(device_map, str):
                    rprint(f"[cyan]📥 Using device mapping: {device_map}[/cyan]")
                else:
                    rprint("[cyan]📥 Using official multi-GPU device mapping[/cyan]")

            # Use official InternVL3 loading pattern with proper device mapping
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_flash_attn=use_flash_attn,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()

            self.create_memory_checkpoint("model_loaded")

            # Step 4: Check for memory spikes
            if self.detect_memory_spike(threshold_gb=3.0):
                rprint("[yellow]⚠️ Large memory spike detected during loading[/yellow]")
                # Perform intermediate cleanup
                torch.cuda.empty_cache()
                gc.collect()

            # Step 5: CUDA placement (only needed for single GPU without device_map)
            if isinstance(device_map, str) and device_map == "auto":
                # For device_map="auto", model is already on appropriate devices
                if self.verbose:
                    rprint(
                        "[cyan]📤 Model automatically placed by device_map='auto'[/cyan]"
                    )
            elif isinstance(device_map, str):
                # For single GPU setups without multi-GPU device mapping
                if self.verbose:
                    rprint("[cyan]📤 Moving model to CUDA...[/cyan]")
                model = model.cuda()
            else:
                # Multi-GPU setup - model components already distributed
                if self.verbose:
                    rprint(
                        "[cyan]📤 Model distributed across GPUs by official device mapping[/cyan]"
                    )

            self.create_memory_checkpoint("model_on_cuda")

            # Step 6: Final memory optimization
            # REMOVED: gradient_checkpointing_enable() - meant for training, not inference
            # This was suspected cause of gibberish responses during inference
            if self.verbose:
                rprint(
                    "[yellow]⚠️ Skipped gradient checkpointing (meant for training, not inference)[/yellow]"
                )

            # Final memory report
            final_checkpoint = self.create_memory_checkpoint("loading_complete")

            if self.verbose:
                param_count = sum(p.numel() for p in model.parameters())
                rprint("[green]✅ InternVL3-8B loaded successfully![/green]")
                rprint(f"[blue]📊 Model parameters: {param_count:,}[/blue]")
                rprint(
                    f"[blue]🎯 Peak memory usage: {self.peak_memory_usage:.2f}GB[/blue]"
                )
                rprint(
                    f"[blue]💾 Final memory usage: {final_checkpoint['allocated_gb']:.2f}GB[/blue]"
                )

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
            "checkpoints": self.memory_checkpoints,
            "peak_memory_gb": self.peak_memory_usage,
            "current_available_gb": get_available_gpu_memory(),
            "loading_stages": len(self.memory_checkpoints),
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
            stage = checkpoint["stage"]
            allocated = checkpoint["allocated_gb"]
            available = checkpoint["available_gb"]

            if i > 0:
                prev_allocated = self.memory_checkpoints[i - 1]["allocated_gb"]
                delta = allocated - prev_allocated
                delta_str = f" ({delta:+.2f}GB)" if delta != 0 else ""
            else:
                delta_str = ""

            rprint(
                f"[cyan]  {i + 1}. {stage}: {allocated:.2f}GB allocated, {available:.2f}GB available{delta_str}[/cyan]"
            )


def load_internvl3_8b_optimized(
    model_path: str,
    torch_dtype: torch.dtype = torch.float16,  # V100-compatible (changed from bfloat16)
    low_cpu_mem_usage: bool = True,
    use_flash_attn: bool = False,
    verbose: bool = True,
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
            use_flash_attn=use_flash_attn,
        )

        if verbose:
            manager.print_memory_report()

        return model, tokenizer

    except Exception as e:
        if verbose:
            rprint(f"[red]❌ InternVL3-8B optimized loading failed: {e}[/red]")
            manager.print_memory_report()
        raise
