"""
Llama 4 Model Loader with Multi-GPU Memory Detection

Specialized loader for Llama 4 Scout/Maverick models with MoE architecture.
Based on llama_model_loader_robust.py but updated for Llama 4 compatibility.

Key Features:
- Llama4ForConditionalGeneration support
- SDPA (scaled dot-product attention) for efficiency
- Robust multi-GPU memory detection
- Intelligent quantization decisions
- V100-specific optimizations
- Comprehensive error handling and fallbacks

Note: Uses SDPA instead of flex_attention due to known bug in flex_attention
with Llama 4 Scout (GitHub issue huggingface/transformers#37352)
"""

import gc
from pathlib import Path
from typing import Tuple

import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Import Llama 4 class
try:
    from transformers import AutoProcessor, Llama4ForConditionalGeneration

    LLAMA4_AVAILABLE = True
except ImportError:
    LLAMA4_AVAILABLE = False
    raise ImportError(
        "Llama4ForConditionalGeneration not available. "
        "Please upgrade transformers: pip install --upgrade transformers>=4.51.0"
    ) from None

# Import GPU optimization utilities
from .gpu_optimization import configure_cuda_memory_allocation


def load_llama4_model(
    model_path: str,
    use_quantization: bool = False,
    device_map: str = "auto",
    max_new_tokens: int = 4000,
    torch_dtype: str = "bfloat16",
    low_cpu_mem_usage: bool = True,
    verbose: bool = True,
) -> Tuple[Llama4ForConditionalGeneration, AutoProcessor]:
    """
    Load Llama 4 model (Scout or Maverick) with robust multi-GPU memory detection.

    Args:
        model_path: Path to the Llama 4 model directory
        use_quantization: Whether to use 8-bit quantization
        device_map: Device mapping strategy ("auto" recommended for multi-GPU)
        max_new_tokens: Maximum new tokens for generation
        torch_dtype: Torch data type ("bfloat16" recommended)
        low_cpu_mem_usage: Whether to use low CPU memory mode
        verbose: Enable detailed logging and diagnostics

    Returns:
        Tuple[Llama4ForConditionalGeneration, AutoProcessor]: Loaded model and processor

    Raises:
        FileNotFoundError: If model path doesn't exist
        RuntimeError: If model loading fails

    Example:
        >>> model, processor = load_llama4_model(
        ...     "/path/to/Llama-4-Scout-17B-16E-Instruct",
        ...     use_quantization=False,
        ...     verbose=True
        ... )
    """
    if verbose:
        rprint(
            "[bold blue]🚀 Loading Llama 4 model with robust multi-GPU optimization...[/bold blue]"
        )
        rprint(
            "[cyan]Features: MoE architecture, SDPA attention, memory management[/cyan]"
        )

    try:
        # PHASE 1: Configure CUDA memory allocation
        if verbose:
            rprint("[blue]🔧 Configuring CUDA memory for Llama 4...[/blue]")

        configure_cuda_memory_allocation(verbose=verbose)

        # Validate model path
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Convert torch_dtype string to actual dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

        # PHASE 2: Detect model type and estimate memory
        model_name = model_path_obj.name
        is_scout = "Scout" in model_name
        is_maverick = "Maverick" in model_name

        if is_scout:
            estimated_memory_needed = 50  # Scout: 109B total, 17B active
            memory_buffer = 10.0
            model_type = "Scout (17B active, 109B total, 16 experts)"
        elif is_maverick:
            estimated_memory_needed = 150  # Maverick: 400B total, 17B active
            memory_buffer = 20.0
            model_type = "Maverick (17B active, 400B total, 128 experts)"
        else:
            estimated_memory_needed = 50  # Default to Scout size
            memory_buffer = 10.0
            model_type = "Llama 4 (unknown variant)"

        # PHASE 3: Robust GPU memory detection
        quantization_config = None
        if torch.cuda.is_available():
            from .robust_gpu_memory import RobustGPUMemoryDetector

            if verbose:
                rprint("[blue]🔍 Performing robust GPU memory detection...[/blue]")

            memory_detector = RobustGPUMemoryDetector(verbose=verbose)
            memory_result = memory_detector.detect_gpu_memory()

            total_gpu_memory = memory_result.total_memory_gb
            total_available_memory = memory_result.total_available_gb
            working_gpus = memory_result.working_gpus

            memory_sufficient = total_available_memory >= (
                estimated_memory_needed + memory_buffer
            )

            if verbose:
                rprint(
                    f"[blue]📊 GPU Hardware: {working_gpus} GPU(s), {total_gpu_memory:.0f}GB total[/blue]"
                )
                rprint(f"[blue]🎯 Model: Llama 4 {model_type}[/blue]")
                rprint(
                    f"[blue]💾 Available Memory: {total_available_memory:.1f}GB[/blue]"
                )
                rprint(
                    f"[blue]💡 Memory sufficient: {'✅ Yes' if memory_sufficient else '❌ No'}[/blue]"
                )

            # Quantization decision
            if not memory_sufficient and not use_quantization:
                if verbose:
                    rprint("[yellow]⚠️ Limited memory, enabling quantization[/yellow]")
                use_quantization = True
            elif memory_sufficient and use_quantization:
                if verbose:
                    rprint(
                        "[green]🚀 Sufficient memory, disabling quantization for better performance[/green]"
                    )
                use_quantization = False

        # Configure quantization if needed
        if use_quantization:
            if verbose:
                rprint(
                    "[yellow]🔧 Configuring Llama 4 compatible 8-bit quantization...[/yellow]"
                )

            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=[
                        "vision_tower",
                        "multi_modal_projector",
                    ],
                    llm_int8_threshold=6.0,
                )

                if verbose:
                    rprint("[green]✅ Quantization configured[/green]")

            except ImportError:
                if verbose:
                    rprint(
                        "[yellow]⚠️ BitsAndBytesConfig not available, using 16-bit[/yellow]"
                    )
                use_quantization = False
        else:
            if verbose:
                rprint(
                    "[green]🚀 Using 16-bit precision for optimal performance[/green]"
                )

        # PHASE 4: Load Llama 4 model
        if verbose:
            rprint("[cyan]Loading Llama 4 model...[/cyan]")
            if (
                device_map == "auto"
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 1
            ):
                rprint(
                    f"[blue]🔄 Auto-distributing model across {torch.cuda.device_count()} GPUs...[/blue]"
                )

        model = Llama4ForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="sdpa",  # Use SDPA instead of flex_attention (known bug in flex_attention for Llama 4 Scout)
            torch_dtype=torch_dtype_obj,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
            quantization_config=quantization_config if use_quantization else None,
            device_map=device_map,
        )

        # Load processor
        if verbose:
            rprint("[cyan]Loading processor...[/cyan]")

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Set generation parameters
        model.config.max_new_tokens = max_new_tokens

        if verbose:
            rprint("[green]✅ Model and processor loaded successfully![/green]")

    except Exception as e:
        if verbose:
            rprint(f"[red]❌ Failed to load Llama 4 model: {e}[/red]")
        raise

    # PHASE 5: Post-loading diagnostics
    if torch.cuda.is_available() and verbose:
        device_count = torch.cuda.device_count()

        if device_count > 1:
            rprint(
                f"[blue]🔄 Multi-GPU Distribution Analysis ({device_count} GPUs):[/blue]"
            )
            total_allocated = 0
            total_reserved = 0
            total_capacity = 0

            for gpu_id in range(device_count):
                gpu_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                gpu_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                gpu_capacity = (
                    torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                )
                gpu_name = torch.cuda.get_device_name(gpu_id)

                total_allocated += gpu_allocated
                total_reserved += gpu_reserved
                total_capacity += gpu_capacity

                usage_pct = (
                    (gpu_reserved / gpu_capacity) * 100 if gpu_capacity > 0 else 0
                )
                rprint(
                    f"   GPU {gpu_id} ({gpu_name}): {gpu_allocated:.1f}GB/{gpu_capacity:.0f}GB ({usage_pct:.1f}%)"
                )

            rprint(
                f"[blue]📊 Total: {total_allocated:.1f}GB allocated, {total_reserved:.1f}GB reserved, {total_capacity:.0f}GB capacity[/blue]"
            )

            if hasattr(model, "hf_device_map") and model.hf_device_map:
                rprint("[green]✅ Model successfully distributed across GPUs[/green]")
        else:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            rprint("[blue]📊 Single GPU Analysis:[/blue]")
            rprint(f"[blue]   Device: {model.device}[/blue]")
            rprint(f"[magenta]   GPU: {torch.cuda.get_device_name(0)}[/magenta]")
            rprint(
                f"[blue]   Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total_memory:.0f}GB total[/blue]"
            )

        # Model configuration table
        if verbose:
            console = Console()
            table = Table(title="🔧 Llama 4 Model Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_column("Status", style="green")

            table.add_row("Model Path", model_path_obj.name, "✅ Valid")
            table.add_row("Model Type", model_type, "✅ Llama 4")
            table.add_row("Device Placement", str(model.device), "✅ Loaded")

            if use_quantization:
                quant_status = "✅ 8-bit (Memory Optimized)"
                quant_method = "8-bit"
            else:
                quant_status = "✅ 16-bit (Performance Optimized)"
                quant_method = "16-bit"
            table.add_row("Quantization", quant_method, quant_status)

            table.add_row("Data Type", str(torch_dtype), "✅ Recommended")
            table.add_row("Max New Tokens", str(max_new_tokens), "✅ Generation Ready")
            table.add_row("Attention", "sdpa", "✅ Efficient SDPA")

            if torch.cuda.is_available():
                gpu_info = (
                    f"{device_count}x {torch.cuda.get_device_name(0)} ({total_capacity:.0f}GB)"
                    if "total_capacity" in locals()
                    else f"{device_count} GPU(s)"
                )
                gpu_status = (
                    f"✅ {total_capacity:.0f}GB Total"
                    if "total_capacity" in locals()
                    else "✅ Available"
                )
            else:
                gpu_info = "CPU"
                gpu_status = "💻 CPU Mode"
            table.add_row("GPU Configuration", gpu_info, gpu_status)

            param_count = sum(p.numel() for p in model.parameters())
            table.add_row("Model Parameters", f"{param_count:,}", "✅ Loaded")

            console.print(table)

    # PHASE 6: Memory cleanup
    if torch.cuda.is_available() and verbose:
        rprint("[cyan]Performing initial memory cleanup...[/cyan]")
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.device_count() > 1:
            total_allocated = sum(
                torch.cuda.memory_allocated(i) / 1e9
                for i in range(torch.cuda.device_count())
            )
            total_reserved = sum(
                torch.cuda.memory_reserved(i) / 1e9
                for i in range(torch.cuda.device_count())
            )
        else:
            total_allocated = torch.cuda.memory_allocated() / 1e9
            total_reserved = torch.cuda.memory_reserved() / 1e9

        fragmentation = total_reserved - total_allocated
        rprint("🧹 Memory cleanup completed")
        rprint(
            f"💾 Final state: Allocated={total_allocated:.2f}GB, Reserved={total_reserved:.2f}GB, Fragmentation={fragmentation:.2f}GB"
        )

    if verbose:
        rprint("[bold green]🎉 Llama 4 model loading complete![/bold green]")
        quant_info = "8-bit quantization" if use_quantization else "16-bit precision"
        rprint(
            f"[blue]🔧 Llama 4 optimizations active: MoE, SDPA attention, {quant_info}[/blue]"
        )

    return model, processor


# Convenience alias matching the original loader's interface
load_llama_model_robust = load_llama4_model
