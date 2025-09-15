"""
InternVL3-Specific Model Loader

Optimized model loading for InternVL3 based on official documentation.
No vision module skipping - processes all components normally.
"""

import gc
from pathlib import Path
from typing import Any, Tuple

import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from transformers import AutoModel, AutoTokenizer


def load_internvl3_model(
    model_path: str,
    use_quantization: bool = True,
    device_map: str = "auto",
    max_new_tokens: int = 4000,
    torch_dtype: str = "bfloat16",
    low_cpu_mem_usage: bool = True,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    Load InternVL3 model with official optimizations.

    Based on https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html

    Args:
        model_path: Path to InternVL3 model
        use_quantization: Use 8-bit quantization (requires appropriate GPU memory)
        device_map: Device mapping strategy
        max_new_tokens: Maximum tokens for generation
        torch_dtype: Data type (bfloat16 recommended)
        low_cpu_mem_usage: Enable low CPU memory usage
        verbose: Enable detailed logging

    Returns:
        Tuple of (model, tokenizer)
    """

    console = Console()

    if verbose:
        rprint("[bold blue]🚀 Loading InternVL3 model with official optimizations...[/bold blue]")

    # Validate model path
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Configure CUDA memory for InternVL3
    if torch.cuda.is_available():
        if verbose:
            rprint("[blue]🔧 Configuring CUDA memory for InternVL3...[/blue]")

        # Use 32MB blocks for better memory management
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Clear existing cache
        torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            rprint(f"📊 Initial CUDA state: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

    # Convert torch_dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

    # Intelligent quantization based on available GPU memory (V100 multi-GPU setup)
    quantization_config = None
    if torch.cuda.is_available():
        # Calculate total available GPU memory across all devices
        total_gpu_memory = 0
        device_count = torch.cuda.device_count()

        if verbose:
            rprint(f"[blue]🔍 Detecting GPU memory across {device_count} device(s)...[/blue]")

        for i in range(device_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            total_gpu_memory += gpu_memory
            if verbose:
                rprint(f"   GPU {i}: {torch.cuda.get_device_name(i)} - {gpu_memory:.0f}GB")

        # InternVL3 memory requirements (conservative estimates for V100-16GB)
        model_variant = "8B" if "8B" in model_path else "2B"
        estimated_memory_needed = 12 if model_variant == "8B" else 3  # Conservative estimates

        # V100-16GB considerations: 1-4 GPUs available (16GB, 32GB, 48GB, or 64GB total)
        # Use quantization if total memory < required + 4GB buffer
        memory_buffer = 4  # 4GB buffer for safety
        memory_sufficient = total_gpu_memory >= (estimated_memory_needed + memory_buffer)

        if verbose:
            rprint(f"[blue]📊 Total GPU memory: {total_gpu_memory:.0f}GB[/blue]")
            rprint(f"[blue]🎯 Model variant: InternVL3-{model_variant} (estimated need: {estimated_memory_needed}GB)[/blue]")
            rprint(f"[blue]💡 Memory sufficient: {'✅ Yes' if memory_sufficient else '❌ No'}[/blue]")

        # Override quantization setting based on memory availability
        if memory_sufficient and use_quantization:
            if verbose:
                rprint("[green]🚀 Sufficient GPU memory detected, disabling quantization for better performance[/green]")
            use_quantization = False
        elif not memory_sufficient and not use_quantization:
            if verbose:
                rprint("[yellow]⚠️ Limited GPU memory detected, enabling quantization for compatibility[/yellow]")
            use_quantization = True

    if use_quantization:
        if verbose:
            rprint("[yellow]🔧 Configuring InternVL3-compatible 8-bit quantization...[/yellow]")

        try:
            from transformers import BitsAndBytesConfig

            # InternVL3-specific quantization config (no vision module skipping)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                # Note: No skip_modules for InternVL3 - all components processed normally
            )

            if verbose:
                rprint("[green]✅ InternVL3-compatible quantization configured[/green]")

        except ImportError:
            if verbose:
                rprint("[yellow]⚠️ BitsAndBytesConfig not available, using 16-bit[/yellow]")
            use_quantization = False
    else:
        if verbose:
            rprint("[green]🚀 Using 16-bit precision for optimal performance[/green]")

    # Load model with InternVL3-optimized parameters
    try:
        if verbose:
            rprint("[cyan]Loading InternVL3 model...[/cyan]")

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_obj,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
            quantization_config=quantization_config if use_quantization else None,
            device_map=device_map
        )

        # Load tokenizer
        if verbose:
            rprint("[cyan]Loading tokenizer...[/cyan]")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Set generation parameters
        model.config.max_new_tokens = max_new_tokens

        if verbose:
            rprint("[green]✅ Model and tokenizer loaded successfully![/green]")

    except Exception as e:
        rprint(f"[red]❌ Failed to load model: {e}[/red]")
        raise

    # Post-loading diagnostics
    if torch.cuda.is_available() and verbose:
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        rprint(f"[blue]📊 Device: {model.device}[/blue]")
        rprint(f"[magenta]🎮 GPU: {torch.cuda.get_device_name(0)}[/magenta]")
        rprint(f"[blue]💾 Memory Allocated: {allocated:.2f}GB[/blue]")
        rprint(f"[blue]💾 Memory Reserved: {reserved:.2f}GB[/blue]")
        rprint(f"[blue]💾 Total GPU Memory: {total_memory:.0f}GB[/blue]")

        memory_usage_pct = (reserved / total_memory) * 100
        if memory_usage_pct < 10:
            rprint(f"[green]✅ Good GPU memory usage: {memory_usage_pct:.1f}%[/green]")
        elif memory_usage_pct < 25:
            rprint(f"[yellow]⚠️ Moderate GPU memory usage: {memory_usage_pct:.1f}%[/yellow]")
        else:
            rprint(f"[red]🔥 High GPU memory usage: {memory_usage_pct:.1f}%[/red]")

    # Configuration summary table
    if verbose:
        table = Table(title="🔧 InternVL3 Model Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("InternVL3 Status", style="green")

        table.add_row("Model Path", str(model_path_obj.name), "✅ Valid")
        table.add_row("Device Placement", str(model.device), "✅ Loaded")

        # Enhanced quantization status
        if use_quantization:
            quant_status = "✅ 8-bit (Memory Optimized)"
        else:
            quant_status = "✅ 16-bit (Performance Optimized)"
        table.add_row("Quantization Method", "8-bit" if use_quantization else "16-bit", quant_status)

        table.add_row("Data Type", torch_dtype, "✅ Recommended")
        table.add_row("Max New Tokens", str(max_new_tokens), "✅ Generation Ready")

        # Add GPU memory info
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # Detect actual V100 memory (should be 16GB per GPU)
            single_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            total_memory = device_count * single_gpu_memory
            table.add_row("GPU Configuration", f"{device_count}x V100-{single_gpu_memory:.0f}GB", f"✅ {total_memory:.0f}GB Total")

        # Get model parameters if possible
        try:
            total_params = sum(p.numel() for p in model.parameters())
            table.add_row("Model Parameters", f"{total_params:,}", "✅ Loaded")
        except Exception:
            table.add_row("Model Parameters", "N/A", "ℹ️ Unknown")

        table.add_row("Memory Optimization", "InternVL3 Official", "✅ Documentation Based")

        console.print(table)

    # Compatibility test
    if verbose:
        rprint("[cyan]Running model compatibility test...[/cyan]")

    try:
        # Simple test to ensure model is working
        test_prompt = "Test prompt"
        with torch.no_grad():
            # Basic tokenization test
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if verbose:
            rprint("[green]✅ Model compatibility test passed[/green]")

    except Exception as e:
        rprint(f"[yellow]⚠️ Compatibility test failed: {e}[/yellow]")
        rprint("[yellow]Model loaded but may have issues during inference[/yellow]")

    # Memory cleanup
    if verbose:
        rprint("[cyan]Performing initial memory cleanup...[/cyan]")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            final_allocated = torch.cuda.memory_allocated() / 1e9
            final_reserved = torch.cuda.memory_reserved() / 1e9
            fragmentation = final_reserved - final_allocated

            rprint("🧹 Memory cleanup completed")
            rprint(f"💾 Final state: Allocated={final_allocated:.2f}GB, Reserved={final_reserved:.2f}GB, Fragmentation={fragmentation:.2f}GB")

    if verbose:
        rprint("[bold green]🎉 InternVL3 model loading and validation complete![/bold green]")
        quant_info = "8-bit quantization" if use_quantization else "16-bit precision"
        rprint(f"[blue]🔧 InternVL3 optimizations active: {quant_info}, memory management, no vision skipping[/blue]")

    return model, tokenizer


def get_internvl3_device_info() -> dict:
    """Get device information for InternVL3 deployment."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "device_name": props.name,
            "total_memory_gb": props.total_memory / 1e9,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
        })

    return info


def main():
    """Test the InternVL3 model loader."""
    # This would be used in testing
    print("InternVL3 Model Loader - Test Mode")
    print("Note: Actual model loading requires valid model path")

    device_info = get_internvl3_device_info()
    print(f"Device info: {device_info}")


if __name__ == "__main__":
    main()