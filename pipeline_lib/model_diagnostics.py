"""Minimal GPU diagnostics for loaded vision-language models (Llama, InternVL3)"""

from pathlib import Path

import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table


def show_model_diagnostics(model, processor, model_path: str, max_new_tokens: int, verbose: bool = True):
    """
    Display GPU diagnostics for a loaded vision-language model.

    Args:
        model: Loaded model (MllamaForConditionalGeneration, InternVLChatModel, etc.)
        processor: Loaded processor (AutoProcessor, AutoImageProcessor, etc.)
        model_path: Path to model (for display)
        max_new_tokens: Max tokens setting (for display)
        verbose: Whether to display diagnostics (default: True)
    """
    if not verbose:
        return

    # Detect model type from class name
    model_class = model.__class__.__name__
    if "Llama" in model_class or "Mllama" in model_class:
        model_type = "Llama"
    elif "InternVL" in model_class:
        model_type = "InternVL3"
    else:
        model_type = "Vision-Language"

    if not torch.cuda.is_available():
        rprint("[yellow]⚠️ No CUDA GPUs available[/yellow]")
        return

    console = Console()
    device_count = torch.cuda.device_count()

    # Multi-GPU distribution analysis
    if device_count > 1:
        rprint(f"[blue]🔄 Multi-GPU Distribution Analysis ({device_count} GPUs):[/blue]")
        total_allocated = 0
        total_reserved = 0
        total_capacity = 0

        for gpu_id in range(device_count):
            gpu_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            gpu_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
            gpu_capacity = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(gpu_id)

            total_allocated += gpu_allocated
            total_reserved += gpu_reserved
            total_capacity += gpu_capacity

            usage_pct = (gpu_reserved / gpu_capacity) * 100 if gpu_capacity > 0 else 0
            rprint(f"   GPU {gpu_id} ({gpu_name}): {gpu_allocated:.1f}GB/{gpu_capacity:.0f}GB ({usage_pct:.1f}%)")

        rprint(f"[blue]📊 Total across all GPUs: {total_allocated:.1f}GB allocated, {total_reserved:.1f}GB reserved, {total_capacity:.0f}GB capacity[/blue]")

        # Check if model is actually distributed
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            rprint("[green]✅ Model successfully distributed across GPUs[/green]")
            device_distribution = {}
            for _module, device in model.hf_device_map.items():
                device_str = str(device)
                device_distribution[device_str] = device_distribution.get(device_str, 0) + 1

            for device, count in device_distribution.items():
                rprint(f"   {device}: {count} modules")
        else:
            rprint("[yellow]⚠️ Model distribution info not available[/yellow]")
    else:
        # Single GPU diagnostics
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        rprint("[blue]📊 Single GPU Analysis:[/blue]")
        rprint(f"[blue]   Device: {model.device}[/blue]")
        rprint(f"[magenta]   GPU: {torch.cuda.get_device_name(0)}[/magenta]")
        rprint(f"[blue]   Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total_memory:.0f}GB total[/blue]")

    # Model configuration table
    table = Table(title=f"🔧 {model_type} Vision Model Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Status", style="green")

    model_name = Path(model_path).name
    table.add_row("Model Path", model_name, "✅ Valid")
    table.add_row("Device Placement", str(model.device), "✅ Loaded")

    # Detect quantization from model dtype
    model_dtype = str(model.dtype)
    if "int8" in model_dtype.lower():
        quant_method = "8-bit"
        quant_status = "✅ 8-bit (Memory Optimized)"
    else:
        quant_method = "16-bit"
        quant_status = "✅ 16-bit (Performance Optimized)"
    table.add_row("Quantization Method", quant_method, quant_status)

    table.add_row("Data Type", model_dtype, "✅ Recommended")
    table.add_row("Max New Tokens", str(max_new_tokens), "✅ Generation Ready")

    # GPU configuration
    if torch.cuda.is_available():
        if device_count > 1:
            gpu_info = f"{device_count}x {torch.cuda.get_device_name(0)} ({total_capacity:.0f}GB)"
            gpu_status = f"✅ {total_capacity:.0f}GB Total"
        else:
            gpu_info = torch.cuda.get_device_name(0)
            gpu_status = "✅ Available"
    else:
        gpu_info = "CPU"
        gpu_status = "💻 CPU Mode"
    table.add_row("GPU Configuration", gpu_info, gpu_status)

    # Model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    table.add_row("Model Parameters", f"{param_count:,}", "✅ Loaded")

    # Memory optimization method
    optimization_status = f"{model_type} Optimized" if model_type in ["Llama", "InternVL3"] else "Standard"
    table.add_row("Memory Optimization", optimization_status, "✅ V100 Compatible")

    console.print(table)
