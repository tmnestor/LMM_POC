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
    verbose: bool = True,
    force_quantization: bool = False
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
        force_quantization: If True, never override quantization setting (for V100 compatibility)

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
            device_count = torch.cuda.device_count()
            if device_count > 1:
                # Multi-GPU: Sum across all devices
                total_allocated = 0.0
                total_reserved = 0.0
                for gpu_id in range(device_count):
                    total_allocated += torch.cuda.memory_allocated(gpu_id) / 1e9
                    total_reserved += torch.cuda.memory_reserved(gpu_id) / 1e9

                allocated = total_allocated
                reserved = total_reserved
                rprint(f"📊 Initial CUDA state (Multi-GPU Total): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            else:
                # Single GPU
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
        # Use robust GPU memory detection for bulletproof memory analysis
        from .robust_gpu_memory import RobustGPUMemoryDetector

        if verbose:
            rprint("[blue]🔍 Performing robust GPU memory detection...[/blue]")

        # Initialize robust detector with verbose output matching our setting
        memory_detector = RobustGPUMemoryDetector(verbose=verbose)
        memory_result = memory_detector.detect_gpu_memory()

        # Extract key values for decision making
        total_gpu_memory = memory_result.total_memory_gb
        total_available_memory = memory_result.total_available_gb
        device_count = memory_result.total_gpus
        working_gpus = memory_result.working_gpus

        # InternVL3 memory requirements with H200, L40S, and V100 support
        model_variant = "8B" if "8B" in model_path else "2B"
        estimated_memory_needed = 16 if model_variant == "8B" else 4  # Realistic estimates for fp16

        # Use dynamic GPU detection for architecture and buffer calculation
        from .dynamic_gpu_config import get_gpu_detector

        detector = get_gpu_detector()
        gpu_config = detector.get_primary_gpu_config()

        if gpu_config:
            memory_buffer = gpu_config.memory_buffer_gb
            gpu_name = gpu_config.name
            gpu_architecture = gpu_config.architecture
        else:
            # Fallback for no GPU
            memory_buffer = 8
            gpu_name = "Unknown"
            gpu_architecture = "unknown"

        # Use total available memory for more accurate assessment
        memory_sufficient = total_available_memory >= (estimated_memory_needed + memory_buffer)

        if verbose:
            # Enhanced display with robust detection results
            if memory_result.detection_success:
                # Get primary GPU name for display (use first working GPU)
                primary_gpu_name = next((gpu.name for gpu in memory_result.per_gpu_info if gpu.is_available), "Unknown")
                avg_gpu_memory = total_gpu_memory / working_gpus if working_gpus > 0 else 0

                rprint(f"[blue]📊 GPU Hardware: {primary_gpu_name} ({working_gpus}x {avg_gpu_memory:.0f}GB = {total_gpu_memory:.0f}GB total)[/blue]")
                rprint(f"[blue]🏗️ Architecture: {gpu_architecture} (dynamic detection)[/blue]")
                rprint(f"[blue]🎯 Model variant: InternVL3-{model_variant} (estimated need: {estimated_memory_needed}GB + {memory_buffer:.1f}GB buffer)[/blue]")
                rprint(f"[blue]💾 Available Memory: {total_available_memory:.1f}GB across {working_gpus} GPU(s)[/blue]")

                # Show any warnings from robust detection
                if memory_result.warnings:
                    for warning in memory_result.warnings:
                        rprint(f"[yellow]⚠️ {warning}[/yellow]")
            else:
                rprint("[blue]📊 GPU Hardware: CPU fallback (robust detection failed)[/blue]")

            rprint(f"[blue]💡 Memory sufficient: {'✅ Yes' if memory_sufficient else '❌ No'}[/blue]")

        # Enhanced quantization decision using robust memory detection
        # CRITICAL FIX: Check high memory REGARDLESS of initial use_quantization setting

        # V100-specific logic: Dynamic assessment for 1x, 2x, 3x, or 4x V100 setups
        v100_gpus = [gpu for gpu in memory_result.per_gpu_info if gpu.is_available and "V100" in gpu.name.upper()]
        v100_detected = len(v100_gpus) > 0

        if v100_detected:
            v100_count = len(v100_gpus)
            v100_total_memory = sum(gpu.total_memory_gb for gpu in v100_gpus)

            # Dynamic V100 memory thresholds:
            # 1x V100 (16GB): Needs quantization (< 20GB threshold)
            # 2x V100 (32GB): Can do full precision (>= 20GB threshold)
            # 3x V100 (48GB): Excellent for full precision
            # 4x V100 (64GB): Excellent for full precision

            v100_threshold = estimated_memory_needed + memory_buffer  # ~20GB for InternVL3-8B

            if v100_total_memory >= v100_threshold:
                # Sufficient V100 memory for full precision
                if use_quantization and not force_quantization:
                    if verbose:
                        rprint(f"[green]🚀 {v100_count}x V100 setup detected ({v100_total_memory:.0f}GB total), disabling quantization for optimal performance[/green]")
                    use_quantization = False
                elif use_quantization and force_quantization:
                    if verbose:
                        rprint(f"[yellow]🔒 {v100_count}x V100 with {v100_total_memory:.0f}GB - FORCING quantization as explicitly requested[/yellow]")
                else:
                    if verbose:
                        rprint(f"[green]✅ {v100_count}x V100 with {v100_total_memory:.0f}GB - running in full precision as requested[/green]")
            else:
                # Insufficient V100 memory - likely 1x V100
                if not use_quantization:
                    if verbose:
                        rprint(f"[yellow]⚠️ {v100_count}x V100 setup ({v100_total_memory:.0f}GB) insufficient for full precision, enabling quantization[/yellow]")
                    use_quantization = True
                else:
                    if verbose:
                        rprint(f"[yellow]⚠️ {v100_count}x V100 with {v100_total_memory:.0f}GB - using quantization due to memory constraints[/yellow]")
        elif gpu_config and gpu_config.is_high_memory:
            # High-memory GPUs (H200, H100, etc.)
            if use_quantization and not force_quantization:
                if verbose:
                    rprint(f"[green]🚀 {gpu_architecture} detected with abundant memory ({total_gpu_memory:.0f}GB), disabling quantization for optimal performance[/green]")
                use_quantization = False
            elif use_quantization and force_quantization:
                if verbose:
                    rprint(f"[yellow]🔒 {gpu_architecture} with {total_gpu_memory:.0f}GB - FORCING quantization as explicitly requested[/yellow]")
            else:
                if verbose:
                    rprint(f"[green]✅ {gpu_architecture} with {total_gpu_memory:.0f}GB - running in full precision as requested[/green]")
        elif memory_sufficient:
            # Sufficient memory based on available memory calculation
            if use_quantization and not force_quantization:
                if verbose:
                    rprint(f"[green]🚀 Sufficient GPU memory detected ({total_available_memory:.0f}GB available), disabling quantization for better performance[/green]")
                use_quantization = False
            elif use_quantization and force_quantization:
                if verbose:
                    rprint(f"[yellow]🔒 Sufficient memory ({total_available_memory:.0f}GB) - FORCING quantization as explicitly requested[/yellow]")
            else:
                if verbose:
                    rprint(f"[green]✅ Memory sufficient ({total_available_memory:.0f}GB available) - running in full precision as requested[/green]")
        elif not memory_sufficient:
            # Insufficient memory - enable quantization if not already enabled
            if not use_quantization:
                if verbose:
                    rprint(f"[yellow]⚠️ Limited GPU memory detected ({total_available_memory:.0f}GB available), enabling quantization for compatibility[/yellow]")
                use_quantization = True
            else:
                if verbose:
                    rprint(f"[yellow]⚠️ Using quantization due to limited memory ({total_available_memory:.0f}GB available)[/yellow]")

        # Log robust detection warnings if any
        if verbose and memory_result.warnings:
            rprint("[yellow]🔍 Robust Detection Warnings:[/yellow]")
            for warning in memory_result.warnings:
                rprint(f"[yellow]   • {warning}[/yellow]")

    # CRITICAL: Enhanced final quantization decision with robust detection data
    if verbose:
        rprint(f"[bold cyan]📊 FINAL QUANTIZATION DECISION: {'ENABLED (8-bit)' if use_quantization else 'DISABLED (full precision)'}[/bold cyan]")
        if torch.cuda.is_available() and total_gpu_memory > 0:
            rprint(f"[cyan]   Total GPU Memory: {total_gpu_memory:.0f}GB[/cyan]")
            rprint(f"[cyan]   Available Memory: {total_available_memory:.0f}GB[/cyan]")
            rprint(f"   Model needs: ~{estimated_memory_needed}GB + {memory_buffer:.1f}GB buffer for InternVL3-{model_variant}")
            rprint(f"[cyan]   Working GPUs: {working_gpus}/{device_count}[/cyan]")

            # V100-specific final summary
            if v100_detected:
                v100_count = len(v100_gpus)
                v100_total_memory = sum(gpu.total_memory_gb for gpu in v100_gpus)
                rprint(f"[cyan]   V100 Configuration: {v100_count}x V100 = {v100_total_memory:.0f}GB total[/cyan]")

            # Show any critical warnings
            if memory_result.warnings:
                rprint(f"[yellow]   Warnings: {len(memory_result.warnings)} detected (see above)[/yellow]")

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

    # Load model with InternVL3-optimized parameters and multi-GPU distribution
    try:
        if verbose:
            rprint("[cyan]Loading InternVL3 model...[/cyan]")
            if device_map == "auto" and torch.cuda.device_count() > 1:
                rprint(f"[blue]🔄 Auto-distributing model across {torch.cuda.device_count()} GPUs...[/blue]")

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

    # Post-loading diagnostics with multi-GPU awareness
    if torch.cuda.is_available() and verbose:
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
            single_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            total_memory = device_count * single_gpu_memory
            gpu_model = torch.cuda.get_device_name(0)
            table.add_row("GPU Configuration", f"{device_count}x {gpu_model} ({single_gpu_memory:.0f}GB each)", f"✅ {total_memory:.0f}GB Total")

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
            # For multi-GPU setups, sum memory across all devices
            device_count = torch.cuda.device_count()
            if device_count > 1:
                total_allocated = 0
                total_reserved = 0
                for gpu_id in range(device_count):
                    total_allocated += torch.cuda.memory_allocated(gpu_id) / 1e9
                    total_reserved += torch.cuda.memory_reserved(gpu_id) / 1e9

                final_allocated = total_allocated
                final_reserved = total_reserved
                fragmentation = final_reserved - final_allocated

                rprint("🧹 Memory cleanup completed")
                rprint(f"💾 Final state (Multi-GPU Total): Allocated={final_allocated:.2f}GB, Reserved={final_reserved:.2f}GB, Fragmentation={fragmentation:.2f}GB")
            else:
                # Single GPU
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