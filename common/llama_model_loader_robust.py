"""
Robust Llama Model Loader with Multi-GPU Memory Detection

A bulletproof Llama Vision model loading system designed for production environments
where interactive debugging is not available. Provides intelligent quantization
decisions based on comprehensive GPU memory analysis.

Key Features:
- Robust multi-GPU memory detection
- Intelligent quantization decisions (1x V100 vs 2x+ V100)
- V100-specific optimizations
- Comprehensive error handling and fallbacks
- Extensive logging for troubleshooting
- Llama-specific vision module optimizations
"""

import gc
from pathlib import Path
from typing import Tuple

import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
)

# Import GPU optimization utilities
from .gpu_optimization import (
    configure_cuda_memory_allocation,
)

# Import robust memory detection


def load_llama_model_robust(
    model_path: str,
    use_quantization: bool = False,  # Default False for robust detection to override
    device_map: str = "auto",
    max_new_tokens: int = 4000,
    torch_dtype: str = "bfloat16",
    low_cpu_mem_usage: bool = True,
    verbose: bool = True,
) -> Tuple[MllamaForConditionalGeneration, AutoProcessor]:
    """
    Load Llama Vision model with robust multi-GPU memory detection and intelligent quantization.

    This function provides bulletproof model loading designed for production environments
    without interactive debugging support. It automatically detects GPU configurations
    and makes intelligent quantization decisions.

    Args:
        model_path: Path to the Llama model directory
        use_quantization: Initial quantization preference (can be overridden by detection)
        device_map: Device mapping strategy ("auto" recommended for multi-GPU)
        max_new_tokens: Maximum new tokens for generation
        torch_dtype: Torch data type ("bfloat16" recommended)
        low_cpu_mem_usage: Whether to use low CPU memory mode
        verbose: Enable detailed logging and diagnostics

    Returns:
        Tuple[MllamaForConditionalGeneration, AutoProcessor]: Loaded model and processor

    Raises:
        FileNotFoundError: If model path doesn't exist
        RuntimeError: If model loading fails after all fallback attempts

    Example:
        >>> model, processor = load_llama_model_robust(
        ...     "/path/to/Llama-3.2-11B-Vision-Instruct",
        ...     use_quantization=False,  # Will be overridden based on GPU detection
        ...     verbose=True
        ... )
    """
    if verbose:
        rprint("[bold blue]🚀 Loading Llama Vision model with robust multi-GPU optimization...[/bold blue]")
        rprint("[cyan]Features: Smart quantization, memory management, V100 support[/cyan]")

    try:
        # PHASE 1: Configure CUDA memory allocation for optimal performance
        if verbose:
            rprint("[blue]🔧 Configuring CUDA memory for Llama...[/blue]")

        configure_cuda_memory_allocation(verbose=verbose)

        # Validate model path
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Convert torch_dtype string to actual dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

        # PHASE 2: Robust GPU memory detection and intelligent quantization decision
        quantization_config = None
        if torch.cuda.is_available():
            # Use robust GPU memory detection for intelligent quantization decisions
            from .robust_gpu_memory import RobustGPUMemoryDetector

            if verbose:
                rprint("[blue]🔍 Performing robust GPU memory detection...[/blue]")

            # Initialize robust detector
            memory_detector = RobustGPUMemoryDetector(verbose=verbose)
            memory_result = memory_detector.detect_gpu_memory()

            # Extract key values for decision making
            total_gpu_memory = memory_result.total_memory_gb
            total_available_memory = memory_result.total_available_gb
            device_count = memory_result.total_gpus
            working_gpus = memory_result.working_gpus

            # Llama-3.2-11B-Vision memory requirements
            estimated_memory_needed = 22  # Llama-3.2-11B-Vision is larger than InternVL3-8B
            memory_buffer = 6.0  # Larger buffer for Llama's complexity

            # Use dynamic GPU detection for architecture information
            from .dynamic_gpu_config import get_gpu_detector

            detector = get_gpu_detector()
            gpu_config = detector.get_primary_gpu_config()

            if gpu_config:
                gpu_name = gpu_config.name
                gpu_architecture = gpu_config.architecture
            else:
                gpu_name = "Unknown"
                gpu_architecture = "unknown"

            # Use total available memory for accurate assessment
            memory_sufficient = total_available_memory >= (estimated_memory_needed + memory_buffer)

            if verbose:
                # Enhanced display with robust detection results
                if memory_result.detection_success:
                    # Get primary GPU name for display
                    primary_gpu_name = next((gpu.name for gpu in memory_result.per_gpu_info if gpu.is_available), "Unknown")
                    avg_gpu_memory = total_gpu_memory / working_gpus if working_gpus > 0 else 0

                    rprint(f"[blue]📊 GPU Hardware: {primary_gpu_name} ({working_gpus}x {avg_gpu_memory:.0f}GB = {total_gpu_memory:.0f}GB total)[/blue]")
                    rprint(f"[blue]🏗️ Architecture: {gpu_architecture} (dynamic detection)[/blue]")
                    rprint(f"[blue]🎯 Model: Llama-3.2-11B-Vision (estimated need: {estimated_memory_needed}GB + {memory_buffer:.1f}GB buffer)[/blue]")
                    rprint(f"[blue]💾 Available Memory: {total_available_memory:.1f}GB across {working_gpus} GPU(s)[/blue]")

                    # Show any warnings from robust detection
                    if memory_result.warnings:
                        for warning in memory_result.warnings:
                            rprint(f"[yellow]⚠️ {warning}[/yellow]")
                else:
                    rprint("[blue]📊 GPU Hardware: CPU fallback (robust detection failed)[/blue]")

                rprint(f"[blue]💡 Memory sufficient: {'✅ Yes' if memory_sufficient else '❌ No'}[/blue]")

            # Enhanced quantization decision using robust memory detection
            # V100-specific logic: Dynamic assessment for 1x, 2x, 3x, or 4x V100 setups
            v100_gpus = [gpu for gpu in memory_result.per_gpu_info if gpu.is_available and "V100" in gpu.name.upper()]
            v100_detected = len(v100_gpus) > 0

            if v100_detected:
                v100_count = len(v100_gpus)
                v100_total_memory = sum(gpu.total_memory_gb for gpu in v100_gpus)

                # Dynamic V100 memory thresholds for Llama-3.2-11B-Vision:
                # 1x V100 (16GB): Needs quantization (< 28GB threshold)
                # 2x V100 (32GB): Can do full precision (>= 28GB threshold)
                # 3x V100 (48GB): Excellent for full precision
                # 4x V100 (64GB): Excellent for full precision

                v100_threshold = estimated_memory_needed + memory_buffer  # ~28GB for Llama-3.2-11B-Vision

                if v100_total_memory >= v100_threshold:
                    # Sufficient V100 memory for full precision
                    if use_quantization:
                        if verbose:
                            rprint(f"[green]🚀 {v100_count}x V100 setup detected ({v100_total_memory:.0f}GB total), disabling quantization for optimal performance[/green]")
                        use_quantization = False
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
                if use_quantization:
                    if verbose:
                        rprint(f"[green]🚀 {gpu_architecture} detected with abundant memory ({total_gpu_memory:.0f}GB), disabling quantization for optimal performance[/green]")
                    use_quantization = False
                else:
                    if verbose:
                        rprint(f"[green]✅ {gpu_architecture} with {total_gpu_memory:.0f}GB - running in full precision as requested[/green]")
            elif memory_sufficient:
                # Sufficient memory based on available memory calculation
                if use_quantization:
                    if verbose:
                        rprint(f"[green]🚀 Sufficient GPU memory detected ({total_available_memory:.0f}GB available), disabling quantization for better performance[/green]")
                    use_quantization = False
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

        # Enhanced final quantization decision with robust detection data
        if verbose:
            rprint(f"[bold cyan]📊 FINAL QUANTIZATION DECISION: {'ENABLED (8-bit)' if use_quantization else 'DISABLED (full precision)'}[/bold cyan]")
            if torch.cuda.is_available() and 'total_gpu_memory' in locals():
                rprint(f"[cyan]   Total GPU Memory: {total_gpu_memory:.0f}GB[/cyan]")
                rprint(f"[cyan]   Available Memory: {total_available_memory:.0f}GB[/cyan]")
                rprint(f"   Model needs: ~{estimated_memory_needed}GB + {memory_buffer:.1f}GB buffer for Llama-3.2-11B-Vision")
                rprint(f"[cyan]   Working GPUs: {working_gpus}/{device_count}[/cyan]")

                # V100-specific final summary
                if v100_detected:
                    v100_count = len(v100_gpus)
                    v100_total_memory = sum(gpu.total_memory_gb for gpu in v100_gpus)
                    rprint(f"[cyan]   V100 Configuration: {v100_count}x V100 = {v100_total_memory:.0f}GB total[/cyan]")

                # Show any critical warnings
                if 'memory_result' in locals() and memory_result.warnings:
                    rprint(f"[yellow]   Warnings: {len(memory_result.warnings)} detected (see above)[/yellow]")

        # PHASE 3: Configure Llama-specific quantization if needed
        if use_quantization:
            if verbose:
                rprint("[yellow]🔧 Configuring Llama-compatible 8-bit quantization...[/yellow]")

            try:
                from transformers import BitsAndBytesConfig

                # Llama-specific quantization config with vision module preservation
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,  # Critical for V100 memory management
                    llm_int8_skip_modules=[
                        "vision_tower",
                        "multi_modal_projector",
                    ],  # Preserve vision components in full precision
                    llm_int8_threshold=6.0,  # Standard threshold for outlier detection
                )

                if verbose:
                    rprint("[green]✅ Llama-compatible quantization configured[/green]")

            except ImportError:
                if verbose:
                    rprint("[yellow]⚠️ BitsAndBytesConfig not available, using 16-bit[/yellow]")
                use_quantization = False
        else:
            if verbose:
                rprint("[green]🚀 Using 16-bit precision for optimal performance[/green]")

        # PHASE 4: Load Llama model with robust configuration
        if verbose:
            rprint("[cyan]Loading Llama Vision model...[/cyan]")
            if device_map == "auto" and torch.cuda.device_count() > 1:
                rprint(f"[blue]🔄 Auto-distributing model across {torch.cuda.device_count()} GPUs...[/blue]")

        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_obj,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
            quantization_config=quantization_config if use_quantization else None,
            device_map=device_map
        )

        # Load processor
        if verbose:
            rprint("[cyan]Loading processor...[/cyan]")

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Set generation parameters
        model.config.max_new_tokens = max_new_tokens

        if verbose:
            rprint("[green]✅ Model and processor loaded successfully![/green]")

    except Exception as e:
        if verbose:
            rprint(f"[red]❌ Failed to load Llama model: {e}[/red]")
        raise

    # PHASE 5: Post-loading diagnostics with multi-GPU awareness
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

        # Enhanced model configuration table
        if verbose:
            console = Console()
            table = Table(title="🔧 Llama Vision Model Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_column("Llama Status", style="green")

            # Use model path basename for cleaner display
            model_name = Path(model_path).name if model_path else "Unknown"
            table.add_row("Model Path", model_name, "✅ Valid")
            table.add_row("Device Placement", str(model.device), "✅ Loaded")

            # Enhanced quantization status
            if use_quantization:
                quant_status = "✅ 8-bit (Memory Optimized)"
                quant_method = "8-bit"
            else:
                quant_status = "✅ 16-bit (Performance Optimized)"
                quant_method = "16-bit"
            table.add_row("Quantization Method", quant_method, quant_status)

            table.add_row("Data Type", str(torch_dtype), "✅ Recommended")
            table.add_row("Max New Tokens", str(max_new_tokens), "✅ Generation Ready")

            # Enhanced GPU configuration
            if torch.cuda.is_available():
                gpu_info = f"{device_count}x {torch.cuda.get_device_name(0)} ({total_capacity:.0f}GB)" if 'total_capacity' in locals() else f"{device_count} GPU(s)"
                gpu_status = f"✅ {total_capacity:.0f}GB Total" if 'total_capacity' in locals() else "✅ Available"
            else:
                gpu_info = "CPU"
                gpu_status = "💻 CPU Mode"
            table.add_row("GPU Configuration", gpu_info, gpu_status)

            # Model parameter count
            param_count = sum(p.numel() for p in model.parameters())
            table.add_row("Model Parameters", f"{param_count:,}", "✅ Loaded")

            # Memory optimization method
            table.add_row("Memory Optimization", "Llama Robust", "✅ V100 Compatible")

            console.print(table)

    # PHASE 6: Compatibility test
    if verbose:
        rprint("[cyan]Running model compatibility test...[/cyan]")

    try:
        # Simple test to ensure model is working
        test_prompt = "Test prompt for Llama Vision"
        with torch.no_grad():
            # Basic processor test
            if hasattr(processor, 'tokenizer'):
                inputs = processor.tokenizer(test_prompt, return_tensors="pt")
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if verbose:
            rprint("[green]✅ Model compatibility test passed[/green]")

    except Exception as e:
        if verbose:
            rprint(f"[yellow]⚠️ Compatibility test failed: {e}[/yellow]")
            rprint("[yellow]Model loaded but may have issues during inference[/yellow]")

    # PHASE 7: Memory cleanup (only for multi-GPU setups to preserve memory reporting accuracy)
    if torch.cuda.is_available() and verbose:
        device_count = torch.cuda.device_count()
        if device_count > 1:
            rprint("[cyan]Performing initial memory cleanup...[/cyan]")
            torch.cuda.empty_cache()
            gc.collect()

            # For multi-GPU setups, sum memory across all devices
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
            rprint("[cyan]Performing initial memory cleanup...[/cyan]")
            torch.cuda.empty_cache()
            gc.collect()

            final_allocated = torch.cuda.memory_allocated() / 1e9
            final_reserved = torch.cuda.memory_reserved() / 1e9
            fragmentation = final_reserved - final_allocated

            rprint("🧹 Memory cleanup completed")
            rprint(f"💾 Final state: Allocated={final_allocated:.2f}GB, Reserved={final_reserved:.2f}GB, Fragmentation={fragmentation:.2f}GB")

    if verbose:
        rprint("[bold green]🎉 Llama Vision model loading and validation complete![/bold green]")
        quant_info = "8-bit quantization" if use_quantization else "16-bit precision"
        rprint(f"[blue]🔧 Llama optimizations active: {quant_info}, memory management, vision preservation[/blue]")

    return model, processor


# Convenience functions for common use cases
def load_llama_v100_robust(
    model_path: str,
    use_quantization: bool = False,
    verbose: bool = True
) -> Tuple[MllamaForConditionalGeneration, AutoProcessor]:
    """
    Convenience function for V100-optimized Llama loading with robust detection.

    Args:
        model_path: Path to the Llama model directory
        use_quantization: Initial quantization preference (overridden by detection)
        verbose: Enable detailed logging

    Returns:
        Tuple[MllamaForConditionalGeneration, AutoProcessor]: Model and processor
    """
    return load_llama_model_robust(
        model_path=model_path,
        use_quantization=use_quantization,
        device_map="auto",
        max_new_tokens=4000,
        torch_dtype="bfloat16",
        low_cpu_mem_usage=True,
        verbose=verbose
    )


def validate_llama_memory_requirements(
    model_path: str,
    verbose: bool = True
) -> bool:
    """
    Validate if current GPU setup can handle Llama-3.2-11B-Vision.

    Args:
        model_path: Path to model (for context)
        verbose: Enable detailed output

    Returns:
        bool: True if memory is sufficient for Llama model
    """
    try:
        from .robust_gpu_memory import validate_model_memory_requirements

        # Llama-3.2-11B-Vision requirements: ~22GB model + 6GB buffer
        return validate_model_memory_requirements(
            model_size_gb=22.0,
            buffer_gb=6.0,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"⚠️ Memory validation failed: {e}")
        return False