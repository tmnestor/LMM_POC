"""
GPU Memory Optimization Utilities for AWS GPUs (A10G, L4)

This module provides comprehensive GPU memory management strategies optimized for
AWS GPU instances (G5 with A10G, G6 with L4).

Key Features:
    - CUDA memory allocation configuration for reduced fragmentation
    - Advanced memory fragmentation detection and defragmentation
    - Model cache clearing utilities
    - Resilient generation with multiple fallback strategies
    - Emergency model reload and CPU fallback capabilities

Supported GPUs:
    - AWS G5 instances: NVIDIA A10G (24GB VRAM)
    - AWS G6 instances: NVIDIA L4 (24GB VRAM)
"""

import gc
import os
from typing import Any, Optional

import torch


def configure_cuda_memory_allocation(
    verbose: bool = True,
    max_split_size_mb: int = 128,
    cudnn_benchmark: bool = True,
):
    """
    Configure CUDA memory allocation to reduce fragmentation.

    Optimized for AWS GPU instances (G5/A10G, G6/L4).

    Args:
        verbose: Whether to print configuration messages
        max_split_size_mb: Maximum CUDA allocator split size in MB
        cudnn_benchmark: Whether to enable cuDNN benchmarking

    Returns:
        bool: True if configuration was applied, False if running on CPU
    """
    if not torch.cuda.is_available():
        return False

    # IMPORTANT: Clear any existing PYTORCH_CUDA_ALLOC_CONF that might have problematic settings
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in current:
            if verbose:
                print(f"⚠️ Removing problematic PYTORCH_CUDA_ALLOC_CONF: {current}")
            del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    # Detect GPU type for logging
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"

    # Configure CUDA memory allocator block size
    cuda_alloc_config = f"max_split_size_mb:{max_split_size_mb}"

    # Apply the configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_config
    if verbose:
        print(f"🔧 GPU detected: {gpu_name}")
        print(f"🔧 CUDA memory allocation configured: {cuda_alloc_config}")

    # Also set cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Log current CUDA memory state
    try:
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # Multi-GPU: Sum across all devices
            total_allocated = 0.0
            total_reserved = 0.0
            for gpu_id in range(device_count):
                total_allocated += torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                total_reserved += torch.cuda.memory_reserved(gpu_id) / (1024**3)  # GB

            allocated = total_allocated
            reserved = total_reserved
            if verbose:
                print(
                    f"📊 Initial CUDA state (Multi-GPU Total): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB"
                )
        else:
            # Single GPU
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            if verbose:
                print(
                    f"📊 Initial CUDA state: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB"
                )
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not check initial CUDA state: {e}")

    return True


def clear_model_caches(
    model: Any, processor: Optional[Any] = None, verbose: bool = True
):
    """
    Phase 1: Enhanced cache clearing for transformer models.

    Args:
        model: The model to clear caches from
        processor: Optional processor/tokenizer to clear caches from
        verbose: Whether to print cleanup messages
    """
    try:
        if verbose:
            print("🧹 Clearing model caches...")

        # Clear KV cache if it exists
        if hasattr(model, "past_key_values"):
            model.past_key_values = None
            if verbose:
                print("  - Cleared past_key_values")

        # Clear generation cache
        if hasattr(model, "_past_key_values"):
            model._past_key_values = None
            if verbose:
                print("  - Cleared _past_key_values")

        # Clear language model caches (for models with separate language model)
        if hasattr(model, "language_model"):
            lang_model = model.language_model
            if hasattr(lang_model, "past_key_values"):
                lang_model.past_key_values = None
                if verbose:
                    print("  - Cleared language_model cache")

        # Clear vision model caches (for multimodal models)
        if hasattr(model, "vision_model"):
            vision_model = model.vision_model
            # Clear any vision processing caches
            for layer in vision_model.modules():
                if hasattr(layer, "past_key_values"):
                    layer.past_key_values = None

        # Clear processor caches if they exist
        if processor and hasattr(processor, "past_key_values"):
            processor.past_key_values = None
            if verbose:
                print("  - Cleared processor cache")

        # Clear any cached attention masks or position IDs
        for module in model.modules():
            if hasattr(module, "past_key_values"):
                module.past_key_values = None
            if hasattr(module, "_past_key_values"):
                module._past_key_values = None
            if hasattr(module, "attention_mask"):
                if hasattr(module.attention_mask, "data"):
                    module.attention_mask = None

        if verbose:
            print("✅ Model caches cleared")

    except Exception as e:
        if verbose:
            print(f"⚠️ Error clearing caches: {e}")
        # Continue anyway - don't fail the entire process


def detect_memory_fragmentation() -> tuple[float, float, float]:
    """
    Detect GPU memory fragmentation across all GPUs.

    Returns:
        tuple: (allocated_gb, reserved_gb, fragmentation_gb) - totals across all GPUs
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    try:
        device_count = torch.cuda.device_count()

        # For multi-GPU setups, sum memory across all devices
        if device_count > 1:
            total_allocated = 0.0
            total_reserved = 0.0

            for gpu_id in range(device_count):
                total_allocated += torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                total_reserved += torch.cuda.memory_reserved(gpu_id) / (1024**3)  # GB

            allocated = total_allocated
            reserved = total_reserved
        else:
            # Single GPU - use default device
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB

        fragmentation = reserved - allocated
        return allocated, reserved, fragmentation
    except Exception:
        return 0.0, 0.0, 0.0


def handle_memory_fragmentation(
    threshold_gb: float = 1.0, aggressive: bool = True, verbose: bool = True
):
    """
    Handle GPU memory fragmentation with various strategies.

    Args:
        threshold_gb: Fragmentation threshold in GB to trigger cleanup
        aggressive: Whether to use aggressive cleanup strategies
        verbose: Whether to print fragmentation messages
    """
    if not torch.cuda.is_available():
        return

    # Removed V100-specific threshold override - let explicit thresholds be respected
    # V100 with 16GB VRAM can handle normal fragmentation thresholds

    allocated, reserved, fragmentation = detect_memory_fragmentation()

    if verbose:
        print(
            f"🧹 Memory state: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Fragmentation={fragmentation:.2f}GB"
        )

    if fragmentation > threshold_gb:
        if verbose:
            print(
                f"⚠️ FRAGMENTATION DETECTED: {fragmentation:.2f}GB gap (allocated vs reserved)"
            )
            print("🔄 Attempting memory pool reset...")

        # Force memory pool cleanup (aggressive strategy)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Clean up IPC memory

        # PyTorch forum suggestion: Reset memory statistics
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Additional synchronization
        torch.cuda.synchronize()

        allocated_after, reserved_after, fragmentation_after = (
            detect_memory_fragmentation()
        )
        print(
            f"📊 Post-cleanup: Allocated={allocated_after:.2f}GB, Reserved={reserved_after:.2f}GB, Fragmentation={fragmentation_after:.2f}GB"
        )

        if aggressive and fragmentation_after > threshold_gb:
            print(
                "🚨 CRITICAL: High fragmentation persists - attempting aggressive defragmentation"
            )
            aggressive_defragmentation()


def aggressive_defragmentation():
    """
    Perform aggressive memory defragmentation for critical fragmentation issues.

    This is the "nuclear option" for severe memory fragmentation.
    """
    print("☢️ Attempting complete memory pool reset...")

    # Step 1: Clear all caches multiple times
    for _ in range(5):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.synchronize()

    # Step 2: Force memory pool compaction by allocating/deallocating
    try:
        # Allocate small tensors to force pool reorganization
        dummy_tensors = []
        for _ in range(10):
            dummy = torch.zeros(1024, 1024, device="cuda")  # 4MB each
            dummy_tensors.append(dummy)

        # Clear them to force deallocation
        del dummy_tensors
        torch.cuda.empty_cache()
        print("✅ Memory pool reorganization attempted")
    except Exception:
        pass  # Ignore if this fails

    # Final cleanup
    for _ in range(2):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.synchronize()

    final_allocated, final_reserved, final_fragmentation = detect_memory_fragmentation()
    print(
        f"🔧 Final state: Allocated={final_allocated:.2f}GB, Reserved={final_reserved:.2f}GB, Fragmentation={final_fragmentation:.2f}GB"
    )


def comprehensive_memory_cleanup(
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    verbose: bool = True,
    fragmentation_threshold_gb: float = 0.5,
):
    """
    Perform comprehensive memory cleanup including cache clearing and defragmentation.

    Args:
        model: Optional model to clear caches from
        processor: Optional processor to clear caches from
        verbose: Whether to print cleanup messages
        fragmentation_threshold_gb: Fragmentation threshold in GB to trigger cleanup
    """
    # Phase 1: Clear model caches if provided
    if model is not None:
        clear_model_caches(model, processor, verbose=verbose)

    # Phase 2: Multi-pass garbage collection
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        # Force synchronization before cleanup
        torch.cuda.synchronize()

        # Multiple empty_cache calls with synchronization
        for _ in range(2):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Reset memory statistics to prevent allocator confusion
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Check and handle fragmentation
        handle_memory_fragmentation(
            threshold_gb=fragmentation_threshold_gb, aggressive=True, verbose=verbose
        )


def get_available_gpu_memory(device: str = "cuda") -> float:
    """
    Get available GPU memory in GB for batch-size decisions.

    For multi-GPU with ``device_map="auto"``, pixel_values live on GPU 0 but
    the KV cache is distributed across all GPUs.  So the effective memory
    budget is the **minimum** free memory across all GPUs scaled by count —
    not the sum (OOM) and not just GPU 0 (too conservative).

    Args:
        device: Device string.
            - ``"cuda"`` — min free × world_size (safe for batching).
            - ``"total"`` — sum across all GPUs.
            - ``"cuda:N"`` — free memory on GPU N only.

    Returns:
        float: Available memory in GB
    """
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0

    try:
        if device == "total":
            from .robust_gpu_memory import get_total_available_gpu_memory

            return get_total_available_gpu_memory()

        # Specific GPU requested
        if ":" in device:
            device_idx = int(device.split(":")[-1])
            total = torch.cuda.get_device_properties(device_idx).total_memory
            alloc = torch.cuda.memory_allocated(device_idx)
            return (total - alloc) / (1024**3)

        # "cuda" — effective memory for batch_chat() sizing.
        # With device_map="auto", the KV cache is distributed across GPUs,
        # so total batch capacity scales with GPU count.  But any single
        # GPU OOMing kills the call, so the per-GPU bottleneck is min(free).
        # We report min(free) * world_size so the thresholds (calibrated
        # for total memory, e.g. 88GB on 4×L4) still work.
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            total = torch.cuda.get_device_properties(0).total_memory
            alloc = torch.cuda.memory_allocated(0)
            return (total - alloc) / (1024**3)

        min_free = float("inf")
        for gpu_id in range(world_size):
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            alloc = torch.cuda.memory_allocated(gpu_id)
            free = (total - alloc) / (1024**3)
            min_free = min(min_free, free)

        return min_free * world_size
    except Exception as e:
        print(f"⚠️ Could not detect GPU memory: {e}")
        return 24.0  # Final fallback for A10G/L4 (24GB)


def diagnose_gpu_memory_comprehensive(verbose: bool = True) -> dict:
    """
    Comprehensive GPU memory diagnostics using robust detection.

    Args:
        verbose: Enable detailed output

    Returns:
        dict: Complete diagnostic information
    """
    try:
        from .robust_gpu_memory import diagnose_gpu_memory

        return diagnose_gpu_memory(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"⚠️ Robust diagnostics failed: {e}")
        # Fallback to basic diagnostics
        basic_diagnostics = {
            "detection_successful": False,
            "error": str(e),
            "fallback_diagnostics": True,
            "recommendations": [
                "Check robust_gpu_memory.py installation",
                "Verify CUDA availability",
            ],
        }
        return basic_diagnostics


def get_total_gpu_memory_robust() -> float:
    """
    Get total GPU memory across all devices using robust detection.

    Returns:
        float: Total GPU memory in GB
    """
    try:
        from .robust_gpu_memory import get_total_gpu_memory

        return get_total_gpu_memory()
    except Exception as e:
        print(f"⚠️ Robust total memory detection failed: {e}")
        # Fallback to basic detection
        if not torch.cuda.is_available():
            return 0.0

        try:
            total = 0.0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.get_device_properties(i).total_memory / (1024**3)
            return total
        except Exception:
            return 0.0


def optimize_model_for_gpu(
    model: Any, verbose: bool = True, dtype: Optional[torch.dtype] = None
):
    """
    Apply GPU optimizations to a model for AWS instances (A10G, L4).

    Args:
        model: The model to optimize
        verbose: Whether to print optimization messages
        dtype: Model dtype (e.g., torch.bfloat16, torch.float32). Used for accurate messaging.
    """
    if not torch.cuda.is_available():
        return

    # Enable TF32 for faster computation on Ampere+ GPUs (A10G, L4)
    # Note: TF32 only affects float32 operations; bfloat16 uses native tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set model to evaluation mode
    model.eval()

    if verbose:
        # Only mention TF32 when using float32 (TF32 is irrelevant for bfloat16)
        if dtype == torch.bfloat16:
            print("🚀 GPU optimizations applied (bfloat16 tensor cores)")
        elif dtype == torch.float32:
            print("🚀 GPU optimizations applied (TF32 enabled)")
        else:
            print("🚀 GPU optimizations applied")


def clear_gpu_cache(
    verbose: bool = True,
    critical_fragmentation_threshold_gb: float = 1.0,
):
    """
    GPU memory cache clearing optimized for AWS instances (A10G, L4).

    Provides comprehensive GPU memory cleanup with detailed reporting of memory
    states before and after clearing. Includes fragmentation detection.

    Args:
        verbose: Whether to print detailed cleanup messages
        critical_fragmentation_threshold_gb: Threshold in GB to flag fragmentation
    """
    if verbose:
        print("🧹 Starting GPU memory cleanup...")

    # Clear Python garbage collection
    gc.collect()

    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        # Get initial memory stats
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        initial_reserved = torch.cuda.memory_reserved() / 1024**3

        if verbose:
            print(
                f"   📊 Initial GPU memory: {initial_memory:.2f}GB allocated, {initial_reserved:.2f}GB reserved"
            )

        # Empty all caches
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force garbage collection again
        gc.collect()

        # Clear any cached allocator stats
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
            torch.cuda.reset_accumulated_memory_stats()

        # Get final memory stats
        final_memory = torch.cuda.memory_allocated() / 1024**3
        final_reserved = torch.cuda.memory_reserved() / 1024**3

        if verbose:
            print(
                f"   ✅ Final GPU memory: {final_memory:.2f}GB allocated, {final_reserved:.2f}GB reserved"
            )
            print(f"   💾 Memory freed: {initial_memory - final_memory:.2f}GB")

        # Memory fragmentation detection
        fragmentation = final_reserved - final_memory
        if fragmentation > critical_fragmentation_threshold_gb:
            if verbose:
                print(f"   ⚠️ FRAGMENTATION DETECTED: {fragmentation:.2f}GB gap")
    else:
        if verbose:
            print("   ℹ️  No CUDA device available, skipping GPU cache clearing")

    if verbose:
        print("✅ GPU memory cleanup complete")


def emergency_cleanup(verbose: bool = True):
    """
    Emergency GPU memory cleanup for critical OOM recovery.

    Performs aggressive memory cleanup including module reference clearing,
    multi-pass garbage collection, and multiple cache clearing iterations.

    Args:
        verbose: Whether to print cleanup messages
    """
    if verbose:
        print("🚨 Running emergency GPU cleanup...")

    # Try to delete any global model references
    import sys

    for name in list(sys.modules.keys()):
        if "transformers" in name or "torch" in name:
            if hasattr(sys.modules[name], "_model"):
                delattr(sys.modules[name], "_model")

    # Multi-pass cleanup: 3x GC + 2x cache clearing
    for _ in range(3):
        gc.collect()

    for _ in range(2):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final comprehensive cleanup
    clear_gpu_cache(verbose=verbose)

    if verbose:
        print("✅ Emergency cleanup complete")


def cleanup_model_handler(
    handler_name: str = "handler", globals_dict: dict = None, verbose: bool = True
):
    """
    Clean up an existing model handler and free GPU memory.

    This function safely removes a model handler from memory, cleaning up all
    associated resources including the model, tokenizer/processor, and GPU cache.
    Commonly used in notebooks before reinitializing models.

    Args:
        handler_name: Name of the handler variable in globals (default: 'handler')
        globals_dict: Dictionary of global variables (default: None, will use caller's globals)
        verbose: Whether to print cleanup messages

    Returns:
        bool: True if cleanup was performed, False if handler didn't exist

    Example:
        >>> # In a notebook:
        >>> from common.gpu_optimization import cleanup_model_handler
        >>> cleanup_model_handler('handler', globals())
        🧹 Cleaning up existing model instances...
           ✅ Previous model cleaned up
    """
    if verbose:
        print("🧹 Cleaning up any existing model instances...")

    # Get globals dict if not provided
    if globals_dict is None:
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            globals_dict = frame.f_back.f_globals
        else:
            if verbose:
                print("   ⚠️ Could not access globals, skipping cleanup")
            return False

    # Check if handler exists
    if handler_name in globals_dict:
        handler = globals_dict[handler_name]

        # Clean up existing handler
        if hasattr(handler, "processor") and handler.processor:
            if hasattr(handler.processor, "model"):
                del handler.processor.model
            if hasattr(handler.processor, "tokenizer"):
                del handler.processor.tokenizer
            del handler.processor

        # Delete the handler itself
        del globals_dict[handler_name]

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print("   ✅ Previous model cleaned up")
        return True
    else:
        if verbose:
            print(f"   ℹ️ No '{handler_name}' found in globals, nothing to clean up")
        return False
