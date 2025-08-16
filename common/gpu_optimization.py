"""
GPU Memory Optimization Utilities for V100 and Other GPUs

This module provides comprehensive GPU memory management strategies optimized for
older GPUs like V100 that have limited memory and fragmentation issues.

Key Features:
    - CUDA memory allocation configuration for reduced fragmentation
    - Advanced memory fragmentation detection and defragmentation
    - Model cache clearing utilities
    - Resilient generation with multiple fallback strategies
    - Emergency model reload and CPU fallback capabilities

Based on insights from:
    - PyTorch forums on CUDA OOM issues
    - worldversant.com memory management articles
    - V100 GPU optimization best practices
"""

import gc
import os
from typing import Any, Dict, Optional

import torch


def configure_cuda_memory_allocation():
    """
    Configure CUDA memory allocation to reduce fragmentation (PyTorch forums insights).

    Based on: https://discuss.pytorch.org/t/keep-getting-cuda-oom-error-with-pytorch-failing-to-allocate-all-free-memory/133896

    Returns:
        bool: True if configuration was applied, False if running on CPU
    """
    if not torch.cuda.is_available():
        return False

    # IMPORTANT: Clear any existing PYTORCH_CUDA_ALLOC_CONF that might have problematic settings
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in current:
            print(f"⚠️ Removing problematic PYTORCH_CUDA_ALLOC_CONF: {current}")
            del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    # Set PYTORCH_CUDA_ALLOC_CONF with more aggressive fragmentation prevention
    # Smaller blocks = better fragmentation handling but slight performance cost
    # 64MB blocks for maximum fragmentation resistance (half of 128MB)
    cuda_alloc_config = "max_split_size_mb:64"

    # Apply the safe configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_config
    print(f"🔧 CUDA memory allocation configured: {cuda_alloc_config}")
    print("💡 Using 64MB memory blocks to reduce fragmentation")

    # Also set cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True

    # Log current CUDA memory state
    try:
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        print(
            f"📊 Initial CUDA state: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB"
        )
    except Exception as e:
        print(f"⚠️ Could not check initial CUDA state: {e}")

    return True


def clear_model_caches(model: Any, processor: Optional[Any] = None):
    """
    Phase 1: Enhanced cache clearing for transformer models.

    Args:
        model: The model to clear caches from
        processor: Optional processor/tokenizer to clear caches from
    """
    try:
        print("🧹 Clearing model caches...")

        # Clear KV cache if it exists
        if hasattr(model, "past_key_values"):
            model.past_key_values = None
            print("  - Cleared past_key_values")

        # Clear generation cache
        if hasattr(model, "_past_key_values"):
            model._past_key_values = None
            print("  - Cleared _past_key_values")

        # Clear language model caches (for models with separate language model)
        if hasattr(model, "language_model"):
            lang_model = model.language_model
            if hasattr(lang_model, "past_key_values"):
                lang_model.past_key_values = None
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

        print("✅ Model caches cleared")

    except Exception as e:
        print(f"⚠️ Error clearing caches: {e}")
        # Continue anyway - don't fail the entire process


def detect_memory_fragmentation() -> tuple[float, float, float]:
    """
    Detect GPU memory fragmentation.

    Returns:
        tuple: (allocated_gb, reserved_gb, fragmentation_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    try:
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        fragmentation = reserved - allocated

        return allocated, reserved, fragmentation
    except Exception:
        return 0.0, 0.0, 0.0


def handle_memory_fragmentation(threshold_gb: float = 1.0, aggressive: bool = True):
    """
    Handle GPU memory fragmentation with various strategies.

    Args:
        threshold_gb: Fragmentation threshold in GB to trigger cleanup
        aggressive: Whether to use aggressive cleanup strategies
    """
    if not torch.cuda.is_available():
        return

    allocated, reserved, fragmentation = detect_memory_fragmentation()

    print(
        f"🧹 Memory state: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Fragmentation={fragmentation:.2f}GB"
    )

    if fragmentation > threshold_gb:
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
    model: Optional[Any] = None, processor: Optional[Any] = None
):
    """
    Perform comprehensive memory cleanup including cache clearing and defragmentation.

    Args:
        model: Optional model to clear caches from
        processor: Optional processor to clear caches from
    """
    # Phase 1: Clear model caches if provided
    if model is not None:
        clear_model_caches(model, processor)

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
        handle_memory_fragmentation(threshold_gb=0.5, aggressive=True)


class ResilientGenerator:
    """
    Resilient generation wrapper with multiple fallback strategies for OOM handling.

    This class provides a robust generation pipeline with:
    1. Standard generation
    2. OffloadedCache fallback
    3. Emergency model reload
    4. CPU fallback as ultimate strategy
    """

    def __init__(self, model, processor=None, model_path=None, model_loader=None):
        """
        Initialize resilient generator.

        Args:
            model: The model to use for generation
            processor: Optional processor/tokenizer
            model_path: Path to model for emergency reload
            model_loader: Function to reload the model
        """
        self.model = model
        self.processor = processor
        self.model_path = model_path
        self.model_loader = model_loader
        self.oom_count = 0
        self.max_oom_retries = 3

    def generate(self, inputs: Dict[str, Any], generation_config: Dict[str, Any] = None, **generation_kwargs) -> Any:
        """
        Generate with automatic fallback on OOM errors.

        Args:
            inputs: Model inputs
            generation_config: Generation configuration dict
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated output
        """
        print("🔍 DEBUG: ResilientGenerator.generate() called!")
        print(f"🔍 DEBUG: inputs type: {type(inputs)}, keys: {list(inputs.keys()) if inputs else 'None'}")
        print(f"🔍 DEBUG: generation_config: {generation_config}")
        print(f"🔍 DEBUG: generation_kwargs: {generation_kwargs}")
        
        # Use generation_config if provided, otherwise fall back to kwargs
        if generation_config is not None:
            generation_kwargs = generation_config
            print(f"🔍 DEBUG: Using generation_config: {generation_kwargs}")
        
        try:
            # First attempt: Standard generation
            print("🔍 DEBUG: About to call _standard_generate")
            print(f"🔍 DEBUG: inputs: {inputs}")
            print(f"🔍 DEBUG: generation_kwargs: {generation_kwargs}")
            result = self._standard_generate(inputs, generation_kwargs)
            print("🔍 DEBUG: _standard_generate completed successfully")
            return result

        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ CUDA OOM detected: {e}")
            self.oom_count += 1

            if self.oom_count <= 1:
                # Strategy 1: Try with OffloadedCache
                return self._offloaded_cache_generate(inputs, generation_kwargs)
            elif self.oom_count <= 2 and self.model_loader:
                # Strategy 2: Emergency model reload
                return self._emergency_reload_generate(inputs, generation_kwargs)
            else:
                # Strategy 3: CPU fallback
                return self._cpu_fallback_generate(inputs, generation_kwargs)

    def _standard_generate(self, inputs: Dict[str, Any], generation_kwargs: Dict[str, Any]) -> Any:
        """Standard generation attempt."""
        print("🔍 DEBUG: _standard_generate method entered!")
        print(f"🔍 DEBUG: hasattr(self.model, 'generate'): {hasattr(self.model, 'generate')}")
        print(f"🔍 DEBUG: hasattr(self.model, 'chat'): {hasattr(self.model, 'chat')}")
        
        # For InternVL3, always use chat method even though generate exists
        # InternVL3's generate method expects different input format
        if hasattr(self.model, "chat") and "tokenizer" in inputs:
            print("🔍 DEBUG: Using model.chat path (InternVL3 forced)")
            # For models like InternVL3 that use chat interface
            # InternVL3 chat method signature: chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=False)
            try:
                print("🔍 DEBUG: Extracting inputs...")
                tokenizer = inputs.get("tokenizer", self.processor)
                print(f"🔍 DEBUG: tokenizer extracted: {type(tokenizer)}")
                pixel_values = inputs.get("pixel_values")
                print(f"🔍 DEBUG: pixel_values extracted: {type(pixel_values)}")
                question = inputs.get("question")
                print(f"🔍 DEBUG: question extracted: {type(question)}")

                # Validate inputs and debug tokenizer state
                print("🔍 DEBUG: Inside ResilientGenerator _standard_generate")
                print(f"🔍 DEBUG: tokenizer type: {type(tokenizer)}")
                print(f"🔍 DEBUG: pixel_values type: {type(pixel_values)}")
                print(f"🔍 DEBUG: question type: {type(question)}")
                print(f"🔍 DEBUG: generation_kwargs: {generation_kwargs}")

                if tokenizer is None:
                    raise ValueError("tokenizer is None in ResilientGenerator")
                if pixel_values is None:
                    raise ValueError("pixel_values is None in ResilientGenerator")
                if question is None:
                    raise ValueError("question is None in ResilientGenerator")
                
                # Test tokenizer inside ResilientGenerator
                print("🔍 DEBUG: Testing tokenizer inside ResilientGenerator...")
                try:
                    tokenizer_test = tokenizer("ResilientGenerator test", return_tensors="pt")
                    print(f"🔍 DEBUG: ResilientGenerator tokenizer test keys: {tokenizer_test.keys()}")
                    print(f"🔍 DEBUG: ResilientGenerator input_ids: {tokenizer_test['input_ids'].shape if tokenizer_test.get('input_ids') is not None else 'None'}")
                except Exception as tok_err:
                    print(f"🔍 DEBUG: Tokenizer test failed inside ResilientGenerator: {tok_err}")
                    raise

                # InternVL3 expects generation_config as a dict, not unpacked kwargs
                return self.model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_kwargs,  # Already a dict, pass directly
                    history=None,
                    return_history=False,
                )
            except Exception as e:
                print(
                    f"🔍 DEBUG: InternVL3 chat failed with: {type(e).__name__}: {str(e)}"
                )
                print(
                    f"🔍 DEBUG: generation_kwargs keys: {list(generation_kwargs.keys())}"
                )
                import traceback

                traceback.print_exc()
                raise
        elif hasattr(self.model, "generate"):
            print("🔍 DEBUG: Using model.generate path (fallback)")
            return self.model.generate(**inputs, **generation_kwargs)
        else:
            raise ValueError("Model does not have generate or chat method")

    def _offloaded_cache_generate(
        self, inputs: Dict[str, Any], generation_kwargs: Dict[str, Any]
    ) -> Any:
        """Generation with OffloadedCache fallback."""
        print("🔄 Retrying with cache_implementation='offloaded'...")

        # Emergency cleanup before retry
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            if hasattr(self.model, "generate"):
                # Add OffloadedCache configuration for generate models
                generation_kwargs["cache_implementation"] = "offloaded"
                return self.model.generate(**inputs, **generation_kwargs)
            else:
                # For chat-based models, offloaded cache may not be supported
                # Just retry with standard generation after cleanup (no cache_implementation)
                return self._standard_generate(inputs, **generation_kwargs)

        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ OffloadedCache also failed: {e}")
            raise

    def _emergency_reload_generate(
        self, inputs: Dict[str, Any], generation_kwargs: Dict[str, Any]
    ) -> Any:
        """Emergency model reload for severe OOM issues."""
        print("🚨 EMERGENCY: Reloading model to force complete memory reset...")

        # Complete cleanup
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor

        # Aggressive memory cleanup
        comprehensive_memory_cleanup()

        # Reload model
        if self.model_loader:
            self.model, self.processor = self.model_loader(self.model_path)
        else:
            raise RuntimeError("No model loader provided for emergency reload")

        # Try generation with fresh model
        # Only add cache_implementation for models that support it (generate method)
        if hasattr(self.model, "generate"):
            generation_kwargs["cache_implementation"] = "offloaded"
        return self._standard_generate(inputs, generation_kwargs)

    def _cpu_fallback_generate(
        self, inputs: Dict[str, Any], generation_kwargs: Dict[str, Any]
    ) -> Any:
        """Ultimate CPU fallback when all GPU strategies fail."""
        print("☢️ ULTIMATE FALLBACK: Processing on CPU (slower but stable)...")

        # Move model to CPU if not already there
        if next(self.model.parameters()).device.type != "cpu":
            self.model = self.model.to("cpu")

        # Move inputs to CPU
        cpu_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                cpu_inputs[key] = value.to("cpu")
            else:
                cpu_inputs[key] = value

        # Remove cache_implementation since we're on CPU
        if "cache_implementation" in generation_kwargs:
            del generation_kwargs["cache_implementation"]

        # Generate on CPU
        with torch.no_grad():
            return self._standard_generate(cpu_inputs, generation_kwargs)


def get_available_gpu_memory(device: str = "cuda") -> float:
    """
    Get available GPU memory in GB.

    Args:
        device: Device string (e.g., "cuda", "cuda:0")

    Returns:
        float: Available memory in GB
    """
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0

    try:
        # Get device index
        if device == "cuda":
            device_idx = torch.cuda.current_device()
        else:
            device_idx = int(device.split(":")[-1])

        # Get total and allocated memory
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_idx)
        available_memory = (total_memory - allocated_memory) / (
            1024**3
        )  # Convert to GB

        return available_memory
    except Exception as e:
        print(f"⚠️ Could not detect GPU memory: {e}")
        return 16.0  # Assume 16GB as default for V100


def optimize_model_for_v100(model: Any):
    """
    Apply V100-specific optimizations to a model.

    Args:
        model: The model to optimize
    """
    if not torch.cuda.is_available():
        return

    # Enable basic V100 optimizations (conservative)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Set model to evaluation mode
    model.eval()

    # Enable gradient checkpointing if available (saves memory)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("🚀 Gradient checkpointing enabled for memory efficiency")

    print("🚀 V100 optimizations applied")
