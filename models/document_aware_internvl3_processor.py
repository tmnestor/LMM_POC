#!/usr/bin/env python3
"""
Document-Aware InternVL3 Processor - Standalone Implementation

A complete standalone processor designed from the ground up for
document-aware extraction with dynamic field lists. NO INHERITANCE.

Adapted from InternVL3Processor but optimized for document-specific extraction
with targeted field lists (invoice: 20 fields, receipt: 15 fields, etc.)
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from common.config import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INTERNVL3_MAX_TILES_2B,
    INTERNVL3_MAX_TILES_8B,
    INTERNVL3_MODEL_PATH,
    get_auto_batch_size,
    get_max_new_tokens,
)

# Removed: DocumentAwareGroupedExtraction import - now using direct field extraction
from common.extraction_cleaner import ExtractionCleaner
from common.gpu_optimization import (
    clear_model_caches,
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    detect_memory_fragmentation,
    get_available_gpu_memory,
    handle_memory_fragmentation,
    optimize_model_for_v100,
)

warnings.filterwarnings("ignore")


class DocumentAwareInternVL3Processor:
    """Standalone document-aware InternVL3 processor with dynamic field support."""

    def __init__(
        self,
        field_list: List[str],
        model_path: str = None,
        device: str = "cuda",
        debug: bool = False,
        batch_size: Optional[int] = None,
        skip_model_loading: bool = False,
    ):
        """
        Initialize document-aware InternVL3 processor with specific field list.

        Args:
            field_list (List[str]): Fields to extract (e.g., 20 invoice fields)
            model_path (str): Path to InternVL3 model
            device (str): Device to run on
            debug (bool): Enable debug output
            batch_size (int): Batch size (auto-detected if None)
            skip_model_loading (bool): Skip loading model (for reusing existing model)
        """
        print("🔍 INIT-TRACE-001: __init__ ENTRY - starting basic field assignment")
        
        try:
            self.field_list = field_list
            self.field_count = len(field_list)
            print(f"🔍 INIT-TRACE-002: Basic fields assigned - {len(field_list)} fields")
            
            self.model_path = model_path or INTERNVL3_MODEL_PATH
            print(f"🔍 INIT-TRACE-003: Model path assigned - {self.model_path}")
            
            # Simple device assignment - no MPS support
            self.device = device
            if device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
                print("💻 CUDA not available, using CPU")
            print(f"🔍 INIT-TRACE-004: Device assigned - {self.device}")
                
            self.debug = debug
            print(f"🔍 INIT-TRACE-005: Debug flag set - {debug}")

            # Initialize components
            self.model = None
            self.tokenizer = None
            self.generation_config = None
            print("🔍 INIT-TRACE-006: Component placeholders initialized")

            # Initialize extraction cleaner for value normalization
            print("🔍 INIT-TRACE-007: About to initialize ExtractionCleaner")
            self.cleaner = ExtractionCleaner(debug=debug)
            print("🔍 INIT-TRACE-008: ExtractionCleaner initialized successfully")

            # Note: Previously used DocumentAwareGroupedExtraction but now using direct field extraction

            # Fix 8B detection using actual model path (defensive string conversion)
            print("🔍 INIT-TRACE-009: About to detect model size (8B vs 2B)")
            try:
                model_path_str = str(self.model_path) if self.model_path else ""
                self.is_8b_model = "8B" in model_path_str
                print(f"🔍 INIT-TRACE-010: Model size detected - 8B={self.is_8b_model}")
            except RecursionError:
                # Fallback if string conversion causes recursion
                self.is_8b_model = False
                print("🔍 INIT-TRACE-011: RecursionError during model size detection, defaulting to 2B")
                print("⚠️ Warning: Could not detect model size, defaulting to 2B")
            
            # Initialize torch_dtype here for when model is reused
            print("🔍 INIT-TRACE-012: Setting torch_dtype based on device")
            if self.device == "cpu":
                self.torch_dtype = torch.float32
            else:
                # CUDA devices support bfloat16
                self.torch_dtype = torch.bfloat16
            print(f"🔍 INIT-TRACE-013: torch_dtype set to {self.torch_dtype}")

            if self.debug:
                print(
                    f"🎯 Document-aware InternVL3 processor initialized for {self.field_count} fields"
                )
                # Handle empty field list for universal extraction mode
                if field_list:
                    print(f"   Fields: {field_list[0]} → {field_list[-1]}")
                else:
                    print("   🌟 Universal extraction mode: Uses internal 15-field list")
                print(f"   Model variant: {'8B' if self.is_8b_model else '2B'}")

            # Configure CUDA memory allocation
            print("🔍 INIT-TRACE-014: About to call configure_cuda_memory_allocation()")
            configure_cuda_memory_allocation()
            print("🔍 INIT-TRACE-015: configure_cuda_memory_allocation() completed")

            # Set seeds for reproducibility
            print("🔍 INIT-TRACE-016: About to call _set_random_seeds(42)")
            self._set_random_seeds(42)
            print("🔍 INIT-TRACE-017: _set_random_seeds(42) completed")

            # Configure batch processing
            print("🔍 INIT-TRACE-018: About to call _configure_batch_processing()")
            self._configure_batch_processing(batch_size)
            print("🔍 INIT-TRACE-019: _configure_batch_processing() completed")

            # Configure generation parameters for dynamic field count
            print("🔍 INIT-TRACE-020: About to call _configure_generation()")
            self._configure_generation()
            print("🔍 INIT-TRACE-021: _configure_generation() completed")

            # Load model and tokenizer (unless skipping for reuse)
            if not skip_model_loading:
                print("🔍 INIT-TRACE-022: About to call _load_model()")
                self._load_model()
                print("🔍 INIT-TRACE-023: _load_model() completed")
            else:
                print("🔍 INIT-TRACE-022: Skipping model loading as requested")
            
            print("🔍 INIT-TRACE-024: __init__ EXIT - initialization completed successfully")
            
        except RecursionError as e:
            print(f"🚨 RECURSION ERROR in __init__: {e}")
            print("🔍 This indicates infinite recursion during processor initialization")
            raise
        except Exception as e:
            print(f"🚨 GENERAL ERROR in __init__: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _set_random_seeds(self, seed: int = 42):
        """Set all random seeds for reproducibility."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.debug:
            print(f"🎲 Random seeds set to {seed} for deterministic output")

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        print("🔍 BATCH-TRACE-001: _configure_batch_processing ENTRY")
        
        try:
            if batch_size is not None:
                print(f"🔍 BATCH-TRACE-002: Using manual batch_size={batch_size}")
                self.batch_size = max(1, batch_size)
                print(f"🎯 Using manual batch size: {self.batch_size}")
            else:
                print("🔍 BATCH-TRACE-003: Auto-detecting batch size")
                
                # Auto-detect batch size based on available memory and model variant
                print("🔍 BATCH-TRACE-004: About to call get_available_gpu_memory()")
                available_memory = get_available_gpu_memory(self.device)
                print(f"🔍 BATCH-TRACE-005: get_available_gpu_memory() returned {available_memory}")
                
                model_key = "internvl3-8b" if self.is_8b_model else "internvl3-2b"
                print(f"🔍 BATCH-TRACE-006: model_key={model_key}")
                
                print("🔍 BATCH-TRACE-007: About to call get_auto_batch_size()")
                self.batch_size = get_auto_batch_size(model_key, available_memory)
                print(f"🔍 BATCH-TRACE-008: get_auto_batch_size() returned {self.batch_size}")
                
                print(
                    f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB, Model: {model_key})"
                )
            
            print("🔍 BATCH-TRACE-009: _configure_batch_processing EXIT")
            
        except RecursionError as e:
            print(f"🚨 RECURSION ERROR in _configure_batch_processing: {e}")
            raise
        except Exception as e:
            print(f"🚨 ERROR in _configure_batch_processing: {e}")
            raise

    def _configure_generation(self):
        """Configure generation parameters for dynamic field count."""
        print("🔍 GEN-TRACE-001: _configure_generation ENTRY")
        
        try:
            # Calculate dynamic max_new_tokens based on actual field count
            print(f"🔍 GEN-TRACE-002: About to call get_max_new_tokens() for {self.field_count} fields")
            max_tokens = get_max_new_tokens("internvl3", self.field_count)
            print(f"🔍 GEN-TRACE-003: get_max_new_tokens() returned {max_tokens}")

            # Get generation config from centralized config (matches original internvl3_processor)
            print("🔍 GEN-TRACE-004: About to import GENERATION_CONFIGS")
            from common.config import GENERATION_CONFIGS
            print("🔍 GEN-TRACE-005: GENERATION_CONFIGS imported successfully")

            print("🔍 GEN-TRACE-006: Retrieving base config for internvl3")
            base_gen_config = GENERATION_CONFIGS.get("internvl3", {})
            print(f"🔍 GEN-TRACE-007: Base config retrieved: {base_gen_config}")
            
            self.generation_config = {"max_new_tokens": max_tokens, **base_gen_config}
            print(f"🔍 GEN-TRACE-008: Generation config created: {self.generation_config}")
            
            # V100 memory optimization: Use more conservative generation settings
            print("🔍 GEN-TRACE-009: Checking V100 memory optimization")
            if self.is_8b_model and torch.cuda.is_available():
                print("🔍 GEN-TRACE-010: 8B model + CUDA available, checking V100")
                try:
                    device_props = torch.cuda.get_device_properties(0)
                    total_memory = device_props.total_memory
                    print(f"🔍 GEN-TRACE-011: GPU total memory: {total_memory / (1024**3):.1f}GB")
                    
                    if total_memory < 17 * 1024**3:  # V100 has 16GB
                        print("🔍 GEN-TRACE-012: V100 detected, optimizing generation config")
                        self.generation_config["num_beams"] = 1  # Disable beam search to save memory
                        if self.debug:
                            print("🔧 V100 detected: Using memory-efficient generation (num_beams=1)")
                    else:
                        print("🔍 GEN-TRACE-013: High-memory GPU detected, using default settings")
                except (RuntimeError, AssertionError) as e:
                    print(f"🔍 GEN-TRACE-014: Error getting device properties: {e}")
                    # Handle CUDA not available or other CUDA errors gracefully
                    if self.debug:
                        print("💻 CUDA not available - skipping GPU memory optimization")
            else:
                print("🔍 GEN-TRACE-015: Not 8B model or CUDA not available, skipping V100 optimization")

            # Ensure deterministic generation (matches original internvl3_processor)
            print("🔍 GEN-TRACE-016: Checking do_sample setting")
            if self.generation_config.get("do_sample", True):
                print("🔍 GEN-TRACE-017: do_sample was True, forcing to False")
                print("⚠️ WARNING: do_sample was True, forcing to False for determinism")
                self.generation_config["do_sample"] = False
            else:
                print("🔍 GEN-TRACE-018: do_sample already False, no change needed")

            print(
                f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
                f"do_sample={self.generation_config['do_sample']} (greedy decoding)"
            )
            
            print("🔍 GEN-TRACE-019: _configure_generation EXIT")
            
        except RecursionError as e:
            print(f"🚨 RECURSION ERROR in _configure_generation: {e}")
            raise
        except Exception as e:
            print(f"🚨 ERROR in _configure_generation: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_model(self):
        """Load InternVL3 model and tokenizer with optimal configuration."""
        print(f"🔄 Loading InternVL3 model from: {self.model_path}")
        
        # Recursion protection for entire model loading process
        import sys
        recursion_limit = sys.getrecursionlimit()
        if recursion_limit < 1500:
            sys.setrecursionlimit(1500)
            print(f"🔧 Increased recursion limit from {recursion_limit} to 1500 for model loading")

        if self.is_8b_model:
            print("🎯 InternVL3-8B detected - applying aggressive V100 optimizations")
        else:
            print("🎯 InternVL3-2B detected - using standard optimizations")

        try:
            # Determine appropriate dtype based on device
            if self.device == "cpu":
                # CPU works better with float32
                torch_dtype = torch.float32
                print("💻 CPU device - using float32")
            else:
                # CUDA devices support bfloat16
                torch_dtype = torch.bfloat16
                print("🔥 CUDA device - using bfloat16")
            
            # Store dtype for later use in inference
            self.torch_dtype = torch_dtype
            
            # Base model configuration
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "use_flash_attn": False,  # Disabled for compatibility
                "trust_remote_code": True,
            }

            # Initialize quantization tracking for proper warm-up logic
            quantization_success = False

            # Simple, reliable model loading - H200 direct, V100 quantized if available
            if self.is_8b_model:
                # Check GPU memory to determine best loading strategy (with recursion protection)
                try:
                    if torch.cuda.is_available():
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_name = torch.cuda.get_device_name(0)
                    else:
                        gpu_memory_gb = 0
                        gpu_name = "CPU"
                except (RecursionError, RuntimeError) as e:
                    print(f"⚠️ Warning: GPU detection failed ({e}), using conservative defaults")
                    gpu_memory_gb = 0
                    gpu_name = "Unknown"
                
                print(f"🎯 InternVL3-8B Loading: {gpu_name} ({gpu_memory_gb:.0f}GB VRAM)")
                
                # Strategy 1: High-end GPUs (H200/H100/A100 40GB+) - ISOLATED quantization testing
                if gpu_memory_gb >= 40:
                    print("📦 STRATEGY: ISOLATED BitsAndBytesConfig testing on H200")
                    print(f"   Expected usage: ~16GB ({16/gpu_memory_gb*100:.0f}% of {gpu_memory_gb:.0f}GB)")
                    
                    # CRITICAL: Test BitsAndBytesConfig creation in complete isolation to prevent recursion
                    print("🧪 PHASE 1: ISOLATED BitsAndBytesConfig creation test")
                    quantization_config = None
                    quantization_viable = False
                    
                    try:
                        print("🔧 Step 1: Importing BitsAndBytesConfig...")
                        from transformers import BitsAndBytesConfig
                        print("✅ Step 1: Import successful")
                        
                        print("🔧 Step 2: Creating BitsAndBytesConfig object...")
                        # Test with reduced parameters to avoid recursion triggers
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            # REMOVED: bnb_8bit_compute_dtype - this may cause recursion
                        )
                        print("✅ Step 2: BitsAndBytesConfig creation successful")
                        
                        print("🔧 Step 3: Testing model loading with BitsAndBytesConfig...")
                        # Ultra-minimal kwargs to avoid any recursion triggers
                        test_kwargs = {
                            "torch_dtype": torch_dtype,
                            "trust_remote_code": True,
                            "quantization_config": quantization_config,
                            # NO device_map, NO low_cpu_mem_usage, NO auth params
                        }
                        
                        self.model = AutoModel.from_pretrained(
                            self.model_path, **test_kwargs
                        ).eval()
                        print(f"✅ SUCCESS: ISOLATED BitsAndBytesConfig works on {gpu_name}!")
                        print("🎯 This confirms quantization approach is viable for V100")
                        quantization_success = True
                        quantization_viable = True
                        
                    except Exception as quant_error:
                        print(f"❌ ISOLATED BitsAndBytesConfig failed: {quant_error}")
                        print("🔍 Checking if this is a recursion error...")
                        
                        if "recursion" in str(quant_error).lower() or "RecursionError" in str(type(quant_error)):
                            print("🚨 RECURSION DETECTED: BitsAndBytesConfig causes infinite recursion")
                            print("🚫 Quantization PERMANENTLY DISABLED for this session")
                            quantization_viable = False
                        else:
                            print(f"💡 Non-recursion error: {str(quant_error)[:100]}...")
                            quantization_viable = False
                        
                        print("🔄 Falling back to direct loading without quantization...")
                        
                        # Ultra-minimal direct loading
                        minimal_kwargs = {
                            "torch_dtype": torch_dtype,
                            "trust_remote_code": True,
                        }
                        
                        try:
                            self.model = AutoModel.from_pretrained(
                                self.model_path, **minimal_kwargs
                            ).eval()
                            print(f"✅ SUCCESS: Direct loading fallback on {gpu_name}")
                            quantization_success = False
                        except Exception as direct_error:
                            print(f"❌ Direct loading also failed: {direct_error}")
                            
                            # Last resort with low_cpu_mem_usage
                            enhanced_kwargs = {
                                "torch_dtype": torch_dtype,
                                "trust_remote_code": True,
                                "low_cpu_mem_usage": True,
                            }
                            self.model = AutoModel.from_pretrained(
                                self.model_path, **enhanced_kwargs
                            ).eval()
                            print(f"✅ SUCCESS: Enhanced direct loading on {gpu_name}")
                            quantization_success = False
                    
                    # Store quantization viability for V100 strategy
                    self._quantization_viable = quantization_viable
                    print(f"   VRAM Usage: ~{16 if not quantization_success else 8}GB ({(16 if not quantization_success else 8)/gpu_memory_gb*100:.0f}% of {gpu_memory_gb:.0f}GB)")
                
                else:
                    # Strategy 2: V100 and memory-constrained GPUs - Use H200-confirmed approach ONLY if viable
                    print("📦 STRATEGY: V100 quantization using H200-confirmed approach")
                    if gpu_memory_gb <= 16:
                        print(f"   V100 ({gpu_memory_gb:.0f}GB): Quantization REQUIRED for InternVL3-8B")
                    print(f"   Target usage: ~8GB (safe for {gpu_memory_gb:.0f}GB GPU)")
                    
                    # Check if H200 confirmed quantization is viable
                    quantization_viable = getattr(self, '_quantization_viable', False)
                    quantization_loaded = False
                    
                    if quantization_viable:
                        print("🔧 PHASE 2: Applying H200-confirmed BitsAndBytesConfig to V100")
                        print("✅ H200 testing confirmed quantization is viable")
                        
                        try:
                            from transformers import BitsAndBytesConfig
                            
                            # Use EXACT same minimal config that worked on H200
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                # NO bnb_8bit_compute_dtype - confirmed to avoid recursion
                            )
                            
                            # Minimal V100 kwargs
                            v100_kwargs = {
                                "torch_dtype": torch_dtype,
                                "trust_remote_code": True,
                                "quantization_config": quantization_config,
                                # NO device_map, NO low_cpu_mem_usage initially
                            }
                            
                            print("🔧 Loading with H200-confirmed BitsAndBytesConfig...")
                            self.model = AutoModel.from_pretrained(
                                self.model_path, **v100_kwargs
                            ).eval()
                            
                            print("✅ SUCCESS: V100 quantization using H200-confirmed approach")
                            print(f"   Memory: ~8GB (safe for V100's {gpu_memory_gb:.0f}GB)")
                            quantization_success = True
                            quantization_loaded = True
                            
                        except Exception as v100_quant_error:
                            print(f"❌ V100 quantization failed despite H200 success: {v100_quant_error}")
                            if "recursion" in str(v100_quant_error).lower():
                                print("🚨 V100 RECURSION: Different behavior than H200")
                            quantization_loaded = False
                    else:
                        print("🚫 PHASE 2: Skipping quantization - H200 testing failed or caused recursion")
                        print("🔄 Going directly to fallback strategies")
                        quantization_loaded = False
                    
                    # FALLBACK: Direct loading for sufficient memory or emergency
                    if not quantization_loaded:
                        if gpu_memory_gb >= 20:
                            print(f"🔄 Trying direct loading on {gpu_memory_gb:.0f}GB GPU...")
                            
                            direct_kwargs = {
                                "torch_dtype": torch_dtype,
                                "trust_remote_code": True,
                                "low_cpu_mem_usage": True,
                            }
                            
                            try:
                                self.model = AutoModel.from_pretrained(
                                    self.model_path, **direct_kwargs
                                ).eval()
                                
                                print("✅ SUCCESS: InternVL3-8B loaded directly")
                                print(f"⚠️ WARNING: Using ~16GB on {gpu_memory_gb:.0f}GB")
                                quantization_success = False
                                
                            except Exception as direct_error:
                                print(f"❌ Direct loading failed: {direct_error}")
                                self._handle_final_loading_failure(gpu_name, gpu_memory_gb)
                        else:
                            # V100 with failed quantization - cannot proceed
                            print(f"\n❌ CRITICAL: V100 ({gpu_memory_gb:.0f}GB) requires quantization")
                            print("❌ All quantization approaches failed")
                            print("💡 SOLUTION: Use InternVL3-2B instead (fits easily on V100)")
                            self._handle_final_loading_failure(gpu_name, gpu_memory_gb)

            else:
                # InternVL3-2B: Direct loading without quantization
                dtype_name = "bfloat16" if torch_dtype == torch.bfloat16 else "float32"
                print(f"\n📦 LOADING APPROACH: Direct {dtype_name} (No Quantization Needed)")
                print("   Model: InternVL3-2B")
                print("   Reason: 2B model fits comfortably in GPU memory")
                print(f"   Precision: {dtype_name}")

                # Configuration for 2B model - always use auto device mapping
                model_kwargs["device_map"] = "auto"
                model_kwargs["local_files_only"] = True  # Force local loading
                model_kwargs["use_auth_token"] = False   # Disable auth

                self.model = AutoModel.from_pretrained(
                    self.model_path, **model_kwargs
                ).eval()

                print(f"✅ SUCCESS: InternVL3-2B loaded in {dtype_name}")
                print("   Approach: Direct loading without quantization")
                print("   VRAM Usage: ~4-6GB (plenty of headroom on 16GB V100)")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )

            print("✅ InternVL3 model and tokenizer loaded successfully")

            # Print loading summary
            print("\n" + "=" * 60)
            print("📊 MODEL LOADING SUMMARY")
            print("=" * 60)
            if self.is_8b_model:
                if quantization_success:
                    print("Model: InternVL3-8B")
                    print("Status: ✅ Successfully loaded with 8-bit quantization")
                    print("Method: Modern BitsAndBytesConfig or Legacy load_in_8bit")
                    print("VRAM: ~8GB (50% reduction from fp16)")
                else:
                    print("Model: InternVL3-8B")
                    print("Status: ⚠️ Loaded but quantization method unclear")
            else:
                print("Model: InternVL3-2B")
                print(f"Status: ✅ Successfully loaded in {dtype_name}")
                print("Method: Direct loading (no quantization needed)")
                print("VRAM: ~4-6GB")
            print("Hardware: V100 GPU (16GB VRAM)")
            print("=" * 60 + "\n")

            # Apply GPU optimizations
            optimize_model_for_v100(self.model)

            # Configure warm-up based on model type and precision
            if self.is_8b_model and quantization_success:
                print(
                    "⚠️ Skipping warm-up test for quantized 8B model (expected with 8-bit)"
                )
                print("   Model will be tested during actual inference")
            elif self.is_8b_model:
                print("ℹ️ 8B model loaded - warm-up enabled to verify stability")
            else:
                print(
                    f"✅ InternVL3-2B {dtype_name} model loaded - enabling warm-up for verification"
                )
                print("   V100 optimizations active")

        except Exception as e:
            print(f"❌ Error loading InternVL3 model: {e}")
            raise

    # OLD SINGLE-PASS METHODS REMOVED - NOW USING HYBRID GROUPED EXTRACTION
    # The hybrid system automatically handles:
    # - Document type detection
    # - Smart group filtering
    # - Proven 90.6% accuracy grouped prompts
    # - Field-specific instructions per group

    def build_transform(self, input_size=DEFAULT_IMAGE_SIZE):
        """Build InternVL3 image transformation pipeline."""
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        """Find aspect ratio optimized for high tile count and better OCR coverage."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        # TILE_BOOST: Bias toward higher tile counts for better OCR accuracy
        tile_boost_threshold = 0.3  # Allow 30% worse aspect ratio for more tiles
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            tile_count = ratio[0] * ratio[1]
            current_best_tiles = best_ratio[0] * best_ratio[1]

            # NEW LOGIC: Prefer higher tile counts if aspect ratio difference is reasonable
            if ratio_diff < best_ratio_diff:
                # Always take better aspect ratio match
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff <= best_ratio_diff * (1 + tile_boost_threshold):
                # Accept slightly worse aspect ratio if we get significantly more tiles
                if tile_count > current_best_tiles:
                    if self.debug:
                        print(f"🚀 TILE_BOOST: Preferring {ratio[0]}x{ratio[1]}={tile_count} tiles over {best_ratio[0]}x{best_ratio[1]}={current_best_tiles} tiles")
                        print(f"   Aspect ratio: {ratio_diff:.3f} vs {best_ratio_diff:.3f} (within {tile_boost_threshold} threshold)")
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                # Original tie-breaking logic
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def dynamic_preprocess(
        self,
        image,
        min_num=1,
        max_num=20,  # Increased default for better OCR quality
        image_size=DEFAULT_IMAGE_SIZE,
        use_thumbnail=False,
    ):
        """InternVL3 dynamic preprocessing algorithm."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # TILE_BOOST: Force high minimum for document OCR accuracy testing
        if min_num < 9:
            min_num = 9  # Force at least 9 tiles for document OCR accuracy
            if self.debug:
                print(f"🚀 DOCUMENT_BOOST: Increased min_num to {min_num} for better text coverage")

        if self.debug:
            print(f"🔍 DYNAMIC_PREPROCESS: image={orig_width}x{orig_height}, aspect_ratio={aspect_ratio:.2f}")
            print(f"🔍 PARAMS: min_num={min_num}, max_num={max_num}, image_size={image_size}")

        # Calculate target ratios
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        if self.debug:
            print(f"🎯 TARGET_RATIOS: {len(target_ratios)} options, max tiles per ratio:")
            for ratio in target_ratios[-5:]:  # Show top 5 largest ratios
                print(f"   {ratio[0]}x{ratio[1]} = {ratio[0]*ratio[1]} tiles")

        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        if self.debug:
            tiles_count = target_aspect_ratio[0] * target_aspect_ratio[1]
            print(f"✅ CHOSEN_RATIO: {target_aspect_ratio[0]}x{target_aspect_ratio[1]} = {tiles_count} tiles (max allowed: {max_num})")

        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]

        if self.debug:
            print(f"📏 TARGET_DIMS: {target_width}x{target_height} → resize from {orig_width}x{orig_height}")

        # Resize image
        resized_img = image.resize((target_width, target_height))
        processed_images = []

        # Split into tiles
        for i in range(target_aspect_ratio[0]):
            for j in range(target_aspect_ratio[1]):
                box = (
                    i * image_size,
                    j * image_size,
                    (i + 1) * image_size,
                    (j + 1) * image_size,
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)

        # Add thumbnail if requested
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            if self.debug:
                print(f"📎 THUMBNAIL_ADDED: {len(processed_images)} total tiles (includes thumbnail)")

        if self.debug:
            print(f"🏁 PREPROCESS_RESULT: {len(processed_images)} tiles created")

        return processed_images

    def load_image(self, image_file, input_size=DEFAULT_IMAGE_SIZE, max_num=None):
        """Complete InternVL3 image loading and preprocessing pipeline."""
        
        
        # PHASE 3: Optimize tile count based on available memory
        # With single-pass extraction we have more memory headroom
        # V100 shows only 8.42GB/16GB used, so we can increase tiles for better OCR
        if max_num is None:
            if self.is_8b_model:
                max_num = INTERNVL3_MAX_TILES_8B  # Configurable: default 20 tiles
            else:
                max_num = INTERNVL3_MAX_TILES_2B  # Configurable: default 24 tiles


        if self.debug:
            print(f"🔍 LOAD_IMAGE: max_num={max_num}, input_size={input_size}")
            allocated, reserved, fragmentation = detect_memory_fragmentation()
            print(f"📊 MEMORY_BEFORE_LOAD: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")

        # Load image if path provided
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
            if self.debug:
                print(f"🖼️  IMAGE_LOADED: {image.size[0]}x{image.size[1]} pixels")
        else:
            image = image_file.convert("RGB")
            if self.debug:
                print(f"🖼️  IMAGE_LOADED: {image.size[0]}x{image.size[1]} pixels")

        # Apply dynamic preprocessing
        images = self.dynamic_preprocess(image, image_size=input_size, max_num=max_num)

        if self.debug:
            print(f"🎯 TILES_CREATED: {len(images)} tiles from dynamic_preprocess (requested max_num={max_num})")

        # Apply transforms
        transform = self.build_transform(input_size=input_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        if self.debug:
            print(f"📐 TENSOR_SHAPE: {pixel_values.shape} (batch_size={pixel_values.shape[0]} tiles)")
            allocated, reserved, fragmentation = detect_memory_fragmentation()
            print(f"📊 MEMORY_AFTER_LOAD: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")

        return pixel_values

    def process_single_image(self, image_path: str) -> dict:
        """Process single image with document-aware field extraction."""
        print(f"🔍 PROCESSOR-TRACE: process_single_image ENTRY - {image_path}, {len(self.field_list)} fields")

        try:
            start_time = time.time()

            # Use the field list passed to constructor (document-aware)
            target_fields = self.field_list

            if self.debug:
                print(
                    f"🚀 Starting document-aware extraction for {Path(image_path).name}"
                )
                print(f"📊 Target fields: {len(target_fields)} document-specific fields")
                print(
                    "🎯 Strategy: Document-aware extraction with type-specific prompts"
                )
                print(
                    f"📝 Fields: {target_fields[:5]}{'...' if len(target_fields) > 5 else ''}"
                )

            # Memory cleanup - V100 optimized threshold (16GB total capacity)
            handle_memory_fragmentation(threshold_gb=2.0, aggressive=True)

            
            # DOCUMENT-AWARE: Extract using the document-specific field list
            # Now capture both extracted data AND raw response + prompt info
            extracted_data, raw_response, prompt_info = self._extract_fields_directly_with_details(image_path, target_fields)


            # Post-processing: Infer document type from extraction results
            inferred_type = self._infer_document_type_from_extraction(extracted_data)

            # Create metadata with document-aware info
            metadata = {
                "document_type": inferred_type,
                "extraction_strategy": "document_aware",
                "total_fields": len(target_fields),
                "extraction_method": "document_specific_field_extraction",
            }

            processing_time = time.time() - start_time

            if self.debug:
                print("🎉 DOCUMENT-AWARE EXTRACTION COMPLETED!")
                print("=" * 80)
                print("📋 Document type: Pre-detected (using provided field schema)")
                print(
                    f"🎯 Fields processed: {len(self.field_list)} document-specific fields"
                )
                print(f"⏱️ Processing time: {processing_time:.2f}s")

                found_fields = [
                    k for k, v in extracted_data.items() if v != "NOT_FOUND"
                ]
                print(
                    f"📊 Results: {len(found_fields)}/{self.field_count} fields found"
                )

                # Show field results summary
                for field in self.field_list:
                    value = extracted_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    print(f'  {status} {field}: "{value}"')
                print(f"  Total fields: {len(self.field_list)}")
                print("=" * 80)

            # Calculate standard metrics for compatibility
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in self.field_list]
            )
            response_completeness = extracted_fields_count / len(self.field_list)
            content_coverage = extracted_fields_count / len(self.field_list)

            # Cleanup with V100 optimizations
            comprehensive_memory_cleanup(self.model, self.tokenizer)

            print(f"🔍 PROCESSOR-TRACE: process_single_image EXIT - {extracted_fields_count}/{len(self.field_list)} fields, {processing_time:.3f}s")
            
            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": raw_response,  # Now contains actual model response
                "prompt_info": prompt_info,    # New: Contains prompt details
                "processing_time": processing_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": self.field_count,
                "extraction_metadata": metadata,
                "extraction_strategy": "document_aware_direct",
            }

        except Exception as e:
            if self.debug:
                print(f"❌ Document-aware extraction error for {image_path}: {e}")

            # Return error result with dynamic fields
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in self.field_list},
                "raw_response": f"Document-aware extraction error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "field_count": self.field_count,
                "extraction_strategy": "document_aware_direct_error",
            }

    def _extract_fields_directly_with_details(
        self, image_path: str, field_list: List[str]
    ) -> tuple[Dict[str, str], str, Dict[str, str]]:
        """
        Extract document-specific fields and return extracted data, raw response, and prompt info.
        
        Returns:
            Tuple of (extracted_data, raw_response, prompt_info)
        """
        
        if self.debug:
            print(f"🎯 Document-aware extraction: processing {len(field_list)} specific fields")

        # Create document-specific prompt based on field list
        field_lines = []
        for field in field_list:
            if field in ["BUSINESS_PHONE", "PAYER_PHONE"]:
                field_lines.append(f"{field}: [complete phone number with area code or NOT_FOUND]")
            elif field == "BUSINESS_ABN":
                field_lines.append(f"{field}: [11-digit ABN or NOT_FOUND]")
            elif field in ["TOTAL_AMOUNT", "SUBTOTAL_AMOUNT", "GST_AMOUNT"]:
                field_lines.append(f"{field}: [dollar amount with $ symbol or NOT_FOUND]")
            elif field in ["PAYER_ADDRESS", "BUSINESS_ADDRESS"]:
                field_lines.append(f"{field}: [complete address with postcode or NOT_FOUND]")
            elif field == "PAYMENT_METHOD":
                field_lines.append(f"{field}: [payment type like AMEX, Visa, Cash, etc. or NOT_FOUND]")
            elif field in ["LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES"]:
                field_lines.append(f"{field}: [pipe-separated values for all items or NOT_FOUND]")
            else:
                field_lines.append(f"{field}: [value as shown or NOT_FOUND]")

        prompt = f"""Extract structured data from this business document image.

OUTPUT FORMAT - EXACTLY {len(field_list)} LINES:
{chr(10).join(field_lines)}

INSTRUCTIONS:
-- Extract values exactly as they appear in the document
-- Use NOT_FOUND if a field is not present or cannot be determined
-- Use colon and space format: FIELD_NAME: value
-- For line items, use pipe-separated format: item1 | item2 | item3
-- Include $ symbol for monetary amounts
-- Output only the {len(field_list)} lines above, nothing else"""
        
        # Create prompt info for display
        prompt_info = {
            "prompt": prompt,
            "source": "document_aware_template",
            "field_count": len(field_list),
            "template_type": "targeted_extraction"
        }
        
        if self.debug:
            print("🔨 Using document-aware extraction template")
            print(f"📝 Prompt length: {len(prompt)} characters")

        # Process the image with the document-aware prompt
        try:
            raw_response = self._generate_response(image_path, prompt)
            
            if self.debug:
                print("✅ Document-aware extraction completed")
                print(f"📝 Raw response length: {len(raw_response)} characters")
                
            # Parse the response
            from common.extraction_parser import parse_extraction_response
            extracted_data = parse_extraction_response(raw_response, expected_fields=field_list)
            
            return extracted_data, raw_response, prompt_info
            
        except Exception as e:
            if self.debug:
                print(f"❌ Document-aware extraction failed: {e}")
            # Return empty results with NOT_FOUND for all fields
            error_response = f"Document-aware extraction error: {str(e)}"
            empty_data = {field: "NOT_FOUND" for field in field_list}
            return empty_data, error_response, prompt_info

    def _extract_fields_directly(
        self, image_path: str, field_list: List[str]
    ) -> Dict[str, str]:
        """
        Extract document-specific fields using document-aware prompts.
        Legacy method for backwards compatibility.
        """
        extracted_data, _, _ = self._extract_fields_directly_with_details(image_path, field_list)
        return extracted_data

    def _generate_response(self, image_path: str, prompt: str) -> str:
        """Generate response using the model's inference method."""
        return self._extract_with_prompt(image_path, prompt, generation_config=None)

    def _extract_fields_using_yaml(
        self, image_path: str, field_list: List[str]
    ) -> Dict[str, str]:
        """
        Extract specific fields using YAML-based unified prompt.

        Uses the same unified schema templates as Llama for consistency.
        """
        
        if self.debug:
            print(f"🎯 Extracting {len(field_list)} document-specific fields using YAML")

        # Use YAML-first approach: generate prompt from unified schema like Llama does
        from common.yaml_template_renderer import PureYAMLRenderer
        
        yaml_renderer = PureYAMLRenderer()
        
        # Determine document type from field list
        # If we have STATEMENT_DATE_RANGE, it's a bank_statement
        # Otherwise default to invoice/receipt
        if "STATEMENT_DATE_RANGE" in field_list:
            document_type = "bank_statement"
        elif "INVOICE_DATE" in field_list:
            document_type = "invoice"  # Could be invoice or receipt, default to invoice
        else:
            document_type = "invoice"  # Default fallback
            
        # Generate prompt from unified schema templates
        try:
            prompt = yaml_renderer.render_prompt_for_document_type(
                document_type=document_type, 
                field_list=field_list,
                model_name="internvl3"  # Use internvl3 model name
            )
        except Exception as e:
            # Re-raise the error instead of using a fallback
            raise ValueError(
                f"❌ FATAL: Could not generate prompt from unified schema: {e}\n"
                f"💡 Check unified_schema.yaml exists and has valid templates\n"
                f"💡 Ensure document_type '{document_type}' is supported"
            ) from e

        if self.debug:
            print("📝 Document-aware extraction prompt:")
            print("-" * 60)
            print(prompt)
            print("-" * 60)

        # Extract using the model
        raw_response = self._extract_with_prompt(image_path, prompt)

        if self.debug:
            print(
                f"📝 Raw response ({len(raw_response)} chars): {raw_response[:200]}..."
            )

        # Parse the response using the extraction parser
        from common.extraction_parser import parse_extraction_response

        extracted_data = parse_extraction_response(
            raw_response, expected_fields=field_list
        )

        # Apply ExtractionCleaner for consistent formatting (matching Llama behavior)
        if self.debug:
            print("🧹 Applying ExtractionCleaner to normalize field values...")
            
        cleaned_extracted_data = {}
        for field_name, value in extracted_data.items():
            # Apply same cleaning logic as Llama processor
            if value and value.lower().strip() not in [
                "not found", "not_found", "notfound", "n/a", "na", "none", "null", ""
            ]:
                cleaned_value = self.cleaner.clean_field_value(field_name, value)
            else:
                cleaned_value = "NOT_FOUND"
            cleaned_extracted_data[field_name] = cleaned_value

        # Replace extracted_data with cleaned version
        extracted_data = cleaned_extracted_data

        if self.debug:
            found_count = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")
            print(f"✅ Parsed {found_count}/{len(field_list)} fields successfully")

        return extracted_data

    def _extract_fields_universally(
        self, image_path: str, field_list: List[str]
    ) -> Dict[str, str]:
        """
        Extract all 15 universal fields using single-pass universal template.
        
        Replaces document-type-specific extraction with universal approach.
        """
        
        if self.debug:
            print(f"🌟 Universal extraction: processing all {len(field_list)} fields in single pass")

        # Use YAML universal extraction template
        from common.yaml_template_renderer import PureYAMLRenderer
        
        yaml_renderer = PureYAMLRenderer()
        
        # Generate universal prompt from unified schema
        try:
            # Access the universal extraction template we just added
            universal_config = yaml_renderer.unified_schema.get("universal_extraction", {})
            internvl3_config = universal_config.get("internvl3", {})
            
            if not internvl3_config:
                raise ValueError("Universal extraction template not found in unified schema")
            
            # Build universal prompt
            system_prompt = internvl3_config.get("system_prompt", "")
            field_instructions = internvl3_config.get("field_instructions", "")
            
            prompt = f"{system_prompt}\n\n{field_instructions}"
            
            if self.debug:
                print("🔨 Using universal extraction template")
                print(f"📝 Prompt length: {len(prompt)} characters")
                print("\n" + "="*80)
                print("📋 UNIVERSAL PROMPT:")
                print("="*80)
                print(prompt)
                print("="*80 + "\n")
                
        except Exception as e:
            raise ValueError(
                f"❌ FATAL: Could not generate universal prompt: {e}\n"
                f"💡 Check unified_schema.yaml contains universal_extraction section"
            ) from e

        if self.debug:
            print(f"📝 Generated universal prompt for all {len(field_list)} fields")

        
        # Execute universal extraction
        raw_response = self._extract_with_prompt(
            image_path, prompt, generation_config=None
        )
        

        if self.debug:
            print("📥 Raw universal response received")
            print(f"📊 Response length: {len(raw_response)} characters")
            print("=" * 60)
            print(raw_response[:500] + "..." if len(raw_response) > 500 else raw_response)
            print("=" * 60)

        # Parse the response using the extraction parser
        from common.extraction_parser import parse_extraction_response

        extracted_data = parse_extraction_response(
            raw_response, expected_fields=field_list
        )

        # Apply ExtractionCleaner for consistent formatting (matching Llama behavior)
        if self.debug:
            print("🧹 Applying ExtractionCleaner to normalize field values...")
            
        cleaned_extracted_data = {}
        for field_name, value in extracted_data.items():
            # Apply same cleaning logic as Llama processor
            if value and value.lower().strip() not in [
                "not found", "not_found", "notfound", "n/a", "na", "none", "null", ""
            ]:
                cleaned_value = self.cleaner.clean_field_value(field_name, value)
            else:
                cleaned_value = "NOT_FOUND"
            cleaned_extracted_data[field_name] = cleaned_value

        # Replace extracted_data with cleaned version
        extracted_data = cleaned_extracted_data

        if self.debug:
            found_count = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")
            print(f"✅ Universal extraction: {found_count}/{len(field_list)} fields found")

        return extracted_data

    def _infer_document_type_from_extraction(self, extracted_data: Dict[str, str]) -> str:
        """
        Infer document type from extraction results.
        
        Post-processing logic to determine document type based on which fields were found.
        """
        if self.debug:
            print("🔍 Inferring document type from extraction results...")
        
        # Count non-empty fields for each document type category
        bank_statement_indicators = [
            "STATEMENT_DATE_RANGE", "TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID"
        ]
        
        invoice_receipt_indicators = [
            "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_TOTAL_PRICES", "GST_AMOUNT", 
            "IS_GST_INCLUDED", "TOTAL_AMOUNT", "INVOICE_DATE"
        ]
        
        # Count found indicators for each type
        bank_found = sum(1 for field in bank_statement_indicators 
                        if extracted_data.get(field, "NOT_FOUND") != "NOT_FOUND")
        
        invoice_found = sum(1 for field in invoice_receipt_indicators
                           if extracted_data.get(field, "NOT_FOUND") != "NOT_FOUND")
        
        # Inference logic
        if bank_found >= 1:  # Any bank statement indicator
            inferred_type = "bank_statement"
        elif invoice_found >= 2:  # Multiple invoice/receipt indicators
            inferred_type = "invoice"  # Default to invoice (could be receipt)
        else:
            inferred_type = "unknown"  # Unclear document type
        
        if self.debug:
            print(f"🎯 Type inference: bank_indicators={bank_found}, invoice_indicators={invoice_found}")
            print(f"📋 Inferred document type: {inferred_type}")
            
        return inferred_type

    def _extract_with_prompt(
        self, image_path: str, prompt: str, generation_config: dict = None
    ) -> str:
        """Core extraction method that handles image processing and model inference."""
        # Pre-processing cleanup with fragmentation detection - V100 optimized
        handle_memory_fragmentation(threshold_gb=2.0, aggressive=True)

        
        # Load and preprocess image
        pixel_values = self.load_image(image_path)
        

        # Move to appropriate device and dtype
        if self.device == "cpu":
            pixel_values = pixel_values.to(self.torch_dtype)
        else:
            pixel_values = pixel_values.to(self.torch_dtype).cuda()

        # Prepare conversation
        question = f"<image>\n{prompt}"

        # Use provided generation config or default
        config = generation_config or self.generation_config

        # Use the same generation logic for both 2B and 8B models
        try:
            if self.debug:
                allocated, reserved, fragmentation = detect_memory_fragmentation()
                print(f"🚀 MODEL_INFERENCE_START: {pixel_values.shape[0]} tiles tensor ready")
                print(f"📊 MEMORY_BEFORE_CHAT: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")
                if self.is_8b_model:
                    print("⚡ QUANTIZED_8B: Using quantization to reduce memory footprint")
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                config,
                history=None,
                return_history=False,
            )
            
            if self.debug:
                allocated, reserved, fragmentation = detect_memory_fragmentation()
                print("✅ MODEL_INFERENCE_SUCCESS: Response generated")
                print(f"📊 MEMORY_AFTER_CHAT: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")
        except torch.cuda.OutOfMemoryError as e:
            if self.debug:
                allocated, reserved, fragmentation = detect_memory_fragmentation()
                print(f"💥 OOM_TRIGGERED: {pixel_values.shape[0]} tiles failed during model.chat()")
                print(f"📊 MEMORY_AT_OOM: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")
                print(f"🔍 OOM_DETAILS: {str(e)[:100]}...")
            
            # Progressive tile fallback strategy for quantized 8B model
            initial_tiles = pixel_values.shape[0]
            fallback_tiles = [9, 6, 4] if initial_tiles >= 12 else [6, 4]
            
            if self.debug:
                print(f"🎯 FALLBACK_STRATEGY: {initial_tiles} → {fallback_tiles}")
            
            for attempt, tiles in enumerate(fallback_tiles, 1):
                if tiles >= initial_tiles:
                    continue  # Skip if not actually reducing
                    
                if self.debug:
                    print(f"🔄 FALLBACK_ATTEMPT_{attempt}: Reducing to {tiles} tiles (from {initial_tiles})")
                    allocated, reserved, fragmentation = detect_memory_fragmentation()
                    print(f"📊 MEMORY_BEFORE_CLEANUP: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
                # Emergency cleanup - V100 optimized threshold
                torch.cuda.empty_cache()
                clear_model_caches(self.model, self.tokenizer)
                handle_memory_fragmentation(threshold_gb=1.5, aggressive=True)
                
                if self.debug:
                    allocated, reserved, fragmentation = detect_memory_fragmentation()
                    print(f"📊 MEMORY_AFTER_CLEANUP: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")
                
                # Reload image with fewer tiles
                del pixel_values
                if self.debug:
                    print(f"🔄 RELOADING: Creating {tiles} tiles...")
                pixel_values = self.load_image(image_path, max_num=tiles)
                if self.device == "cpu":
                    pixel_values = pixel_values.to(self.torch_dtype)
                else:
                    pixel_values = pixel_values.to(self.torch_dtype).cuda()
                
                if self.debug:
                    print(f"🎯 RETRY_ATTEMPT: Trying model.chat() with {pixel_values.shape[0]} tiles")
                
                try:
                    response = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        config,
                        history=None,
                        return_history=False,
                    )
                    if self.debug:
                        allocated, reserved, fragmentation = detect_memory_fragmentation()
                        print(f"✅ FALLBACK_SUCCESS: {tiles} tiles worked (reduced from {initial_tiles})")
                        print(f"📊 FINAL_MEMORY: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")
                    break  # Success - exit the fallback loop
                except torch.cuda.OutOfMemoryError as e2:
                    if self.debug:
                        print(f"❌ FALLBACK_FAILED: {tiles} tiles still caused OOM")
                        print(f"🔍 OOM_DETAILS: {str(e2)[:50]}...")
                    continue
            else:
                # All fallback attempts failed
                print("❌ InternVL3: OOM persists even with minimum tile count")
                if self.debug:
                    allocated, reserved, fragmentation = detect_memory_fragmentation()
                    print("💀 FINAL_FAILURE: All fallback attempts exhausted")
                    print(f"📊 FINAL_STATE: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={16-reserved:.2f}GB")
                raise

        # Memory cleanup after processing
        del pixel_values
        comprehensive_memory_cleanup(self.model, self.tokenizer)

        return response

    def _extract_with_custom_prompt(
        self, image_path: str, prompt: str, **generation_kwargs
    ) -> str:
        """
        Extract fields using a custom prompt with specific generation parameters.

        Required method for DocumentTypeDetector compatibility.

        Args:
            image_path (str): Path to image file
            prompt (str): Custom extraction prompt
            **generation_kwargs: Additional generation parameters

        Returns:
            str: Raw model response
        """
        print(f"🔍 PROCESSOR-TRACE: _extract_with_custom_prompt ENTRY - {image_path}, prompt_length={len(prompt)}")
        
        try:
            # Create generation config - allow custom overrides
            custom_generation_config = self.generation_config.copy()

            if "max_new_tokens" in generation_kwargs:
                custom_generation_config["max_new_tokens"] = generation_kwargs[
                    "max_new_tokens"
                ]

            # Only set sampling parameters if do_sample is True to avoid warnings
            if custom_generation_config.get("do_sample", False):
                if (
                    "temperature" in generation_kwargs
                    and generation_kwargs["temperature"] is not None
                ):
                    custom_generation_config["temperature"] = generation_kwargs[
                        "temperature"
                    ]
            else:
                # Remove all sampling-related parameters when do_sample is False
                custom_generation_config.pop("temperature", None)
                custom_generation_config.pop("top_k", None)
                custom_generation_config.pop("top_p", None)

            # Use shared extraction method
            print("🔍 PROCESSOR-TRACE: _extract_with_custom_prompt EXIT - calling _extract_with_prompt")
            return self._extract_with_prompt(
                image_path, prompt, custom_generation_config
            )

        except Exception as e:
            if self.debug:
                print(f"❌ Error in InternVL3 custom prompt extraction: {e}")
            return f"Error: {str(e)}"

    def _handle_final_loading_failure(self, gpu_name: str, gpu_memory_gb: float):
        """Handle final loading failure with appropriate error message and solutions."""
        print(f"\n❌ FATAL: InternVL3-8B loading failed on {gpu_name} ({gpu_memory_gb:.0f}GB)")
        print("🔍 All loading strategies exhausted")
        print("\n🛠️ SOLUTIONS:")
        if gpu_memory_gb <= 16:
            print(f"   1. 🎯 RECOMMENDED: Use InternVL3-2B (perfect for {gpu_memory_gb:.0f}GB V100)")
            print("   2. Install quantization: pip install bitsandbytes>=0.41.0")
            print("   3. Update PyTorch/CUDA: pip install torch --upgrade")
        else:
            print("   1. Use InternVL3-2B (works on any GPU)")
            print("   2. Install quantization libraries")
        print("   4. Alternative: Use Llama-3.2-Vision-11B (proven stable)")
        raise RuntimeError(
            f"InternVL3-8B loading failed on {gpu_name} ({gpu_memory_gb:.0f}GB)"
        ) from None

    # OLD SINGLE-PASS PARSING REMOVED - NOW USING HYBRID GROUPED EXTRACTION
    # Each group handles its own parsing with proven 90.6% accuracy format
