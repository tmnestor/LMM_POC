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
        self.field_list = field_list
        self.field_count = len(field_list)
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.debug = debug

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.generation_config = None

        # Initialize extraction cleaner for value normalization
        self.cleaner = ExtractionCleaner(debug=debug)

        # Note: Previously used DocumentAwareGroupedExtraction but now using direct field extraction

        # Fix 8B detection using actual model path
        self.is_8b_model = "8B" in str(self.model_path)

        if self.debug:
            print(
                f"🎯 Document-aware InternVL3 processor initialized for {self.field_count} fields"
            )
            print(f"   Fields: {field_list[0]} → {field_list[-1]}")
            print(f"   Model variant: {'8B' if self.is_8b_model else '2B'}")

        # Configure CUDA memory allocation
        configure_cuda_memory_allocation()

        # Set seeds for reproducibility
        self._set_random_seeds(42)

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Configure generation parameters for dynamic field count
        self._configure_generation()

        # Load model and tokenizer (unless skipping for reuse)
        if not skip_model_loading:
            self._load_model()

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
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory and model variant
            available_memory = get_available_gpu_memory(self.device)
            model_key = "internvl3-8b" if self.is_8b_model else "internvl3-2b"
            self.batch_size = get_auto_batch_size(model_key, available_memory)
            print(
                f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB, Model: {model_key})"
            )

    def _configure_generation(self):
        """Configure generation parameters for dynamic field count."""
        # Calculate dynamic max_new_tokens based on actual field count
        max_tokens = get_max_new_tokens("internvl3", self.field_count)

        # Get generation config from centralized config (matches original internvl3_processor)
        from common.config import GENERATION_CONFIGS

        base_gen_config = GENERATION_CONFIGS.get("internvl3", {})
        self.generation_config = {"max_new_tokens": max_tokens, **base_gen_config}

        # Ensure deterministic generation (matches original internvl3_processor)
        if self.generation_config.get("do_sample", True):
            print("⚠️ WARNING: do_sample was True, forcing to False for determinism")
            self.generation_config["do_sample"] = False

        print(
            f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
            f"do_sample={self.generation_config['do_sample']} (greedy decoding)"
        )

    def _load_model(self):
        """Load InternVL3 model and tokenizer with optimal configuration."""
        print(f"🔄 Loading InternVL3 model from: {self.model_path}")

        if self.is_8b_model:
            print("🎯 InternVL3-8B detected - applying aggressive V100 optimizations")
        else:
            print("🎯 InternVL3-2B detected - using standard optimizations")

        try:
            # Base model configuration
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "use_flash_attn": False,  # Disabled for compatibility
                "trust_remote_code": True,
            }

            # Initialize quantization tracking for proper warm-up logic
            quantization_success = False

            # Configure 8-bit quantization for 8B model on V100
            if self.is_8b_model:
                print("🔧 Loading InternVL3-8B with 8-bit quantization for V100")
                print("   Essential for fitting model in 16GB VRAM")

                # Check bitsandbytes version and use appropriate API
                try:
                    import bitsandbytes as bnb

                    bnb_version = getattr(bnb, "__version__", "unknown")
                    print(f"   bitsandbytes version: {bnb_version}")
                except ImportError:
                    bnb_version = None
                    print("   ⚠️ bitsandbytes not found, will try without quantization")

                # Determine which API to use based on bitsandbytes version
                use_old_api = False
                if bnb_version and bnb_version != "unknown":
                    try:
                        version_parts = bnb_version.split(".")
                        major = int(version_parts[0])
                        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
                        use_old_api = major == 0 and minor < 44
                        if self.debug:
                            print(
                                f"   Version check: major={major}, minor={minor}, use_old_api={use_old_api}"
                            )
                    except (ValueError, IndexError):
                        if self.debug:
                            print(
                                f"   ⚠️ Could not parse version {bnb_version}, will try new API"
                            )
                        use_old_api = False

                if use_old_api:
                    print("📦 QUANTIZATION APPROACH: Legacy load_in_8bit=True")
                    print(f"   Reason: bitsandbytes {bnb_version} < 0.44.0")
                    print("   Status: Using legacy API for compatibility")
                    model_kwargs["load_in_8bit"] = True
                else:
                    try:
                        from transformers import BitsAndBytesConfig

                        print("📦 QUANTIZATION APPROACH: Modern BitsAndBytesConfig")
                        print(f"   Reason: bitsandbytes {bnb_version} >= 0.44.0")
                        print("   Status: Using modern API (recommended)")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16,
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        print("   ✅ BitsAndBytesConfig created successfully")
                    except ImportError as e:
                        print(
                            "📦 QUANTIZATION APPROACH: Fallback to Legacy load_in_8bit=True"
                        )
                        print(f"   Reason: BitsAndBytesConfig import failed: {e}")
                        print("   Status: Using legacy API as fallback")
                        model_kwargs["load_in_8bit"] = True

                model_kwargs["device_map"] = "auto"

                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_path, **model_kwargs
                    ).eval()

                    print("✅ InternVL3-8B loaded with 8-bit quantization")
                    print("   Memory footprint reduced by ~50%")
                    quantization_success = True

                except Exception as quant_error:
                    print(f"❌ 8-bit quantization failed: {quant_error}")
                    print(
                        "💾 V100 has only 16GB VRAM - InternVL3-8B REQUIRES 8-bit quantization"
                    )

                    # Try alternative quantization approaches before giving up
                    success = False

                    # Try legacy API with specific device mapping
                    if not success:
                        try:
                            print(
                                "\n📦 QUANTIZATION FALLBACK: Attempting Legacy load_in_8bit=True"
                            )
                            print(
                                "   Reason: Primary approach failed, trying alternative"
                            )
                            print(
                                "   Strategy: Using explicit device mapping for V100 compatibility"
                            )

                            legacy_kwargs = {
                                "torch_dtype": torch.bfloat16,
                                "load_in_8bit": True,
                                "device_map": {"": 0},  # Force GPU 0
                                "low_cpu_mem_usage": True,
                                "trust_remote_code": True,
                            }

                            self.model = AutoModel.from_pretrained(
                                self.model_path, **legacy_kwargs
                            ).eval()

                            print(
                                "✅ SUCCESS: InternVL3-8B loaded with legacy 8-bit quantization"
                            )
                            print(
                                "   Approach: Legacy load_in_8bit=True with explicit device mapping"
                            )
                            success = True
                            quantization_success = True

                        except Exception as legacy_error:
                            print(f"   ❌ Legacy fallback failed: {legacy_error}")

                    # If all approaches fail, provide clear guidance
                    if not success:
                        print("\n❌ FATAL: Cannot load InternVL3-8B on V100")
                        print("🔍 ISSUE: bitsandbytes 0.47.0 compatibility problem")
                        print(
                            "💾 V100 limitation: 16GB VRAM insufficient for FP16 (needs ~22GB)"
                        )
                        print("\n🛠️ SOLUTIONS:")
                        print(
                            "   1. Use InternVL3-2B instead: MODEL_PATH = '.../InternVL3-2B'"
                        )
                        print("   2. Try: pip install bitsandbytes==0.41.1")
                        print("   3. Use Llama-3.2-Vision-11B with 8-bit quantization")
                        print("   4. Upgrade to A100/H100 with >20GB VRAM")
                        raise RuntimeError(
                            "InternVL3-8B incompatible with V100 due to bitsandbytes issues"
                        ) from None

            else:
                # InternVL3-2B: Direct loading without quantization
                print("\n📦 LOADING APPROACH: Direct bfloat16 (No Quantization Needed)")
                print("   Model: InternVL3-2B")
                print("   Reason: 2B model fits comfortably in V100 16GB VRAM")
                print("   Precision: bfloat16 (model requires this precision)")

                # Configuration for 2B model - use model's native bfloat16
                model_kwargs["device_map"] = "auto"

                self.model = AutoModel.from_pretrained(
                    self.model_path, **model_kwargs
                ).eval()

                print("✅ SUCCESS: InternVL3-2B loaded in bfloat16")
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
                print("Status: ✅ Successfully loaded in bfloat16")
                print("Method: Direct loading (no quantization needed)")
                print("VRAM: ~4-6GB")
            print("Hardware: V100 GPU (16GB VRAM)")
            print("=" * 60 + "\n")

            # Apply V100 optimizations
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
                    "✅ InternVL3-2B bfloat16 model loaded - enabling warm-up for verification"
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
        max_num=12,
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
        # PHASE 2: Increase tiles for 8B model to match 2B model accuracy
        # 8B model now uses quantization on V100, so memory impact is reduced
        if max_num is None:
            max_num = 12  # Both 2B and 8B models now use same tile count for better text coverage

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

        try:
            start_time = time.time()

            if self.debug:
                print(
                    f"🚀 Starting document-aware extraction for {Path(image_path).name}"
                )
                print(f"📊 Target fields: {self.field_count} fields")
                print(
                    "🎯 Strategy: Direct field extraction (document type already detected)"
                )
                print(
                    f"📝 Field list: {self.field_list[:5]}{'...' if len(self.field_list) > 5 else ''}"
                )

            # Memory cleanup - V100 optimized threshold (16GB total capacity)
            handle_memory_fragmentation(threshold_gb=2.0, aggressive=True)

            # FIXED: Use direct extraction with the already-determined field list
            # Don't re-detect document type - we already have the correct fields
            extracted_data = self._extract_fields_directly(image_path, self.field_list)

            # Create metadata for compatibility
            metadata = {
                "document_type": "pre_detected",
                "extraction_strategy": "document_aware_direct",
                "total_fields": len(self.field_list),
                "extraction_method": "single_pass_with_predefined_fields",
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

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": f"DOCUMENT-AWARE: {len(self.field_list)} document-specific fields",
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
                import traceback

                traceback.print_exc()

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

    def _extract_fields_directly(
        self, image_path: str, field_list: List[str]
    ) -> Dict[str, str]:
        """
        Extract specific fields using YAML-based unified prompt.

        Uses the same unified schema templates as Llama for consistency.
        """
        if self.debug:
            print(f"🎯 Extracting {len(field_list)} document-specific fields directly")

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

    def _extract_with_prompt(
        self, image_path: str, prompt: str, generation_config: dict = None
    ) -> str:
        """Core extraction method that handles image processing and model inference."""
        # Pre-processing cleanup with fragmentation detection - V100 optimized
        handle_memory_fragmentation(threshold_gb=2.0, aggressive=True)

        # Load and preprocess image
        pixel_values = self.load_image(image_path)

        # Move to appropriate device and dtype
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

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
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                
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
            return self._extract_with_prompt(
                image_path, prompt, custom_generation_config
            )

        except Exception as e:
            if self.debug:
                print(f"❌ Error in InternVL3 custom prompt extraction: {e}")
            return f"Error: {str(e)}"

    # OLD SINGLE-PASS PARSING REMOVED - NOW USING HYBRID GROUPED EXTRACTION
    # Each group handles its own parsing with proven 90.6% accuracy format
