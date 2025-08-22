"""
InternVL3-specific processor for vision model evaluation.

This module contains all InternVL3-specific code including model loading,
image preprocessing, and batch processing logic.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from common.config import (
    BATCH_SIZE_FALLBACK_STEPS,
    CLEAR_GPU_CACHE_AFTER_BATCH,
    DEFAULT_EXTRACTION_MODE,
    DEFAULT_IMAGE_SIZE,
    ENABLE_BATCH_SIZE_FALLBACK,
    EXTRACTION_FIELDS,
    FIELD_COUNT,
    FIELD_INSTRUCTIONS,
    GENERATION_CONFIGS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INTERNVL3_MODEL_PATH,
    INTERNVL3_TOKEN_LIMITS,
    get_auto_batch_size,
    get_max_new_tokens,
    get_model_name_with_size,
)
from common.evaluation_utils import parse_extraction_response
from common.gpu_optimization import (
    clear_model_caches,
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    get_available_gpu_memory,
    handle_memory_fragmentation,
    optimize_model_for_v100,
)
from common.grouped_extraction import get_extraction_strategy


class InternVL3Processor:
    """Processor for InternVL3 vision-language model."""

    def __init__(
        self,
        model_path=None,
        device="cuda",
        batch_size=None,
        extraction_mode=None,
        debug=False,
        grouping_strategy="8_groups",
    ):
        """
        Initialize InternVL3 processor with model and tokenizer.

        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
            batch_size (int): Batch size for processing (auto-detected if None)
            extraction_mode (str): Extraction mode ('single_pass', 'grouped', 'adaptive')
            debug (bool): Enable debug logging for extraction
            grouping_strategy (str): Grouping strategy ('8_groups' or '6_groups')
        """
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.model = None
        self.tokenizer = None

        # Configure extraction strategy
        self.extraction_mode = extraction_mode or DEFAULT_EXTRACTION_MODE
        self.debug = debug
        self.extraction_strategy = get_extraction_strategy(self.extraction_mode, debug, grouping_strategy)
        self.generation_config = None
        # Fix 8B detection to use actual model path (after setting default)
        self.is_8b_model = "8B" in str(self.model_path)

        # Configure CUDA memory allocation for V100 optimization
        configure_cuda_memory_allocation()

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Initialize model and tokenizer
        self._load_model()

        # Setup generation config from centralized configuration
        model_size = "8b" if self.is_8b_model else "2b"

        # Get token limit from config
        max_tokens = INTERNVL3_TOKEN_LIMITS.get(model_size)
        if max_tokens is None:
            max_tokens = get_max_new_tokens("internvl3", FIELD_COUNT)
            print(
                f"🎯 InternVL3-{model_size.upper()}: Using calculated tokens ({max_tokens})"
            )
        else:
            print(
                f"🎯 InternVL3-{model_size.upper()}: Using configured tokens ({max_tokens})"
            )

        # Get generation config from centralized config
        base_gen_config = GENERATION_CONFIGS.get("internvl3", {})
        self.generation_config = {"max_new_tokens": max_tokens, **base_gen_config}

        print(
            f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
            f"do_sample={self.generation_config['do_sample']}"
        )

        # Don't use ResilientGenerator - it breaks the 8B model
        # Both 2B and 8B should use model.chat() directly as per documentation
        if self.is_8b_model:
            print("✅ InternVL3-8B will use direct model.chat() like 2B model")

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        model_key = "internvl3-8b" if self.is_8b_model else "internvl3-2b"

        # Use configured batch size if none provided
        if batch_size is None:
            from common.config import DEFAULT_BATCH_SIZES

            self.batch_size = DEFAULT_BATCH_SIZES.get(model_key, 1)
            print(f"🎯 Using configured batch size: {self.batch_size} for {model_key}")
        else:
            self.batch_size = max(1, batch_size)  # Ensure minimum batch size of 1
            print(f"🎯 Using manual batch size: {self.batch_size}")

            # Warn if manual batch size exceeds configured safe limits
            from common.config import DEFAULT_BATCH_SIZES

            safe_limit = DEFAULT_BATCH_SIZES.get(model_key, 1)
            if self.batch_size > safe_limit:
                print(
                    f"⚠️ Warning: batch_size={self.batch_size} exceeds safe limit ({safe_limit}) for {model_key}"
                )

        # Auto-detection fallback (shouldn't be reached with current logic)
        if not hasattr(self, "batch_size"):
            # Auto-detect batch size based on available memory and model size
            available_memory = get_available_gpu_memory(self.device)
            size_aware_model_name = get_model_name_with_size(
                "internvl3", self.model_path, self.is_8b_model
            )
            self.batch_size = get_auto_batch_size(
                size_aware_model_name, available_memory
            )
            print(
                f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB, Model: {size_aware_model_name})"
            )

    def _load_model(self):
        """Load InternVL3 model and tokenizer with compatibility settings."""
        print(f"🔧 Loading InternVL3 model from: {self.model_path}")

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
                    # Parse version more carefully
                    try:
                        version_parts = bnb_version.split(".")
                        major = int(version_parts[0])
                        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
                        # Use old API for versions < 0.44.0
                        use_old_api = major == 0 and minor < 44
                        print(
                            f"   Version check: major={major}, minor={minor}, use_old_api={use_old_api}"
                        )
                    except (ValueError, IndexError):
                        print(
                            f"   ⚠️ Could not parse version {bnb_version}, will try new API"
                        )
                        use_old_api = False

                if use_old_api:
                    print(
                        f"   Using load_in_8bit for bitsandbytes {bnb_version} (old API)"
                    )
                    model_kwargs["load_in_8bit"] = True
                else:
                    # Try to use BitsAndBytesConfig for newer versions
                    try:
                        from transformers import BitsAndBytesConfig

                        print(
                            f"   Using BitsAndBytesConfig for bitsandbytes {bnb_version} (new API)"
                        )
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16,
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        print("   ✅ BitsAndBytesConfig created successfully")
                    except ImportError as e:
                        # If BitsAndBytesConfig not available, fall back to old API
                        print(f"   ⚠️ BitsAndBytesConfig import failed: {e}")
                        print(
                            "   Falling back to load_in_8bit (expect deprecation warning)"
                        )
                        model_kwargs["load_in_8bit"] = True

                model_kwargs["device_map"] = "auto"  # Let it handle device placement

                self.model = AutoModel.from_pretrained(
                    self.model_path, **model_kwargs
                ).eval()

                print("✅ InternVL3-8B loaded with 8-bit quantization")
                print("   Memory footprint reduced by ~50%")

            else:
                # 2B model doesn't need quantization
                print("🔧 Loading InternVL3-2B model...")
                model_kwargs["device_map"] = "auto"

                self.model = AutoModel.from_pretrained(
                    self.model_path, **model_kwargs
                ).eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )

            print("✅ InternVL3 model and tokenizer loaded successfully")

            # Apply V100 optimizations
            optimize_model_for_v100(self.model)

            # Note: Gradient checkpointing removed - only useful for training, not inference

            # Skip warm-up for 8B model with quantization - causes shape mismatch warnings
            # This is expected behavior with 8-bit quantization
            if self.is_8b_model:
                print(
                    "⚠️ Skipping warm-up test for quantized 8B model (expected with 8-bit)"
                )
                print("   Model will be tested during actual inference")

        except Exception as e:
            print(f"❌ Error loading InternVL3 model: {e}")
            raise

    def get_extraction_prompt(self):
        """Get the extraction prompt for InternVL3."""
        # Use the same comprehensive prompt for both models now that 8B has proper quantization
        prompt = f"""Extract data from this business document. 
Output ALL fields below with their exact keys. 
Use "N/A" if field is not visible or not present.

OUTPUT FORMAT ({FIELD_COUNT} required fields):
"""
        # Add all fields with centralized field-specific instructions
        for field in EXTRACTION_FIELDS:
            instruction = FIELD_INSTRUCTIONS.get(field, "[value or N/A]")
            prompt += f"{field}: {instruction}\n"

        prompt += f"""
INSTRUCTIONS:
- Keep field names EXACTLY as shown above
- Use "N/A" for any missing/unclear information
- Do not add explanations or comments
- Extract actual values from the document image
- Output exactly {FIELD_COUNT} lines, one for each field"""

        if self.debug:
            print(f"📝 SINGLE-PASS PROMPT: {len(prompt)} chars, {len(EXTRACTION_FIELDS)} fields")
            print("📝 PROMPT CONTENT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)

        return prompt

    def build_transform(self, input_size=DEFAULT_IMAGE_SIZE):
        """
        Build InternVL3 image transformation pipeline.

        Args:
            input_size: Target size for image resizing

        Returns:
            torchvision.transforms.Compose: Transformation pipeline
        """
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
        """Find closest aspect ratio for InternVL3 dynamic preprocessing."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)

            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
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
        """
        InternVL3 dynamic preprocessing algorithm.

        Args:
            image: PIL Image to preprocess
            min_num: Minimum number of tiles
            max_num: Maximum number of tiles
            image_size: Size of each tile
            use_thumbnail: Whether to include thumbnail

        Returns:
            list: List of preprocessed image tiles
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

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

        return processed_images

    def load_image(self, image_file, input_size=DEFAULT_IMAGE_SIZE, max_num=None):
        """
        Complete InternVL3 image loading and preprocessing pipeline.

        Args:
            image_file: Path to image file or PIL Image
            input_size: Size for each tile
            max_num: Maximum number of tiles

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # For 8B model, use fewer tiles to reduce memory
        if max_num is None:
            max_num = 6 if self.is_8b_model else 12  # Reduce tiles for 8B model

        # Load image if path provided
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        else:
            image = image_file.convert("RGB")

        # Apply dynamic preprocessing
        images = self.dynamic_preprocess(image, image_size=input_size, max_num=max_num)

        # Apply transforms
        transform = self.build_transform(input_size=input_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values

    def _extract_with_prompt(self, image_path: str, prompt: str, generation_config: dict = None) -> str:
        """
        Core extraction method that handles image processing and model inference.
        
        This is the shared implementation used by both single-pass and grouped extraction.
        
        Args:
            image_path (str): Path to image file
            prompt (str): Extraction prompt
            generation_config (dict): Optional custom generation config
            
        Returns:
            str: Raw model response
        """
        # Pre-processing cleanup with fragmentation detection
        handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

        # Load and preprocess image
        pixel_values = self.load_image(image_path)

        # Move to appropriate device and dtype
        # Both models now use GPU with bfloat16
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # Prepare conversation
        question = f"<image>\n{prompt}"
        
        # Use provided generation config or default
        config = generation_config or self.generation_config

        # Use the same generation logic for both 2B and 8B models
        # Documentation shows using model.chat() directly for both
        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                config,
                history=None,
                return_history=False,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ InternVL3 OOM: {e}")
            print("🔄 Attempting emergency cleanup and retry...")

            # Emergency cleanup (skip normal post-processing cleanup)
            torch.cuda.empty_cache()
            clear_model_caches(self.model, self.tokenizer)
            handle_memory_fragmentation(threshold_gb=0.5, aggressive=True)

            # Retry once
            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    config,
                    history=None,
                    return_history=False,
                )
            except torch.cuda.OutOfMemoryError:
                print("❌ InternVL3: Even with cleanup, OOM persists")
                raise

        # Memory cleanup after processing
        del pixel_values
        comprehensive_memory_cleanup(self.model, self.tokenizer)
        
        return response

    def process_single_image(self, image_path):
        """
        Process a single image through InternVL3 extraction pipeline.

        Args:
            image_path (str): Path to image file

        Returns:
            dict: Extraction results with metadata
        """
        try:
            start_time = time.time()
            
            # Use shared extraction method
            response = self._extract_with_prompt(image_path, self.get_extraction_prompt())
            
            processing_time = time.time() - start_time

            # Parse response
            extracted_data = parse_extraction_response(response)

            if self.debug:
                print("🔍 RAW MODEL RESPONSE (single-pass):")
                print("-" * 40)
                print(response)
                print("-" * 40)
                print("🔍 PARSED DATA (single-pass):")
                for field, value in list(extracted_data.items())[:5]:  # Show first 5 fields
                    print(f"  {field}: {value}")
                print(f"  ... and {len(extracted_data) - 5} more fields")
                print()

            # Calculate metrics - count ALL fields that are present (including correct N/A)
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]
            )
            response_completeness = extracted_fields_count / len(EXTRACTION_FIELDS)
            content_coverage = extracted_fields_count / len(EXTRACTION_FIELDS)

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": processing_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "raw_response_length": len(response),
            }

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

            # Emergency cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("🧹 Emergency cleanup after error")

            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "N/A" for field in EXTRACTION_FIELDS},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "raw_response_length": 0,
            }

    def _extract_with_custom_prompt(
        self, image_path: str, prompt: str, **generation_kwargs
    ) -> str:
        """
        Extract fields using a custom prompt with specific generation parameters.

        Args:
            image_path (str): Path to image file
            prompt (str): Custom extraction prompt
            **generation_kwargs: Additional generation parameters (for group-specific settings)

        Returns:
            str: Raw model response
        """
        try:
            # Create generation config - allow group-specific overrides
            custom_generation_config = self.generation_config.copy()
            if "max_new_tokens" in generation_kwargs:
                custom_generation_config["max_new_tokens"] = generation_kwargs["max_new_tokens"]
            # Only set temperature if do_sample is True to avoid warnings
            if custom_generation_config.get("do_sample", False):
                if "temperature" in generation_kwargs and generation_kwargs["temperature"] is not None:
                    custom_generation_config["temperature"] = generation_kwargs["temperature"]
            else:
                # Remove temperature if do_sample is False
                custom_generation_config.pop("temperature", None)

            # Use shared extraction method
            return self._extract_with_prompt(image_path, prompt, custom_generation_config)

        except Exception as e:
            print(f"❌ Error in InternVL3 custom prompt extraction: {e}")
            return f"Error: {str(e)}"

    def process_single_image_grouped(self, image_path: str) -> dict:
        """
        Process a single image using grouped extraction strategy.

        Args:
            image_path (str): Path to image file

        Returns:
            dict: Extraction results with group metadata
        """
        if self.extraction_mode == "single_pass":
            # Fallback to single-pass extraction
            return self.process_single_image(image_path)

        start_time = time.time()

        try:
            if self.debug:
                print(
                    f"🔍 Processing {Path(image_path).name} with InternVL3 {self.extraction_mode} mode"
                )

            # Memory cleanup before processing
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            if self.extraction_mode == "grouped":
                # Use grouped extraction strategy
                extracted_data, metadata = (
                    self.extraction_strategy.extract_fields_grouped(
                        image_path, self._extract_with_custom_prompt
                    )
                )
            elif self.extraction_mode == "adaptive":
                # Use adaptive extraction strategy
                extracted_data, metadata = (
                    self.extraction_strategy.extract_fields_adaptive(
                        image_path,
                        lambda path: self.process_single_image(path)["extracted_data"],
                        self._extract_with_custom_prompt,
                    )
                )
            else:
                raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")

            # Calculate standard metrics for compatibility - count ALL present fields
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]
            )
            response_completeness = extracted_fields_count / len(EXTRACTION_FIELDS)
            content_coverage = extracted_fields_count / len(EXTRACTION_FIELDS)

            total_processing_time = time.time() - start_time

            result = {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "processing_time": total_processing_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "extraction_mode": self.extraction_mode,
                "group_metadata": metadata,
            }

            if self.debug:
                print(
                    f"✅ InternVL3 grouped extraction completed in {total_processing_time:.2f}s"
                )
                print(
                    f"📊 Extracted {extracted_fields_count}/{len(EXTRACTION_FIELDS)} fields"
                )

            return result

        except Exception as e:
            print(f"❌ Error in InternVL3 grouped extraction for {image_path}: {e}")

            # Emergency cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "N/A" for field in EXTRACTION_FIELDS},
                "processing_time": time.time() - start_time,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "extraction_mode": self.extraction_mode,
                "error": str(e),
            }

    def process_image_batch(
        self, image_files: List[str], progress_callback=None
    ) -> Tuple[list, dict]:
        """
        Process batch of images through InternVL3 extraction pipeline with true batch processing.

        Args:
            image_files (List[str]): List of image file paths
            progress_callback (callable): Optional callback for progress updates

        Returns:
            Tuple[list, dict]: (results, statistics) - Extraction results and batch statistics
        """
        if not image_files:
            return [], {
                "total_images": 0,
                "successful_extractions": 0,
                "total_processing_time": 0,
                "average_processing_time": 0,
                "success_rate": 0,
            }

        print(
            f"\n🚀 Processing {len(image_files)} images with InternVL3 (batch_size={self.batch_size})..."
        )

        results = []
        total_processing_time = 0
        successful_extractions = 0

        # Process images in batches
        for batch_start in range(0, len(image_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]

            # Progress update for batch
            if progress_callback:
                progress_callback(
                    batch_end,
                    len(image_files),
                    f"Batch {batch_start // self.batch_size + 1}",
                )
            else:
                print(
                    f"\n[Batch {batch_start // self.batch_size + 1}] Processing images {batch_start + 1}-{batch_end} of {len(image_files)}"
                )

            # Process current batch with fallback mechanism
            batch_results = self._process_batch_with_fallback(batch_files)
            results.extend(batch_results)

            # Update statistics
            for result in batch_results:
                total_processing_time += result["processing_time"]
                if result["response_completeness"] > 0:
                    successful_extractions += 1

            # Clear GPU cache after each batch with comprehensive cleanup
            # For 8B model, do more aggressive cleanup
            if CLEAR_GPU_CACHE_AFTER_BATCH and torch.cuda.is_available():
                if self.is_8b_model:
                    # Extra aggressive cleanup for 8B model
                    comprehensive_memory_cleanup(self.model, self.tokenizer)
                    handle_memory_fragmentation(threshold_gb=0.5, aggressive=True)
                    print(
                        f"   🧹 Aggressive memory cleanup after batch {batch_start // self.batch_size + 1}"
                    )
                else:
                    comprehensive_memory_cleanup(self.model, self.tokenizer)

        # Calculate final statistics
        batch_statistics = {
            "total_images": len(image_files),
            "successful_extractions": successful_extractions,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(image_files)
            if image_files
            else 0,
            "success_rate": successful_extractions / len(image_files)
            if image_files
            else 0,
            "effective_batch_size": self.batch_size,
        }
        
        # Add group processing statistics if in grouped mode
        if self.extraction_mode == "grouped" and self.extraction_strategy:
            batch_statistics["total_groups_processed"] = self.extraction_strategy.stats.get("total_groups_processed", 0)
            batch_statistics["successful_groups"] = self.extraction_strategy.stats.get("successful_groups", 0)
            batch_statistics["failed_groups"] = self.extraction_strategy.stats.get("failed_groups", 0)

        print("\n📊 Batch Processing Complete:")
        print(f"   Total images: {batch_statistics['total_images']}")
        print(
            f"   Successful extractions: {batch_statistics['successful_extractions']}"
        )
        print(f"   Success rate: {batch_statistics['success_rate']:.1%}")
        print(
            f"   Average processing time: {batch_statistics['average_processing_time']:.2f}s"
        )
        print(f"   Effective batch size: {batch_statistics['effective_batch_size']}")

        return results, batch_statistics

    def _process_batch_with_fallback(self, batch_files: List[str]) -> List[dict]:
        """
        Process a batch of images with automatic fallback on OOM errors.

        Args:
            batch_files (List[str]): List of image file paths for this batch

        Returns:
            List[dict]: Results for this batch
        """
        if len(batch_files) == 1:
            # Single image processing (no batching needed)
            if self.extraction_mode == "single_pass":
                return [self.process_single_image(batch_files[0])]
            else:
                return [self.process_single_image_grouped(batch_files[0])]

        # Try true batch processing first
        if ENABLE_BATCH_SIZE_FALLBACK:
            return self._process_batch_with_retry(batch_files)
        else:
            return self._process_true_batch(batch_files)

    def _process_batch_with_retry(self, batch_files: List[str]) -> List[dict]:
        """
        Process batch with automatic retry on memory errors.

        Args:
            batch_files (List[str]): List of image file paths

        Returns:
            List[dict]: Processing results
        """
        current_batch_size = len(batch_files)

        # Try smaller batch sizes if needed
        for fallback_size in BATCH_SIZE_FALLBACK_STEPS:
            if fallback_size >= current_batch_size:
                continue

            try:
                return self._process_true_batch(batch_files)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"   ⚠️ OOM with batch size {current_batch_size}, trying smaller batches..."
                    )
                    # Split into smaller batches
                    results = []
                    for i in range(0, len(batch_files), fallback_size):
                        sub_batch = batch_files[i : i + fallback_size]
                        try:
                            sub_results = self._process_true_batch(sub_batch)
                            results.extend(sub_results)
                        except Exception as sub_e:
                            print(
                                f"   ❌ Sub-batch failed, falling back to individual processing: {sub_e}"
                            )
                            # Ultimate fallback: process individually
                            for file in sub_batch:
                                if self.extraction_mode == "single_pass":
                                    results.append(self.process_single_image(file))
                                else:
                                    results.append(
                                        self.process_single_image_grouped(file)
                                    )

                        # Clear cache between sub-batches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    return results
                else:
                    raise e

        # Ultimate fallback: process individually
        print(
            f"   🔄 Falling back to individual processing for {len(batch_files)} images"
        )
        if self.extraction_mode == "single_pass":
            return [self.process_single_image(file) for file in batch_files]
        else:
            return [self.process_single_image_grouped(file) for file in batch_files]

    def _process_true_batch(self, batch_files: List[str]) -> List[dict]:
        """
        Process multiple images in a true batch (parallel processing).

        Args:
            batch_files (List[str]): List of image file paths

        Returns:
            List[dict]: Processing results for each image
        """
        if len(batch_files) == 1:
            if self.extraction_mode == "single_pass":
                return [self.process_single_image(batch_files[0])]
            else:
                return [self.process_single_image_grouped(batch_files[0])]

        start_time = time.time()

        try:
            # Load all images in batch
            images = []
            pixel_values_list = []
            valid_files = []

            for file_path in batch_files:
                try:
                    # Load and preprocess image using InternVL3 pipeline
                    pixel_values = self.load_image(file_path)
                    pixel_values_list.append(pixel_values.to(torch.bfloat16).cuda())
                    valid_files.append(file_path)
                except Exception as e:
                    print(f"   ❌ Failed to load {Path(file_path).name}: {e}")
                    continue

            if not pixel_values_list:
                return [
                    self._create_error_result(file, "Image loading failed")
                    for file in batch_files
                ]

            # Process batch
            results = []
            prompt_template = f"<image>\n{self.get_extraction_prompt()}"

            for _idx, (pixel_values, file_path) in enumerate(
                zip(pixel_values_list, valid_files, strict=False)
            ):
                try:
                    # Use the same generation logic for both 2B and 8B models
                    try:
                        response = self.model.chat(
                            self.tokenizer,
                            pixel_values,
                            prompt_template,
                            self.generation_config,
                            history=None,
                            return_history=False,
                        )
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"⚠️ InternVL3 batch OOM on {Path(file_path).name}: {e}")
                        print("🔄 Attempting emergency cleanup and retry...")

                        # Emergency cleanup (skip normal post-processing cleanup)
                        torch.cuda.empty_cache()
                        clear_model_caches(self.model, self.tokenizer)
                        handle_memory_fragmentation(threshold_gb=0.5, aggressive=True)

                        # Retry once
                        try:
                            response = self.model.chat(
                                self.tokenizer,
                                pixel_values,
                                prompt_template,
                                self.generation_config,
                                history=None,
                                return_history=False,
                            )
                        except torch.cuda.OutOfMemoryError:
                            print(
                                f"❌ InternVL3: Even with cleanup, OOM persists for {Path(file_path).name}"
                            )
                            raise

                    # Parse response
                    extracted_data = parse_extraction_response(response)

                    # Calculate metrics - count ALL fields that are present (including correct N/A)
                    extracted_fields_count = len(
                        [k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]
                    )
                    response_completeness = extracted_fields_count / len(EXTRACTION_FIELDS)
                    content_coverage = extracted_fields_count / len(EXTRACTION_FIELDS)

                    result = {
                        "image_name": Path(file_path).name,
                        "extracted_data": extracted_data,
                        "raw_response": response,
                        "processing_time": (time.time() - start_time)
                        / len(valid_files),  # Approximate per-image time
                        "response_completeness": response_completeness,
                        "content_coverage": content_coverage,
                        "extracted_fields_count": extracted_fields_count,
                        "raw_response_length": len(response),
                    }

                    results.append(result)

                    print(
                        f"     ✅ {Path(file_path).name}: {extracted_fields_count}/{FIELD_COUNT} fields"
                    )

                except Exception as e:
                    print(f"     ❌ {Path(file_path).name}: {e}")
                    results.append(self._create_error_result(file_path, str(e)))

            # Add error results for any files that failed to load
            failed_files = set(batch_files) - set(valid_files)
            for failed_file in failed_files:
                results.append(
                    self._create_error_result(failed_file, "Image loading failed")
                )

            total_time = time.time() - start_time
            print(
                f"   ⏱️ Batch processing time: {total_time:.2f}s ({total_time / len(batch_files):.2f}s per image)"
            )

            return results

        except Exception as e:
            print(f"   ❌ Batch processing failed: {e}")
            # Fallback to individual processing
            if self.extraction_mode == "single_pass":
                return [self.process_single_image(file) for file in batch_files]
            else:
                return [self.process_single_image_grouped(file) for file in batch_files]

    def _create_error_result(self, file_path: str, error_message: str) -> dict:
        """Create standardized error result for failed processing."""
        return {
            "image_name": Path(file_path).name,
            "extracted_data": {field: "N/A" for field in EXTRACTION_FIELDS},
            "raw_response": f"Error: {error_message}",
            "processing_time": 0,
            "response_completeness": 0,
            "content_coverage": 0,
            "extracted_fields_count": 0,
            "raw_response_length": 0,
        }
