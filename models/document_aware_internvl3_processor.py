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
from typing import List, Optional

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
from common.extraction_cleaner import ExtractionCleaner
from common.gpu_optimization import (
    clear_model_caches,
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
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
        skip_model_loading: bool = False
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

        # Fix 8B detection using actual model path
        self.is_8b_model = "8B" in str(self.model_path)

        if self.debug:
            print(f"🎯 Document-aware InternVL3 processor initialized for {self.field_count} fields")
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
            print(f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB, Model: {model_key})")

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
                            print(f"   Version check: major={major}, minor={minor}, use_old_api={use_old_api}")
                    except (ValueError, IndexError):
                        if self.debug:
                            print(f"   ⚠️ Could not parse version {bnb_version}, will try new API")
                        use_old_api = False

                if use_old_api:
                    print(f"   Using load_in_8bit for bitsandbytes {bnb_version} (old API)")
                    model_kwargs["load_in_8bit"] = True
                else:
                    try:
                        from transformers import BitsAndBytesConfig
                        print(f"   Using BitsAndBytesConfig for bitsandbytes {bnb_version} (new API)")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16,
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        print("   ✅ BitsAndBytesConfig created successfully")
                    except ImportError as e:
                        print(f"   ⚠️ BitsAndBytesConfig import failed: {e}")
                        print("   Falling back to load_in_8bit (expect deprecation warning)")
                        model_kwargs["load_in_8bit"] = True

                model_kwargs["device_map"] = "auto"

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

            # Skip warm-up for 8B model with quantization
            if self.is_8b_model:
                print("⚠️ Skipping warm-up test for quantized 8B model (expected with 8-bit)")
                print("   Model will be tested during actual inference")

        except Exception as e:
            print(f"❌ Error loading InternVL3 model: {e}")
            raise

    def generate_dynamic_prompt(self) -> str:
        """Generate prompt for specific field list - EXACT COPY of Llama's successful approach."""
        
        # Use Llama's exact simple prompt approach (no schema loading, no fallbacks)
        return self._generate_simple_prompt()

    # Removed unnecessary YAML loading methods - using Llama's simple approach only

    def _generate_simple_prompt(self) -> str:
        """Generate simple fallback prompt with dynamic fields - matches successful Llama structure."""
        
        prompt = f"""Extract structured data from this business document image.

CRITICAL INSTRUCTIONS:
-- Output ONLY the structured data below
-- Do NOT include any conversation text
-- Do NOT repeat the user's request
-- Do NOT include <image> tokens
-- Start immediately with {self.field_list[0]}
-- Stop immediately after {self.field_list[-1]}

REQUIRED OUTPUT FORMAT - EXACTLY {self.field_count} LINES:
"""
        
        # Add each field with type-aware instruction
        for field in self.field_list:
            instruction = self._get_field_type_instruction(field)
            prompt += f"{field}: {instruction}\n"
        
        prompt += f"""
OUTPUT RULES:
-- NEVER use: **KEY:** or **KEY** or *KEY* or any formatting  
-- Plain text only - NO markdown, NO bold, NO italic
-- Include ALL {self.field_count} keys even if value is NOT_FOUND
-- Output ONLY these {self.field_count} lines, nothing else
-- Use exact text from document (e.g., "TAX INVOICE" not "Invoice")
-- Use pipe separators for lists (e.g., "item1 | item2 | item3")
-- Be conservative: use NOT_FOUND if field is truly missing
-- For boolean fields: use "true" or "false" (not "yes"/"no")
-- For calculated fields: show computed values with proper formatting
-- For transaction lists: use pipe separators between transactions

STOP after {self.field_list[-1]} line. Do not add explanations or comments."""

        return prompt

    def _get_field_type_instruction(self, field: str) -> str:
        """Get field type-specific instruction for v4 schema fields - EXACT COPY from Llama."""
        from common.config import (
            get_boolean_fields,
            get_calculated_fields,
            get_transaction_list_fields,
        )
        
        try:
            # Check field type and provide appropriate instruction
            if field in get_boolean_fields():
                if "IS_GST_INCLUDED" in field:
                    return "[true if GST included in prices, false if GST separate, or NOT_FOUND]"
                else:
                    return "[true or false, or NOT_FOUND]"
            
            elif field in get_calculated_fields():
                if "LINE_ITEM_TOTAL_PRICES" in field:
                    return "[pipe-separated calculated totals: qty×price for each item, or NOT_FOUND]"
                elif "DISCOUNT" in field:
                    return "[discount amount calculated from line items, or NOT_FOUND]"
                else:
                    return "[calculated value or NOT_FOUND]"
            
            elif field in get_transaction_list_fields():
                if "TRANSACTION_DATES" in field:
                    return "[pipe-separated transaction dates in chronological order, or NOT_FOUND]"
                elif "TRANSACTION_AMOUNTS" in field:
                    return "[pipe-separated transaction amounts with signs (+/-), or NOT_FOUND]"
                elif "TRANSACTION_DESCRIPTIONS" in field:
                    return "[pipe-separated transaction descriptions/references, or NOT_FOUND]"
                elif "RUNNING_BALANCES" in field:
                    return "[pipe-separated account balances after each transaction, or NOT_FOUND]"
                else:
                    return "[pipe-separated transaction data or NOT_FOUND]"
            
            else:
                # Default instruction for standard field types
                return "[value or NOT_FOUND]"
                
        except Exception:
            # Fallback if field type checking fails
            return "[value or NOT_FOUND]"

    def build_transform(self, input_size=DEFAULT_IMAGE_SIZE):
        """Build InternVL3 image transformation pipeline."""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=DEFAULT_IMAGE_SIZE, use_thumbnail=False):
        """InternVL3 dynamic preprocessing algorithm."""
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
        """Complete InternVL3 image loading and preprocessing pipeline."""
        # For 8B model, use fewer tiles to reduce memory
        if max_num is None:
            max_num = 6 if self.is_8b_model else 12

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

    def process_single_image(self, image_path: str) -> dict:
        """Process single image with document-aware extraction."""

        try:
            start_time = time.time()

            # Memory cleanup
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            # Generate prompt for specific field list
            prompt = self.generate_dynamic_prompt()

            if self.debug:
                print(f"📝 Generated prompt for {self.field_count} fields")
                print(f"   Fields: {self.field_list[:3]}{'...' if len(self.field_list) > 3 else ''}")
                print(f"🔍 DOCUMENT-AWARE PROMPT ({len(prompt)} chars):")
                print("=" * 80)
                print(prompt)
                print("=" * 80)

            # Extract with document-specific prompt
            response = self._extract_with_prompt(image_path, prompt)

            processing_time = time.time() - start_time

            if self.debug:
                print(f"📄 RAW MODEL RESPONSE ({len(response)} chars):")
                print("=" * 80)
                print(response)
                print("=" * 80)

            # Parse response using document-specific field list
            extracted_data = self._parse_document_aware_response(response)

            if self.debug:
                print("📊 PARSED EXTRACTION RESULTS:")
                print("=" * 80)
                for field in self.field_list:
                    value = extracted_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    print(f"  {status} {field}: \"{value}\"")
                print("=" * 80)

                found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
                print(f"✅ Extracted {len(found_fields)}/{self.field_count} fields")
                if found_fields:
                    print(f"   Found: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}")

            # Calculate metrics
            extracted_fields_count = len([k for k in extracted_data.keys() if k in self.field_list])
            response_completeness = extracted_fields_count / len(self.field_list)
            content_coverage = extracted_fields_count / len(self.field_list)

            # Cleanup with V100 optimizations
            comprehensive_memory_cleanup(self.model, self.tokenizer)

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": processing_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": self.field_count
            }

        except Exception as e:
            if self.debug:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()

            # Return error result with dynamic fields
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in self.field_list},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "field_count": self.field_count
            }

    def _extract_with_prompt(self, image_path: str, prompt: str, generation_config: dict = None) -> str:
        """Core extraction method that handles image processing and model inference."""
        # Pre-processing cleanup with fragmentation detection
        handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

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

            # Emergency cleanup
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

    def _extract_with_custom_prompt(self, image_path: str, prompt: str, **generation_kwargs) -> str:
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
                custom_generation_config["max_new_tokens"] = generation_kwargs["max_new_tokens"]
                
            # Only set sampling parameters if do_sample is True to avoid warnings
            if custom_generation_config.get("do_sample", False):
                if "temperature" in generation_kwargs and generation_kwargs["temperature"] is not None:
                    custom_generation_config["temperature"] = generation_kwargs["temperature"]
            else:
                # Remove all sampling-related parameters when do_sample is False
                custom_generation_config.pop("temperature", None)
                custom_generation_config.pop("top_k", None)
                custom_generation_config.pop("top_p", None)

            # Use shared extraction method
            return self._extract_with_prompt(image_path, prompt, custom_generation_config)

        except Exception as e:
            if self.debug:
                print(f"❌ Error in InternVL3 custom prompt extraction: {e}")
            return f"Error: {str(e)}"

    def _parse_document_aware_response(self, response_text: str) -> dict:
        """Parse extraction response for document-specific field list."""
        import re

        if not response_text:
            return {field: "NOT_FOUND" for field in self.field_list}

        # Initialize with NOT_FOUND for all document-specific fields
        extracted_data = {field: "NOT_FOUND" for field in self.field_list}

        # Clean response text
        response_text = response_text.strip()

        # Process each line looking for key-value pairs
        lines = response_text.split("\n")

        for line in lines:
            # Skip empty lines and non-key-value lines
            if not line.strip() or ":" not in line:
                continue

            # Clean the line from various formatting issues
            clean_line = line.strip()
            
            # Remove markdown formatting
            clean_line = re.sub(r"\*\*([^*]+)\*\*:", r"\1:", clean_line)
            clean_line = re.sub(r"\*([^*]+)\*:", r"\1:", clean_line)
            
            # Fix GST field name variations
            clean_line = re.sub(r"^GST[_\s]*\d*%?:", "GST_AMOUNT:", clean_line)

            # Extract key and value
            parts = clean_line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()
                value = parts[1].strip()

                # Store if it's in our document-specific field list
                if key in self.field_list:
                    # Clean the extracted value using the centralized cleaner
                    cleaned_value = self.cleaner.clean_field_value(key, value) if value else "NOT_FOUND"
                    extracted_data[key] = cleaned_value

        return extracted_data
    
