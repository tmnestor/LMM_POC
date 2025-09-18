#!/usr/bin/env python3
"""
Document-Aware InternVL3 Hybrid Processor - SAFE COPY WITH INTERNVL3 MODEL

A hybrid processor that combines:
- InternVL3 model (for better accuracy potential)
- Llama's proven processing pipeline (for reliable processing)
- ExtractionCleaner integration (for 🧹 CLEANER CALLED output)

ZERO RISK to existing Llama processor - this is a completely independent copy.

DOCUMENT AWARE REDUCTION OPTIMIZED:
- Invoice/Receipt: 11 fields (62-67% reduction) → ~50% faster processing
- Bank Statement: 5 fields (75% reduction) → ~75% faster processing
- Dynamic max_new_tokens automatically scales with reduced field counts
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
    INTERNVL3_MAX_TILES_2B,
    INTERNVL3_MAX_TILES_8B,
    INTERNVL3_MODEL_PATH,
    get_auto_batch_size,
    get_max_new_tokens,
)
from common.extraction_cleaner import ExtractionCleaner
from common.gpu_optimization import (
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    detect_memory_fragmentation,
    get_available_gpu_memory,
    handle_memory_fragmentation,
    optimize_model_for_v100,
)
from common.simple_prompt_loader import SimplePromptLoader, load_llama_prompt

warnings.filterwarnings("ignore")


class DocumentAwareInternVL3HybridProcessor:
    """Hybrid processor: InternVL3 model + Llama's proven processing pipeline."""

    def __init__(
        self,
        field_list: List[str],
        model_path: str = None,
        device: str = "cuda",
        debug: bool = False,
        batch_size: Optional[int] = None,
        skip_model_loading: bool = False,
        pre_loaded_model=None,
        pre_loaded_tokenizer=None,
    ):
        """
        Initialize hybrid processor with InternVL3 model and Llama processing logic.

        Args:
            field_list (List[str]): Fields to extract
            model_path (str): Path to InternVL3 model
            device (str): Device to run on
            debug (bool): Enable debug output
            batch_size (int): Batch size (auto-detected if None)
            skip_model_loading (bool): Skip loading model (for reusing existing model)
            pre_loaded_model: Pre-loaded InternVL3 model (avoids reloading)
            pre_loaded_tokenizer: Pre-loaded InternVL3 tokenizer (avoids reloading)
        """
        self.field_list = field_list
        self.field_count = len(field_list)
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.debug = debug

        # Initialize components (InternVL3 specific)
        self.model = pre_loaded_model
        self.tokenizer = pre_loaded_tokenizer
        self.generation_config = None

        # Detect model variant (2B vs 8B) for tile optimization
        self.is_8b_model = "8B" in self.model_path

        # Initialize extraction cleaner for value normalization (🧹 CLEANER CALLED output)
        self.cleaner = ExtractionCleaner(debug=debug)

        if self.debug:
            print(
                f"🎯 InternVL3 Hybrid processor initialized for {self.field_count} fields: {field_list[0]} → {field_list[-1]}"
            )

        # Configure CUDA memory allocation
        configure_cuda_memory_allocation()

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Configure generation parameters for dynamic field count
        self._configure_generation()

        # Load model and tokenizer (unless skipping for reuse or already provided)
        if not skip_model_loading and (self.model is None or self.tokenizer is None):
            self._load_model()
        elif self.model is not None and self.tokenizer is not None:
            if self.debug:
                print("✅ Using pre-loaded InternVL3 model and tokenizer")
                print(f"🔧 Device: {self.model.device}")
                print(f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            # Apply V100 optimizations to pre-loaded model
            optimize_model_for_v100(self.model)

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            if self.debug:
                print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = get_available_gpu_memory(self.device)
            self.batch_size = get_auto_batch_size("internvl3", available_memory)
            if self.debug:
                print(
                    f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)"
                )

    def _configure_generation(self):
        """Configure generation parameters for InternVL3."""
        # InternVL3 generation config (adapted from Llama logic)
        self.generation_config = {
            "max_new_tokens": get_max_new_tokens("internvl3", self.field_count),
            "temperature": 0.0,
            "do_sample": False,
            "top_p": 0.9,
            "use_cache": True,
        }

        if self.debug:
            performance_gain = max(
                0, (29 - self.field_count) / 29 * 100
            )  # vs old invoice max
            print(
                f"🎯 DOCUMENT AWARE REDUCTION: {self.field_count} fields (~{performance_gain:.0f}% fewer than original 29)"
            )
            print(
                f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
                f"temperature={self.generation_config['temperature']}, "
                f"do_sample={self.generation_config['do_sample']}"
            )

    def _load_model(self):
        """Load InternVL3 model and tokenizer with optimal configuration."""
        if self.debug:
            print(f"🔄 Loading InternVL3 model from: {self.model_path}")

        try:
            # Load InternVL3 model with V100 optimizations
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )

            if self.debug:
                print("✅ InternVL3 model loaded successfully")
                print(f"🔧 Device: {self.model.device}")
                print(
                    f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )

            # Apply V100 optimizations
            optimize_model_for_v100(self.model)

        except Exception as e:
            if self.debug:
                print(f"❌ Error loading InternVL3 model: {e}")
            raise

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

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio

        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=24, image_size=DEFAULT_IMAGE_SIZE):
        """InternVL3 dynamic preprocessing with tiling."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Target ratios for optimal tiling
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Resize image
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
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

        assert len(processed_images) == target_aspect_ratio[0] * target_aspect_ratio[1]
        assert len(processed_images) <= max_num
        return processed_images

    def load_image(self, image_file, input_size=DEFAULT_IMAGE_SIZE, max_num=None):
        """Complete InternVL3 image loading and preprocessing pipeline."""
        if max_num is None:
            if self.is_8b_model:
                max_num = INTERNVL3_MAX_TILES_8B  # Configurable: default 20 tiles
            else:
                max_num = INTERNVL3_MAX_TILES_2B  # Configurable: default 24 tiles

        if self.debug:
            print(f"🔍 LOAD_IMAGE: max_num={max_num}, input_size={input_size}")

        # Load image
        image = Image.open(image_file).convert("RGB")

        # Process into tiles
        images = self.dynamic_preprocess(
            image, min_num=1, max_num=max_num, image_size=input_size
        )

        # Apply transforms
        transform = self.build_transform(input_size=input_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        if self.debug:
            print(
                f"📐 TENSOR_SHAPE: {pixel_values.shape} (batch_size={pixel_values.shape[0]} tiles)"
            )

        return pixel_values

    def get_extraction_prompt(self, document_type: str = None) -> str:
        """
        Get extraction prompt for document type - uses Llama prompts for consistency.

        Args:
            document_type: Document type ('invoice', 'receipt', 'bank_statement')
                          If None, uses 'universal' prompt

        Returns:
            Complete prompt string loaded from YAML
        """
        if document_type is None:
            document_type = "universal"

        if self.debug:
            print(f"📝 Loading {document_type} prompt for InternVL3 Hybrid")

        try:
            return load_llama_prompt(document_type)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to load {document_type} prompt, falling back to universal")
            return load_llama_prompt("universal")

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return SimplePromptLoader.get_available_prompts("llama_prompts.yaml")

    def detect_document_type(self, field_list: List[str] = None) -> str:
        """
        Detect document type based on field composition.
        Simple heuristic for backward compatibility.
        """
        fields_to_check = field_list or self.field_list
        field_set = set(fields_to_check)

        # Simple heuristic based on unique fields
        if "STATEMENT_DATE_RANGE" in field_set or "TRANSACTION_DATES" in field_set:
            return "bank_statement"
        elif "INVOICE_DATE" in field_set and "GST_AMOUNT" in field_set:
            # Both invoice and receipt have these, default to invoice
            return "invoice"
        else:
            # Fallback to invoice for unknown patterns
            return "invoice"

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading image {image_path}: {e}")
            raise

    def _resilient_generate(self, pixel_values, question, **generation_kwargs):
        """Resilient generation with OOM fallback using InternVL3 chat method."""

        # Build clean generation parameters
        clean_generation_kwargs = {
            "max_new_tokens": generation_kwargs.get(
                "max_new_tokens", self.generation_config["max_new_tokens"]
            ),
            "temperature": generation_kwargs.get(
                "temperature", self.generation_config["temperature"]
            ),
            "do_sample": generation_kwargs.get(
                "do_sample", self.generation_config["do_sample"]
            ),
            "top_p": generation_kwargs.get("top_p", self.generation_config["top_p"]),
            "use_cache": generation_kwargs.get(
                "use_cache", self.generation_config["use_cache"]
            ),
        }

        try:
            # Use InternVL3 chat method instead of generate
            return self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config=clean_generation_kwargs,
                history=None,
                return_history=False
            )
        except torch.cuda.OutOfMemoryError:
            if self.debug:
                print("⚠️ CUDA OOM during generation, attempting recovery...")

            # Clear cache and retry
            torch.cuda.empty_cache()
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            try:
                return self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config=clean_generation_kwargs,
                    history=None,
                    return_history=False
                )
            except torch.cuda.OutOfMemoryError:
                if self.debug:
                    print("❌ Still OOM after cleanup, falling back to CPU")

                # CPU fallback
                pixel_values_cpu = pixel_values.cpu()
                self.model = self.model.cpu()

                response = self.model.chat(
                    self.tokenizer,
                    pixel_values_cpu,
                    question,
                    generation_config=clean_generation_kwargs,
                    history=None,
                    return_history=False
                )

                # Move back to GPU
                self.model = self.model.to(self.device)
                return response

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: Optional[str] = None,
        custom_max_tokens: Optional[int] = None,
    ) -> dict:
        """Process single image with document-aware extraction using InternVL3."""

        try:
            start_time = time.time()

            # Memory cleanup
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            # Use custom prompt if provided, otherwise generate from schema
            if custom_prompt:
                prompt = custom_prompt
                document_type = "CUSTOM"  # Indicate custom prompt usage
            else:
                # Get document-aware prompt
                document_type = self.detect_document_type()
                prompt = self.get_extraction_prompt(document_type=document_type)

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                if custom_prompt:
                    sys.stdout.write("📝 Using custom YAML prompt\n")
                    sys.stdout.write(f"🔍 CUSTOM YAML PROMPT ({len(prompt)} chars):\n")
                else:
                    sys.stdout.write(
                        f"📝 Generated prompt for {self.field_count} fields\n"
                    )
                    sys.stdout.write(
                        f"   Fields: {self.field_list[:3]}{'...' if len(self.field_list) > 3 else ''}\n"
                    )
                    sys.stdout.write(
                        f"🔍 DOCUMENT-AWARE PROMPT ({len(prompt)} chars):\n"
                    )
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(prompt + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Load and preprocess image using InternVL3 pipeline
            pixel_values = self.load_image(image_path)

            # Move to appropriate device and convert dtype
            if self.device == "cpu":
                pixel_values = pixel_values.to(torch.float32)
            else:
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

            # Prepare question for InternVL3
            question = f"<image>\n{prompt}"

            # Use custom max_tokens if provided (for YAML prompts)
            generation_config = self.generation_config.copy()
            if custom_max_tokens:
                generation_config["max_new_tokens"] = custom_max_tokens

            if self.debug:
                print(f"🖼️  Input tensor shape: {pixel_values.shape}")
                print(
                    f"💭 Generating with max_new_tokens={generation_config['max_new_tokens']}"
                )

            # Generate response using InternVL3 chat method
            response = self._resilient_generate(pixel_values, question, **generation_config)

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write(f"📄 RAW MODEL RESPONSE ({len(response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse response using robust extraction parser with field filtering
            from common.extraction_parser import parse_extraction_response

            extracted_data = parse_extraction_response(
                response, expected_fields=self.field_list
            )

            # Apply ExtractionCleaner for value normalization (🧹 CLEANER CALLED output)
            cleaned_data = {}
            for field in self.field_list:
                raw_value = extracted_data.get(field, "NOT_FOUND")
                if raw_value != "NOT_FOUND":
                    cleaned_value = self.cleaner.clean_field_value(field, raw_value)
                    cleaned_data[field] = cleaned_value
                else:
                    cleaned_data[field] = "NOT_FOUND"

            # Replace extracted_data with cleaned version
            extracted_data = cleaned_data

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write("📊 PARSED EXTRACTION RESULTS:\n")
                sys.stdout.write("=" * 80 + "\n")
                for field in self.field_list:
                    value = extracted_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    sys.stdout.write(f'  {status} {field}: "{value}"\n')
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

                found_fields = [
                    k for k, v in extracted_data.items() if v != "NOT_FOUND"
                ]
                print(f"✅ Extracted {len(found_fields)}/{self.field_count} fields")
                if found_fields:
                    print(
                        f"   Found: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}"
                    )

            # Calculate metrics
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in self.field_list]
            )
            response_completeness = extracted_fields_count / len(self.field_list)
            content_coverage = extracted_fields_count / len(self.field_list)

            # Cleanup with V100 optimizations
            del pixel_values
            comprehensive_memory_cleanup(self.model, None, verbose=self.debug)

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": time.time() - start_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": self.field_count,
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
                "field_count": self.field_count,
            }

    def get_model_info(self) -> dict:
        """Get information about the loaded model for debugging."""
        if self.model is None:
            return {"status": "not_loaded", "model_path": self.model_path}

        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": str(self.model.device),
            "is_8b_model": self.is_8b_model,
            "field_count": self.field_count,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
        }