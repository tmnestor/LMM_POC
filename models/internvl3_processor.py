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
    DEFAULT_IMAGE_SIZE,
    ENABLE_BATCH_SIZE_FALLBACK,
    EXTRACTION_FIELDS,
    FIELD_COUNT,
    FIELD_INSTRUCTIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INTERNVL3_GENERATION_CONFIG,
    INTERNVL3_MODEL_PATH,
    get_auto_batch_size,
    get_max_new_tokens,
)
from common.evaluation_utils import parse_extraction_response


class InternVL3Processor:
    """Processor for InternVL3 vision-language model."""

    def __init__(self, model_path=None, device="cuda", batch_size=None):
        """
        Initialize InternVL3 processor with model and tokenizer.

        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
            batch_size (int): Batch size for processing (auto-detected if None)
        """
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.model = None
        self.tokenizer = None
        self.generation_config = None

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Initialize model and tokenizer
        self._load_model()

        # Setup generation config from centralized configuration
        # Remove configuration constants that aren't model parameters
        self.generation_config = {
            "max_new_tokens": get_max_new_tokens("internvl3", FIELD_COUNT),
            "do_sample": INTERNVL3_GENERATION_CONFIG["do_sample"],
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        print(
            f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
            f"do_sample={self.generation_config['do_sample']}"
        )

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)  # Ensure minimum batch size of 1
            print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = self._get_available_gpu_memory()
            self.batch_size = get_auto_batch_size("internvl3", available_memory)
            print(
                f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)"
            )

    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available() or self.device == "cpu":
            return 0.0

        try:
            # Get total and allocated memory
            device_idx = (
                torch.cuda.current_device()
                if self.device == "cuda"
                else int(self.device.split(":")[-1])
            )
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_idx)
            available_memory = (total_memory - allocated_memory) / (
                1024**3
            )  # Convert to GB

            return available_memory
        except Exception as e:
            print(f"⚠️ Could not detect GPU memory: {e}")
            return 16.0  # Assume 16GB as default for V100

    def _load_model(self):
        """Load InternVL3 model and tokenizer with compatibility settings."""
        print(f"🔧 Loading InternVL3 model from: {self.model_path}")

        try:
            # Load model with compatibility settings
            self.model = (
                AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,  # Official recommendation
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,  # Disabled for compatibility
                    trust_remote_code=True,
                )
                .eval()
                .to(self.device)
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )

            print("✅ InternVL3 model and tokenizer loaded successfully")

        except Exception as e:
            print(f"❌ Error loading InternVL3 model: {e}")
            raise

    def get_extraction_prompt(self):
        """Get the extraction prompt for InternVL3."""
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

    def load_image(self, image_file, input_size=DEFAULT_IMAGE_SIZE, max_num=12):
        """
        Complete InternVL3 image loading and preprocessing pipeline.

        Args:
            image_file: Path to image file or PIL Image
            input_size: Size for each tile
            max_num: Maximum number of tiles

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
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

            # Load and preprocess image
            pixel_values = self.load_image(image_path)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            # Prepare conversation
            question = f"<image>\n{self.get_extraction_prompt()}"

            # Generate response
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                self.generation_config,
                history=None,
                return_history=False,
            )

            processing_time = time.time() - start_time

            # Parse response
            extracted_data = parse_extraction_response(response)

            # Calculate metrics
            extracted_fields_count = sum(
                1 for v in extracted_data.values() if v != "N/A"
            )
            response_completeness = len(
                [k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]
            ) / len(EXTRACTION_FIELDS)
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

            # Clear GPU cache after each batch
            if CLEAR_GPU_CACHE_AFTER_BATCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            return [self.process_single_image(batch_files[0])]

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
                                results.append(self.process_single_image(file))

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
        return [self.process_single_image(file) for file in batch_files]

    def _process_true_batch(self, batch_files: List[str]) -> List[dict]:
        """
        Process multiple images in a true batch (parallel processing).

        Args:
            batch_files (List[str]): List of image file paths

        Returns:
            List[dict]: Processing results for each image
        """
        if len(batch_files) == 1:
            return [self.process_single_image(batch_files[0])]

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
                    # Generate response for this image
                    response = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        prompt_template,
                        self.generation_config,
                        history=None,
                        return_history=False,
                    )

                    # Parse response
                    extracted_data = parse_extraction_response(response)

                    # Calculate metrics
                    extracted_fields_count = sum(
                        1 for v in extracted_data.values() if v != "N/A"
                    )
                    response_completeness = len(
                        [k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]
                    ) / len(EXTRACTION_FIELDS)
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
            return [self.process_single_image(file) for file in batch_files]

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
