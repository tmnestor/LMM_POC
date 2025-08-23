"""
Llama-specific processor for vision model evaluation.

This module contains all Llama-3.2-11B-Vision-Instruct-specific code including
model loading, image preprocessing, and batch processing logic.
"""

import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from common.config import (
    BATCH_SIZE_FALLBACK_STEPS,
    CLEAR_GPU_CACHE_AFTER_BATCH,
    DEFAULT_EXTRACTION_MODE,
    ENABLE_BATCH_SIZE_FALLBACK,
    EXTRACTION_FIELDS,
    FIELD_COUNT,
    FIELD_INSTRUCTIONS,
    LLAMA_GENERATION_CONFIG,
    LLAMA_MODEL_PATH,
    get_auto_batch_size,
    get_max_new_tokens,
)
from common.extraction_parser import parse_extraction_response
from common.gpu_optimization import (
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    get_available_gpu_memory,
    handle_memory_fragmentation,
    optimize_model_for_v100,
)
from common.grouped_extraction import get_extraction_strategy

warnings.filterwarnings("ignore")


class LlamaProcessor:
    """Processor for Llama-3.2-11B-Vision-Instruct model."""

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
        Initialize Llama processor with model and processor.

        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
            batch_size (int): Batch size for processing (auto-detected if None)
            extraction_mode (str): Extraction mode ('single_pass', 'grouped', 'adaptive')
            debug (bool): Enable debug logging for extraction
            grouping_strategy (str): Grouping strategy ('8_groups' or '6_groups')
        """
        self.model_path = model_path or LLAMA_MODEL_PATH
        self.device = device
        self.model = None
        self.processor = None

        # Configure extraction strategy
        self.extraction_mode = extraction_mode or DEFAULT_EXTRACTION_MODE
        self.debug = debug
        self.extraction_strategy = get_extraction_strategy(
            self.extraction_mode, debug, grouping_strategy, model_name="llama"
        )

        # Configure CUDA memory allocation strategy (from PyTorch forums)
        configure_cuda_memory_allocation()

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Configure generation parameters from config.py
        self._configure_generation()

        # Initialize model and processor
        self._load_model()

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)  # Ensure minimum batch size of 1
            print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = get_available_gpu_memory(self.device)
            self.batch_size = get_auto_batch_size("llama", available_memory)
            print(
                f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)"
            )

    def _configure_generation(self):
        """Configure generation parameters from config.py."""
        # Initialize generation config using centralized configuration
        self.generation_config = LLAMA_GENERATION_CONFIG.copy()

        # Calculate dynamic max_new_tokens based on field count
        self.generation_config["max_new_tokens"] = get_max_new_tokens(
            "llama", FIELD_COUNT
        )

        print(
            f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
            f"temperature={self.generation_config['temperature']}, "
            f"do_sample={self.generation_config['do_sample']}"
        )

    def _load_model(self):
        """Load Llama Vision model and processor with optimal configuration."""
        print(f"🔄 Loading Llama Vision model from: {self.model_path}")

        try:
            # Configure simple 8-bit quantization for V100 compatibility
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Standard setting
                llm_int8_skip_modules=[
                    "vision_tower",
                    "multi_modal_projector",
                ],  # Skip vision modules that cause tensor issues
                llm_int8_threshold=6.0,
            )

            # Load model with simple, stable configuration
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,  # Memory-efficient 16-bit precision
                device_map="auto",  # Automatic device mapping
                quantization_config=quantization_config,  # Simple 8-bit quantization
            )

            # Load processor for multimodal inputs
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Call tie_weights() method after model loading (warning will persist but model works)
            try:
                self.model.tie_weights()
                print("✅ Llama Vision model loaded successfully (tie_weights called)")
            except Exception as e:
                print(f"⚠️ Llama Vision model loaded (tie_weights warning ignored): {e}")
                print("ℹ️ Model will function correctly despite tie_weights warning")
            print(f"🔧 Device: {self.model.device}")
            print(
                f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

            # Apply V100 optimizations
            optimize_model_for_v100(self.model)

        except Exception as e:
            print(f"❌ Error loading Llama model: {e}")
            raise

    def get_extraction_prompt(self):
        """Get the extraction prompt optimized for Llama Vision."""
        # Use centralized field instructions from config.py

        prompt = f"""Extract key-value data from this business document image.

CRITICAL INSTRUCTIONS:
- Output ONLY the structured data below
- Do NOT include any conversation text
- Do NOT repeat the user's request
- Do NOT include <image> tokens
- Start immediately with {EXTRACTION_FIELDS[0]}
- Stop immediately after {EXTRACTION_FIELDS[-1]}

REQUIRED OUTPUT FORMAT - EXACTLY {FIELD_COUNT} LINES:
"""

        # Add each field dynamically using centralized instructions
        for field in EXTRACTION_FIELDS:
            instruction = FIELD_INSTRUCTIONS.get(field, "[value or N/A]")
            prompt += f"{field}: {instruction}\n"

        prompt += f"""
FORMAT RULES:
- Use exactly: KEY: value (colon and space)
- NEVER use: **KEY:** or **KEY** or *KEY* or any formatting
- Plain text only - NO markdown, NO bold, NO italic
- Include ALL {FIELD_COUNT} keys even if value is N/A
- Output ONLY these {FIELD_COUNT} lines, nothing else

STOP after {EXTRACTION_FIELDS[-1]} line. Do not add explanations or comments."""

        if self.debug:
            print(
                f"📝 SINGLE-PASS PROMPT: {len(prompt)} chars, {len(EXTRACTION_FIELDS)} fields"
            )
            print("📝 PROMPT CONTENT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)

        return prompt

    def load_document_image(self, image_path):
        """
        Load document image with error handling.

        Args:
            image_path (str): Path to document image

        Returns:
            PIL.Image: Loaded document image
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            raise

    def _resilient_generate(self, inputs, **generation_kwargs):
        """
        Resilient generation with automatic OffloadedCache fallback on CUDA OOM.

        Args:
            inputs: Model inputs
            **generation_kwargs: Generation parameters

        Returns:
            Generated output tensor
        """
        oom_occurred = False

        try:
            # First attempt: Standard generation with use_cache=True
            return self.model.generate(**inputs, **generation_kwargs)

        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ CUDA OOM detected: {e}")
            print("🔄 Retrying with cache_implementation='offloaded'...")
            oom_occurred = True

        if oom_occurred:
            # Emergency cleanup before retry
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            try:
                # Retry with OffloadedCache fallback
                generation_kwargs["cache_implementation"] = "offloaded"
                return self.model.generate(**inputs, **generation_kwargs)

            except torch.cuda.OutOfMemoryError as e2:
                print(f"⚠️ OffloadedCache also failed: {e2}")
                print("🚨 EMERGENCY: Reloading model to force complete memory reset...")

                # Emergency Strategy 4: Complete model reload
                return self._emergency_model_reload_generate(
                    inputs, **generation_kwargs
                )

    def _emergency_model_reload_generate(self, inputs, **generation_kwargs):
        """
        Emergency model reload to force complete memory reset when all other strategies fail.

        Args:
            inputs: Model inputs
            **generation_kwargs: Generation parameters

        Returns:
            Generated output tensor
        """
        print("🔄 EMERGENCY: Forcing complete model reload...")

        # Step 1: Complete model cleanup
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor

        # Step 2: Aggressive memory cleanup
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(
                f"🧹 Post-deletion cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f}GB VRAM"
            )

        # Step 3: Reload model with minimal configuration
        print("🔄 Reloading model with emergency configuration...")
        self._load_model()

        # Step 4: Process with OffloadedCache from the start
        generation_kwargs["cache_implementation"] = "offloaded"
        print("🎯 Processing with emergency OffloadedCache configuration...")

        try:
            with torch.no_grad():
                return self.model.generate(**inputs, **generation_kwargs)

        except torch.cuda.OutOfMemoryError as e3:
            print(f"🚨 CRITICAL: Model reload with OffloadedCache failed: {e3}")
            print(
                "🔄 STRATEGY 4.5: Attempting fresh GPU model reload WITHOUT OffloadedCache..."
            )

            # Strategy 4.5: Try fresh GPU reload WITHOUT offloaded cache
            # The offloaded cache itself might be causing issues
            try:
                # Complete cleanup before fresh GPU reload
                if self.model is not None:
                    del self.model
                if self.processor is not None:
                    del self.processor

                import gc

                for _ in range(3):
                    gc.collect()

                if torch.cuda.is_available():
                    # Maximum cleanup effort
                    for _ in range(5):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

                    memory_after_cleanup = torch.cuda.memory_allocated() / (1024**3)
                    print(
                        f"🧹 Ultra-cleanup complete: {memory_after_cleanup:.2f}GB VRAM"
                    )

                # Reload fresh GPU model
                print("🔄 Loading fresh GPU model (no OffloadedCache)...")
                self._load_model()

                # Try generation with fresh GPU model (no offloaded cache)
                generation_kwargs_fresh = generation_kwargs.copy()
                if "cache_implementation" in generation_kwargs_fresh:
                    del generation_kwargs_fresh["cache_implementation"]

                print("🎯 Attempting generation with fresh GPU model...")
                with torch.no_grad():
                    return self.model.generate(**inputs, **generation_kwargs_fresh)

            except torch.cuda.OutOfMemoryError as e4:
                print(f"❌ Fresh GPU reload also failed: {e4}")
                print("☢️ FINAL FALLBACK: Forcing CPU-only processing...")

                # Ultimate Strategy 5: Force CPU processing
                return self._ultimate_cpu_fallback_generate(inputs, **generation_kwargs)

    def _ultimate_cpu_fallback_generate(self, inputs, **generation_kwargs):
        """
        Ultimate CPU fallback: Load fresh model on CPU when GPU strategies fail.

        Args:
            inputs: Model inputs
            **generation_kwargs: Generation parameters

        Returns:
            Generated output tensor
        """
        print("☢️ ULTIMATE FALLBACK: Loading fresh CPU-only model...")

        # Step 1: Complete cleanup of GPU model
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor

        # Step 2: Clear all GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(
                f"🧹 GPU cleared: {torch.cuda.memory_allocated() / (1024**3):.2f}GB VRAM"
            )

        # Step 3: Load fresh model directly on CPU (no device_map, no quantization)
        print("🔄 Loading fresh CPU-only model (no quantization)...")
        try:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use FP32 for optimal CPU performance
                device_map="cpu",  # Force CPU only
                # NO quantization_config - causes meta device issues
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Call tie_weights() method after model loading (CPU version)
            try:
                self.model.tie_weights()
                print(
                    "✅ Fresh CPU model loaded successfully (FP32, tie_weights called)"
                )
            except Exception as e:
                print(
                    f"⚠️ Fresh CPU model loaded (FP32, tie_weights warning ignored): {e}"
                )
                print("ℹ️ CPU model will function correctly despite tie_weights warning")

        except Exception as e:
            print(f"❌ CPU model loading failed: {e}")
            raise RuntimeError(f"All fallback strategies failed: {e}") from e

        # Step 4: Move inputs to CPU
        cpu_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                cpu_inputs[key] = value.to("cpu")
            else:
                cpu_inputs[key] = value

        # Step 5: Process on CPU (slower but guaranteed)
        print("🐌 Processing on CPU (slower but stable)...")

        # Remove cache_implementation since we're on CPU
        cpu_generation_kwargs = generation_kwargs.copy()
        if "cache_implementation" in cpu_generation_kwargs:
            del cpu_generation_kwargs["cache_implementation"]

        with torch.no_grad():
            output = self.model.generate(**cpu_inputs, **cpu_generation_kwargs)

        print("✅ CPU processing completed successfully")
        return output

    def process_single_image(self, image_path):
        """
        Process a single image through Llama extraction pipeline.

        Args:
            image_path (str): Path to image file

        Returns:
            dict: Extraction results with metadata
        """
        try:
            start_time = time.time()

            # STRATEGY 3: Pre-processing cleanup with fragmentation detection
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            # Load image
            image = self.load_document_image(image_path)

            # Create multimodal conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.get_extraction_prompt()},
                    ],
                }
            ]

            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.model.device
            )

            # STRATEGY 3: Resilient generation with OffloadedCache fallback
            generation_kwargs = {
                "max_new_tokens": self.generation_config["max_new_tokens"],
                "temperature": self.generation_config["temperature"],
                "do_sample": self.generation_config["do_sample"],
                "top_p": self.generation_config["top_p"],
                "use_cache": self.generation_config["use_cache"],
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }

            # Generate response with resilient fallback
            with torch.no_grad():
                output = self._resilient_generate(inputs, **generation_kwargs)

            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "assistant\n\n" in response:
                response = response.split("assistant\n\n")[-1].strip()
            elif "assistant" in response:
                response = response.split("assistant")[-1].strip()

            processing_time = time.time() - start_time

            # Parse response with Llama-specific cleaning
            extracted_data = parse_extraction_response(
                response, clean_conversation_artifacts=True
            )

            if self.debug:
                print("🔍 RAW MODEL RESPONSE (single-pass):")
                print("-" * 40)
                print(response)
                print("-" * 40)
                print("🔍 PARSED DATA (single-pass):")
                for field, value in list(extracted_data.items())[
                    :5
                ]:  # Show first 5 fields
                    print(f"  {field}: {value}")
                print(f"  ... and {len(extracted_data) - 5} more fields")
                print()

            # Calculate metrics - count ALL fields that are present (including correct N/A)
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]
            )
            response_completeness = extracted_fields_count / len(EXTRACTION_FIELDS)
            content_coverage = extracted_fields_count / len(EXTRACTION_FIELDS)

            # STRATEGY 3: Comprehensive memory cleanup and cache clearing + OffloadedCache fallback
            del inputs, output, image

            # Use comprehensive cleanup from common module
            comprehensive_memory_cleanup(self.model, self.processor)

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

            # STRATEGY 3: Emergency cleanup on error
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
            **generation_kwargs: Additional generation parameters

        Returns:
            str: Raw model response
        """
        try:
            # Load image
            image = self.load_document_image(image_path)

            # Create multimodal conversation with custom prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(image, input_text, return_tensors="pt").to(
                self.model.device
            )

            # Merge generation kwargs with defaults
            final_generation_kwargs = {
                "do_sample": self.generation_config["do_sample"],
                "top_p": self.generation_config["top_p"],
                "use_cache": self.generation_config["use_cache"],
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            final_generation_kwargs.update(generation_kwargs)

            # Clean up temperature if do_sample is False to avoid warnings
            if not final_generation_kwargs.get("do_sample", False):
                final_generation_kwargs.pop("temperature", None)
                final_generation_kwargs.pop("top_p", None)

            # Generate response with resilient fallback
            with torch.no_grad():
                output = self._resilient_generate(inputs, **final_generation_kwargs)

            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "assistant\n\n" in response:
                response = response.split("assistant\n\n")[-1].strip()
            elif "assistant" in response:
                response = response.split("assistant")[-1].strip()

            # Cleanup
            del inputs, output, image
            comprehensive_memory_cleanup(self.model, self.processor)

            return response

        except Exception as e:
            print(f"❌ Error in custom prompt extraction: {e}")
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
                    f"🔍 Processing {Path(image_path).name} with {self.extraction_mode} mode"
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
                    f"✅ Grouped extraction completed in {total_processing_time:.2f}s"
                )
                print(
                    f"📊 Extracted {extracted_fields_count}/{len(EXTRACTION_FIELDS)} fields"
                )

            return result

        except Exception as e:
            print(f"❌ Error in grouped extraction for {image_path}: {e}")

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
        Process batch of images through Llama extraction pipeline with true batch processing.

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
            f"\n🚀 Processing {len(image_files)} images with Llama Vision (batch_size={self.batch_size})..."
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

        # Add group processing statistics if in grouped mode
        if self.extraction_mode == "grouped" and self.extraction_strategy:
            batch_statistics["total_groups_processed"] = (
                self.extraction_strategy.stats.get("total_groups_processed", 0)
            )
            batch_statistics["successful_groups"] = self.extraction_strategy.stats.get(
                "successful_groups", 0
            )
            batch_statistics["failed_groups"] = self.extraction_strategy.stats.get(
                "failed_groups", 0
            )

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
        Process multiple images using the proven single image method.

        Note: Llama processor uses sequential processing for reliability.
        Each image is processed using the existing process_single_image method
        which is known to work correctly.

        Args:
            batch_files (List[str]): List of image file paths

        Returns:
            List[dict]: Processing results for each image
        """
        print(f"   🔄 Processing {len(batch_files)} images with single-image method...")

        results = []
        start_time = time.time()

        for idx, file_path in enumerate(batch_files):
            try:
                # Use the appropriate image processing method based on extraction mode
                if self.extraction_mode == "single_pass":
                    result = self.process_single_image(file_path)
                else:
                    result = self.process_single_image_grouped(file_path)
                results.append(result)

                print(
                    f"     ✅ {Path(file_path).name}: {result['extracted_fields_count']}/{FIELD_COUNT} fields ({result['processing_time']:.1f}s)"
                )

            except Exception as e:
                print(f"     ❌ {Path(file_path).name}: {e}")
                results.append(self._create_error_result(file_path, str(e)))

            # Clear GPU cache periodically to prevent memory buildup
            if (idx + 1) % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        avg_time = (
            sum(r["processing_time"] for r in results if r["processing_time"] > 0)
            / len(results)
            if results
            else 0
        )

        print(
            f"   ⏱️ Batch processing time: {total_time:.2f}s (avg: {avg_time:.2f}s per image)"
        )

        return results

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
