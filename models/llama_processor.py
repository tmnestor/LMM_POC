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
    ENABLE_BATCH_SIZE_FALLBACK,
    EXTRACTION_FIELDS,
    FIELD_COUNT,
    FIELD_INSTRUCTIONS,
    LLAMA_MODEL_PATH,
    get_auto_batch_size,
)
from common.evaluation_utils import parse_extraction_response

warnings.filterwarnings('ignore')


class LlamaProcessor:
    """Processor for Llama-3.2-11B-Vision-Instruct model."""
    
    def __init__(self, model_path=None, device='cuda', batch_size=None):
        """
        Initialize Llama processor with model and processor.
        
        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
            batch_size (int): Batch size for processing (auto-detected if None)
        """
        self.model_path = model_path or LLAMA_MODEL_PATH
        self.device = device
        self.model = None
        self.processor = None
        
        # Configure batch processing
        self._configure_batch_processing(batch_size)
        
        # Initialize model and processor
        self._load_model()
    
    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)  # Ensure minimum batch size of 1
            print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = self._get_available_gpu_memory()
            self.batch_size = get_auto_batch_size('llama', available_memory)
            print(f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)")
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available() or self.device == 'cpu':
            return 0.0
        
        try:
            # Get total and allocated memory
            device_idx = torch.cuda.current_device() if self.device == 'cuda' else int(self.device.split(':')[-1])
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_idx)
            available_memory = (total_memory - allocated_memory) / (1024 ** 3)  # Convert to GB
            
            return available_memory
        except Exception as e:
            print(f"⚠️ Could not detect GPU memory: {e}")
            return 16.0  # Assume 16GB as default for V100
    
    def _load_model(self):
        """Load Llama Vision model and processor with optimal configuration."""
        print(f"🔄 Loading Llama Vision model from: {self.model_path}")
        
        try:
            # Configure 8-bit quantization properly
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            
            # Load model with optimal configuration for 16GB VRAM
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,  # Memory-efficient 16-bit precision
                device_map="auto",           # Automatic device mapping
                quantization_config=quantization_config,  # Proper 8-bit quantization
            )
            
            # Load processor for multimodal inputs
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            print("✅ Llama Vision model loaded successfully")
            print(f"🔧 Device: {self.model.device}")
            print(f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
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
            instruction = FIELD_INSTRUCTIONS.get(field, '[value or N/A]')
            prompt += f"{field}: {instruction}\n"
        
        prompt += f"""
FORMAT RULES:
- Use exactly: KEY: value (colon and space)
- NEVER use: **KEY:** or **KEY** or *KEY* or any formatting
- Plain text only - NO markdown, NO bold, NO italic
- Include ALL {FIELD_COUNT} keys even if value is N/A
- Output ONLY these {FIELD_COUNT} lines, nothing else

STOP after {EXTRACTION_FIELDS[-1]} line. Do not add explanations or comments."""
        
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
            
            # Load image
            image = self.load_document_image(image_path)
            
            # Create multimodal conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.get_extraction_prompt()}
                    ]
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.processor(
                image,
                input_text,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max(800, FIELD_COUNT * 40),  # Scale tokens with field count
                    temperature=0.1,       # Near-deterministic
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "assistant\n\n" in response:
                response = response.split("assistant\n\n")[-1].strip()
            elif "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
            processing_time = time.time() - start_time
            
            # Parse response with Llama-specific cleaning
            extracted_data = parse_extraction_response(response, clean_conversation_artifacts=True)
            
            # Calculate metrics
            extracted_fields_count = sum(1 for v in extracted_data.values() if v != "N/A")
            response_completeness = len([k for k in extracted_data.keys() if k in EXTRACTION_FIELDS]) / len(EXTRACTION_FIELDS)
            content_coverage = extracted_fields_count / len(EXTRACTION_FIELDS)
            
            return {
                'image_name': Path(image_path).name,
                'extracted_data': extracted_data,
                'raw_response': response,
                'processing_time': processing_time,
                'response_completeness': response_completeness,
                'content_coverage': content_coverage,
                'extracted_fields_count': extracted_fields_count,
                'raw_response_length': len(response)
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return {
                'image_name': Path(image_path).name,
                'extracted_data': {field: "N/A" for field in EXTRACTION_FIELDS},
                'raw_response': f"Error: {str(e)}",
                'processing_time': 0,
                'response_completeness': 0,
                'content_coverage': 0,
                'extracted_fields_count': 0,
                'raw_response_length': 0
            }
    
    def process_image_batch(self, image_files: List[str], progress_callback=None) -> Tuple[list, dict]:
        """
        Process batch of images through Llama extraction pipeline with true batch processing.
        
        Args:
            image_files (List[str]): List of image file paths
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            Tuple[list, dict]: (results, statistics) - Extraction results and batch statistics
        """
        if not image_files:
            return [], {'total_images': 0, 'successful_extractions': 0, 'total_processing_time': 0, 'average_processing_time': 0, 'success_rate': 0}
        
        print(f"\n🚀 Processing {len(image_files)} images with Llama Vision (batch_size={self.batch_size})...")
        
        results = []
        total_processing_time = 0
        successful_extractions = 0
        
        # Process images in batches
        for batch_start in range(0, len(image_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            # Progress update for batch
            if progress_callback:
                progress_callback(batch_end, len(image_files), f"Batch {batch_start//self.batch_size + 1}")
            else:
                print(f"\n[Batch {batch_start//self.batch_size + 1}] Processing images {batch_start+1}-{batch_end} of {len(image_files)}")
            
            # Process current batch with fallback mechanism
            batch_results = self._process_batch_with_fallback(batch_files)
            results.extend(batch_results)
            
            # Update statistics
            for result in batch_results:
                total_processing_time += result['processing_time']
                if result['response_completeness'] > 0:
                    successful_extractions += 1
            
            # Clear GPU cache after each batch
            if CLEAR_GPU_CACHE_AFTER_BATCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final statistics
        batch_statistics = {
            'total_images': len(image_files),
            'successful_extractions': successful_extractions,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(image_files) if image_files else 0,
            'success_rate': successful_extractions / len(image_files) if image_files else 0,
            'effective_batch_size': self.batch_size
        }
        
        print("\n📊 Batch Processing Complete:")
        print(f"   Total images: {batch_statistics['total_images']}")
        print(f"   Successful extractions: {batch_statistics['successful_extractions']}")
        print(f"   Success rate: {batch_statistics['success_rate']:.1%}")
        print(f"   Average processing time: {batch_statistics['average_processing_time']:.2f}s")
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
                    print(f"   ⚠️ OOM with batch size {current_batch_size}, trying smaller batches...")
                    # Split into smaller batches
                    results = []
                    for i in range(0, len(batch_files), fallback_size):
                        sub_batch = batch_files[i:i+fallback_size]
                        try:
                            sub_results = self._process_true_batch(sub_batch)
                            results.extend(sub_results)
                        except Exception as sub_e:
                            print(f"   ❌ Sub-batch failed, falling back to individual processing: {sub_e}")
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
        print(f"   🔄 Falling back to individual processing for {len(batch_files)} images")
        return [self.process_single_image(file) for file in batch_files]
    
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
        print(f"   🔄 Processing {len(batch_files)} images with proven single-image method...")
        
        results = []
        start_time = time.time()
        
        for idx, file_path in enumerate(batch_files):
            try:
                # Use the proven single image processing method
                result = self.process_single_image(file_path)
                results.append(result)
                
                print(f"     ✅ {Path(file_path).name}: {result['extracted_fields_count']}/{FIELD_COUNT} fields ({result['processing_time']:.1f}s)")
                
            except Exception as e:
                print(f"     ❌ {Path(file_path).name}: {e}")
                results.append(self._create_error_result(file_path, str(e)))
            
            # Clear GPU cache periodically to prevent memory buildup
            if (idx + 1) % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        avg_time = sum(r['processing_time'] for r in results if r['processing_time'] > 0) / len(results) if results else 0
        
        print(f"   ⏱️ Batch processing time: {total_time:.2f}s (avg: {avg_time:.2f}s per image)")
        
        return results
    
    def _create_error_result(self, file_path: str, error_message: str) -> dict:
        """Create standardized error result for failed processing."""
        return {
            'image_name': Path(file_path).name,
            'extracted_data': {field: "N/A" for field in EXTRACTION_FIELDS},
            'raw_response': f"Error: {error_message}",
            'processing_time': 0,
            'response_completeness': 0,
            'content_coverage': 0,
            'extracted_fields_count': 0,
            'raw_response_length': 0
        }