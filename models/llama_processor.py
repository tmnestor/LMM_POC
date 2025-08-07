"""
Llama-specific processor for vision model evaluation.

This module contains all Llama-3.2-11B-Vision-Instruct-specific code including
model loading, image preprocessing, and batch processing logic.
"""

import time
import warnings
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from common.config import EXTRACTION_FIELDS, FIELD_COUNT, LLAMA_MODEL_PATH
from common.evaluation_utils import parse_extraction_response

warnings.filterwarnings('ignore')


class LlamaProcessor:
    """Processor for Llama-3.2-11B-Vision-Instruct model."""
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize Llama processor with model and processor.
        
        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
        """
        self.model_path = model_path or LLAMA_MODEL_PATH
        self.device = device
        self.model = None
        self.processor = None
        
        # Initialize model and processor
        self._load_model()
    
    def _load_model(self):
        """Load Llama Vision model and processor with optimal configuration."""
        print(f"🔄 Loading Llama Vision model from: {self.model_path}")
        
        try:
            # Load model with optimal configuration
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,  # Memory-efficient 16-bit precision
                device_map="auto",           # Automatic device mapping
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
        # Build field-specific instructions dynamically
        field_instructions = {
            'ABN': '[11-digit Australian Business Number or N/A]',
            'BANK_ACCOUNT_NUMBER': '[account number from bank statements only or N/A]',
            'BANK_NAME': '[bank name from bank statements only or N/A]',
            'BSB_NUMBER': '[6-digit BSB from bank statements only or N/A]',
            'CLOSING_BALANCE': '[closing balance amount in dollars or N/A]',
            'DESCRIPTIONS': '[list of transaction descriptions or N/A]',
            'GST': '[GST amount in dollars or N/A]',
            'OPENING_BALANCE': '[opening balance amount in dollars or N/A]',
            'PRICES': '[individual prices in dollars or N/A]',
            'QUANTITIES': '[list of quantities or N/A]',
            'SUBTOTAL': '[subtotal amount in dollars or N/A]',
            'TOTAL': '[total amount in dollars or N/A]'
        }
        
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
        
        # Add each field dynamically
        for field in EXTRACTION_FIELDS:
            instruction = field_instructions.get(field, '[value or N/A]')
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
    
    def process_image_batch(self, image_files, progress_callback=None):
        """
        Process batch of images through Llama extraction pipeline.
        
        Args:
            image_files (list): List of image file paths
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            tuple: (results, statistics) - Extraction results and batch statistics
        """
        results = []
        total_processing_time = 0
        successful_extractions = 0
        
        print(f"\n🚀 Processing {len(image_files)} images with Llama Vision...")
        
        for idx, image_path in enumerate(image_files, 1):
            # Progress update
            if progress_callback:
                progress_callback(idx, len(image_files), image_path)
            else:
                print(f"\n[{idx}/{len(image_files)}] Processing: {Path(image_path).name}")
            
            # Process image
            result = self.process_single_image(image_path)
            results.append(result)
            
            # Update statistics
            total_processing_time += result['processing_time']
            if result['response_completeness'] > 0:
                successful_extractions += 1
            
            # Show extraction status
            print(f"   ⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"   📊 Fields extracted: {result['extracted_fields_count']}/{FIELD_COUNT}")
            print(f"   ✅ Response completeness: {result['response_completeness']:.1%}")
        
        # Calculate batch statistics
        batch_statistics = {
            'total_images': len(image_files),
            'successful_extractions': successful_extractions,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(image_files) if image_files else 0,
            'success_rate': successful_extractions / len(image_files) if image_files else 0
        }
        
        print("\n📊 Batch Processing Complete:")
        print(f"   Total images: {batch_statistics['total_images']}")
        print(f"   Successful extractions: {batch_statistics['successful_extractions']}")
        print(f"   Success rate: {batch_statistics['success_rate']:.1%}")
        print(f"   Average processing time: {batch_statistics['average_processing_time']:.2f}s")
        
        return results, batch_statistics