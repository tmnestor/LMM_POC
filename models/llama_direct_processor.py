"""
Llama-specific processor for vision model evaluation using direct prompting.

This module contains all Llama-3.2-11B-Vision (base model) specific code for
direct prompting without chat templates, similar to InternVL3's approach.
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
    CLEAR_GPU_CACHE_AFTER_BATCH,
    EXTRACTION_FIELDS,
    FIELD_COUNT,
    LLAMA_DIRECT_MODEL_PATH,
    get_auto_batch_size,
)

warnings.filterwarnings("ignore")


class LlamaDirectProcessor:
    """Processor for Llama-3.2-11B-Vision base model using direct prompting."""

    def __init__(self, model_path=None, device="cuda", batch_size=None):
        """
        Initialize Llama direct processor with model and processor.

        Args:
            model_path (str): Path to model weights (uses default if None)
            device (str): Device to run model on
            batch_size (int): Batch size for processing (auto-detected if None)
        """
        self.model_path = model_path or LLAMA_DIRECT_MODEL_PATH
        self.device = device
        self.model = None
        self.processor = None

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Initialize model and processor
        self._load_model()

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is None:
            self.batch_size = get_auto_batch_size("llama")
            print(
                f"🔧 Auto-detected batch size: {self.batch_size} (model: llama direct)"
            )
        else:
            self.batch_size = batch_size
            print(f"🔧 Using specified batch size: {self.batch_size}")

        # Store original batch size for fallback
        self.original_batch_size = self.batch_size

    def _load_model(self):
        """Load Llama Vision base model and processor with optimal configuration."""
        print(f"🔄 Loading Llama Vision direct model from: {self.model_path}")

        # Configure 8-bit quantization for V100 compatibility
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
        )

        try:
            # Load model with quantization
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            print("✅ Llama Vision direct model loaded successfully")
            if hasattr(self.model, "parameters"):
                print(
                    f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )
            print("🔧 Using direct prompting (no chat template)")

        except Exception as e:
            print(f"❌ Error loading Llama Vision direct model: {e}")
            raise

    def get_extraction_prompt(self):
        """Get the extraction prompt optimized for direct Llama Vision."""
        # Use completion-style prompt with exact field names
        prompt = f"""Extract the following data from this business document:

{EXTRACTION_FIELDS[0]}: """
        
        return prompt

    def _parse_direct_response(self, response):
        """Parse the direct model response format into structured fields."""
        import re
        
        # Initialize with N/A for all fields
        extracted_data = {field: "N/A" for field in EXTRACTION_FIELDS}
        
        # Look for exact field names first (if model uses them)
        for field in EXTRACTION_FIELDS:
            # Try exact field name match
            exact_match = re.search(rf'{field}:\s*([^\n]+)', response, re.IGNORECASE)
            if exact_match:
                value = exact_match.group(1).strip()
                if value and value != 'N/A' and len(value) > 0:
                    extracted_data[field] = value

        # Extract ABN patterns (various formats) if not found above
        if extracted_data['ABN'] == 'N/A':
            abn_patterns = [
                r'(\d{11})',  # 11 digits together
                r'(\d{2,3}\s+\d{3}\s+\d{3})',  # With spaces
                r'(\d{2,3}\s+\d{3}\s+\d{3}\s+\d{3})',  # With more spaces
            ]
            for pattern in abn_patterns:
                abn_match = re.search(pattern, response)
                if abn_match:
                    extracted_data['ABN'] = abn_match.group(1).replace(' ', '')
                    break
        
        # Extract account holder patterns
        account_patterns = [
            r'Account Holder:\s*([^[\]\n]+)',
            r'ACCOUNT_HOLDER:\s*([^[\]\n]+)',
            r'Name:\s*([A-Za-z\s]+)(?=\s*Address|\s*Phone|\s*$)',
            r'Account Name:\s*\[ACCOUNT_HOLDER:\s*([^\]]+)\]',
        ]
        for pattern in account_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_data['ACCOUNT_HOLDER'] = match.group(1).strip()
                break
        
        # Extract account numbers
        account_num_patterns = [
            r'Account Number:\s*(\d+)',
            r'ACCOUNT_NUMBER:\s*(\d+)',
        ]
        for pattern in account_num_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_data['BANK_ACCOUNT_NUMBER'] = match.group(1).strip()
                break
        
        # Extract invoice information
        invoice_patterns = {
            'INVOICE_NUMBER': [r'Invoice\s*#?\s*([A-Z0-9-]+)', r'Receipt:\s*([A-Z0-9-]+)'],
            'INVOICE_DATE': [r'Invoice Date\s+(\d{2}/\d{2}/\d{4})', r'Date:\s*(\d{2}/\d{2}/\d{4})'],
            'DUE_DATE': [r'Due Date\s+(\d{2}/\d{2}/\d{4})'],
        }
        
        for field, patterns in invoice_patterns.items():
            if field in extracted_data:
                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        extracted_data[field] = match.group(1).strip()
                        break
        
        # Extract addresses from various formats
        address_patterns = [
            r'Bill To:\s*\[BILL_TO:\s*([^\]]+)\]',
            r'BILLING_ADDRESS:\s*([^[\]\n]+)',
            r'Address:\s*([^[\]\n]+?)(?=\s*Phone|\s*Email|\s*$)',
        ]
        for pattern in address_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                extracted_data['BILLING_ADDRESS'] = address
                # Try to extract company/person name from context
                if 'Account Holder:' in response or 'Name:' in response:
                    # Address is for a person
                    pass
                else:
                    # Might be company address
                    first_part = address.split(',')[0].strip()
                    if not re.match(r'^\d', first_part) and len(first_part) > 3:
                        extracted_data['COMPANY_NAME'] = first_part
                break
        
        # Extract totals and financial amounts
        total_patterns = [
            r'Balance:\s*\$?([\d,]+\.?\d*)',
            r'TOTAL\s*\$?([\d,]+\.?\d*)',
            r'Total.*?\$?([\d,]+\.?\d*)',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_data['TOTAL'] = match.group(1).replace(',', '')
                break
        
        return extracted_data

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
        Process a single image using direct prompting approach.

        Args:
            image_path (str): Path to document image

        Returns:
            dict: Extraction results with metadata
        """
        try:
            start_time = time.time()

            # Load image
            image = self.load_document_image(image_path)

            # Use correct base model format with begin_of_text token
            direct_prompt = f"<|begin_of_text|><|image|>{self.get_extraction_prompt()}"

            # Process inputs directly - image first, then text
            inputs = self.processor(image, direct_prompt, return_tensors="pt").to(
                self.model.device
            )

            # Generate response with stable parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max(800, FIELD_COUNT * 40),  # Standard tokens
                    temperature=0.1,  # Near-deterministic
                    do_sample=True,  # Standard sampling
                    top_p=0.95,
                    use_cache=True,  # Enable KV caching
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode response and extract only the generated part
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the newly generated text (after the prompt)
            input_length = len(self.processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
            full_response = self.processor.decode(output[0], skip_special_tokens=True)
            response = full_response[input_length:].strip()
            
            # Additional cleanup: remove any remaining prompt artifacts
            if "CRITICAL INSTRUCTIONS:" in response:
                # Find where actual data starts (after instructions)
                lines = response.split('\n')
                data_start = -1
                for i, line in enumerate(lines):
                    if ':' in line and not line.strip().startswith('-') and 'INSTRUCTIONS' not in line:
                        data_start = i
                        break
                if data_start >= 0:
                    response = '\n'.join(lines[data_start:]).strip()

            processing_time = time.time() - start_time

            # Custom parsing for direct model output format
            extracted_data = self._parse_direct_response(response)

            # TEMPORARY DEBUG: Print full debugging info
            print(f"  🔍 DEBUG - Full response length: {len(full_response)}")
            print(f"  🔍 DEBUG - Input length: {input_length}")
            print(f"  🔍 DEBUG - Generated response: '{response[:300]}{'...' if len(response) > 300 else ''}'")
            print(f"  🔍 DEBUG - Response empty?: {len(response) == 0}")
            print(f"  📊 Extracted {len([k for k, v in extracted_data.items() if v != 'N/A'])} fields:")
            for key, value in list(extracted_data.items())[:5]:  # Show first 5 fields
                print(f"    {key}: {value}")
            if len(extracted_data) > 5:
                print(f"    ... and {len(extracted_data) - 5} more fields")
            print()

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
                "raw_response_length": len(response),
                "extracted_fields_count": extracted_fields_count,
            }

        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return self._create_error_result(image_path, str(e))

    def _create_error_result(self, image_path, error_message):
        """Create error result structure."""
        return {
            "image_name": Path(image_path).name,
            "extracted_data": {field: "ERROR" for field in EXTRACTION_FIELDS},
            "raw_response": f"ERROR: {error_message}",
            "processing_time": 0.0,
            "response_completeness": 0.0,
            "content_coverage": 0.0,
            "raw_response_length": 0,
            "extracted_fields_count": 0,
            "error": True,
        }

    def process_image_batch(self, image_files: List[str]) -> Tuple[List[dict], dict]:
        """
        Process a batch of images using direct prompting with fallback support.

        Args:
            image_files (list): List of image file paths

        Returns:
            tuple: (results_list, batch_statistics)
        """
        print(
            f"\n🚀 Processing {len(image_files)} images with Llama Vision Direct (batch_size={self.batch_size})..."
        )

        results = []
        total_processing_time = 0
        successful_extractions = 0

        # Reset batch size to original for new batch
        self.batch_size = self.original_batch_size

        try:
            # Process images individually for now (batch processing can be added later)
            for i, image_path in enumerate(image_files, 1):
                print(
                    f"📄 Processing image {i}/{len(image_files)}: {Path(image_path).name}"
                )

                try:
                    result = self.process_single_image(image_path)
                    results.append(result)

                    total_processing_time += result["processing_time"]
                    if not result.get("error", False):
                        successful_extractions += 1

                    # Clear GPU cache periodically
                    if CLEAR_GPU_CACHE_AFTER_BATCH and i % 5 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"⚠️ Error processing image {i}: {e}")
                    error_result = self._create_error_result(image_path, str(e))
                    results.append(error_result)

        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            # Fill remaining results with errors
            for remaining_path in image_files[len(results) :]:
                error_result = self._create_error_result(
                    remaining_path, "Batch processing failed"
                )
                results.append(error_result)

        # Calculate batch statistics
        avg_processing_time = (
            total_processing_time / len(image_files) if image_files else 0
        )
        success_rate = successful_extractions / len(image_files) if image_files else 0

        batch_stats = {
            "total_images": len(image_files),
            "successful_extractions": successful_extractions,
            "failed_extractions": len(image_files) - successful_extractions,
            "success_rate": success_rate,
            "total_processing_time": total_processing_time,
            "average_processing_time": avg_processing_time,
            "final_batch_size": self.batch_size,
        }

        print("✅ Batch processing completed:")
        print(f"   📊 Success rate: {success_rate:.1%}")
        print(f"   ⏱️ Average time per image: {avg_processing_time:.2f}s")
        print(f"   🔧 Final batch size: {self.batch_size}")

        return results, batch_stats

    def __del__(self):
        """Cleanup method to clear GPU memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
