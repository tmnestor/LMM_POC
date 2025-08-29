#!/usr/bin/env python3
"""
Document-Aware Llama Processor - Standalone Implementation

A complete rewrite of Llama processor designed from the ground up for 
document-aware extraction with dynamic field lists. NO INHERITANCE.
"""

import time
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from common.config import (
    LLAMA_GENERATION_CONFIG,
    LLAMA_MODEL_PATH,
    get_auto_batch_size,
    get_max_new_tokens,
)
from common.extraction_cleaner import ExtractionCleaner
from common.gpu_optimization import (
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    get_available_gpu_memory,
    handle_memory_fragmentation,
    optimize_model_for_v100,
)

warnings.filterwarnings("ignore")


class DocumentAwareLlamaProcessor:
    """Standalone document-aware Llama processor with dynamic field support."""
    
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
        Initialize document-aware processor with specific field list.
        
        Args:
            field_list (List[str]): Fields to extract (e.g., 20 invoice fields)
            model_path (str): Path to Llama model
            device (str): Device to run on  
            debug (bool): Enable debug output
            batch_size (int): Batch size (auto-detected if None)
            skip_model_loading (bool): Skip loading model (for reusing existing model)
        """
        self.field_list = field_list
        self.field_count = len(field_list)
        self.model_path = model_path or LLAMA_MODEL_PATH
        self.device = device
        self.debug = debug
        
        # Initialize components
        self.model = None
        self.processor = None
        self.generation_config = None
        
        # Initialize extraction cleaner for value normalization
        self.cleaner = ExtractionCleaner(debug=debug)
        
        if self.debug:
            print(f"🎯 Document-aware processor initialized for {self.field_count} fields: {field_list[0]} → {field_list[-1]}")
        
        # Configure CUDA memory allocation
        configure_cuda_memory_allocation()
        
        # Configure batch processing
        self._configure_batch_processing(batch_size)
        
        # Configure generation parameters for dynamic field count
        self._configure_generation()
        
        # Load model and processor (unless skipping for reuse)
        if not skip_model_loading:
            self._load_model()
    
    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = get_available_gpu_memory(self.device)
            self.batch_size = get_auto_batch_size("llama", available_memory)
            print(f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)")
    
    def _configure_generation(self):
        """Configure generation parameters for dynamic field count."""
        # Initialize generation config
        self.generation_config = LLAMA_GENERATION_CONFIG.copy()
        
        # Calculate dynamic max_new_tokens based on actual field count
        self.generation_config["max_new_tokens"] = get_max_new_tokens("llama", self.field_count)
        
        print(
            f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
            f"temperature={self.generation_config['temperature']}, "
            f"do_sample={self.generation_config['do_sample']}"
        )
    
    def _load_model(self):
        """Load Llama Vision model and processor with optimal configuration."""
        print(f"🔄 Loading Llama Vision model from: {self.model_path}")
        
        try:
            # Configure 8-bit quantization for V100 compatibility
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                llm_int8_threshold=6.0,
            )
            
            # Load model
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )
            
            # Load processor for multimodal inputs
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Call tie_weights() after loading
            try:
                self.model.tie_weights()
                print("✅ Llama Vision model loaded successfully (tie_weights called)")
            except Exception as e:
                print(f"⚠️ Llama Vision model loaded (tie_weights warning ignored): {e}")
            
            print(f"🔧 Device: {self.model.device}")
            print(f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Apply V100 optimizations
            optimize_model_for_v100(self.model)
            
        except Exception as e:
            print(f"❌ Error loading Llama model: {e}")
            raise
    
    def generate_dynamic_prompt(self) -> str:
        """Generate prompt for specific field list with v4 field type support."""
        
        # Try to load YAML configuration
        yaml_config = self._load_yaml_config()
        
        if yaml_config:
            return self._generate_yaml_prompt(yaml_config)
        else:
            return self._generate_simple_prompt()
    
    def _load_yaml_config(self) -> dict:
        """Load YAML configuration if available."""
        try:
            yaml_path = Path(__file__).parent.parent / "common" / "llama_single_pass_prompts.yaml"
            if yaml_path.exists():
                with yaml_path.open("r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                    return yaml_data.get("single_pass", {})
        except Exception as e:
            if self.debug:
                print(f"⚠️ Could not load YAML config: {e}")
        return {}
    
    def _generate_yaml_prompt(self, yaml_config: dict) -> str:
        """Generate prompt using YAML configuration with dynamic fields and v4 field type support."""
        
        persona = yaml_config.get("persona", "You are an expert document analyzer.")
        task_description = yaml_config.get("task_description", "Extract key information from business documents.")
        output_format = yaml_config.get("output_format", "Provide information in the following format:")
        
        prompt = f"{persona}\n\n{task_description}\n\n{output_format}\n"
        
        # Add field instructions using dynamic field list with v4 field type awareness
        field_instructions = yaml_config.get("field_instructions", {})
        for field in self.field_list:
            # Get field type-specific instruction or use default
            instruction = field_instructions.get(field)
            if not instruction:
                instruction = self._get_field_type_instruction(field)
            prompt += f"{field}: {instruction}\n"
        
        # Add stop instruction
        stop_instruction = yaml_config.get("stop_instruction", "Stop after providing all required information.")
        prompt += f"\n{stop_instruction}"
        
        return prompt
    
    def _generate_simple_prompt(self) -> str:
        """Generate simple fallback prompt with dynamic fields."""
        
        prompt = f"""Extract structured data from this business document image.

CRITICAL INSTRUCTIONS:
-- Output ONLY the structured data below
-- Do NOT include any conversation text
-- Do NOT repeat the user's request
-- Do NOT include <image> tokens
- Start immediately with {self.field_list[0]}
- Stop immediately after {self.field_list[-1]}

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
    
    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            raise
    
    def _resilient_generate(self, inputs, **generation_kwargs):
        """Resilient generation with OOM fallback."""
        
        # Build clean generation parameters following llama_processor_v2 pattern
        clean_generation_kwargs = {
            "max_new_tokens": generation_kwargs.get("max_new_tokens", self.generation_config["max_new_tokens"]),
            "temperature": generation_kwargs.get("temperature", self.generation_config["temperature"]),
            "do_sample": generation_kwargs.get("do_sample", self.generation_config["do_sample"]),
            "top_p": generation_kwargs.get("top_p", self.generation_config["top_p"]),
            "use_cache": generation_kwargs.get("use_cache", self.generation_config["use_cache"]),
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        try:
            # Standard generation with clean parameters
            return self.model.generate(**inputs, **clean_generation_kwargs)
        except torch.cuda.OutOfMemoryError:
            if self.debug:
                print("⚠️ CUDA OOM during generation, attempting recovery...")
            
            # Clear cache and retry
            torch.cuda.empty_cache()
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            
            try:
                return self.model.generate(**inputs, **clean_generation_kwargs)
            except torch.cuda.OutOfMemoryError:
                if self.debug:
                    print("❌ Still OOM after cleanup, falling back to CPU")
                
                # CPU fallback
                inputs_cpu = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in inputs.items()}
                self.model = self.model.cpu()
                
                output = self.model.generate(**inputs_cpu, **clean_generation_kwargs)
                
                # Move back to GPU
                self.model = self.model.to(self.device)
                return output
    
    def process_single_image(self, image_path: str) -> dict:
        """Process single image with document-aware extraction."""
        
        try:
            start_time = time.time()
            
            # Memory cleanup
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            
            # Load image
            image = self.load_document_image(image_path)
            
            # Generate prompt for specific field list
            prompt = self.generate_dynamic_prompt()
            
            if self.debug:
                print(f"📝 Generated prompt for {self.field_count} fields")
                print(f"   Fields: {self.field_list[:3]}{'...' if len(self.field_list) > 3 else ''}")
                print(f"🔍 DOCUMENT-AWARE PROMPT ({len(prompt)} chars):")
                print("=" * 80)
                print(prompt)
                print("=" * 80)
            
            # Create multimodal conversation
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
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.device)
            
            if self.debug:
                print(f"🖼️  Input tensor shape: {inputs['input_ids'].shape}")
                print(f"💭 Generating with max_new_tokens={self.generation_config['max_new_tokens']}")
            
            # Generate response
            output = self._resilient_generate(inputs, **self.generation_config)
            
            # Decode response
            full_response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract assistant's response - handle multiple chat template formats
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                # Llama 3.2 official format
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[1]
                response = response.split("<|eot_id|>")[0].strip()
            elif "\nassistant\n\n" in full_response:
                # Common format: user\n[prompt]\nassistant\n\n[response]
                response = full_response.split("\nassistant\n\n")[1].strip()
            elif "\nassistant\n" in full_response:
                # Variant format: user\n[prompt]\nassistant\n[response]
                response = full_response.split("\nassistant\n")[1].strip()
            elif "assistant" in full_response:
                # Fallback: split on assistant keyword and take the last part
                parts = full_response.split("assistant")
                response = parts[-1].strip()
            else:
                # No chat template detected, use full response
                response = full_response.strip()
            
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
            del inputs, output, image
            comprehensive_memory_cleanup(self.model, self.processor)
            
            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": time.time() - start_time,
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
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.device)
            
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
            if "assistant\\n\\n" in response:
                response = response.split("assistant\\n\\n")[-1].strip()
            elif "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
            # Cleanup with V100 optimizations
            del inputs, output, image
            comprehensive_memory_cleanup(self.model, self.processor)
            
            return response
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error in custom prompt extraction: {e}")
            return f"Error: {str(e)}"
    
    def _parse_document_aware_response(self, response_text: str) -> dict:
        """Parse extraction response for document-specific field list."""
        import re
        
        if not response_text:
            return {field: "NOT_FOUND" for field in self.field_list}
        
        # Initialize with NOT_FOUND for all document-specific fields
        extracted_data = {field: "NOT_FOUND" for field in self.field_list}
        
        # Clean Llama-specific conversation artifacts and extract only assistant response
        if "assistant" in response_text:
            # Split on 'assistant' and take the last part (the actual response)
            parts = response_text.split("assistant")
            response_text = parts[-1].strip()
        
        # Additional cleaning patterns
        clean_patterns = [
            r"I'll extract.*?\n",
            r"I can extract.*?\n", 
            r"Here (?:is|are) the.*?\n",
            r"Based on.*?\n",
            r"Looking at.*?\n",
            r"<\|start_header_id\|>.*?<\|end_header_id\|>",
            r"<image>",
            r"^\s*Extract.*?below\.\s*\n",
        ]
        
        for pattern in clean_patterns:
            response_text = re.sub(
                pattern, "", response_text, flags=re.IGNORECASE | re.MULTILINE
            )
        
        # Process each line looking for key-value pairs
        lines = response_text.strip().split("\n")
        
        for line in lines:
            # Skip empty lines and non-key-value lines
            if not line.strip() or ":" not in line:
                continue
            
            # Clean the line from various formatting issues
            clean_line = line
            # Remove markdown formatting - handle **KEY**: value pattern specifically
            clean_line = re.sub(r"\*\*([^*]+)\*\*:", r"\1:", clean_line)
            # Fix various prefix issues
            clean_line = re.sub(r"^KEY:\s*([A-Z_]+):", r"\1:", clean_line)
            clean_line = re.sub(r"^KEY\s+([A-Z_]+):", r"\1:", clean_line)
            # Fix GST field name variations (after markdown removal)
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
    
    def _get_field_type_instruction(self, field: str) -> str:
        """Get field type-specific instruction for v4 schema fields."""
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
                # Field-specific instructions for common fields
                if field == "SUPPLIER_NAME":
                    return "[business name from document header or NOT_FOUND]"
                elif field == "BUSINESS_ABN":
                    return "[11-digit Australian Business Number (e.g. XX XXX XXX XXX or XXXXXXXXXXX) or NOT_FOUND]"
                elif field == "BUSINESS_PHONE":
                    return "[business phone number (not customer phone) (e.g. (XX) XXXX XXXX) or NOT_FOUND]"
                elif field == "DOCUMENT_TYPE":
                    return "[document type (INVOICE/RECEIPT/STATEMENT) or NOT_FOUND]"
                elif field == "LINE_ITEM_PRICES":
                    return "[numbers in the right most column with 2 decimal places or NOT_FOUND]"
                else:
                    # Default instruction for other field types
                    return "[value or NOT_FOUND]"
                
        except Exception:
            # Fallback if field type checking fails
            return "[value or NOT_FOUND]"