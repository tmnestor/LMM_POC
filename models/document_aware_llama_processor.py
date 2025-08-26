#!/usr/bin/env python3
"""
Document-Aware Llama Processor - Phase 4 Implementation

A proper document-aware processor that inherits from LlamaProcessor but uses
document-specific field sets instead of the hardcoded unified 25-field approach.
"""

import time
from pathlib import Path

import yaml

from models.llama_processor import LlamaProcessor


class DocumentAwareLlamaProcessor(LlamaProcessor):
    """Document-aware Llama processor supporting dynamic field sets."""
    
    def __init__(
        self,
        model_path=None,
        device="cuda",
        batch_size=None,
        extraction_mode=None,
        debug=False,
        grouping_strategy="detailed_grouped",
        field_list=None
    ):
        """
        Initialize document-aware Llama processor.
        
        Args:
            field_list (List[str]): List of fields to extract (overrides default EXTRACTION_FIELDS)
            All other args: Same as parent LlamaProcessor
        """
        # Initialize parent processor
        super().__init__(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            extraction_mode=extraction_mode,
            debug=debug,
            grouping_strategy=grouping_strategy
        )
        
        # Set document-specific fields
        self.extraction_fields = field_list or []
        self.field_count = len(self.extraction_fields)
        
        if self.debug and field_list:
            print(f"🎯 Document-aware processor using {len(field_list)} fields: {field_list[0]} → {field_list[-1]}")
    
    def generate_single_pass_prompt(self, yaml_config):
        """Override to use document-specific fields."""
        # Build structured prompt with document-specific fields
        persona = yaml_config.get("persona", "You are an expert document analyzer.")
        task_description = yaml_config.get("task_description", "Extract key information from business documents.")
        output_format = yaml_config.get("output_format", "Provide information in the following format:")
        
        prompt = f"{persona}\n\n{task_description}\n\n{output_format}\n"
        
        # Add field instructions using document-specific fields
        field_instructions = yaml_config.get("field_instructions", {})
        for field in self.extraction_fields:
            instruction = field_instructions.get(
                field, f"[{field.lower()} or NOT_FOUND]"
            )
            prompt += f"{field}: {instruction}\n"
        
        # Add structured output instructions
        stop_instruction = yaml_config.get(
            "stop_instruction", 
            "Stop after providing all required information."
        )
        prompt += f"\n{stop_instruction}"
        
        if self.debug:
            print(f"📝 SINGLE-PASS PROMPT: {len(prompt)} chars, {len(self.extraction_fields)} fields")
            print("📝 PROMPT CONTENT:")
            print("-" * 40)
            print(prompt)
            print("-" * 40)
        
        return prompt
    
    def get_simple_single_pass_prompt(self):
        """Override to use document-specific fields."""
        # Simple fallback prompt with document-specific fields
        prompt = f"""Extract structured data from this business document image.

CRITICAL INSTRUCTIONS:
-- Output ONLY the structured data below
-- Do NOT include any conversation text
-- Do NOT repeat the user's request
-- Do NOT include <image> tokens
- Start immediately with {self.extraction_fields[0]}
- Stop immediately after {self.extraction_fields[-1]}

REQUIRED OUTPUT FORMAT - EXACTLY {self.field_count} LINES:
"""
        
        # Add each field with simple fallback instruction
        for field in self.extraction_fields:
            instruction = "[value or NOT_FOUND]"
            prompt += f"{field}: {instruction}\n"
        
        prompt += f"""
OUTPUT RULES:
-- NEVER use: **KEY:** or **KEY** or *KEY* or any formatting
-- Plain text only - NO markdown, NO bold, NO italic
-- Include ALL {self.field_count} keys even if value is NOT_FOUND
-- Output ONLY these {self.field_count} lines, nothing else

STOP after {self.extraction_fields[-1]} line. Do not add explanations or comments."""
        
        return prompt
    
    def process_single_image(self, image_path):
        """Override to use document-specific fields in processing and result handling."""
        try:
            start_time = time.time()
            
            # Use parent's memory handling
            from common.gpu_optimization import handle_memory_fragmentation
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            
            # Load and preprocess image
            image = self.load_document_image(image_path)
            
            # Load YAML configuration
            yaml_file = Path(__file__).parent.parent / "common" / "llama_single_pass_prompts.yaml"
            yaml_config = {}
            if yaml_file.exists():
                with yaml_file.open("r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                    yaml_config = yaml_data.get("single_pass", {})
            
            # Generate prompt with document-specific fields
            if yaml_config:
                prompt = self.generate_single_pass_prompt(yaml_config)
            else:
                prompt = self.get_simple_single_pass_prompt()
            
            # Process with model
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            if self.debug:
                print(f"🖼️  Input tensor shape: {inputs['input_ids'].shape}")
                print(f"💭 Generating with max_new_tokens={self.generation_config.max_new_tokens}")
            
            # Generate response
            output = self.model.generate(**inputs, **self.generation_config)
            
            # Decode response
            full_response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the assistant's response (remove the user prompt)
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[1]
                response = response.split("<|eot_id|>")[0].strip()
            else:
                response = full_response.strip()
            
            # Parse response using document-specific fields
            from common.extraction_parser import parse_extraction_response
            extracted_data = parse_extraction_response(response, field_names=self.extraction_fields)
            
            if self.debug:
                print("🎯 EXTRACTION RESULTS:")
                print("=" * 50)
                for i, (key, value) in enumerate(extracted_data.items()):
                    if i < 5:  # Show first 5 results
                        print(f"{key}: {value}")
                if len(extracted_data) > 5:
                    print(f"  ... and {len(extracted_data) - 5} more fields")
                print()
            
            # Calculate metrics using document-specific fields
            extracted_fields_count = len([k for k in extracted_data.keys() if k in self.extraction_fields])
            response_completeness = extracted_fields_count / len(self.extraction_fields)
            content_coverage = extracted_fields_count / len(self.extraction_fields)
            
            # Cleanup
            del inputs, output, image
            
            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": time.time() - start_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "extraction_mode": self.extraction_mode,
                "field_count": self.field_count
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error in single image processing: {e}")
                import traceback
                traceback.print_exc()
            
            # Return error result with document-specific fields
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in self.extraction_fields},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "extraction_mode": self.extraction_mode,
                "field_count": self.field_count
            }
    
    def _create_error_result(self, file_path: str, error_message: str) -> dict:
        """Override error result creation to use document-specific fields."""
        return {
            "image_name": Path(file_path).name,
            "extracted_data": {field: "NOT_FOUND" for field in self.extraction_fields},
            "raw_response": f"Error: {error_message}",
            "processing_time": 0,
            "response_completeness": 0,
            "content_coverage": 0,
            "extracted_fields_count": 0,
            "extraction_mode": self.extraction_mode,
            "field_count": self.field_count
        }