#!/usr/bin/env python3
"""
InternVL3 Document-Aware Handler - Simplified for New Architecture

This module implements a simple document-aware extraction handler for InternVL3 models,
compatible with the unified BatchDocumentProcessor and simplified architecture.

Key Features:
- Compatible with BatchDocumentProcessor model type detection
- Uses SimplePromptLoader (no complex templates)
- Uses SimpleModelEvaluator (no legacy evaluation)
- Simple constructor accepting model and tokenizer directly
- Direct InternVL3 processing without complex wrappers

Usage:
    handler = DocumentAwareInternVL3Handler(model=model, tokenizer=tokenizer, config=config)
    classification_info = handler.detect_and_classify_document(image_path)
    result = handler.process_document_aware(image_path, classification_info)
"""

import time
from typing import Any, Dict

from rich import print as rprint

from common.simple_model_evaluator import SimpleModelEvaluator
from common.simple_prompt_loader import SimplePromptLoader


class DocumentAwareInternVL3Handler:
    """Simple Document-Aware InternVL3 Handler for unified BatchDocumentProcessor."""

    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        """
        Initialize handler with model components and config.

        Args:
            model: Loaded InternVL3 model
            tokenizer: InternVL3 tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize simplified components
        self.prompt_loader = SimplePromptLoader()
        self.evaluator = SimpleModelEvaluator()

        # Extract prompt config from main config
        self.prompt_config = config.get('PROMPT_CONFIG', {})

        rprint("[green]✅ DocumentAwareInternVL3Handler initialized with simplified architecture[/green]")

    def detect_and_classify_document(self, image_path: str) -> Dict[str, Any]:
        """
        Detect document type using simple prompt-based approach.

        Args:
            image_path: Path to document image

        Returns:
            Dict with document type and processing info
        """
        try:
            # Load detection prompt
            detection_file = self.prompt_config.get('detection_file', 'prompts/document_type_detection.yaml')
            detection_key = self.prompt_config.get('detection_key', 'detection')

            # Extract just the filename from the path for SimplePromptLoader
            detection_filename = detection_file.split('/')[-1] if '/' in detection_file else detection_file
            detection_prompt = self.prompt_loader.load_prompt(detection_filename, detection_key)

            # Process image with detection prompt
            # For simplicity, use a basic processor approach
            from models.document_aware_internvl3_processor import (
                DocumentAwareInternVL3Processor,
            )

            # Create a temporary processor for image loading
            temp_processor = DocumentAwareInternVL3Processor(
                field_list=["DOCUMENT_TYPE"],
                pre_loaded_model=self.model,
                pre_loaded_tokenizer=self.tokenizer,
                skip_model_loading=True,  # Use existing model
                debug=False
            )

            # Load and preprocess image
            pixel_values = temp_processor.load_image(image_path)

            # Move to appropriate device and convert dtype (match working processor)
            import torch
            if torch.cuda.is_available():
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            else:
                pixel_values = pixel_values.to(torch.float32)

            # Use InternVL3 chat method for detection
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                detection_prompt,
                generation_config=dict(
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.0
                ),
                history=None,
                return_history=False
            )

            # Parse document type from response
            doc_type = self._parse_document_type(response)

            # Map to field information
            doc_type_upper = doc_type.upper()
            field_info = self._get_field_info_for_type(doc_type_upper)

            return {
                "document_type": doc_type,
                "field_count": field_info["field_count"],
                "field_names": field_info["field_names"],
                "extraction_prompt_key": field_info["prompt_key"]
            }

        except Exception as e:
            rprint(f"[red]❌ Document detection failed: {e}[/red]")
            # Fallback to invoice
            fallback_info = self._get_field_info_for_type("INVOICE")
            return {
                "document_type": "invoice",
                "field_count": fallback_info["field_count"],
                "field_names": fallback_info["field_names"],
                "extraction_prompt_key": fallback_info["prompt_key"]
            }

    def process_document_aware(self, image_path: str, classification_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document with type-aware extraction.

        Args:
            image_path: Path to document image
            classification_info: Result from detect_and_classify_document

        Returns:
            Dict with extraction results
        """
        start_time = time.time()

        try:
            doc_type = classification_info["document_type"]
            prompt_key = classification_info["extraction_prompt_key"]

            # Load extraction prompt for document type
            extraction_files = self.prompt_config.get('extraction_files', {})
            doc_type_upper = doc_type.upper()

            # Get the prompt file for this document type
            prompt_file = extraction_files.get(doc_type_upper, 'prompts/internvl3_prompts.yaml')

            # Extract just the filename from the path for SimplePromptLoader
            prompt_filename = prompt_file.split('/')[-1] if '/' in prompt_file else prompt_file
            extraction_prompt = self.prompt_loader.load_prompt(prompt_filename, prompt_key)

            # Process image with extraction prompt
            # Use the same processor approach for consistency
            from models.document_aware_internvl3_processor import (
                DocumentAwareInternVL3Processor,
            )

            # Create a temporary processor for image loading
            temp_processor = DocumentAwareInternVL3Processor(
                field_list=classification_info["field_names"],
                pre_loaded_model=self.model,
                pre_loaded_tokenizer=self.tokenizer,
                skip_model_loading=True,  # Use existing model
                debug=False
            )

            # Load and preprocess image
            pixel_values = temp_processor.load_image(image_path)

            # Move to appropriate device and convert dtype (match working processor)
            import torch
            if torch.cuda.is_available():
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            else:
                pixel_values = pixel_values.to(torch.float32)

            # Use InternVL3 chat method for extraction
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                extraction_prompt,
                generation_config=dict(
                    max_new_tokens=self.config.get('MAX_NEW_TOKENS', 800),
                    do_sample=False,
                    temperature=0.0
                ),
                history=None,
                return_history=False
            )

            # Parse extraction response into structured data
            extracted_data = self._parse_extraction_response(response, classification_info["field_names"])

            processing_time = time.time() - start_time

            return {
                "document_type": doc_type,
                "extracted_data": extracted_data,
                "processing_time": processing_time,
                "response_text": response,
                "prompt_used": f"internvl3_{prompt_key}"
            }

        except Exception as e:
            processing_time = time.time() - start_time
            rprint(f"[red]❌ Document processing failed: {e}[/red]")

            return {
                "document_type": classification_info.get("document_type", "unknown"),
                "extracted_data": {},
                "processing_time": processing_time,
                "error": str(e)
            }

    def _parse_document_type(self, response: str) -> str:
        """Parse document type from detection response with enhanced debugging."""
        rprint("[dim]🔍 DEBUG: Parsing document type from response[/dim]")
        if response:
            rprint(f"[dim]📝 Detection response: {response[:100]}...[/dim]")

        if not response:
            rprint("[dim]⚠️ Empty detection response - defaulting to invoice[/dim]")
            return "invoice"

        response_lower = response.lower().strip()

        # Look for document type keywords with enhanced matching
        detected_type = "invoice"  # Default

        if "receipt" in response_lower:
            detected_type = "receipt"
        elif "bank" in response_lower or "statement" in response_lower:
            detected_type = "bank_statement"
        elif "invoice" in response_lower or "bill" in response_lower:
            detected_type = "invoice"

        rprint(f"[dim]📋 Detected document type: {detected_type}[/dim]")
        return detected_type

    def _get_field_info_for_type(self, doc_type: str) -> Dict[str, Any]:
        """Get field information for document type."""
        # Define field sets for each document type
        field_sets = {
            "INVOICE": {
                "field_names": [
                    "DOCUMENT_TYPE", "BUSINESS_ABN", "SUPPLIER_NAME", "BUSINESS_ADDRESS",
                    "PAYER_NAME", "PAYER_ADDRESS", "INVOICE_DATE", "LINE_ITEM_DESCRIPTIONS",
                    "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES",
                    "IS_GST_INCLUDED", "GST_AMOUNT", "TOTAL_AMOUNT"
                ],
                "prompt_key": "invoice"
            },
            "RECEIPT": {
                "field_names": [
                    "DOCUMENT_TYPE", "BUSINESS_ABN", "SUPPLIER_NAME", "BUSINESS_ADDRESS",
                    "PAYER_NAME", "PAYER_ADDRESS", "INVOICE_DATE", "LINE_ITEM_DESCRIPTIONS",
                    "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES",
                    "IS_GST_INCLUDED", "GST_AMOUNT", "TOTAL_AMOUNT"
                ],
                "prompt_key": "receipt"
            },
            "BANK_STATEMENT": {
                "field_names": [
                    "DOCUMENT_TYPE", "STATEMENT_DATE_RANGE", "LINE_ITEM_DESCRIPTIONS",
                    "TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID", "TRANSACTION_AMOUNTS_RECEIVED",
                    "ACCOUNT_BALANCE"
                ],
                "prompt_key": "bank_statement"
            }
        }

        field_info = field_sets.get(doc_type, field_sets["INVOICE"])
        field_info["field_count"] = len(field_info["field_names"])

        return field_info

    def _parse_extraction_response(self, response: str, expected_fields: list) -> Dict[str, str]:
        """Parse extraction response into structured field data with enhanced debugging."""
        extracted_data = {}

        # Debug logging
        rprint(f"[dim]🔍 DEBUG: Parsing response ({len(response) if response else 0} chars)[/dim]")
        if response:
            rprint(f"[dim]📝 Raw response preview: {response[:200]}...[/dim]")

        if not response:
            rprint("[dim]⚠️ Empty response - returning NOT_FOUND for all fields[/dim]")
            return {field: "NOT_FOUND" for field in expected_fields}

        # Enhanced parsing: handle multiple formats
        lines = response.strip().split('\n')

        rprint(f"[dim]📋 Processing {len(lines)} response lines[/dim]")

        for line in lines:
            line = line.strip()
            if ':' in line:
                # Split on first colon only
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field_name = parts[0].strip()
                    field_value = parts[1].strip()

                    # Clean up field name (remove any prefixes, brackets, numbers)
                    clean_field_name = field_name.upper()

                    # Remove common prefixes that might be added
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                                   '11.', '12.', '13.', '14.', '15.', '-', '*', '•']:
                        if clean_field_name.startswith(prefix):
                            clean_field_name = clean_field_name[len(prefix):].strip()

                    # Check if this matches any expected field
                    for expected_field in expected_fields:
                        if clean_field_name == expected_field.upper():
                            extracted_data[expected_field] = field_value
                            rprint(f"[dim]✅ Found {expected_field}: {field_value[:50]}...[/dim]")
                            break
                    else:
                        rprint(f"[dim]❌ Unknown field '{clean_field_name}' (original: '{field_name}')[/dim]")

        # Ensure all expected fields are present
        for field in expected_fields:
            if field not in extracted_data:
                extracted_data[field] = "NOT_FOUND"

        found_count = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")
        rprint(f"[dim]📊 Extraction summary: {found_count}/{len(expected_fields)} fields found[/dim]")

        return extracted_data