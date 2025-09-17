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

from PIL import Image
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

            detection_prompt = self.prompt_loader.load_prompt(detection_file, detection_key)

            # Process image with detection prompt
            image = Image.open(image_path).convert('RGB')

            # Use InternVL3 chat method for detection
            response = self.model.chat(
                self.tokenizer,
                pixel_values=None,  # Will be processed internally
                question=detection_prompt,
                generation_config=dict(
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.0
                ),
                image=image
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
            extraction_prompt = self.prompt_loader.load_prompt(prompt_file, prompt_key)

            # Process image with extraction prompt
            image = Image.open(image_path).convert('RGB')

            # Use InternVL3 chat method for extraction
            response = self.model.chat(
                self.tokenizer,
                pixel_values=None,  # Will be processed internally
                question=extraction_prompt,
                generation_config=dict(
                    max_new_tokens=self.config.get('MAX_NEW_TOKENS', 800),
                    do_sample=False,
                    temperature=0.0
                ),
                image=image
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
        """Parse document type from detection response."""
        if not response:
            return "invoice"

        response_lower = response.lower().strip()

        # Look for document type keywords
        if "receipt" in response_lower:
            return "receipt"
        elif "bank" in response_lower or "statement" in response_lower:
            return "bank_statement"
        elif "invoice" in response_lower or "bill" in response_lower:
            return "invoice"
        else:
            return "invoice"  # Default fallback

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
        """Parse extraction response into structured field data."""
        extracted_data = {}

        if not response:
            # Return NOT_FOUND for all expected fields
            return {field: "NOT_FOUND" for field in expected_fields}

        # Simple parsing: look for "FIELD_NAME: value" patterns
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line:
                # Split on first colon only
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field_name = parts[0].strip()
                    field_value = parts[1].strip()

                    # Clean up field name (remove any prefixes)
                    if field_name in expected_fields:
                        extracted_data[field_name] = field_value

        # Ensure all expected fields are present
        for field in expected_fields:
            if field not in extracted_data:
                extracted_data[field] = "NOT_FOUND"

        return extracted_data