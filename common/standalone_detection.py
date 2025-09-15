"""
Standalone Document Type Detection

A simplified document type detection module for the clean architecture.
Uses YAML-first detection approach without complex dependencies.
"""

import time
from pathlib import Path
from typing import Any, Dict, Tuple

from rich import print as rprint


class StandaloneDocumentDetector:
    """Standalone document type detector using YAML prompts."""

    def __init__(self):
        """Initialize the detector with type mappings."""
        self.type_mappings = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "commercial invoice": "invoice",
            "bill": "invoice",
            "estimate": "invoice",
            "quote": "invoice",
            "quotation": "invoice",
            "receipt": "receipt",
            "purchase receipt": "receipt",
            "payment receipt": "receipt",
            "sales receipt": "receipt",
            "bank statement": "bank_statement",
            "account statement": "bank_statement",
            "credit card statement": "bank_statement",
            "statement": "bank_statement"
        }

        self.supported_types = ["invoice", "receipt", "bank_statement"]

    def detect_document_type(self, image_path: Path, processor: Any, prompt_config: Dict) -> Tuple[str, Dict]:
        """
        Detect document type using YAML-first approach.

        Args:
            image_path: Path to the image file
            processor: Model processor instance
            prompt_config: Prompt configuration dictionary

        Returns:
            Tuple of (document_type, detection_info)
        """

        try:
            # Load detection prompt using the prompt loader
            from .prompt_loader import PromptLoader
            loader = PromptLoader()

            detection_prompt, prompt_name, prompt_description = loader.load_prompt(
                prompt_file=prompt_config['detection_file'],
                prompt_key=prompt_config['detection_key'],
                document_type="detection",
                verbose=False
            )

            # Create detection info for metadata
            detection_info = {
                "prompt": detection_prompt,
                "source": "yaml_prompt_loader",
                "field_count": 0,  # Detection doesn't extract fields
                "template_type": "document_type_detection",
                "prompt_file": prompt_config['detection_file'],
                "prompt_key": prompt_config['detection_key'],
                "prompt_name": prompt_name,
                "prompt_description": prompt_description
            }

            # Use the processor to detect document type
            start_time = time.perf_counter()

            # Use the internal method to get response
            if hasattr(processor, '_extract_with_prompt'):
                response = processor._extract_with_prompt(str(image_path), detection_prompt)
            elif hasattr(processor, '_generate_response'):
                response = processor._generate_response(str(image_path), detection_prompt)
            else:
                raise ValueError("Processor doesn't have expected response generation method")

            detection_time = time.perf_counter() - start_time
            detection_info["detection_time"] = detection_time

            if response:
                # Clean and normalize the response
                doc_type = self._normalize_response(response.strip())

                # Validate detected type
                if doc_type in self.supported_types:
                    detection_info["detection_confidence"] = "high"
                    return doc_type, detection_info
                else:
                    # Try partial matching
                    doc_type = self._partial_match(response.strip())
                    if doc_type:
                        detection_info["detection_confidence"] = "medium"
                        return doc_type, detection_info
                    else:
                        # Fallback to invoice
                        detection_info["detection_confidence"] = "low_fallback"
                        return "invoice", detection_info
            else:
                raise ValueError("No response from processor")

        except Exception as e:
            rprint(f"[yellow]⚠️ YAML document detection failed: {e}[/yellow]")
            rprint("[yellow]   Falling back to filename-based detection[/yellow]")

            # Fallback to filename-based detection
            doc_type = self._filename_based_detection(image_path)

            fallback_info = {
                "prompt": f"Filename-based detection (YAML failed: {str(e)})",
                "source": "filename_pattern_matching_fallback",
                "field_count": 0,
                "template_type": "error_fallback_detection",
                "prompt_file": "N/A",
                "prompt_key": "error_fallback",
                "detection_confidence": "fallback"
            }

            return doc_type, fallback_info

    def _normalize_response(self, response: str) -> str:
        """Normalize detection response using type mappings."""
        response_lower = response.lower().strip()

        # Direct mapping lookup
        if response_lower in self.type_mappings:
            return self.type_mappings[response_lower]

        # Extract first word if it contains known types
        first_word = response_lower.split()[0] if response_lower.split() else ""
        if first_word in self.type_mappings:
            return self.type_mappings[first_word]

        return response_lower

    def _partial_match(self, response: str) -> str:
        """Try partial matching for common variations."""
        response_lower = response.lower()

        if "receipt" in response_lower:
            return "receipt"
        elif "statement" in response_lower or "bank" in response_lower:
            return "bank_statement"
        elif "invoice" in response_lower or "bill" in response_lower:
            return "invoice"

        return None

    def _filename_based_detection(self, image_path: Path) -> str:
        """Fallback filename-based detection."""
        image_name = image_path.name.lower()

        if "statement" in image_name or "bank" in image_name:
            return "bank_statement"
        elif "receipt" in image_name:
            return "receipt"
        else:
            return "invoice"  # Default fallback


def main():
    """Test the standalone detector."""
    detector = StandaloneDocumentDetector()

    # Test normalization
    test_responses = [
        "INVOICE",
        "TAX INVOICE",
        "RECEIPT",
        "BANK STATEMENT",
        "bank_statement",
        "unknown document"
    ]

    for response in test_responses:
        normalized = detector._normalize_response(response)
        print(f"'{response}' -> '{normalized}'")


if __name__ == "__main__":
    main()