"""
Lightweight Document Type Detector for V4 Schema Integration

Provides fast document type classification to enable intelligent field filtering
without requiring heavy model loading for document-aware extraction.
"""

import re
from pathlib import Path
from typing import Optional


class LightweightDocumentDetector:
    """
    Lightweight document type detector using filename and basic heuristics.
    
    Designed for V4 schema integration to enable document-aware field filtering
    without requiring heavy model inference for document classification.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the lightweight detector."""
        self.debug = debug
        
        # Document type patterns for filename-based detection
        self.filename_patterns = {
            "invoice": [
                r"invoice", r"inv_", r"bill", r"tax_invoice", 
                r"commercial_invoice", r"proforma"
            ],
            "receipt": [
                r"receipt", r"rcpt", r"purchase", r"transaction",
                r"pos_", r"retail", r"store"
            ],
            "statement": [
                r"statement", r"stmt", r"bank", r"account",
                r"credit_card", r"financial", r"balance"
            ]
        }
        
        if self.debug:
            print("🔍 LightweightDocumentDetector initialized")
            print(f"   Supported types: {list(self.filename_patterns.keys())}")
    
    def classify_document_type(self, image_path: str) -> str:
        """
        Classify document type based on filename patterns and heuristics.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Document type ("invoice", "receipt", "statement", or "invoice" as default)
        """
        try:
            # Extract filename for pattern matching
            filename = Path(image_path).stem.lower()
            
            if self.debug:
                print(f"🔍 Analyzing filename: {filename}")
            
            # Check filename patterns
            for doc_type, patterns in self.filename_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, filename):
                        if self.debug:
                            print(f"📄 Detected type: {doc_type} (pattern: {pattern})")
                        return doc_type
            
            # Default classification logic based on common patterns
            if any(keyword in filename for keyword in ["synthetic_invoice", "sample_invoice"]):
                return "invoice"
            elif any(keyword in filename for keyword in ["synthetic_receipt", "sample_receipt"]):
                return "receipt"
            elif any(keyword in filename for keyword in ["synthetic_statement", "sample_statement"]):
                return "statement"
            
            # Final fallback - default to invoice (most common document type)
            if self.debug:
                print(f"📄 No specific pattern matched, defaulting to: invoice")
            return "invoice"
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Document detection failed: {e}, defaulting to invoice")
            return "invoice"
    
    def get_confidence_score(self, image_path: str, detected_type: str) -> float:
        """
        Get confidence score for document type detection.
        
        Args:
            image_path (str): Path to image file
            detected_type (str): Detected document type
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        try:
            filename = Path(image_path).stem.lower()
            
            # High confidence for explicit patterns
            high_confidence_patterns = {
                "invoice": ["invoice", "tax_invoice", "commercial_invoice"],
                "receipt": ["receipt", "purchase", "transaction"],
                "statement": ["statement", "bank", "account"]
            }
            
            if detected_type in high_confidence_patterns:
                for pattern in high_confidence_patterns[detected_type]:
                    if pattern in filename:
                        return 0.9  # High confidence
            
            # Medium confidence for partial matches
            if detected_type != "invoice":  # Non-default classification
                return 0.7
            
            # Low confidence for default classification
            return 0.5
            
        except Exception:
            return 0.3  # Very low confidence on error
    
    def get_supported_document_types(self) -> list:
        """Get list of supported document types."""
        return list(self.filename_patterns.keys())
    
    def validate_document_path(self, image_path: str) -> bool:
        """
        Validate that the image path exists and is accessible.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            bool: True if path is valid and accessible
        """
        try:
            path = Path(image_path)
            return path.exists() and path.is_file()
        except Exception:
            return False


def create_lightweight_detector(debug: bool = False) -> LightweightDocumentDetector:
    """
    Factory function to create a lightweight document detector.
    
    Args:
        debug (bool): Enable debug logging
        
    Returns:
        LightweightDocumentDetector: Configured detector instance
    """
    return LightweightDocumentDetector(debug=debug)


# Quick test function for development
if __name__ == "__main__":
    # Test the detector with sample filenames
    detector = LightweightDocumentDetector(debug=True)
    
    test_filenames = [
        "synthetic_invoice_001.png",
        "sample_receipt_retail.png",
        "bank_statement_monthly.png",
        "tax_invoice_commercial.png",
        "unknown_document.png"
    ]
    
    print("\n=== DOCUMENT TYPE DETECTION TEST ===")
    for filename in test_filenames:
        doc_type = detector.classify_document_type(filename)
        confidence = detector.get_confidence_score(filename, doc_type)
        print(f"📄 {filename} → {doc_type} (confidence: {confidence:.1f})")