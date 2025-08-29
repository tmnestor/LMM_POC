#!/usr/bin/env python3
"""
Document Type Detection - Phase 1 Implementation

Lightweight classification system to identify document types before field extraction.
Supports: invoice, bank_statement, receipt with confidence scoring and fallback.
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Optional


class DocumentTypeDetector:
    """
    Lightweight document type classification for targeted field extraction.
    
    Uses minimal prompts to quickly identify document type before full extraction,
    enabling document-type-specific schema selection.
    """
    
    def __init__(self, model_processor=None):
        """
        Initialize document type detector.
        
        Args:
            model_processor: Processor instance (LlamaProcessor or InternVL3Processor)
        """
        self.processor = model_processor
        self.supported_types = ["invoice", "bank_statement", "receipt"]
        self.confidence_threshold = 0.85
        
        # Map document variants to canonical schema types
        self.type_to_schema = {
            "invoice": "invoice",
            "tax invoice": "invoice",
            "bill": "invoice",
            "estimate": "invoice",
            "quote": "invoice",
            "quotation": "invoice",
            "proforma invoice": "invoice",
            "receipt": "receipt",
            "purchase receipt": "receipt",
            "payment receipt": "receipt",
            "bank statement": "bank_statement",
            "account statement": "bank_statement",
            "credit card statement": "bank_statement",
            "statement": "bank_statement"
        }
        
        # Classification prompts optimized for speed and accuracy
        self.classification_prompts = {
            "llama": {
                "system": "You are a document classifier. Analyze this business document image.",
                "prompt": """What type of business document is this?

Look for document types like:
- invoice, tax invoice, bill, estimate, quote, quotation
- bank_statement, account statement, credit card statement
- receipt, purchase receipt

Answer with the exact document type you see (e.g., 'invoice', 'estimate', 'receipt', 'bank_statement', etc.).""",
                "max_tokens": 50
            },
            
            "internvl3": {
                "prompt": """What document type is this?

Types: invoice, estimate, quote, receipt, bank_statement, credit card statement

Answer with the exact type:""",
                "max_tokens": 20
            }
        }
    
    def detect_document_type(self, image_path: str) -> Dict:
        """
        Detect document type with confidence scoring.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Dict with 'type', 'confidence', 'reasoning', 'processing_time'
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if not self.processor:
            raise ValueError("No processor configured for classification")
        
        print(f"🔍 Classifying document type: {Path(image_path).name}")
        
        start_time = time.time()
        
        try:
            # Get model-specific classification prompt
            model_name = self._get_processor_type()
            prompt_config = self.classification_prompts.get(model_name, self.classification_prompts["llama"])
            
            # Run classification
            if hasattr(self.processor, '_extract_with_custom_prompt'):
                response = self.processor._extract_with_custom_prompt(
                    image_path, 
                    prompt_config["prompt"],
                    max_new_tokens=prompt_config.get("max_tokens", 100),
                    temperature=0.0,
                    do_sample=False
                )
            else:
                raise AttributeError(f"Processor {type(self.processor)} doesn't support custom prompts")
            
            processing_time = time.time() - start_time
            
            # Debug: Show raw response
            print(f"🔍 Raw model response: '{response.strip()}'")
            
            # Parse classification result
            parsed_result = self._parse_classification_response(response)
            parsed_result['processing_time'] = processing_time
            parsed_result['raw_response'] = response
            
            print(f"✅ Classification: {parsed_result['type']} (confidence: {parsed_result['confidence']:.2f})")
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'reasoning': f'Classification error: {str(e)}',
                'processing_time': time.time() - start_time,
                'fallback_used': True
            }
    
    def _get_processor_type(self) -> str:
        """Determine processor type for prompt selection."""
        processor_class = type(self.processor).__name__.lower()
        if 'llama' in processor_class:
            return 'llama'
        elif 'internvl' in processor_class:
            return 'internvl3'
        else:
            return 'llama'  # Default fallback
    
    def _parse_classification_response(self, response: str) -> Dict:
        """
        Parse classification response and extract type/confidence.
        
        Args:
            response: Raw model response
            
        Returns:
            Dict with parsed classification results
        """
        original_response = response.strip()
        response_lower = response.strip().lower()
        
        # Initialize result
        result = {
            'type': 'unknown',
            'confidence': 0.0,
            'reasoning': 'Failed to parse response',
            'fallback_used': False
        }
        
        # Extract document type
        doc_type = self._extract_document_type(response_lower)
        if doc_type:
            result['type'] = doc_type
            result['reasoning'] = f'Detected {doc_type} from response'
        
        # Extract confidence if present
        confidence = self._extract_confidence(response_lower)
        if confidence is not None:
            result['confidence'] = confidence
        else:
            # Assign default confidence based on type detection strength
            if result['type'] != 'unknown':
                # Higher confidence if type is clearly mentioned multiple times
                type_mentions = response_lower.count(result['type'])
                if 'invoice' in response_lower and result['type'] == 'invoice':
                    type_mentions += response_lower.count('bill') + response_lower.count('tax invoice')
                elif 'statement' in response_lower and result['type'] == 'bank_statement':
                    type_mentions += response_lower.count('bank') + response_lower.count('account')
                elif 'receipt' in response_lower and result['type'] == 'receipt':
                    type_mentions += response_lower.count('payment') + response_lower.count('purchase')
                
                # Set confidence based on strength of detection
                if type_mentions >= 3:
                    result['confidence'] = 0.95
                elif type_mentions >= 2:
                    result['confidence'] = 0.90
                else:
                    result['confidence'] = 0.85
        
        # Extract reasoning if present
        reasoning = self._extract_reasoning(original_response)
        if reasoning:
            result['reasoning'] = reasoning
        elif result['type'] != 'unknown':
            # Provide basic reasoning based on detection
            result['reasoning'] = f'Identified as {result["type"]} based on document characteristics'
        
        # Apply fallback logic if confidence too low
        if result['confidence'] < self.confidence_threshold and result['type'] == 'unknown':
            result = self._apply_fallback_classification(response_lower, result)
        
        return result
    
    def _extract_document_type(self, response: str) -> Optional[str]:
        """Extract document type from response."""
        response_lower = response.lower().strip()
        
        # First check for exact matches to any known type
        for doc_type in self.type_to_schema.keys():
            if doc_type.lower() in response_lower:
                # Return the actual detected type (not the schema type)
                return doc_type.lower()
        
        # Look for explicit format with type field
        for line in response.split('\n'):
            line = line.strip().lower()
            if 'document_type:' in line or 'type:' in line:
                # Check all known types
                for doc_type in self.type_to_schema.keys():
                    if doc_type.lower() in line:
                        return doc_type.lower()
        
        # Fallback: look for supported schema types
        for doc_type in self.supported_types:
            if doc_type in response_lower:
                return doc_type
        
        return None
    
    def _extract_confidence(self, response: str) -> Optional[float]:
        """Extract confidence score from response."""
        import re
        
        # Look for confidence patterns
        patterns = [
            r'confidence:\s*([0-9]*\.?[0-9]+)',
            r'confidence\s*=\s*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*confidence'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    confidence = float(match.group(1))
                    # Normalize to 0-1 range if needed
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    return min(max(confidence, 0.0), 1.0)
                except ValueError:
                    continue
        
        return None
    
    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from response."""
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('reasoning:'):
                return line[10:].strip()  # Remove "reasoning:" prefix
        
        return None
    
    def _apply_fallback_classification(self, response: str, current_result: Dict) -> Dict:
        """Apply fallback logic for low-confidence classifications."""
        
        # Keyword-based fallback classification
        keywords = {
            'invoice': ['invoice', 'due date', 'bill to', 'line item', 'subtotal', 'tax invoice'],
            'bank_statement': ['account', 'statement', 'balance', 'transaction', 'bsb', 'deposit', 'withdrawal', 'credit card', 'visa', 'mastercard', 'amex'],
            'receipt': ['receipt', 'payment', 'cash', 'card', 'store', 'purchase', 'thank you']
        }
        
        # Count keyword matches
        keyword_scores = {}
        response_lower = response.lower()
        
        for doc_type, type_keywords in keywords.items():
            score = sum(1 for keyword in type_keywords if keyword in response_lower)
            keyword_scores[doc_type] = score
        
        # Find best keyword match
        if keyword_scores:
            best_type = max(keyword_scores, key=keyword_scores.get)
            best_score = keyword_scores[best_type]
            
            if best_score > 0:
                fallback_confidence = min(0.6 + (best_score * 0.1), 0.9)
                return {
                    'type': best_type,
                    'confidence': fallback_confidence,
                    'reasoning': f'Keyword-based classification (score: {best_score})',
                    'fallback_used': True
                }
        
        # Ultimate fallback - return as unknown but mark for manual review
        return {
            'type': 'unknown',
            'confidence': 0.0,
            'reasoning': 'Unable to classify - requires manual review',
            'fallback_used': True,
            'manual_review_needed': True
        }
    
    def batch_classify_images(self, image_directory: str) -> List[Dict]:
        """
        Classify multiple images in a directory.
        
        Args:
            image_directory: Directory containing images to classify
            
        Returns:
            List of classification results
        """
        try:
            # Simple image discovery without circular imports
            image_path = Path(image_directory)
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
            images = [str(f) for f in image_path.rglob('*') if f.suffix.lower() in image_extensions]
        except Exception as e:
            print(f"❌ Error discovering images: {e}")
            return []
        
        if not images:
            print(f"❌ No images found in {image_directory}")
            return []
        
        print(f"🔬 Batch classifying {len(images)} images...")
        
        results = []
        for i, image_path in enumerate(images, 1):
            print(f"\n📄 Image {i}/{len(images)}: {Path(image_path).name}")
            try:
                result = self.detect_document_type(str(image_path))
                result['image_path'] = str(image_path)
                result['image_name'] = Path(image_path).name
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to classify {Path(image_path).name}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'image_name': Path(image_path).name,
                    'type': 'error',
                    'confidence': 0.0,
                    'reasoning': f'Classification error: {str(e)}',
                    'error': True
                })
        
        return results
    
    def generate_classification_report(self, results: List[Dict]) -> str:
        """Generate summary report of batch classification results."""
        if not results:
            return "No classification results to report."
        
        # Count results by type
        type_counts = {}
        confidence_sum = {}
        errors = 0
        fallbacks = 0
        manual_reviews = 0
        
        for result in results:
            if result.get('error'):
                errors += 1
                continue
                
            doc_type = result['type']
            confidence = result['confidence']
            
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            confidence_sum[doc_type] = confidence_sum.get(doc_type, 0) + confidence
            
            if result.get('fallback_used'):
                fallbacks += 1
            if result.get('manual_review_needed'):
                manual_reviews += 1
        
        # Generate report
        total = len(results)
        successful = total - errors
        
        report = f"""
📊 DOCUMENT TYPE CLASSIFICATION REPORT
{'='*50}

📈 OVERVIEW:
   Total Images: {total}
   Successfully Classified: {successful}
   Errors: {errors}
   Fallbacks Used: {fallbacks}
   Manual Review Needed: {manual_reviews}

📋 CLASSIFICATION RESULTS:
"""
        
        for doc_type in sorted(type_counts.keys()):
            count = type_counts[doc_type]
            avg_confidence = confidence_sum[doc_type] / count if count > 0 else 0
            percentage = (count / successful * 100) if successful > 0 else 0
            
            report += f"   {doc_type.upper()}: {count} documents ({percentage:.1f}%) - Avg confidence: {avg_confidence:.2f}\n"
        
        # Success metrics
        high_confidence = sum(1 for r in results if not r.get('error') and r['confidence'] >= self.confidence_threshold)
        success_rate = (high_confidence / successful * 100) if successful > 0 else 0
        
        report += f"""
🎯 QUALITY METRICS:
   High Confidence Classifications: {high_confidence}/{successful} ({success_rate:.1f}%)
   Confidence Threshold: {self.confidence_threshold}
   Overall Success Rate: {(successful/total*100):.1f}%
"""
        
        if manual_reviews > 0:
            report += f"\n⚠️  ATTENTION: {manual_reviews} documents need manual review\n"
        
        return report


def main():
    """Test document type detection with sample images."""
    print("🚀 Document Type Detection - Phase 1 Testing")
    
    # For testing, we need a processor instance
    print("\n💡 Note: This requires running on a machine with model access")
    print("💡 Import and initialize LlamaProcessor or InternVL3Processor first")
    
    # Example usage:
    print("""
Example usage:
    from models.llama_processor import LlamaProcessor
    from common.document_type_detector import DocumentTypeDetector
    
    # Initialize processor and detector
    processor = LlamaProcessor()
    detector = DocumentTypeDetector(processor)
    
    # Classify single image
    result = detector.detect_document_type("evaluation_data/synthetic_invoice_001.png")
    print(result)
    
    # Batch classify directory
    results = detector.batch_classify_images("evaluation_data/")
    report = detector.generate_classification_report(results)
    print(report)
    """)


# ============================================================================
# LIGHTWEIGHT DOCUMENT DETECTOR FOR V4 INTEGRATION
# ============================================================================

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
            
            # Check for generic filename patterns that need content-based detection
            if re.match(r'^(image|doc|document|file|test)_?\d*$', filename):
                if self.debug:
                    print(f"🔍 Generic filename detected: {filename}")
                    print("💡 Consider using content-based detection or descriptive filenames")
                # Keep original default - invoices are typically most common business documents
                return "invoice"
            
            # Final fallback - default to invoice (most common business document type)
            if self.debug:
                print("📄 No specific pattern matched, defaulting to: invoice")
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


if __name__ == "__main__":
    main()