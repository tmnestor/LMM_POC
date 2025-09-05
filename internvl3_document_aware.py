#!/usr/bin/env python3
"""
InternVL3 Vision Document-Aware Key-Value Extraction - V4 Schema Implementation

This module implements the V4 document-aware extraction pipeline for the
InternVL3 model (2B/8B variants), featuring:
- Document type detection and classification
- Type-specific field schema routing
- Comprehensive extraction (invoice: 29 fields, receipt: 20 fields, bank_statement: 16 fields)
- ATO compliance validation for invoices
- Complete field coverage with v4 schema including payment tracking

Pipeline Flow:
    1. Document Type Detection - Classify document type with confidence scoring
    2. Schema Routing - Select appropriate field schema for document type
    3. Targeted Extraction - Extract only relevant fields for document type
    4. Type-Specific Evaluation - Apply document-specific validation rules
    5. ATO Compliance Check - Validate tax invoice compliance (Australian requirements)
    6. Performance Reporting - Generate document-type-specific metrics

Usage:
    python internvl3_document_aware.py --limit-images 5 --debug
    python internvl3_document_aware.py --document-type invoice
    python internvl3_document_aware.py --image-path "path/to/single_image.jpg" --debug
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from common.document_type_metrics import DocumentTypeEvaluator
from common.evaluation_metrics import load_ground_truth
from common.extraction_parser import create_extraction_dataframe, discover_images

# Import unified schema loader (simplified YAML-first)
from common.unified_schema import DocumentTypeFieldSchema
from models.document_aware_internvl3_processor import DocumentAwareInternVL3Processor

# Universal extraction field list - eliminates document type detection
UNIVERSAL_FIELDS = [
    "DOCUMENT_TYPE", "INVOICE_DATE", "SUPPLIER_NAME", "BUSINESS_ABN", "BUSINESS_ADDRESS",
    "PAYER_NAME", "PAYER_ADDRESS", "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_TOTAL_PRICES", 
    "GST_AMOUNT", "IS_GST_INCLUDED", "TOTAL_AMOUNT", "STATEMENT_DATE_RANGE",
    "TRANSACTION_DATES", "TRANSACTION_AMOUNTS_PAID"
]


class DocumentAwareInternVL3Handler:
    """V4 Document-Aware InternVL3 Vision Processor with comprehensive field coverage."""

    def __init__(self, model_path: str, debug: bool = False):
        """Initialize document-aware processor."""
        self.debug = debug
        self.model_path = model_path

        print(
            "🚀 Initializing InternVL3 Vision processor for V4 document-aware extraction..."
        )

        # We'll create processors on-demand to avoid loading multiple models
        self.base_processor = None
        self.model_loaded = False

        # Initialize V4 components
        self.schema_loader = DocumentTypeFieldSchema()
        self.evaluator = DocumentTypeEvaluator()

        # Load document detection prompts from unified schema (single source of truth)
        self.detection_config = self.schema_loader.load_detection_prompts()

        if self.debug:
            print("📝 YAML-first prompt loader initialized")
            print(
                f"   Detection config version: {self.detection_config.get('version', 'unknown')}"
            )
            print(
                f"   Supported types: {len(self.detection_config.get('supported_types', []))}"
            )

        # Schema loader initialized - ready for document-aware processing

        # Note: Document detector will be configured when we need it
        self.document_detector = None

        print(
            "✅ Document-aware InternVL3 handler initialized (model will load on first use)"
        )

    def detect_and_classify_document(self, image_path: str) -> Dict[str, Any]:
        """Detect document type and get appropriate schema."""

        if self.debug:
            print(f"📋 Detecting document type for: {image_path}")

        # Initialize processor for direct extraction (more reliable than detector)
        if not self.base_processor:
            # Create processor for document type detection using single field
            detection_fields = ["DOCUMENT_TYPE"]
            self.base_processor = DocumentAwareInternVL3Processor(
                field_list=detection_fields,
                model_path=self.model_path,
                debug=self.debug,
            )

        # Phase 1: YAML-first document type detection
        if self.debug:
            print("🔍 Phase 1: YAML-first DOCUMENT_TYPE detection")

        try:
            # Use YAML-first document detection instead of generic field extraction
            doc_type = self._detect_document_type_yaml(image_path)

            if self.debug:
                print(f"   YAML-based detection result: '{doc_type}'")

        except Exception as e:
            if self.debug:
                print(f"   ⚠️ YAML detection failed: {e}, using fallback")
            doc_type = self.detection_config["detection_config"].get(
                "fallback_type", "invoice"
            )

        # Get schema for correctly detected document type
        schema = self.schema_loader.get_document_schema(doc_type)

        if self.debug:
            print(f"   Document Type: {doc_type}")
            print(f"   Schema Fields: {len(schema['fields'])} fields")
            print(f"   Extraction Mode: {schema['extraction_mode']}")
            print(f"   🔍 DEBUG: First 5 schema fields: {schema['fields'][:5]}")
            print(f"   🔍 DEBUG: Total schema fields: {len(schema['fields'])}")

        # Extract field names
        field_names = (
            schema["fields"]
            if isinstance(schema["fields"][0], str)
            else [f["name"] for f in schema["fields"]]
        )

        if self.debug:
            print(f"   🔍 DEBUG: Field names length: {len(field_names)}")
            print(f"   🔍 DEBUG: First 5 field names: {field_names[:5]}")

        return {
            "document_type": doc_type,
            "schema": schema,
            "field_count": len(schema["fields"]),
            "field_names": field_names,
        }

    def _normalize_document_type(self, raw_type: str) -> str:
        """Normalize raw document type to canonical schema type."""
        if not raw_type or raw_type.lower() in ["unknown", "not_found"]:
            return "invoice"  # Default fallback

        normalized = raw_type.lower().strip()

        # Direct mapping of common variations
        type_mapping = {
            # Bank statement variations
            "statement": "bank_statement",
            "bank statement": "bank_statement",
            "account statement": "bank_statement",
            "credit card statement": "bank_statement",
            "financial statement": "bank_statement",
            # Invoice variations
            "invoice": "invoice",
            "tax invoice": "invoice",
            "bill": "invoice",
            "estimate": "invoice",
            "quote": "invoice",
            # Receipt variations
            "receipt": "receipt",
            "purchase receipt": "receipt",
            "sales receipt": "receipt",
        }

        return type_mapping.get(normalized, "invoice")  # Default to invoice if no match

    def _detect_document_type_yaml(self, image_path: str) -> str:
        """
        YAML-first document type detection using configurable prompts.

        Uses prompts/document_type_detection.yaml for maintainable prompt management.
        """
        if self.debug:
            print("📝 Using YAML-first document detection approach")

        # Get InternVL3-specific prompt from YAML configuration
        internvl3_config = self.detection_config["detection_prompts"]["internvl3"]
        doc_type_prompt = internvl3_config["user_prompt"]
        max_tokens = internvl3_config.get("max_tokens", 20)

        if self.debug:
            print(
                f"   YAML config version: {self.detection_config.get('version', 'unknown')}"
            )
            print(f"   Max tokens: {max_tokens}")
            print(f"   Prompt: {doc_type_prompt[:100]}...")

        # Use the base processor to extract with YAML prompt
        response = self.base_processor._extract_with_custom_prompt(
            image_path, doc_type_prompt, max_new_tokens=max_tokens
        )

        # Parse and normalize using YAML type mappings
        doc_type = self._parse_document_type_response_yaml(response)

        if self.debug:
            print(f"   Raw response: '{response.strip()}'")
            print(f"   Parsed type: '{doc_type}'")

        return doc_type

    def _parse_document_type_response_yaml(self, response: str) -> str:
        """Parse document type response using YAML-configured type mappings."""
        if not response:
            return self.detection_config["detection_config"].get(
                "fallback_type", "invoice"
            )

        response_lower = response.lower().strip()

        # First try to extract any document type mentioned in response
        raw_type = None
        supported_types = self.detection_config.get("supported_types", [])

        for doc_type in supported_types:
            if doc_type in response_lower:
                raw_type = doc_type
                break

        # If no direct match, look in type mappings
        if not raw_type:
            type_mappings = self.detection_config.get("type_mappings", {})
            for variant, canonical in type_mappings.items():
                if variant.lower() in response_lower:
                    raw_type = canonical
                    break

        # Final fallback
        if not raw_type:
            raw_type = self.detection_config["detection_config"].get(
                "fallback_type", "invoice"
            )

        return raw_type

    def process_document_aware(
        self, image_path: str, classification_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single document with type-aware extraction reusing loaded model."""

        start_time = time.perf_counter()

        # Extract using document-specific fields
        field_names = classification_info["field_names"]
        doc_type = classification_info["document_type"]

        if self.debug:
            print(f"🔍 Extracting {len(field_names)} {doc_type} fields...")
            print(
                f"   Target fields: {field_names[:5]}{'...' if len(field_names) > 5 else ''}"
            )
            print(
                f"   🔍 DEBUG: process_document_aware received {len(field_names)} fields for {doc_type}"
            )
            print(
                f"   🔍 DEBUG: Classification info field_count: {classification_info.get('field_count', 'MISSING')}"
            )
            print(f"   🔍 DEBUG: First 10 field names: {field_names[:10]}")

        # Create document-specific processor, skip model loading if we can reuse
        skip_loading = self.base_processor and hasattr(self.base_processor, "model")

        if self.debug:
            print(
                f"   🔍 DEBUG: Creating DocumentAwareInternVL3Processor with {len(field_names)} fields"
            )
            print(
                f"   🔍 DEBUG: Field list being passed: {field_names[:5]}{'...' if len(field_names) > 5 else ''}"
            )

        document_processor = DocumentAwareInternVL3Processor(
            field_list=field_names,
            model_path=self.model_path,
            debug=self.debug,
            skip_model_loading=skip_loading,
        )

        # CRITICAL OPTIMIZATION: Reuse the already-loaded model from base processor
        if skip_loading:
            if self.debug:
                print("   🔄 Reusing already loaded model (avoiding redundant load)")

            document_processor.model = self.base_processor.model
            document_processor.tokenizer = self.base_processor.tokenizer

        if self.debug:
            print(
                f"   🎯 Processor ready for {len(field_names)} {doc_type}-specific fields"
            )

        # Extract with document-specific processor
        extraction_result = document_processor.process_single_image(image_path)

        if self.debug:
            extracted_data = extraction_result.get("extracted_data", {})
            found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
            print(
                f"   ✅ Found {len(found_fields)} fields: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}"
            )

        # Extract the data from the processor result
        extracted_data = extraction_result.get("extracted_data", {})

        processing_time = time.perf_counter() - start_time

        return {
            "image_file": Path(image_path).name,
            "document_type": doc_type,
            "detected_fields": len(
                [v for v in extracted_data.values() if v != "NOT_FOUND"]
            ),
            "total_fields": len(field_names),
            "processing_time": processing_time,
            "extracted_data": extracted_data,
            "raw_response": extraction_result.get("raw_response", ""),
        }

    def process_universal_single_pass(self, image_path: str) -> Dict[str, Any]:
        """Process single document with universal single-pass extraction."""
        
        print("🔥 DEBUG_MARKER_200: process_universal_single_pass() ENTRY POINT")
        
        start_time = time.perf_counter()

        if self.debug:
            print("🚀 Starting universal single-pass extraction (eliminates double tiling)")
            
        # Create universal processor (no document-specific field list needed)
        from models.document_aware_internvl3_processor import (
            DocumentAwareInternVL3Processor,
        )
        
        # Use explicit universal field list for proper token configuration
        universal_fields = UNIVERSAL_FIELDS  # All 15 universal fields for single-pass extraction
        
        # Create processor, skip model loading if we can reuse  
        skip_loading = self.base_processor and hasattr(self.base_processor, "model")
        
        document_processor = DocumentAwareInternVL3Processor(
            field_list=universal_fields,
            model_path=self.model_path,  # Uses correct path from args
            debug=self.debug,
            skip_model_loading=skip_loading,
        )

        # Copy model references if reusing to avoid reloading
        if skip_loading:
            document_processor.model = self.base_processor.model
            document_processor.tokenizer = self.base_processor.tokenizer
            document_processor.generation_config = self.base_processor.generation_config
            if self.debug:
                print("   ♻️ Reusing loaded model (skip model loading)")

        if self.debug:
            print(
                f"   🎯 Processor ready for universal extraction ({len(universal_fields)} explicit fields)"
            )

        print("🔥 DEBUG_MARKER_201: About to call document_processor.process_single_image()")
        
        # Execute universal single-pass extraction
        extraction_result = document_processor.process_single_image(image_path)
        
        print("🔥 DEBUG_MARKER_202: document_processor.process_single_image() completed")

        processing_time = time.perf_counter() - start_time

        # Format result for compatibility with existing evaluation pipeline
        extracted_data = extraction_result.get("extracted_data", {})
        metadata = extraction_result.get("metadata", {})
        
        detected_fields = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")
        
        result = {
            "image_path": str(image_path),
            "document_type": metadata.get("document_type", "unknown"),
            "extraction_strategy": metadata.get("extraction_strategy", "universal_single_pass"),
            "total_fields": len(universal_fields),  # Explicit universal field count
            "detected_fields": detected_fields,
            "extraction_results": extracted_data,
            "metadata": metadata,
            "processing_time": processing_time,
        }

        if self.debug:
            print(f"   ✅ Universal extraction complete: {detected_fields}/{len(universal_fields)} fields")
            print(f"   📋 Inferred type: {metadata.get('document_type', 'unknown')}")
            print(f"   ⏱️ Processing time: {processing_time:.2f}s")

        # Store processor reference for model reuse
        self.base_processor = document_processor
        
        return result

    def evaluate_document_aware(
        self, results: List[Dict[str, Any]], ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate document-aware extraction results."""

        if self.debug:
            print("🔬 Evaluating document-aware extraction results...")

        # Group results by document type for evaluation
        by_document_type = {}
        evaluation_results = []

        for result in results:
            doc_type = result["document_type"]
            if doc_type not in by_document_type:
                by_document_type[doc_type] = []
            by_document_type[doc_type].append(result)

        # Evaluate each document type separately
        for doc_type, type_results in by_document_type.items():
            if self.debug:
                print(f"📊 Evaluating {len(type_results)} {doc_type} documents...")

            for result in type_results:
                image_name = result["image_file"]
                extracted_data = result["extracted_data"]

                # Get ground truth for this image
                image_truth = ground_truth.get(image_name, {})
                if not image_truth:
                    if self.debug:
                        print(f"   ⚠️ No ground truth found for {image_name}")
                    continue

                # Evaluate using document-type-specific metrics
                evaluation = self.evaluator.evaluate_extraction(
                    extracted_data, image_truth, doc_type
                )

                # Add image metadata
                evaluation["image_file"] = image_name
                evaluation["detected_fields"] = result["detected_fields"]
                evaluation["total_fields"] = result["total_fields"]
                evaluation["processing_time"] = result["processing_time"]

                evaluation_results.append(evaluation)
                
                # Display detailed comparison for single image processing
                if len(results) == 1:
                    self._display_detailed_comparison(result, image_truth, evaluation)

        # Calculate aggregate metrics
        if not evaluation_results:
            return {"error": "No evaluation results generated"}

        # Overall metrics across all document types
        overall_accuracies = [
            r["overall_metrics"]["overall_accuracy"] for r in evaluation_results
        ]
        processing_times = [r["processing_time"] for r in evaluation_results]

        # Document type breakdown
        type_breakdown = {}
        for result in evaluation_results:
            doc_type = result["document_type"]
            if doc_type not in type_breakdown:
                type_breakdown[doc_type] = {
                    "count": 0,
                    "accuracies": [],
                    "ato_compliant": 0 if doc_type == "invoice" else None,
                }

            type_breakdown[doc_type]["count"] += 1
            type_breakdown[doc_type]["accuracies"].append(
                result["overall_metrics"]["overall_accuracy"]
            )

            # Track ATO compliance for invoices
            if doc_type == "invoice" and result["overall_metrics"].get("ato_compliant"):
                type_breakdown[doc_type]["ato_compliant"] += 1

        # Generate summary report
        summary_report = {
            "model": "InternVL3",
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_documents": len(evaluation_results),
            "document_type_breakdown": {},
            "overall_metrics": {
                "average_accuracy": sum(overall_accuracies) / len(overall_accuracies)
                if overall_accuracies
                else 0,
                "average_processing_time": sum(processing_times) / len(processing_times)
                if processing_times
                else 0,
                "documents_above_80_percent": sum(
                    1 for acc in overall_accuracies if acc >= 0.8
                ),
                "documents_above_90_percent": sum(
                    1 for acc in overall_accuracies if acc >= 0.9
                ),
            },
        }

        # Add per-document-type metrics
        for doc_type, stats in type_breakdown.items():
            avg_accuracy = sum(stats["accuracies"]) / len(stats["accuracies"])
            summary_report["document_type_breakdown"][doc_type] = {
                "documents": stats["count"],
                "average_accuracy": avg_accuracy,
                "accuracy_percentage": f"{avg_accuracy * 100:.1f}%",
            }

            # Add ATO compliance for invoices
            if doc_type == "invoice" and stats["ato_compliant"] is not None:
                compliance_rate = (
                    stats["ato_compliant"] / stats["count"] if stats["count"] > 0 else 0
                )
                summary_report["document_type_breakdown"][doc_type][
                    "ato_compliance"
                ] = {
                    "compliant_documents": stats["ato_compliant"],
                    "compliance_rate": f"{compliance_rate * 100:.0f}%",
                }

        return {"summary": summary_report, "detailed_results": evaluation_results}
    
    def _display_detailed_comparison(self, result: Dict, ground_truth: Dict, evaluation: Dict):
        """Display detailed field-by-field comparison like in the notebook."""
        
        print("\\n" + "="*120)
        print("📋 STEP 4: Extracted Data Results with Ground Truth Comparison")  
        print("="*120)
        
        extracted_data = result["extracted_data"]
        processing_time = result["processing_time"]
        
        # Display extracted data first
        print("\\n🔍 EXTRACTED DATA:")
        for field, value in extracted_data.items():
            if value != "NOT_FOUND":
                print(f"✅ {field}: {value}")
            else:
                print(f"❌ {field}: {value}")
        
        # Ground truth comparison table
        print(f"\\n📊 Ground truth loaded for {result['image_file']}")
        print("-"*120)
        
        field_scores = evaluation.get("field_scores", {})
        
        # Table header
        print(f"{'STATUS':<8} {'FIELD':<25} {'EXTRACTED':<40} {'GROUND TRUTH':<40}")
        print("="*120)
        
        # Field-by-field comparison
        fields_found = 0
        exact_matches = 0
        total_fields = len(field_scores)
        
        for field, score in field_scores.items():
            extracted_val = extracted_data.get(field, "NOT_FOUND")
            ground_val = ground_truth.get(field, "NOT_FOUND")
            
            # Determine status symbol
            if score.get("accuracy", 0) == 1.0:
                status = "✅"
                exact_matches += 1
            elif score.get("accuracy", 0) >= 0.8:
                status = "≈"
            else:
                status = "❌"
            
            if extracted_val != "NOT_FOUND":
                fields_found += 1
                
            # Truncate long values for display
            extracted_display = str(extracted_val)[:38] + ("..." if len(str(extracted_val)) > 38 else "")
            ground_display = str(ground_val)[:38] + ("..." if len(str(ground_val)) > 38 else "")
            
            print(f"{status:<8} {field:<25} {extracted_display:<40} {ground_display:<40}")
        
        # Summary section
        overall_accuracy = evaluation["overall_metrics"]["overall_accuracy"]
        
        print("\\n📊 EXTRACTION SUMMARY:")
        print(f"✅ Fields Found: {fields_found}/{total_fields} ({fields_found/total_fields*100:.1f}%)")
        print(f"🎯 Exact Matches: {exact_matches}/{total_fields} ({exact_matches/total_fields*100:.1f}%)")  
        print(f"📈 Extraction Success Rate: {overall_accuracy*100:.1f}%")
        print(f"⏱️ Accuracy (matches/total): {overall_accuracy*100:.1f}%")
        print(f"⏰ Processing Time: {processing_time:.3f}s")
        print(f"🤖 Document Type: {result['document_type']}")
        print("🔧 Model: InternVL3-8B")
        
        # Additional metrics
        meets_threshold = evaluation["overall_metrics"].get("meets_threshold", False)
        threshold = evaluation["overall_metrics"].get("document_type_threshold", 0.8)
        print("\\n≈ = Partial match")  
        print("✗ = No match")
        print(f"Note: Meets accuracy threshold ({threshold*100:.0f}%): {'✅ Yes' if meets_threshold else '❌ No'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="InternVL3 Document-Aware Extraction Pipeline"
    )
    parser.add_argument(
        "--model-path",
        default="/efs/shared/PTM/InternVL3-8B",
        help="Path to InternVL3 model",
    )
    parser.add_argument(
        "--data-dir", default="evaluation_data", help="Directory with images"
    )
    parser.add_argument("--image-path", help="Path to single image file for testing")
    parser.add_argument(
        "--ground-truth",
        default="evaluation_data/ground_truth.csv",
        help="Ground truth CSV file",
    )
    parser.add_argument(
        "--limit-images", type=int, help="Limit number of images to process"
    )
    parser.add_argument(
        "--document-type",
        choices=["invoice", "receipt", "bank_statement"],
        help="Process only specific document type",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Enable debug OCR mode for raw markdown output (requires --debug)",
    )

    args = parser.parse_args()

    print("🚀 InternVL3 Vision Document-Aware Extraction Pipeline")
    print("=" * 80)

    # Determine if single image or batch processing
    if args.image_path:
        print(f"🖼️  Single image: {args.image_path}")

        # Verify single image exists
        if not Path(args.image_path).exists():
            print(f"❌ ERROR: Image file not found: {args.image_path}")
            return

        print(f"🔧 Model: {args.model_path or 'Default InternVL3 path'}")
        print(f"🐛 Debug mode: {args.debug}")
        
        # Handle debug OCR mode
        if args.debug_ocr:
            if not args.debug:
                print("❌ ERROR: --debug-ocr requires --debug flag")
                print(
                    "💡 Usage: python internvl3_document_aware.py --image-path IMAGE --debug --debug-ocr"
                )
                return
            
            print("\n🔍 DEBUG OCR MODE ENABLED")
            print(
                "🎯 Processing will output raw text extraction for OCR debugging"
            )
            
            # Create a simple OCR debugging handler
            print(f"\n📄 Processing {Path(args.image_path).name} in debug OCR mode...")
            
            # Initialize processor for OCR mode
            processor = DocumentAwareInternVL3Handler(
                model_path=args.model_path, debug=True
            )
            
            # Use a simple OCR prompt to get raw text
            from models.document_aware_internvl3_processor import (
                DocumentAwareInternVL3Processor,
            )
            
            # Create minimal processor for OCR
            ocr_fields = ["DOCUMENT_TYPE"]  # Minimal field to trigger model loading
            ocr_processor = DocumentAwareInternVL3Processor(
                field_list=ocr_fields,
                model_path=args.model_path,
                debug=True
            )
            
            # Simple OCR prompt
            ocr_prompt = """Read and transcribe ALL text visible in this document image.

Output the complete text content exactly as shown, preserving layout where possible.
Include all headers, labels, values, numbers, and any other visible text.

COMPLETE TEXT TRANSCRIPTION:"""
            
            print("\n📝 Debug OCR Prompt:")
            print("-" * 60)
            print(ocr_prompt)
            print("-" * 60)
            
            try:
                # Get raw OCR output
                raw_text = ocr_processor._extract_with_prompt(args.image_path, ocr_prompt)
                
                print("\n📄 RAW OCR OUTPUT:")
                print("=" * 80)
                print(raw_text)
                print("=" * 80)
                
                print("\n✅ Debug OCR complete")
                print("💡 This shows what text the model can read from the image")
                print("💡 Compare with expected field values to diagnose extraction issues")
                
            except Exception as e:
                print(f"❌ Error in debug OCR mode: {e}")
                import traceback
                if args.debug:
                    traceback.print_exc()
            
            return

        # Single image mode (normal processing)
        processor = DocumentAwareInternVL3Handler(
            model_path=args.model_path, debug=args.debug
        )

        print(f"\\n📄 Processing single image: {Path(args.image_path).name}...")

        try:
            # SINGLE-PASS: Universal extraction with post-processing type inference  
            result = processor.process_universal_single_pass(args.image_path)

            # Display results
            print("\\n📋 RESULTS:")
            print(f"   Document Type: {result['document_type']}")
            print(
                f"   Fields Found: {result['detected_fields']}/{result['total_fields']}"
            )
            print(f"   Processing Time: {result['processing_time']:.3f}s")

            print("\\n📊 EXTRACTED DATA:")
            extracted_data = result["extracted_data"]
            for field_name, value in extracted_data.items():
                status = "✅" if value != "NOT_FOUND" else "❌"
                print(f"   {status} {field_name}: {value}")

            # If ground truth available, evaluate single result
            if Path(args.ground_truth).exists():
                ground_truth = load_ground_truth(args.ground_truth)
                image_name = Path(args.image_path).name

                if image_name in ground_truth:
                    evaluation_report = processor.evaluate_document_aware(
                        [result], ground_truth
                    )
                    summary = evaluation_report["summary"]
                    print("\\n📈 EVALUATION vs Ground Truth:")
                    print(
                        f"   Accuracy: {summary['overall_metrics']['average_accuracy'] * 100:.1f}%"
                    )

                    # Document type specific metrics
                    doc_type = result["document_type"]
                    type_stats = summary["document_type_breakdown"].get(doc_type, {})
                    if "ato_compliance" in type_stats:
                        print(
                            f"   ATO Compliant: {type_stats['ato_compliance']['compliance_rate']}"
                        )
                else:
                    print(f"\\n⚠️  No ground truth available for {image_name}")

            print("\\n✅ Single image processing complete!")
            return

        except Exception as e:
            print(f"❌ Error processing {args.image_path}: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return

    else:
        # Batch processing mode
        print(f"📂 Data directory: {args.data_dir}")
        print(f"📋 Ground truth: {args.ground_truth}")
        if args.limit_images:
            print(f"🔢 Limiting to: {args.limit_images} images")
        if args.document_type:
            print(f"📄 Document type filter: {args.document_type}")
        print()

        # Initialize processor
        processor = DocumentAwareInternVL3Handler(
            model_path=args.model_path, debug=args.debug
        )

        # Load ground truth
        print("📚 Loading ground truth data...")
        ground_truth = load_ground_truth(args.ground_truth)
        print(f"✅ Loaded ground truth for {len(ground_truth)} documents")

        # Discover images
        print("🔍 Discovering document images...")
        image_files = discover_images(args.data_dir)

        if not image_files:
            print("❌ No image files found!")
            return

        print(f"📁 Found {len(image_files)} document images")

        # Process documents with document-aware pipeline
        print("\n🔬 PHASE 4: DOCUMENT-AWARE EXTRACTION")
        print("=" * 80)

        results = []
        start_time = time.perf_counter()
        processed_count = 0

        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {Path(image_path).name}")

            try:
                # SINGLE-PASS: Universal extraction with post-processing type inference
                result = processor.process_universal_single_pass(image_path)
                
                # Extract results for display
                detected = result["detected_fields"]  
                total = result["total_fields"]
                time_taken = result["processing_time"]
                inferred_type = result.get("document_type", "unknown")
                
                print(
                    f"  📋 Inferred Type: {inferred_type} (universal extraction: {detected}/{total} fields)"
                )

                # Optional: Filter by document type (using inferred type)
                if args.document_type and inferred_type != args.document_type:
                    print(f"  ⏭️ Skipping - looking for {args.document_type} documents")
                    continue

                results.append(result)
                processed_count += 1

                print(
                    f"  ✅ {inferred_type}: {detected}/{total} fields (single-pass)"
                )
                print(f"  ⏱️ Processing time: {time_taken:.2f}s")

                # Check if we've processed enough documents of the target type
                if args.limit_images and processed_count >= args.limit_images:
                    print(
                        f"\n🎯 Reached limit: processed {processed_count} {args.document_type or 'document'}(s)"
                    )
                    break

            except Exception as e:
                print(f"  ❌ Error processing {Path(image_path).name}: {e}")
                if args.debug:
                    import traceback

                    traceback.print_exc()
                continue

        total_time = time.perf_counter() - start_time

        # Results summary
        print("\n📊 PROCESSING COMPLETE")
        print("=" * 80)
        print(f"✅ Processed: {len(results)} documents")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(
            f"🏎️ Average time: {total_time / len(results):.2f}s per document"
            if results
            else "No results"
        )

        # Evaluate results
        if results:
            evaluation_report = processor.evaluate_document_aware(results, ground_truth)

            # Display summary
            summary = evaluation_report["summary"]
            print("\n📈 DOCUMENT-AWARE EVALUATION RESULTS")
            print("=" * 80)
            print(
                f"📊 Overall Accuracy: {summary['overall_metrics']['average_accuracy'] * 100:.1f}%"
            )
            print(
                f"⚡ Average Processing Time: {summary['overall_metrics']['average_processing_time']:.2f}s"
            )
            print(
                f"🎯 Documents >80%: {summary['overall_metrics']['documents_above_80_percent']}/{summary['total_documents']}"
            )
            print(
                f"🏆 Documents >90%: {summary['overall_metrics']['documents_above_90_percent']}/{summary['total_documents']}"
            )

            print("\n📄 BY DOCUMENT TYPE:")
            print("-" * 40)
            for doc_type, stats in summary["document_type_breakdown"].items():
                print(
                    f"  {doc_type.upper()}: {stats['accuracy_percentage']} ({stats['documents']} docs)"
                )
                if "ato_compliance" in stats:
                    print(
                        f"    ATO Compliance: {stats['ato_compliance']['compliance_rate']}"
                    )

            print("\n💡 INSIGHTS:")
            if summary["overall_metrics"]["documents_above_90_percent"] > 0:
                print(
                    f"  • {summary['overall_metrics']['documents_above_90_percent']} documents achieved >90% accuracy"
                )

            # Save detailed results
            from common.config import OUTPUT_DIR

            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = (
                output_dir / f"internvl3_document_aware_report_{timestamp}.json"
            )

            with report_path.open("w") as f:
                json.dump(evaluation_report, f, indent=2, default=str)

            # Generate CSV exports (ported from keyvalue functionality)
            print("\n📊 Creating extraction DataFrames...")
            try:
                main_df, metadata_df = create_extraction_dataframe(results)

                # Save main extraction results
                extraction_csv = (
                    output_dir / f"internvl3_document_aware_extraction_{timestamp}.csv"
                )
                main_df.to_csv(extraction_csv, index=False)
                print(f"💾 Extraction results saved: {extraction_csv}")

                # Save processing metadata
                if not metadata_df.empty:
                    metadata_csv = (
                        output_dir
                        / f"internvl3_document_aware_metadata_{timestamp}.csv"
                    )
                    metadata_df.to_csv(metadata_csv, index=False)
                    print(f"💾 Extraction metadata saved: {metadata_csv}")

            except Exception as e:
                print(f"⚠️  Error creating CSV exports: {e}")
                if args.debug:
                    import traceback

                    traceback.print_exc()

            print(f"\n✅ Detailed report saved to: {report_path}")
            print("\n" + "=" * 80)
            print("✅ PHASE 4 DOCUMENT-AWARE EXTRACTION COMPLETE")
            print("=" * 80)
        else:
            print("❌ No results to evaluate")


if __name__ == "__main__":
    main()
