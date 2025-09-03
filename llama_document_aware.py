#!/usr/bin/env python3
"""
Llama Vision Document-Aware Key-Value Extraction - V4 Schema Implementation

This module implements the V4 document-aware extraction pipeline for the
Llama-3.2-11B-Vision-Instruct model, featuring:
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
    python llama_document_aware.py --limit-images 5 --debug
    python llama_document_aware.py --document-type invoice
    python llama_document_aware.py --image-path "path/to/single_image.jpg" --debug
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
from models.document_aware_llama_processor import DocumentAwareLlamaProcessor


class DocumentAwareLlamaHandler:
    """V4 Document-Aware Llama Vision Processor with comprehensive field coverage."""

    def __init__(self, model_path: str, debug: bool = False):
        """Initialize document-aware processor."""
        self.debug = debug
        self.model_path = model_path

        print(
            "🚀 Initializing Llama Vision processor for V4 document-aware extraction..."
        )

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

        # Create a single processor that will be reused for all operations
        # Initially with just DOCUMENT_TYPE for detection, then reconfigured for extraction
        self.processor = None

        print(
            "✅ Document-aware Llama handler initialized (model will load on first use)"
        )

    def _detect_document_type_yaml(self, image_path: str) -> str:
        """
        YAML-first document type detection using configurable prompts.

        Uses prompts/document_type_detection.yaml for maintainable prompt management.
        """
        if self.debug:
            print("📝 Using YAML-first document detection approach")

        # Get Llama-specific prompt from YAML configuration
        llama_config = self.detection_config["detection_prompts"]["llama"]
        doc_type_prompt = llama_config["user_prompt"]
        max_tokens = llama_config.get("max_tokens", 50)

        if self.debug:
            print(
                f"   YAML config version: {self.detection_config.get('version', 'unknown')}"
            )
            print(f"   Max tokens: {max_tokens}")
            print(f"   Prompt: {doc_type_prompt[:100]}...")

        # Ensure processor exists for simple detection
        if not self.processor:
            self.processor = DocumentAwareLlamaProcessor(
                field_list=["DOCUMENT_TYPE"],  # Single field for detection
                model_path=self.model_path,
                debug=self.debug,
            )

        # Use the processor to extract with YAML prompt
        response = self.processor._extract_with_custom_prompt(
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

    def detect_and_classify_document(self, image_path: str) -> Dict[str, Any]:
        """
        Detect document type using YAML-first approach and get appropriate schema.

        Args:
            image_path: Path to document image

        Returns:
            Dict with document type, schema, and field information
        """

        if self.debug:
            print(f"📋 Detecting document type for: {image_path}")
            print("   Using YAML-first detection approach")

        try:
            # Use YAML-first document detection
            doc_type = self._detect_document_type_yaml(image_path)

            if self.debug:
                print(f"   📄 Detected document type: {doc_type}")

        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Detection failed: {e}, using fallback")
            doc_type = self.detection_config["detection_config"].get(
                "fallback_type", "invoice"
            )

        # Get schema for detected document type
        schema = self.schema_loader.get_document_schema(doc_type)

        if self.debug:
            print(f"   Schema Fields: {len(schema['fields'])} fields")
            print(f"   Extraction Mode: {schema['extraction_mode']}")

        return {
            "document_type": doc_type,
            "schema": schema,
            "field_count": len(schema["fields"]),
            "field_names": schema["fields"]
            if isinstance(schema["fields"][0], str)
            else [f["name"] for f in schema["fields"]],
        }

    def process_document_aware(
        self, image_path: str, classification_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single document with type-aware extraction."""

        start_time = time.perf_counter()

        # Extract using document-specific fields
        field_names = classification_info["field_names"]
        doc_type = classification_info["document_type"]

        if self.debug:
            print(f"🔍 Extracting {len(field_names)} {doc_type} fields...")
            print(
                f"   Target fields: {field_names[:5]}{'...' if len(field_names) > 5 else ''}"
            )

        # Reconfigure existing processor with new field list
        # This is more efficient than creating a new processor
        if self.processor:
            # Simply update the field list on the existing processor
            self.processor.field_list = field_names
            self.processor.field_count = len(field_names)
            # Reconfigure generation parameters for new field count
            self.processor._configure_generation()

            if self.debug:
                print(
                    f"   🔄 Reconfigured processor for {len(field_names)} {doc_type}-specific fields"
                )
        else:
            # Create processor if it doesn't exist yet
            self.processor = DocumentAwareLlamaProcessor(
                field_list=field_names, model_path=self.model_path, debug=self.debug
            )

        # Extract with reconfigured processor
        extraction_result = self.processor.process_single_image(image_path)

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
        }

    def evaluate_document_aware(
        self, results: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate results with document-type-specific metrics."""

        print("\\n📊 Evaluating with document-type-specific metrics...")

        evaluations = []

        for result in results:
            image_file = result["image_file"]
            gt_data = ground_truth.get(image_file, {})

            if not gt_data:
                print(f"⚠️  No ground truth for {image_file}")
                continue

            # Use document-type-specific evaluator
            evaluation = self.evaluator.evaluate_extraction(
                result["extracted_data"], gt_data, result["document_type"]
            )

            evaluation["image_file"] = image_file
            evaluation["processing_time"] = result["processing_time"]
            evaluations.append(evaluation)
            
            # Display detailed comparison for single image processing
            if len(results) == 1:
                self._display_detailed_comparison(result, gt_data, evaluation)

        return self._generate_document_aware_report(evaluations)
    
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
        print("🔧 Model: Llama-3.2-11B-Vision-Instruct")
        
        # Additional metrics
        meets_threshold = evaluation["overall_metrics"].get("meets_threshold", False)
        threshold = evaluation["overall_metrics"].get("document_type_threshold", 0.8)
        print("\\n≈ = Partial match")  
        print("✗ = No match")
        print(f"Note: Meets accuracy threshold ({threshold*100:.0f}%): {'✅ Yes' if meets_threshold else '❌ No'}")

    def _generate_document_aware_report(
        self, evaluations: List[Dict]
    ) -> Dict[str, Any]:
        """Generate comprehensive document-aware evaluation report."""

        # Group by document type
        by_type = {}
        for eval_result in evaluations:
            doc_type = eval_result["document_type"]
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(eval_result)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(evaluations),
            "evaluation_mode": "document_aware_phase4",
            "by_document_type": {},
        }

        print("\\n" + "=" * 80)
        print("📊 DOCUMENT-AWARE EVALUATION RESULTS")
        print("=" * 80)

        for doc_type, type_evaluations in by_type.items():
            # Calculate type-specific metrics
            accuracies = [
                e["overall_metrics"]["overall_accuracy"] for e in type_evaluations
            ]
            meets_threshold = sum(
                1
                for e in type_evaluations
                if e["overall_metrics"].get("meets_threshold", False)
            )
            critical_perfect = sum(
                1
                for e in type_evaluations
                if e["overall_metrics"].get("critical_fields_perfect", False)
            )

            # ATO compliance for invoices
            ato_compliant = 0
            if doc_type == "invoice":
                ato_compliant = sum(
                    1
                    for e in type_evaluations
                    if e["overall_metrics"].get("ato_compliant", False)
                )

            type_metrics = {
                "count": len(type_evaluations),
                "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "meets_threshold": meets_threshold,
                "critical_perfect": critical_perfect,
                "ato_compliant": ato_compliant if doc_type == "invoice" else None,
            }

            report["by_document_type"][doc_type] = type_metrics

            # Print results
            print(f"\\n📋 {doc_type.upper()}:")
            print(f"   Documents: {type_metrics['count']}")
            print(f"   Avg Accuracy: {type_metrics['avg_accuracy'] * 100:.1f}%")
            print(f"   Meeting Threshold: {meets_threshold}/{len(type_evaluations)}")
            print(
                f"   Critical Fields Perfect: {critical_perfect}/{len(type_evaluations)}"
            )

            if doc_type == "invoice" and ato_compliant is not None:
                print(
                    f"   ATO Compliant: {ato_compliant}/{len(type_evaluations)} ({ato_compliant / len(type_evaluations) * 100:.0f}%)"
                )

        return report


def main():
    """Run document-aware Llama evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Document-Aware Llama Vision Extraction - Phase 4"
    )
    parser.add_argument(
        "--model-path",
        default="/efs/shared/PTM/Llama-3.2-11B-Vision-Instruct",
        help="Path to Llama model",
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

    print("\\n" + "=" * 80)
    print("🚀 LLAMA DOCUMENT-AWARE EXTRACTION - PHASE 4")
    print("=" * 80)

    # Determine if single image or batch processing
    if args.image_path:
        print(f"🖼️  Single image: {args.image_path}")

        # Verify single image exists
        if not Path(args.image_path).exists():
            print(f"❌ ERROR: Image file not found: {args.image_path}")
            return

        print(f"🔧 Model: {args.model_path}")
        print(f"🐛 Debug mode: {args.debug}")

        # Handle debug OCR mode
        if args.debug_ocr:
            if not args.debug:
                print("❌ ERROR: --debug-ocr requires --debug flag")
                print(
                    "💡 Usage: python llama_document_aware.py --image-path IMAGE --debug --debug-ocr"
                )
                return

            print("\n🔍 DEBUG OCR MODE ENABLED")
            print(
                "🎯 Processing will output raw markdown OCR instead of structured extraction"
            )
            print("💡 This helps diagnose OCR vs document understanding issues")

            # Simple OCR processing using basic model interaction
            print(f"\n📄 Processing {Path(args.image_path).name} in debug OCR mode...")

            # TODO: Add simple OCR debug functionality here
            print(
                "⚠️  Debug OCR mode is not yet implemented in document-aware processor"
            )
            print("💡 Use llama_keyvalue.py --debug-ocr for full OCR debugging")
            return

        # Single image mode
        processor = DocumentAwareLlamaHandler(args.model_path, debug=args.debug)

        print(f"\\n📄 Processing single image: {Path(args.image_path).name}...")

        try:
            # Step 1: Detect document type and get schema (YAML-first approach)
            classification_info = processor.detect_and_classify_document(
                args.image_path
            )

            # Step 2: Extract with document-specific schema
            result = processor.process_document_aware(
                args.image_path, classification_info
            )

            # Display results
            print("\\n📋 RESULTS:")
            print(f"   Document Type: {result['document_type']}")
            print(
                f"   Fields Found: {result['detected_fields']}/{result['total_fields']}"
            )
            print(f"   Processing Time: {result['processing_time']:.3f}s")

            efficiency = (
                (25 - result["total_fields"]) / 25 * 100
            )  # vs unified 25 fields
            print(
                f"   Field Reduction: {efficiency:.0f}% fewer fields than unified approach"
            )

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
                    print("\\n📈 EVALUATION vs Ground Truth:")

                    # Find this document's evaluation
                    doc_type = result["document_type"]
                    type_metrics = evaluation_report["by_document_type"].get(
                        doc_type, {}
                    )
                    print(
                        f"   Accuracy: {type_metrics.get('avg_accuracy', 0) * 100:.1f}%"
                    )
                    print(
                        f"   Meets Threshold: {'Yes' if type_metrics.get('meets_threshold', 0) > 0 else 'No'}"
                    )
                    if doc_type == "invoice":
                        print(
                            f"   ATO Compliant: {'Yes' if type_metrics.get('ato_compliant', 0) > 0 else 'No'}"
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
        print(f"📁 Data directory: {args.data_dir}")
        print(f"📊 Ground truth: {args.ground_truth}")
        print(f"🔧 Model: {args.model_path}")
        print(f"🎯 Document type filter: {args.document_type or 'All types'}")
        print(f"📸 Image limit: {args.limit_images or 'No limit'}")
        print(f"🐛 Debug mode: {args.debug}")

        # Verify files exist for batch processing
        if not Path(args.ground_truth).exists():
            print(f"❌ ERROR: Ground truth file not found: {args.ground_truth}")
            return

        if not Path(args.data_dir).exists():
            print(f"❌ ERROR: Data directory not found: {args.data_dir}")
            return

        # Initialize document-aware processor for batch processing
        processor = DocumentAwareLlamaHandler(args.model_path, debug=args.debug)

        # Load ground truth and discover images
        ground_truth = load_ground_truth(args.ground_truth)
        all_images = discover_images(args.data_dir)

        # Filter images if document type specified
        if args.document_type:
            filtered_images = []
            for img in all_images:
                img_name = Path(img).name
                if img_name in ground_truth:
                    gt_doc_type = (
                        ground_truth[img_name].get("DOCUMENT_TYPE", "").lower()
                    )
                    if args.document_type.lower() in gt_doc_type:
                        filtered_images.append(img)
            all_images = filtered_images
            print(f"🔍 Filtered to {len(all_images)} {args.document_type} documents")

        # Limit images if specified
        if args.limit_images:
            all_images = all_images[: args.limit_images]
            print(f"📸 Processing {len(all_images)} images")

        # Process each image with document-aware extraction
        results = []

        print(f"\\n🔄 Processing {len(all_images)} documents...")

        for i, image_path in enumerate(all_images, 1):
            print(
                f"\\n📄 [{i}/{len(all_images)}] Processing {Path(image_path).name}..."
            )

            try:
                # Step 1: Detect document type and get schema (YAML-first approach)
                classification_info = processor.detect_and_classify_document(image_path)

                # Step 2: Extract with document-specific schema
                result = processor.process_document_aware(
                    image_path, classification_info
                )
                results.append(result)

                # Show progress
                print(
                    f"   ✅ {result['document_type']}: {result['detected_fields']}/{result['total_fields']} fields"
                )
                print(f"   ⏱️  Processing time: {result['processing_time']:.3f}s")

            except Exception as e:
                print(f"   ❌ Error processing {image_path}: {e}")
                if args.debug:
                    import traceback

                    traceback.print_exc()

        # Evaluate results
        if results:
            evaluation_report = processor.evaluate_document_aware(results, ground_truth)

            # Save detailed results
            from common.config import OUTPUT_DIR

            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = output_dir / f"llama_document_aware_report_{timestamp}.json"

            with Path(report_path).open("w") as f:
                json.dump(evaluation_report, f, indent=2, default=str)

            # Generate CSV exports (ported from keyvalue functionality)
            print("\\n📊 Creating extraction DataFrames...")
            try:
                main_df, metadata_df = create_extraction_dataframe(results)

                # Save main extraction results
                extraction_csv = (
                    output_dir / f"llama_document_aware_extraction_{timestamp}.csv"
                )
                main_df.to_csv(extraction_csv, index=False)
                print(f"💾 Extraction results saved: {extraction_csv}")

                # Save processing metadata
                if not metadata_df.empty:
                    metadata_csv = (
                        output_dir / f"llama_document_aware_metadata_{timestamp}.csv"
                    )
                    metadata_df.to_csv(metadata_csv, index=False)
                    print(f"💾 Extraction metadata saved: {metadata_csv}")

            except Exception as e:
                print(f"⚠️  Error creating CSV exports: {e}")
                if args.debug:
                    import traceback

                    traceback.print_exc()

            print(f"\\n✅ Detailed report saved to: {report_path}")
            print("\\n" + "=" * 80)
            print("✅ PHASE 4 DOCUMENT-AWARE EXTRACTION COMPLETE")
            print("=" * 80)
        else:
            print("❌ No results to evaluate")


if __name__ == "__main__":
    main()
