#!/usr/bin/env python3
"""
InternVL3 Vision Document-Aware Key-Value Extraction - Phase 4 Implementation

This module implements the Phase 4 document-aware extraction pipeline for the 
InternVL3 model (2B/8B variants), featuring:
- Document type detection and classification
- Type-specific field schema routing 
- Targeted extraction (invoice: 20 fields, receipt: 15 fields, bank_statement: 15 fields)
- ATO compliance validation for invoices
- Performance optimization through reduced field sets

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
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Import Phase 4 document-aware components
from common.document_schema_loader import DocumentTypeFieldSchema
from common.document_type_detector import DocumentTypeDetector
from common.document_type_metrics import DocumentTypeEvaluator
from common.evaluation_metrics import load_ground_truth
from common.extraction_parser import discover_images
from models.document_aware_internvl3_processor import DocumentAwareInternVL3Processor


class DocumentAwareInternVL3Handler:
    """Phase 4 Document-Aware InternVL3 Vision Processor."""
    
    def __init__(self, model_path: str, debug: bool = False):
        """Initialize document-aware processor."""
        self.debug = debug
        self.model_path = model_path
        
        print("🚀 Initializing InternVL3 Vision processor for document-aware extraction...")
        
        # We'll create processors on-demand to avoid loading multiple models
        self.base_processor = None
        self.model_loaded = False
        
        # Initialize Phase 4 components
        self.schema_loader = DocumentTypeFieldSchema()
        self.evaluator = DocumentTypeEvaluator()
        
        # Note: Document detector will be configured when we need it
        self.document_detector = None
        
        print("✅ Document-aware InternVL3 handler initialized (model will load on first use)")
    
    def detect_and_classify_document(self, image_path: str) -> Dict[str, Any]:
        """Detect document type and get appropriate schema."""
        
        if self.debug:
            print(f"📋 Detecting document type for: {image_path}")
        
        # Initialize document detector on first use
        if not self.document_detector:
            # Create a minimal processor just for document type detection
            detection_fields = ["DOCUMENT_TYPE"]  # Only need this for classification
            self.base_processor = DocumentAwareInternVL3Processor(
                field_list=detection_fields,
                model_path=self.model_path,
                debug=self.debug
            )
            self.document_detector = DocumentTypeDetector(model_processor=self.base_processor)
            self.schema_loader.set_document_detector(self.document_detector)
        
        # Detect document type
        doc_type = self.schema_loader.detect_document_type(image_path)
        schema = self.schema_loader.get_document_schema(doc_type)
        
        if self.debug:
            print(f"   Document Type: {doc_type}")
            print(f"   Schema Fields: {len(schema['fields'])} fields")
            print(f"   Extraction Mode: {schema['extraction_mode']}")
        
        return {
            "document_type": doc_type,
            "schema": schema,
            "field_count": len(schema["fields"]),
            "field_names": [f["name"] for f in schema["fields"]]
        }
    
    def process_document_aware(self, image_path: str, 
                              classification_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process single document with type-aware extraction reusing loaded model."""
        
        start_time = time.perf_counter()
        
        # Extract using document-specific fields
        field_names = classification_info["field_names"]
        doc_type = classification_info["document_type"]
        
        if self.debug:
            print(f"🔍 Extracting {len(field_names)} {doc_type} fields...")
            print(f"   Target fields: {field_names[:5]}{'...' if len(field_names) > 5 else ''}")
        
        # Create document-specific processor, skip model loading if we can reuse
        skip_loading = self.base_processor and hasattr(self.base_processor, 'model')
        
        document_processor = DocumentAwareInternVL3Processor(
            field_list=field_names,
            model_path=self.model_path,
            debug=self.debug,
            skip_model_loading=skip_loading
        )
        
        # CRITICAL OPTIMIZATION: Reuse the already-loaded model from base processor
        if skip_loading:
            if self.debug:
                print("   🔄 Reusing already loaded model (avoiding redundant load)")
            
            document_processor.model = self.base_processor.model
            document_processor.tokenizer = self.base_processor.tokenizer
        
        if self.debug:
            print(f"   🎯 Processor ready for {len(field_names)} {doc_type}-specific fields")
        
        # Extract with document-specific processor
        extraction_result = document_processor.process_single_image(image_path)
        
        if self.debug:
            extracted_data = extraction_result.get("extracted_data", {})
            found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
            print(f"   ✅ Found {len(found_fields)} fields: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}")
        
        # Extract the data from the processor result
        extracted_data = extraction_result.get("extracted_data", {})
        
        processing_time = time.perf_counter() - start_time
        
        return {
            "image_file": Path(image_path).name,
            "document_type": doc_type,
            "detected_fields": len([v for v in extracted_data.values() if v != "NOT_FOUND"]),
            "total_fields": len(field_names),
            "field_reduction": f"{((25 - len(field_names)) / 25 * 100):.0f}%",
            "processing_time": processing_time,
            "extracted_data": extracted_data,
            "raw_response": extraction_result.get("raw_response", "")
        }
    
    def evaluate_document_aware(self, results: List[Dict[str, Any]], ground_truth: Dict) -> Dict[str, Any]:
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
                evaluation["field_reduction"] = result["field_reduction"]
                
                evaluation_results.append(evaluation)
        
        # Calculate aggregate metrics
        if not evaluation_results:
            return {"error": "No evaluation results generated"}
        
        # Overall metrics across all document types
        overall_accuracies = [r["overall_metrics"]["overall_accuracy"] for r in evaluation_results]
        processing_times = [r["processing_time"] for r in evaluation_results]
        field_reductions = [float(r["field_reduction"].rstrip('%')) for r in evaluation_results if r["field_reduction"] != "0%"]
        
        # Document type breakdown
        type_breakdown = {}
        for result in evaluation_results:
            doc_type = result["document_type"]
            if doc_type not in type_breakdown:
                type_breakdown[doc_type] = {
                    "count": 0,
                    "accuracies": [],
                    "ato_compliant": 0 if doc_type == "invoice" else None
                }
            
            type_breakdown[doc_type]["count"] += 1
            type_breakdown[doc_type]["accuracies"].append(result["overall_metrics"]["overall_accuracy"])
            
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
                "average_accuracy": sum(overall_accuracies) / len(overall_accuracies) if overall_accuracies else 0,
                "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "average_field_reduction": sum(field_reductions) / len(field_reductions) if field_reductions else 0,
                "documents_above_80_percent": sum(1 for acc in overall_accuracies if acc >= 0.8),
                "documents_above_90_percent": sum(1 for acc in overall_accuracies if acc >= 0.9),
            }
        }
        
        # Add per-document-type metrics
        for doc_type, stats in type_breakdown.items():
            avg_accuracy = sum(stats["accuracies"]) / len(stats["accuracies"])
            summary_report["document_type_breakdown"][doc_type] = {
                "documents": stats["count"],
                "average_accuracy": avg_accuracy,
                "accuracy_percentage": f"{avg_accuracy * 100:.1f}%"
            }
            
            # Add ATO compliance for invoices
            if doc_type == "invoice" and stats["ato_compliant"] is not None:
                compliance_rate = stats["ato_compliant"] / stats["count"] if stats["count"] > 0 else 0
                summary_report["document_type_breakdown"][doc_type]["ato_compliance"] = {
                    "compliant_documents": stats["ato_compliant"],
                    "compliance_rate": f"{compliance_rate * 100:.0f}%"
                }
        
        return {
            "summary": summary_report,
            "detailed_results": evaluation_results
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="InternVL3 Document-Aware Extraction Pipeline")
    parser.add_argument("--model-path", default=None, help="Path to InternVL3 model")
    parser.add_argument("--data-dir", default="evaluation_data", help="Directory with images")
    parser.add_argument("--ground-truth", default="evaluation_data/ground_truth.csv", 
                       help="Ground truth CSV file")
    parser.add_argument("--limit-images", type=int, help="Limit number of images to process")
    parser.add_argument("--document-type", choices=["invoice", "receipt", "bank_statement"],
                       help="Process only specific document type")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    print("🚀 InternVL3 Vision Document-Aware Extraction Pipeline")
    print("=" * 80)
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📋 Ground truth: {args.ground_truth}")
    if args.limit_images:
        print(f"🔢 Limiting to: {args.limit_images} images")
    if args.document_type:
        print(f"📄 Document type filter: {args.document_type}")
    print()

    # Initialize processor
    processor = DocumentAwareInternVL3Handler(model_path=args.model_path, debug=args.debug)

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
            # Phase 1: Document Type Detection & Schema Routing
            classification_info = processor.detect_and_classify_document(image_path)
            print(f"  📋 Document Type: {classification_info['document_type']} ({classification_info['field_count']} fields)")
            
            # Optional: Filter by document type
            if args.document_type and classification_info['document_type'] != args.document_type:
                print(f"  ⏭️ Skipping - looking for {args.document_type} documents")
                continue
            
            # Phase 2: Document-Aware Extraction
            result = processor.process_document_aware(image_path, classification_info)
            results.append(result)
            processed_count += 1
            
            # Display results
            reduction = result["field_reduction"] 
            detected = result["detected_fields"]
            total = result["total_fields"]
            time_taken = result["processing_time"]
            
            print(f"  ✅ {classification_info['document_type']}: {detected}/{total} fields ({reduction} field reduction)")
            print(f"  ⏱️ Processing time: {time_taken:.2f}s")
            
            # Check if we've processed enough documents of the target type
            if args.limit_images and processed_count >= args.limit_images:
                print(f"\n🎯 Reached limit: processed {processed_count} {args.document_type or 'document'}(s)")
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
    print(f"🏎️ Average time: {total_time / len(results):.2f}s per document" if results else "No results")

    # Evaluate results
    if results:
        evaluation_report = processor.evaluate_document_aware(results, ground_truth)
        
        # Display summary
        summary = evaluation_report["summary"]
        print("\n📈 DOCUMENT-AWARE EVALUATION RESULTS")
        print("=" * 80)
        print(f"📊 Overall Accuracy: {summary['overall_metrics']['average_accuracy'] * 100:.1f}%")
        print(f"⚡ Average Processing Time: {summary['overall_metrics']['average_processing_time']:.2f}s")
        print(f"🔧 Average Field Reduction: {summary['overall_metrics']['average_field_reduction']:.0f}%")
        print(f"🎯 Documents >80%: {summary['overall_metrics']['documents_above_80_percent']}/{summary['total_documents']}")
        print(f"🏆 Documents >90%: {summary['overall_metrics']['documents_above_90_percent']}/{summary['total_documents']}")
        
        print("\n📄 BY DOCUMENT TYPE:")
        print("-" * 40)
        for doc_type, stats in summary["document_type_breakdown"].items():
            print(f"  {doc_type.upper()}: {stats['accuracy_percentage']} ({stats['documents']} docs)")
            if "ato_compliance" in stats:
                print(f"    ATO Compliance: {stats['ato_compliance']['compliance_rate']}")
        
        print("\n💡 INSIGHTS:")
        if summary['overall_metrics']['average_field_reduction'] > 0:
            print(f"  • Field reduction improved efficiency by {summary['overall_metrics']['average_field_reduction']:.0f}%")
        if summary['overall_metrics']['documents_above_90_percent'] > 0:
            print(f"  • {summary['overall_metrics']['documents_above_90_percent']} documents achieved >90% accuracy")
        
        # Save detailed results
        from common.config import OUTPUT_DIR
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"internvl3_document_aware_report_{timestamp}.json"
        
        with report_path.open('w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"\n✅ Detailed report saved to: {report_path}")
        print("\n" + "=" * 80)
        print("✅ PHASE 4 DOCUMENT-AWARE EXTRACTION COMPLETE")
        print("=" * 80)
    else:
        print("❌ No results to evaluate")


if __name__ == "__main__":
    main()