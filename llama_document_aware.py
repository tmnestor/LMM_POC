#!/usr/bin/env python3
"""
Llama Vision Document-Aware Key-Value Extraction - Phase 4 Implementation

This module implements the Phase 4 document-aware extraction pipeline for the 
Llama-3.2-11B-Vision-Instruct model, featuring:
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
    python llama_document_aware.py --limit-images 5 --debug
    python llama_document_aware.py --document-type invoice
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
from models.document_aware_llama_processor import DocumentAwareLlamaProcessor


class DocumentAwareLlamaHandler:
    """Phase 4 Document-Aware Llama Vision Processor."""
    
    def __init__(self, model_path: str, debug: bool = False):
        """Initialize document-aware processor."""
        self.debug = debug
        self.model_path = model_path
        
        print("🚀 Initializing Llama Vision processor for document-aware extraction...")
        
        # We'll create processors on-demand to avoid loading multiple models
        self.base_processor = None
        self.model_loaded = False
        
        # Initialize Phase 4 components
        self.schema_loader = DocumentTypeFieldSchema()
        self.evaluator = DocumentTypeEvaluator()
        
        # Note: Document detector will be configured when we need it
        self.document_detector = None
        
        print("✅ Document-aware Llama handler initialized (model will load on first use)")
    
    def detect_and_classify_document(self, image_path: str) -> Dict[str, Any]:
        """Detect document type and get appropriate schema."""
        
        if self.debug:
            print(f"📋 Detecting document type for: {image_path}")
        
        # Initialize document detector on first use
        if not self.document_detector:
            # Create a minimal processor just for document type detection
            detection_fields = ["DOCUMENT_TYPE"]  # Only need this for classification
            self.base_processor = DocumentAwareLlamaProcessor(
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
        
        document_processor = DocumentAwareLlamaProcessor(
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
            document_processor.processor = self.base_processor.processor
        
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
            "processing_time": processing_time,
            "extracted_data": extracted_data
        }
    
    def evaluate_document_aware(self, results: List[Dict], ground_truth: Dict) -> Dict[str, Any]:
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
                result["extracted_data"],
                gt_data,
                result["document_type"]
            )
            
            evaluation["image_file"] = image_file
            evaluation["processing_time"] = result["processing_time"]
            evaluations.append(evaluation)
        
        return self._generate_document_aware_report(evaluations)
    
    def _generate_document_aware_report(self, evaluations: List[Dict]) -> Dict[str, Any]:
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
            "by_document_type": {}
        }
        
        print("\\n" + "=" * 80)
        print("📊 DOCUMENT-AWARE EVALUATION RESULTS")
        print("=" * 80)
        
        for doc_type, type_evaluations in by_type.items():
            # Calculate type-specific metrics
            accuracies = [e["overall_metrics"]["overall_accuracy"] for e in type_evaluations]
            meets_threshold = sum(1 for e in type_evaluations 
                                if e["overall_metrics"].get("meets_threshold", False))
            critical_perfect = sum(1 for e in type_evaluations 
                                 if e["overall_metrics"].get("critical_fields_perfect", False))
            
            # ATO compliance for invoices
            ato_compliant = 0
            if doc_type == "invoice":
                ato_compliant = sum(1 for e in type_evaluations 
                                  if e["overall_metrics"].get("ato_compliant", False))
            
            type_metrics = {
                "count": len(type_evaluations),
                "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "meets_threshold": meets_threshold,
                "critical_perfect": critical_perfect,
                "ato_compliant": ato_compliant if doc_type == "invoice" else None
            }
            
            report["by_document_type"][doc_type] = type_metrics
            
            # Print results
            print(f"\\n📋 {doc_type.upper()}:")
            print(f"   Documents: {type_metrics['count']}")
            print(f"   Avg Accuracy: {type_metrics['avg_accuracy']*100:.1f}%")
            print(f"   Meeting Threshold: {meets_threshold}/{len(type_evaluations)}")
            print(f"   Critical Fields Perfect: {critical_perfect}/{len(type_evaluations)}")
            
            if doc_type == "invoice" and ato_compliant is not None:
                print(f"   ATO Compliant: {ato_compliant}/{len(type_evaluations)} ({ato_compliant/len(type_evaluations)*100:.0f}%)")
        
        return report


def main():
    """Run document-aware Llama evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Document-Aware Llama Vision Extraction - Phase 4")
    parser.add_argument("--model-path", default="/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct",
                       help="Path to Llama model")
    parser.add_argument("--data-dir", default="evaluation_data", help="Directory with images")
    parser.add_argument("--ground-truth", default="evaluation_data/ground_truth.csv", 
                       help="Ground truth CSV file")
    parser.add_argument("--limit-images", type=int, help="Limit number of images to process")
    parser.add_argument("--document-type", choices=["invoice", "receipt", "bank_statement"],
                       help="Process only specific document type")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    print("\\n" + "=" * 80)
    print("🚀 LLAMA DOCUMENT-AWARE EXTRACTION - PHASE 4")
    print("=" * 80)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Ground truth: {args.ground_truth}")
    print(f"🔧 Model: {args.model_path}")
    print(f"🎯 Document type filter: {args.document_type or 'All types'}")
    print(f"📸 Image limit: {args.limit_images or 'No limit'}")
    print(f"🐛 Debug mode: {args.debug}")
    
    # Verify files exist
    if not Path(args.ground_truth).exists():
        print(f"❌ ERROR: Ground truth file not found: {args.ground_truth}")
        return
    
    if not Path(args.data_dir).exists():
        print(f"❌ ERROR: Data directory not found: {args.data_dir}")
        return
    
    # Initialize document-aware processor
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
                gt_doc_type = ground_truth[img_name].get("DOCUMENT_TYPE", "").lower()
                if args.document_type.lower() in gt_doc_type:
                    filtered_images.append(img)
        all_images = filtered_images
        print(f"🔍 Filtered to {len(all_images)} {args.document_type} documents")
    
    # Limit images if specified
    if args.limit_images:
        all_images = all_images[:args.limit_images]
        print(f"📸 Processing {len(all_images)} images")
    
    # Process each image with document-aware extraction
    results = []
    
    print(f"\\n🔄 Processing {len(all_images)} documents...")
    
    for i, image_path in enumerate(all_images, 1):
        print(f"\\n📄 [{i}/{len(all_images)}] Processing {Path(image_path).name}...")
        
        try:
            # Step 1: Detect document type and get schema
            classification_info = processor.detect_and_classify_document(image_path)
            
            # Step 2: Extract with document-specific schema
            result = processor.process_document_aware(image_path, classification_info)
            results.append(result)
            
            # Show progress
            efficiency = (25 - result["total_fields"]) / 25 * 100  # vs unified 25 fields
            print(f"   ✅ {result['document_type']}: {result['detected_fields']}/{result['total_fields']} fields ({efficiency:.0f}% field reduction)")
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
        
        with Path(report_path).open('w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"\\n✅ Detailed report saved to: {report_path}")
        print("\\n" + "=" * 80)
        print("✅ PHASE 4 DOCUMENT-AWARE EXTRACTION COMPLETE")
        print("=" * 80)
    else:
        print("❌ No results to evaluate")


if __name__ == "__main__":
    main()