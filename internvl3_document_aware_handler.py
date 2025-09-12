#!/usr/bin/env python3
"""
InternVL3 Document-Aware Handler - Following Llama Success Pattern

This module implements a document-aware extraction handler for InternVL3 models,
following the exact same pattern as the successful DocumentAwareLlamaHandler.

Key Features:
- Handler pattern with lazy model loading
- Simple constructor without immediate model loading  
- Processor lifecycle management
- H200/high-end GPU direct loading support
- V100 quantization fallback

Usage:
    handler = DocumentAwareInternVL3Handler(model_path, debug=True)
    classification_info = handler.detect_and_classify_document(image_path)
    result = handler.process_document_aware(image_path, classification_info)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from common.document_type_metrics import DocumentTypeEvaluator
from common.unified_schema import DocumentTypeFieldSchema
from models.document_aware_internvl3_processor import DocumentAwareInternVL3Processor


class DocumentAwareInternVL3Handler:
    """Document-Aware InternVL3 Handler using proven Llama success pattern."""

    def __init__(self, model_path: str, device: str = "cuda", debug: bool = False):
        """Initialize document-aware handler with lazy loading."""
        self.debug = debug
        self.model_path = model_path
        self.device = device

        print("🚀 Initializing InternVL3 processor for document-aware extraction...")

        # Initialize components (same pattern as Llama)
        self.schema_loader = DocumentTypeFieldSchema()
        self.evaluator = DocumentTypeEvaluator()

        # Load document detection prompts from unified schema
        self.detection_config = self.schema_loader.load_detection_prompts()

        if self.debug:
            print("📝 YAML-first prompt loader initialized")
            print(
                f"   Detection config version: {self.detection_config.get('version', 'unknown')}"
            )
            print(
                f"   Supported types: {len(self.detection_config.get('supported_types', []))}"
            )

        # CRITICAL: Lazy loading pattern - processor created only when needed
        self.processor = None

        print("✅ Document-aware InternVL3 handler initialized (model will load on first use)")

    def _detect_document_type_yaml(self, image_path: str) -> str:
        """YAML-first document type detection using configurable prompts."""
        if self.debug:
            print("📝 Using YAML-first document detection approach")

        # Get InternVL3-specific prompt from YAML configuration
        internvl3_config = self.detection_config["detection_prompts"]["internvl3"]
        doc_type_prompt = internvl3_config["user_prompt"]
        max_tokens = internvl3_config.get("max_tokens", 50)

        if self.debug:
            print(
                f"   YAML config version: {self.detection_config.get('version', 'unknown')}"
            )
            print(f"   Max tokens: {max_tokens}")
            print(f"   Prompt: {doc_type_prompt[:100]}...")

        # Ensure processor exists for detection (lazy loading)
        if not self.processor:
            self.processor = DocumentAwareInternVL3Processor(
                field_list=["DOCUMENT_TYPE"],  # Single field for detection
                model_path=self.model_path,
                device=self.device,
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

        # Reconfigure existing processor with new field list (same pattern as Llama)
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
            self.processor = DocumentAwareInternVL3Processor(
                field_list=field_names,
                model_path=self.model_path,
                device=self.device,
                debug=self.debug,
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
        print("🔧 Model: InternVL3 (2B/8B)")
        
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
            "evaluation_mode": "internvl3_document_aware",
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