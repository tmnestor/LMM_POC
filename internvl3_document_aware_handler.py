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

        # Recursion protection
        self._recursion_depth = 0
        self._max_recursion_depth = 5
        self._evaluation_calls = 0
        self._max_evaluation_calls = 10
        
        # Comprehensive tracing
        self._trace_counter = 0
        self._method_calls = {}
        import time
        self._start_time = time.time()

        print(f"🔍 TRACE-000: __init__ ENTRY at {time.time():.3f}")
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

        print(f"🔍 TRACE-001: __init__ EXIT at {time.time():.3f}")
        print("✅ Document-aware InternVL3 handler initialized (model will load on first use)")

    def _trace_method(self, method_name: str, action: str, **kwargs):
        """Helper method for comprehensive tracing."""
        import time
        self._trace_counter += 1
        trace_id = f"TRACE-{self._trace_counter:03d}"
        timestamp = time.time()
        elapsed = timestamp - self._start_time
        
        if action == "ENTRY":
            if method_name not in self._method_calls:
                self._method_calls[method_name] = 0
            self._method_calls[method_name] += 1
            call_count = self._method_calls[method_name]
            
            print(f"🔍 {trace_id}: {method_name} {action} at {timestamp:.3f}s (elapsed: {elapsed:.3f}s) [call #{call_count}]")
            if kwargs:
                print(f"   Args: {kwargs}")
            if call_count > 10:
                print(f"🚨 WARNING: {method_name} called {call_count} times - possible infinite recursion!")
        else:
            print(f"🔍 {trace_id}: {method_name} {action} at {timestamp:.3f}s (elapsed: {elapsed:.3f}s)")
            
        return trace_id

    def _detect_document_type_yaml(self, image_path: str) -> str:
        """YAML-first document type detection using configurable prompts."""
        self._trace_method("_detect_document_type_yaml", "ENTRY", image_path=image_path)
        
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
            print("🔍 TRACE: Creating processor for document detection...")
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Processor creation timed out after 60 seconds")
            
            try:
                # Set a timeout for processor creation
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 60 second timeout
                
                start_time = time.time()
                print(f"🔍 TRACE: Initializing DocumentAwareInternVL3Processor with model_path={self.model_path}")
                
                self.processor = DocumentAwareInternVL3Processor(
                    field_list=["DOCUMENT_TYPE"],  # Single field for detection
                    model_path=self.model_path,
                    device=self.device,
                    debug=self.debug,
                    skip_model_loading=False,  # Load model normally
                )
                
                signal.alarm(0)  # Cancel timeout
                elapsed = time.time() - start_time
                print(f"🔍 TRACE: Processor creation completed successfully in {elapsed:.3f}s")
                
            except (Exception, TimeoutError) as e:
                signal.alarm(0)  # Cancel timeout
                print(f"🚨 TRACE: Processor creation failed: {e}")
                # Fallback: return a default document type to prevent infinite loop
                self._trace_method("_detect_document_type_yaml", "EXIT", doc_type="invoice_fallback_due_to_processor_error")
                return self.detection_config["detection_config"].get("fallback_type", "invoice")

        # Use the processor to extract with YAML prompt
        response = self.processor._extract_with_custom_prompt(
            image_path, doc_type_prompt, max_new_tokens=max_tokens
        )

        # Parse and normalize using YAML type mappings
        doc_type = self._parse_document_type_response_yaml(response)

        if self.debug:
            print(f"   Raw response: '{response.strip()}'")
            print(f"   Parsed type: '{doc_type}'")

        self._trace_method("_detect_document_type_yaml", "EXIT", doc_type=doc_type)
        return doc_type

    def _parse_document_type_response_yaml(self, response: str) -> str:
        """Parse document type response using YAML-configured type mappings."""
        self._trace_method("_parse_document_type_response_yaml", "ENTRY", response_length=len(response) if response else 0)
        
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

        self._trace_method("_parse_document_type_response_yaml", "EXIT", raw_type=raw_type)
        return raw_type

    def detect_and_classify_document(self, image_path: str) -> Dict[str, Any]:
        """
        Detect document type using YAML-first approach and get appropriate schema.

        Args:
            image_path: Path to document image

        Returns:
            Dict with document type, schema, and field information
        """
        self._trace_method("detect_and_classify_document", "ENTRY", image_path=image_path)
        
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

        result = {
            "document_type": doc_type,
            "schema": schema,
            "field_count": len(schema["fields"]),
            "field_names": schema["fields"]
            if isinstance(schema["fields"][0], str)
            else [f["name"] for f in schema["fields"]],
        }
        
        self._trace_method("detect_and_classify_document", "EXIT", 
                          doc_type=doc_type, field_count=result["field_count"])
        return result

    def process_document_aware(
        self, image_path: str, classification_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single document with type-aware extraction."""
        self._trace_method("process_document_aware", "ENTRY", 
                          image_path=image_path, 
                          doc_type=classification_info.get('document_type', 'unknown'))
        
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

        result = {
            "image_file": Path(image_path).name,
            "document_type": doc_type,
            "detected_fields": len(
                [v for v in extracted_data.values() if v != "NOT_FOUND"]
            ),
            "total_fields": len(field_names),
            "processing_time": processing_time,
            "extracted_data": extracted_data,
        }
        
        self._trace_method("process_document_aware", "EXIT", 
                          doc_type=doc_type, 
                          detected_fields=result["detected_fields"],
                          processing_time=processing_time)
        return result

    def evaluate_document_aware(
        self, results: List[Dict], ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate results with document-type-specific metrics."""
        # Recursion protection
        self._evaluation_calls += 1
        if self._evaluation_calls > self._max_evaluation_calls:
            print("🚨 RECURSION PROTECTION: Too many evaluation calls, aborting to prevent infinite loop")
            return {"error": "recursion_protection_triggered", "summary": {"overall_metrics": {}, "document_type_breakdown": {}}}
        
        self._recursion_depth += 1
        if self._recursion_depth > self._max_recursion_depth:
            self._recursion_depth -= 1
            print("🚨 RECURSION PROTECTION: Maximum recursion depth exceeded")
            return {"error": "max_recursion_depth_exceeded", "summary": {"overall_metrics": {}, "document_type_breakdown": {}}}

        try:
            import traceback
            print(f"🔍 TRACE: evaluate_document_aware called with {len(results)} results, recursion_depth={self._recursion_depth}, eval_calls={self._evaluation_calls}")
            print("🔍 TRACE: Call stack:")
            for line in traceback.format_stack()[-3:-1]:  # Show last 2 stack frames (excluding current)
                print(f"   {line.strip()}")
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
                # TEMPORARILY DISABLED TO PREVENT INFINITE RECURSION
                # if len(results) == 1:
                #     self._display_detailed_comparison(result, gt_data, evaluation)

            return self._generate_document_aware_report(evaluations)
        finally:
            # Always decrement recursion depth
            self._recursion_depth -= 1

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
        # Recursion protection
        self._recursion_depth += 1
        if self._recursion_depth > self._max_recursion_depth:
            self._recursion_depth -= 1
            print("🚨 RECURSION PROTECTION: Maximum recursion depth exceeded in _generate_document_aware_report")
            return {"error": "max_recursion_depth_exceeded", "summary": {"overall_metrics": {}, "document_type_breakdown": {}}}
        
        try:
            print(f"🔍 TRACE: _generate_document_aware_report called with {len(evaluations)} evaluations, recursion_depth={self._recursion_depth}")
            
            # Group by document type
            by_type = {}
            for eval_result in evaluations:
                doc_type = eval_result["document_type"]
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(eval_result)

            # Calculate overall metrics across all documents
            all_accuracies = []
            total_meets_threshold = 0
            total_critical_perfect = 0
            total_ato_compliant = 0
            invoice_count = 0
            
            document_type_breakdown = {}

            # TEMPORARILY DISABLED TO PREVENT INFINITE RECURSION
            # print("\\n" + "=" * 80)
            # print("📊 DOCUMENT-AWARE EVALUATION RESULTS")
            # print("=" * 80)

            for doc_type, type_evaluations in by_type.items():
                # Calculate type-specific metrics
                accuracies = [
                    e["overall_metrics"]["overall_accuracy"] for e in type_evaluations
                ]
                all_accuracies.extend(accuracies)
                
                meets_threshold = sum(
                    1
                    for e in type_evaluations
                    if e["overall_metrics"].get("meets_threshold", False)
                )
                total_meets_threshold += meets_threshold
                
                critical_perfect = sum(
                    1
                    for e in type_evaluations
                    if e["overall_metrics"].get("critical_fields_perfect", False)
                )
                total_critical_perfect += critical_perfect

                # ATO compliance for invoices
                ato_compliant = 0
                if doc_type == "invoice":
                    invoice_count += len(type_evaluations)
                    ato_compliant = sum(
                        1
                        for e in type_evaluations
                        if e["overall_metrics"].get("ato_compliant", False)
                    )
                    total_ato_compliant += ato_compliant

                type_metrics = {
                    "documents": len(type_evaluations),
                    "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                    "accuracy_percentage": f"{(sum(accuracies) / len(accuracies) * 100 if accuracies else 0):.1f}%",
                    "meets_threshold": meets_threshold,
                    "critical_perfect": critical_perfect,
                }
                
                if doc_type == "invoice":
                    type_metrics["ato_compliance"] = {
                        "compliant": ato_compliant,
                        "total": len(type_evaluations),
                        "compliance_rate": f"{(ato_compliant / len(type_evaluations) * 100 if type_evaluations else 0):.0f}%"
                    }

                document_type_breakdown[doc_type] = type_metrics

            # TEMPORARILY DISABLED TO PREVENT INFINITE RECURSION
            # print(f"\\n📋 {doc_type.upper()}:")
            # print(f"   Documents: {type_metrics['documents']}")
            # print(f"   Avg Accuracy: {type_metrics['accuracy'] * 100:.1f}%")
            # print(f"   Meeting Threshold: {meets_threshold}/{len(type_evaluations)}")
            # print(
            #     f"   Critical Fields Perfect: {critical_perfect}/{len(type_evaluations)}"
            # )

            # if doc_type == "invoice" and 'ato_compliance' in type_metrics:
            #     ato_info = type_metrics['ato_compliance']
            #     print(
            #         f"   ATO Compliant: {ato_info['compliant']}/{ato_info['total']} ({ato_info['compliance_rate']})"
            #     )

            # Build the report with the expected 'summary' structure
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(evaluations),
                "evaluation_mode": "internvl3_document_aware",
                "summary": {
                    "overall_metrics": {
                        "average_accuracy": sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0,
                        "total_documents": len(evaluations),
                        "meets_threshold_count": total_meets_threshold,
                        "critical_perfect_count": total_critical_perfect,
                        "ato_compliant_count": total_ato_compliant if invoice_count > 0 else None,
                    },
                    "document_type_breakdown": document_type_breakdown
                }
            }

            return report
        finally:
            # Always decrement recursion depth
            self._recursion_depth -= 1