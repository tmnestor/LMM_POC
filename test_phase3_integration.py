#!/usr/bin/env python3
"""
Phase 3 Integration Testing - End-to-End Document-Aware Processing

Complete testing of the document-type-specific extraction pipeline:
- Document detection (Phase 1)
- Schema routing (Phase 2) 
- Processor integration (Phase 3)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from common.extraction_parser import discover_images
    from models.internvl3_processor_v2 import DocumentAwareInternVL3Processor
    from models.llama_processor_v2 import DocumentAwareLlamaProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all Phase 3 files exist and you're running from project root")
    sys.exit(1)


class Phase3IntegrationTester:
    """Comprehensive tester for Phase 3 document-aware integration."""
    
    def __init__(self, model_type: str = "llama"):
        """
        Initialize integration tester.
        
        Args:
            model_type: Model to test ("llama" or "internvl3")
        """
        self.model_type = model_type.lower()
        self.processor = None
        
        # Test results tracking
        self.test_results = {
            "initialization": False,
            "single_image_processing": False,
            "document_type_detection": False,
            "schema_routing": False,
            "field_reduction": False,
            "batch_processing": False,
            "performance_validation": False,
            "error_handling": False
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_tests": 0,
            "successful_extractions": 0,
            "document_types_detected": set(),
            "avg_processing_time": 0,
            "avg_field_reduction": 0,
            "efficiency_gains": []
        }
    
    def initialize_processor(self) -> bool:
        """Initialize the document-aware processor."""
        print(f"🚀 Initializing {self.model_type.upper()} document-aware processor...")
        
        try:
            if self.model_type == "llama":
                self.processor = DocumentAwareLlamaProcessor(
                    debug=True,
                    enable_document_awareness=True
                )
            elif self.model_type == "internvl3":
                self.processor = DocumentAwareInternVL3Processor(
                    debug=True,
                    enable_document_awareness=True
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            print(f"✅ {self.model_type.upper()} processor initialized successfully")
            self.test_results["initialization"] = True
            return True
            
        except Exception as e:
            print(f"❌ Processor initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_single_image_processing(self, image_path: str) -> Dict:
        """Test single image processing with document awareness."""
        print(f"\n📄 Testing single image processing: {Path(image_path).name}")
        
        if not Path(image_path).exists():
            print(f"❌ Test image not found: {image_path}")
            return {"success": False, "error": "Image not found"}
        
        try:
            start_time = time.time()
            
            # Process with document awareness
            result = self.processor.process_single_image(image_path)
            
            processing_time = time.time() - start_time
            
            # Validate result structure
            required_keys = ["document_awareness", "enhanced_processor"]
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"❌ Missing required keys in result: {missing_keys}")
                return {"success": False, "error": f"Missing keys: {missing_keys}"}
            
            # Extract metrics
            doc_type = result.get("detected_document_type", "unknown")
            field_count = result.get("fields_extracted", 25)
            field_reduction = result.get("field_reduction", 0)
            efficiency_gain = result.get("efficiency_gain", "0%")
            
            print("✅ Processing completed successfully:")
            print(f"   Document type: {doc_type}")
            print(f"   Fields extracted: {field_count}")
            print(f"   Field reduction: {field_reduction}")
            print(f"   Efficiency gain: {efficiency_gain}")
            print(f"   Processing time: {processing_time:.2f}s")
            
            # Update metrics
            self.performance_metrics["total_tests"] += 1
            self.performance_metrics["successful_extractions"] += 1
            self.performance_metrics["document_types_detected"].add(doc_type)
            self.performance_metrics["avg_field_reduction"] += field_reduction
            
            if efficiency_gain != "0%":
                efficiency_pct = float(efficiency_gain.strip("%"))
                self.performance_metrics["efficiency_gains"].append(efficiency_pct)
            
            self.test_results["single_image_processing"] = True
            
            return {
                "success": True,
                "document_type": doc_type,
                "field_count": field_count,
                "field_reduction": field_reduction,
                "efficiency_gain": efficiency_gain,
                "processing_time": processing_time,
                "result": result
            }
            
        except Exception as e:
            print(f"❌ Single image processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_document_type_detection(self, test_images: List[str]) -> bool:
        """Test document type detection across different document types."""
        print(f"\n🔍 Testing document type detection on {len(test_images)} images...")
        
        detection_results = {}
        successful_detections = 0
        
        for image_path in test_images:
            try:
                if not Path(image_path).exists():
                    print(f"⚠️ Skipping missing image: {Path(image_path).name}")
                    continue
                
                # Test detection
                result = self.test_single_image_processing(image_path)
                
                if result["success"]:
                    doc_type = result["document_type"]
                    image_name = Path(image_path).name
                    
                    detection_results[image_name] = doc_type
                    successful_detections += 1
                    
                    print(f"✅ {image_name}: {doc_type}")
                else:
                    print(f"❌ {Path(image_path).name}: Detection failed")
                    
            except Exception as e:
                print(f"❌ Error processing {Path(image_path).name}: {e}")
        
        # Analyze detection results
        detected_types = set(detection_results.values())
        detection_rate = (successful_detections / len(test_images)) * 100 if test_images else 0
        
        print("\n📊 Detection Results:")
        print(f"   Successful detections: {successful_detections}/{len(test_images)} ({detection_rate:.1f}%)")
        print(f"   Document types found: {detected_types}")
        
        # Success criteria: >80% detection rate and >1 document type
        success = detection_rate >= 80 and len(detected_types) > 1
        
        if success:
            print("🎉 Document type detection test PASSED!")
            self.test_results["document_type_detection"] = True
        else:
            print("⚠️ Document type detection needs improvement")
        
        return success
    
    def test_schema_routing(self) -> bool:
        """Test that different document types use different schemas."""
        print("\n⚙️ Testing schema routing...")
        
        # Test with known document types if available
        test_cases = [
            {"type": "invoice", "expected_fields": 19},
            {"type": "bank_statement", "expected_fields": 15},
            {"type": "receipt", "expected_fields": 15}
        ]
        
        routing_tests_passed = 0
        
        for case in test_cases:
            try:
                # Get schema for document type
                if hasattr(self.processor, 'schema_loader') and self.processor.schema_loader:
                    schema = self.processor.schema_loader.get_document_schema(case["type"])
                    actual_fields = schema["total_fields"]
                    
                    if actual_fields == case["expected_fields"]:
                        print(f"✅ {case['type']}: {actual_fields} fields (correct)")
                        routing_tests_passed += 1
                    else:
                        print(f"❌ {case['type']}: {actual_fields} fields (expected {case['expected_fields']})")
                else:
                    print("⚠️ Schema loader not available")
                    break
                    
            except Exception as e:
                print(f"❌ Schema routing test failed for {case['type']}: {e}")
        
        success = routing_tests_passed == len(test_cases)
        
        if success:
            print("🎉 Schema routing test PASSED!")
            self.test_results["schema_routing"] = True
        else:
            print("⚠️ Schema routing needs attention")
        
        return success
    
    def test_field_reduction(self) -> bool:
        """Test that field reduction is working as expected."""
        print("\n📊 Testing field reduction efficiency...")
        
        if not self.performance_metrics["efficiency_gains"]:
            print("⚠️ No efficiency gains recorded yet")
            return False
        
        avg_efficiency = sum(self.performance_metrics["efficiency_gains"]) / len(self.performance_metrics["efficiency_gains"])
        field_reduction_threshold = 15  # Minimum 15% improvement expected
        
        print(f"   Average efficiency gain: {avg_efficiency:.1f}%")
        print(f"   Efficiency threshold: {field_reduction_threshold}%")
        
        success = avg_efficiency >= field_reduction_threshold
        
        if success:
            print("🎉 Field reduction test PASSED!")
            self.test_results["field_reduction"] = True
        else:
            print("⚠️ Field reduction below threshold")
        
        return success
    
    def test_batch_processing(self, image_files: List[str]) -> bool:
        """Test batch processing with document awareness."""
        print(f"\n🔬 Testing batch processing with {len(image_files)} images...")
        
        if len(image_files) < 2:
            print("⚠️ Need at least 2 images for batch testing")
            return False
        
        try:
            # Test batch processing
            batch_start_time = time.time()
            results, stats = self.processor.process_image_batch(image_files)
            batch_time = time.time() - batch_start_time
            
            # Validate results
            successful = stats.get("successful", 0)
            total = stats.get("total_images", 0)
            success_rate = (successful / total) * 100 if total > 0 else 0
            
            print("✅ Batch processing completed:")
            print(f"   Total images: {total}")
            print(f"   Successful: {successful}")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Batch time: {batch_time:.2f}s")
            
            # Check document-aware statistics
            if "document_type_distribution" in stats:
                print(f"   Document types: {stats['document_type_distribution']}")
                print(f"   Avg field reduction: {stats.get('avg_field_reduction', 0)}")
                print(f"   Efficiency estimate: {stats.get('efficiency_gain_estimate', '0%')}")
            
            # Success criteria: >70% success rate
            success = success_rate >= 70
            
            if success:
                print("🎉 Batch processing test PASSED!")
                self.test_results["batch_processing"] = True
            else:
                print("⚠️ Batch processing success rate below threshold")
            
            return success
            
        except Exception as e:
            print(f"❌ Batch processing test failed: {e}")
            return False
    
    def test_performance_validation(self) -> bool:
        """Validate overall performance improvements."""
        print("\n⚡ Testing performance validation...")
        
        metrics = self.performance_metrics
        
        # Calculate averages
        if metrics["total_tests"] > 0:
            avg_field_reduction = metrics["avg_field_reduction"] / metrics["total_tests"]
            avg_efficiency = sum(metrics["efficiency_gains"]) / len(metrics["efficiency_gains"]) if metrics["efficiency_gains"] else 0
        else:
            avg_field_reduction = 0
            avg_efficiency = 0
        
        print(f"   Total tests: {metrics['total_tests']}")
        print(f"   Successful extractions: {metrics['successful_extractions']}")
        print(f"   Document types detected: {len(metrics['document_types_detected'])}")
        print(f"   Average field reduction: {avg_field_reduction:.1f}")
        print(f"   Average efficiency gain: {avg_efficiency:.1f}%")
        
        # Success criteria
        success_criteria = [
            metrics["total_tests"] > 0,
            metrics["successful_extractions"] > 0,
            len(metrics["document_types_detected"]) > 1,
            avg_efficiency > 10  # At least 10% efficiency improvement
        ]
        
        success = all(success_criteria)
        
        if success:
            print("🎉 Performance validation PASSED!")
            self.test_results["performance_validation"] = True
        else:
            print("⚠️ Performance validation needs improvement")
        
        return success
    
    def test_error_handling(self) -> bool:
        """Test error handling and fallback mechanisms."""
        print("\n🛡️ Testing error handling...")
        
        error_tests = [
            {"name": "Non-existent image", "path": "non_existent_image.png"},
            {"name": "Invalid image path", "path": ""},
        ]
        
        error_handling_success = 0
        
        for test in error_tests:
            try:
                print(f"   Testing: {test['name']}")
                result = self.processor.process_single_image(test["path"])
                
                # Should handle error gracefully without crashing
                if "error" in result or not result.get("enhanced_processor", False):
                    print("   ✅ Error handled gracefully")
                    error_handling_success += 1
                else:
                    print("   ❌ Error not handled properly")
                    
            except Exception as e:
                # This is actually bad - we want graceful error handling
                print(f"   ❌ Unhandled exception: {e}")
        
        success = error_handling_success == len(error_tests)
        
        if success:
            print("🎉 Error handling test PASSED!")
            self.test_results["error_handling"] = True
        else:
            print("⚠️ Error handling needs improvement")
        
        return success
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        report = {
            "model_type": self.model_type,
            "test_results": self.test_results,
            "overall_success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "performance_metrics": self.performance_metrics,
            "status": "PASSED" if success_rate >= 85 else "NEEDS_IMPROVEMENT",
            "recommendations": []
        }
        
        # Add recommendations based on failed tests
        for test_name, passed in self.test_results.items():
            if not passed:
                report["recommendations"].append(f"Improve {test_name.replace('_', ' ')}")
        
        return report


def main():
    """Run comprehensive Phase 3 integration testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 3 document-aware integration")
    parser.add_argument('--model', choices=['llama', 'internvl3'], default='llama',
                       help='Model to test (default: llama)')
    parser.add_argument('--test-dir', type=str, default='evaluation_data',
                       help='Directory containing test images')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer images')
    
    args = parser.parse_args()
    
    print("🚀 Phase 3 Integration Testing - Document-Aware Processing")
    print("=" * 70)
    print(f"📊 Model: {args.model.upper()}")
    print(f"📂 Test directory: {args.test_dir}")
    
    # Initialize tester
    tester = Phase3IntegrationTester(args.model)
    
    # Test 1: Processor initialization
    print("\n1️⃣ PROCESSOR INITIALIZATION TEST")
    if not tester.initialize_processor():
        print("❌ Critical failure - cannot continue testing")
        return
    
    # Discover test images
    try:
        test_images = discover_images(args.test_dir)
        if args.quick:
            test_images = test_images[:3]  # Limit for quick testing
        print(f"📁 Found {len(test_images)} test images")
    except Exception as e:
        print(f"❌ Failed to discover images: {e}")
        test_images = []
    
    if not test_images:
        print("❌ No test images available - cannot run tests")
        return
    
    # Test 2: Single image processing
    print("\n2️⃣ SINGLE IMAGE PROCESSING TEST")
    tester.test_single_image_processing(str(test_images[0]))
    
    # Test 3: Document type detection
    print("\n3️⃣ DOCUMENT TYPE DETECTION TEST")
    tester.test_document_type_detection([str(img) for img in test_images])
    
    # Test 4: Schema routing
    print("\n4️⃣ SCHEMA ROUTING TEST")
    tester.test_schema_routing()
    
    # Test 5: Field reduction
    print("\n5️⃣ FIELD REDUCTION TEST")
    tester.test_field_reduction()
    
    # Test 6: Batch processing
    if len(test_images) >= 2:
        print("\n6️⃣ BATCH PROCESSING TEST")
        batch_images = [str(img) for img in test_images[:3]]  # Test with 3 images
        tester.test_batch_processing(batch_images)
    
    # Test 7: Performance validation
    print("\n7️⃣ PERFORMANCE VALIDATION TEST")
    tester.test_performance_validation()
    
    # Test 8: Error handling
    print("\n8️⃣ ERROR HANDLING TEST")
    tester.test_error_handling()
    
    # Generate final report
    print("\n" + "=" * 70)
    print("📊 FINAL TEST REPORT")
    print("=" * 70)
    
    report = tester.generate_final_report()
    
    print(f"🎯 Model: {report['model_type'].upper()}")
    print(f"🏆 Overall Success Rate: {report['overall_success_rate']:.1f}%")
    print(f"✅ Passed Tests: {report['passed_tests']}/{report['total_tests']}")
    print(f"📈 Status: {report['status']}")
    
    print("\n📋 Test Results:")
    for test_name, passed in report['test_results'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        print(f"   {status} {test_display}")
    
    # Performance summary
    metrics = report['performance_metrics']
    if metrics['total_tests'] > 0:
        avg_efficiency = sum(metrics['efficiency_gains']) / len(metrics['efficiency_gains']) if metrics['efficiency_gains'] else 0
        print("\n⚡ Performance Summary:")
        print(f"   Document types detected: {len(metrics['document_types_detected'])}")
        print(f"   Average efficiency gain: {avg_efficiency:.1f}%")
        print(f"   Total successful extractions: {metrics['successful_extractions']}")
    
    # Recommendations
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    # Final verdict
    if report['status'] == "PASSED":
        print("\n🎉 PHASE 3 INTEGRATION TEST PASSED!")
        print(f"✅ Document-aware {args.model.upper()} processor is ready for production!")
    else:
        print("\n⚠️ Phase 3 integration needs improvement")
        print("💡 Address the failed tests above before production deployment")


if __name__ == "__main__":
    main()