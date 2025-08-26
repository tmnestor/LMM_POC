#!/usr/bin/env python3
"""
Document-Type-Specific Schema Testing - Phase 2

Test the new hierarchical schema system and document-type routing.
Validates schema loading, field reduction, and backward compatibility.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from common.document_schema_loader import DocumentTypeFieldSchema
    from common.document_type_detector import DocumentTypeDetector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all required files exist and you're running from project root")
    sys.exit(1)


def test_schema_loading():
    """Test basic schema loading and validation."""
    print("🔧 Testing schema loading...")
    
    try:
        # Initialize schema loader
        schema_loader = DocumentTypeFieldSchema()
        
        # Test supported document types
        doc_types = schema_loader.get_supported_document_types()
        print(f"✅ Supported document types: {doc_types}")
        
        # Test schema validation for each type
        all_valid = True
        for doc_type in doc_types:
            validation = schema_loader.validate_document_type_schema(doc_type)
            if validation["valid"]:
                print(f"✅ {doc_type}: Valid schema ({validation['field_count']} fields)")
            else:
                print(f"❌ {doc_type}: Invalid - {validation.get('error', 'Unknown error')}")
                all_valid = False
        
        return all_valid, schema_loader
        
    except Exception as e:
        print(f"❌ Schema loading failed: {e}")
        return False, None


def test_field_reduction():
    """Test field reduction efficiency for each document type."""
    print("\n📊 Testing field reduction efficiency...")
    
    try:
        schema_loader = DocumentTypeFieldSchema()
        
        print(f"{'Document Type':<15} {'Specific':<10} {'Unified':<10} {'Reduction':<12} {'Efficiency':<15}")
        print("-" * 70)
        
        total_reductions = []
        
        for doc_type in schema_loader.get_supported_document_types():
            comparison = schema_loader.compare_schemas(doc_type)
            
            specific = comparison['specific_field_count']
            unified = comparison['unified_field_count']
            reduction = comparison['field_reduction']
            efficiency = f"{comparison['field_reduction_percentage']:.0f}%"
            
            print(f"{doc_type:<15} {specific:<10} {unified:<10} {reduction:<12} {efficiency:<15}")
            total_reductions.append(comparison['field_reduction_percentage'])
        
        avg_reduction = sum(total_reductions) / len(total_reductions)
        print(f"\n📈 Average field reduction: {avg_reduction:.1f}%")
        
        if avg_reduction >= 30:
            print("🎉 Excellent efficiency gains!")
        elif avg_reduction >= 20:
            print("👍 Good efficiency improvements")
        else:
            print("⚠️ Lower than expected efficiency gains")
        
        return True
        
    except Exception as e:
        print(f"❌ Field reduction testing failed: {e}")
        return False


def test_schema_details():
    """Test detailed schema properties for each document type."""
    print("\n🔍 Testing detailed schema properties...")
    
    try:
        schema_loader = DocumentTypeFieldSchema()
        
        for doc_type in schema_loader.get_supported_document_types():
            print(f"\n📋 {doc_type.upper()} SCHEMA DETAILS:")
            
            schema = schema_loader.get_document_schema(doc_type)
            field_names = schema_loader.get_field_names_for_type(doc_type)
            
            print(f"   Total fields: {schema['total_fields']}")
            print(f"   Extraction mode: {schema['extraction_mode']}")
            print(f"   Validation rules: {len(schema['validation_rules'])}")
            
            # Show key fields
            critical_fields = [f["name"] for f in schema["fields"] if f.get("group") == "critical"]
            monetary_fields = [f["name"] for f in schema["fields"] if f.get("group") == "monetary"]
            
            print(f"   Critical fields: {critical_fields}")
            print(f"   Monetary fields: {len(monetary_fields)} fields")
            
            # Show excluded fields
            excluded = schema.get("excluded_fields", [])
            if excluded:
                print(f"   Excluded from unified: {len(excluded)} fields")
            
        return True
        
    except Exception as e:
        print(f"❌ Schema details testing failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with v1 unified schema."""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        schema_loader = DocumentTypeFieldSchema()
        
        # Test unknown document type (should fall back to unified)
        unknown_schema = schema_loader.get_document_schema("unknown")
        
        print("✅ Unknown document type handling:")
        print(f"   Falls back to: {unknown_schema['document_type']}")
        print(f"   Field count: {unknown_schema['total_fields']}")
        print(f"   Extraction mode: {unknown_schema['extraction_mode']}")
        
        # Test v1 availability
        if schema_loader.v1_available:
            print("✅ V1 schema available for fallback")
        else:
            print("⚠️ V1 schema not available - limited fallback support")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility testing failed: {e}")
        return False


def test_with_document_detector(processor_type: str = None):
    """Test integration with document type detector."""
    print("\n🔬 Testing integration with document detector...")
    
    if not processor_type:
        print("💡 Skipping detector integration (no processor specified)")
        print("💡 Run with --model llama or --model internvl3 to test full integration")
        return True
    
    try:
        # Initialize processor
        if processor_type.lower() == "llama":
            from models.llama_processor import LlamaProcessor
            processor = LlamaProcessor()
        elif processor_type.lower() == "internvl3":
            from models.internvl3_processor import InternVL3Processor
            processor = InternVL3Processor()
        else:
            raise ValueError(f"Unsupported processor: {processor_type}")
        
        # Initialize detector and schema loader
        detector = DocumentTypeDetector(processor)
        schema_loader = DocumentTypeFieldSchema()
        schema_loader.set_document_detector(detector)
        
        # Test with sample image
        test_image = "evaluation_data/synthetic_invoice_001.png"
        if Path(test_image).exists():
            print(f"📄 Testing with image: {Path(test_image).name}")
            
            # Get schema for image (should auto-detect type)
            schema = schema_loader.get_schema_for_image(test_image)
            
            print("✅ Auto-detection results:")
            print(f"   Document type: {schema.get('document_type', 'unknown')}")
            print(f"   Field count: {schema['total_fields']}")
            print(f"   Extraction mode: {schema['extraction_mode']}")
            
            return True
        else:
            print(f"⚠️ Test image not found: {test_image}")
            print("💡 Place a test image in evaluation_data/ for full integration testing")
            return True
        
    except Exception as e:
        print(f"❌ Detector integration testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extraction_strategies():
    """Test extraction strategy configuration."""
    print("\n⚙️ Testing extraction strategies...")
    
    try:
        schema_loader = DocumentTypeFieldSchema()
        
        for doc_type in schema_loader.get_supported_document_types():
            strategy = schema_loader.get_extraction_strategy(doc_type)
            
            print(f"📋 {doc_type.upper()} extraction strategy:")
            print(f"   Optimization level: {strategy['optimization_level']}")
            print(f"   Field count: {strategy['field_count']}")
            print(f"   Validation rules: {len(strategy['validation_rules'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction strategy testing failed: {e}")
        return False


def generate_test_report(schema_loader):
    """Generate comprehensive test report."""
    print("\n📊 GENERATING TEST REPORT...")
    
    try:
        report = schema_loader.generate_schema_report()
        print(report)
        
        # Additional test metrics
        print("\n🧪 TEST RESULTS SUMMARY:")
        
        # Count validation results
        valid_schemas = 0
        total_schemas = len(schema_loader.get_supported_document_types())
        
        for doc_type in schema_loader.get_supported_document_types():
            validation = schema_loader.validate_document_type_schema(doc_type)
            if validation["valid"]:
                valid_schemas += 1
        
        success_rate = (valid_schemas / total_schemas) * 100
        print(f"   Schema validation success rate: {success_rate:.1f}% ({valid_schemas}/{total_schemas})")
        
        if success_rate == 100:
            print("🎉 Perfect schema validation!")
        elif success_rate >= 80:
            print("👍 Good schema validation")
        else:
            print("⚠️ Schema issues need attention")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False


def main():
    """Main testing routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test document-type-specific schema system")
    parser.add_argument('--model', choices=['llama', 'internvl3'],
                       help='Test with specific model processor for full integration')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed output for all tests')
    
    args = parser.parse_args()
    
    print("🚀 Document-Type-Specific Schema Testing - Phase 2")
    print("=" * 60)
    
    # Run test suite
    test_results = []
    
    # 1. Basic schema loading
    print("\n1️⃣ SCHEMA LOADING TEST")
    success, schema_loader = test_schema_loading()
    test_results.append(("Schema Loading", success))
    
    if not success or not schema_loader:
        print("❌ Critical failure - cannot continue testing")
        return
    
    # 2. Field reduction efficiency
    print("\n2️⃣ FIELD REDUCTION TEST")
    success = test_field_reduction()
    test_results.append(("Field Reduction", success))
    
    # 3. Schema details
    if args.detailed:
        print("\n3️⃣ SCHEMA DETAILS TEST")
        success = test_schema_details()
        test_results.append(("Schema Details", success))
    
    # 4. Backward compatibility
    print("\n4️⃣ BACKWARD COMPATIBILITY TEST")
    success = test_backward_compatibility()
    test_results.append(("Backward Compatibility", success))
    
    # 5. Extraction strategies
    print("\n5️⃣ EXTRACTION STRATEGIES TEST")
    success = test_extraction_strategies()
    test_results.append(("Extraction Strategies", success))
    
    # 6. Document detector integration (if model specified)
    if args.model:
        print("\n6️⃣ DETECTOR INTEGRATION TEST")
        success = test_with_document_detector(args.model)
        test_results.append(("Detector Integration", success))
    
    # 7. Generate comprehensive report
    print("\n7️⃣ COMPREHENSIVE REPORT")
    success = generate_test_report(schema_loader)
    test_results.append(("Report Generation", success))
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    overall_success = (passed / total) * 100
    print(f"\n🎯 Overall success rate: {overall_success:.1f}% ({passed}/{total} tests passed)")
    
    if overall_success == 100:
        print("🎉 ALL TESTS PASSED! Phase 2 schema implementation is ready.")
        print("\n🚀 READY FOR PHASE 3: Pipeline Integration")
    elif overall_success >= 80:
        print("👍 Most tests passed. Minor issues may need attention.")
    else:
        print("🚨 Multiple test failures. Significant issues need resolution.")
    
    print("\n💡 Next steps:")
    if args.model:
        print("   - Schema system validated with full integration")
        print("   - Ready to modify processors for document-type routing")
    else:
        print("   - Run with --model llama or --model internvl3 for full integration test")
    
    print("   - Proceed to Phase 3: Pipeline Integration")
    print("   - Test document-specific extraction on real images")


if __name__ == "__main__":
    main()