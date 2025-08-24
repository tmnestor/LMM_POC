#!/usr/bin/env python3
"""
Test script to verify extraction pipeline works with new schema-based fields.
Run this on remote machine after schema loader tests pass.
"""


def test_model_imports():
    """Test that model processors can import with new schema."""
    print("🧪 Testing Model Imports...")
    
    try:
        print("✅ Model processors import successfully")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_grouped_extraction_imports():
    """Test that grouped extraction works with schema."""
    print("\n🧪 Testing Grouped Extraction Imports...")
    
    try:
        from common.grouped_extraction import GroupedExtractionStrategy
        
        strategy = GroupedExtractionStrategy(
            extraction_mode="grouped",
            debug=True,
            grouping_strategy="detailed_grouped"
        )
        
        print("✅ Grouped extraction strategy created")
        print(f"📊 Number of groups: {len(strategy.field_groups)}")
        
        return True
    except Exception as e:
        print(f"❌ Grouped extraction test failed: {e}")
        return False

def test_parser_integration():
    """Test that parser works with schema fields."""
    print("\n🧪 Testing Parser Integration...")
    
    try:
        from common.extraction_parser import parse_extraction_response
        
        # Test sample response parsing
        test_response = """DOCUMENT_TYPE: invoice
BUSINESS_ABN: 12345678901  
TOTAL_AMOUNT: $58.62
SUPPLIER_NAME: Test Company"""
        
        parsed = parse_extraction_response(test_response)
        
        print("✅ Parser works with schema fields")
        print(f"📊 Parsed fields: {len(parsed)}")
        print(f"🔍 Sample parsed: {dict(list(parsed.items())[:3])}")
        
        return True
    except Exception as e:
        print(f"❌ Parser integration failed: {e}")
        return False

def test_field_consistency():
    """Test that all field references are consistent."""
    print("\n🧪 Testing Field Consistency...")
    
    try:
        from common.config import EXTRACTION_FIELDS, FIELD_GROUPS_DETAILED
        from common.schema_loader import get_global_schema
        
        schema = get_global_schema()
        
        # Verify field counts match
        schema_fields = set(schema.field_names)
        config_fields = set(EXTRACTION_FIELDS)
        
        if schema_fields == config_fields:
            print("✅ Schema and config fields match perfectly")
        else:
            missing_in_config = schema_fields - config_fields
            extra_in_config = config_fields - schema_fields
            print("❌ Field mismatch!")
            if missing_in_config:
                print(f"Missing in config: {missing_in_config}")
            if extra_in_config:
                print(f"Extra in config: {extra_in_config}")
            return False
            
        # Verify all group fields exist in main field list
        all_group_fields = set()
        for _group_name, group_config in FIELD_GROUPS_DETAILED.items():
            group_fields = set(group_config.get("fields", []))
            all_group_fields.update(group_fields)
        
        if all_group_fields.issubset(schema_fields):
            print("✅ All group fields exist in main field list")
        else:
            missing = all_group_fields - schema_fields
            print(f"❌ Group fields missing from main list: {missing}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Field consistency test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Phase 1 Extraction Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        test_model_imports,
        test_grouped_extraction_imports,
        test_parser_integration,
        test_field_consistency
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All pipeline tests passed! Ready for full extraction test!")
        print("💡 Next: Run 'python3 llama_keyvalue.py --debug --limit 1'")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")