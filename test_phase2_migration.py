#!/usr/bin/env python3
"""
Test script for Phase 2 schema migration - Replace remaining hardcoded configurations.
Run this to validate that all hardcoded field configurations have been eliminated.
"""

def test_dynamic_field_groups():
    """Test that field groups are dynamically generated."""
    print("🧪 Testing Dynamic Field Groups...")
    
    try:
        from common.config import FIELD_GROUPS_DETAILED, FIELD_GROUPS_COGNITIVE
        
        print(f"✅ FIELD_GROUPS_DETAILED loaded: {len(FIELD_GROUPS_DETAILED)} groups")
        print(f"✅ FIELD_GROUPS_COGNITIVE loaded: {len(FIELD_GROUPS_COGNITIVE)} groups")
        
        # Test specific groups exist
        expected_detailed = ["critical", "monetary", "dates", "business_entity", "payer_info", "banking", "item_details", "metadata"]
        for group in expected_detailed:
            if group in FIELD_GROUPS_DETAILED:
                print(f"✅ Detailed group found: {group}")
            else:
                print(f"❌ Missing detailed group: {group}")
                return False
        
        expected_cognitive = ["regulatory_financial", "entity_contacts", "transaction_details", "temporal_data", "banking_payment", "document_metadata"]
        for group in expected_cognitive:
            if group in FIELD_GROUPS_COGNITIVE:
                print(f"✅ Cognitive group found: {group}")
            else:
                print(f"❌ Missing cognitive group: {group}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Dynamic field groups test failed: {e}")
        return False

def test_dynamic_field_types():
    """Test that field type groupings are dynamically generated."""
    print("\n🧪 Testing Dynamic Field Types...")
    
    try:
        from common.config import MONETARY_FIELDS, DATE_FIELDS, NUMERIC_ID_FIELDS, LIST_FIELDS, TEXT_FIELDS
        
        print(f"✅ MONETARY_FIELDS: {len(MONETARY_FIELDS)} fields")
        print(f"✅ DATE_FIELDS: {len(DATE_FIELDS)} fields") 
        print(f"✅ NUMERIC_ID_FIELDS: {len(NUMERIC_ID_FIELDS)} fields")
        print(f"✅ LIST_FIELDS: {len(LIST_FIELDS)} fields")
        print(f"✅ TEXT_FIELDS: {len(TEXT_FIELDS)} fields")
        
        # Verify some expected field assignments
        if "TOTAL_AMOUNT" in MONETARY_FIELDS:
            print("✅ TOTAL_AMOUNT correctly classified as monetary")
        else:
            print("❌ TOTAL_AMOUNT not found in MONETARY_FIELDS")
            return False
            
        if "BUSINESS_ABN" in NUMERIC_ID_FIELDS:
            print("✅ BUSINESS_ABN correctly classified as numeric_id")
        else:
            print("❌ BUSINESS_ABN not found in NUMERIC_ID_FIELDS")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Dynamic field types test failed: {e}")
        return False

def test_schema_field_consistency():
    """Test that schema and config field assignments are consistent."""
    print("\n🧪 Testing Schema-Config Consistency...")
    
    try:
        from common.config import EXTRACTION_FIELDS, FIELD_TYPES, FIELD_DESCRIPTIONS
        from common.schema_loader import get_global_schema
        
        schema = get_global_schema()
        
        # Test field count consistency
        if len(EXTRACTION_FIELDS) == schema.total_fields:
            print(f"✅ Field counts match: {len(EXTRACTION_FIELDS)}")
        else:
            print(f"❌ Field count mismatch: config={len(EXTRACTION_FIELDS)}, schema={schema.total_fields}")
            return False
        
        # Test field types consistency
        schema_types = schema.generate_field_types_mapping()
        if FIELD_TYPES == schema_types:
            print("✅ Field types match between config and schema")
        else:
            print("❌ Field types mismatch between config and schema")
            return False
            
        # Test field descriptions consistency  
        schema_descriptions = schema.generate_field_descriptions_mapping()
        if FIELD_DESCRIPTIONS == schema_descriptions:
            print("✅ Field descriptions match between config and schema")
        else:
            print("❌ Field descriptions mismatch between config and schema")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Schema-config consistency test failed: {e}")
        return False

def test_cognitive_grouping_generation():
    """Test that cognitive grouping is properly generated."""
    print("\n🧪 Testing Cognitive Grouping Generation...")
    
    try:
        from common.schema_loader import get_global_schema
        
        schema = get_global_schema()
        cognitive_strategy = schema.get_grouping_strategy("field_grouped")
        
        print(f"✅ Cognitive strategy generated with {len(cognitive_strategy['group_configs'])} groups")
        
        # Test specific cognitive group field assignments
        regulatory_group = cognitive_strategy["group_configs"].get("regulatory_financial")
        if regulatory_group:
            reg_fields = regulatory_group["fields"]
            print(f"✅ Regulatory financial group has {len(reg_fields)} fields")
            
            # Should contain critical + monetary fields
            if "BUSINESS_ABN" in reg_fields and "TOTAL_AMOUNT" in reg_fields:
                print("✅ Regulatory group contains expected critical fields")
            else:
                print("❌ Regulatory group missing expected critical fields")
                return False
                
        else:
            print("❌ Regulatory financial group not generated")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Cognitive grouping test failed: {e}")
        return False

def test_validation_rules_dynamic():
    """Test that validation rules are loaded from schema."""
    print("\n🧪 Testing Dynamic Validation Rules...")
    
    try:
        from common.config import GROUP_VALIDATION_RULES
        
        print(f"✅ Validation rules loaded for {len(GROUP_VALIDATION_RULES)} groups")
        
        # Check that validation rules exist for key groups
        if "critical" in GROUP_VALIDATION_RULES:
            print("✅ Critical group validation rules loaded")
        else:
            print("❌ Critical group validation rules missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Validation rules test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Phase 2 Migration Test Suite")
    print("=" * 60)
    print("Testing that all hardcoded configurations have been replaced with schema-driven dynamic generation")
    
    tests = [
        test_dynamic_field_groups,
        test_dynamic_field_types,
        test_schema_field_consistency,
        test_cognitive_grouping_generation,
        test_validation_rules_dynamic
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 Phase 2 migration successful!")
        print("✨ All hardcoded configurations replaced with schema-driven generation")
        print("💡 Ready for Phase 3: Dynamic prompt generation")
    else:
        print("❌ Some tests failed. Check errors above.")
        print("🔧 Fix issues before proceeding to Phase 3")