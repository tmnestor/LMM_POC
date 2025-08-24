#!/usr/bin/env python3
"""
Test script for Phase 1 schema loader functionality.
Run this on remote machine to validate the single source of truth migration.
"""

def test_schema_loading():
    """Test basic schema loading functionality."""
    print("🧪 Testing Schema Loader...")
    
    try:
        from common.schema_loader import FieldSchema
        schema = FieldSchema()
        
        print('✅ Schema loaded successfully!')
        print(f'📊 Total fields: {schema.total_fields}')
        print(f'🔤 First 5 fields: {schema.field_names[:5]}')
        print(f'🔤 Last 5 fields: {schema.field_names[-5:]}')
        print(f'💰 Monetary fields: {schema.get_fields_by_type("monetary")}')
        print(f'🏦 Banking fields: {schema.get_fields_by_group("banking")}')
        return True
        
    except Exception as e:
        print(f'❌ Schema loading failed: {e}')
        return False

def test_config_integration():
    """Test that config.py works with new schema."""
    print("\n🧪 Testing Config Integration...")
    
    try:
        from common.config import EXTRACTION_FIELDS, FIELD_COUNT
        
        print('✅ Config loads with schema!')
        print(f'📊 Field count: {FIELD_COUNT}')
        print(f'🔤 First field: {EXTRACTION_FIELDS[0]}')
        print(f'🔤 Last field: {EXTRACTION_FIELDS[-1]}')
        print(f'🎯 Semantic ordering preserved: {EXTRACTION_FIELDS[0]} → {EXTRACTION_FIELDS[-1]}')
        
        # Verify critical fields are present
        expected_critical = ["BUSINESS_ABN", "TOTAL_AMOUNT"]
        for field in expected_critical:
            if field in EXTRACTION_FIELDS:
                print(f'✅ Critical field found: {field}')
            else:
                print(f'❌ Missing critical field: {field}')
                return False
                
        return True
        
    except Exception as e:
        print(f'❌ Config integration failed: {e}')
        return False

def test_field_groups():
    """Test dynamic field group generation."""
    print("\n🧪 Testing Field Groups...")
    
    try:
        from common.config import FIELD_GROUPS_DETAILED
        
        print('✅ Field groups loaded!')
        print(f'📊 Number of groups: {len(FIELD_GROUPS_DETAILED)}')
        
        # Test specific groups
        critical_group = FIELD_GROUPS_DETAILED.get("critical", {})
        if critical_group:
            print(f'✅ Critical group: {critical_group.get("fields", [])}')
        else:
            print('❌ Missing critical group')
            return False
            
        return True
        
    except Exception as e:
        print(f'❌ Field groups test failed: {e}')
        return False

if __name__ == "__main__":
    print("🚀 Phase 1 Schema Migration Test Suite")
    print("=" * 50)
    
    tests = [
        test_schema_loading,
        test_config_integration, 
        test_field_groups
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Phase 1 migration successful!")
    else:
        print("❌ Some tests failed. Check errors above.")