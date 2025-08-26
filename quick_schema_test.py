#!/usr/bin/env python3
"""
Quick Schema Test - Test fixed Phase 2 implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from common.document_schema_loader import DocumentTypeFieldSchema
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_schema_basics():
    """Test basic schema functionality after bug fixes."""
    print("🧪 Testing schema basics...")
    
    try:
        # Initialize loader
        schema_loader = DocumentTypeFieldSchema()
        print("✅ Schema loader initialized")
        
        # Test each document type
        for doc_type in schema_loader.get_supported_document_types():
            # Get schema
            schema = schema_loader.get_document_schema(doc_type)
            field_count = schema['total_fields']
            
            # Test comparison (this was failing before)
            comparison = schema_loader.compare_schemas(doc_type)
            
            print(f"✅ {doc_type.upper()}:")
            print(f"   Fields: {field_count}")
            print(f"   Efficiency: {comparison['efficiency_gain']}")
            print(f"   Reduction: {comparison['field_reduction']} fields")
            
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_field_counts():
    """Verify field counts are correct."""
    print("\n📊 Testing field counts...")
    
    expected_counts = {
        "invoice": 19,      # 6 common + 13 specific
        "bank_statement": 15,   # 6 common + 9 specific  
        "receipt": 15       # 6 common + 9 specific
    }
    
    try:
        schema_loader = DocumentTypeFieldSchema()
        
        for doc_type, expected_count in expected_counts.items():
            actual_count = schema_loader.get_field_count_for_type(doc_type)
            
            if actual_count == expected_count:
                print(f"✅ {doc_type}: {actual_count} fields (correct)")
            else:
                print(f"❌ {doc_type}: {actual_count} fields (expected {expected_count})")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Field count test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("🚀 Quick Schema Test - Bug Fix Verification")
    print("=" * 50)
    
    # Test 1: Basic functionality
    test1 = test_schema_basics()
    
    # Test 2: Field counts
    test2 = test_field_counts()
    
    # Results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Schema Basics: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Field Counts: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 All tests passed! Bug fixes successful.")
        print("💡 You can now run: python test_document_schema_v2.py --model llama")
    else:
        print("\n🚨 Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()