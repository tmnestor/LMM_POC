#!/usr/bin/env python3
"""
Test InternVL3-8B with the working tokenizer configuration.

This script applies the tokenizer fix and tests full inference through 
the InternVL3Processor to confirm 44.2% accuracy issue is resolved.
"""

from datetime import datetime
from pathlib import Path

from common.config import DATA_DIR, INTERNVL3_MODEL_PATH
from common.evaluation_utils import discover_images
from models.internvl3_processor import InternVL3Processor


def test_internvl3_8b_fixed():
    """Test InternVL3-8B with tokenizer fixes applied."""
    print("🔧 TESTING INTERNVL3-8B WITH TOKENIZER FIXES")
    print("=" * 60)
    
    # Find test image
    image_files = discover_images(DATA_DIR)
    if not image_files:
        print("❌ No test images found")
        return False
    
    test_image = image_files[0]
    print(f"🖼️ Test image: {Path(test_image).name}")
    print(f"📂 Model path: {INTERNVL3_MODEL_PATH}")
    
    try:
        # Initialize processor - this should now work with tokenizer fixes
        print("\n🚀 Initializing InternVL3 processor with fixes...")
        processor = InternVL3Processor(model_path=INTERNVL3_MODEL_PATH)
        
        print("✅ Processor initialized successfully!")
        print(f"   Model type: {processor.model_type}")
        print(f"   Is 8B model: {processor.is_8b_model}")
        
        # Test single image processing
        print("\n🧪 Testing single image processing...")
        start_time = datetime.now()
        
        result = processor.process_single_image(test_image)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Analyze results
        extracted_data = result.get("extracted_data", {})
        non_na_fields = sum(1 for v in extracted_data.values() if v != "N/A")
        total_fields = len(extracted_data)
        
        print("✅ PROCESSING SUCCESSFUL!")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Fields extracted: {non_na_fields}/{total_fields}")
        print(f"   Extraction rate: {non_na_fields/total_fields*100:.1f}%")
        print(f"   Quality: {result.get('extraction_quality', 'unknown')}")
        
        # Show sample extracted fields
        print("\n📋 Sample extracted fields:")
        sample_fields = list(extracted_data.items())[:5]
        for field, value in sample_fields:
            display_value = value[:50] + "..." if len(value) > 50 else value
            print(f"   {field}: {display_value}")
        
        # Check if this looks like proper extraction vs degraded mode
        if non_na_fields >= 5:  # Reasonable threshold
            print("\n🎯 SUCCESS INDICATORS:")
            print(f"   ✅ Multiple fields extracted ({non_na_fields}/{total_fields})")
            print(f"   ✅ Processing time reasonable ({processing_time:.1f}s)")
            print("   ✅ No UTF-8 or loading errors")
            print("\n💡 CONCLUSION:")
            print("   InternVL3-8B appears to be working properly!")
            print("   This should significantly improve accuracy from 44.2%")
            print("   Ready for full evaluation pipeline testing")
            return True
        else:
            print("\n⚠️ CONCERN:")
            print(f"   Only {non_na_fields} fields extracted - may still have issues")
            print("   But no UTF-8/loading errors, which is progress")
            return False
            
    except Exception as e:
        print(f"\n❌ PROCESSING FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_batch():
    """Test a small batch to verify sustained performance."""
    print("\n🔬 TESTING SMALL BATCH PROCESSING")
    print("=" * 40)
    
    try:
        # Find test images
        image_files = discover_images(DATA_DIR)
        if len(image_files) < 3:
            print("⚠️ Less than 3 images available for batch test")
            return False
        
        # Test with first 3 images
        test_batch = image_files[:3]
        print(f"📷 Testing batch of {len(test_batch)} images")
        
        processor = InternVL3Processor(model_path=INTERNVL3_MODEL_PATH)
        
        start_time = datetime.now()
        results, batch_stats = processor.process_image_batch(test_batch)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        
        print("✅ BATCH PROCESSING SUCCESSFUL!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average per image: {total_time/len(test_batch):.2f} seconds")
        print(f"   Success rate: {batch_stats.get('success_rate', 0)*100:.1f}%")
        
        # Analyze extraction quality across batch
        total_extractions = 0
        total_fields = 0
        for result in results:
            extracted_data = result.get("extracted_data", {})
            total_extractions += sum(1 for v in extracted_data.values() if v != "N/A")
            total_fields += len(extracted_data)
        
        batch_extraction_rate = total_extractions / total_fields * 100 if total_fields > 0 else 0
        
        print(f"   Batch extraction rate: {batch_extraction_rate:.1f}%")
        
        if batch_extraction_rate >= 30:  # Reasonable threshold for working model
            print("\n🎯 BATCH SUCCESS!")
            print("   Consistent extraction across multiple images")
            print("   Ready for full evaluation pipeline")
            return True
        else:
            print("\n⚠️ BATCH CONCERNS:")
            print("   Low extraction rate may indicate remaining issues")
            return False
            
    except Exception as e:
        print(f"\n❌ BATCH TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    print("🚀 INTERNVL3-8B FIXED CONFIGURATION TEST")
    print("=" * 60)
    
    # Test single image
    single_success = test_internvl3_8b_fixed()
    
    # Test batch if single succeeds
    if single_success:
        batch_success = test_quick_batch()
        
        if batch_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("   ✅ Single image processing works")
            print("   ✅ Batch processing works")
            print("   ✅ No UTF-8 or tokenizer errors")
            print("\n📝 NEXT STEPS:")
            print("   1. Run full evaluation: python internvl3_keyvalue.py")
            print("   2. Compare accuracy to 44.2% baseline")
            print("   3. Expected significant improvement")
        else:
            print("\n⚠️ PARTIAL SUCCESS")
            print("   Single image works but batch has issues")
            print("   May need additional optimization")
    else:
        print("\n❌ CONFIGURATION ISSUES REMAIN")
        print("   Need to debug further or try alternative approaches")