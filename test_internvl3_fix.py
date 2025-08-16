#!/usr/bin/env python3
"""
Test script for InternVL3-8B ResilientGenerator fixes.

This script will test the fixes for the embedding indices issue by:
1. Loading the InternVL3-8B model 
2. Creating a small test image
3. Testing both ResilientGenerator and direct chat methods
4. Validating that the tokenizer works correctly

Run this on the H200 development machine to verify the fixes.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from models.internvl3_processor import InternVL3Processor
from common.config import INTERNVL3_MODEL_PATH

def create_test_image():
    """Create a simple test image for validation."""
    # Create a simple white image with some text-like patterns
    img_array = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Add some simple patterns that might look like text/numbers
    img_array[50:100, 50:150] = 0  # Black rectangle
    img_array[120:140, 60:140] = 128  # Gray rectangle
    
    return Image.fromarray(img_array)

def test_tokenizer(processor):
    """Test that the tokenizer is working correctly."""
    print("\n🧪 Testing tokenizer functionality...")
    
    try:
        # Test basic tokenization
        test_text = "Extract data from this image"
        tokens = processor.tokenizer(test_text, return_tensors="pt")
        
        print(f"✅ Tokenizer working: {tokens['input_ids'].shape}")
        
        # Test with longer text similar to our prompt
        long_text = processor.get_extraction_prompt()
        long_tokens = processor.tokenizer(long_text, return_tensors="pt")
        
        print(f"✅ Long prompt tokenization: {long_tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_direct_chat(processor, test_image):
    """Test direct chat method without ResilientGenerator."""
    print("\n🧪 Testing direct chat method...")
    
    try:
        # Load image through processor
        pixel_values = processor.load_image(test_image, input_size=224, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # Create simple question
        question = "<image>\nWhat do you see in this image?"
        
        # Simple generation config
        gen_config = {
            "max_new_tokens": 50,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.eos_token_id,
        }
        
        # Call chat directly
        response = processor.model.chat(
            processor.tokenizer,
            pixel_values,
            question,
            gen_config,
            history=None,
            return_history=False,
        )
        
        print(f"✅ Direct chat successful: {len(response)} chars")
        print(f"   Response preview: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct chat test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resilient_generator(processor, test_image):
    """Test ResilientGenerator method."""
    print("\n🧪 Testing ResilientGenerator method...")
    
    try:
        # Process single image using the processor's method
        # This will use ResilientGenerator if it's an 8B model
        result = processor.process_single_image("test_image.png")  # Dummy path
        
        if result and result.get("raw_response"):
            print(f"✅ ResilientGenerator successful: {len(result['raw_response'])} chars")
            print(f"   Response preview: {result['raw_response'][:100]}...")
            return True
        else:
            print("❌ ResilientGenerator returned empty result")
            return False
        
    except Exception as e:
        print(f"❌ ResilientGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔬 InternVL3-8B ResilientGenerator Fix Test")
    print("=" * 50)
    
    # Check if this is an 8B model path
    if "8B" not in str(INTERNVL3_MODEL_PATH):
        print("⚠️ Warning: Model path doesn't contain '8B', fixes may not be tested")
    
    print(f"📁 Model path: {INTERNVL3_MODEL_PATH}")
    
    try:
        # Initialize processor
        print("\n🚀 Loading InternVL3 processor...")
        processor = InternVL3Processor(model_path=INTERNVL3_MODEL_PATH)
        
        print(f"🎯 Model type: {'8B' if processor.is_8b_model else '2B'}")
        print(f"🤖 ResilientGenerator: {'Enabled' if processor.resilient_generator else 'Disabled'}")
        
        # Create test image
        test_image = create_test_image()
        print("🖼️ Test image created")
        
        # Run tests
        tests_passed = 0
        total_tests = 3
        
        if test_tokenizer(processor):
            tests_passed += 1
            
        if test_direct_chat(processor, test_image):
            tests_passed += 1
            
        if test_resilient_generator(processor, test_image):
            tests_passed += 1
        
        # Summary
        print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print("✅ All tests passed! The fixes appear to be working.")
        elif tests_passed > 0:
            print("⚠️ Some tests passed. Partial functionality restored.")
        else:
            print("❌ All tests failed. Further investigation needed.")
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()