#!/usr/bin/env python3
"""
Diagnostic script to test Llama model loading and basic image processing.
This helps isolate whether tensor issues are from quantization or other factors.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


def test_core_single_image():
    """Test core single image processing without batch logic."""
    print("🧪 Testing core single image processing...")
    
    try:
        from common.evaluation_utils import discover_images
        from models.llama_processor import LlamaProcessor
        
        processor = LlamaProcessor(batch_size=1)
        image_files = discover_images('./evaluation_data')
        
        if image_files:
            print(f"Testing with: {image_files[0]}")
            try:
                result = processor.process_single_image(image_files[0])
                print(f"✅ Success: {result['extracted_fields_count']}/25 fields")
                return True
            except Exception as e:
                print(f"❌ Core single image processing failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ No test images found")
            return False
            
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_quantization():
    """Test model loading without quantization."""
    print("\n🧪 Testing without quantization...")
    
    model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
    
    try:
        # Load without quantization
        print("Loading model without quantization...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # No quantization
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Model loaded successfully without quantization")
        
        # Test with a single image
        from common.evaluation_utils import discover_images
        
        image_files = discover_images('./evaluation_data')
        if image_files:
            print(f"Testing basic image processing with: {Path(image_files[0]).name}")
            image = Image.open(image_files[0])
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see in this image?"}
                ]
            }]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(model.device)
            
            print("Generating response...")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            response = processor.decode(output[0], skip_special_tokens=True)
            print("✅ Basic image processing successful")
            print(f"Response length: {len(response)} characters")
            
            # Show a snippet of the response
            if "assistant" in response:
                assistant_response = response.split("assistant")[-1].strip()
                print(f"Assistant response snippet: {assistant_response[:200]}...")
            
            return True
            
        else:
            print("❌ No test images found")
            return False
    
    except Exception as e:
        print(f"❌ Test without quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_proper_quantization():
    """Test model loading with proper BitsAndBytesConfig."""
    print("\n🧪 Testing with proper quantization config...")
    
    model_path = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
    
    try:
        from transformers import BitsAndBytesConfig
        
        # Configure 8-bit quantization properly
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        
        print("Loading model with proper quantization config...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Model loaded successfully with proper quantization")
        
        # Test with a single image
        from common.evaluation_utils import discover_images
        
        image_files = discover_images('./evaluation_data')
        if image_files:
            print(f"Testing with quantized model: {Path(image_files[0]).name}")
            image = Image.open(image_files[0])
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see in this image?"}
                ]
            }]
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(model.device)
            
            print("Generating response with quantized model...")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            response = processor.decode(output[0], skip_special_tokens=True)
            print("✅ Quantized model processing successful")
            print(f"Response length: {len(response)} characters")
            
            return True
            
        else:
            print("❌ No test images found")
            return False
    
    except Exception as e:
        print(f"❌ Test with proper quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("=" * 80)
    print("🔍 LLAMA DIAGNOSTIC TESTS")
    print("=" * 80)
    
    # Test 1: Core single image processing with current setup
    success1 = test_core_single_image()
    
    # Test 2: Without quantization
    success2 = test_without_quantization()
    
    # Test 3: With proper quantization config
    success3 = test_with_proper_quantization()
    
    print("\n" + "=" * 80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Core single image processing: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Without quantization: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"With proper quantization: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    if success2 and not success3:
        print("\n💡 RECOMMENDATION: Quantization is causing the issue. Use without quantization.")
    elif success3:
        print("\n💡 RECOMMENDATION: Use the updated quantization configuration.")
    elif not success1 and not success2 and not success3:
        print("\n💡 RECOMMENDATION: There may be a deeper model or environment issue.")
    
    return success1 or success2 or success3

if __name__ == "__main__":
    main()