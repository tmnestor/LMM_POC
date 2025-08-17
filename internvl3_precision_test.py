#!/usr/bin/env python3
"""
InternVL3-8B V100 Precision Diagnostic Test

This script tests different precision and quantization configurations 
to diagnose the 44.2% accuracy issue on V100 hardware.

Usage: python internvl3_precision_test.py
"""

from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from common.config import DATA_DIR as data_dir
from common.config import INTERNVL3_MODEL_PATH as model_path
from common.evaluation_utils import discover_images, parse_extraction_response


def test_model_configuration(config_name: str, model_kwargs: dict, test_image_path: str):
    """Test a specific model configuration and return accuracy metrics."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING CONFIGURATION: {config_name}")
    print(f"{'='*60}")
    print(f"Model kwargs: {model_kwargs}")
    
    try:
        # Load model with specific configuration
        print("📦 Loading model...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully")
        print(f"🔧 Model device: {next(model.parameters()).device}")
        print(f"🔧 Model dtype: {next(model.parameters()).dtype}")
        
        # Test inference on single image
        print(f"🖼️ Testing inference on: {Path(test_image_path).name}")
        
        # Simple extraction test
        question = "Extract the following information from this document: INVOICE_NUMBER, TOTAL, ABN. Return in JSON format."
        
        # Load and process image
        from PIL import Image
        image = Image.open(test_image_path).convert('RGB')
        pixel_values = model.build_transform()(image).unsqueeze(0)
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate response
        start_time = datetime.now()
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config={
                "max_new_tokens": 600,
                "temperature": 0.1,
                "do_sample": True,
                "top_p": 0.95,
                "use_cache": True,
            },
            history=None,
            return_history=False,
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📝 Raw response: {response[:200]}...")
        
        # Parse response for quality assessment
        extracted_data = parse_extraction_response(response)
        non_na_fields = sum(1 for v in extracted_data.values() if v != "N/A")
        
        result = {
            "config_name": config_name,
            "processing_time": processing_time,
            "raw_response": response,
            "extracted_fields": non_na_fields,
            "extraction_quality": "good" if non_na_fields >= 3 else "poor" if non_na_fields >= 1 else "failed",
            "model_dtype": str(next(model.parameters()).dtype),
            "model_device": str(next(model.parameters()).device),
            "success": True
        }
        
        print(f"✅ Extraction quality: {result['extraction_quality']} ({non_na_fields} fields)")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        torch.cuda.empty_cache()
        return {
            "config_name": config_name,
            "success": False,
            "error": str(e)
        }


def main():
    """Run precision diagnostic tests."""
    print("🔬 InternVL3-8B V100 Precision Diagnostic")
    print("=" * 60)
    
    # Find test image
    image_files = discover_images(data_dir)
    if not image_files:
        print("❌ No test images found")
        return
    
    test_image = image_files[0]  # Use first image for consistent testing
    print(f"🖼️ Test image: {Path(test_image).name}")
    
    # Configuration tests to run
    test_configs = [
        {
            "name": "1_Current_bfloat16_8bit",
            "kwargs": {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "use_flash_attn": False,
                "load_in_8bit": True,  # Current problematic setting
            }
        },
        {
            "name": "2_No_Quantization_bfloat16", 
            "kwargs": {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "use_flash_attn": False,
                # NO quantization
            }
        },
        {
            "name": "3_Float16_No_Quantization",
            "kwargs": {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "use_flash_attn": False,
                # NO quantization
            }
        },
        {
            "name": "4_Float32_CPU_Baseline",
            "kwargs": {
                "torch_dtype": torch.float32,
                "device_map": "cpu",  # CPU baseline for comparison
                "low_cpu_mem_usage": True,
            }
        },
        {
            "name": "5_Float32_GPU_Full_Precision",
            "kwargs": {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "use_flash_attn": False,
                # Full precision on GPU
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = test_model_configuration(
                config["name"], 
                config["kwargs"], 
                test_image
            )
            results.append(result)
            
            # Brief pause between tests
            import time
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Test {config['name']} failed completely: {e}")
            results.append({
                "config_name": config["name"],
                "success": False,
                "error": str(e)
            })
    
    # Summary report
    print("\n" + "=" * 80)
    print("📊 PRECISION TEST SUMMARY")
    print("=" * 80)
    
    for result in results:
        if result["success"]:
            print(f"✅ {result['config_name']:<25} | "
                  f"Quality: {result['extraction_quality']:<6} | "
                  f"Fields: {result['extracted_fields']:<2} | "
                  f"Time: {result['processing_time']:.1f}s | "
                  f"DType: {result['model_dtype']}")
        else:
            print(f"❌ {result['config_name']:<25} | FAILED: {result.get('error', 'Unknown error')}")
    
    print("\n🎯 ANALYSIS:")
    successful_configs = [r for r in results if r["success"]]
    
    if successful_configs:
        best_quality = max(successful_configs, key=lambda x: x.get("extracted_fields", 0))
        print(f"🏆 Best configuration: {best_quality['config_name']}")
        print(f"   - Extracted fields: {best_quality['extracted_fields']}")
        print(f"   - Processing time: {best_quality['processing_time']:.1f}s")
        print(f"   - Model dtype: {best_quality['model_dtype']}")
        
        # Check if removing quantization helps
        current_result = next((r for r in results if "Current" in r["config_name"] and r["success"]), None)
        no_quant_result = next((r for r in results if "No_Quantization_bfloat16" in r["config_name"] and r["success"]), None)
        
        if current_result and no_quant_result:
            improvement = no_quant_result["extracted_fields"] - current_result["extracted_fields"]
            if improvement > 0:
                print(f"🔍 CRITICAL: Removing 8-bit quantization improved extraction by {improvement} fields!")
                print("   → 8-bit quantization may be causing accuracy degradation on V100")
            elif improvement < 0:
                print(f"🔍 Quantization removal decreased performance by {abs(improvement)} fields")
            else:
                print("🔍 Quantization removal had no significant impact")
    
    print("\n📝 Next steps based on results:")
    print("1. If Float32 > Float16/bfloat16: V100 precision issue")
    print("2. If No_Quantization > 8bit: Quantization incompatibility") 
    print("3. If CPU > GPU: V100 hardware-specific problem")
    print("4. If all configs similar: Issue is elsewhere (preprocessing, attention, etc.)")


if __name__ == "__main__":
    main()