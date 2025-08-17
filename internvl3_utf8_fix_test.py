#!/usr/bin/env python3
"""
InternVL3-8B UTF-8 Error Diagnostic and Fix Test

This script tests different model loading approaches to resolve the 
"stream did not contain valid UTF-8" error affecting InternVL3-8B.

Usage: python internvl3_utf8_fix_test.py
"""

from datetime import datetime
from pathlib import Path

import torch

from common.config import DATA_DIR as data_dir
from common.config import INTERNVL3_MODEL_PATH as model_path
from common.evaluation_utils import discover_images


def test_utf8_safe_loading(config_name: str, model_kwargs: dict, tokenizer_kwargs: dict, test_image_path: str):
    """Test model loading with UTF-8 safe configurations."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING UTF-8 SAFE CONFIG: {config_name}")
    print(f"{'='*60}")
    print(f"Model kwargs: {model_kwargs}")
    print(f"Tokenizer kwargs: {tokenizer_kwargs}")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Load tokenizer first with UTF-8 safety
        print("📝 Loading tokenizer with UTF-8 safety...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            **tokenizer_kwargs
        )
        print("✅ Tokenizer loaded successfully")
        
        # Load model with UTF-8 safe configuration
        print("📦 Loading model with UTF-8 safety...")
        model = AutoModel.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        print("✅ Model loaded successfully")
        print(f"🔧 Model device: {next(model.parameters()).device}")
        print(f"🔧 Model dtype: {next(model.parameters()).dtype}")
        
        # Test basic inference
        print(f"🖼️ Testing inference on: {Path(test_image_path).name}")
        
        # Simple test question
        question = "What type of document is this?"
        
        # Load and process image
        from PIL import Image
        image = Image.open(test_image_path).convert('RGB')
        pixel_values = model.build_transform()(image).unsqueeze(0)
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate response with UTF-8 safe configuration
        start_time = datetime.now()
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config={
                "max_new_tokens": 100,  # Short response for quick test
                "temperature": 0.1,
                "do_sample": True,
                "use_cache": True,
            },
            history=None,
            return_history=False,
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📝 Response length: {len(response)} characters")
        print(f"📝 Response preview: {response[:100]}...")
        
        # Check if response contains actual content
        response_quality = "good" if len(response) > 10 and "error" not in response.lower() else "poor"
        
        result = {
            "config_name": config_name,
            "processing_time": processing_time,
            "response_length": len(response),
            "response_preview": response[:200],
            "response_quality": response_quality,
            "model_dtype": str(next(model.parameters()).dtype),
            "model_device": str(next(model.parameters()).device),
            "success": True
        }
        
        print(f"✅ Test successful - Response quality: {response_quality}")
        
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        torch.cuda.empty_cache()
        return {
            "config_name": config_name,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def main():
    """Run UTF-8 safe loading tests."""
    print("🔬 InternVL3-8B UTF-8 Error Fix Diagnostic")
    print("=" * 60)
    
    # Find test image
    image_files = discover_images(data_dir)
    if not image_files:
        print("❌ No test images found")
        return
    
    test_image = image_files[0]
    print(f"🖼️ Test image: {Path(test_image).name}")
    print(f"📁 Model path: {model_path}")
    
    # UTF-8 safe configuration tests
    test_configs = [
        {
            "name": "1_Explicit_UTF8_Encoding",
            "model_kwargs": {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_safetensors": True,  # Force safetensors loading
            },
            "tokenizer_kwargs": {
                "trust_remote_code": True,
                "use_fast": False,  # Disable fast tokenizer
                "clean_up_tokenization_spaces": True,
            }
        },
        {
            "name": "2_Safe_Tensors_Only",
            "model_kwargs": {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_safetensors": True,
                "force_download": False,
            },
            "tokenizer_kwargs": {
                "trust_remote_code": True,
                "use_fast": True,  # Try fast tokenizer
                "padding_side": "right",
            }
        },
        {
            "name": "3_No_Trust_Remote_Code",
            "model_kwargs": {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": False,  # Disable custom code
            },
            "tokenizer_kwargs": {
                "trust_remote_code": False,
                "use_fast": False,
            }
        },
        {
            "name": "4_Legacy_Loading_Mode",
            "model_kwargs": {
                "torch_dtype": torch.float16,  # Different dtype
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_safetensors": False,  # Force pickle loading
            },
            "tokenizer_kwargs": {
                "trust_remote_code": True,
                "use_fast": False,
                "legacy": True,
            }
        },
        {
            "name": "5_Minimal_Configuration",
            "model_kwargs": {
                "trust_remote_code": True,
                # Minimal kwargs to avoid conflicts
            },
            "tokenizer_kwargs": {
                "trust_remote_code": True,
                # Minimal kwargs
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = test_utf8_safe_loading(
                config["name"], 
                config["model_kwargs"], 
                config["tokenizer_kwargs"],
                test_image
            )
            results.append(result)
            
            # Brief pause between tests
            import time
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Test {config['name']} failed completely: {e}")
            results.append({
                "config_name": config["name"],
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    # Summary report
    print("\n" + "=" * 80)
    print("📊 UTF-8 FIX TEST SUMMARY")
    print("=" * 80)
    
    successful_configs = []
    for result in results:
        if result["success"]:
            successful_configs.append(result)
            print(f"✅ {result['config_name']:<30} | "
                  f"Quality: {result['response_quality']:<6} | "
                  f"Length: {result['response_length']:<4} | "
                  f"Time: {result['processing_time']:.1f}s")
        else:
            error_type = result.get('error_type', 'Unknown')
            print(f"❌ {result['config_name']:<30} | "
                  f"FAILED: {error_type} - {result.get('error', 'Unknown error')[:50]}...")
    
    print("\n🎯 ANALYSIS:")
    
    if successful_configs:
        print(f"🏆 {len(successful_configs)} configurations worked successfully!")
        best_config = max(successful_configs, key=lambda x: x.get("response_length", 0))
        print(f"   Best configuration: {best_config['config_name']}")
        print(f"   Response length: {best_config['response_length']} characters")
        print(f"   Processing time: {best_config['processing_time']:.1f}s")
        
        print("\n✅ SOLUTION FOUND:")
        print("   - UTF-8 error can be resolved with proper model loading configuration")
        print(f"   - Working config: {best_config['config_name']}")
        print("   - This should restore normal InternVL3-8B accuracy")
        
    else:
        print("❌ ALL CONFIGURATIONS FAILED")
        print("   This suggests a deeper issue with the InternVL3-8B model files")
        print("   Possible causes:")
        print("   1. Model files are corrupted or incomplete")
        print("   2. Wrong model variant (need InternVL3-2B path)")
        print("   3. Environment/dependency issues beyond UTF-8")
        print("   4. Hardware-specific incompatibility")
    
    print("\n📝 Next steps:")
    if successful_configs:
        print("1. ✅ Apply working configuration to main InternVL3 processor")
        print("2. 🧪 Test full evaluation pipeline with fixed configuration")
        print("3. 📊 Compare accuracy before/after UTF-8 fix")
    else:
        print("1. 🔍 Check if InternVL3-2B model path works")
        print("2. 🔍 Verify model files integrity and completeness")
        print("3. 🔍 Test with different model variants or sources")


if __name__ == "__main__":
    main()