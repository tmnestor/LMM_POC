#!/usr/bin/env python3
"""
Fix InternVL3-8B tokenizer UTF-8 loading issue.

This script tests different tokenizer configurations to resolve the UTF-8 error
that's preventing proper InternVL3-8B loading and causing 44.2% accuracy.
"""


import torch

from common.config import INTERNVL3_MODEL_PATH


def test_tokenizer_configs():
    """Test different tokenizer loading configurations."""
    print("🔧 INTERNVL3-8B TOKENIZER FIX")
    print("=" * 50)
    
    print(f"📂 Model path: {INTERNVL3_MODEL_PATH}")
    
    # Test configurations for tokenizer loading
    tokenizer_configs = [
        {
            "name": "1_Disable_Fast_Tokenizer",
            "kwargs": {
                "trust_remote_code": True,
                "use_fast": False,  # Disable fast tokenizer
                "legacy": False,
            }
        },
        {
            "name": "2_Legacy_Mode",
            "kwargs": {
                "trust_remote_code": True,
                "use_fast": False,
                "legacy": True,  # Enable legacy mode
            }
        },
        {
            "name": "3_Force_Revision",
            "kwargs": {
                "trust_remote_code": True,
                "use_fast": False,
                "revision": "main",  # Force specific revision
            }
        },
        {
            "name": "4_Disable_Trust_Remote_Code",
            "kwargs": {
                "trust_remote_code": False,  # Try without custom code
                "use_fast": False,
            }
        },
        {
            "name": "5_Clean_Tokenization",
            "kwargs": {
                "trust_remote_code": True,
                "use_fast": False,
                "clean_up_tokenization_spaces": True,
                "strip_accents": False,
            }
        },
        {
            "name": "6_Manual_Loading",
            "kwargs": {
                "trust_remote_code": True,
                "use_fast": False,
                "from_tf": False,
                "from_flax": False,
                "force_download": False,
                "resume_download": True,
            }
        }
    ]
    
    working_configs = []
    
    for config in tokenizer_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   Config: {config['kwargs']}")
        
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                INTERNVL3_MODEL_PATH,
                **config['kwargs']
            )
            
            # Test basic tokenizer functionality
            test_text = "What type of document is this?"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            
            print("   ✅ SUCCESS!")
            print(f"      Vocab size: {tokenizer.vocab_size}")
            print(f"      Test encoding: {len(tokens)} tokens")
            print(f"      Decoded correctly: {decoded == test_text}")
            
            working_configs.append({
                "name": config['name'],
                "kwargs": config['kwargs'],
                "vocab_size": tokenizer.vocab_size,
                "test_passed": decoded == test_text
            })
            
            # Cleanup
            del tokenizer
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print(f"      Error type: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("📊 TOKENIZER FIX RESULTS")
    print("=" * 60)
    
    if working_configs:
        print(f"🎉 {len(working_configs)} configuration(s) worked!")
        
        for config in working_configs:
            print(f"✅ {config['name']}")
            print(f"   Vocab size: {config['vocab_size']}")
            print(f"   Test passed: {config['test_passed']}")
        
        # Recommend best config
        best_config = working_configs[0]  # First working config
        print("\n💡 RECOMMENDED CONFIGURATION:")
        print(f"   Use: {best_config['name']}")
        print(f"   Parameters: {best_config['kwargs']}")
        
        # Test with model loading
        print("\n🔬 TESTING FULL MODEL + TOKENIZER LOADING:")
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print("   Loading tokenizer with working config...")
            tokenizer = AutoTokenizer.from_pretrained(
                INTERNVL3_MODEL_PATH,
                **best_config['kwargs']
            )
            
            print("   Loading model...")
            model = AutoModel.from_pretrained(
                INTERNVL3_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            print("   ✅ FULL MODEL + TOKENIZER LOADED SUCCESSFULLY!")
            print(f"      Model device: {next(model.parameters()).device}")
            print(f"      Model dtype: {next(model.parameters()).dtype}")
            
            # Quick inference test
            print("   🧪 Testing basic inference...")
            question = "What is this?"
            
            # Create dummy image for test
            import numpy as np
            from PIL import Image
            
            # Create a small test image
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            pixel_values = model.build_transform()(test_image).unsqueeze(0)
            
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config={
                    "max_new_tokens": 50,
                    "temperature": 0.1,
                    "use_cache": True,
                },
                history=None,
                return_history=False,
            )
            
            print("   ✅ INFERENCE SUCCESSFUL!")
            print(f"      Response length: {len(response)} characters")
            print(f"      Response preview: {response[:100]}...")
            
            print("\n🎯 SOLUTION CONFIRMED:")
            print("   - InternVL3-8B can be loaded successfully with correct tokenizer config")
            print("   - This should fix the 44.2% accuracy issue")
            print("   - Apply this configuration to the main processor")
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Full model test failed: {e}")
            print("      Need to debug model loading separately")
    
    else:
        print("❌ NO WORKING CONFIGURATIONS FOUND")
        print("   InternVL3-8B tokenizer has serious corruption/compatibility issues")
        print("   May need to:")
        print("   1. Re-download the model")
        print("   2. Use InternVL3-2B instead")
        print("   3. Check for model version compatibility")
    
    return working_configs


if __name__ == "__main__":
    working_configs = test_tokenizer_configs()
    
    if working_configs:
        print("\n📝 NEXT STEPS:")
        print("1. Update InternVL3Processor to use working tokenizer configuration")
        print("2. Test full evaluation pipeline with fixed tokenizer")
        print("3. Confirm accuracy improves from 44.2% to expected levels")
        print("\n💻 Apply this config to models/internvl3_processor.py tokenizer loading")
    else:
        print("\n📝 ALTERNATIVE APPROACHES:")
        print("1. Try InternVL3-2B model (known working at 73.7%)")
        print("2. Re-download InternVL3-8B model")
        print("3. Check model version compatibility")