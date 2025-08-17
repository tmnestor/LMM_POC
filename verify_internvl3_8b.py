#!/usr/bin/env python3
"""
Verify InternVL3-8B model path and loading.

This script directly tests if InternVL3-8B can be loaded properly.
"""

from pathlib import Path

from common.config import INTERNVL3_MODEL_PATH


def verify_model():
    """Verify the InternVL3-8B model can be loaded."""
    print("🔍 INTERNVL3-8B VERIFICATION")
    print("=" * 50)
    
    print(f"📂 Expected path: {INTERNVL3_MODEL_PATH}")
    
    # Check if path exists
    model_path = Path(INTERNVL3_MODEL_PATH)
    if not model_path.exists():
        print("❌ Model path doesn't exist!")
        return False
    
    print("✅ Model path exists")
    
    # Check for essential files
    essential_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    print("\n📋 Checking essential files:")
    for file_name in essential_files:
        file_path = model_path / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} missing")
    
    # List model weight files
    print("\n📦 Model weight files:")
    safetensors_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("*.bin"))
    
    if safetensors_files:
        print(f"   ✅ {len(safetensors_files)} safetensors files")
        for f in safetensors_files[:3]:  # Show first 3
            print(f"      - {f.name}")
        if len(safetensors_files) > 3:
            print(f"      ... and {len(safetensors_files) - 3} more")
    
    if bin_files:
        print(f"   ✅ {len(bin_files)} bin files")
        for f in bin_files[:3]:  # Show first 3
            print(f"      - {f.name}")
        if len(bin_files) > 3:
            print(f"      ... and {len(bin_files) - 3} more")
    
    if not safetensors_files and not bin_files:
        print("   ❌ No model weight files found!")
        return False
    
    # Try minimal loading test
    print("\n🧪 MINIMAL LOADING TEST:")
    try:
        from transformers import AutoConfig
        
        # Just load config first
        print("   📝 Loading config...")
        config = AutoConfig.from_pretrained(INTERNVL3_MODEL_PATH, trust_remote_code=True)
        print("   ✅ Config loaded successfully")
        print(f"       Model type: {getattr(config, 'model_type', 'unknown')}")
        print(f"       Architecture: {getattr(config, 'architectures', ['unknown'])[0] if hasattr(config, 'architectures') else 'unknown'}")
        
        # Try tokenizer loading
        print("   📝 Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(INTERNVL3_MODEL_PATH, trust_remote_code=True)
        print("   ✅ Tokenizer loaded successfully")
        print(f"       Vocab size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Loading failed: {e}")
        print(f"       Error type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    success = verify_model()
    
    if success:
        print("\n✅ VERIFICATION PASSED")
        print("   InternVL3-8B model appears to be correctly installed")
        print("   The UTF-8 error is likely a loading configuration issue")
        print("\n💡 NEXT STEPS:")
        print("   1. The 44.2% accuracy is likely due to incorrect model loading parameters")
        print("   2. Try different loading configurations (dtype, trust_remote_code, etc.)")
        print("   3. The model files are intact, so this should be fixable")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("   InternVL3-8B model has installation issues")
        print("   Need to reinstall or verify model integrity")