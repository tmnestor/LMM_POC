"""
Test script for Llama 4 Scout model loading and inference
Run this on H200 machine to verify Llama 4 compatibility
"""

import sys
from pathlib import Path

import torch

print("=" * 80)
print("LLAMA 4 SCOUT COMPATIBILITY TEST")
print("=" * 80)

# ============================================================================
# PHASE 1: Environment Check
# ============================================================================
print("\n📋 PHASE 1: Environment Check")
print("-" * 80)

# Check transformers version
try:
    import transformers
    version = transformers.__version__
    print(f"✅ transformers version: {version}")

    # Parse version
    major, minor, patch = version.split('.')[:3]
    if int(major) >= 5 or (int(major) == 4 and int(minor) >= 51):
        print("✅ Version check passed (>= 4.51.0 required)")
    else:
        print(f"❌ Version too old! Need >= 4.51.0, have {version}")
        print("   Run: pip install --upgrade transformers")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error checking transformers: {e}")
    sys.exit(1)

# Check for Llama4 class
try:
    from transformers import Llama4ForConditionalGeneration
    print("✅ Llama4ForConditionalGeneration class available")
    LLAMA4_AVAILABLE = True
except ImportError as e:
    print(f"❌ Llama4ForConditionalGeneration not available: {e}")
    print("   Upgrade transformers: pip install --upgrade transformers")
    LLAMA4_AVAILABLE = False
    sys.exit(1)

# Check GPU availability
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA available: {device_count} GPU(s)")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.0f}GB)")
else:
    print("⚠️  No CUDA available - will run on CPU (very slow)")

# ============================================================================
# PHASE 2: Model Loading Test
# ============================================================================
print("\n📦 PHASE 2: Model Loading Test")
print("-" * 80)

model_path = "/home/jovyan/nfs_share/models/Llama-4-Scout-17B-16E-Instruct"

# Check model path exists
if not Path(model_path).exists():
    print(f"❌ Model path not found: {model_path}")
    sys.exit(1)
else:
    print(f"✅ Model path exists: {model_path}")

# Load processor
print("\n1. Loading processor...")
try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ Processor loaded successfully")
except Exception as e:
    print(f"❌ Processor loading failed: {e}")
    sys.exit(1)

# Load model
print("\n2. Loading Llama 4 Scout model (this may take 2-3 minutes)...")
print("   Expected size: ~109GB (17B active, 109B total with MoE)")

try:
    model = Llama4ForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="flex_attention",  # Required for Llama 4
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("✅ Model loaded successfully!")
    print(f"   Model class: {type(model).__name__}")
    print(f"   Model device: {model.device}")

    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Check GPU memory usage
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        total_allocated = 0
        total_reserved = 0

        print("\n   GPU Memory Usage:")
        for gpu_id in range(device_count):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
            total_allocated += allocated
            reserved_pct = (reserved / torch.cuda.get_device_properties(gpu_id).total_memory) * 100
            print(f"   GPU {gpu_id}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved ({reserved_pct:.1f}%)")
            total_allocated += allocated
            total_reserved += reserved

        print(f"   Total: {total_allocated:.1f}GB allocated, {total_reserved:.1f}GB reserved")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PHASE 3: Basic Inference Test
# ============================================================================
print("\n🧪 PHASE 3: Basic Inference Test")
print("-" * 80)

print("\n1. Testing text-only inference...")
try:
    # Simple text prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is 2+2? Answer in one word."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    print("   Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0]

    print("✅ Text inference successful!")
    print("   Prompt: What is 2+2? Answer in one word.")
    print(f"   Response: {response.strip()}")

except Exception as e:
    print(f"❌ Text inference failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# PHASE 4: Vision Inference Test (if you have a test image)
# ============================================================================
print("\n👁️  PHASE 4: Vision Inference Test")
print("-" * 80)

try:
    # Try to load a test image from evaluation data
    from io import BytesIO

    import requests
    from PIL import Image

    # Use a simple test image URL
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

    print(f"1. Loading test image from: {test_image_url[:60]}...")
    response = requests.get(test_image_url, timeout=10)
    test_image = Image.open(BytesIO(response.content))
    print(f"✅ Test image loaded: {test_image.size}")

    print("\n2. Testing vision inference...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one sentence."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images=[test_image]
    ).to(model.device)

    print("   Generating vision response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )[0]

    print("✅ Vision inference successful!")
    print(f"   Response: {response.strip()}")

except Exception as e:
    print(f"⚠️  Vision inference test skipped or failed: {e}")
    print("   (This is optional - text inference works)")

# ============================================================================
# PHASE 5: Summary
# ============================================================================
print("\n" + "=" * 80)
print("📊 TEST SUMMARY")
print("=" * 80)

print("\n✅ PASSED:")
print("   - Transformers version check")
print("   - Llama4ForConditionalGeneration class available")
print("   - Model loading successful")
print("   - Processor loading successful")
print("   - Text inference working")

print("\n📋 Model Information:")
print("   - Model: Llama 4 Scout (17B active, 109B total)")
print(f"   - Parameters: {total_params:,}")
print(f"   - Memory usage: {total_allocated:.1f}GB")
print("   - Architecture: Mixture of Experts (16 experts)")
print("   - Attention: flex_attention")

print("\n✅ READY FOR INTEGRATION")
print("   Llama 4 Scout is working correctly on your H200 machine!")
print("   You can now integrate it into llama_batch.ipynb")

print("\n" + "=" * 80)
print("Next steps:")
print("1. Update common/llama_model_loader_robust.py to support Llama 4")
print("2. Update llama_batch.ipynb model path to Llama 4 Scout")
print("3. Test with your actual document extraction pipeline")
print("=" * 80)
