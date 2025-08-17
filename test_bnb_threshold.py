#!/usr/bin/env python3
"""
Test different bitsandbytes thresholds for 0.43.3 compatibility.

Usage:
    # Test default threshold (2.0)
    python test_bnb_threshold.py
    
    # Test specific threshold
    export BNB_THRESHOLD=1.0 && python test_bnb_threshold.py
    export BNB_THRESHOLD=0.5 && python test_bnb_threshold.py
    export BNB_THRESHOLD=0.0 && python test_bnb_threshold.py  # Maximum accuracy (if fits in memory)
"""

import os
import sys
from pathlib import Path

from common.config import DATA_DIR, INTERNVL3_MODEL_PATH
from common.evaluation_utils import discover_images
from models.internvl3_processor import InternVL3Processor


def test_threshold():
    """Test current threshold setting."""
    threshold = os.environ.get("BNB_THRESHOLD", "2.0")
    
    print("=" * 70)
    print(f"🧪 TESTING BITSANDBYTES 0.43.3 WITH THRESHOLD={threshold}")
    print("=" * 70)
    
    # Find test image
    image_files = discover_images(DATA_DIR)
    if not image_files:
        print("❌ No test images found")
        return 0
    
    test_image = image_files[0]
    print(f"📷 Test image: {Path(test_image).name}")
    
    try:
        # Check bitsandbytes version
        import bitsandbytes as bnb
        bnb_version = getattr(bnb, "__version__", "unknown")
        print(f"📦 bitsandbytes version: {bnb_version}")
        
        if not bnb_version.startswith("0.43"):
            print(f"⚠️  This test is designed for 0.43.x, you have {bnb_version}")
        
        # Initialize processor
        print(f"\n🚀 Loading InternVL3-8B with threshold={threshold}...")
        processor = InternVL3Processor(model_path=INTERNVL3_MODEL_PATH)
        
        # Process image
        print("\n🔄 Processing image...")
        result = processor.process_single_image(test_image)
        
        # Count extracted fields
        extracted_data = result.get("extracted_data", {})
        non_na_fields = sum(1 for v in extracted_data.values() if v != "N/A")
        total_fields = len(extracted_data)
        accuracy = non_na_fields / total_fields * 100
        
        print("\n" + "=" * 70)
        print(f"📊 RESULTS WITH THRESHOLD={threshold}:")
        print(f"   Fields extracted: {non_na_fields}/{total_fields}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        # Show first few extracted fields
        print("\n📋 Sample fields:")
        for i, (field, value) in enumerate(extracted_data.items()):
            if i >= 5:
                break
            status = "✓" if value != "N/A" else "✗"
            print(f"   {status} {field}: {value[:30]}...")
        
        print("\n💡 RECOMMENDATIONS:")
        if accuracy >= 70:
            print(f"   ✅ Threshold {threshold} works well! ({accuracy:.1f}% accuracy)")
        elif accuracy >= 50:
            print(f"   🔶 Threshold {threshold} is acceptable ({accuracy:.1f}% accuracy)")
            print("   Try lower threshold for better accuracy (if memory allows)")
        else:
            print(f"   ❌ Threshold {threshold} gives poor accuracy ({accuracy:.1f}%)")
            print("   Try lower values: 1.0, 0.5, or 0.0")
        
        return non_na_fields
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ OUT OF MEMORY!")
        print(f"   Threshold {threshold} uses too much memory")
        print("   Try higher threshold values (3.0, 4.0, 6.0)")
        return -1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return -1


def test_multiple_thresholds():
    """Test multiple threshold values automatically."""
    print("\n" + "=" * 70)
    print("🔬 TESTING MULTIPLE THRESHOLDS")
    print("=" * 70)
    
    thresholds = ["0.0", "0.5", "1.0", "2.0", "3.0", "6.0"]
    results = {}
    
    for threshold in thresholds:
        print(f"\n🧪 Testing threshold={threshold}...")
        os.environ["BNB_THRESHOLD"] = threshold
        
        try:
            # Quick test - just check if it loads and runs
            import subprocess
            result = subprocess.run(
                [sys.executable, __file__, "--single"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse accuracy from output
            for line in result.stdout.split("\n"):
                if "Fields extracted:" in line:
                    # Extract "X/25" pattern
                    parts = line.split(":")[-1].strip().split("/")
                    if len(parts) == 2:
                        extracted = int(parts[0])
                        results[threshold] = extracted
                        break
            
            if threshold not in results:
                if "OUT OF MEMORY" in result.stdout:
                    results[threshold] = "OOM"
                else:
                    results[threshold] = "ERROR"
                    
        except subprocess.TimeoutExpired:
            results[threshold] = "TIMEOUT"
        except Exception as e:
            results[threshold] = f"ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 THRESHOLD TEST SUMMARY:")
    print("=" * 70)
    
    best_threshold = None
    best_score = 0
    
    for threshold, result in results.items():
        if isinstance(result, int):
            accuracy = result / 25 * 100
            print(f"   Threshold {threshold}: {result}/25 fields ({accuracy:.1f}%)")
            if result > best_score:
                best_score = result
                best_threshold = threshold
        else:
            print(f"   Threshold {threshold}: {result}")
    
    if best_threshold:
        print(f"\n✅ BEST THRESHOLD: {best_threshold} ({best_score}/25 fields)")
        print(f"   Use: export BNB_THRESHOLD={best_threshold}")
    else:
        print("\n❌ No successful threshold found")
        print("   The 0.43.3 quantization may be fundamentally broken")


if __name__ == "__main__":
    import torch
    
    if "--single" in sys.argv or len(sys.argv) == 1:
        # Single threshold test
        test_threshold()
    else:
        # Test multiple thresholds
        test_multiple_thresholds()