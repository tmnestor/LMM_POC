#!/usr/bin/env python3
"""
Debug script to identify why document detection returns empty responses
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_detection():
    """Test document detection directly to see the error."""

    print("🔍 DEBUG: Testing InternVL3 Document Detection")
    print("=" * 60)

    try:
        # Import the hybrid processor
        from models.document_aware_internvl3_hybrid_processor import (
            DocumentAwareInternVL3HybridProcessor,
        )

        # Initialize with minimal setup
        print("🚀 Initializing processor...")
        processor = DocumentAwareInternVL3HybridProcessor(
            field_list=["DOCUMENT_TYPE"],
            model_path="/home/jovyan/nfs_share/models/InternVL3-8B",
            debug=True
        )
        print("✅ Processor initialized")

        # Test image path
        test_image = "/home/jovyan/nfs_share/tod/evaluation_data/image_001.png"
        print(f"📷 Testing with: {test_image}")

        # Load detection config
        import yaml
        config_path = Path('prompts/document_type_detection.yaml')
        with config_path.open('r') as f:
            detection_config = yaml.safe_load(f)

        detection_prompt = detection_config['prompts']['detection_simple']['prompt']
        print(f"📝 Detection prompt: '{detection_prompt}'")

        # Load image
        print("🖼️ Loading image...")
        pixel_values = processor.load_image(test_image)
        print(f"✅ Image loaded: {pixel_values.shape}")

        # Try detection with verbose error handling
        print("🤖 Calling model for detection...")

        try:
            response = processor._resilient_generate(
                pixel_values=pixel_values,
                question=detection_prompt,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                is_detection=True
            )

            print(f"📄 Raw response: '{response}'")
            print(f"📏 Response length: {len(response)}")

            if not response:
                print("❌ EMPTY RESPONSE - This is the root cause!")
            else:
                print("✅ Got non-empty response")

        except Exception as e:
            print(f"❌ Exception in _resilient_generate: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Setup error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_detection()