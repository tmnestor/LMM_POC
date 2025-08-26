#!/usr/bin/env python3
"""
Quick Prompt Test - Ultra-simple experimental prompt testing

For when you just want to test a quick idea without command-line arguments.
Edit the EXPERIMENTAL_PROMPT and run: python quick_prompt_test.py
"""

import sys
from pathlib import Path

# Add project root to path - ensure we can import project modules
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Debug: Check if required files exist
required_files = [
    'common/extraction_parser.py',
    'models/llama_processor.py', 
    'models/internvl3_processor.py',
    'experimental_prompt_tester.py'
]

missing_files = []
for file_path in required_files:
    if not (project_root / file_path).exists():
        missing_files.append(file_path)

if missing_files:
    print("❌ Missing required files:")
    for file in missing_files:
        print(f"   - {file}")
    print(f"💡 Project root: {project_root}")
    print("💡 Please check file locations")
    sys.exit(1)

# Try importing with explicit path debugging
try:
    import experimental_prompt_tester
    ExperimentalPromptTester = experimental_prompt_tester.ExperimentalPromptTester
except ImportError as e:
    print(f"❌ Import error during experimental_prompt_tester import: {e}")
    print(f"💡 Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Try direct import to isolate the issue
    try:
        import common.extraction_parser  # noqa: F401
        print("✅ common.extraction_parser imports successfully")
    except ImportError as e2:
        print(f"❌ Direct common.extraction_parser import failed: {e2}")
        print("💡 This suggests a PYTHONPATH or conda environment issue")
    
    sys.exit(1)

# ============================================================================
# EDIT THIS SECTION FOR YOUR EXPERIMENT
# ============================================================================

# Choose model: "llama" or "internvl3"
MODEL = "llama"

# Your experimental prompt (edit this!)
EXPERIMENTAL_PROMPT = """
Extract all text from this image and format it as proper markdown. Maintain the visual hierarchy:
- Main titles as # headers
- Subtitles as ## or ### 
- Bullet points as - lists
- Numbered items as 1. 2. 3.
- Preserve line breaks and spacing

Markdown output:
"""

# Test image (or None to use first available)
# TEST_IMAGE = None  # Will auto-select from evaluation_data/
TEST_IMAGE = "evaluation_data/synthetic_invoice_001.png"  # Will auto-select from evaluation_data/

# Compare with baseline? (True/False)
COMPARE_WITH_BASELINE = True

# ============================================================================
# NO NEED TO EDIT BELOW THIS LINE
# ============================================================================


def main():
    print("🚀 Quick Experimental Prompt Test")
    print(f"📊 Model: {MODEL}")
    print(f"📝 Prompt: {len(EXPERIMENTAL_PROMPT)} characters")

    # Auto-select image if none specified
    image_path = TEST_IMAGE
    if not image_path:
        try:
            from common.extraction_parser import discover_images

            images = discover_images("evaluation_data/")
            if images:
                image_path = str(images[0])
                print(f"🎯 Using: {Path(image_path).name}")
            else:
                print("❌ No test images found in evaluation_data/")
                return
        except Exception as e:
            print(f"❌ Error finding images: {e}")
            return

    # Run the test
    try:
        tester = ExperimentalPromptTester(MODEL)

        if COMPARE_WITH_BASELINE:
            print("\n⚖️  Running comparison with baseline...")
            result = tester.compare_with_baseline(EXPERIMENTAL_PROMPT, image_path)

            # Simple analysis
            baseline_response = result["baseline"]["response"]
            experimental_response = result["experimental"]["response"]

            print("\n📊 QUICK ANALYSIS:")
            print(f"Baseline length: {len(baseline_response)} chars")
            print(f"Experimental length: {len(experimental_response)} chars")
            print(f"Time difference: {result['time_difference']:+.2f}s")

        else:
            print("\n🧪 Running experimental prompt only...")
            result = tester.test_prompt(EXPERIMENTAL_PROMPT, image_path)

        print("\n✅ Test completed! Review the output above.")
        print("\n💡 To modify the test:")
        print("   1. Edit EXPERIMENTAL_PROMPT in this file")
        print("   2. Change MODEL if needed")
        print("   3. Run again: python quick_prompt_test.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
