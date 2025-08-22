#!/usr/bin/env python3
"""
Debug why 6-groups shows 0.0s processing time
"""

import time


def test_extraction_flow():
    """Test if extraction is actually happening."""
    
    from common.grouped_extraction import GroupedExtractionStrategy
    
    # Test 6-group strategy
    strategy = GroupedExtractionStrategy("grouped", debug=False, grouping_strategy="6_groups")
    
    print(f"Strategy initialized with {len(strategy.field_groups)} groups")
    print(f"Groups: {list(strategy.field_groups.keys())}")
    
    # Create a dummy extraction function that simulates work
    def dummy_extract(image_path, prompt, **kwargs):
        print(f"  EXTRACT CALLED: image={image_path}, prompt_len={len(prompt)}, kwargs={kwargs}")
        time.sleep(0.1)  # Simulate some work
        return "ABN: 123\nTOTAL: $100"
    
    # Try to extract
    print("\nCalling extract_fields_grouped...")
    start = time.time()
    result, metadata = strategy.extract_fields_grouped(
        "dummy_image.png",
        dummy_extract
    )
    elapsed = time.time() - start
    
    print(f"\nElapsed time: {elapsed:.2f}s")
    print(f"Metadata: {metadata}")
    print(f"Stats: {strategy.stats}")
    
    if elapsed < 0.5:
        print("\n❌ PROBLEM: Extraction completed too quickly!")
        print("This suggests groups aren't being processed")
    else:
        print("\n✅ Extraction seems to be working")

if __name__ == "__main__":
    test_extraction_flow()