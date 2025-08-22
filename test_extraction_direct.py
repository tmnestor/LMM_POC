#!/usr/bin/env python3
"""
Test the extraction directly to isolate the issue
"""
import sys
sys.path.append('.')

def test_direct():
    from models.llama_processor import LlamaProcessor
    
    print("Testing direct extraction with 6-groups...")
    
    # Initialize processor
    processor = LlamaProcessor(
        extraction_mode="grouped",
        debug=True,
        grouping_strategy="6_groups"
    )
    
    print(f"\nProcessor initialized:")
    print(f"  extraction_mode: {processor.extraction_mode}")
    print(f"  extraction_strategy exists: {processor.extraction_strategy is not None}")
    
    if processor.extraction_strategy:
        print(f"  field_groups count: {len(processor.extraction_strategy.field_groups)}")
        print(f"  groups: {list(processor.extraction_strategy.field_groups.keys())}")
    
    # Test the extraction function directly
    print("\n\nTesting _extract_with_custom_prompt directly...")
    try:
        from common.config import DATA_DIR
        from common.evaluation_utils import discover_images
        
        images = discover_images(DATA_DIR)
        if images:
            test_image = images[0]
            print(f"Test image: {test_image}")
            
            # Call the extraction function directly
            response = processor._extract_with_custom_prompt(
                test_image,
                "Extract: ABN: [value]\nTOTAL: [value]",
                max_new_tokens=100,
                temperature=0.1
            )
            
            print(f"\nDirect extraction response length: {len(response)}")
            print(f"Response preview: {response[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct()