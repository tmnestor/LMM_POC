#!/usr/bin/env python3
"""Debug script to compare YAML loading between direct and processor methods."""

import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, '/Users/tod/Desktop/LMM_POC')

def direct_yaml_load():
    """Load YAML directly like our manual test."""
    print("=== DIRECT YAML LOADING ===")
    yaml_path = Path('prompts/llama_single_pass_high_performance.yaml')
    with yaml_path.open('r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        single_pass = yaml_data.get('single_pass', {})
        field_instructions = single_pass.get('field_instructions', {})
        
    print(f"Total field instructions: {len(field_instructions)}")
    print(f"GST_AMOUNT present: {'GST_AMOUNT' in field_instructions}")
    print("Field instruction keys:")
    for key in field_instructions.keys():
        print(f"  - {key}")
    return field_instructions

def processor_yaml_load():
    """Load YAML using processor method."""
    print("\n=== PROCESSOR YAML LOADING ===")
    from models.document_aware_llama_processor import DocumentAwareLlamaProcessor
    
    # Create minimal processor instance
    processor = DocumentAwareLlamaProcessor(['DOCUMENT_TYPE'])
    yaml_config = processor._load_yaml_config()
    field_instructions = yaml_config.get('field_instructions', {})
    
    print(f"Total field instructions: {len(field_instructions)}")
    print(f"GST_AMOUNT present: {'GST_AMOUNT' in field_instructions}")
    print("Field instruction keys:")
    for key in field_instructions.keys():
        print(f"  - {key}")
    return field_instructions

def compare_results():
    """Compare both loading methods."""
    direct = direct_yaml_load()
    processor = processor_yaml_load()
    
    print("\n=== COMPARISON ===")
    direct_keys = set(direct.keys())
    processor_keys = set(processor.keys())
    
    missing_in_processor = direct_keys - processor_keys
    extra_in_processor = processor_keys - direct_keys
    
    if missing_in_processor:
        print(f"❌ Missing in processor: {missing_in_processor}")
    if extra_in_processor:
        print(f"➕ Extra in processor: {extra_in_processor}")
    if not missing_in_processor and not extra_in_processor:
        print("✅ Both methods load identical field instructions")

if __name__ == "__main__":
    compare_results()