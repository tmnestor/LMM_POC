# Old Prompt Conversion Plan

## Problem Statement
The old prompts in `Old_prompts/` directory have different YAML key structures than the current system expects. We need a way to test these historically high-performing prompts without significant changes to the notebooks (`ivl3_batch.ipynb` and `llama_batch.ipynb`).

## Current System Analysis

### Current Prompt Structure
Located in `prompts/` directory:
```yaml
prompts:
  invoice:      # Document type as key
    name: "Invoice Extraction"
    description: "..."
    prompt: |
      ...
  receipt:      # Document type as key
    name: "Receipt Extraction"
    description: "..."
    prompt: |
      ...
  bank_statement:  # Document type as key
    name: "Bank Statement Extraction"
    description: "..."
    prompt: |
      ...
```

### Old Prompt Structure
Located in `Old_prompts/` directory:
```yaml
prompts:
  extraction:   # Generic key (multiple files)
  standard:     # Generic key (multiple files)
  flat:         # Bank statement specific
  date_grouped: # Bank statement specific
  flat_optimized: # Bank statement specific
  minimal:      # Bank statement specific
  universal:    # Universal extraction
```

### Key Differences
1. **Key naming**: Old prompts use generic keys (`extraction`, `standard`) or specific keys (`date_grouped`, `flat`)
2. **File organization**: Old prompts often have document type in filename instead of key
3. **Same YAML structure**: Both use `prompts:` → `key:` → `name/description/prompt`

## Solution: Prompt Conversion Script

### Script: `convert_old_prompts.py`

#### Core Functionality
1. Read all YAML files from `Old_prompts/`
2. Detect document type from filename
3. Map old keys to new document type keys
4. Generate new YAML files with correct structure
5. Save to `prompts/converted/` directory

#### Conversion Mappings

```python
# Filename patterns to document types
FILENAME_TO_DOCTYPE = {
    'invoice': 'invoice',
    'receipt': 'receipt',
    'bank_statement': 'bank_statement',
    'universal': 'universal'
}

# Old prompt keys to new keys (with filename context)
OLD_KEY_MAPPINGS = {
    'extraction': 'auto_detect',     # Use filename to determine
    'standard': 'auto_detect',       # Use filename to determine
    'flat': 'bank_statement',
    'date_grouped': 'bank_statement',
    'flat_optimized': 'bank_statement',
    'minimal': 'bank_statement',
    'universal': 'universal'
}

# Special handling for model-specific prompts
MODEL_PREFIXES = ['llama', 'internvl3', 'enhanced_internvl3', 'simple_internvl3']
```

#### Conversion Algorithm
```python
def convert_prompt(old_file_path):
    # 1. Parse filename to detect document type
    filename = Path(old_file_path).stem
    doc_type = detect_document_type(filename)

    # 2. Load old YAML
    old_data = yaml.safe_load(old_file_path)

    # 3. Extract prompt content from old structure
    old_prompts = old_data['prompts']
    old_key = list(old_prompts.keys())[0]  # Usually single key

    # 4. Create new structure
    new_data = {
        'prompts': {
            doc_type: {
                'name': old_prompts[old_key]['name'],
                'description': old_prompts[old_key]['description'],
                'prompt': old_prompts[old_key]['prompt']
            }
        }
    }

    # 5. Save with descriptive name
    model_prefix = detect_model_prefix(filename)
    output_name = f"{model_prefix}_{doc_type}_{old_key}.yaml"
    return new_data, output_name
```

#### Output Directory Structure
```
prompts/
├── converted/
│   ├── internvl3_invoice_extraction.yaml
│   ├── internvl3_receipt_extraction.yaml
│   ├── internvl3_bank_statement_date_grouped.yaml
│   ├── enhanced_internvl3_invoice_extraction.yaml
│   ├── enhanced_internvl3_receipt_extraction.yaml
│   ├── enhanced_internvl3_bank_statement_extraction.yaml
│   ├── llama_invoice_standard.yaml
│   └── ...
├── llama_prompts.yaml (original)
└── internvl3_prompts.yaml (original)
```

## Notebook Configuration Updates

### Minimal Changes Required

#### Option 1: Simple Flag Toggle
Add to CONFIG section in both notebooks:
```python
CONFIG = {
    # ... existing config ...

    # Test old prompts - just change this flag
    'USE_OLD_PROMPTS': True,
    'OLD_PROMPT_SET': 'enhanced',  # 'enhanced', 'simple', 'date_grouped', etc.
}

# Conditional prompt config based on flag
if CONFIG.get('USE_OLD_PROMPTS', False):
    if CONFIG['OLD_PROMPT_SET'] == 'enhanced':
        PROMPT_CONFIG = {
            'extraction_files': {
                'INVOICE': 'prompts/converted/enhanced_internvl3_invoice_extraction.yaml',
                'RECEIPT': 'prompts/converted/enhanced_internvl3_receipt_extraction.yaml',
                'BANK_STATEMENT': 'prompts/converted/enhanced_internvl3_bank_statement_extraction.yaml'
            },
            'extraction_keys': {
                'INVOICE': 'invoice',
                'RECEIPT': 'receipt',
                'BANK_STATEMENT': 'bank_statement'
            }
        }
    # ... other sets
else:
    # Use existing PROMPT_CONFIG
    pass
```

#### Option 2: Direct Path Override
Simply change the paths in existing PROMPT_CONFIG:
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        # Just change these paths to test different prompts
        'INVOICE': 'prompts/converted/enhanced_internvl3_invoice_extraction.yaml',
        'RECEIPT': 'prompts/converted/enhanced_internvl3_receipt_extraction.yaml',
        'BANK_STATEMENT': 'prompts/converted/internvl3_bank_statement_date_grouped.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

## Implementation Steps

1. **Create conversion script** (`convert_old_prompts.py`)
   - Load all old prompts
   - Apply conversion mappings
   - Validate output structure
   - Save to `prompts/converted/`

2. **Run conversion**
   ```bash
   python convert_old_prompts.py
   ```

3. **Update notebooks** (minimal changes)
   - Add USE_OLD_PROMPTS flag to CONFIG
   - Add conditional PROMPT_CONFIG selection
   - Or directly modify PROMPT_CONFIG paths

4. **Test converted prompts**
   - Run notebooks with different prompt sets
   - Compare performance metrics
   - Identify best performing prompts

## High-Performing Prompt Recommendations

Based on filenames, these appear to be optimized versions worth testing:

### InternVL3 Model
- `enhanced_internvl3_invoice.yaml` - Enhanced with layout examples
- `enhanced_internvl3_receipt.yaml` - Enhanced with layout examples
- `enhanced_internvl3_bank_statement.yaml` - Enhanced with layout examples
- `bank_statement_date_grouped.yaml` - Optimized for grouped transactions

### Llama Model
- `llama_single_pass_v4.yaml` - Version 4 optimization
- `bank_statement_flat_optimized.yaml` - Optimized for flat statements
- `universal_extraction.yaml` - Single-pass universal extraction

## Benefits

1. **No core system changes** - Works with existing `SimplePromptLoader`
2. **Preserves originals** - Old prompts remain untouched in `Old_prompts/`
3. **Easy A/B testing** - Switch between prompts via simple config change
4. **Organized structure** - Converted prompts clearly labeled in `prompts/converted/`
5. **Reversible** - Can always revert to original prompts
6. **Performance comparison** - Easy to benchmark old vs new prompts

## Example Conversion Script Structure

```python
#!/usr/bin/env python3
"""Convert old prompt format to current system format."""

import yaml
from pathlib import Path
from typing import Dict, Tuple

def detect_document_type(filename: str) -> str:
    """Detect document type from filename."""
    # Implementation here
    pass

def convert_prompt_file(input_path: Path) -> Tuple[Dict, str]:
    """Convert single prompt file to new format."""
    # Implementation here
    pass

def main():
    """Convert all old prompts to new format."""
    old_prompts_dir = Path("Old_prompts")
    output_dir = Path("prompts/converted")
    output_dir.mkdir(exist_ok=True)

    for yaml_file in old_prompts_dir.glob("*.yaml"):
        try:
            new_data, output_name = convert_prompt_file(yaml_file)
            output_path = output_dir / output_name

            with open(output_path, 'w') as f:
                yaml.dump(new_data, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)

            print(f"✅ Converted: {yaml_file.name} → {output_name}")
        except Exception as e:
            print(f"❌ Failed: {yaml_file.name}: {e}")

if __name__ == "__main__":
    main()
```

## Next Steps

1. Review and approve this plan
2. Implement the conversion script
3. Run conversion on all old prompts
4. Update notebook configurations
5. Test and benchmark performance
6. Select best performing prompts for production use