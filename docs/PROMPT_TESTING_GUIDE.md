# Prompt Testing Guide for Batch Notebooks

This guide explains how to test different prompt sets in `ivl3_batch.ipynb` and `llama_batch.ipynb` using the unified prompt system.

## Quick Start

To test different prompts, simply update the `PROMPT_CONFIG` section in your notebook configuration.

## Available Prompt Sets

### 🚀 Enhanced Performance Prompts
- **File**: `prompts/unified/enhanced_internvl3.yaml`
- **Best for**: High accuracy, complex document layouts
- **Description**: Enhanced prompts with layout-specific guidance and examples

### ⚡ Simple & Fast Prompts
- **File**: `prompts/unified/simple_internvl3.yaml`
- **Best for**: Fast processing, straightforward documents
- **Description**: Streamlined prompts for reliable, quick extraction

### 📋 Standard Prompts
- **File**: `prompts/unified/internvl3_standard.yaml`
- **Best for**: Consistent key-value format output
- **Description**: Standard prompts with enforced formatting

### 🏦 Bank Statement Specialists
- **File**: `prompts/unified/old_bank_specialized.yaml`
- **Best for**: Complex bank statements with grouped transactions
- **Description**: Multiple specialized bank statement prompt variants

### 📜 Legacy Prompts
- **File**: `prompts/unified/generic_legacy.yaml`
- **Best for**: Baseline comparison
- **Description**: Original generic prompts for comparison testing

## Configuration Examples

### Method 1: Simple Replacement (Recommended)

Find the `PROMPT_CONFIG` section in your notebook and replace it:

#### Enhanced Prompts (High Performance)
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': 'prompts/unified/enhanced_internvl3.yaml',
        'RECEIPT': 'prompts/unified/enhanced_internvl3.yaml',
        'BANK_STATEMENT': 'prompts/unified/enhanced_internvl3.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

#### Simple Prompts (Fast & Reliable)
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': 'prompts/unified/simple_internvl3.yaml',
        'RECEIPT': 'prompts/unified/simple_internvl3.yaml',
        'BANK_STATEMENT': 'prompts/unified/simple_internvl3.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

### Method 2: Easy A/B Testing

Add a prompt selection variable at the top of your configuration:

```python
# CONFIG SECTION - CHANGE THIS TO TEST DIFFERENT PROMPTS
USE_PROMPT_SET = 'enhanced_internvl3'  # Options: 'enhanced_internvl3', 'simple_internvl3', 'internvl3_standard', 'old_bank_specialized', 'generic_legacy'

# Automatic prompt configuration
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': f'prompts/unified/{USE_PROMPT_SET}.yaml',
        'RECEIPT': f'prompts/unified/{USE_PROMPT_SET}.yaml',
        'BANK_STATEMENT': f'prompts/unified/{USE_PROMPT_SET}.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

### Method 3: Mixed Prompt Testing

Test different prompts for different document types:

```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': 'prompts/unified/enhanced_internvl3.yaml',        # Enhanced for invoices
        'RECEIPT': 'prompts/unified/simple_internvl3.yaml',          # Simple for receipts
        'BANK_STATEMENT': 'prompts/unified/old_bank_specialized.yaml' # Specialized for bank statements
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

### Method 4: Bank Statement Specialist Testing

For specialized bank statement testing with different variants:

```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': 'prompts/unified/enhanced_internvl3.yaml',
        'RECEIPT': 'prompts/unified/enhanced_internvl3.yaml',
        'BANK_STATEMENT': 'prompts/unified/old_bank_specialized.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'  # Note: old_bank_specialized.yaml contains multiple variants
    }
}
```

## Step-by-Step Instructions

### For ivl3_batch.ipynb:

1. **Open the notebook**: `ivl3_batch.ipynb`

2. **Find the configuration cell** (usually cell #6 or #7):
   ```python
   # Configuration
   console = Console()

   CONFIG = {
       # ... existing config ...
   }

   # InternVL3 prompt configuration
   PROMPT_CONFIG = {
       # THIS IS WHERE YOU MAKE CHANGES
   }
   ```

3. **Replace the PROMPT_CONFIG** with one of the examples above

4. **Run the notebook** from the configuration cell onwards

### For llama_batch.ipynb:

1. **Open the notebook**: `llama_batch.ipynb`

2. **Find the configuration cell** containing PROMPT_CONFIG

3. **Replace with the same configuration** (the unified prompts work for both models)

4. **Run the notebook** from the configuration cell onwards

## Performance Comparison Workflow

To systematically test prompt performance:

### 1. Baseline Run
```python
USE_PROMPT_SET = 'simple_internvl3'  # Start with simple prompts
```
Run notebook → Save results as `simple_results.csv`

### 2. Enhanced Run
```python
USE_PROMPT_SET = 'enhanced_internvl3'  # Test enhanced prompts
```
Run notebook → Save results as `enhanced_results.csv`

### 3. Specialized Run
```python
USE_PROMPT_SET = 'old_bank_specialized'  # Test specialized prompts
```
Run notebook → Save results as `specialized_results.csv`

### 4. Compare Results
Use `model_comparison.ipynb` to analyze the different result sets.

## Expected Output Changes

When you change prompts, you should see:

### In Console Output:
```
📝 Using invoice prompt: 1234 characters
📝 Using receipt prompt: 1456 characters
📝 Using bank_statement prompt: 2345 characters
```

### In Debug Mode:
If `CONFIG['VERBOSE'] = True`, you'll see the actual prompt being used:
```
📝 Prompt: Extract ALL data from this invoice image. Respond in exact format...
```

### In Results:
- Different accuracy scores
- Different processing times
- Different field extraction patterns

## Troubleshooting

### Error: "Prompt file not found"
```
❌ Prompt file not found: prompts/unified/enhanced_internvl3.yaml
```
**Solution**: Run the conversion scripts first:
```bash
python convert_old_prompts.py
python consolidate_prompts.py
```

### Error: "Prompt key not found"
```
❌ Prompt 'invoice' not found in enhanced_internvl3.yaml
```
**Solution**: Check available prompts:
```python
from common.simple_prompt_loader import SimplePromptLoader
available = SimplePromptLoader.get_available_prompts('unified/enhanced_internvl3.yaml')
print(f"Available: {available}")
```

### Error: "No prompts section"
This indicates a malformed YAML file. Check the file structure:
```yaml
prompts:
  invoice:
    name: "..."
    description: "..."
    prompt: |
      ...
```

## Tips for Prompt Testing

### 1. Document Your Tests
Keep track of which prompts you've tested:
```python
# Add this to your notebook
CURRENT_TEST = {
    'prompt_set': 'enhanced_internvl3',
    'date': '2025-01-22',
    'notes': 'Testing enhanced prompts for invoice accuracy'
}
print(f"🧪 Current test: {CURRENT_TEST}")
```

### 2. Use Consistent Test Data
Always test with the same image set for fair comparison:
```python
CONFIG = {
    'MAX_IMAGES': 10,  # Use same number for all tests
    'DOCUMENT_TYPES': None,  # Test all types consistently
}
```

### 3. Monitor Memory Usage
Enhanced prompts may use more GPU memory:
```bash
# On remote machine
nvidia-smi --loop=5
```

### 4. Save Results with Descriptive Names
```python
# Update the timestamp to include prompt info
BATCH_TIMESTAMP = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_enhanced_prompts"
```

## Integration with Existing Workflow

The unified prompts work seamlessly with your existing:
- ✅ `BatchDocumentProcessor`
- ✅ `DocumentAwareInternVL3HybridProcessor`
- ✅ `DocumentAwareLlamaProcessor`
- ✅ `SimplePromptLoader`
- ✅ Model comparison notebooks
- ✅ Evaluation metrics
- ✅ Report generation

No code changes needed - just configuration updates!

## Next Steps

1. **Test enhanced prompts** on your current dataset
2. **Compare accuracy metrics** between prompt sets
3. **Identify best-performing prompts** for each document type
4. **Update production notebooks** with optimal prompt configuration
5. **Document findings** for future reference

Happy prompt testing! 🚀