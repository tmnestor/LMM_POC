# Unified Prompt Usage Examples

## Available Prompt Sets

### 🚀 Enhanced InternVL3 (High Performance)
- **File**: `prompts/unified/enhanced_internvl3.yaml`
- **Best for**: Complex documents, high accuracy requirements
- **Contains**: invoice, receipt, bank_statement

### 🤖 Generated Llama (Schema-Based)
- **File**: `prompts/unified/generated_llama.yaml`
- **Best for**: Llama model, consistent schema-based extraction
- **Contains**: invoice, receipt, bank_statement

### ⚡ Simple InternVL3 (Fast & Reliable)
- **File**: `prompts/unified/simple_internvl3.yaml`
- **Best for**: Fast processing, straightforward documents
- **Contains**: invoice, receipt, bank_statement

### 📋 Standard InternVL3
- **File**: `prompts/unified/internvl3_standard.yaml`
- **Best for**: Consistent key-value format output
- **Contains**: invoice, receipt, bank_statement

### 🏦 Bank Statement Specialists
- **File**: `prompts/unified/old_bank_specialized.yaml`
- **Best for**: Complex bank statements with different layouts
- **Contains**: Multiple bank statement variants

### 📜 Legacy Prompts
- **File**: `prompts/unified/generic_legacy.yaml`
- **Best for**: Baseline comparison
- **Contains**: invoice, receipt, universal

## Configuration Examples

### For InternVL3 Notebooks (ivl3_batch.ipynb)

#### Enhanced InternVL3 Prompts
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

### For Llama Notebooks (llama_batch.ipynb)

#### Generated Llama Prompts (Recommended for Llama)
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': 'prompts/unified/generated_llama.yaml',
        'RECEIPT': 'prompts/unified/generated_llama.yaml',
        'BANK_STATEMENT': 'prompts/unified/generated_llama.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

#### Or Enhanced InternVL3 Prompts (Cross-Model Testing)
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

## Easy A/B Testing

```python
# Choose your prompt set
PROMPT_SET = 'enhanced_internvl3'  # or 'generated_llama', 'simple_internvl3', etc.

PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection_simple',
    'extraction_files': {
        'INVOICE': f'prompts/unified/{PROMPT_SET}.yaml',
        'RECEIPT': f'prompts/unified/{PROMPT_SET}.yaml',
        'BANK_STATEMENT': f'prompts/unified/{PROMPT_SET}.yaml'
    },
    'extraction_keys': {
        'INVOICE': 'invoice',
        'RECEIPT': 'receipt',
        'BANK_STATEMENT': 'bank_statement'
    }
}
```

## Model-Specific Recommendations

### For InternVL3:
1. **Best Performance**: `enhanced_internvl3.yaml`
2. **Fast Processing**: `simple_internvl3.yaml`
3. **Standard Format**: `internvl3_standard.yaml`

### For Llama:
1. **Recommended**: `generated_llama.yaml` (designed for Llama)
2. **Cross-Model**: `enhanced_internvl3.yaml` (test compatibility)
3. **Legacy**: `generic_legacy.yaml` (baseline comparison)

### For Bank Statements:
- **Complex Layouts**: `old_bank_specialized.yaml`
- **Date-Grouped**: Use bank_statement_date_grouped variant
- **Flat Statements**: Use bank_statement_flat_optimized variant

## Quick Test Setup

To quickly test different prompts:

1. **Save your current config**
2. **Replace PROMPT_CONFIG** with desired set
3. **Run notebook**
4. **Compare results**
5. **Document findings**

Happy testing! 🚀
