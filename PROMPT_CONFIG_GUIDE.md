# Prompt Configuration Guide

## Overview

The `PROMPT_CONFIG` in `llama_batch.ipynb` now provides **explicit, transparent control** over which prompt files and keys are used for each document type. No hidden logic.

## Configuration Structure

```python
PROMPT_CONFIG = {
    # Document type detection
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',

    # Extraction prompt files (REQUIRED)
    'extraction_files': {
        'INVOICE': 'path/to/invoice_prompts.yaml',
        'RECEIPT': 'path/to/receipt_prompts.yaml',
        'BANK_STATEMENT': 'path/to/bank_statement_prompts.yaml'
    },

    # Extraction prompt keys (OPTIONAL)
    'extraction_keys': {
        'INVOICE': 'specific_key',
        'RECEIPT': 'specific_key',
        'BANK_STATEMENT': 'specific_key'
    }
}
```

## How It Works

### 1. File Selection
**Always explicit** - specified in `extraction_files`:
```python
'extraction_files': {
    'INVOICE': 'prompts/generated/llama_invoice_prompt.yaml'
}
```

### 2. Key Selection (3 options)

#### Option A: Auto-derived (default)
If you **don't specify** `extraction_keys`, the key is derived from document type:
- `INVOICE` → `invoice`
- `RECEIPT` → `receipt`
- `BANK_STATEMENT` → `bank_statement` + structure suffix

```python
# Just specify files
'extraction_files': {
    'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
}
# Result: Uses key 'bank_statement_flat' or 'bank_statement_date_grouped'
```

#### Option B: Partial override
Specify a base key, structure suffix still appended for bank statements:
```python
'extraction_files': {
    'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
},
'extraction_keys': {
    'BANK_STATEMENT': 'bank_statement'
}
# Result: Uses 'bank_statement_flat' or 'bank_statement_date_grouped'
```

#### Option C: Full override
Specify complete key with suffix to skip structure classification:
```python
'extraction_files': {
    'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
},
'extraction_keys': {
    'BANK_STATEMENT': 'bank_statement_flat'  # Full key with suffix
}
# Result: ALWAYS uses 'bank_statement_flat', ignores vision classification
```

## Bank Statement Structure Logic

**Only applies when:**
- Document type is `BANK_STATEMENT`
- Key doesn't already contain `_flat` or `_date_grouped`

**Process:**
1. Vision-based classification determines structure (`flat` or `date_grouped`)
2. Structure suffix is appended to base key
3. Full key used to load prompt

**To bypass:**
Provide full key in `extraction_keys` with the suffix already included.

## Common Configurations

### Use Generated Prompts (Current Default)
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',
    'extraction_files': {
        'INVOICE': 'prompts/generated/llama_invoice_prompt.yaml',
        'RECEIPT': 'prompts/generated/llama_receipt_prompt.yaml',
        'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
    }
}
```

### Use Legacy Prompts
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',
    'extraction_files': {
        'INVOICE': 'prompts/llama_prompts.yaml',
        'RECEIPT': 'prompts/llama_prompts.yaml',
        'BANK_STATEMENT': 'prompts/llama_prompts.yaml'
    }
}
```

### Mix Old and New
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',
    'extraction_files': {
        'INVOICE': 'prompts/generated/llama_invoice_prompt.yaml',
        'RECEIPT': 'prompts/llama_prompts.yaml',  # Old
        'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
    }
}
```

### Force Specific Bank Statement Structure
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',
    'extraction_files': {
        'INVOICE': 'prompts/generated/llama_invoice_prompt.yaml',
        'RECEIPT': 'prompts/generated/llama_receipt_prompt.yaml',
        'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
    },
    'extraction_keys': {
        'BANK_STATEMENT': 'bank_statement_flat'  # Always use flat, never date_grouped
    }
}
```

## Debugging

The batch processor will print which file and key it's using:
```
📁 Using prompt: prompts/generated/llama_bank_statement_prompt.yaml
🔑 Loading key: bank_statement_flat
```

## Implementation Details

**File:** `common/batch_processor.py` (lines 943-980)

**Logic:**
1. Get `extraction_files` dict from config
2. Look up file path for document type
3. Get `extraction_keys` dict from config (may not exist)
4. If key specified in `extraction_keys`, use it
5. Otherwise, derive from document type (lowercase)
6. For bank statements only: append structure suffix if not already present
7. Load prompt using SimplePromptLoader

**No hidden fallbacks or legacy modes** - configuration is explicit and traceable.

## Changes Made

### Before (Hidden Logic)
- Looked for non-existent `extraction_keys` config
- Fell back to old `llama_prompts.yaml` silently
- Bank structure suffix appended invisibly

### After (Explicit Configuration)
- Clear two-level config: `extraction_files` (required) + `extraction_keys` (optional)
- All logic documented in notebook with examples
- Bank structure suffix logic clearly explained
- Full override capability via explicit keys

## Testing

All scenarios tested and working:
- ✅ Default behavior (key auto-derived)
- ✅ Explicit key override
- ✅ Legacy prompt files
- ✅ Mixed old/new prompts
- ✅ Forced bank statement structure

See test output in terminal for verification.
