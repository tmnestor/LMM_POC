# Prompt Selection in llama_batch.ipynb

## Overview
The prompt selection in `llama_batch.ipynb` is a **multi-stage process** involving document type detection, optional structure classification, and then extraction prompt selection. The selection logic is centralized in the `BatchDocumentProcessor._process_llama_image()` method.

---

## Configuration (Where Selection Starts)

**Location**: Cell 6 of the notebook

```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',

    'extraction_files': {
        'INVOICE': 'prompts/generated/llama_invoice_prompt.yaml',
        'RECEIPT': 'prompts/generated/llama_receipt_prompt.yaml',
        'BANK_STATEMENT': 'prompts/generated/llama_bank_statement_prompt.yaml'
    },

    # Optional - for explicit key control
    'extraction_keys': {
        # 'INVOICE': 'invoice',
        # 'BANK_STATEMENT': 'bank_statement_flat'  # Force specific structure
    }
}
```

This config is passed to `BatchDocumentProcessor` in Cell 7.

---

## Prompt Selection Flow (4 Phases)

### **Phase 1: Document Type Detection**
**Location**: `batch_processor.py:942-1040`

```python
# 1. Load detection config from YAML
detection_path = Path(self.prompt_config["detection_file"])
detection_config = yaml.safe_load(detection_path.open("r"))

# 2. Get the detection prompt using configured key
detection_prompt_key = self.prompt_config.get("detection_key")
doc_type_prompt = detection_config["prompts"][detection_prompt_key]["prompt"]

# 3. Send to model for classification
# ... model inference code ...

# 4. Parse response to determine document type
document_type = self._parse_document_type_response(response, detection_config)
# Returns: "INVOICE", "RECEIPT", or "BANK_STATEMENT"
```

**Result**: `document_type` variable set (e.g., `"BANK_STATEMENT"`)

---

### **Phase 2: Bank Statement Structure Classification**
**Location**: `batch_processor.py:1049-1070`

**Only runs if `document_type == "BANK_STATEMENT"`**

```python
if document_type == "BANK_STATEMENT":
    from .vision_bank_statement_classifier import classify_bank_statement_structure_vision

    # Use vision analysis to determine structure
    bank_structure = classify_bank_statement_structure_vision(
        image_path=image_path,
        model=self.model,
        processor=self.processor,
        verbose=verbose
    )
    # Returns: "flat" or "date_grouped"
```

**Result**: `bank_structure` variable set (e.g., `"flat"` or `"date_grouped"`)

---

### **Phase 3: Extraction Prompt Selection**
**Location**: `batch_processor.py:1072-1116`

This is **the core prompt selection logic**:

```python
# 1. Get document type in uppercase
doc_type_upper = document_type.upper()  # "BANK_STATEMENT"

# 2. Get prompt FILE from config
extraction_files = self.prompt_config.get('extraction_files', {})
extraction_file = extraction_files.get(
    doc_type_upper,
    'prompts/llama_prompts.yaml'  # fallback
)
# Result: 'prompts/generated/llama_bank_statement_prompt.yaml'

# 3. Get prompt KEY from config (or derive)
extraction_keys = self.prompt_config.get('extraction_keys', {})

if doc_type_upper in extraction_keys:
    # Use explicitly configured key
    extraction_key = extraction_keys[doc_type_upper]
else:
    # Derive key from document type
    extraction_key = document_type.lower()  # "bank_statement"

# 4. For bank statements: append structure suffix (unless full key specified)
if document_type == "BANK_STATEMENT" and bank_structure:
    if "_flat" not in extraction_key and "_date_grouped" not in extraction_key:
        extraction_key = f"{extraction_key}_{bank_structure}"
        # Result: "bank_statement_flat" or "bank_statement_date_grouped"

# 5. Load the prompt
extraction_prompt = prompt_loader.load_prompt(extraction_file, extraction_key)
```

**Example Results**:
- **Invoice**: File=`llama_invoice_prompt.yaml`, Key=`invoice`
- **Receipt**: File=`llama_receipt_prompt.yaml`, Key=`receipt`
- **Bank Statement (flat)**: File=`llama_bank_statement_prompt.yaml`, Key=`bank_statement_flat`
- **Bank Statement (grouped)**: File=`llama_bank_statement_prompt.yaml`, Key=`bank_statement_date_grouped`

---

### **Phase 4: Extraction**
**Location**: `batch_processor.py:1165-1262`

```python
# Use the selected prompt for extraction
messageDataStructure = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": extraction_prompt},  # ← Selected prompt used here
    ]
}]

# Generate extraction response
output = self.model.generate(**inputs, max_new_tokens=max_tokens)
response = self.processor.decode(output[0])

# Parse the response
parsed_data = parse_extraction_response(response, expected_fields=field_list)
```

---

## Summary: Where Selection Occurs

| **Step** | **Location** | **Purpose** |
|----------|--------------|-------------|
| **Config Setup** | `llama_batch.ipynb` Cell 6 | Define PROMPT_CONFIG mapping |
| **Document Detection** | `batch_processor.py:942-1040` | Determine doc type (INVOICE/RECEIPT/BANK_STATEMENT) |
| **Structure Classification** | `batch_processor.py:1049-1070` | For bank statements: flat vs date_grouped |
| **Prompt Selection** | `batch_processor.py:1072-1116` | **CORE LOGIC**: Map doc type → file + key |
| **Extraction** | `batch_processor.py:1165-1262` | Use selected prompt for extraction |

---

## Key Design Features

1. **Explicit Configuration**: `PROMPT_CONFIG` allows full control over which files and keys are used

2. **Fallback Mechanism**: If no explicit key is configured, derives from document type name

3. **Dynamic Suffix**: Bank statements get `_flat` or `_date_grouped` appended automatically (unless you specify the full key)

4. **Override Capability**: You can force a specific prompt by setting `extraction_keys['BANK_STATEMENT'] = 'bank_statement_flat'` to skip vision classification

5. **Single Source of Truth**: All prompt selection goes through `PROMPT_CONFIG` - no hardcoded paths

The selection is **deterministic and traceable** - you can always know which prompt was used by checking the `prompt_name` in the batch results.

---

## Configuration Examples

### Use Old Prompts
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

### Mix Old and New Prompts
```python
PROMPT_CONFIG = {
    'detection_file': 'prompts/document_type_detection.yaml',
    'detection_key': 'detection',
    'extraction_files': {
        'INVOICE': 'prompts/generated/llama_invoice_prompt.yaml',
        'RECEIPT': 'prompts/llama_prompts.yaml',  # Old prompt
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
        'BANK_STATEMENT': 'bank_statement_flat'  # Ignores vision classification
    }
}
```
