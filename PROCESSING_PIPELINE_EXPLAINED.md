# Processing Pipeline Documentation

**Document Purpose**: Comprehensive explanation of how raw model responses are transformed into structured extraction data in the LMM_POC project.

**Last Updated**: 2025-10-09

---

## Table of Contents

1. [Overview](#overview)
2. [Batch Results Data Structure](#batch-results-data-structure)
3. [Raw Response to Extracted Data Pipeline](#raw-response-to-extracted-data-pipeline)
4. [Detailed Processing Steps](#detailed-processing-steps)
5. [Llama vs InternVL3 Comparison](#llama-vs-internvl3-comparison)
6. [Code References](#code-references)

---

## Overview

This document explains the complete data processing pipeline used in the LMM_POC project for extracting structured business document information using vision-language models (Llama-3.2-Vision and InternVL3).

**Key Pipeline Stages**:
1. Image → Model → Raw text response
2. Raw text → Hybrid parser → Parsed dictionary
3. Parsed dictionary → ExtractionCleaner → Final structured data

---

## Batch Results Data Structure

### What is `batch_results`?

`batch_results` is a **Python list** containing dictionaries, where each dictionary represents the extraction results for one image.

**Type**: `List[Dict[str, Any]]`

### Structure Example

```python
batch_results = [
    {
        'image_path': '/path/to/image_001.png',
        'image_name': 'image_001.png',
        'document_type': 'receipt',
        'extraction_result': {
            'extracted_data': {
                'DOCUMENT_TYPE': 'RECEIPT',
                'SUPPLIER_NAME': 'Pizza Hut',
                'TOTAL_AMOUNT': '$97.95',
                'IS_GST_INCLUDED': 'true',
                'BUSINESS_ABN': '12 345 678 901',
                # ... all 14 or 19 fields depending on document type
            },
            'raw_response': 'DOCUMENT_TYPE: RECEIPT\nSUPPLIER_NAME: Pizza Hut\n...',
            'processing_time': 5.97,
            'response_completeness': 0.93,
            'content_coverage': 0.93,
            'extracted_fields_count': 13,
            'field_count': 14
        },
        'evaluation': {  # Only present in evaluation mode (not inference-only)
            'overall_accuracy': 0.9286,  # Float between 0-1
            'fields_matched': 13,
            'fields_extracted': 14,
            'total_fields': 14,
            'missing_fields': ['PAYER_NAME'],
            'incorrect_fields': []
        }
    },
    # ... more results for each image
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `image_path` | str | Full path to the image file |
| `image_name` | str | Image filename only |
| `document_type` | str | Detected document type (invoice/receipt/bank_statement) |
| `extraction_result` | dict | All extraction-related data |
| `extraction_result.extracted_data` | dict | Final structured fields (key-value pairs) |
| `extraction_result.raw_response` | str | Unprocessed model output text |
| `extraction_result.processing_time` | float | Seconds taken to process image |
| `evaluation` | dict | Accuracy metrics (only in evaluation mode) |
| `evaluation.overall_accuracy` | float | Accuracy as decimal (0.9286 = 92.86%) |

---

## Raw Response to Extracted Data Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Model Generation                                            │
│ Input:  Image pixel values + Extraction prompt                      │
│ Output: Raw text response from vision-language model                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Hybrid Parsing                                              │
│ 2a. Try JSON parsing (fast path for complex documents)              │
│ 2b. Fallback to plain text parsing (two-pass strategy)              │
│ Output: Dictionary with field-value pairs                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Value Cleaning & Normalization                              │
│ Apply ExtractionCleaner to standardize formats                      │
│ Output: Final extracted_data dictionary                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Summary Flow

```
Raw Text Response (from model)
    ↓
hybrid_parse_response() → Try JSON first, fallback to text parsing
    ↓
Parsed Dictionary (field: value pairs)
    ↓
ExtractionCleaner.clean_field_value() → Normalize values
    ↓
Final extracted_data (clean, structured)
```

---

## Detailed Processing Steps

### Step 1: Model Generation

**Location**:
- InternVL3: `models/document_aware_internvl3_processor.py:1024`
- Llama: `models/document_aware_llama_processor.py:445`

**Process**:

#### InternVL3 Generation
```python
# Generate response using InternVL3 chat method
response = self._resilient_generate(
    pixel_values,
    question="<image>\n{prompt}",
    **generation_config
)
```

#### Llama Generation
```python
# Generate response using Llama generate method
output = self._resilient_generate(inputs, **generation_config)

# Decode tokens to text
full_response = self.processor.decode(output[0], skip_special_tokens=True)

# Extract assistant's response from chat template
response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[1]
```

**Example Raw Response**:
```
DOCUMENT_TYPE: RECEIPT
SUPPLIER_NAME: Pizza Hut
BUSINESS_ABN: 12 345 678 901
BUSINESS_ADDRESS: 123 Main St, Melbourne VIC 3000
INVOICE_DATE: 15/03/2024
LINE_ITEM_DESCRIPTIONS: Pepperoni Pizza | Garlic Bread | Coke
LINE_ITEM_QUANTITIES: 1 | 2 | 1
LINE_ITEM_PRICES: $18.95 | $5.00 | $3.50
LINE_ITEM_TOTAL_PRICES: $18.95 | $10.00 | $3.50
IS_GST_INCLUDED: true
GST_AMOUNT: $2.95
TOTAL_AMOUNT: $32.45
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
```

---

### Step 2: Hybrid Parsing

**Location**: `common/extraction_parser.py`

**Entry Points**:
- InternVL3: Calls `hybrid_parse_response()` (line 213)
- Llama: Calls `parse_extraction_response()` (line 253)

#### Step 2a: JSON Parsing (Fast Path)

**Function**: `_try_parse_json()` (line 59-104)

**Purpose**: Handle model responses that use JSON format (common for bank statements with complex structures)

**Process**:
1. **Fast JSON Detection** (line 28-56)
   - Check if response starts with `{` and has JSON structure
   - Handle markdown code blocks (```json ... ```)
   - Minimal overhead - no full parsing unless likely JSON

2. **JSON Repair** (line 106-210)
   - Fix truncated JSON (missing closing braces)
   - Fix incomplete string values (missing quotes)
   - Fix incomplete field assignments

3. **Parse with orjson** (if available) or stdlib json
   - orjson is 3-5x faster than stdlib
   - Automatically used if installed

4. **Convert to expected format**
   ```python
   extracted_data = {field: "NOT_FOUND" for field in expected_fields}
   for field in expected_fields:
       if field in json_data:
           value = json_data[field]
           extracted_data[field] = str(value) if value else "NOT_FOUND"
   ```

**Example JSON Response**:
```json
{
  "DOCUMENT_TYPE": "RECEIPT",
  "SUPPLIER_NAME": "Pizza Hut",
  "TOTAL_AMOUNT": "$32.45",
  "IS_GST_INCLUDED": true
}
```

#### Step 2b: Plain Text Parsing (Fallback)

**Function**: `parse_extraction_response()` (line 253-490)

**Purpose**: Handle plain text key-value format (most common for invoices/receipts)

**Two-Pass Strategy**:

##### First Pass: Standard Line-by-Line Parsing (lines 316-363)
Works for clean outputs like Llama and well-formatted InternVL3 responses.

**Process**:
1. Split response by newlines
2. For each line:
   - Skip empty lines
   - Skip lines without `:` separator
   - Clean markdown formatting (`**FIELD:** → FIELD:`)
   - Fix common issues:
     - `KEY: FIELD_NAME:` → `FIELD_NAME:`
     - `LINE_ITEM_DESCRIPTION:` → `LINE_ITEM_DESCRIPTIONS:`
   - Split on first `:` to get key and value
   - Convert value to string (handles boolean/numeric values)
   - Store if field is in expected fields

3. **Success Criteria**: If ≥50% of fields found with actual values (not "NOT_FOUND")

**Example Processing**:
```python
# Input line:
"**SUPPLIER_NAME:** Pizza Hut"

# After cleaning:
"SUPPLIER_NAME: Pizza Hut"

# After split:
key = "SUPPLIER_NAME"
value = "Pizza Hut"

# Store:
extracted_data["SUPPLIER_NAME"] = "Pizza Hut"
```

##### Second Pass: Markdown Multi-Line Handling (lines 365-490)
Fallback for problematic InternVL3 output with multi-line values.

**Process**:
1. Look for markdown key patterns: `**FIELD_NAME:**`
2. Handle values on same line or next line
3. Collect multi-line values:
   - For `LINE_ITEM_*` fields: Join with ` | ` separator
   - For regular fields: Join with spaces
4. Stop collecting when next field detected

**Example Multi-Line Processing**:
```
**LINE_ITEM_DESCRIPTIONS:**
* Pepperoni Pizza
* Garlic Bread
* Coke

**LINE_ITEM_PRICES:**
$18.95
$5.00
$3.50
```

Becomes:
```python
{
    "LINE_ITEM_DESCRIPTIONS": "Pepperoni Pizza | Garlic Bread | Coke",
    "LINE_ITEM_PRICES": "$18.95 | $5.00 | $3.50"
}
```

---

### Step 3: Value Cleaning & Normalization

**Location**:
- InternVL3: `models/document_aware_internvl3_processor.py:1047-1058`
- Llama: `models/document_aware_llama_processor.py:488-499`

**Purpose**: Standardize field values for consistency and evaluation

**Process**:

```python
cleaned_data = {}
for field in document_fields:
    raw_value = extracted_data.get(field, "NOT_FOUND")
    if raw_value != "NOT_FOUND":
        cleaned_value = self.cleaner.clean_field_value(field, raw_value)
        cleaned_data[field] = cleaned_value
    else:
        cleaned_data[field] = "NOT_FOUND"
```

**ExtractionCleaner Operations** (`common/extraction_cleaner.py`):

| Field Type | Cleaning Operation | Example |
|------------|-------------------|---------|
| **ABN** | Remove spaces, validate 11 digits | `12 345 678 901` → `12345678901` |
| **Currency** | Add $ symbol if missing | `100.00` → `$100.00` |
| **Dates** | Standardize to DD/MM/YYYY | `2024-03-15` → `15/03/2024` |
| **Boolean** | Normalize to lowercase | `True` → `true`, `FALSE` → `false` |
| **List Fields** | Clean each item, join with ` \| ` | `"Item1\|Item2"` → `"Item1 | Item2"` |
| **Addresses** | Trim whitespace, fix formatting | `"  123 Main St  "` → `"123 Main St"` |

**Example Cleaning**:
```python
# Before cleaning:
{
    'BUSINESS_ABN': '12 345 678 901',
    'TOTAL_AMOUNT': '100.00',
    'IS_GST_INCLUDED': 'True',
    'INVOICE_DATE': '2024-03-15'
}

# After cleaning:
{
    'BUSINESS_ABN': '12345678901',
    'TOTAL_AMOUNT': '$100.00',
    'IS_GST_INCLUDED': 'true',
    'INVOICE_DATE': '15/03/2024'
}
```

---

## Llama vs InternVL3 Comparison

### Are the Processing Pipelines the Same?

**Answer**: YES, both models use the exact same processing pipeline.

### Entry Point Differences

| Aspect | InternVL3 | Llama |
|--------|-----------|-------|
| **Entry Function** | `hybrid_parse_response()` | `parse_extraction_response()` |
| **JSON Check** | Explicit in `hybrid_parse_response()` | Built-in to `parse_extraction_response()` |
| **Fallback** | Calls `parse_extraction_response()` | Continues with plain text parsing |

### Implementation Details

**InternVL3 Path**:
```python
# models/document_aware_internvl3_processor.py:1043
extracted_data = hybrid_parse_response(
    response, expected_fields=document_fields
)
```

Internally:
```python
# common/extraction_parser.py:213-250
def hybrid_parse_response(response_text, expected_fields):
    # Step 1: Try JSON first
    json_result = _try_parse_json(response_text.strip(), expected_fields)
    if json_result is not None:
        return json_result

    # Step 2: Fallback to plain text parser
    return parse_extraction_response(
        response_text=response_text,
        clean_conversation_artifacts=False,
        expected_fields=expected_fields
    )
```

**Llama Path**:
```python
# models/document_aware_llama_processor.py:484
extracted_data = parse_extraction_response(
    response, expected_fields=self.field_list
)
```

Internally:
```python
# common/extraction_parser.py:311-314
def parse_extraction_response(response_text, clean_conversation_artifacts, expected_fields):
    # HYBRID PARSING: Try JSON first (fast path for complex documents)
    json_result = _try_parse_json(response_text.strip(), expected_fields)
    if json_result is not None:
        return json_result

    # Continue with plain text parsing...
```

### Conclusion

**Both models use identical parsing logic**:
1. Try JSON parsing first
2. Fallback to plain text parsing (two-pass strategy)
3. Apply ExtractionCleaner for normalization

The only difference is the entry point:
- **InternVL3**: Calls `hybrid_parse_response()` → JSON check → `parse_extraction_response()`
- **Llama**: Calls `parse_extraction_response()` → JSON check (built-in) → plain text parsing

**Result**: Identical processing behavior for both models.

---

## Code References

### Key Files

| File | Purpose |
|------|---------|
| `common/extraction_parser.py` | All parsing logic (JSON + plain text) |
| `common/extraction_cleaner.py` | Value normalization and cleaning |
| `models/document_aware_internvl3_processor.py` | InternVL3 model integration |
| `models/document_aware_llama_processor.py` | Llama model integration |
| `common/batch_processor.py` | Batch processing orchestration |

### Important Functions

#### Parsing Functions
- `hybrid_parse_response()` - line 213 - Main entry for InternVL3
- `parse_extraction_response()` - line 253 - Main entry for Llama
- `_try_parse_json()` - line 59 - JSON parsing with repair
- `_repair_truncated_json()` - line 106 - Fix common JSON issues
- `_fast_json_detection()` - line 28 - Detect JSON format

#### Model-Specific Processing
- **InternVL3**:
  - `process_single_image()` - line 894 - Main processing method
  - `_resilient_generate()` - line 729 - Generation with OOM handling
  - `load_image()` - line 317 - Image preprocessing

- **Llama**:
  - `process_single_image()` - line 322 - Main processing method
  - `_resilient_generate()` - line 272 - Generation with OOM handling
  - `load_document_image()` - line 263 - Image loading

#### Cleaning Functions
- `ExtractionCleaner.clean_field_value()` - Field-specific cleaning
- `ExtractionCleaner.clean_abn()` - ABN normalization
- `ExtractionCleaner.clean_currency()` - Currency formatting
- `ExtractionCleaner.clean_date()` - Date standardization

### Prompt Loading

Both models load prompts from YAML files in `prompts/`:

| Model | Prompt File | Loader Function |
|-------|-------------|-----------------|
| **Llama** | `prompts/llama_prompts.yaml` | `load_llama_prompt()` |
| **InternVL3** | `prompts/internvl3_prompts.yaml` | `load_internvl3_prompt()` |

**Common Infrastructure**: `common/simple_prompt_loader.py`
- `SimplePromptLoader.load_prompt()` - Load prompt from YAML
- `SimplePromptLoader.get_available_prompts()` - List available prompts

---

## Complete Example: End-to-End Processing

### Input: Receipt Image

**Prompt** (from `prompts/internvl3_prompts.yaml`):
```yaml
receipt:
  prompt: |
    Extract ALL data from this receipt image. Respond in exact format below with actual values or NOT_FOUND.

    DOCUMENT_TYPE: RECEIPT
    BUSINESS_ABN: NOT_FOUND
    SUPPLIER_NAME: NOT_FOUND
    ...
```

### Step 1: Model Response

**Raw Response** (from model):
```
DOCUMENT_TYPE: RECEIPT
BUSINESS_ABN: 12 345 678 901
SUPPLIER_NAME: Pizza Hut
BUSINESS_ADDRESS: 123 Main St, Melbourne VIC 3000
PAYER_NAME: NOT_FOUND
PAYER_ADDRESS: NOT_FOUND
INVOICE_DATE: 15/03/2024
LINE_ITEM_DESCRIPTIONS: Pepperoni Pizza | Garlic Bread | Coke
LINE_ITEM_QUANTITIES: 1 | 2 | 1
LINE_ITEM_PRICES: $18.95 | $5.00 | $3.50
LINE_ITEM_TOTAL_PRICES: $18.95 | $10.00 | $3.50
IS_GST_INCLUDED: true
GST_AMOUNT: $2.95
TOTAL_AMOUNT: $32.45
```

### Step 2: Parsing

**After `hybrid_parse_response()`**:
```python
{
    'DOCUMENT_TYPE': 'RECEIPT',
    'BUSINESS_ABN': '12 345 678 901',
    'SUPPLIER_NAME': 'Pizza Hut',
    'BUSINESS_ADDRESS': '123 Main St, Melbourne VIC 3000',
    'PAYER_NAME': 'NOT_FOUND',
    'PAYER_ADDRESS': 'NOT_FOUND',
    'INVOICE_DATE': '15/03/2024',
    'LINE_ITEM_DESCRIPTIONS': 'Pepperoni Pizza | Garlic Bread | Coke',
    'LINE_ITEM_QUANTITIES': '1 | 2 | 1',
    'LINE_ITEM_PRICES': '$18.95 | $5.00 | $3.50',
    'LINE_ITEM_TOTAL_PRICES': '$18.95 | $10.00 | $3.50',
    'IS_GST_INCLUDED': 'true',
    'GST_AMOUNT': '$2.95',
    'TOTAL_AMOUNT': '$32.45'
}
```

### Step 3: Cleaning

**After `ExtractionCleaner.clean_field_value()`**:
```python
{
    'DOCUMENT_TYPE': 'RECEIPT',
    'BUSINESS_ABN': '12345678901',  # Spaces removed
    'SUPPLIER_NAME': 'Pizza Hut',
    'BUSINESS_ADDRESS': '123 Main St, Melbourne VIC 3000',
    'PAYER_NAME': 'NOT_FOUND',
    'PAYER_ADDRESS': 'NOT_FOUND',
    'INVOICE_DATE': '15/03/2024',
    'LINE_ITEM_DESCRIPTIONS': 'Pepperoni Pizza | Garlic Bread | Coke',
    'LINE_ITEM_QUANTITIES': '1 | 2 | 1',
    'LINE_ITEM_PRICES': '$18.95 | $5.00 | $3.50',
    'LINE_ITEM_TOTAL_PRICES': '$18.95 | $10.00 | $3.50',
    'IS_GST_INCLUDED': 'true',  # Normalized to lowercase
    'GST_AMOUNT': '$2.95',
    'TOTAL_AMOUNT': '$32.45'
}
```

### Final Result in `batch_results`

```python
{
    'image_path': '/home/jovyan/nfs_share/tod/evaluation_data/image_001.png',
    'image_name': 'image_001.png',
    'document_type': 'receipt',
    'extraction_result': {
        'extracted_data': {
            # Cleaned data shown above
        },
        'raw_response': 'DOCUMENT_TYPE: RECEIPT\nBUSINESS_ABN: 12 345 678 901\n...',
        'processing_time': 5.975193,
        'response_completeness': 1.0,
        'content_coverage': 1.0,
        'extracted_fields_count': 14,
        'field_count': 14
    },
    'evaluation': {
        'overall_accuracy': 0.9286,  # 92.86%
        'fields_matched': 13,
        'fields_extracted': 14,
        'total_fields': 14,
        'missing_fields': [],
        'incorrect_fields': ['PAYER_NAME']  # Expected value didn't match
    }
}
```

---

## Summary

### Key Takeaways

1. **`batch_results` is a list of dictionaries** - One dict per processed image

2. **Processing pipeline is identical for both models**:
   - Try JSON parsing first (fast path)
   - Fallback to plain text parsing (two-pass strategy)
   - Apply value cleaning and normalization

3. **Three main steps**:
   - **Generation**: Model produces raw text response
   - **Parsing**: Convert text to structured dictionary
   - **Cleaning**: Normalize values for consistency

4. **Robust parsing handles multiple formats**:
   - JSON responses (with repair for truncation)
   - Plain text key-value pairs
   - Multi-line markdown formatting
   - Bullet point lists

5. **ExtractionCleaner ensures consistency**:
   - ABN formatting (remove spaces)
   - Currency formatting (add $ symbol)
   - Date standardization (DD/MM/YYYY)
   - Boolean normalization (lowercase)

---

**Document Version**: 1.0
**Generated**: 2025-10-09
**Author**: Claude Code Analysis
