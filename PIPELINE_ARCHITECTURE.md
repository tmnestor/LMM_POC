# Document-Aware Processing Pipeline Architecture

## Pipeline Flow Explanation

### 1. WRE Document (Top)
This represents the input - a "Whole Receipt/Invoice/Statement Extraction" document image that enters the system.

### 2. Document Type Detection Prompt (Blue)
This corresponds to the document type detection phase in your code. The system first identifies what type of document it's processing using prompts from `prompts/document_type_detection.yaml`.

### 3. Document Type Decision (Diamond)
The system classifies the document into one of three categories:
- Invoice
- Receipt
- Bank Statement

### 4. Document-Specific Prompts (Purple boxes)
Based on the detected document type, the system selects the appropriate extraction prompt:
- **Invoice Prompt** → Uses prompts from `prompts/llama_prompts.yaml` with key `'invoice'`
- **Receipt Prompt** → Uses prompts from `prompts/llama_prompts.yaml` with key `'receipt'`
- **Bank Statement Prompt** → Uses prompts from `prompts/llama_prompts.yaml` with key `'bank_statement'`

### 5. Vision-Language Model Processing (Red)
This is where either Llama-3.2-Vision or InternVL3 processes the image with the selected prompt to extract structured data.

### 6. Response Processing & Value Normalization (Yellow)
The raw model output goes through:
- Parsing of the extraction response
- Field validation
- Value normalization (dates, amounts, etc.)
- Fuzzy matching for evaluation

### 7. Structured Extraction (Green)
The final output contains the 25 structured fields defined in your system:
- ABN: Business number
- TOTAL: $137.50
- And 23 other fields as defined in `EXTRACTION_FIELDS`

## Code Implementation

This pipeline is implemented in the `BatchDocumentProcessor` class (`common/batch_processor.py`):

1. **Document type detection** → `_detect_document_type()` method
2. **Prompt selection** → `_get_extraction_prompt()` method using `PROMPT_CONFIG`
3. **Model processing** → `_process_with_vlm()` method
4. **Response parsing** → Uses `parse_extraction_response()` from `common/extraction_parser.py`
5. **Structured output** → Returns extracted data matching `EXTRACTION_FIELDS` schema

This architecture enables document-aware processing where each document type receives optimized prompts, significantly improving extraction accuracy compared to a one-size-fits-all approach.