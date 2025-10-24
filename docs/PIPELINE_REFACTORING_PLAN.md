# Pipeline Refactoring Plan: llama_batch_pipeline.ipynb

## Executive Summary

Transform `llama_batch_adaptive.ipynb` into `llama_batch_pipeline.ipynb` - a transparent, pandas-based pipeline architecture optimized for processing 10,000+ images with clear separation of concerns and integrated parsing/cleaning capabilities.

## Objectives

1. **Transparent data flow**: Each stage (extraction → parsing → cleaning → evaluation) is visible and inspectable
2. **Scalability**: Use pandas `.apply()` for efficient batch processing of 10,000+ images
3. **Debuggability**: Inspect intermediate results at each pipeline stage
4. **Integration**: Incorporate `extraction_parser.py` and `extraction_cleaner.py` capabilities
5. **Maintainability**: Clear, linear code flow that's easy to understand and modify

## Current Architecture vs. Proposed Architecture

### Current Architecture (llama_batch_adaptive.ipynb)

```python
# Single-pass processing loop
for image_path in images:
    # Stage 0: Document type detection
    doctype_answer, messages = chat_with_mllm(...)
    document_type = parse_document_type(doctype_answer)

    # Stage 1: Structure classification (bank statements only)
    if document_type == "BANK_STATEMENT":
        structure_answer, messages = chat_with_mllm(...)
        structure_type = parse_structure_type(structure_answer)

    # Stage 2: Extraction
    extraction_result, messages = chat_with_mllm(...)
    extracted_fields = parse_extraction(extraction_result)  # Simple parsing

    results.append({...})  # No cleaning, no evaluation
```

**Issues:**
- ❌ Simple parsing doesn't handle JSON, markdown, multi-line values
- ❌ No field value cleaning (commas, markdown artifacts remain)
- ❌ No business knowledge validation
- ❌ No ground truth evaluation
- ❌ Can't inspect intermediate stages
- ❌ Hard to debug failures

### Proposed Architecture (llama_batch_pipeline.ipynb)

```python
# Create DataFrame with images
df = pd.DataFrame({'image_path': image_files})

# STAGE 0: Document Type Detection
df['doctype_raw'] = df.apply(stage_0_doctype_detection, axis=1)
df['document_type'] = df['doctype_raw'].apply(parse_document_type)

# STAGE 1: Structure Classification (bank statements only)
df['structure_raw'] = df.apply(stage_1_structure_classification, axis=1)
df['structure_type'] = df['structure_raw'].apply(parse_structure_type)

# STAGE 2: Extraction
df['extraction_raw'] = df.apply(stage_2_extraction, axis=1)

# STAGE 3: Parsing (Text → Fields)
df['parsed_fields'] = df['extraction_raw'].apply(stage_3_parsing)

# STAGE 4: Cleaning (Normalize & Validate)
df['cleaned_fields'] = df['parsed_fields'].apply(stage_4_cleaning)

# STAGE 5: Evaluation (Optional - if ground truth available)
if ground_truth_available:
    df['evaluation'] = df.apply(stage_5_evaluation, axis=1)

# Can inspect at any stage:
print(df[['image_path', 'document_type']].head())
print(df.loc[0, 'extraction_raw'])  # Debug specific image
print(df.loc[0, 'parsed_fields'])
print(df.loc[0, 'cleaned_fields'])
```

**Benefits:**
- ✅ Clear separation of concerns
- ✅ Inspectable at every stage
- ✅ Easy to debug specific images
- ✅ Can checkpoint between stages
- ✅ Robust parsing with JSON + markdown support
- ✅ Field value cleaning and validation
- ✅ Optional ground truth evaluation
- ✅ Scalable to 10,000+ images

## Pipeline Stage Definitions

### Stage 0: Document Type Detection

**Purpose**: Classify document as INVOICE, RECEIPT, or BANK_STATEMENT

**Implementation:**
```python
def stage_0_doctype_detection(row):
    """
    Stage 0: Document type detection.

    Args:
        row: DataFrame row with 'image_path' column

    Returns:
        dict: {'raw_response': str, 'processing_time': float}
    """
    image_path = row['image_path']
    image = Image.open(image_path)
    images = [image]
    messages = []

    start_time = time.time()

    doctype_answer, messages = chat_with_mllm(
        model, processor, DOCTYPE_PROMPT, images, messages,
        max_new_tokens=CONFIG['MAX_NEW_TOKENS_DOCTYPE']
    )

    processing_time = time.time() - start_time
    image.close()

    return {
        'raw_response': doctype_answer,
        'processing_time': processing_time,
        'messages': messages  # Preserve for multi-turn chat
    }

# Apply to DataFrame
df['doctype_raw'] = df.apply(stage_0_doctype_detection, axis=1)

# Extract components
df['doctype_response'] = df['doctype_raw'].apply(lambda x: x['raw_response'])
df['doctype_time'] = df['doctype_raw'].apply(lambda x: x['processing_time'])
df['messages_after_stage0'] = df['doctype_raw'].apply(lambda x: x['messages'])

# Parse document type
df['document_type'] = df['doctype_response'].apply(parse_document_type)
```

### Stage 1: Structure Classification (Bank Statements Only)

**Purpose**: Classify bank statement structure as FLAT or DATE_GROUPED

**Implementation:**
```python
def stage_1_structure_classification(row):
    """
    Stage 1: Structure classification for bank statements.

    Args:
        row: DataFrame row with 'image_path', 'document_type', 'messages_after_stage0'

    Returns:
        dict: {'raw_response': str, 'processing_time': float, 'messages': list}
             or None if not a bank statement
    """
    if row['document_type'] != 'BANK_STATEMENT':
        return None

    image_path = row['image_path']
    image = Image.open(image_path)
    images = [image]
    messages = row['messages_after_stage0'].copy()

    start_time = time.time()

    structure_answer, messages = chat_with_mllm(
        model, processor, STRUCTURE_CLASSIFICATION_PROMPT, images, messages,
        max_new_tokens=CONFIG['MAX_NEW_TOKENS_STRUCTURE']
    )

    processing_time = time.time() - start_time
    image.close()

    return {
        'raw_response': structure_answer,
        'processing_time': processing_time,
        'messages': messages
    }

# Apply to DataFrame
df['structure_raw'] = df.apply(stage_1_structure_classification, axis=1)

# Extract components (handle None for non-bank-statements)
df['structure_response'] = df['structure_raw'].apply(
    lambda x: x['raw_response'] if x else 'N/A'
)
df['structure_time'] = df['structure_raw'].apply(
    lambda x: x['processing_time'] if x else 0
)
df['messages_after_stage1'] = df['structure_raw'].apply(
    lambda x: x['messages'] if x else None
)

# Parse structure type
df['structure_type'] = df['structure_response'].apply(parse_structure_type)
```

### Stage 2: Extraction

**Purpose**: Extract field values using document-type and structure-aware prompts

**Implementation:**
```python
def stage_2_extraction(row):
    """
    Stage 2: Field extraction with document-type-aware prompts.

    Args:
        row: DataFrame row with all previous stage data

    Returns:
        dict: {'raw_response': str, 'processing_time': float, 'prompt_used': str}
    """
    image_path = row['image_path']
    document_type = row['document_type']
    structure_type = row['structure_type']

    # Determine which prompt to use
    if document_type == 'BANK_STATEMENT':
        extraction_prompt = BANK_PROMPTS[structure_type]
        prompt_key = f"bank_statement_{structure_type}"
        messages = row['messages_after_stage1'].copy()
    elif document_type == 'INVOICE':
        extraction_prompt = INVOICE_PROMPT
        prompt_key = "invoice"
        messages = row['messages_after_stage0'].copy()
    elif document_type == 'RECEIPT':
        extraction_prompt = RECEIPT_PROMPT
        prompt_key = "receipt"
        messages = row['messages_after_stage0'].copy()
    else:
        # Fallback
        extraction_prompt = INVOICE_PROMPT
        prompt_key = "invoice_fallback"
        messages = row['messages_after_stage0'].copy()

    image = Image.open(image_path)
    images = [image]

    start_time = time.time()

    extraction_result, messages = chat_with_mllm(
        model, processor, extraction_prompt, images, messages,
        max_new_tokens=CONFIG['MAX_NEW_TOKENS_EXTRACT']
    )

    processing_time = time.time() - start_time
    image.close()

    return {
        'raw_response': extraction_result,
        'processing_time': processing_time,
        'prompt_used': prompt_key
    }

# Apply to DataFrame
df['extraction_raw'] = df.apply(stage_2_extraction, axis=1)

# Extract components
df['extraction_response'] = df['extraction_raw'].apply(lambda x: x['raw_response'])
df['extraction_time'] = df['extraction_raw'].apply(lambda x: x['processing_time'])
df['prompt_used'] = df['extraction_raw'].apply(lambda x: x['prompt_used'])

# Calculate total processing time
df['total_time'] = df['doctype_time'] + df['structure_time'] + df['extraction_time']
```

### Stage 3: Parsing (Text → Fields)

**Purpose**: Convert raw text responses into structured field dictionaries

**Implementation:**
```python
from common.extraction_parser import hybrid_parse_response

# Define expected fields (matching ground truth)
FIELD_COLUMNS = [
    'DOCUMENT_TYPE', 'BUSINESS_ABN', 'SUPPLIER_NAME', 'BUSINESS_ADDRESS',
    'PAYER_NAME', 'PAYER_ADDRESS', 'INVOICE_DATE', 'LINE_ITEM_DESCRIPTIONS',
    'LINE_ITEM_QUANTITIES', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES',
    'IS_GST_INCLUDED', 'GST_AMOUNT', 'TOTAL_AMOUNT', 'STATEMENT_DATE_RANGE',
    'TRANSACTION_DATES', 'TRANSACTION_AMOUNTS_PAID'
]

def stage_3_parsing(extraction_response):
    """
    Stage 3: Parse extraction response into structured fields.

    Uses hybrid_parse_response which handles:
    - JSON format responses
    - Plain text key:value format
    - Markdown formatted responses
    - Multi-line values
    - Truncated JSON repair

    Args:
        extraction_response: Raw text response from VLM

    Returns:
        dict: Parsed fields (all fields present, NOT_FOUND if missing)
    """
    parsed_fields = hybrid_parse_response(
        response_text=extraction_response,
        expected_fields=FIELD_COLUMNS
    )
    return parsed_fields

# Apply to DataFrame
df['parsed_fields'] = df['extraction_response'].apply(stage_3_parsing)

# Can inspect parsed fields
print("Sample parsed fields:")
print(df.loc[0, 'parsed_fields'])
```

**What hybrid_parse_response provides:**
- ✅ **JSON parsing**: Handles `{"FIELD": "value"}` format
- ✅ **JSON repair**: Fixes truncated JSON responses
- ✅ **Plain text parsing**: Handles `FIELD: value` format
- ✅ **Markdown removal**: Strips `**`, `*`, bullet points
- ✅ **Multi-line values**: Collects bullet lists for LINE_ITEM fields
- ✅ **Conversation artifacts**: Removes Llama chat patterns
- ✅ **Document type normalization**: STATEMENT → BANK_STATEMENT
- ✅ **List field conversion**: Commas/markdown → pipe-separated
- ✅ **Date normalization**: Various formats → DD/MM/YYYY

### Stage 4: Cleaning (Normalize & Validate)

**Purpose**: Clean and normalize field values, apply business knowledge validation

**Implementation:**
```python
from common.extraction_cleaner import ExtractionCleaner

# Initialize cleaner
cleaner = ExtractionCleaner(debug=CONFIG.get('VERBOSE', False))

def stage_4_cleaning(parsed_fields):
    """
    Stage 4: Clean and normalize field values.

    Uses ExtractionCleaner which provides:
    - Monetary field cleaning (remove suffixes, standardize currency, remove commas)
    - List field cleaning (markdown removal, consistent pipe-separation)
    - Address field cleaning (remove phone numbers, emails, commas)
    - ID field normalization (ABN, BSB formatting)
    - Business knowledge validation (pricing, GST calculations)

    Args:
        parsed_fields: Dictionary of parsed field values

    Returns:
        dict: Cleaned and validated field values
    """
    cleaned_fields = cleaner.clean_extraction_dict(parsed_fields)
    return cleaned_fields

# Apply to DataFrame
df['cleaned_fields'] = df['parsed_fields'].apply(stage_4_cleaning)

# Can inspect cleaning results
print("Sample cleaned fields:")
print(df.loc[0, 'cleaned_fields'])

# Can compare before/after cleaning
print("\nBefore cleaning:")
print(df.loc[0, 'parsed_fields']['TOTAL_AMOUNT'])
print("\nAfter cleaning:")
print(df.loc[0, 'cleaned_fields']['TOTAL_AMOUNT'])
```

**What ExtractionCleaner provides:**

**Monetary Fields** (AMOUNT, PRICE, GST, TOTAL):
- ✅ Remove "each", "per item", "per unit" suffixes
- ✅ Standardize currency symbols (AUD → $)
- ✅ **Remove commas**: `$4,834.03` → `$4834.03` (critical for ground truth matching)
- ✅ Clean up spacing

**List Fields** (LINE_ITEM_*, TRANSACTION_*):
- ✅ Remove markdown from each item
- ✅ Convert all formats to pipe-separated: `item1 | item2 | item3`
- ✅ Preserve positional arrays for bank statement TRANSACTION_AMOUNTS_*
- ✅ Clean individual items (monetary cleaning for price lists)

**Address Fields**:
- ✅ Remove embedded phone numbers
- ✅ Remove email addresses
- ✅ Remove contact info suffixes like "P: (03) 1234 5678"
- ✅ **Remove commas**: `123 Main St, City` → `123 Main St City`

**ID Fields** (ABN, BSB):
- ✅ Normalize spacing: `12 345 678 901` format for ABN
- ✅ Normalize BSB format: `123-456`

**Business Knowledge Validation**:
- ✅ LINE_ITEM_PRICES validation: unit_price × quantity = total_price
- ✅ GST consistency: Check GST calculations match totals
- ✅ Cross-field validation

### Stage 5: Evaluation (Optional)

**Purpose**: Compare cleaned extractions against ground truth

**Implementation:**
```python
from common.evaluation_metrics import load_ground_truth, evaluate_extraction

# Load ground truth (conditional based on mode)
if not CONFIG.get('INFERENCE_ONLY', False) and CONFIG.get('GROUND_TRUTH'):
    ground_truth = load_ground_truth(CONFIG['GROUND_TRUTH'], verbose=True)
else:
    ground_truth = None
    print("Running in inference-only mode - no ground truth evaluation")

def stage_5_evaluation(row):
    """
    Stage 5: Evaluate extraction against ground truth.

    Args:
        row: DataFrame row with 'image_path' and 'cleaned_fields'

    Returns:
        dict: Evaluation metrics or None if no ground truth
    """
    if ground_truth is None:
        return None

    image_name = Path(row['image_path']).name

    if image_name not in ground_truth:
        return {'error': 'No ground truth available'}

    gt_data = ground_truth[image_name]
    extracted_data = row['cleaned_fields']

    evaluation_result = evaluate_extraction(
        extracted_data=extracted_data,
        ground_truth=gt_data
    )

    return evaluation_result

# Apply to DataFrame (only if ground truth available)
if ground_truth is not None:
    df['evaluation'] = df.apply(stage_5_evaluation, axis=1)

    # Extract accuracy metrics
    df['overall_accuracy'] = df['evaluation'].apply(
        lambda x: x.get('overall_accuracy', 0) * 100 if x else None
    )
    df['fields_matched'] = df['evaluation'].apply(
        lambda x: x.get('fields_matched', 0) if x else None
    )
    df['fields_extracted'] = df['evaluation'].apply(
        lambda x: x.get('fields_extracted', 0) if x else None
    )

    # Display evaluation summary
    print(f"\nAverage Accuracy: {df['overall_accuracy'].mean():.2f}%")
    print(f"Median Accuracy: {df['overall_accuracy'].median():.2f}%")
else:
    df['evaluation'] = None
    df['overall_accuracy'] = None
    print("\nInference-only mode - no accuracy metrics")
```

## Pandas Apply Patterns

### Basic Apply Pattern

```python
# Single column apply
df['result'] = df['input_column'].apply(function)

# Multi-column apply (access full row)
df['result'] = df.apply(lambda row: function(row['col1'], row['col2']), axis=1)

# Or with named function
def process_row(row):
    return function(row['col1'], row['col2'])

df['result'] = df.apply(process_row, axis=1)
```

### Progress Tracking with Apply

```python
from rich.progress import track

# Manual iteration with progress bar
results = []
for idx, row in track(df.iterrows(), total=len(df), description="Processing"):
    result = stage_function(row)
    results.append(result)

df['result'] = results
```

### Parallel Processing (for 10,000+ images)

```python
# Using pandarallel for parallel apply
from pandarallel import pandarallel

# Initialize parallel processing
pandarallel.initialize(nb_workers=4, progress_bar=True)

# Use parallel_apply instead of apply
df['result'] = df.parallel_apply(stage_function, axis=1)
```

## Checkpointing Strategy

### Save Intermediate Results

```python
# Save after each major stage
CHECKPOINT_DIR = Path(CONFIG['OUTPUT_DIR']) / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

# After Stage 2: Extraction
df[['image_path', 'extraction_response', 'extraction_time']].to_csv(
    CHECKPOINT_DIR / f'stage2_extraction_{TIMESTAMP}.csv',
    index=False
)

# After Stage 3: Parsing
df[['image_path', 'parsed_fields']].to_pickle(
    CHECKPOINT_DIR / f'stage3_parsed_{TIMESTAMP}.pkl'
)

# After Stage 4: Cleaning
df[['image_path', 'cleaned_fields']].to_pickle(
    CHECKPOINT_DIR / f'stage4_cleaned_{TIMESTAMP}.pkl'
)
```

### Resume from Checkpoint

```python
# Resume from parsing stage
df = pd.read_pickle(CHECKPOINT_DIR / 'stage3_parsed_20251022_143000.pkl')

# Continue with cleaning stage
df['cleaned_fields'] = df['parsed_fields'].apply(stage_4_cleaning)
```

## Debugging Utilities

### Inspect Specific Image at Each Stage

```python
def inspect_pipeline(df, image_name, stages=['all']):
    """
    Inspect all pipeline stages for a specific image.

    Args:
        df: DataFrame with pipeline results
        image_name: Name of image to inspect (e.g., 'image_003.png')
        stages: List of stages to show or 'all'
    """
    row = df[df['image_path'].str.contains(image_name)].iloc[0]

    print(f"\n{'='*80}")
    print(f"Pipeline Inspection: {image_name}")
    print(f"{'='*80}")

    if 'all' in stages or 'doctype' in stages:
        print(f"\n[STAGE 0: Document Type Detection]")
        print(f"Response: {row['doctype_response']}")
        print(f"Parsed: {row['document_type']}")
        print(f"Time: {row['doctype_time']:.2f}s")

    if 'all' in stages or 'structure' in stages:
        print(f"\n[STAGE 1: Structure Classification]")
        print(f"Response: {row['structure_response']}")
        print(f"Parsed: {row['structure_type']}")
        print(f"Time: {row['structure_time']:.2f}s")

    if 'all' in stages or 'extraction' in stages:
        print(f"\n[STAGE 2: Extraction]")
        print(f"Prompt Used: {row['prompt_used']}")
        print(f"Response ({len(row['extraction_response'])} chars):")
        print(row['extraction_response'][:500] + "..." if len(row['extraction_response']) > 500 else row['extraction_response'])
        print(f"Time: {row['extraction_time']:.2f}s")

    if 'all' in stages or 'parsing' in stages:
        print(f"\n[STAGE 3: Parsing]")
        parsed = row['parsed_fields']
        found_fields = [k for k, v in parsed.items() if v != 'NOT_FOUND']
        print(f"Found {len(found_fields)}/{len(parsed)} fields:")
        for field in found_fields[:5]:  # Show first 5
            value = parsed[field]
            print(f"  {field}: {value[:50]}..." if len(value) > 50 else f"  {field}: {value}")

    if 'all' in stages or 'cleaning' in stages:
        print(f"\n[STAGE 4: Cleaning]")
        cleaned = row['cleaned_fields']
        found_fields = [k for k, v in cleaned.items() if v != 'NOT_FOUND']
        print(f"Cleaned {len(found_fields)}/{len(cleaned)} fields:")

        # Show before/after for changed fields
        for field in found_fields[:5]:
            parsed_val = row['parsed_fields'][field]
            cleaned_val = cleaned[field]
            if parsed_val != cleaned_val:
                print(f"  {field}:")
                print(f"    Before: {parsed_val[:50]}..." if len(parsed_val) > 50 else f"    Before: {parsed_val}")
                print(f"    After:  {cleaned_val[:50]}..." if len(cleaned_val) > 50 else f"    After:  {cleaned_val}")

    if 'all' in stages or 'evaluation' in stages:
        if 'evaluation' in row and row['evaluation'] is not None:
            print(f"\n[STAGE 5: Evaluation]")
            eval_data = row['evaluation']
            print(f"Overall Accuracy: {row['overall_accuracy']:.2f}%")
            print(f"Fields Matched: {row['fields_matched']}/{row['fields_extracted']}")

    print(f"\n{'='*80}\n")

# Usage
inspect_pipeline(df, 'image_003.png')
inspect_pipeline(df, 'image_005.png', stages=['parsing', 'cleaning'])
```

### Compare Parsing vs Cleaning

```python
def compare_parsing_cleaning(df, image_name):
    """Show side-by-side comparison of parsed vs cleaned fields."""
    row = df[df['image_path'].str.contains(image_name)].iloc[0]

    parsed = row['parsed_fields']
    cleaned = row['cleaned_fields']

    print(f"\nParsing vs Cleaning Comparison: {image_name}")
    print(f"{'Field':<30} {'Parsed':<40} {'Cleaned':<40}")
    print("="*110)

    for field in FIELD_COLUMNS:
        parsed_val = parsed.get(field, 'NOT_FOUND')
        cleaned_val = cleaned.get(field, 'NOT_FOUND')

        if parsed_val != cleaned_val:
            # Truncate long values
            p_display = parsed_val[:37] + "..." if len(parsed_val) > 40 else parsed_val
            c_display = cleaned_val[:37] + "..." if len(cleaned_val) > 40 else cleaned_val

            print(f"{field:<30} {p_display:<40} {c_display:<40}")

# Usage
compare_parsing_cleaning(df, 'image_001.png')
```

### Field Coverage Report

```python
def field_coverage_report(df):
    """Generate field coverage statistics across all images."""
    print("\n" + "="*80)
    print("FIELD COVERAGE REPORT")
    print("="*80)

    coverage_data = []

    for field in FIELD_COLUMNS:
        # Count how many images have this field
        parsed_count = sum(
            1 for idx, row in df.iterrows()
            if row['parsed_fields'].get(field, 'NOT_FOUND') != 'NOT_FOUND'
        )
        cleaned_count = sum(
            1 for idx, row in df.iterrows()
            if row['cleaned_fields'].get(field, 'NOT_FOUND') != 'NOT_FOUND'
        )

        parsed_pct = (parsed_count / len(df)) * 100
        cleaned_pct = (cleaned_count / len(df)) * 100

        coverage_data.append({
            'Field': field,
            'Parsed': f"{parsed_count}/{len(df)} ({parsed_pct:.1f}%)",
            'Cleaned': f"{cleaned_count}/{len(df)} ({cleaned_pct:.1f}%)",
            'Change': cleaned_count - parsed_count
        })

    coverage_df = pd.DataFrame(coverage_data)
    print(coverage_df.to_string(index=False))
    print("="*80 + "\n")

# Usage
field_coverage_report(df)
```

## Performance Optimization

### Memory Management

```python
# Clear GPU cache between stages
import gc

# After extraction stage
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Process in batches for very large datasets
BATCH_SIZE = 1000

for i in range(0, len(df), BATCH_SIZE):
    batch_df = df.iloc[i:i+BATCH_SIZE].copy()

    # Process batch
    batch_df['parsed_fields'] = batch_df['extraction_response'].apply(stage_3_parsing)

    # Update main DataFrame
    df.iloc[i:i+BATCH_SIZE, df.columns.get_loc('parsed_fields')] = batch_df['parsed_fields']

    # Checkpoint
    df.to_pickle(CHECKPOINT_DIR / f'checkpoint_batch_{i}.pkl')

    # Cleanup
    gc.collect()
```

### Parallel Processing Setup

```python
# Install pandarallel
# pip install pandarallel

from pandarallel import pandarallel

# Initialize with optimal worker count
import multiprocessing
num_workers = min(multiprocessing.cpu_count() - 1, 8)  # Leave 1 CPU free

pandarallel.initialize(
    nb_workers=num_workers,
    progress_bar=True,
    verbose=1
)

# Use parallel apply for CPU-intensive stages
df['parsed_fields'] = df['extraction_response'].parallel_apply(stage_3_parsing)
df['cleaned_fields'] = df['parsed_fields'].parallel_apply(stage_4_cleaning)

# Note: GPU stages (extraction) cannot be parallelized this way
```

### Progress Monitoring

```python
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

# Custom progress tracking
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
) as progress:

    # Stage 3: Parsing
    task = progress.add_task("[cyan]Stage 3: Parsing...", total=len(df))
    parsed_results = []
    for idx, row in df.iterrows():
        parsed_results.append(stage_3_parsing(row['extraction_response']))
        progress.update(task, advance=1)
    df['parsed_fields'] = parsed_results

    # Stage 4: Cleaning
    task = progress.add_task("[green]Stage 4: Cleaning...", total=len(df))
    cleaned_results = []
    for idx, row in df.iterrows():
        cleaned_results.append(stage_4_cleaning(row['parsed_fields']))
        progress.update(task, advance=1)
    df['cleaned_fields'] = cleaned_results
```

## Output Formats

### CSV Export with Expanded Fields

```python
# Export cleaned results to CSV with individual field columns
export_data = []

for idx, row in df.iterrows():
    record = {
        'image_file': Path(row['image_path']).name,
        'document_type': row['document_type'],
        'structure_type': row['structure_type'],
        'prompt_used': row['prompt_used'],
        'total_time': row['total_time'],
    }

    # Add all cleaned field values as individual columns
    cleaned = row['cleaned_fields']
    for field in FIELD_COLUMNS:
        record[field] = cleaned.get(field, 'NOT_FOUND')

    # Add evaluation metrics if available
    if row.get('evaluation') is not None:
        record['overall_accuracy'] = row['overall_accuracy']
        record['fields_matched'] = row['fields_matched']
        record['fields_extracted'] = row['fields_extracted']

    export_data.append(record)

export_df = pd.DataFrame(export_data)

# Save
csv_output = OUTPUT_DIRS['csv'] / f'llama_pipeline_results_{TIMESTAMP}.csv'
export_df.to_csv(csv_output, index=False)

print(f"✅ Exported {len(export_df)} rows to {csv_output}")
print(f"   Columns: {len(export_df.columns)}")
```

### JSON Export with Full Pipeline Data

```python
# Export complete pipeline data including intermediate stages
json_data = []

for idx, row in df.iterrows():
    record = {
        'image_file': Path(row['image_path']).name,
        'pipeline_stages': {
            'stage_0_doctype': {
                'raw_response': row['doctype_response'],
                'document_type': row['document_type'],
                'processing_time': row['doctype_time']
            },
            'stage_1_structure': {
                'raw_response': row['structure_response'],
                'structure_type': row['structure_type'],
                'processing_time': row['structure_time']
            },
            'stage_2_extraction': {
                'raw_response': row['extraction_response'],
                'prompt_used': row['prompt_used'],
                'processing_time': row['extraction_time']
            },
            'stage_3_parsing': {
                'parsed_fields': row['parsed_fields']
            },
            'stage_4_cleaning': {
                'cleaned_fields': row['cleaned_fields']
            }
        },
        'total_processing_time': row['total_time']
    }

    if row.get('evaluation') is not None:
        record['evaluation'] = row['evaluation']

    json_data.append(record)

# Save
json_output = OUTPUT_DIRS['batch'] / f'llama_pipeline_full_{TIMESTAMP}.json'
with open(json_output, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"✅ Exported full pipeline data to {json_output}")
```

## Migration Plan: llama_batch_adaptive.ipynb → llama_batch_pipeline.ipynb

### Step 1: Copy Base Notebook

```bash
cp llama_batch_adaptive.ipynb llama_batch_pipeline.ipynb
```

### Step 2: Add New Imports

Add after existing imports:
```python
from common.extraction_parser import hybrid_parse_response
from common.extraction_cleaner import ExtractionCleaner
from common.evaluation_metrics import load_ground_truth, evaluate_extraction
```

### Step 3: Replace Batch Processing Loop

**Old approach (llama_batch_adaptive.ipynb):**
```python
results = []
for idx, image_path in enumerate(track(image_files)):
    # Multi-turn chat processing
    # Direct append to results list
    results.append({...})
```

**New approach (llama_batch_pipeline.ipynb):**
```python
# Create DataFrame
df = pd.DataFrame({'image_path': image_files})

# Apply each stage
# (as detailed in Pipeline Stage Definitions above)
```

### Step 4: Add Pipeline Stage Functions

Add cells for each stage function:
- `stage_0_doctype_detection()`
- `stage_1_structure_classification()`
- `stage_2_extraction()`
- `stage_3_parsing()`
- `stage_4_cleaning()`
- `stage_5_evaluation()` (optional)

### Step 5: Add Debugging Utilities

Add cells with:
- `inspect_pipeline()`
- `compare_parsing_cleaning()`
- `field_coverage_report()`

### Step 6: Update Results Export

Replace simple CSV export with expanded field export as shown in Output Formats section.

### Step 7: Add Checkpointing

Add checkpoint saves after each major stage for resumability.

## Testing Strategy

### Unit Test Each Stage

```python
# Test Stage 3: Parsing
test_response = """
DOCUMENT_TYPE: INVOICE
SUPPLIER_NAME: Test Company
TOTAL_AMOUNT: $1,234.56
"""
parsed = stage_3_parsing(test_response)
assert parsed['DOCUMENT_TYPE'] == 'INVOICE'
assert parsed['SUPPLIER_NAME'] == 'Test Company'
print("✅ Parsing stage test passed")

# Test Stage 4: Cleaning
parsed = {'TOTAL_AMOUNT': '$1,234.56'}
cleaned = stage_4_cleaning(parsed)
assert cleaned['TOTAL_AMOUNT'] == '$1234.56'  # Comma removed
print("✅ Cleaning stage test passed")
```

### Integration Test with Sample Images

```python
# Test with first 3 images
test_df = df.head(3).copy()

# Run through pipeline
test_df['extraction_raw'] = test_df.apply(stage_2_extraction, axis=1)
test_df['parsed_fields'] = test_df['extraction_raw'].apply(
    lambda x: stage_3_parsing(x['raw_response'])
)
test_df['cleaned_fields'] = test_df['parsed_fields'].apply(stage_4_cleaning)

# Inspect results
for idx, row in test_df.iterrows():
    print(f"\n{row['image_path']}")
    inspect_pipeline(test_df, Path(row['image_path']).name)
```

## Expected Outcomes

### Immediate Benefits

1. **Transparency**: Can see exactly what happens at each stage
2. **Debuggability**: Can inspect any image at any stage
3. **Flexibility**: Can skip or modify stages easily
4. **Quality**: Robust parsing and cleaning improves accuracy
5. **Scalability**: Pandas operations efficient for large datasets

### Performance Metrics

For 10,000 images:
- **Sequential processing**: ~8 hours (with 2.9s per image)
- **With checkpointing**: Can resume if interrupted
- **With parallel parsing/cleaning**: ~6 hours (GPU extraction still sequential)
- **Memory usage**: Manageable with batch processing

### Accuracy Improvements

Expected accuracy improvement from adding parsing + cleaning:
- **Before** (simple parsing): ~70-85% accuracy
- **After** (hybrid parsing + cleaning): ~85-95% accuracy

Main improvements from:
- Comma removal in monetary values
- Address field cleaning
- List field normalization
- Document type normalization

## Conclusion

The pipeline architecture provides:
- ✅ Clear separation of concerns
- ✅ Full transparency and inspectability
- ✅ Robust parsing with JSON + markdown support
- ✅ Comprehensive field value cleaning
- ✅ Business knowledge validation
- ✅ Optional ground truth evaluation
- ✅ Scalability to 10,000+ images
- ✅ Easy debugging and maintenance

This architecture makes the codebase easier to understand, debug, and extend while maintaining high performance and accuracy.
