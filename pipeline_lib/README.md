# pipeline_lib

**Lightweight, self-contained library for llama_batch_pipeline**

Replaces bloated `common/` dependencies with clean, minimal implementation.

## Overview

- **Total size**: 1,700 lines (vs 3,313 lines from `common/`)
- **Reduction**: 49% smaller
- **Dependencies**: None on `common/` - fully self-contained
- **Technical debt**: Zero - fresh, clean code

## Modules

### `parser.py` (375 lines)
- `hybrid_parse_response()` - Main parsing function
- Handles JSON, plain text, and markdown formats
- Automatic JSON repair for truncated responses
- Date normalization and field post-processing

**Reduced from**: 783 lines (52% smaller)

### `cleaner.py` (733 lines)
- `ExtractionCleaner` class - Field value normalization
- Monetary field cleaning (remove commas, standardize currency)
- List field cleaning (pipe-separation)
- Address field cleaning (remove phone/email)
- Business knowledge validation

**Source**: Direct copy from `common/extraction_cleaner.py` (already clean, no dependencies)

### `evaluator.py` (262 lines)
- `load_ground_truth()` - CSV loading
- `calculate_field_accuracy()` - Field-level accuracy scoring
- Simplified implementation without heavy `config` dependencies

**Reduced from**: 1,797 lines (85% smaller!)

### `stages.py` (292 lines)
- `stage_3_parsing()` - Parse VLM response into structured fields
- `stage_4_cleaning()` - Clean and normalize field values
- `stage_5_evaluation()` - Evaluate against ground truth
- Integrates parser, cleaner, and evaluator into pipeline workflow

**New module** - Provides pandas-compatible pipeline functions

### `__init__.py` (37 lines)
Clean API exports - no legacy code.

## Usage

### Pipeline Workflow (Recommended)
```python
from pipeline_lib import (
    stage_3_parsing,
    stage_4_cleaning,
    stage_5_evaluation,
    ExtractionCleaner,
    load_ground_truth,
)

# Initialize cleaner
cleaner = ExtractionCleaner(debug=False)

# Load ground truth (optional)
ground_truth = load_ground_truth("ground_truth.csv", verbose=True)

# Apply pipeline stages with pandas
df['parsed_fields'] = df['extraction_response'].apply(
    lambda x: stage_3_parsing(x, expected_fields=FIELD_COLUMNS)
)

df['cleaned_fields'] = df['parsed_fields'].apply(
    lambda x: stage_4_cleaning(x, cleaner=cleaner)
)

df['evaluation'] = df.apply(
    lambda row: stage_5_evaluation(row, ground_truth, expected_fields=FIELD_COLUMNS),
    axis=1
)
```

### Individual Functions
```python
from pipeline_lib import (
    hybrid_parse_response,
    ExtractionCleaner,
    load_ground_truth,
    calculate_field_accuracy,
)

# Parse VLM responses
parsed_fields = hybrid_parse_response(response_text, expected_fields)

# Clean extracted values
cleaner = ExtractionCleaner(debug=False)
cleaned_fields = cleaner.clean_extraction_dict(parsed_fields)

# Load ground truth
ground_truth = load_ground_truth("ground_truth.csv", verbose=True)

# Calculate accuracy
score = calculate_field_accuracy(
    extracted_value="$1234.56",
    ground_truth_value="$1,234.56",
    field_name="TOTAL_AMOUNT",
    debug=False
)
```

## Comparison with common/

| Aspect | common/ | pipeline_lib/ |
|--------|---------|---------------|
| **Total lines** | 3,313 | 1,700 |
| **Dependencies** | config, unified_schema, etc. | None - self-contained |
| **Technical debt** | High | Zero |
| **Maintainability** | Complex | Simple |
| **Purpose** | General framework | Pipeline-specific |
| **Pipeline stages** | Scattered | Unified in stages.py |

## Key Improvements

### 1. No Config Dependencies
- **Before**: Depends on `common.config` for field types
- **After**: Field types inlined or inferred from field names

### 2. No Schema Dependencies
- **Before**: Depends on `common.unified_schema`
- **After**: Always passes `expected_fields` explicitly

### 3. Simplified Evaluation
- **Before**: Complex field type system with 10+ helper functions
- **After**: Simple pattern matching on field names

### 4. Self-Contained
- Each module can be used independently
- No circular dependencies
- Easy to understand and modify

## Future

When ready, `common/` can be deleted. All notebook functionality is now in `pipeline_lib/`.

## Version

**v1.0.0** - Initial extraction from `common/` (2025-01-22)
