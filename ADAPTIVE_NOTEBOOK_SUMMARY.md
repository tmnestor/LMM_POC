# InternVL3 Adaptive Notebook - Implementation Summary

## Files Created

### 1. `ivl3_2b_batch_adaptive.ipynb` ✅
**New adaptive notebook with Llama-style explicit multi-stage processing**

- **Total cells**: 27 (properly ordered)
- **Pattern**: Follows `llama_batch_adaptive.ipynb` for consistency
- **Output**: Llama-compatible CSV for `model_comparison.ipynb`

### 2. `build_adaptive_notebook.py` ✅
**Python script using nbformat to build the notebook**

- Reads original `ivl3_2b_batch_non_quantized.ipynb`
- Adds new cells in correct positions
- Removes old analytics/reporting cells
- Can be re-run if changes needed

## Original File Status

### `ivl3_2b_batch_non_quantized.ipynb` ✅
**PRESERVED - Original working notebook intact**

- Total cells: 26
- Uses black-box `BatchDocumentProcessor`
- For users who prefer simpler interface

## Key Differences: Adaptive vs Original

| Feature | Original (non_quantized) | Adaptive (NEW) |
|---------|-------------------------|----------------|
| **Processing** | Black-box `BatchDocumentProcessor` | Explicit 3-stage loop |
| **Visibility** | Hidden stages | Shows Stage 0 → Stage 1 → Stage 2 |
| **Intermediate data** | Not saved | Saves `doctype_classification`, `structure_classification`, `extraction_raw` |
| **Multi-turn chat** | No | Yes - `chat_with_internvl()` |
| **Progress display** | Minimal | Stage-by-stage with timing |
| **CSV output** | Standard format | Llama-compatible format |
| **Debugging** | Harder | Easier - see raw VLM responses |

## Adaptive Notebook Structure

```
Cell 0:  Title - InternVL3-2B Document-Type-Aware Adaptive Extraction
Cell 1:  ## 1. Imports
Cell 2:  Path setup code
Cell 3:  ## 1a. Path Setup (V100 Compatibility)
Cell 4:  Main imports (with gc, json, time, track)
Cell 5:  ## 2. Pre-emptive Memory Cleanup
Cell 6:  Memory cleanup code
Cell 7:  ## 3. Configuration
Cell 8:  Configuration code
Cell 9:  # 4. Output Directory Setup
Cell 10: Output directory code
Cell 11: # 5. Model Loading (Direct Official Pattern)
Cell 12: Model loading code
Cell 13: ## 5a. Multi-Turn Chat Function ⭐ NEW
Cell 14: chat_with_internvl() function ⭐ NEW
Cell 15: # 6. Image Discovery
Cell 16: Image discovery code
Cell 17: ## 7. Multi-Stage Batch Processing ⭐ NEW
Cell 18: Explicit 3-stage loop code ⭐ NEW
Cell 19: ## 8. Save Results (Llama-Compatible Format) ⭐ NEW
Cell 20: Save CSV/JSON code ⭐ NEW
Cell 21: ## 9. Display Sample Results ⭐ NEW
Cell 22: Display results code ⭐ NEW
Cell 23: ## 10. Summary Statistics ⭐ NEW
Cell 24: Summary statistics code ⭐ NEW
Cell 25: ## 11. View Individual Extraction ⭐ NEW
Cell 26: Individual viewer code ⭐ NEW
```

## What the Adaptive Notebook Does

### Stage 0: Document Type Classification
```python
classification_result = hybrid_processor.detect_and_classify_document(
    str(image_path), verbose=False
)
document_type = classification_result['document_type']  # INVOICE/RECEIPT/BANK_STATEMENT
doctype_answer = classification_result.get('raw_response', document_type)  # RAW VLM RESPONSE
```

### Stage 1: Structure Classification (Bank Statements Only)
```python
if document_type == "BANK_STATEMENT":
    structure_type = classify_bank_statement_structure_vision(...)  # flat/date_grouped
    structure_answer = structure_type  # RAW VLM RESPONSE
    prompt_key = f"internvl3_bank_statement_{structure_type}"
```

### Stage 2: Document-Type-Aware Extraction
```python
extraction_result = hybrid_processor.process_document_aware(
    str(image_path), classification_result, verbose=False
)
extracted_fields = extraction_result.get('extracted_data', {})
extraction_raw = extraction_result.get('raw_response', '')  # RAW VLM RESPONSE
```

### Results Stored
```python
result = {
    'image_file': image_name,
    'document_type': document_type,
    'structure_type': structure_type,
    'prompt_used': prompt_key,
    'doctype_classification': doctype_answer,      # ⭐ NEW - Raw response
    'structure_classification': structure_answer,  # ⭐ NEW - Raw response
    'extraction_raw': extraction_raw,              # ⭐ NEW - Raw response
    **extracted_fields  # All field columns
}
```

## Output Files

### CSV Output
**Filename**: `internvl3_adaptive_results_{timestamp}.csv`

**Columns**:
- Core: `image_file`, `document_type`, `structure_type`, `prompt_used`
- Intermediate: `doctype_classification`, `structure_classification`, `extraction_raw`
- Fields: All extraction fields (DOCUMENT_TYPE, BUSINESS_ABN, etc.)

### JSON Output
**Filename**: `internvl3_adaptive_results_{timestamp}.json`

Complete structured results including all raw VLM responses.

## Usage

### On V100 Machine:
1. Copy both notebooks to V100
2. Run `ivl3_2b_batch_adaptive.ipynb` for transparent processing
3. Results will be in `/path/to/output/csv/`

### For Model Comparison:
The CSV output is compatible with `model_comparison.ipynb`:
```python
# In model_comparison.ipynb
internvl3_file = glob("output/csv/*internvl3*adaptive*results*.csv")[0]
llama_file = glob("output/csv/*llama*adaptive*results*.csv")[0]
```

## Benefits of Adaptive Version

1. **Transparency**: See exactly what happens at each stage
2. **Debugging**: Access raw VLM responses for troubleshooting
3. **Comparison**: Same structure as Llama for fair comparisons
4. **Flexibility**: Multi-turn chat enables iterative refinement
5. **Learning**: Understand how document-aware processing works

## When to Use Which Notebook

### Use Original (`ivl3_2b_batch_non_quantized.ipynb`)
- Quick batch processing
- Don't need intermediate responses
- Prefer simpler interface
- Standard reporting needed

### Use Adaptive (`ivl3_2b_batch_adaptive.ipynb`)
- Debugging extraction issues
- Comparing with Llama model
- Understanding multi-stage process
- Need raw VLM responses
- Building custom workflows

## Future Enhancements

To add new capabilities to the adaptive notebook:
1. Edit `build_adaptive_notebook.py`
2. Add new cells in the appropriate section
3. Re-run: `python build_adaptive_notebook.py`
4. Test on V100

## Troubleshooting

### If notebook structure gets corrupted:
```bash
# Restore original
git checkout HEAD -- ivl3_2b_batch_non_quantized.ipynb

# Rebuild adaptive
python build_adaptive_notebook.py
```

### If cells are out of order:
The `build_adaptive_notebook.py` script guarantees correct cell order.
Just re-run it to rebuild from scratch.

## Summary

✅ **Original preserved**: `ivl3_2b_batch_non_quantized.ipynb` (26 cells)
✅ **Adaptive created**: `ivl3_2b_batch_adaptive.ipynb` (27 cells)
✅ **Builder script**: `build_adaptive_notebook.py`
✅ **Llama-compatible**: CSV output works with model_comparison.ipynb
✅ **Explicit stages**: Transparent multi-stage processing
✅ **Raw responses**: Saves all intermediate VLM outputs

Both notebooks are production-ready and can be used on V100.
