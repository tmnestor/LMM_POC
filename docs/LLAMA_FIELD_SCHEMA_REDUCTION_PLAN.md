# Llama Field Schema Reduction - Implementation Plan

## Overview
The boss has decided to reduce our current 49-field schema to a focused subset of essential fields. **This document covers implementation for the Llama model only.** InternVL3 updates will be documented in a separate implementation plan.

This plan outlines the changes needed to implement the new reduced field schema for Llama document-aware extraction.

## New Field Schema

### Invoice Documents

#### Basic Level Fields (9 fields)
- `SUPPLIER_ABN` - Where's the invoice from: Supplier ABN
- `SUPPLIER_NAME` - Supplier name  
- `SUPPLIER_ADDRESS` - Supplier address
- `PAYER_NAME` - Who's the invoice to: Payer name
- `PAYER_ADDRESS` - Payer address
- `INVOICE_DATE` - When's the invoice due: Date of the invoice
- `TOTAL_AMOUNT` - How much it is due: Total
- `IS_GST_INCLUDED` - Is it GST included: Yes/No
- `GST_AMOUNT` - Total GST amount

#### Advanced Level Fields (2 additional fields)
- `LINE_ITEM_DESCRIPTIONS` - Line item descriptions
- `LINE_ITEM_TOTAL_PRICES` - Line item total amount

**Total Invoice Fields: 11 fields** (vs current 29 fields - 62% reduction)

### Receipt Documents
**IMPORTANT**: Receipt documents will use the **same 11-field Invoice schema** above. The system still maintains 3 document types (invoice, receipt, bank_statement) for detection purposes, but receipts and invoices now share the same extraction field set.

**Total Receipt Fields: 11 fields** (same as invoices - uses Invoice prompt)

### Bank Statement Documents (4 fields)
- `STATEMENT_DATE_RANGE` - Statement period
- `TRANSACTION_DATES` - Transaction date
- `TRANSACTION_DESCRIPTIONS` - Transaction description  
- `TRANSACTION_AMOUNTS` - Transaction amount

**Total Bank Statement Fields: 4 fields** (vs current 16 fields - 75% reduction)

## Implementation Changes Required

**IMPORTANT**: For all changes, original fields that are to be removed should be **commented out** rather than deleted, making it easy to see what has changed and enabling quick rollback if needed.

### 1. Schema Configuration Updates

**File**: `common/unified_schema.py`
- Update `get_document_schema()` method to return new reduced field sets
- Comment out unused invoice fields (18 fields) while keeping new 11-field set active
- Comment out unused bank statement fields (12 fields) while keeping new 4-field set active  
- **Keep receipt document type** but map it to use the same 11-field Invoice schema
- Update schema logic: `if doc_type in ['invoice', 'receipt']:` return invoice schema
- Use `# SUPER_SET:` prefix for commented out fields for easy identification

### 2. Field Mapping Updates

**File**: `common/config.py`
- Comment out unused field definitions with `# SUPER_SET:` prefix
- Keep new schema field constants active
- Update field count calculations (comment out old values, add new ones)
- Comment out unused field type classifications while maintaining active ones
- Example pattern:
  ```python
  # SUPER_SET: "SUPPLIER_EMAIL",
  # SUPER_SET: "SUPPLIER_WEBSITE", 
  "SUPPLIER_ABN",     # SUBSET: Essential supplier identifier
  "SUPPLIER_NAME",    # SUBSET: Essential supplier info
  ```

### 3. Prompt Template Updates

**Files**: `prompts/*.yaml`
- Comment out unused field instructions with `# SUPER_SET:` prefix
- Keep new field instructions active with `# SUBSET:` comments
- Simplify field instructions for reduced complexity
- Update field-specific extraction instructions for remaining fields
- Reduce prompt length for better performance
- Preserve original YAML structure for easy comparison

### 4. Document-Aware Processor Updates

**File**: 
- `models/document_aware_llama_processor.py`

Changes needed:
- Comment out field type instruction logic for unused fields with `# SUPER_SET:`
- Keep field handling for new schema active with `# SUBSET:` comments
- Update dynamic prompt generation (comment out old field references)
- Adjust max_new_tokens calculations for smaller field sets (keep old calculations commented)
- Preserve method structure for easy rollback

### 5. Evaluation System Updates

**Files**:
- `common/evaluation_metrics.py`
- `common/document_type_metrics.py`

Changes needed:
- Comment out evaluation logic for unused fields with `# SUPER_SET:`
- Update ground truth loading to focus on new field schema
- Adjust evaluation metrics for reduced field counts (keep old metrics commented)
- Update accuracy thresholds (preserve old thresholds as comments)
- Modify ATO compliance checking for invoice basic fields only
- Keep evaluation method signatures intact for compatibility

### 6. Notebook Updates

**File**:
- `llama_document_aware.ipynb`

Changes needed:
- Update field count displays with commented old values:
  ```python
  # SUPER_SET: Old count was 49 → 29 for invoices  
  # SUBSET: New count is 49 → 11 for invoices
  # SUPER_SET: Old count was 49 → 16 for statements
  # SUBSET: New count is 49 → 4 for statements
  ```
- Update efficiency calculations (preserve old calculations as comments)
- Adjust ground truth comparison tables (may show fewer rows)
- Update documentation and field descriptions in markdown cells
- Add comments explaining the reduction for clarity

### 7. Ground Truth Data Updates

**Files**: 
- `evaluation_data/ground_truth.csv` (current working file)
- `evaluation_data/ground_truth_49_fields.csv` (backup of original 49-field schema)

**IMPORTANT**: The original 49-field ground truth has already been backed up and saved as `evaluation_data/ground_truth_49_fields.csv` for preservation and potential rollback.

Changes needed:
- Verify all new required fields exist in current ground truth
- **Do NOT delete unused field columns** - leave them for potential rollback  
- Update evaluation scripts to ignore unused columns rather than removing them
- Validate data quality for reduced field set
- Add comments in evaluation scripts indicating which fields are being ignored
- Reference backup file in documentation: `ground_truth_49_fields.csv` contains full original schema

## Performance Impact Analysis

### Expected Improvements
- **Processing Speed**: ~60-75% reduction in fields should improve inference speed significantly
- **Memory Usage**: Reduced prompt size and shorter responses
- **Accuracy**: Focusing on essential fields may improve per-field accuracy
- **Cost**: Reduced token usage for API-based deployments

### Potential Risks
- **Information Loss**: Some previously captured data will no longer be extracted
- **Use Case Limitations**: Advanced analytics requiring detailed fields may be affected
- **Backwards Compatibility**: Existing integrations expecting 49 fields will break

## Implementation Priority

### Phase 1: Core Schema (High Priority)
1. Update schema definitions and field mappings
2. Update document-aware processors 
3. Test basic functionality with new field sets

### Phase 2: Evaluation & Validation (Medium Priority)
1. Update evaluation system
2. Validate ground truth compatibility
3. Performance testing and optimization

### Phase 3: Documentation & Notebooks (Lower Priority)
1. Update interactive notebooks
2. Update documentation
3. User interface adjustments

## Testing Strategy

### Unit Tests
- Test schema loading with new field definitions
- Validate prompt generation for reduced fields
- Test document type detection (invoice, receipt, bank_statement) with receipt→invoice mapping
- Verify that receipts get routed to invoice schema correctly
- Verify ground truth loading works with both files:
  - `ground_truth.csv` (current working file)
  - `ground_truth_49_fields.csv` (backup for rollback testing)

### Integration Tests  
- End-to-end extraction with new field schema
- Test all 3 document types: invoice (11 fields), receipt (11 fields), bank_statement (4 fields)
- Verify receipt documents use invoice prompts correctly
- Ground truth evaluation accuracy
- Performance benchmarking vs current 49-field system

### User Acceptance Testing
- Validate business requirements are met with reduced fields
- Confirm essential information is still captured
- Performance meets production requirements

## Migration Considerations

### Data Migration
- Existing extractions may need field mapping to new schema
- **Original 49-field ground truth**: Already backed up as `evaluation_data/ground_truth_49_fields.csv`
- Archive or backup current 49-field extraction results before schema change
- **Advantage of commenting approach**: Easy rollback by uncommenting fields
- **Advantage of ground truth backup**: Can easily restore full evaluation capability
- Plan transition period for dual schema support if needed
- Git history will clearly show what was removed vs kept

### API Compatibility
- Update API responses to return new field structure
- Consider versioned APIs if backwards compatibility needed
- Update client integrations expecting 49 fields

## Success Metrics

### Primary Goals
- **Extraction Speed**: Target >60% improvement in processing time
- **Accuracy**: Maintain or improve per-field accuracy on essential fields  
- **Resource Usage**: Reduce memory and compute requirements

### Quality Assurance
- All 11 invoice fields extracted with >95% accuracy
- All 4 bank statement fields extracted with >95% accuracy
- System stability maintained on V100 GPU hardware
- No regression in core functionality

## Comment Prefixes for Easy Tracking

To make changes easily visible and reversible, use these standardized comment prefixes:

- `# SUPER_SET:` - Fields/code removed due to boss's schema reduction
- `# SUBSET:` - Fields/code kept in new schema (for emphasis)
- `# OLD_COUNT:` - Previous field counts and calculations  
- `# NEW_COUNT:` - Updated field counts and calculations

### Example Implementation Pattern:
```python
# Invoice field definitions
invoice_fields = [
    "SUPPLIER_ABN",           # SUBSET: Essential supplier identifier
    "SUPPLIER_NAME",          # SUBSET: Essential supplier info  
    "SUPPLIER_ADDRESS",       # SUBSET: Essential supplier info
    # SUPER_SET: "SUPPLIER_EMAIL",
    # SUPER_SET: "SUPPLIER_WEBSITE", 
    # SUPER_SET: "SUPPLIER_PHONE",
    "PAYER_NAME",            # SUBSET: Essential payer info
    "PAYER_ADDRESS",         # SUBSET: Essential payer info  
    # SUPER_SET: "PAYER_ABN",
    # SUPER_SET: "PAYER_PHONE",
    # SUPER_SET: "PAYER_EMAIL",
]

# OLD_COUNT: 29 invoice fields
# NEW_COUNT: 11 invoice fields  
INVOICE_FIELD_COUNT = 11  # SUBSET: Reduced from 29
```

This approach ensures:
- **Clear visibility** of what changed
- **Easy rollback** by uncommenting lines
- **Preserved functionality** during testing
- **Git diff clarity** showing exactly what the boss requested
- **SUPER_SET** clearly indicates the full original field set
- **SUBSET** clearly indicates the reduced field set

---

## Related Documentation

**InternVL3 Implementation**: A separate implementation plan will be created for InternVL3 model updates: `INTERNVL3_FIELD_SCHEMA_REDUCTION_PLAN.md`

---

**Next Steps**: Await approval to proceed with Phase 1 Llama implementation of the reduced field schema using the commenting approach.