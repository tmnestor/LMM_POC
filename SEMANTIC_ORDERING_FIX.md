# Semantic Field Ordering Fix - Performance Regression Solution

## Problem Analysis

After implementing Phases 2-4 of the refactoring (Priority 4 & 5), both Llama and InternVL3 models experienced significant performance degradation. The root cause was **semantic ordering disruption**, not field name recognition issues.

### Key Issues Identified:
1. **Alphabetical ordering** put bank-specific fields first (`ACCOUNT_CLOSING_BALANCE`), confusing models about document type
2. **Critical instruction mismatch**: "Start with ACCOUNT_CLOSING_BALANCE" (irrelevant for most docs) vs original "Start with ABN" (relevant for all business docs)
3. **Loss of semantic flow**: Original ABN→TOTAL sequence was intuitive, new ACCOUNT_CLOSING_BALANCE→TOTAL_AMOUNT was not

## Solution: Semantic Field Ordering

### What We Implemented:

1. **Defined Semantic Field Order** in `common/config.py`:
   ```python
   SEMANTIC_FIELD_ORDER = [
       # Document identifiers - start with most universal fields
       "DOCUMENT_TYPE", "BUSINESS_ABN", "SUPPLIER_NAME",
       
       # Business entity details  
       "BUSINESS_ADDRESS", "BUSINESS_PHONE", "SUPPLIER_WEBSITE",
       
       # Payer information
       "PAYER_NAME", "PAYER_ADDRESS", "PAYER_PHONE", "PAYER_EMAIL",
       
       # Temporal data
       "INVOICE_DATE", "DUE_DATE", "STATEMENT_DATE_RANGE",
       
       # Line item details
       "LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES",
       
       # Banking information
       "BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_HOLDER",
       "ACCOUNT_OPENING_BALANCE", "ACCOUNT_CLOSING_BALANCE",
       
       # Financial totals - end with most important field
       "SUBTOTAL_AMOUNT", "GST_AMOUNT", "TOTAL_AMOUNT"
   ]
   ```

2. **Updated YAML Prompt Files**:
   - `llama_single_pass_prompts.yaml` - reordered field_instructions to semantic sequence
   - `internvl3_prompts.yaml` - reordered field_instructions to semantic sequence
   - Updated critical instructions to "Start immediately with DOCUMENT_TYPE"

3. **Modified Field Discovery**:
   - `discover_fields_from_yaml()` now returns fields in semantic order
   - `EXTRACTION_FIELDS` follows semantic sequence instead of alphabetical

4. **Created Debug Tools**:
   - `debug_semantic_ordering.py` - comprehensive prompt and field analysis
   - `test_semantic_fix.py` - validation tests for local verification

### Semantic Flow Logic:

```
🏢 Document Identifiers → 🏪 Business Details → 👤 Payer Info → 
📅 Dates → 📋 Line Items → 🏦 Banking → 💰 Financial Totals
```

This mimics the original ~84% accuracy prompt structure while maintaining all standardized field names.

## Key Improvements:

✅ **Universal start field**: `DOCUMENT_TYPE` instead of `ACCOUNT_CLOSING_BALANCE`  
✅ **Logical grouping**: Related fields clustered together  
✅ **Intuitive flow**: Document type → business info → payer → details → totals  
✅ **Maintains standardization**: All field names from Priority 5 preserved  
✅ **Clear instructions**: "Start with DOCUMENT_TYPE" is universally applicable  

## Files Modified:

- `common/config.py` - Added SEMANTIC_FIELD_ORDER and semantic ordering logic
- `llama_single_pass_prompts.yaml` - Reordered fields and updated instructions
- `internvl3_prompts.yaml` - Reordered fields to match semantic sequence
- `debug_semantic_ordering.py` - New debug/analysis tool
- `test_semantic_fix.py` - New validation test script

## Validation Results:

```
🧪 SEMANTIC ORDERING FIX - VALIDATION TESTS
Tests passed: 3/3
🎉 ALL TESTS PASSED!

✅ Semantic ordering fix is ready for remote testing
✅ Configuration loads correctly  
✅ Field order is optimized for model performance
✅ Prompts are generated with correct structure
```

## Next Steps for Remote Testing:

1. **Deploy changes to remote GPU environment**
2. **Run model tests**:
   ```bash
   python llama_keyvalue.py
   python internvl3_keyvalue.py
   ```
3. **Compare accuracy** against original ~84% performance baseline
4. **Verify both models** show improved extraction rates

## Expected Results:

- **Performance recovery**: Back to ~84% accuracy levels
- **Better document type handling**: Models start with universal identifiers
- **Improved semantic understanding**: Logical field flow aids comprehension
- **Maintained standardization**: All benefits of Priority 5 field naming preserved

The fix addresses the core issue: **models perform better with semantically ordered prompts** that match natural document processing workflows, rather than alphabetical field lists.