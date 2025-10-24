# Field Schema v4 Implementation Plan

## Overview
Comprehensive upgrade from field_schema_v3.yaml to field_schema_v4.yaml incorporating all boss-mandated fields with updates to extraction prompts, evaluation logic, and ground truth structure.

## Phase 1: Schema Foundation (field_schema_v4.yaml)
1. **Create field_schema_v4.yaml** with updated version and metadata
2. **Expand Invoice Schema** from 20 → 25 fields:
   - Add SUPPLIER_EMAIL (text field)
   - Add IS_GST_INCLUDED (boolean field - new field type)
   - Add TOTAL_DISCOUNT_AMOUNT (monetary field)
   - Add TOTAL_AMOUNT_PAID (monetary field)  
   - Add BALANCE_OF_PAYMENT (monetary field)
   - Add TOTAL_AMOUNT_PAYABLE (monetary field)
   - Add LINE_ITEM_TOTAL_PRICES (list field)
   - Add LINE_ITEM_GST_AMOUNTS (list field)
   - Add LINE_ITEM_DISCOUNT_AMOUNTS (list field)

3. **Expand Bank Statement Schema** from 12 → 17 fields:
   - Add CREDIT_CARD_DUE_DATE (date field)
   - Add TRANSACTION_DATES (list field)
   - Modify TRANSACTION_DESCRIPTIONS (enhanced list field)
   - Add TRANSACTION_AMOUNTS_PAID (list field)
   - Add TRANSACTION_AMOUNTS_RECEIVED (list field)
   - Add TRANSACTION_BALANCES (list field)
   - Update BANK_BSB_NUMBER with conditional logic for credit cards

4. **Receipt Schema** - No changes required (19 fields maintained)

## Phase 2: Field Type System Enhancement
1. **Add new field type**: `boolean` for IS_GST_INCLUDED
2. **Add new field type**: `calculated` for derived fields (LINE_ITEM_TOTAL_PRICES)
3. **Add new field type**: `transaction_list` for structured transaction data
4. **Update evaluation logic** mapping in document_type_metrics.py

## Phase 3: Extraction Prompt Updates
1. **Document-Aware Llama Processor**:
   - Update `generate_dynamic_prompt()` method to handle new field types
   - Add boolean field prompt templates
   - Add calculated field instructions
   - Update line-item extraction prompts for GST/discount/totals

2. **Document-Aware InternVL3 Processor**:
   - Mirror prompt updates from Llama processor
   - Optimize for InternVL3's concise response style

3. **Prompt Configuration** (if YAML-based prompts exist):
   - Update field_instructions section with new fields
   - Add specialized instructions for calculated fields
   - Update transaction-level extraction prompts

## Phase 4: Evaluation Logic Enhancement
1. **Create new evaluation methods** in evaluation_metrics.py:
   - `evaluate_boolean_field()` for IS_GST_INCLUDED
   - `evaluate_calculated_field()` for computed line items
   - `evaluate_transaction_list()` for individual transactions

2. **Update calculate_field_accuracy()**:
   - Add boolean field evaluation logic
   - Add calculated field validation (quantity × price = total)
   - Add transaction-level matching algorithms

3. **Document Type Metrics**:
   - Update DocumentTypeMetrics class with new required fields
   - Add validation rules for payment status fields
   - Update ATO compliance rules for additional invoice fields

## Phase 5: Ground Truth Data Structure
1. **CSV Schema Migration**:
   - Current: 34 columns
   - Target: 49 columns (+15 new fields)
   - Create migration script: `migrate_ground_truth_v3_to_v4.py`

2. **New Column Order**:
   ```
   # Existing columns maintained + new insertions:
   SUPPLIER_EMAIL (after BUSINESS_PHONE)
   IS_GST_INCLUDED (after GST_AMOUNT)
   TOTAL_DISCOUNT_AMOUNT (after SUBTOTAL_AMOUNT)
   TOTAL_AMOUNT_PAID (after TOTAL_AMOUNT)
   BALANCE_OF_PAYMENT (after TOTAL_AMOUNT_PAID)
   TOTAL_AMOUNT_PAYABLE (after BALANCE_OF_PAYMENT)
   LINE_ITEM_TOTAL_PRICES (after LINE_ITEM_PRICES)
   LINE_ITEM_GST_AMOUNTS (after LINE_ITEM_TOTAL_PRICES)
   LINE_ITEM_DISCOUNT_AMOUNTS (after LINE_ITEM_GST_AMOUNTS)
   CREDIT_CARD_DUE_DATE (in bank statement section)
   TRANSACTION_DATES (after existing transaction fields)
   TRANSACTION_AMOUNTS_PAID (after TRANSACTION_DATES)
   TRANSACTION_AMOUNTS_RECEIVED (after TRANSACTION_AMOUNTS_PAID)
   TRANSACTION_BALANCES (after TRANSACTION_AMOUNTS_RECEIVED)
   ```

3. **Data Migration Tasks**:
   - Auto-populate NOT_FOUND for new fields in existing 43 rows
   - Validate existing field mappings remain correct
   - Create backup: ground_truth_v3_backup.csv

## Phase 6: System Integration Updates
1. **Config Management**:
   - Update schema_config.py to reference field_schema_v4.yaml
   - Update field count constants and mappings
   - Add new field type definitions to config.py

2. **Document Schema Loader**:
   - Update DocumentTypeFieldSchema class to handle v4 schema
   - Add validation for new field types
   - Update get_document_schema() method for new field counts

3. **Processing Pipeline**:
   - Update document-aware handlers to use v4 schema
   - Test backward compatibility with v3 processors (if needed)
   - Update field reduction calculations (25 vs unified 25 = 0% reduction)

## Phase 7: Testing & Validation
1. **Unit Tests**:
   - Test v4 schema loading and compilation
   - Test new field type evaluation methods
   - Test prompt generation with new fields

2. **Integration Tests**:
   - Test end-to-end extraction with v4 schema
   - Validate ground truth CSV loading with new columns
   - Test document-aware pipeline with expanded fields

3. **Accuracy Validation**:
   - Re-annotate sample documents with new fields
   - Run accuracy tests to ensure no regression
   - Validate calculated fields produce correct results

## Phase 8: Documentation & Deployment
1. **Update Technical Documentation**:
   - Schema v4 field reference guide
   - New evaluation logic documentation
   - Ground truth annotation guidelines for new fields

2. **User Documentation**:
   - Field mapping guide (boss's names → our names)
   - Extraction capabilities update
   - Performance impact documentation

3. **Deployment**:
   - Update environment.yml if new dependencies needed
   - Deploy v4 schema as default
   - Create rollback procedure to v3 if issues

## Implementation Timeline
- **Phase 1-2** (Schema & Types): 1 day
- **Phase 3** (Prompts): 1 day  
- **Phase 4** (Evaluation): 2 days
- **Phase 5** (Ground Truth): 1 day
- **Phase 6** (Integration): 1 day
- **Phase 7** (Testing): 2 days
- **Phase 8** (Documentation): 1 day

**Total Estimated Time**: 9 days

## Files to Create/Modify
### New Files:
- `common/field_schema_v4.yaml`
- `scripts/migrate_ground_truth_v3_to_v4.py`
- `v4_implementation_plan.md` (this document)

### Modified Files:
- `common/schema_config.py`
- `common/document_schema_loader.py`
- `common/evaluation_metrics.py`
- `common/document_type_metrics.py`
- `models/document_aware_llama_processor.py`
- `models/document_aware_internvl3_processor.py`
- `evaluation_data/ground_truth.csv`

## Success Criteria
1. ✅ All 15 new fields successfully extracted from test documents
2. ✅ No regression in existing field accuracy (maintain >90%)
3. ✅ New evaluation logic correctly validates calculated fields
4. ✅ Ground truth CSV loads without errors in expanded format
5. ✅ Document-aware pipeline processes all document types with v4 schema
6. ✅ Backward compatibility maintained for critical system components

## Risk Mitigation
- **Schema Validation**: Comprehensive testing before deployment
- **Data Backup**: Full backup of v3 ground truth before migration
- **Rollback Plan**: Quick reversion to v3 schema if critical issues
- **Phased Testing**: Validate each phase before proceeding to next