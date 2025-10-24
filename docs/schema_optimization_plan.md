# Schema Optimization Plan for Document-Aware Extraction

## Executive Summary

Analysis of the document-aware extraction system reveals significant schema-evaluation alignment issues causing reported accuracy of 58.7% when actual model performance is 83.7%. The current field_schema_v2.yaml has structural inconsistencies that prevent proper evaluation and integration with the document-aware processing pipeline.

## Key Issues Identified

### 1. Schema-Evaluation Format Mismatch
- **Issue**: Schema instructs models to use "comma-separated" lists but evaluation system expects "pipe-separated" format
- **Impact**: Systematic evaluation failures for all list fields (LINE_ITEM_DESCRIPTIONS, LINE_ITEM_QUANTITIES, LINE_ITEM_PRICES)
- **Evidence**: Manual testing shows 83.7% actual accuracy vs 58.7% reported

### 2. Field Type Classification Problems
- **Issue**: Phone numbers classified as text instead of numeric_id fields
- **Impact**: Wrong evaluation logic applied (fuzzy text matching vs exact numeric matching)
- **Evidence**: BUSINESS_PHONE "(08) 4482 2347" vs "(38) 4482 2347" scores 0.0 instead of partial match

### 3. Schema Structure Inconsistency
- **Issue**: field_schema_v2.yaml uses new document-specific structure but some components expect unified format
- **Impact**: Schema loading conflicts, field type detection failures
- **Evidence**: DocumentTypeFieldSchema compatibility issues with base FieldSchema class

### 4. Ground Truth Synchronization Issues
- **Issue**: Schema field definitions don't align with ground truth CSV expectations
- **Impact**: Evaluation uses wrong comparison logic for specific field types
- **Evidence**: Multiple OCR precision issues treated as evaluation failures

### 5. Document-Aware Integration Gaps
- **Issue**: Evaluation system not optimized for document-specific schemas
- **Impact**: Performance metrics don't account for reduced field sets per document type
- **Evidence**: Receipt schema (19 fields) evaluated against full unified expectations

## Optimization Plan

### Phase 1: Fix Schema-Evaluation Alignment (High Priority)

**Objective**: Immediate fix for accuracy reporting discrepancy

1. **Update field instructions in field_schema_v2.yaml**:
   - Change all "comma-separated list" instructions to "pipe-separated list"
   - Update LINE_ITEM field instructions: `[pipe-separated items or NOT_FOUND]`
   - Ensure output format consistency across all list-type fields

2. **Fix field type classifications**:
   - Reclassify phone number fields as "numeric_id" type for proper validation
   - Update BUSINESS_PHONE and PAYER_PHONE field types
   - Verify MONETARY_FIELDS, LIST_FIELDS, NUMERIC_ID_FIELDS population

3. **Align evaluation logic with schema**:
   - Verify evaluation_metrics.py uses correct field type mappings
   - Fix phone number evaluation to allow partial matches for OCR errors
   - Update tolerance settings for document-specific field types

**Success Metrics**: 
- Reported accuracy matches manual calculation (83.7%+)
- List field evaluation passes consistently
- Phone number fields get appropriate partial credit

### Phase 2: Schema Structure Modernization (Medium Priority)

**Objective**: Seamless integration between v1/v2 schemas and document-aware processing

1. **Modernize DocumentTypeFieldSchema integration**:
   - Fix base class compatibility with new schema structure
   - Ensure field type detection works with document-specific schemas
   - Update schema loading to handle both unified and document-specific modes

2. **Optimize prompt generation for document types**:
   - Add document-specific field emphasis in prompts
   - Include validation hints for problematic fields (ABN vs BSB distinctions)
   - Implement field-specific extraction guidance for critical fields

3. **Enhance schema validation**:
   - Add schema structure validation for v2 format
   - Implement field inheritance validation (common + specific fields)
   - Create schema migration tools for v1 to v2 compatibility

**Success Metrics**:
- No schema loading errors or compatibility warnings
- Document-specific prompts improve extraction accuracy by 5-10%
- Seamless fallback to unified schema for unknown document types

### Phase 3: Evaluation System Enhancement (Low Priority)

**Objective**: Document-aware evaluation metrics and validation

1. **Create document-aware evaluation metrics**:
   - Different accuracy thresholds per document type (invoice: 90%, receipt: 85%, bank: 80%)
   - Field importance weighting (critical vs optional fields)
   - Document-specific validation rules (financial consistency, date sequences)

2. **Add comprehensive field validation**:
   - Schema-driven field format validation using field_schemas section
   - Cross-field consistency checks (subtotal + GST = total)
   - Ground truth alignment verification with detailed error reporting

3. **Implement evaluation analytics**:
   - Field-level accuracy tracking across document types
   - Performance comparison between document-aware and unified approaches
   - OCR precision vs extraction logic error categorization

**Success Metrics**:
- Document type-specific accuracy targets met consistently
- Field validation catches format errors before evaluation
- Clear distinction between OCR and extraction logic errors

### Phase 4: Ground Truth Synchronization

**Objective**: Ensure ground truth data matches schema expectations

1. **Verify ground truth data consistency**:
   - Audit all CSV fields against schema field definitions
   - Fix format inconsistencies (phone numbers, addresses, monetary values)
   - Validate field mappings between schema names and CSV columns

2. **Implement ground truth validation**:
   - Schema-driven ground truth format validation
   - Automated detection of field type mismatches
   - Ground truth quality metrics and reporting

3. **Create ground truth maintenance tools**:
   - Automated ground truth format correction
   - Schema-ground truth alignment verification
   - Field-level data quality monitoring

**Success Metrics**:
- Zero format mismatches between schema and ground truth
- Automated ground truth validation passes for all document types
- Ground truth maintenance tools reduce manual correction time by 80%

## Implementation Timeline

### Week 1: Critical Fixes (Phase 1)
- [ ] Fix schema instruction format (comma → pipe separators)
- [ ] Update field type classifications  
- [ ] Align evaluation logic with schema types
- [ ] Test accuracy reporting fix

### Week 2: Integration Improvements (Phase 2)  
- [ ] Fix DocumentTypeFieldSchema compatibility
- [ ] Enhance document-specific prompt generation
- [ ] Add schema validation improvements
- [ ] Test document-aware processing pipeline

### Week 3: Evaluation Enhancement (Phase 3)
- [ ] Implement document-aware evaluation metrics
- [ ] Add comprehensive field validation
- [ ] Create evaluation analytics dashboard
- [ ] Test performance monitoring

### Week 4: Data Synchronization (Phase 4)
- [ ] Audit and fix ground truth inconsistencies
- [ ] Implement ground truth validation tools
- [ ] Create maintenance automation
- [ ] Final system integration testing

## Expected Outcomes

### Immediate (Phase 1 Complete):
- **Accuracy reporting**: 58.7% → 83.7%+ (matching actual performance)
- **List field evaluation**: 100% pass rate for correctly formatted lists
- **Phone field evaluation**: Partial credit for OCR precision errors

### Medium-term (Phase 2 Complete):
- **Schema reliability**: Zero loading errors or compatibility issues
- **Document-specific optimization**: 20-30% fewer fields processed per document
- **Processing efficiency**: 15-25% faster extraction times

### Long-term (All Phases Complete):
- **Evaluation reliability**: Consistent metrics across all document types
- **Schema maintainability**: Single source of truth for field definitions  
- **System robustness**: Automated validation and error detection
- **Performance optimization**: Document-aware processing meets 90%+ accuracy targets

## Risk Mitigation

### Backward Compatibility
- Maintain v1 schema support throughout migration
- Implement gradual rollout with fallback mechanisms
- Comprehensive testing with existing evaluation datasets

### Data Integrity
- Backup ground truth data before modifications
- Implement validation checkpoints at each phase
- Rollback procedures for each optimization step

### Performance Impact
- Monitor processing times throughout optimization
- A/B testing for schema changes
- Performance regression prevention measures

## Success Criteria

1. **Accuracy Alignment**: Reported accuracy matches manual calculation within 2%
2. **Format Consistency**: 100% of list fields use pipe separators consistently
3. **Type Classification**: All field types correctly classified and evaluated
4. **Document-Aware Performance**: Each document type meets specific accuracy targets
5. **System Reliability**: Zero schema loading errors or evaluation failures