# Single-Pass Universal Extraction Implementation Plan

## Executive Summary

**Objective**: Eliminate double tiling issue by removing document type detection and implementing universal 15-field extraction in a single pass.

**Current Problem**: 
- InternVL3 loads/tiles images twice (detection + extraction)  
- Memory fragmentation triggers aggressive cleanup and retiling
- Performance penalty: 2x memory usage, 2x processing time

**Proposed Solution**: 
- Single model call extracting all 15 universal fields
- Post-processing document type inference from extraction results
- 50% speed improvement, 50% memory reduction

---

## Risk Assessment

### 🔴 HIGH RISKS ⚠️ UPDATED ASSESSMENT

#### 1. **Accuracy Regression Risk** - REDUCED RISK
- **Current**: Invoice/Receipt prompts are identical (no specialization loss)
- **Bank Statements**: Only need 5 fields vs 15 universal (low complexity increase)
- **Historical Evidence**: Universal prompts worked well in previous implementations
- **Impact**: Risk significantly lower than initially assessed

#### 2. **Prompt Engineering Complexity** ⚠️ AMENDED
- **Current**: Invoice and Receipt prompts are IDENTICAL, Bank Statement only needs 5 fields
- **Reality**: Universal prompts have worked well previously in this codebase
- **Key Success Factor**: Critical to reinforce "if you cannot see it then return NOT_FOUND"
- **Impact**: Lower risk than initially assessed - mainly need strong NOT_FOUND reinforcement

#### 3. **Post-Processing Logic Risk**
- **Current**: Document type known before field processing
- **Risk**: Type inference from extraction results may be unreliable
- **Impact**: Wrong type inference → wrong field cleaning → poor evaluation scores

#### 4. **Field Conflict Resolution**
- **Current**: Type-specific field lists prevent conflicts
- **Risk**: Universal field list may confuse model (e.g., INVOICE_DATE vs STATEMENT_DATE_RANGE)
- **Impact**: Model may extract wrong date types or mix field semantics

### 🟡 MEDIUM RISKS

#### 5. **YAML Schema Migration**
- **Current**: Well-tested type-specific prompts in unified_schema.yaml
- **Risk**: Need to create new universal prompt template
- **Impact**: Extensive testing required across all document types

#### 6. **ExtractionCleaner Compatibility** 
- **Current**: Cleaner works with known document types
- **Risk**: Type inference timing may affect cleaning logic
- **Impact**: Currency formatting, address normalization issues may persist

#### 7. **Evaluation Pipeline Changes**
- **Current**: Ground truth evaluation expects specific document types
- **Risk**: Evaluation metrics may need recalibration for universal approach
- **Impact**: Cannot directly compare new vs old accuracy metrics

### 🟢 LOW RISKS

#### 8. **Memory Management**
- **Current**: V100 optimizations already implemented
- **Risk**: Single-pass should improve memory usage
- **Impact**: Positive impact expected

#### 9. **Backward Compatibility**
- **Current**: Existing interfaces and output formats
- **Risk**: Output format changes minimal
- **Impact**: Minimal API changes required

---

## Implementation Strategy

### Phase 1: Preparation (Low Risk)

#### 1.1 Backup Current State
```bash
# Create backup branch
git checkout -b backup-before-single-pass
git push -u origin backup-before-single-pass

# Create implementation branch  
git checkout -b single-pass-extraction
```

#### 1.2 Research Current Field Usage
- Analyze ground truth data to confirm 15-field coverage
- Verify no missing fields across all test images
- Document current accuracy baseline per field

#### 1.3 Design Universal Prompt Template
- Create new YAML template section for universal extraction
- Test prompt variations with different field orderings
- Validate prompt token count (target: <800 tokens)

### Phase 2: Core Implementation (Medium Risk)

#### 2.1 Modify InternVL3 Processor Architecture

**File**: `models/document_aware_internvl3_processor.py`

**Changes Required**:
```python
class DocumentAwareInternVL3Processor:
    def __init__(self, field_list=None, debug=False):
        # Remove document-type-specific initialization
        self.field_list = UNIVERSAL_FIELD_LIST  # Always use 15 fields
        self.universal_mode = True
        # ... rest unchanged
    
    def process_single_image(self, image_path):
        """Single-pass universal extraction - NO document type detection"""
        # REMOVE: doc_type = self._detect_document_type_yaml(image_path)  
        # REMOVE: schema = self.schema_loader.get_document_schema(doc_type)
        
        # Direct extraction with universal field list
        extracted_data = self._extract_fields_directly(image_path, UNIVERSAL_FIELD_LIST)
        
        # NEW: Post-processing type inference
        inferred_type = self._infer_document_type_from_extraction(extracted_data)
        
        # Metadata update
        metadata = {
            "document_type": inferred_type,
            "extraction_strategy": "single_pass_universal", 
            "total_fields": 15,
            "extraction_method": "universal_field_extraction"
        }
        
        return extracted_data, metadata
```

#### 2.2 Implement Document Type Inference
```python
def _infer_document_type_from_extraction(self, extracted_data):
    """Infer document type from extraction results"""
    
    # Bank statement indicators
    if (extracted_data.get("STATEMENT_DATE_RANGE", "NOT_FOUND") != "NOT_FOUND" or
        extracted_data.get("TRANSACTION_DATES", "NOT_FOUND") != "NOT_FOUND"):
        return "bank_statement"
    
    # Invoice/Receipt indicators  
    elif (extracted_data.get("LINE_ITEM_DESCRIPTIONS", "NOT_FOUND") != "NOT_FOUND" or
          extracted_data.get("GST_AMOUNT", "NOT_FOUND") != "NOT_FOUND"):
        return "invoice"  # Could be receipt, but use invoice as default
    
    # Fallback
    else:
        return "unknown"
```

#### 2.3 Create Universal YAML Template

**File**: `config/unified_schema.yaml`

**New Section**:
```yaml
universal_extraction:
  internvl3:
    system_prompt: |
      You are a document analysis expert. Extract information from this business document.
      
      🚨 CRITICAL RULE: If you cannot clearly see a field in the document, return "NOT_FOUND"
      🚨 DO NOT GUESS or INFER missing information
      🚨 Only extract what is explicitly visible in the image
      
      FORMATTING REQUIREMENTS:
      1. Fields not visible/applicable: Return "NOT_FOUND"
      2. Monetary values: Include currency symbols ($8.62, not 8.62)
      3. Addresses: Extract complete multi-line addresses  
      4. Dates: Use DD/MM/YYYY format
      5. Lists: Use pipe separation (Item1 | Item2 | Item3)
      6. Boolean values: true/false (lowercase)
      
    field_instructions: |
      DOCUMENT_TYPE: NOT_FOUND (will be auto-determined)
      INVOICE_DATE: Date of invoice/receipt (DD/MM/YYYY)
      SUPPLIER_NAME: Company/business name providing goods/services  
      BUSINESS_ABN: 11-digit Australian Business Number
      BUSINESS_ADDRESS: Complete supplier business address
      PAYER_NAME: Customer/payer name
      PAYER_ADDRESS: Customer/payer address
      LINE_ITEM_DESCRIPTIONS: Item names/descriptions (pipe-separated)
      LINE_ITEM_TOTAL_PRICES: Item prices (pipe-separated with $)
      GST_AMOUNT: GST/tax amount (with $)
      IS_GST_INCLUDED: true/false - whether GST is included in total
      TOTAL_AMOUNT: Final total amount (with $)
      STATEMENT_DATE_RANGE: Date range for statements (DD/MM/YYYY - DD/MM/YYYY) 
      TRANSACTION_DATES: Transaction dates (pipe-separated, DD/MM/YYYY)
      TRANSACTION_AMOUNTS_PAID: Transaction amounts (pipe-separated with $)
```

### Phase 3: Testing and Validation (High Risk)

#### 3.1 Create Test Suite
```python
# test_single_pass_extraction.py
def test_universal_extraction():
    """Test universal extraction against current baseline"""
    
    test_cases = [
        ("invoice_sample.jpg", "invoice", expected_invoice_fields),
        ("receipt_sample.jpg", "receipt", expected_receipt_fields), 
        ("statement_sample.jpg", "bank_statement", expected_statement_fields)
    ]
    
    for image_path, expected_type, expected_fields in test_cases:
        # Test extraction
        result, metadata = processor.process_single_image(image_path)
        
        # Validate type inference
        assert metadata["document_type"] == expected_type
        
        # Validate field extraction accuracy  
        accuracy = calculate_field_accuracy(result, expected_fields)
        assert accuracy >= MINIMUM_ACCEPTABLE_ACCURACY  # e.g., 70%
```

#### 3.2 A/B Testing Protocol
1. **Baseline Test**: Run current two-pass extraction on test set
2. **Universal Test**: Run new single-pass extraction on same test set  
3. **Comparison**: Field-by-field accuracy comparison
4. **Performance Test**: Memory usage and speed benchmarking

#### 3.3 Acceptance Criteria
- **Accuracy**: Universal extraction ≥ 70% accuracy (current InternVL3 baseline)
- **Performance**: ≥ 40% speed improvement 
- **Memory**: ≥ 30% memory reduction
- **Reliability**: No crashes or OOM errors on test set

### Phase 4: Llama Integration (Medium Risk)

#### 4.1 Apply Same Changes to Llama Processor
- Modify `models/document_aware_llama_processor.py` with identical approach
- Test that Llama maintains ~100% accuracy with universal extraction
- Ensure both models use identical processing pipeline

#### 4.2 Cross-Model Validation
- Compare Llama vs InternVL3 results on identical universal pipeline
- Verify ExtractionCleaner works consistently for both models
- Validate that accuracy differences are model-specific, not pipeline-specific

---

## Rollback Strategy

### Immediate Rollback (if critical issues found)
```bash
git checkout main
git branch -D single-pass-extraction  # Delete problematic branch
# Continue with current two-pass approach
```

### Partial Rollback (if accuracy insufficient)
```bash
# Keep single-pass for InternVL3 (faster), revert Llama to two-pass (accuracy)
git checkout main
git cherry-pick <internvl3_changes>  # Keep only InternVL3 modifications
```

### Gradual Rollback (if performance issues)
```bash  
# Add --single-pass flag for optional behavior
python internvl3_document_aware.py --single-pass  # New behavior
python internvl3_document_aware.py               # Original behavior
```

---

## Success Metrics

### Performance Targets
- **Speed**: Processing time reduced by ≥40%
- **Memory**: Peak memory usage reduced by ≥30% 
- **Tiling**: Eliminate double tiling (1 load cycle vs 2)

### Accuracy Targets  
- **InternVL3**: Maintain or improve current 74.3% accuracy
- **Llama**: Maintain ≥95% accuracy (slight regression acceptable for performance gains)
- **Field-Specific**: No critical field accuracy drops >10%

### Quality Targets
- **Currency Formatting**: Maintain $ signs in GST_AMOUNT, TOTAL_AMOUNT
- **Address Extraction**: Maintain multi-line address extraction
- **Document Type Inference**: ≥95% correct type classification

---

## Timeline and Milestones

### Week 1: Research and Design
- [ ] Baseline accuracy measurement
- [ ] Universal prompt template design
- [ ] Implementation architecture finalization

### Week 2: Core Implementation  
- [ ] InternVL3 processor modifications
- [ ] Document type inference implementation
- [ ] Universal YAML template creation

### Week 3: Testing and Validation
- [ ] Unit test development
- [ ] A/B testing execution  
- [ ] Performance benchmarking

### Week 4: Integration and Rollout
- [ ] Llama processor modifications
- [ ] Cross-model validation
- [ ] Production deployment or rollback decision

---

## Dependencies and Prerequisites

### Technical Requirements
- [ ] Current codebase backed up to separate branch
- [ ] Test dataset with ground truth for all document types
- [ ] Performance benchmarking tools ready
- [ ] V100 GPU access for memory testing

### Validation Requirements  
- [ ] Accuracy evaluation pipeline functional
- [ ] Memory monitoring tools configured
- [ ] Rollback procedures tested

---

## Conclusion

This implementation plan provides a systematic approach to high-risk architectural change while maintaining safety through comprehensive testing, clear rollback strategies, and incremental validation.

**Updated Recommendation**: Risk assessment significantly reduced based on user insights:
- Invoice/Receipt prompts are already identical (no specialization loss)
- Universal prompts have historical success in this codebase  
- Bank statements only need 5/15 fields (low complexity)

**Proceed with confidence to Phase 2 (Core Implementation)** with emphasis on strong "NOT_FOUND" reinforcement in prompts.