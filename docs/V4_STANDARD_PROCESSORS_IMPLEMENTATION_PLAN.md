# Implementation Plan: Upgrade Standard Processors to v4 Schema (49 Fields)

## Executive Summary

**Objective**: Upgrade Standard Llama and InternVL3 Processors from legacy 25-field extraction to the new v4 schema supporting 49 comprehensive fields with document-type awareness.

**Current Problem**: 
- Standard processors hard-coded to extract only 25 fields from `EXTRACTION_FIELDS` 
- v4 schema defines 49 total fields (25 invoice + 19 receipt + 17 bank statement)
- Standard processors missing new boss-mandated fields (payment status, GST details, transaction lists)

**Success Criteria**:
- Standard processors extract all relevant fields for each document type
- Document detection filters fields appropriately (25/19/17 per type)
- Evaluation works with both v3 and v4 ground truth formats
- Memory usage remains manageable with increased field count

## Phase 1: Schema Integration

### Task 1.1: Update `common/config.py`
**File**: `/Users/tod/Desktop/LMM_POC/common/config.py`
**Priority**: High
**Estimated Time**: 2 hours

**Current State**:
```python
EXTRACTION_FIELDS = []  # Will be set on first access - currently 25 fields
FIELD_COUNT = None      # Currently set to 25
```

**Required Changes**:
1. **Replace hardcoded field loading** with v4 schema integration
2. **Add `get_v4_field_list()` function** that returns all 49 unique fields from v4 schema
3. **Update `FIELD_COUNT` constant** to reflect actual v4 schema field count (49)
4. **Add document-type field filtering functions**:
   - `get_invoice_fields()` → 25 fields
   - `get_receipt_fields()` → 19 fields  
   - `get_statement_fields()` → 17 fields

**Implementation Details**:
```python
def get_v4_field_list() -> List[str]:
    """Get all 49 unique fields from v4 schema."""
    from .schema_config import get_schema_config
    config = get_schema_config()
    return config.extraction_fields  # Should return 49 fields

def get_document_type_fields(document_type: str) -> List[str]:
    """Get fields specific to document type."""
    from .document_schema_loader import DocumentTypeFieldSchema
    loader = DocumentTypeFieldSchema("field_schema_v4.yaml")
    schema = loader.get_document_schema(document_type)
    return [field["name"] for field in schema["fields"]]
```

**Validation**:
- Verify `get_field_count()` returns 49
- Verify `get_v4_field_list()` contains v4-specific fields like `IS_GST_INCLUDED`, `TOTAL_AMOUNT_PAID`
- Test document-type specific field filtering

### Task 1.2: Add Document Type Detection Utility
**File**: `/Users/tod/Desktop/LMM_POC/common/document_type_detector.py`
**Priority**: Medium
**Estimated Time**: 3 hours

**Purpose**: Lightweight document type detection for standard processors to enable field filtering

**Implementation**:
```python
class LightweightDocumentDetector:
    """Simple document type detection for standard processors."""
    
    def detect_document_type(self, image_path: str, model_extract_func) -> str:
        """Detect document type using minimal model call."""
        prompt = """Look at this document and determine its type.
        
        OUTPUT FORMAT:
        DOCUMENT_TYPE: [invoice, receipt, or bank_statement]
        
        Instructions:
        - invoice: Has invoice number, due date, business details
        - receipt: Has transaction receipt, purchase details  
        - bank_statement: Has account statements, transaction history
        """
        
        response = model_extract_func(image_path, prompt, max_new_tokens=50)
        return self._parse_document_type(response)
```

## Phase 2: Standard Processor Updates

### Task 2.1: Modify Standard Llama Processor
**File**: `/Users/tod/Desktop/LMM_POC/models/llama_processor.py`
**Priority**: High
**Estimated Time**: 4 hours

**Current Architecture**:
```python
def get_extraction_prompt(self):
    # Uses schema.generate_dynamic_prompt() with fixed 25 fields
    schema = get_global_schema()
    prompt = schema.generate_dynamic_prompt(
        model_name="llama", 
        strategy="single_pass"
    )
```

**Required Changes**:
1. **Add document type detection**:
   ```python
   def __init__(self, ..., enable_v4_schema=True):
       self.enable_v4_schema = enable_v4_schema
       self.document_detector = LightweightDocumentDetector()
   ```

2. **Update prompt generation for variable fields**:
   ```python
   def get_extraction_prompt(self, image_path=None):
       if self.enable_v4_schema and image_path:
           # Detect document type
           classification_info = self.document_detector.detect_and_classify_document(
               image_path
           )
           doc_type = classification_info['document_type']
           # Get document-specific fields
           fields = get_document_type_fields(doc_type)
       else:
           # Legacy: use all 49 fields
           fields = get_v4_field_list()
       
       # Generate prompt with specific field list
       return self._generate_prompt_for_fields(fields)
   ```

3. **Add field-specific prompt generation**:
   ```python
   def _generate_prompt_for_fields(self, field_list: List[str]) -> str:
       """Generate extraction prompt for specific field list."""
       # Use schema loader but with custom field filtering
   ```

4. **Update processing methods**:
   ```python
   def process_single_image(self, image_path):
       # Pass image_path to get_extraction_prompt for document detection
       prompt = self.get_extraction_prompt(image_path)
       # ... rest of processing
   ```

### Task 2.2: Modify Standard InternVL3 Processor
**File**: `/Users/tod/Desktop/LMM_POC/models/internvl3_processor.py`
**Priority**: High
**Estimated Time**: 4 hours

**Required Changes**: Mirror the Llama processor changes:
1. Add `enable_v4_schema` parameter to constructor
2. Integrate `LightweightDocumentDetector`
3. Update `get_extraction_prompt()` with document-type awareness
4. Add field-specific prompt generation
5. Update `process_single_image()` to pass image path for detection

**InternVL3-Specific Considerations**:
- Ensure document detection works with InternVL3's direct prompting format
- Update token limits for larger field counts (49 vs 25 fields)
- Test with both 2B and 8B models

## Phase 3: Response Processing Updates

### Task 3.1: Update Extraction Parser
**File**: `/Users/tod/Desktop/LMM_POC/common/extraction_parser.py`
**Priority**: Medium
**Estimated Time**: 2 hours

**Current Issue**: Parser expects exactly 25 fields, needs to handle variable field counts

**Required Changes**:
1. **Make field validation flexible**:
   ```python
   def parse_extraction_response(response: str, expected_fields: List[str] = None):
       expected_fields = expected_fields or get_v4_field_list()
       # Parse with flexible field list
   ```

2. **Add v4 field type support**:
   - Boolean field parsing (`IS_GST_INCLUDED: true/false`)
   - Calculated field validation (`LINE_ITEM_TOTAL_PRICES`)
   - Transaction list parsing (`TRANSACTION_DATES`)

### Task 3.2: Update Evaluation Logic
**File**: `/Users/tod/Desktop/LMM_POC/common/evaluation_metrics.py`
**Priority**: Medium
**Estimated Time**: 3 hours

**Required Changes**:
1. **Add v4 field type evaluation**:
   ```python
   def evaluate_boolean_field(extracted: str, ground_truth: str) -> float:
       """Evaluate boolean fields like IS_GST_INCLUDED."""
   
   def evaluate_calculated_field(extracted: str, ground_truth: str) -> float:
       """Evaluate calculated fields with validation."""
   
   def evaluate_transaction_list(extracted: str, ground_truth: str) -> float:
       """Evaluate transaction list fields."""
   ```

2. **Update `calculate_field_accuracy()`** to handle new field types
3. **Add document-type aware accuracy calculation**

## Phase 4: Ground Truth Integration

### Task 4.1: Update Batch Processing
**File**: Multiple processor files
**Priority**: High
**Estimated Time**: 3 hours

**Required Changes**:
1. **Update CSV loading/saving** to handle 49-column structure
2. **Add v3/v4 ground truth detection**:
   ```python
   def detect_ground_truth_version(csv_path: str) -> str:
       """Detect if ground truth is v3 (34 cols) or v4 (49 cols)."""
   ```

3. **Add backward compatibility** for v3 ground truth files
4. **Update batch statistics** to handle variable field counts

### Task 4.2: Memory Management Updates
**File**: Both processor files  
**Priority**: Medium
**Estimated Time**: 2 hours

**Required Changes**:
1. **Recalculate `max_new_tokens`** for 49-field responses vs 25-field
2. **Update batch sizes** to handle larger token requirements
3. **Test memory usage** with expanded extraction

## Phase 5: Integration Testing

### Task 5.1: Validation Suite
**Priority**: High
**Estimated Time**: 4 hours

**Test Cases**:
1. **Field Count Validation**:
   - Verify `get_field_count()` returns 49
   - Verify document-type filtering works (25/19/17)
   - Test all v4-specific fields are included

2. **Document Detection**:
   - Test invoice detection → 25 fields extracted  
   - Test receipt detection → 19 fields extracted
   - Test statement detection → 17 fields extracted

3. **Backward Compatibility**:
   - Test with v3 ground truth (34 columns)
   - Test with v4 ground truth (49 columns)
   - Verify no regression in accuracy

4. **Memory & Performance**:
   - Test memory usage with 49 fields vs 25 fields
   - Verify batch processing still works
   - Test token limits are adequate

### Task 5.2: Integration Tests
**Priority**: High  
**Estimated Time**: 3 hours

**Test Scenarios**:
```python
# Test standard Llama with v4 schema
processor = LlamaProcessor(enable_v4_schema=True)
result = processor.process_single_image("test_invoice.png")
assert len(result["extracted_data"]) <= 25  # Invoice fields only

# Test standard InternVL3 with v4 schema  
processor = InternVL3Processor(enable_v4_schema=True)
result = processor.process_single_image("test_receipt.png")
assert len(result["extracted_data"]) <= 19  # Receipt fields only
```

## Implementation Timeline

| Phase | Tasks | Duration | Dependencies |
|-------|--------|----------|--------------|
| **Phase 1** | Schema Integration | 5 hours | v4 schema files |
| **Phase 2** | Processor Updates | 8 hours | Phase 1 complete |
| **Phase 3** | Response Processing | 5 hours | Phase 2 complete |
| **Phase 4** | Ground Truth Integration | 5 hours | Phase 3 complete |
| **Phase 5** | Testing & Validation | 7 hours | All phases complete |
| **Total** | **Full Implementation** | **30 hours** | **~4 working days** |

## Risk Mitigation

### High Risk Items
1. **Memory Usage**: 49 fields may require more VRAM
   - **Mitigation**: Implement document-type filtering to reduce active field count
   - **Fallback**: Add `enable_v4_schema=False` parameter for backward compatibility

2. **Accuracy Regression**: Changes may affect existing performance
   - **Mitigation**: Extensive testing with v3 ground truth
   - **Fallback**: Feature flag to disable v4 schema

3. **Token Limit Issues**: Larger responses may exceed model limits
   - **Mitigation**: Dynamic token calculation based on document type
   - **Fallback**: Chunk responses if needed

### Medium Risk Items
1. **Document Detection Accuracy**: Incorrect type detection affects field filtering
   - **Mitigation**: Conservative detection with fallback to full extraction
2. **Ground Truth Compatibility**: v3/v4 format conflicts  
   - **Mitigation**: Automatic format detection and conversion

## Files Modified

| File | Purpose | Change Scope |
|------|---------|-------------|
| `common/config.py` | Field definitions | Major - core field loading |
| `common/document_type_detector.py` | Document detection | New file |
| `models/llama_processor.py` | Standard Llama | Major - prompt generation |
| `models/internvl3_processor.py` | Standard InternVL3 | Major - prompt generation |
| `common/extraction_parser.py` | Response parsing | Medium - flexible validation |
| `common/evaluation_metrics.py` | Field evaluation | Medium - new field types |

## Success Metrics

1. **Functional Success**:
   - ✅ Standard processors extract 49 total unique fields
   - ✅ Document detection correctly filters fields by type  
   - ✅ All v4-specific fields (boolean, calculated, transaction_list) work
   - ✅ Backward compatibility with v3 ground truth maintained

2. **Performance Success**:
   - ✅ Memory usage increase < 20% compared to 25-field extraction
   - ✅ Processing time increase < 30% due to document detection overhead
   - ✅ Accuracy maintained or improved vs baseline

3. **Quality Success**:
   - ✅ All existing tests pass
   - ✅ New v4 field evaluation works correctly
   - ✅ Ground truth loading handles both v3 and v4 formats

## Rollback Plan

If implementation fails or causes regressions:

1. **Immediate Rollback**: Set `enable_v4_schema=False` by default
2. **Partial Rollback**: Disable document detection, use full 49-field extraction
3. **Full Rollback**: Revert to v3 schema with 25-field EXTRACTION_FIELDS
4. **Emergency**: Use git to revert all changes to last stable commit

## Next Steps After Implementation

1. **Performance Optimization**: Profile memory usage and optimize hot paths
2. **Advanced Features**: Add Chain-of-Thought prompting for complex fields
3. **Production Deployment**: Update production configurations to use v4 schema
4. **Documentation**: Update technical documentation with v4 schema details