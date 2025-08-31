# Risk-Averse Migration Plan: Llama Document Detection Simplification

## Executive Summary
Migrate Llama-3.2-Vision from complex DocumentTypeDetector (518 lines) to InternVL3's simple YAML-first approach (63 lines) while preserving accuracy and maintaining rollback capabilities.

## Current Status: ✅ Phase 1 Complete

### ✅ Phase 1: Parallel Implementation (COMPLETE)
**Duration:** Completed in 1 day
**Risk Level:** LOW ✅

**Implemented Changes:**
1. **Added YAML-first detection to Llama handler** ✅
   - Created `_detect_document_type_yaml_simple()` method in `DocumentAwareLlamaHandler`
   - Copied InternVL3's proven YAML approach exactly
   - Uses same YAML config: `prompts/document_type_detection.yaml`

2. **Implemented A/B Testing Framework** ✅
   - Added `--detection-method` CLI flag (`complex|simple|both`)
   - `both` mode runs both methods and compares results in real-time
   - Logs detection differences with agreement analysis

3. **Parallel Method Support** ✅
   - Complex method: Uses existing DocumentTypeDetector (default)
   - Simple method: Uses InternVL3-style YAML-first approach  
   - Both method: A/B tests both approaches and shows comparison

## Usage Examples

### Test Simple Method
```bash
python llama_document_aware.py --image-path evaluation_data/image_004.png --detection-method simple --debug
```

### A/B Test Both Methods
```bash
python llama_document_aware.py --image-path evaluation_data/image_004.png --detection-method both --debug
```

### Batch A/B Testing
```bash
python llama_document_aware.py --detection-method both --limit-images 10 --debug
```

## Next Phases

### Phase 2: Controlled Testing (READY)
**Duration:** 1-2 days  
**Risk Level:** LOW-MEDIUM

**Tasks:**
1. **Simple Testing**
   - Test simple method: `--detection-method simple`
   - Compare with complex method manually
   - Validate edge cases

2. **A/B Testing**
   - Run batch tests: `--detection-method both`
   - Monitor agreement rates in debug output
   - Check for systematic detection failures

3. **Quality Gate Criteria**
   - Simple method should work on receipt/invoice/bank_statement docs
   - Agreement with complex method should be high (>90%)
   - No systematic failures on any document type

### Phase 3: Gradual Migration (WAITING)
**Duration:** 1 day
**Risk Level:** MEDIUM

**Tasks:**
1. **Default to Simple Method**
   - Change CLI default from `complex` to `simple`
   - Keep complex method available via `--detection-method complex`
   - Monitor for detection issues

2. **Field Testing Period**
   - Use simple method for regular testing
   - Monitor accuracy and user feedback

### Phase 4: Code Cleanup (WAITING)
**Duration:** 1 day
**Risk Level:** LOW

**Tasks:**
1. **Remove Complex Detection** (only after validation)
   - Archive `DocumentTypeDetector` to `legacy/document_type_detector.py`
   - Remove complex detection imports and CLI options
   - Clean up documentation

## Implementation Details

### Files Modified
- **`llama_document_aware.py`** - Added parallel detection methods
  - `_detect_document_type_yaml_simple()` - Simple detection method
  - `_parse_document_type_response_yaml_simple()` - Simple response parser
  - `detect_and_classify_document()` - Updated with method selection
  - `_detect_with_complex_method()` - Complex method wrapper
  - `_detect_with_simple_method()` - Simple method wrapper
  - `_run_ab_detection_test()` - A/B testing implementation

### CLI Arguments Added
- `--detection-method {complex,simple,both}` - Choose detection approach
  - `complex`: Uses DocumentTypeDetector (default, current behavior)
  - `simple`: Uses YAML-first approach (InternVL3 style)
  - `both`: A/B tests both methods and shows comparison

### Debug Output
When using `--debug`, the system shows:
- Detection method being used
- Raw detection responses  
- Parsed document types
- For A/B testing: Agreement/disagreement between methods
- Confidence scores (complex method only)

## Rollback Strategy

**Immediate Rollback:** Use `--detection-method complex` (current default)
**Full Rollback:** Remove simple detection code (if needed)
**No Risk:** Complex method remains unchanged and fully functional

## Expected Benefits (After Full Migration)

1. **Code Simplification:** 91% reduction in detection code (518 → 63 lines)
2. **Consistency:** Same YAML-first approach across both models (Llama + InternVL3)
3. **Maintainability:** All detection prompts in YAML, no hardcoded logic
4. **Performance:** Potentially faster detection with simpler parsing
5. **A/B Testing:** Easy prompt experimentation and optimization

## Risk Assessment: LOW ✅

**Mitigation Factors:**
- ✅ Parallel Implementation: Both methods available during transition
- ✅ Easy Testing: CLI flags make testing straightforward
- ✅ Zero Risk Rollback: Default remains unchanged
- ✅ Real-time Comparison: A/B testing shows agreement immediately
- ✅ Gradual Approach: No sudden changes to behavior

## Quality Gates (For Phase 2 → Phase 3 Transition)

1. **Detection Accuracy:** Simple method should correctly identify document types
2. **Method Agreement:** >90% agreement between simple and complex methods  
3. **Edge Case Handling:** No systematic failures on any document type
4. **Performance:** No significant speed regression

## Next Actions

1. **Test the implementation:**
   ```bash
   # Test simple method
   python llama_document_aware.py --image-path evaluation_data/image_004.png --detection-method simple --debug
   
   # A/B test both methods
   python llama_document_aware.py --image-path evaluation_data/image_004.png --detection-method both --debug
   ```

2. **Monitor agreement rates** in A/B testing output

3. **Identify any disagreements** and analyze causes

4. **Proceed to Phase 3** once quality gates are met

---

**Migration Lead:** Claude Code Assistant  
**Status:** Phase 1 Complete ✅ | Ready for Phase 2 Testing  
**Last Updated:** $(date)  
**Risk Level:** LOW (with proper validation)