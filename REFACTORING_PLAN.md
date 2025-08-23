# LMM_POC Refactoring Plan

## Overview
This document outlines key refactoring opportunities identified during the evaluation system debugging. These improvements will make the codebase more robust, maintainable, and less prone to subtle data handling issues.

## Priority 1: Missing Value Convention Change

### Issue
- Current system uses "N/A" for missing fields
- Pandas automatically converts "N/A" to NaN during CSV loading
- Causes evaluation mismatches and debugging complexity

### Solution
**Change from "N/A" to "NOT_FOUND"**

### Files to Update
1. **Ground Truth Data**
   - `evaluation_data/evaluation_ground_truth.csv` - Replace all "N/A" with "NOT_FOUND"

2. **Model Prompts**
   - `llama_prompts.yaml` - Update all prompt instructions
   - Any InternVL3 prompt configurations
   - Change instructions from "Use N/A if field is not visible" to "Use NOT_FOUND if field is not visible"

3. **Evaluation Logic**
   - `common/evaluation_metrics.py` - Update `is_missing_value()` function
   - Remove pandas NaN handling complexity
   - Simplify to just check for "NOT_FOUND"

4. **Documentation**
   - Update any examples or documentation showing expected outputs

### Benefits
- ✅ No pandas auto-conversion interference
- ✅ Clearer model instructions  
- ✅ Easier debugging and maintenance
- ✅ More semantic meaning ("NOT_FOUND" vs generic "N/A")

---

## Priority 2: Prompt Engineering Improvements

### Issue
- Current prompts have complex "CRITICAL" warnings that may confuse models
- Inconsistent separator formats (comma vs pipe)
- Long, prescriptive instructions

### Solution
**Simplify and standardize prompts**

### Improvements
1. **Reduce "CRITICAL" emphasis** - InternVL3 gets confused by too many warnings
2. **Standardize list separators** - Choose either comma or pipe consistently
3. **Streamline instructions** - More concise, action-oriented language
4. **Field-specific guidance** - Clearer examples for complex fields

---

## Priority 3: Configuration Management

### Issue
- Model paths are hardcoded in multiple places
- Strategy names are unclear ("6_groups" vs semantic names)
- Configuration scattered across files

### Solution
**Centralized, flexible configuration**

### Improvements
1. **Environment Variables**
   - `LLAMA_MODEL_PATH` - for Llama model location
   - `INTERNVL3_MODEL_PATH` - for InternVL3 model location
   - `GROUND_TRUTH_PATH` - for evaluation data location

2. **Strategy Naming**
   - "6_groups" → "grouped_extraction" or "field_grouped"
   - More descriptive names that indicate extraction approach

3. **Config Consolidation**
   - Single configuration file for all deployment environments
   - Clear separation of dev/test/production settings

---

## Priority 4: Field Naming Consistency

### Issue
- Mixed naming conventions in extraction fields
- Some fields unclear or don't match business document standards

### Solution
**Standardize field names and ensure business alignment**

### Review Areas
1. **Naming Conventions**
   - Consistent use of underscores vs camelCase
   - Business-standard terminology

2. **Field Clarity**
   - Ensure field names are self-documenting
   - Match common invoice/statement terminology

3. **Logical Grouping**
   - Related fields use consistent prefixes (PAYER_, BUSINESS_, etc.)

---

## Priority 5: Error Handling & Validation

### Issue
- Limited validation of ground truth CSV format
- Generic error messages for model loading
- No preprocessing validation

### Solution
**Comprehensive error handling and validation**

### Improvements
1. **Ground Truth Validation**
   - Check CSV column names match expected fields
   - Validate data types and formats
   - Clear error messages for malformed data

2. **Model Loading**
   - Better error messages for missing model files
   - Memory requirement checks before loading
   - Fallback strategies for different hardware

3. **Input Validation**
   - Image file format validation
   - Path existence checks
   - Clear user guidance for fixes

---

## Priority 6: Testing & Quality Assurance

### Issue
- No automated tests for evaluation logic
- Manual verification of accuracy changes
- Risk of regression during refactoring

### Solution
**Automated testing suite**

### Test Areas
1. **Evaluation Logic Tests**
   - Unit tests for field comparison functions
   - Known input/output pairs for accuracy verification
   - Edge case handling (empty fields, malformed data)

2. **Integration Tests**
   - Full pipeline tests with sample data
   - Cross-model consistency verification
   - Performance regression detection

3. **Data Validation Tests**
   - Ground truth CSV format validation
   - Model output parsing verification

---

## Implementation Strategy

### Phase 1: Core Data Issues (High Impact, Low Risk)
- [ ] Missing value convention change (N/A → NOT_FOUND)
- [ ] Ground truth CSV update
- [ ] Basic evaluation logic cleanup

### Phase 2: Configuration & Prompts (Medium Impact, Medium Risk)
- [ ] Prompt simplification and standardization
- [ ] Configuration centralization
- [ ] Environment variable support

### Phase 3: Structural Improvements (High Impact, Higher Risk)
- [ ] Field naming standardization
- [ ] Error handling improvements
- [ ] Testing suite implementation

### Success Criteria
- Maintain 87.6% accuracy after each change
- Improved code maintainability and readability
- Reduced debugging complexity
- Better error messages and user experience

---

## Notes
- Each refactoring should be tested individually
- Keep git commits focused on single improvements
- Verify accuracy benchmarks after each change
- Document any breaking changes clearly