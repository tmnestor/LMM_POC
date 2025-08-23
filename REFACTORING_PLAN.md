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

## Priority 3: YAML-First Prompt Configuration

### Issue
- Prompts are scattered between YAML files and hardcoded Python
- Single-pass mode uses YAML, but grouped and InternVL3 use hardcoded prompts
- Inconsistent configuration approach makes prompt optimization difficult

### Solution
**YAML as single source of truth for ALL prompt configuration**

### Current State
- ✅ **Llama Single-pass**: YAML (`llama_single_pass_prompts.yaml`) - ✅ **87.2% accuracy**
- ❌ **Llama Grouped**: Hardcoded in `common/config.py`
- ❌ **InternVL3**: Hardcoded in `common/config.py`
- ❌ **Fallback**: Hardcoded in `common/config.py`

### Migration Plan

#### **Phase 1: Create YAML Files**
```
llama_single_pass_prompts.yaml     ✅ Done (87.2% accuracy verified)
llama_grouped_prompts.yaml         📝 TODO
internvl3_prompts.yaml             📝 TODO 
```

#### **Phase 2: Update Processors**
```python
# All processors will use YAML loading:
llama_processor.py                 🔄 Partially done (single-pass only)
internvl3_processor.py            📝 TODO
grouped_extraction.py             📝 TODO
```

#### **Phase 3: Remove Hardcoded Prompts**
```python
# Remove from common/config.py:
FIELD_INSTRUCTIONS                📝 TODO (after all migrated)
FIELD_DEFINITIONS["instruction"]  📝 TODO (keep other metadata)
```

#### **Phase 4: Clean Architecture**
```python
# Final state - config.py only contains:
FIELD_DEFINITIONS = {
    "ABN": {
        "type": "numeric_id",
        # "instruction": REMOVED - now in YAML
        "evaluation_logic": "exact_numeric_match", 
        "description": "Australian Business Number",
        "required": True,
    }
}
```

### Benefits of Complete YAML Migration
- 🎯 **Single Source of Truth**: All prompts in version-controlled YAML
- ⚡ **Easy A/B Testing**: Swap YAML files to test different prompts
- 🔧 **Non-programmer Editing**: Domain experts can modify prompts
- 📊 **Prompt Analytics**: Track which prompts perform best
- 🚀 **Deployment Flexibility**: Update prompts without code changes

---

## Priority 4: Environment & Deployment Configuration

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

## Priority 5: Field Naming Consistency

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

## Priority 6: Error Handling & Validation

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

## Priority 7: Testing & Quality Assurance

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
- [x] Missing value convention change (N/A → NOT_FOUND) - ✅ **COMPLETED**
- [x] Ground truth CSV update - ✅ **COMPLETED**
- [x] Basic evaluation logic cleanup - ✅ **COMPLETED**

### Phase 2: YAML-First Configuration (High Impact, Medium Risk)
- [x] Llama single-pass YAML prompts - ✅ **COMPLETED (87.2% accuracy)**
- [ ] Llama grouped extraction YAML prompts
- [ ] InternVL3 YAML prompts
- [ ] Remove hardcoded prompts from config.py

### Phase 3: Configuration & Environment Management (Medium Impact, Medium Risk)
- [ ] Prompt simplification and standardization
- [ ] Configuration centralization
- [ ] Environment variable support

### Phase 4: Structural Improvements (High Impact, Higher Risk)
- [ ] Field naming standardization
- [ ] Error handling improvements
- [ ] Testing suite implementation

### Success Criteria
- Maintain 87.6% accuracy after each change
- Improved code maintainability and readability
- Reduced debugging complexity
- Better error messages and user experience

---

## Recent Achievements

### ✅ **YAML Single-Pass Implementation (Completed)**
- **Date**: August 2025
- **Achievement**: Successfully migrated Llama single-pass prompts from hardcoded Python to YAML configuration
- **File**: `llama_single_pass_prompts.yaml`
- **Performance**: ✅ **87.2% accuracy maintained** (identical to hardcoded version)
- **Benefits**: 
  - Clean separation of configuration from code
  - Easy prompt modification without Python changes
  - Version control for prompt evolution
  - Deterministic results with temperature=0.0

### 🎯 **Next Milestone**: Complete YAML Migration
- Target: All prompts (Llama grouped, InternVL3) in YAML format
- Goal: Zero hardcoded prompts in Python codebase
- Expected Outcome: Full configurability and easier prompt optimization

---

## Notes
- Each refactoring should be tested individually
- Keep git commits focused on single improvements
- Verify accuracy benchmarks after each change
- Document any breaking changes clearly
- **Temperature=0.0** provides deterministic results for consistent testing