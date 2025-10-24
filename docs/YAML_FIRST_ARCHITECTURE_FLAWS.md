# YAML-First Architecture Design Flaws & Remediation Plan

**Document Status**: Architectural Review  
**Date**: 2025-01-14  
**Issue**: Prompt field reordering required cascade changes across 4+ files  
**Priority**: High - Violates core architectural principles  

## Executive Summary

A simple field reordering change in `prompts/llama_single_pass_high_performance.yaml` cascaded through multiple Python files and configuration files, revealing fundamental design flaws in our YAML-first architecture implementation. This violates the core principle that YAML should be the single source of truth with zero Python overrides.

## Current Architecture Problems

### 1. Multiple Sources of Truth

**Flaw**: Field ordering defined in multiple places:
- `prompts/llama_single_pass_high_performance.yaml` (prompt field order)
- `config/fields.yaml` (schema field order)  
- `config/document_metrics.yaml` (evaluation field lists)

**Impact**: Changes require manual synchronization across files
**Example**: Moving GST_AMOUNT required editing both prompt YAML and schema YAML

### 2. Python Dynamic Overrides

**Flaw**: Python code dynamically modifies YAML configurations:
```python
# BAD: Dynamic field count injection
output_format = output_format.replace("49 FIELDS", f"{len(field_list)} FIELDS")

# BAD: Dynamic stop instruction generation  
stop_instruction = f"STOP after {last_field} line. Do not add explanations or comments."
```

**Impact**: YAML files become unreliable - Python can override anything
**Violation**: Breaks YAML-first principle completely

### 3. Tight Coupling Between Systems

**Flaw**: Document schema structure directly coupled to prompt generation:
- Prompt field order must match schema field order
- Document-aware fields filtering affects prompt rendering
- Schema changes break prompt generation

**Impact**: Interdependent systems - changes cascade unpredictably

### 4. Fragmented Configuration Management

**Flaw**: Related configurations scattered across multiple files:
- Field definitions: `config/fields.yaml`
- Document metrics: `config/document_metrics.yaml` 
- Prompt structure: `prompts/*.yaml`
- Field validation: Python code

**Impact**: No single view of system configuration

### 5. Inconsistent Architecture Implementation

**Flaw**: Mixed implementation approaches:
- Some processors use pure YAML (`llama_processor.py` - partial)
- Others have Python overrides (`document_aware_llama_processor.py`)
- Schema loader has hardcoded logic

**Impact**: Unpredictable behavior - some changes work, others cascade

## Cascade Failure Analysis

### Recent Example: GST_AMOUNT Field Reordering

**Initial Change**: Move GST_AMOUNT in prompt semantic order
**Required Cascading Changes**:

1. **prompts/llama_single_pass_high_performance.yaml**
   - Reorder fields to: `LINE_ITEM_DESCRIPTIONS → LINE_ITEM_TOTAL_PRICES → IS_GST_INCLUDED → GST_AMOUNT → TOTAL_AMOUNT`

2. **models/document_aware_llama_processor.py** 
   - Remove dynamic stop instruction generation
   - Remove dynamic field count injection
   - Remove dynamic format rule modifications

3. **models/llama_processor.py**
   - Remove dynamic output format modification 
   - Remove field count replacements

4. **config/fields.yaml**
   - Reorder invoice field list to match prompt order
   - Update semantic sequence comments

**Root Cause**: No single source of truth for field ordering

## Architectural Design Principles Violated

### 1. Single Responsibility Principle
- YAML files should define structure
- Python should only render YAML (no modification)

### 2. Don't Repeat Yourself (DRY)
- Field order defined in multiple places
- Validation rules duplicated across files

### 3. Open/Closed Principle  
- System should be open for extension (new fields)
- Closed for modification (no Python overrides)

### 4. Separation of Concerns
- Configuration mixed with business logic
- Prompt structure coupled to data schema

## Proposed Architecture Improvements

### Phase 1: Consolidate Configuration

**Objective**: Single source of truth for all field definitions

**Implementation**:
```yaml
# config/unified_field_schema.yaml
field_definitions:
  semantic_order:
    - DOCUMENT_TYPE
    - BUSINESS_ABN  
    - SUPPLIER_NAME
    # ... complete semantic order
    
  document_types:
    invoice:
      required_fields: [DOCUMENT_TYPE, BUSINESS_ABN, ...]
      field_order: semantic  # Use semantic_order above
      
  prompts:
    llama_single_pass:
      output_format: "REQUIRED OUTPUT FORMAT - BOSS REDUCED SCHEMA:"
      stop_instruction: "STOP AFTER the TOTAL_AMOUNT. Do not add explanations or comments."
```

### Phase 2: Pure YAML Rendering

**Objective**: Eliminate all Python dynamic modifications

**Implementation**:
```python
class PureYAMLProcessor:
    def generate_prompt(self) -> str:
        """Pure YAML rendering - no Python modifications allowed."""
        config = self.load_yaml_config()  # Read only
        return self.render_template(config)  # Template only
        # NO .replace(), NO f-strings, NO dynamic generation
```

### Phase 3: Configuration Validation

**Objective**: Ensure YAML consistency at startup

**Implementation**:
```python
def validate_yaml_consistency():
    """Fail-fast validation of YAML configuration consistency."""
    # Verify field orders match across all YAML files
    # Validate required fields are defined
    # Check prompt templates reference valid fields
    # Ensure no Python overrides exist
```

### Phase 4: Schema-Driven Prompt Generation

**Objective**: Prompt structure derived from unified schema

**Implementation**:
```yaml
# Single schema drives everything
unified_schema:
  fields:
    DOCUMENT_TYPE:
      order: 1
      instruction: "[document type (INVOICE/RECEIPT/STATEMENT) or NOT_FOUND]"
      applicable_documents: [invoice, receipt, bank_statement]
      
  document_types:
    invoice:
      includes: [1, 2, 3, 15, 16, 17, 18, 19]  # Reference field order numbers
      
  prompt_templates:
    llama_single_pass:
      format: "{instruction_header}\n{field_list}\n{format_rules}\n{stop_instruction}"
```

## Implementation Roadmap

### Immediate (Next Sprint)
- [ ] Document current cascade failures
- [ ] Create unified schema design specification
- [ ] Plan migration strategy

### Short Term (Next Month)  
- [ ] Implement unified field schema YAML
- [ ] Remove all Python dynamic modifications
- [ ] Add YAML consistency validation

### Medium Term (Next Quarter)
- [ ] Migrate all processors to pure YAML rendering
- [ ] Implement schema-driven prompt generation
- [ ] Create automated YAML consistency tests

### Long Term (Future Releases)
- [ ] Visual configuration management UI
- [ ] Real-time YAML validation
- [ ] Template-based prompt generation system

## Success Criteria

### Definition of Done: True YAML-First Architecture

1. **Single File Changes**: Prompt modifications should only require editing ONE YAML file
2. **Zero Python Overrides**: No Python code modifies YAML-defined structures
3. **Fail-Fast Validation**: System validates YAML consistency at startup
4. **Predictable Behavior**: Configuration changes have predictable, isolated effects

### Validation Tests

```bash
# Test 1: Field reordering should only touch one file
git log --oneline | grep "field reorder" | wc -l  # Should be 1

# Test 2: No Python string replacement in processors  
grep -r "\.replace(" models/ | wc -l  # Should be 0

# Test 3: YAML validation passes
python -c "from config import validate_yaml_consistency; validate_yaml_consistency()" # Should pass
```

## Risk Assessment

### High Risk
- **Breaking Changes**: Migration will require significant refactoring
- **Testing Complexity**: Need comprehensive validation of all prompt variations

### Medium Risk  
- **Performance Impact**: Pure YAML rendering might be slower than dynamic generation
- **Developer Learning**: Team needs to learn new configuration patterns

### Low Risk
- **Backward Compatibility**: Can implement incrementally
- **Tool Support**: YAML editing tools are mature

## Conclusion

The current YAML-first implementation has fundamental design flaws that violate core architectural principles. Field reordering should be a trivial YAML edit, not a cascade failure across multiple systems. 

Implementing true YAML-first architecture will:
- Eliminate cascade failures
- Improve system maintainability  
- Reduce development time
- Prevent configuration drift
- Enable confident rapid iteration

**Next Step**: Approve this architectural review and begin Phase 1 implementation planning.

---

*This document represents the current state analysis and should be updated as implementation progresses.*