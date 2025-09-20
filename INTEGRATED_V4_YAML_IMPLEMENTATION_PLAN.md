# Integrated V4 Schema + YAML-First Prompt Implementation Plan

## Executive Summary

**Dual Objective**: 
1. Upgrade Standard Processors from legacy 25-field extraction to v4 schema (49 fields) with document-type awareness
2. Implement YAML-first configurable prompt system to eliminate hardcoded paths and simplify maintenance

**Strategic Integration**: The YAML-first approach provides the foundation for managing v4's expanded prompt requirements, while the v4 upgrade provides the content for the new configurable prompt system.

## Current State Analysis

**V4 Schema Requirements**:
- Standard processors stuck on 25 legacy fields vs v4's 49 total fields
- Missing boss-mandated fields (payment status, GST details, transaction lists)
- Need document intelligence while maintaining standard processor interface

**Prompt Management Issues**:
- Hardcoded paths: `Path(__file__).parent.parent / "internvl3_prompts.yaml"`
- Maintenance requires Python file edits to change prompt locations
- No easy way to experiment with different prompt strategies

## Integrated Architecture Solution

### Phase 1: YAML-First Prompt Foundation (Day 1 - 4 hours)

#### Task 1.1: Create Configurable Prompt System
**File**: `/Users/tod/Desktop/LMM_POC/prompts/prompt_config.yaml`

```yaml
# prompts/prompt_config.yaml - Single source for all prompt paths
prompts:
  llama:
    single_pass: "llama_single_pass_v4.yaml"
    grouped: "llama_grouped_v4.yaml"
  internvl3:
    single_pass: "internvl3_single_pass_v4.yaml" 
    grouped: "internvl3_grouped_v4.yaml"

# Base directory for all prompt files
base_path: "prompts/"

# Schema compatibility
schema_version: "v4"
field_count: 49
```

#### Task 1.2: Simple Prompt Loader Implementation
**File**: `/Users/tod/Desktop/LMM_POC/common/prompt_loader.py`

```python
class PromptLoader:
    def __init__(self, config_file="prompts/prompt_config.yaml"):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
    
    def get_prompt_path(self, model_name: str, strategy: str) -> Path:
        """Get full path to prompt file"""
        prompt_file = self.config["prompts"][model_name][strategy]
        base_path = Path(self.config["base_path"])
        return base_path / prompt_file
        
    def get_schema_version(self) -> str:
        return self.config.get("schema_version", "v4")
        
    def get_field_count(self) -> int:
        return self.config.get("field_count", 49)
```

#### Task 1.3: Directory Structure Setup

```
LMM_POC/
├── prompts/
│   ├── prompt_config.yaml              # Master configuration
│   ├── llama_single_pass_v4.yaml       # V4-ready Llama prompts
│   ├── llama_grouped_v4.yaml
│   ├── internvl3_single_pass_v4.yaml   # V4-ready InternVL3 prompts
│   ├── internvl3_grouped_v4.yaml
│   └── experimental/                   # Optional: A/B testing
│       ├── llama_cot_enhanced.yaml
│       └── internvl3_precision_v2.yaml
```

### Phase 2: V4 Schema Integration (Day 2 - 6 hours)

#### Task 2.1: Update `common/config.py` for V4 Schema
**Integration Point**: Use YAML-configured prompts with v4 field loading

```python
def get_v4_field_list() -> List[str]:
    """Get all 49 unique fields from v4 schema."""
    from .schema_config import get_schema_config
    config = get_schema_config()
    return config.extraction_fields  # Returns 49 fields

def get_document_type_fields(document_type: str) -> List[str]:
    """Get fields specific to document type (25/19/17 per type)."""
    from .document_schema_loader import DocumentTypeFieldSchema
    loader = DocumentTypeFieldSchema("field_schema_v4.yaml")
    schema = loader.get_document_schema(document_type)
    return [field["name"] for field in schema["fields"]]

# Update field count constant
FIELD_COUNT = 49  # Updated from 25
```

#### Task 2.2: Document Type Detection Utility
**File**: `/Users/tod/Desktop/LMM_POC/common/document_type_detector.py`

```python
class LightweightDocumentDetector:
    """Simple document type detection for standard processors."""
    
    def __init__(self, prompt_loader):
        self.prompt_loader = prompt_loader
    
    def detect_document_type(self, image_path: str, model_extract_func) -> str:
        """Detect document type using configurable prompt."""
        # Load detection prompt from YAML config
        detection_prompt = self._get_detection_prompt()
        response = model_extract_func(image_path, detection_prompt, max_new_tokens=50)
        return self._parse_document_type(response)
        
    def _get_detection_prompt(self) -> str:
        """Get document type detection prompt from configuration."""
        # Could be loaded from prompt_config.yaml or inline
```

### Phase 3: Standard Processor Updates (Day 2-3 - 8 hours)

#### Task 3.1: Upgrade Llama Processor with Integrated System
**File**: `/Users/tod/Desktop/LMM_POC/models/llama_processor.py`

**Current Architecture**:
```python
def _load_single_pass_prompts(self):
    # OLD: Hardcoded path + 25 fields
    yaml_path = Path(__file__).parent.parent / "llama_single_pass_prompts.yaml"
```

**Integrated V4 + YAML Architecture**:
```python
def __init__(self, ..., enable_v4_schema=True):
    self.enable_v4_schema = enable_v4_schema
    self.prompt_loader = PromptLoader()
    self.document_detector = LightweightDocumentDetector(self.prompt_loader)

def _load_single_pass_prompts(self):
    # NEW: Configurable path from YAML config
    yaml_path = self.prompt_loader.get_prompt_path("llama", "single_pass")
    
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("single_pass", {})

def get_extraction_prompt(self, image_path=None):
    if self.enable_v4_schema and image_path:
        # V4: Document type detection + field filtering
        classification_info = self.document_detector.detect_and_classify_document(
            image_path
        )
        doc_type = classification_info['document_type']
        fields = get_document_type_fields(doc_type)  # 25/19/17 fields
    else:
        # V4: All 49 fields (fallback mode)
        fields = get_v4_field_list()
    
    return self._generate_prompt_for_fields(fields)
```

#### Task 3.2: Upgrade InternVL3 Processor with Integrated System
**File**: `/Users/tod/Desktop/LMM_POC/models/internvl3_processor.py`

**Mirror Llama integration**:
```python
def __init__(self, ..., enable_v4_schema=True):
    self.enable_v4_schema = enable_v4_schema
    self.prompt_loader = PromptLoader()
    self.document_detector = LightweightDocumentDetector(self.prompt_loader)

def _load_single_pass_prompts(self):
    # NEW: Configurable path + v4 schema support
    yaml_path = self.prompt_loader.get_prompt_path("internvl3", "single_pass")
    
    with yaml_path.open("r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts.get("single_pass", {})

def get_extraction_prompt(self, image_path=None):
    # Same v4 integration as Llama processor
```

### Phase 4: V4 Prompt Content Creation (Day 3 - 4 hours)

#### Task 4.1: Create V4-Ready Prompt Files
**Files**: All prompt YAML files in `prompts/` directory

**Enhanced V4 Llama Prompts**:
```yaml
# prompts/llama_single_pass_v4.yaml
single_pass:
  persona: "You are an expert document analyzer with comprehensive field extraction capabilities."
  field_instructions:
    # V4 Schema - All 49 fields with enhanced instructions
    DOCUMENT_TYPE: "[document type (invoice/receipt/bank_statement) or NOT_FOUND]"
    BUSINESS_ABN: "[11-digit Australian Business Number or NOT_FOUND]"
    
    # V4 New Fields
    IS_GST_INCLUDED: "[true/false for GST inclusion or NOT_FOUND]"
    TOTAL_AMOUNT_PAID: "[amount paid with $ symbol or NOT_FOUND]"
    BALANCE_OF_PAYMENT: "[remaining balance with $ symbol or NOT_FOUND]"
    LINE_ITEM_GST_AMOUNTS: "[GST per item with pipe separator or NOT_FOUND]"
    TRANSACTION_DATES: "[transaction dates with pipe separator or NOT_FOUND]"
    
    # ... all 49 fields
  
  closing_instruction: "Extract ALL visible fields. Use NOT_FOUND only when field is genuinely absent."
```

#### Task 4.2: Document-Type Specific Field Filtering
**Integration**: Prompt system supports document-aware field reduction

```python
def _generate_prompt_for_fields(self, field_list: List[str]) -> str:
    """Generate extraction prompt for specific field subset."""
    # Load base template from YAML
    template_path = self.prompt_loader.get_prompt_path(self.model_name, "single_pass")
    
    with template_path.open("r") as f:
        template = yaml.safe_load(f)
    
    # Filter field instructions to only requested fields
    all_instructions = template["single_pass"]["field_instructions"]
    filtered_instructions = {
        field: all_instructions[field] 
        for field in field_list 
        if field in all_instructions
    }
    
    # Build dynamic prompt with filtered fields
    return self._build_prompt_from_template(template, filtered_instructions)
```

### Phase 5: Response Processing Updates (Day 4 - 5 hours)

#### Task 5.1: Update Extraction Parser for V4
**File**: `/Users/tod/Desktop/LMM_POC/common/extraction_parser.py`

```python
def parse_extraction_response(response: str, expected_fields: List[str] = None):
    # NEW: Flexible field validation
    expected_fields = expected_fields or get_v4_field_list()  # 49 fields
    
    # V4 field type support
    if field_name in get_boolean_fields():
        parsed_value = parse_boolean_field(raw_value)
    elif field_name in get_calculated_fields():
        parsed_value = validate_calculated_field(raw_value)
    elif field_name in get_transaction_list_fields():
        parsed_value = parse_transaction_list(raw_value)
    
    return extracted_data
```

#### Task 5.2: Update Evaluation Logic for V4
**File**: `/Users/tod/Desktop/LMM_POC/common/evaluation_metrics.py`

```python
def calculate_field_accuracy(extracted: str, ground_truth: str, field_name: str):
    # V4 field type evaluation
    if field_name in get_boolean_fields():
        return evaluate_boolean_field(extracted, ground_truth)
    elif field_name in get_calculated_fields():
        return evaluate_calculated_field(extracted, ground_truth)
    elif field_name in get_transaction_list_fields():
        return evaluate_transaction_list(extracted, ground_truth)
    else:
        # Standard evaluation logic
        return standard_field_accuracy(extracted, ground_truth)
```

### Phase 6: Integration Testing & Validation (Day 4 - 3 hours)

#### Task 6.1: Comprehensive Integration Tests

```python
def test_integrated_v4_yaml_system():
    """Test both YAML configuration and V4 schema integration."""
    
    # Test 1: YAML prompt loading
    processor = LlamaProcessor(enable_v4_schema=True)
    assert processor.prompt_loader.get_field_count() == 49
    
    # Test 2: Document type detection + field filtering
    result = processor.process_single_image("test_invoice.png")
    assert len(result["extracted_data"]) <= 25  # Invoice fields only
    
    # Test 3: Configurable prompt paths
    assert "prompts/llama_single_pass_v4.yaml" in str(processor.prompt_loader.get_prompt_path("llama", "single_pass"))
    
    # Test 4: V4 field extraction
    v4_fields = ["IS_GST_INCLUDED", "TOTAL_AMOUNT_PAID", "TRANSACTION_DATES"]
    extracted_fields = result["extracted_data"].keys()
    v4_present = any(field in extracted_fields for field in v4_fields)
    assert v4_present, "V4-specific fields should be extracted"
```

#### Task 6.2: Backward Compatibility Validation
```python
def test_backward_compatibility():
    """Ensure system works with both v3 and v4 ground truth."""
    
    # Test with v3 ground truth (34 columns)
    processor = LlamaProcessor(enable_v4_schema=False)
    # Should still work with legacy 25-field extraction
    
    # Test with v4 ground truth (49 columns)  
    processor = LlamaProcessor(enable_v4_schema=True)
    # Should work with full v4 field set
```

## Integrated Implementation Timeline

| Day | Phase | YAML Tasks | V4 Schema Tasks | Duration |
|-----|-------|------------|-----------------|----------|
| **Day 1** | Foundation | Create prompt config system | Update config.py for 49 fields | 6 hours |
| **Day 2** | Processors | Update prompt loading | Add document detection + v4 integration | 8 hours |
| **Day 3** | Content | Create v4 prompt files | Implement field filtering logic | 8 hours |
| **Day 4** | Validation | Test prompt configuration | Test v4 schema + response processing | 6 hours |
| **Total** | **Integration** | **YAML + V4 Combined** | **Both systems working together** | **28 hours** |

## Integrated Benefits

### 1. **Maintenance Revolution**
```bash
# Before: Hunt through Python files for both prompts AND field definitions
vim models/llama_processor.py          # Change hardcoded prompt path
vim common/config.py                   # Change hardcoded field list

# After: Edit two clean configuration files
vim prompts/prompt_config.yaml         # Change any prompt file
vim common/field_schema_v4.yaml        # Change field definitions
```

### 2. **Easy V4 Experimentation**
```yaml
# Test different V4 prompt strategies by editing config
prompts:
  llama:
    single_pass: "experimental/llama_cot_v4_enhanced.yaml"  # Switch here
```

### 3. **Document-Type Intelligence**
- **Invoice detection** → 25 relevant fields → Faster processing
- **Receipt detection** → 19 relevant fields → Reduced tokens
- **Statement detection** → 17 relevant fields → Optimized accuracy

### 4. **Flexible Deployment**
- **Mac**: Test prompt changes without model loading
- **H200**: Full validation with v4 field extraction
- **V100**: Production deployment with battle-tested prompts

## Success Metrics

### YAML-First Success:
1. ✅ **Zero Hardcoded Paths**: All prompt files configurable via YAML
2. ✅ **Easy Maintenance**: Prompt changes via config file edits only
3. ✅ **Experimentation Ready**: Quick A/B testing of prompt strategies

### V4 Schema Success:
1. ✅ **49-Field Extraction**: All v4 schema fields accessible to processors  
2. ✅ **Document Intelligence**: Smart field filtering (25/19/17 per document type)
3. ✅ **Backward Compatibility**: Works with both v3 and v4 ground truth
4. ✅ **Performance**: Memory usage manageable, accuracy maintained/improved

### Integration Success:
1. ✅ **Seamless Operation**: Both systems work together without conflicts
2. ✅ **Clean Architecture**: Configuration separated from code logic
3. ✅ **Future-Proof**: Easy to add new models, prompts, or field types

## Risk Mitigation Strategy

### High Risk Items:
1. **Integration Complexity**: Two major changes simultaneously
   - **Mitigation**: Phase-by-phase implementation with testing at each step
   - **Fallback**: Feature flags to disable either YAML or v4 systems individually

2. **Memory Usage**: 49 fields may require more VRAM than 25 fields
   - **Mitigation**: Document-type filtering reduces active field count
   - **Fallback**: `enable_v4_schema=False` parameter for backward compatibility

3. **Prompt-Schema Mismatch**: YAML prompts not aligned with v4 schema
   - **Mitigation**: Validation system ensures prompt fields match schema fields
   - **Fallback**: Automatic field validation and warning system

### Rollback Plan:
1. **Immediate**: Set `enable_v4_schema=False` and use legacy prompt paths
2. **Partial**: Disable document detection, use full 49-field extraction  
3. **Full**: Revert to v3 schema with hardcoded prompt paths
4. **Emergency**: Git revert to last stable commit before integration

## Files Modified/Created

### New Files:
| File | Purpose | Priority |
|------|---------|----------|
| `prompts/prompt_config.yaml` | Master prompt configuration | High |
| `common/prompt_loader.py` | YAML-based prompt path resolution | High |
| `common/document_type_detector.py` | Lightweight document detection | Medium |
| `prompts/llama_single_pass_v4.yaml` | V4-ready Llama prompts | High |
| `prompts/internvl3_single_pass_v4.yaml` | V4-ready InternVL3 prompts | High |

### Modified Files:
| File | Changes | Integration Level |
|------|---------|-------------------|
| `common/config.py` | Add v4 field loading + prompt config support | Major |
| `models/llama_processor.py` | Replace hardcoded paths + add v4 support | Major |
| `models/internvl3_processor.py` | Replace hardcoded paths + add v4 support | Major |
| `common/extraction_parser.py` | Add v4 field type support | Medium |
| `common/evaluation_metrics.py` | Add v4 field evaluation | Medium |

This integrated plan delivers both maintainable YAML-first prompt management AND comprehensive v4 schema support in a single coherent implementation that's greater than the sum of its parts.