# YAML Single Source of Truth Migration Plan

## Executive Summary

Convert the current fragmented field definition system to a single authoritative YAML schema that dynamically generates all field configurations, eliminating hardcoded field names throughout the codebase.

## Current State Analysis

### Hardcoded Field Name Locations
1. **YAML Files** (Multiple files with duplicated definitions)
   - `llama_single_pass_prompts.yaml`
   - `internvl3_prompts.yaml`
   - Model-specific prompt variations

2. **Python Configuration** (`common/config.py`)
   - `FIELD_GROUPS_DETAILED` - 25 hardcoded field names
   - `FIELD_GROUPS_COGNITIVE` - 25 hardcoded field names  
   - `FIELD_DEFINITIONS` - 25+ hardcoded field metadata
   - `GROUP_VALIDATION_RULES` - Hardcoded required field lists

3. **Processing Logic** (`common/grouped_extraction.py`)
   - Hardcoded field references in prompt templates
   - Fixed group-to-field mappings

4. **Parser Logic** (`common/extraction_parser.py`)
   - Hardcoded field name cleaning rules
   - Static field name mappings and transformations

5. **Validation & Metadata**
   - Field type classifications (NUMERIC_ID_FIELDS, MONETARY_FIELDS, etc.)
   - Required vs optional field categorizations

### Problems This Creates
- **25+ field names × 5+ locations = 125+ potential inconsistency points**
- **Deployment risk**: Different environments with mismatched definitions
- **Maintenance overhead**: Every field change requires 5+ file updates
- **Bug risk**: Easy to miss updating one location (as we just experienced)

## Target Architecture

### Single Schema File: `field_schema.yaml`
```yaml
# Master field schema - SINGLE SOURCE OF TRUTH
schema_version: "1.0"
total_fields: 25

fields:
  - name: "DOCUMENT_TYPE"
    type: "text"
    evaluation_logic: "fuzzy_text_match" 
    required: false
    group: "metadata"
    priority: 8
    description: "Type of business document"
    instruction: "[document type (invoice/receipt/statement) or NOT_FOUND]"
    
  - name: "BUSINESS_ABN"
    type: "numeric_id"
    evaluation_logic: "exact_numeric_match"
    required: true
    group: "critical" 
    priority: 1
    description: "Australian Business Number"
    instruction: "[11-digit Australian Business Number or NOT_FOUND]"

  # ... all 25 fields defined once with complete metadata

groups:
  critical:
    name: "Critical Business Identifiers"
    priority: 1
    max_tokens: 300
    temperature: 0.0
    description: "Most important fields for business validation"
    
  monetary:
    name: "Monetary Values"
    priority: 2
    max_tokens: 400
    temperature: 0.0
    description: "Financial amounts and calculations"
```

### Dynamic Configuration System
```python
# common/schema_loader.py
class FieldSchema:
    """Single source of truth for all field definitions"""
    
    def __init__(self, schema_file="field_schema.yaml"):
        self.schema = self._load_schema(schema_file)
        self._validate_schema()
    
    @property
    def field_names(self) -> List[str]:
        """Get ordered list of all field names"""
        return [f["name"] for f in self.schema["fields"]]
    
    def get_field_metadata(self, field_name: str) -> dict:
        """Get complete metadata for a field"""
        
    def get_group_config(self, group_name: str) -> dict:
        """Get group configuration with dynamic field assignment"""
        
    def get_fields_by_type(self, field_type: str) -> List[str]:
        """Get all fields of specific type (monetary, text, etc.)"""
```

## Migration Implementation Plan

### Phase 1: Create Master Schema (Week 1)
1. **Analyze Current Fields**
   ```bash
   # Audit all field definitions across files
   grep -r "BUSINESS_ABN\|TOTAL_AMOUNT\|SUPPLIER_NAME" --include="*.py" --include="*.yaml" .
   ```

2. **Create `field_schema.yaml`**
   - Consolidate all 25 fields from YAML files
   - Add complete metadata for each field
   - Define group structures
   - Add validation rules

3. **Build Schema Loader**
   ```python
   # common/schema_loader.py
   class FieldSchema:
       # Dynamic configuration generation
       # Field validation
       # Group management
   ```

### Phase 2: Replace config.py Hardcoding (Week 2)
1. **Remove Hardcoded Definitions**
   ```python
   # BEFORE (hardcoded)
   FIELD_GROUPS_DETAILED = {
       "critical": {"fields": ["BUSINESS_ABN", "TOTAL_AMOUNT"]},
       # ...
   }
   
   # AFTER (dynamic)
   schema = FieldSchema()
   FIELD_GROUPS_DETAILED = schema.generate_group_config("detailed_grouped")
   EXTRACTION_FIELDS = schema.field_names
   ```

2. **Dynamic Field Type Generation**
   ```python
   # Replace static lists with dynamic generation
   MONETARY_FIELDS = schema.get_fields_by_type("monetary")
   DATE_FIELDS = schema.get_fields_by_type("date")
   REQUIRED_FIELDS = schema.get_required_fields()
   ```

### Phase 3: Update Processing Logic (Week 2)
1. **Grouped Extraction Refactoring**
   ```python
   # common/grouped_extraction.py
   def generate_group_prompt(self, group_name: str) -> str:
       # Get fields dynamically from schema
       group_config = self.schema.get_group_config(group_name)
       fields = group_config["fields"]
       
       # Generate prompt dynamically
       for field in fields:
           field_meta = self.schema.get_field_metadata(field)
           instruction = field_meta["instruction"]
           prompt += f"{field}: {instruction}\n"
   ```

2. **Parser Dynamic Mapping**
   ```python
   # common/extraction_parser.py  
   def parse_extraction_response(response_text: str) -> Dict[str, str]:
       schema = FieldSchema()
       # Use dynamic field list instead of hardcoded EXTRACTION_FIELDS
       expected_fields = schema.field_names
   ```

### Phase 4: Model-Specific Prompt Generation (Week 3)
1. **Dynamic Prompt Generation**
   ```python
   # Generate model-specific prompts from schema
   def generate_model_prompts(model_name: str, schema: FieldSchema):
       template = schema.get_prompt_template(model_name)
       fields = schema.field_names
       # Generate complete prompt dynamically
   ```

2. **Remove Hardcoded YAML Files**
   - Replace `llama_single_pass_prompts.yaml` with dynamic generation
   - Replace `internvl3_prompts.yaml` with dynamic generation
   - Keep only `field_schema.yaml` as source

### Phase 5: Validation & Testing (Week 3)
1. **Schema Validation**
   ```python
   def validate_schema_completeness():
       # Ensure no hardcoded field names remain
       # Verify all 25 fields defined
       # Check group assignments complete
   ```

2. **Regression Testing**
   - Test field extraction accuracy maintained
   - Verify all model configurations work
   - Confirm grouped extraction functions

3. **Migration Verification**
   ```bash
   # Search for any remaining hardcoded field names
   grep -r "BUSINESS_ABN\|TOTAL_AMOUNT" --include="*.py" . 
   # Should only find dynamic references
   ```

## Implementation Priority

### High Priority (Week 1)
- [x] Create `field_schema.yaml` master file
- [x] Build `FieldSchema` loader class
- [x] Replace `EXTRACTION_FIELDS` with dynamic loading

### Medium Priority (Week 2)  
- [x] Replace `FIELD_GROUPS_DETAILED` and `FIELD_GROUPS_COGNITIVE`
- [x] Update grouped extraction to use dynamic fields
- [x] Replace parser hardcoded field lists

### Lower Priority (Week 3)
- [x] Dynamic prompt generation
- [x] Remove old YAML files
- [x] Comprehensive testing and validation

## Success Metrics

1. **Single Point of Change**: All field modifications require only `field_schema.yaml` updates
2. **Zero Hardcoding**: No field names hardcoded in Python files
3. **Consistency Guaranteed**: Impossible to have field name mismatches
4. **Maintainability**: Adding new fields requires single schema entry
5. **Performance Maintained**: 82.6%+ accuracy preserved after migration

## Risk Mitigation

1. **Backup Current State**: Git branch before migration
2. **Incremental Migration**: Phase-by-phase with testing
3. **Regression Testing**: Verify accuracy maintained at each phase
4. **Rollback Plan**: Can revert to hardcoded approach if needed

## Files to Modify

### Create New
- `common/field_schema.yaml` - Master schema
- `common/schema_loader.py` - Dynamic configuration system

### Modify Existing  
- `common/config.py` - Replace hardcoded with dynamic
- `common/grouped_extraction.py` - Dynamic field references
- `common/extraction_parser.py` - Dynamic field handling
- `models/llama_processor.py` - Use dynamic configuration
- `models/internvl3_processor.py` - Use dynamic configuration

### Remove Eventually
- `llama_single_pass_prompts.yaml` - Replace with dynamic generation
- `internvl3_prompts.yaml` - Replace with dynamic generation

This migration will eliminate the root cause of field name inconsistency bugs and create a truly maintainable system with single source of truth.