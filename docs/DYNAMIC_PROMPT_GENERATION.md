# Dynamic Prompt Generation System

**Schema-Driven Vision-Language Model Prompt Generation**

This document provides comprehensive guidance on the dynamic prompt generation system that powers the LMM_POC vision-language document extraction pipeline. The system generates model-specific prompts from YAML schema templates, replacing hardcoded prompt files with flexible, maintainable configuration.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Schema Configuration](#schema-configuration)
- [Extraction Strategies](#extraction-strategies)
- [Model-Specific Templates](#model-specific-templates)
- [CLI Usage Examples](#cli-usage-examples)
- [Prompt Generation Process](#prompt-generation-process)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

### Single Source of Truth
All prompt generation is driven by `common/field_schema.yaml`:
```
field_schema.yaml
├── fields[]                    # 25 document fields with metadata
├── grouping_strategies         # field_grouped vs detailed_grouped
├── cognitive_groups           # 6-group logical organization  
└── model_prompt_templates     # Model-specific generation rules
    ├── llama                  # Llama-3.2-Vision templates
    └── internvl3              # InternVL3 templates
```

### Core Components
1. **`common/schema_loader.py`** - Schema parsing and prompt generation engine
2. **`common/field_schema.yaml`** - Central configuration and templates  
3. **`models/*_processor.py`** - Model-specific prompt integration
4. **`common/grouped_extraction.py`** - Multi-group extraction orchestration

## Schema Configuration

### Field Definition Structure
```yaml
fields:
  - name: "BUSINESS_ABN"
    type: "numeric_id"
    evaluation_logic: "exact_numeric_match"
    required: true
    group: "regulatory_financial"  # Cognitive group assignment
    description: "11-digit Australian Business Number"
    instruction: "[11-digit Australian Business Number or NOT_FOUND]"
```

### Grouping Strategies
Two extraction approaches are supported:

#### 1. Single-Pass Strategy
```yaml
single_pass:
  name: "Complete Document Extraction"  
  description: "Extract all 25 fields in one model call"
  max_tokens: 800
  temperature: 0.0
```

#### 2. Field-Grouped Strategy (6 Groups)
```yaml
field_grouped:
  name: "6-Group Cognitive Optimization"
  description: "Cognitive load research-based grouping"
  groups:
    - regulatory_financial     # Business ID and financial totals
    - entity_contacts         # Supplier and payer information  
    - transaction_details     # Line items and pricing
    - temporal_data          # Dates and time ranges
    - banking_payment        # Banking and payment details
    - document_metadata      # Document type and classification
```

#### 3. Detailed-Grouped Strategy (8 Groups)
```yaml
detailed_grouped:
  name: "8-Group Detailed Extraction"
  description: "Proven stable strategy with 8 focused groups"
  groups:
    - critical              # Most important: ABN, TOTAL_AMOUNT
    - monetary             # All financial amounts
    - dates               # Temporal information  
    - business_entity     # Supplier details
    - payer_info         # Customer details
    - banking            # Banking information
    - item_details       # Line items and quantities
    - metadata           # Document classification
```

## Model-Specific Templates

### Llama-3.2-Vision Templates
```yaml
model_prompt_templates:
  llama:
    single_pass:
      opening_text: "Extract key-value data from this business document image."
      critical_instructions:
        - "Use EXACT field names as shown below"
        - "Output format: FIELD_NAME: value"
        - "Use 'NOT_FOUND' if field not visible"
      
    field_grouped:
      regulatory_financial:
        expertise_frame: "Extract critical business identifiers and ALL financial amounts with precision."
        cognitive_context: "The Australian Business Number (BUSINESS_ABN) is an 11 digit number..."
        focus_instruction: "Extract the BUSINESS_ABN (11 digits total) and all monetary amounts..."
```

### InternVL3 Templates  
```yaml
model_prompt_templates:
  internvl3:
    single_pass:
      opening_text: "Extract data from this business document."
      output_instruction: "Output ALL fields below with their exact keys."
      
    field_grouped:
      regulatory_financial:
        expertise_frame: "Extract business ID and financial amounts."
        cognitive_context: "BUSINESS_ABN is 11 digits. TOTAL_AMOUNT is final amount due..."
        focus_instruction: "Find ABN (11 digits) and all dollar amounts..."
```

## CLI Usage Examples

### Available CLI Arguments

#### InternVL3 Key-Value Extraction
```bash
python internvl3_keyvalue.py [-h] [--extraction-mode {single_pass,field_grouped,detailed_grouped,adaptive}] [--debug] [--limit-images LIMIT_IMAGES]

Arguments:
  --extraction-mode    Extraction strategy (default: detailed_grouped)
                      • single_pass: All 25 fields in one model call
                      • field_grouped: 6 logical groups (regulatory_financial, entity_contacts, etc.)
                      • detailed_grouped: 8 focused groups (critical, monetary, dates, etc.)  
                      • adaptive: Dynamic strategy selection based on document
  --debug             Enable debug output and detailed logging
  --limit-images N    Process only first N images (useful for testing)
```

#### Llama Key-Value Extraction
```bash
python llama_keyvalue.py [-h] [--extraction-mode {single_pass,field_grouped,detailed_grouped,adaptive}] [--debug] [--limit-images LIMIT_IMAGES]

Arguments: (Same as InternVL3)
```

### 1. Single Document Processing

#### InternVL3 Field-Grouped (6 logical groups)
```bash
# Use field_grouped extraction (regulatory_financial, entity_contacts, etc.)
python internvl3_keyvalue.py \
  --extraction-mode field_grouped \
  --limit-images 1 \
  --debug
```

#### InternVL3 Detailed-Grouped (8 focused groups - Production)
```bash
# Use detailed_grouped extraction (critical, monetary, dates, etc.)
python internvl3_keyvalue.py \
  --extraction-mode detailed_grouped \
  --limit-images 1 \
  --debug
```

#### InternVL3 Single-Pass Extraction
```bash  
# Use single-pass extraction (all fields at once)
python internvl3_keyvalue.py \
  --extraction-mode single_pass \
  --limit-images 1 \
  --debug
```

### 2. Batch Processing

#### Full Dataset Evaluation
```bash
# Process all 20 evaluation images with production strategy (detailed_grouped default)
python internvl3_keyvalue.py --debug
```

#### Strategy Performance Comparison
```bash
# Compare all extraction strategies on same dataset
python internvl3_keyvalue.py --extraction-mode single_pass --limit-images 5 --debug
python internvl3_keyvalue.py --extraction-mode field_grouped --limit-images 5 --debug  
python internvl3_keyvalue.py --extraction-mode detailed_grouped --limit-images 5 --debug
python internvl3_keyvalue.py --extraction-mode adaptive --limit-images 5 --debug
```

### 3. Development and Testing

#### Schema Validation
```bash
# Test schema loading and prompt generation
python test_phase4_migration.py
```

#### Interactive Development
```bash
# Start Jupyter for prompt development
jupyter notebook
# Open: notebooks/internvl3_keyvalue.ipynb
```

#### Prompt Generation Testing
```python
# Test prompt generation in Python REPL
from common.schema_loader import get_global_schema

schema = get_global_schema()

# Generate single-pass prompt for InternVL3
prompt = schema.generate_dynamic_prompt('internvl3', 'single_pass')
print(f"Generated prompt ({len(prompt)} chars):")
print(prompt)

# Generate grouped prompt for specific group
llama_prompt = schema.generate_dynamic_prompt(
    'llama', 'field_grouped', 'regulatory_financial'
)
```

## Prompt Generation Process

### 1. Schema Loading
```python
# Automatic schema loading with caching
from common.schema_loader import get_global_schema
schema = get_global_schema()  # Loads common/field_schema.yaml
```

### 2. Strategy Selection
The system supports three extraction strategies:
- **`single_pass`**: All 25 fields in one call
- **`field_grouped`**: 6 cognitive groups (research-optimized)  
- **`detailed_grouped`**: 8 focused groups (production-stable)

### 3. Dynamic Generation
```python
# Single-pass generation
prompt = schema.generate_dynamic_prompt(
    model_name='internvl3',     # 'llama' or 'internvl3'
    strategy='single_pass'       # Strategy selection
)

# Grouped generation  
prompt = schema.generate_dynamic_prompt(
    model_name='llama',
    strategy='field_grouped',
    group_name='regulatory_financial',  # Specific group
    fields=['BUSINESS_ABN', 'TOTAL_AMOUNT']  # Override fields
)
```

### 4. Template Resolution
The system resolves templates using this hierarchy:
1. **Model-specific template** (`model_prompt_templates.{model}.{strategy}`)
2. **Group-specific template** (for grouped strategies)
3. **Field metadata** (descriptions, instructions, formatting rules)
4. **Output formatting** (field list, validation rules)

## Advanced Configuration

### Custom Field Groupings
```yaml
# Define custom cognitive groups
cognitive_groups:
  regulatory_financial:
    name: "Regulatory and Financial Core"
    description: "Business identification and monetary amounts"
    cognitive_load: "high"
    fields:
      - BUSINESS_ABN
      - TOTAL_AMOUNT  
      - GST_AMOUNT
      - SUBTOTAL_AMOUNT
```

### Model-Specific Optimizations
```yaml
# Llama-specific optimizations
model_prompt_templates:
  llama:
    field_grouped:
      regulatory_financial:
        # Detailed cognitive context for complex reasoning
        cognitive_context: "The Australian Business Number (BUSINESS_ABN) is an 11 digit number structured as a 9 digit identifier with two leading check digits..."
        # Specific error-prevention instructions
        focus_instruction: "Extract the BUSINESS_ABN (11 digits total) and all monetary amounts. Double-check decimal places..."

# InternVL3-specific optimizations  
model_prompt_templates:
  internvl3:
    field_grouped:
      regulatory_financial:
        # Simplified for faster processing
        cognitive_context: "BUSINESS_ABN is 11 digits. TOTAL_AMOUNT is final amount due..."
        # Concise instructions
        focus_instruction: "Find ABN (11 digits) and all dollar amounts. Check decimal places carefully."
```

## Troubleshooting

### Common Issues

#### 1. Schema Loading Errors
```bash
❌ FATAL: Schema file not found: common/field_schema.yaml
💡 Expected location: /Users/tod/Desktop/LMM_POC/common/field_schema.yaml
💡 Verify: File exists and is readable
```
**Solution**: Ensure `common/field_schema.yaml` exists and is properly formatted.

#### 2. Template Resolution Failures  
```bash
❌ FATAL: Schema-based prompt generation failed for InternVL3
💡 Root cause: No template found for model 'internvl3' strategy 'single_pass'
💡 Expected: Model templates in common/field_schema.yaml
💡 Fix: Ensure schema contains model_prompt_templates.internvl3.single_pass
```
**Solution**: Add missing template to `model_prompt_templates` section.

#### 3. Group Name Mismatches
```bash
❌ FATAL: Schema-based prompt generation failed for group 'regulatory_financial'
💡 Root cause: Group 'regulatory_financial' not found in strategy 'detailed_grouped'
💡 Available groups: ['critical', 'monetary', 'dates', 'business_entity', 'payer_info', 'banking', 'item_details', 'metadata']
```
**Solution**: Use correct group names for the selected strategy or switch strategies.

### Debugging Commands

#### Schema Validation
```bash
# Validate schema structure
/opt/homebrew/Caskroom/miniforge/base/envs/unified_vision_processor/bin/python -c "
from common.schema_loader import get_global_schema
schema = get_global_schema()
print(f'✅ Schema loaded: {len(schema.fields)} fields')
print(f'✅ Available models: {list(schema.schema.get(\"model_prompt_templates\", {}).keys())}')
"
```

#### Template Inspection
```bash
# Inspect available templates
python -c "
from common.schema_loader import get_global_schema
schema = get_global_schema()
for model in ['llama', 'internvl3']:
    for strategy in ['single_pass', 'field_grouped']:
        try:
            template = schema.get_model_prompt_template(model, strategy)
            print(f'✅ {model}.{strategy}: {len(str(template))} chars')
        except Exception as e:
            print(f'❌ {model}.{strategy}: {e}')
"
```

### Performance Optimization

#### Memory Management
```bash
# Monitor memory usage during generation
python -c "
import torch
from common.schema_loader import get_global_schema
print(f'Memory before: {torch.cuda.memory_allocated()/1e9:.2f}GB')
schema = get_global_schema() 
prompt = schema.generate_dynamic_prompt('internvl3', 'single_pass')
print(f'Memory after: {torch.cuda.memory_allocated()/1e9:.2f}GB')
print(f'Prompt length: {len(prompt)} characters')
"
```

#### Batch Processing Optimization
```bash
# Optimize for batch processing with field_grouped strategy
export PYTHONPATH=/path/to/LMM_POC:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python internvl3_keyvalue.py \
  --extraction-mode field_grouped \
  --limit-images 10 \
  --debug
```

## Migration Notes

### From Hardcoded YAML Files
This system replaces the following legacy files:
- ❌ `llama_prompts.yaml` → ✅ Schema-driven generation
- ❌ `internvl3_prompts.yaml` → ✅ Schema-driven generation  
- ❌ `llama_single_pass_prompts.yaml` → ✅ Schema-driven generation

### Backward Compatibility
The system uses **fail-fast architecture** with explicit diagnostics instead of graceful fallbacks:
- **No silent fallbacks** to prevent mysterious failures
- **Explicit error messages** with remediation steps
- **Clear validation** at startup to catch configuration issues early

### Future Enhancements
1. **Multi-language Support**: Extend templates for non-English documents
2. **Dynamic Field Selection**: Runtime field filtering based on document type  
3. **Performance Profiling**: Built-in timing and memory usage tracking
4. **Template Versioning**: Support multiple template versions with migration tools

---

**Last Updated**: August 24, 2025  
**Version**: Phase 4 - Schema-Driven Architecture  
**Status**: Production Ready ✅