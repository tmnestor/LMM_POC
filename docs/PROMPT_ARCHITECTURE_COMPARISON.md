# Llama vs InternVL3: Prompt Generation & Processing Architecture Comparison

## Executive Summary

This document provides a comprehensive comparison of prompt generation and processing strategies across four vision-language model implementations in the LMM_POC codebase:

1. **Standard Processors**: Full 25-field extraction using schema-driven single-pass prompts
2. **Document-Aware Processors**: Dynamic field extraction using hybrid grouped strategies with document type detection

The analysis reveals fundamental architectural differences in how Llama and InternVL3 handle prompt construction, model integration, and response processing.

## Quick Reference: Implementation Matrix

| Processor Type | File Path | Primary Strategy | Field Count | Prompt Source |
|----------------|-----------|------------------|-------------|---------------|
| **Llama Standard** | `models/llama_processor.py` | Single-pass or Grouped | 25 fixed | Schema templates (`field_schema.yaml`) |
| **Llama Document-Aware** | `models/document_aware_llama_processor.py` | Dynamic single-pass | Variable | YAML config + field type detection |
| **InternVL3 Standard** | `models/internvl3_processor.py` | Single-pass or Grouped | 25 fixed | Schema templates (`field_schema.yaml`) |
| **InternVL3 Document-Aware** | `models/document_aware_internvl3_processor.py` | Hybrid grouped extraction | Variable | `DocumentAwareGroupedExtraction` class |

## Core Architecture Differences

### Standard Processors (Llama & InternVL3)
- **Fixed Field Extraction**: Always extract 25 predefined fields from `EXTRACTION_FIELDS`
- **Schema-Driven Prompts**: Use `schema_loader.generate_dynamic_prompt()` with model templates
- **Multiple Strategies**: Support single-pass, grouped, and adaptive extraction modes
- **Consistent Interface**: Both inherit similar patterns for batch processing

### Document-Aware Processors  
- **Dynamic Field Lists**: Accept variable field lists based on document type
- **Hybrid Intelligence**: Combine document type detection with grouped extraction
- **Specialized Prompting**: Field-type aware instructions and cleaner integration
- **No Inheritance**: Complete standalone implementations optimized for flexibility

## Prompt Generation Detailed Comparison

### 1. Standard Llama Processor

```python
def get_extraction_prompt(self):
    """Schema-based dynamic generation for Llama"""
    if self.extraction_mode == "single_pass":
        # Use centralized schema system
        schema = get_global_schema()
        prompt = schema.generate_dynamic_prompt(
            model_name="llama", 
            strategy="single_pass"
        )
        return prompt
    else:
        # Grouped mode uses extraction strategy
        return self._get_config_prompt()
```

**Key Features:**
- Uses chat template formatting: `processor.apply_chat_template(messages, add_generation_prompt=True)`
- Multimodal message structure with image and text components
- Fallback to hardcoded prompts if schema unavailable
- Complex conversation artifact cleaning for response parsing

### 2. Document-Aware Llama Processor

```python
def generate_dynamic_prompt(self) -> str:
    """Generate prompt for specific field list with v4 field type support"""
    yaml_config = self._load_yaml_config()
    
    if yaml_config:
        return self._generate_yaml_prompt(yaml_config)
    else:
        return self._generate_simple_prompt()
        
def _get_field_type_instruction(self, field: str) -> str:
    """Generate field-specific instructions based on field type"""
    # Field type detection for monetary, list, phone fields etc.
```

**Key Features:**
- Dynamic field list support (not fixed to 25 fields)
- Field type awareness (monetary, list, phone, address detection)
- YAML configuration with fallback to programmatic generation
- Integration with `ExtractionCleaner` for value normalization

### 3. Standard InternVL3 Processor

```python
def get_extraction_prompt(self):
    """Schema-driven generation for InternVL3"""
    schema = get_global_schema()
    prompt = schema.generate_dynamic_prompt(
        model_name="internvl3",
        strategy="single_pass"
    )
    return prompt
```

**Key Features:**
- Direct prompt format: `<image>\n{prompt}` (no chat template)
- Dynamic image preprocessing with tile-based approach
- Deterministic generation enforced (`do_sample=False`)
- Model size awareness (2B vs 8B) for memory optimization

### 4. Document-Aware InternVL3 Processor

```python
def process_single_image(self, image_path: str) -> dict:
    """Hybrid document-aware grouped extraction"""
    # USE THE HYBRID SYSTEM!
    extracted_data, metadata = self.grouped_extractor.extract_with_document_awareness(
        image_path,
        self._extract_with_custom_prompt,  # Model extractor function
        self.field_list  # All fields for fallback
    )
```

**Key Features:**
- Delegates to `DocumentAwareGroupedExtraction` class
- Two-phase extraction: document type detection → grouped field extraction
- Smart group filtering based on document type
- No direct prompt generation - uses grouped extraction system

## Prompt Template Systems

### Schema-Based Templates (`field_schema.yaml`)

Both standard processors use centralized templates:

```yaml
model_prompt_templates:
  llama:
    single_pass:
      critical_instructions:
        - "Output ONLY the structured data below"
        - "Start immediately with {first_field}"
        - "Stop immediately after {last_field}"
      
  internvl3:
    single_pass:
      opening_text: "Extract structured data from this business document image."
      output_instruction: "Output ALL fields below with their exact keys."
```

### Document-Aware Grouped Templates

Document-aware processors use the `DocumentAwareGroupedExtraction` class:

```python
DOCUMENT_AWARE_FIELD_GROUPS = {
    "regulatory_financial": {
        "fields": ["BUSINESS_ABN", "TOTAL_AMOUNT", ...],
        "expertise_frame": "Extract business ID and financial amounts.",
        "cognitive_context": "BUSINESS_ABN is 11 digits...",
        "focus_instruction": "Find ABN and all dollar amounts..."
    },
    # ... more groups
}
```

## Model Integration Patterns

### Llama Integration
```python
# Standard & Document-Aware both use:
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]
}]
input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = self.processor(image, input_text, return_tensors="pt")
```

### InternVL3 Integration
```python
# Both InternVL3 processors use:
question = f"<image>\n{prompt}"
response = self.model.chat(
    self.tokenizer,
    pixel_values,
    question,
    generation_config,
    history=None,
    return_history=False
)
```

## Response Processing & Parsing

### Standard Processors
- Use `parse_extraction_response()` from `extraction_parser.py`
- Llama: Requires `clean_conversation_artifacts=True` to remove chat markers
- InternVL3: Direct parsing without artifact cleaning
- Both validate field presence against `EXTRACTION_FIELDS`

### Document-Aware Processors
- Llama: Custom parsing with field type normalization via `ExtractionCleaner`
- InternVL3: Delegates parsing to grouped extraction system
- Both support variable field validation based on dynamic field lists

## Memory Management & Optimization

### Llama Resilient Generation
```python
def _resilient_generate(self, inputs, **generation_kwargs):
    """Multi-tier OOM fallback strategy"""
    try:
        # Tier 1: Standard generation
        return self.model.generate(**inputs, **generation_kwargs)
    except torch.cuda.OutOfMemoryError:
        # Tier 2: OffloadedCache
        generation_kwargs["cache_implementation"] = "offloaded"
        # Tier 3: Model reload
        # Tier 4: CPU fallback
```

### InternVL3 Optimizations
- 8B model: Automatic 8-bit quantization with bitsandbytes
- 2B model: Standard bfloat16 precision
- Both: Aggressive memory cleanup between batches
- Dynamic tile reduction for 8B model (max_num=6 vs 12)

## Performance Characteristics

| Metric | Llama Standard | Llama Doc-Aware | InternVL3 Standard | InternVL3 Doc-Aware |
|--------|---------------|-----------------|-------------------|---------------------|
| **Memory Usage** | ~22GB (11B model) | ~22GB | 4GB (2B) / 8-10GB (8B) | 4GB / 8-10GB |
| **Batch Size** | Auto-detected | Auto-detected | Configured per model | Auto-detected |
| **Token Limit** | Dynamic (25 fields) | Dynamic (variable) | Model-specific | Dynamic |
| **Extraction Accuracy** | 91.8% reported | Not measured | 90.6% (grouped) | 69.3% current |
| **Processing Strategy** | Single/Grouped | Single-pass | Single/Grouped | Hybrid grouped |

## Key Implementation Decisions

### Why Standard Processors Use Schema Templates
1. **Consistency**: Centralized prompt management across models
2. **Versioning**: Easy to update prompts without code changes  
3. **Fail-Fast**: Explicit errors if templates missing
4. **Optimization**: Model-specific tuning in YAML

### Why Document-Aware Uses Hybrid Approach
1. **Flexibility**: Variable field lists per document type
2. **Intelligence**: Document type detection reduces unnecessary extraction
3. **Accuracy**: Grouped extraction with focused prompts (90.6% proven)
4. **Efficiency**: Only process relevant field groups

## Best Practices & Recommendations

### When to Use Standard Processors
- Fixed extraction requirements (always 25 fields)
- Batch processing of similar documents
- Need for consistent output structure
- Integration with existing evaluation pipelines

### When to Use Document-Aware Processors
- Variable document types (invoices, receipts, statements)
- Field requirements vary by document
- Optimization for specific document types needed
- Memory-constrained environments (fewer fields = less tokens)

### Prompt Engineering Guidelines

1. **Llama Models**: 
   - Use conversation structure with clear role separation
   - Include explicit start/stop instructions
   - Clean conversation artifacts from responses

2. **InternVL3 Models**:
   - Use direct prompting without conversation markers
   - Enforce deterministic generation (`do_sample=False`)
   - Adjust tile count based on model size

3. **Document-Aware Systems**:
   - Implement field type detection for better instructions
   - Use grouped extraction for related fields
   - Include document type detection as first step

## Architecture Evolution Path

The codebase shows clear evolution:

1. **Phase 1**: Standard single-pass extraction (both models)
2. **Phase 2**: Grouped extraction for accuracy improvement
3. **Phase 3**: Document-aware intelligence layer
4. **Phase 4**: Hybrid systems combining all approaches

Current focus is on pushing document-aware InternVL3 from 69.3% to 90%+ accuracy through prompt refinement and ground truth alignment.

## Conclusion

The comparison reveals two distinct architectural philosophies:

- **Standard processors** prioritize consistency and comprehensive extraction
- **Document-aware processors** prioritize efficiency and contextual intelligence

Both Llama and InternVL3 implementations follow similar patterns within their categories, with model-specific optimizations for memory management and prompt formatting. The hybrid document-aware approach represents the most sophisticated evolution, combining multiple strategies for optimal results.