# InternVL3 Notebook Versions - Complete Comparison

## Three Versions Available

### 1. `ivl3_2b_batch_non_quantized.ipynb` (ORIGINAL)
**Black-box processor version** - Simple and fast

### 2. `ivl3_2b_batch_adaptive.ipynb` (HYBRID)
**Explicit stages with processor** - Transparent but uses wrapper

### 3. `ivl3_2b_batch_pure_adaptive.ipynb` (PURE) ⭐ NEW
**TRUE multi-turn with direct model calls** - Exactly like Llama

---

## Quick Comparison Table

| Feature | Original | Hybrid | Pure |
|---------|----------|--------|------|
| **Cells** | 26 | 27 | 26 |
| **Processor** | DocumentAwareInternVL3HybridProcessor | DocumentAwareInternVL3HybridProcessor | None (direct calls) |
| **Multi-turn chat** | ❌ No | ❌ Function exists but unused | ✅ YES - fully functional |
| **Conversation history** | ❌ No | ❌ No | ✅ YES - across all stages |
| **Stage visibility** | ❌ Hidden | ✅ Explicit loop | ✅ Explicit loop |
| **Prompt loading** | Automatic (processor) | Automatic (processor) | Manual from YAML |
| **Response parsing** | Automatic (processor) | Automatic (processor) | Manual parsing |
| **Intermediate data** | ❌ Not saved | ✅ Saved | ✅ Saved |
| **Code verbosity** | Low | Medium | High |
| **Control level** | Low | Medium | Full |
| **Like Llama?** | No | Partially | ✅ Exactly |

---

## Detailed Comparison

### 1. Original Version (`ivl3_2b_batch_non_quantized.ipynb`)

**Purpose**: Simple batch processing with proven infrastructure

**How it works**:
```python
# One-liner batch processing
batch_results, processing_times, document_types_found = processor.process_batch(
    all_images, verbose=CONFIG['VERBOSE']
)
```

**Pros**:
- ✅ Simplest code
- ✅ Fastest to write/modify
- ✅ Proven reliable
- ✅ Comprehensive analytics

**Cons**:
- ❌ Black-box processing
- ❌ No visibility into stages
- ❌ Can't access intermediate responses
- ❌ No multi-turn capability
- ❌ Different from Llama pattern

**When to use**:
- Quick batch processing
- Don't need debugging info
- Want standard reports
- Prefer simplicity

---

### 2. Hybrid Version (`ivl3_2b_batch_adaptive.ipynb`)

**Purpose**: Transparent processing while keeping processor convenience

**How it works**:
```python
# Explicit loop but using processor methods
for image_path in track(all_images):
    # Stage 0: Document type
    classification_result = hybrid_processor.detect_and_classify_document(image_path)

    # Stage 1: Structure (if bank statement)
    if bank_statement:
        structure_type = classify_bank_statement_structure_vision(...)

    # Stage 2: Extraction
    extraction_result = hybrid_processor.process_document_aware(image_path, classification_result)
```

**Pros**:
- ✅ Explicit stage visibility
- ✅ Saves intermediate responses
- ✅ Still uses processor convenience
- ✅ Easier than pure version
- ✅ Llama-compatible CSV output

**Cons**:
- ❌ Multi-turn function exists but NEVER USED
- ❌ No conversation history (processor doesn't support it)
- ❌ Still semi-black-box (processor methods)
- ❌ Not exactly like Llama

**When to use**:
- Want transparency but not complexity
- Need intermediate responses
- Comparing with Llama results
- Don't need true multi-turn

---

### 3. Pure Version (`ivl3_2b_batch_pure_adaptive.ipynb`) ⭐

**Purpose**: TRUE multi-turn chat with full control (Llama pattern)

**How it works**:
```python
# Manual everything - full control
for image_path in track(image_files):
    # Load image manually
    pixel_values = load_image(str(image_path))

    # Initialize conversation
    messages = []

    # Stage 0: Document type (first turn)
    doctype_answer, messages = chat_with_internvl(
        model, tokenizer, DOCTYPE_PROMPT, pixel_values, messages,
        max_new_tokens=50
    )

    # Stage 1: Structure (second turn - SAME conversation)
    if bank_statement:
        structure_answer, messages = chat_with_internvl(
            model, tokenizer, STRUCTURE_PROMPT, pixel_values, messages,
            max_new_tokens=50
        )

    # Stage 2: Extraction (third turn - SAME conversation)
    extraction_result, messages = chat_with_internvl(
        model, tokenizer, extraction_prompt, pixel_values, messages,
        max_new_tokens=2000
    )
```

**Pros**:
- ✅ TRUE multi-turn conversation
- ✅ Full conversation history maintained
- ✅ Direct model access
- ✅ Complete control over parameters
- ✅ Exactly mirrors Llama pattern
- ✅ No processor abstraction
- ✅ Can modify any aspect

**Cons**:
- ⚠️ More verbose code
- ⚠️ Manual prompt loading
- ⚠️ Manual response parsing
- ⚠️ More to maintain

**When to use**:
- Need TRUE multi-turn capability
- Want exact Llama equivalence
- Need full control
- Building custom workflows
- Research/experimentation

---

## File Organization

```
LMM_POC/
├── ivl3_2b_batch_non_quantized.ipynb       # Original (26 cells)
├── ivl3_2b_batch_adaptive.ipynb            # Hybrid (27 cells)
├── ivl3_2b_batch_pure_adaptive.ipynb       # Pure (26 cells) ⭐ NEW
├── build_adaptive_notebook.py              # Builder for hybrid
├── build_pure_adaptive_notebook.py         # Builder for pure ⭐ NEW
└── NOTEBOOK_VERSIONS_COMPARISON.md         # This file
```

---

## Multi-Turn Chat: The Key Difference

### Original & Hybrid: NO Multi-Turn

Even though the hybrid version has `chat_with_internvl()` defined, it's **never used**:

```python
# Hybrid version - processor methods don't use conversation history
classification_result = hybrid_processor.detect_and_classify_document(...)  # Fresh call
extraction_result = hybrid_processor.process_document_aware(...)            # Fresh call
# No messages passed between them!
```

### Pure: TRUE Multi-Turn

The pure version maintains conversation across all stages:

```python
# Pure version - TRUE conversation flow
messages = []  # Empty history

# Turn 1: "What type of document is this?"
doctype_answer, messages = chat_with_internvl(...)
# messages now contains: [user_prompt, assistant_response]

# Turn 2: "What structure does it have?" (model remembers previous conversation)
structure_answer, messages = chat_with_internvl(..., messages=messages)
# messages now contains: [turn1_user, turn1_assistant, turn2_user, turn2_assistant]

# Turn 3: "Extract the fields" (model remembers entire conversation)
extraction_result, messages = chat_with_internvl(..., messages=messages)
# Full conversation history maintained
```

**Why this matters**:
- Model can reference previous responses
- Can ask follow-up questions
- Can correct misunderstandings
- More natural conversation flow

---

## Output Files

All three versions produce Llama-compatible CSV files:

### Original:
`internvl3_non_quantized_batch_results_{timestamp}.csv`

### Hybrid:
`internvl3_adaptive_results_{timestamp}.csv`

### Pure:
`internvl3_pure_adaptive_results_{timestamp}.csv`

All include the same columns:
- Core: `image_file`, `document_type`, `structure_type`, `prompt_used`
- Intermediate: `doctype_classification`, `structure_classification`, `extraction_raw`
- Fields: All extraction fields

---

## Which Version Should You Use?

### Use **Original** if:
- ✅ You want simplest code
- ✅ You don't need intermediate responses
- ✅ You want standard analytics/reports
- ✅ You trust the processor

### Use **Hybrid** if:
- ✅ You want to see explicit stages
- ✅ You need intermediate VLM responses
- ✅ You want Llama-compatible output
- ✅ You don't need multi-turn
- ✅ You want processor convenience

### Use **Pure** if:
- ✅ You need TRUE multi-turn conversations
- ✅ You want exact Llama equivalence
- ✅ You need full control over everything
- ✅ You're doing research/experiments
- ✅ You want to modify the conversation flow

---

## Code Example: Multi-Turn Conversation

### Hybrid (NO multi-turn):
```python
# Each call is independent
result1 = hybrid_processor.detect_and_classify_document(image_path)
# Model forgets what it just said ⬆️

result2 = hybrid_processor.process_document_aware(image_path, result1)
# Model has no memory of previous call ⬆️
```

### Pure (TRUE multi-turn):
```python
messages = []

# First turn
response1, messages = chat_with_internvl(model, tokenizer, "What type?", pixels, messages)
# messages = [[user: "What type?", assistant: "Invoice"]]

# Second turn - model REMEMBERS first turn
response2, messages = chat_with_internvl(model, tokenizer, "Extract fields from that invoice", pixels, messages)
# messages = [[user: "What type?", assistant: "Invoice"],
#            [user: "Extract fields from that invoice", assistant: "FIELD1: value1..."]]
# Model knows "that invoice" refers to the invoice from turn 1! ⬆️
```

---

## Rebuilding Notebooks

If you need to modify and rebuild:

```bash
# Rebuild hybrid version
python build_adaptive_notebook.py

# Rebuild pure version
python build_pure_adaptive_notebook.py
```

Both scripts:
- ✅ Guarantee correct cell order
- ✅ Can be re-run safely
- ✅ Easy to modify for new features

---

## Summary

You now have **three options** for InternVL3 batch processing:

1. **Original** - Fast & simple, black-box
2. **Hybrid** - Transparent stages, uses processor
3. **Pure** - TRUE multi-turn, full control ⭐

All three are production-ready for V100. Choose based on your needs:
- **Speed → Original**
- **Transparency → Hybrid**
- **Control → Pure**

The **Pure version** is the only one with true multi-turn conversation capability, making it equivalent to the Llama pattern.
