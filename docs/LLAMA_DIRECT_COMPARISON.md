# Llama Direct Prompting vs Chat-based Comparison

## Overview
This document compares the two Llama implementation approaches now available in the LMM_POC project.

## Implementations

### 1. Chat-based Llama (Original)
**File**: `llama_keyvalue.py`  
**Processor**: `models/llama_processor.py`  
**Model**: `Llama-3.2-11B-Vision-Instruct`

```python
# Uses conversation template
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Requires heavy post-processing
if "assistant\n\n" in response:
    response = response.split("assistant\n\n")[-1].strip()
extracted_data = parse_extraction_response(response, clean_conversation_artifacts=True)
```

### 2. Direct Prompting Llama (New)
**File**: `llama_keyvalue_direct.py`  
**Processor**: `models/llama_direct_processor.py`  
**Model**: `Llama-3.2-11B-Vision` (base)

```python
# Uses direct prompting
direct_prompt = "<image>\n{prompt}"
inputs = processor(image, direct_prompt, return_tensors="pt")

# Minimal post-processing
extracted_data = parse_extraction_response(response, clean_conversation_artifacts=False)
```

## Key Differences

| Aspect | **Chat-based** | **Direct** |
|--------|----------------|------------|
| **Model Type** | -Instruct (conversation-trained) | Base (pre-trained) |
| **Prompting** | Chat template with roles | Direct `<image>\nprompt` |
| **Post-processing** | Heavy artifact cleaning | Minimal cleaning |
| **Output Quality** | Verbose with conversation artifacts | Clean, direct responses |
| **Similarity to** | Conversational AI systems | InternVL3 approach |

## Usage

### Run Chat-based Version
```bash
python llama_keyvalue.py
```
**Outputs**: `llama_batch_extraction_*.csv`, `llama_evaluation_results_*.json`

### Run Direct Version
```bash
python llama_keyvalue_direct.py
```
**Outputs**: `llama_direct_batch_extraction_*.csv`, `llama_direct_evaluation_results_*.json`

## Configuration
Update `common/config.py` to set model paths:
```python
# Chat-based (existing)
LLAMA_MODEL_PATH = f"{MODELS_BASE}/Llama-3.2-11B-Vision-Instruct"

# Direct prompting (new)
LLAMA_DIRECT_MODEL_PATH = f"{MODELS_BASE}/Llama-3.2-11B-Vision"
```

## Expected Benefits of Direct Approach
1. **Cleaner outputs** - No conversation artifacts to remove
2. **Simpler pipeline** - Direct input → direct output
3. **Better comparison** - More similar to InternVL3's clean approach
4. **Reduced processing** - Less post-processing overhead

## Comparison Methodology
Both implementations:
- Use identical prompts and evaluation metrics
- Process the same images with same fields
- Generate comparable CSV and report outputs
- Support the same batch processing and error handling

This enables direct performance comparison to isolate the impact of prompting strategy vs. model architecture differences.

## Notes
- Both versions require the same hardware (16GB+ VRAM or 8-bit quantization)
- Direct version expects base model, not instruct model
- Chat version may have better instruction following due to conversation training
- Direct version produces outputs more similar to InternVL3 format