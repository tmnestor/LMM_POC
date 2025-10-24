# Multi-GPU Inference Issues: Exclamation Marks and Garbage Output

## Executive Summary

When using multi-GPU inference with vision-language models like InternVL3-8B, improper tensor device placement can cause the model to generate garbage output, including repetitive exclamation marks ("!!!!!!..."). This is a well-documented issue across multiple frameworks and models.

**Root Cause:** When model layers are distributed across GPUs without ensuring the first and last layers of the language model are on the same device, tensor device mismatches occur during forward passes, resulting in corrupted output.

---

## References

### 1. Exclamation Marks Symptom (Specific Case)

**Issue:** [Bug]: Llama-3.1-405B-Instruct-FP8 only generates exclamation marks
**Source:** vllm-project/vllm Issue #13035
**URL:** https://github.com/vllm-project/vllm/issues/13035
**Date:** Reported February 10, 2025
**Description:**
- Model generates endless exclamation marks in vLLM v0.7.2
- Issue resolved by downgrading to vLLM v0.6.6
- Affects Meta-Llama-3.1-405B-Instruct-FP8 model
- Occurred even with frequency penalties and different temperatures

**Relevance:** Demonstrates that "exclamation mark only" output is a documented symptom of inference configuration issues in large language models.

---

### 2. General Multi-GPU Gibberish/Garbage Output Issues

#### 2a. Transformers with Accelerate (RTX 4090)

**Issue:** Multi-GPU inference using accelerate giving inaccurate/gibberish results on RTX 4090s
**Source:** huggingface/transformers Issue #21720
**URL:** https://github.com/huggingface/transformers/issues/21720
**Description:**
- Gibberish output when using `device_map="auto"` or `device_map="balanced"` to spread model weights across multiple RTX 4090s
- Same setup works fine with 2x RTX 3090 and 2x A5000 GPUs
- Hardware-specific manifestation of multi-GPU inference issues

**Relevance:** Shows that multi-GPU inference with Accelerate's device_map can produce garbage output depending on configuration and hardware.

---

#### 2b. Hugging Face Forum Discussion

**Title:** Multi-GPU inference with LLM produces gibberish
**Source:** Hugging Face Transformers Forum
**URL:** https://discuss.huggingface.co/t/multi-gpu-inference-with-llm-produces-gibberish/35904
**Description:**
- Gibberish output when running LLaMA-7b with `device_map="auto"` across multiple GPUs
- Single GPU produces reasonable outputs
- Multi-GPU setup produces corrupted/nonsensical text

**Relevance:** Community-reported issue demonstrating that improper multi-GPU configuration leads to garbage output across different model sizes.

---

#### 2c. DeepSpeed Multi-GPU Inference

**Issue:** Garbage GPT-Neo-X output when using multi-gpu inference
**Source:** Microsoft DeepSpeed
**URL:** https://lightrun.com/answers/microsoft-deepspeed-bug-master-garbage-gpt-neo-x-output-when-using-multi-gpu-inference
**Description:**
- Garbage output issues with multi-GPU fp16 inference for GPT-Neo-X
- Affects DeepSpeed framework specifically

**Relevance:** Demonstrates that multi-GPU inference issues are framework-agnostic and affect multiple inference backends.

---

### 3. InternVL-Specific Multi-GPU Requirements

#### 3a. Official InternVL 3.0 Documentation

**Source:** InternVL 3.0 Quick Start Documentation
**URL:** https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html
**Key Quote:**
> "The code is designed to avoid errors during multi-GPU inference by ensuring that the first and last layers of the large language model (LLM) are on the same device."

**Implementation:**
- Provides `split_model()` function to create proper device_map
- Vision model placed on GPU 0
- Critical components (embeddings, output layers, rotary_emb) on GPU 0
- **Last LLM layer explicitly placed on GPU 0** (same as first layer)
- Middle layers distributed across available GPUs

**Also Documented In:**
- InternVL 1.5 Quick Start: https://internvl.readthedocs.io/en/latest/internvl1.5/quick_start.html
- InternVL 2.0 Quick Start: https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html
- InternVL 2.5 Quick Start: https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html

**Relevance:** Official documentation explicitly addresses the requirement for first/last layer placement to prevent multi-GPU inference errors.

---

## Technical Analysis

### Why First and Last Layers Must Be on Same Device

**Problem:**
When model layers are distributed across GPUs:
1. Input tensors enter the model on one device (e.g., GPU 0)
2. Intermediate layers may be on different devices (GPU 1, 2, 3)
3. Output layers may be on a different device than input layers
4. Tensor transfers between devices can cause:
   - Device mismatch errors
   - Corrupted gradients/activations
   - Garbage output (including repetitive tokens like "!!!!")

**Solution:**
Ensure critical path components are co-located:
- First layer (embeddings) on GPU 0
- Last layer (output projections) on GPU 0
- This ensures input→output path starts and ends on same device
- Intermediate layers can be distributed for parallelism

### InternVL3-8B Specific Issue

**Observation:**
- InternVL3-2B works correctly with standard device_map
- InternVL3-8B produces only exclamation marks with same configuration

**Root Cause:**
The provided `split_model()` function had a bug:
```python
# BUGGY CODE:
for i, num_layer in enumerate(num_layers_per_gpu):
    for j in range(num_layer):
        if layer_cnt < num_layers:  # Includes last layer in loop
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

# Later assignment may not override:
device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
```

**Fix:**
```python
# FIXED CODE:
for i, num_layer in enumerate(num_layers_per_gpu):
    for _ in range(num_layer):
        if layer_cnt < num_layers - 1:  # EXCLUDE last layer from loop
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

# Explicitly assign last layer to GPU 0:
device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
```

---

## Findings Summary

### 1. Documented Issues
- ✅ Exclamation mark output is a documented symptom (vLLM Issue #13035)
- ✅ Multi-GPU gibberish/garbage output is well-documented across frameworks
- ✅ InternVL explicitly requires first/last layers on same device (official docs)

### 2. Root Cause
Tensor device mismatches during multi-GPU inference when layers are improperly distributed

### 3. Framework Scope
Issue affects multiple frameworks:
- vLLM
- Hugging Face Transformers + Accelerate
- DeepSpeed
- Vision-language models (InternVL family)

### 4. Solution
Proper device_map configuration ensuring:
- First and last LLM layers on same device
- Vision components co-located with LLM input/output
- Critical components (embeddings, norms, output projections) on primary device

---

## Recommendations

### For InternVL3 Users:
1. Use the corrected `split_model()` function (see `split_model_fixed.py`)
2. Verify last layer placement with debug output
3. Test output quality after loading model to confirm proper distribution

### For Reporting:
- Multi-GPU garbage output is a **known, documented issue** across multiple frameworks
- InternVL **explicitly documents** the requirement for proper layer placement
- The exclamation mark symptom, while less common, is **documented in vLLM**
- Fix is straightforward: ensure first/last layers are co-located

---

## Additional Resources

- **Accelerate Multi-GPU Issues:** https://github.com/huggingface/accelerate/issues/1518
- **InternVL GitHub:** https://github.com/OpenGVLab/InternVL
- **Transformers Device Mapping:** https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling

---

**Document Version:** 1.0
**Date:** 2025-01-21
**Author:** Technical analysis based on publicly available documentation and issue trackers
