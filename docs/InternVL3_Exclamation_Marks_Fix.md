# InternVL3-8B Exclamation Marks Output Issue - Root Cause and Solution

## Problem Description

When running InternVL3-8B on a 4x V100 GPU system, the model outputs only exclamation marks (!!!!!...) instead of proper text responses, despite successful model loading and image preprocessing.

**Symptoms:**
- Model loads successfully with `device_map="auto"`
- Image preprocessing completes without errors
- GPU memory allocation appears normal across all 4 GPUs
- Generation completes successfully (no errors thrown)
- Output consists entirely of exclamation marks

## Root Cause Analysis

### Primary Issue: `device_map="auto"` Incompatibility

The root cause is using `device_map="auto"` for model loading when calling the `chat()` method:

```python
# PROBLEMATIC CODE
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"  # <-- CAUSES DEVICE MISMATCH
).eval()
```

**Why this fails:**

1. **Layer Distribution**: `device_map="auto"` distributes model layers across multiple GPUs (e.g., vision encoder on GPU 0, language model on GPU 1-3)

2. **chat() Method Assumptions**: The InternVL3 `chat()` method expects:
   - The entire model to be on a single device, OR
   - PyTorch's native multi-GPU handling (DataParallel/DDP)
   - NOT Hugging Face's device_map distribution

3. **Device Mismatch**: When pixel_values are on GPU 0 but language model layers are on GPU 1-3:
   - Internal tensor operations fail silently
   - Token generation produces invalid token IDs
   - Decoder maps invalid IDs to fallback characters (exclamation marks)

### Secondary Issues

1. **pad_token_id Configuration**: Explicitly setting `pad_token_id=tokenizer.eos_token_id` can interfere with InternVL3's internal tokenization

2. **do_sample=False Incompatibility**: Pure greedy decoding (`do_sample=False`) may not work correctly with InternVL3's vision-language architecture

## Solution

### 1. Replace device_map with DataParallel

**Before:**
```python
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()
```

**After:**
```python
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()

# Use DataParallel for 4x V100 multi-GPU utilization
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.cuda()
else:
    model = model.cuda()
```

### 2. Ensure Image Tensors on GPU

**Before:**
```python
pixel_values = load_image(imageName, max_num=12).to(torch.bfloat16)  # CPU
```

**After:**
```python
pixel_values = load_image(imageName, max_num=12).to(torch.bfloat16).cuda()
```

### 3. Use Deterministic Sampling-Based Generation

**Before:**
```python
generation_config = dict(
    max_new_tokens=2000,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

response, history = model.chat(
    tokenizer,
    pixel_values,
    formatted_question,
    generation_config,
    history=None,
    return_history=True
)
```

**After:**
```python
generation_config = dict(
    max_new_tokens=2000,
    do_sample=True,
    temperature=0.0,  # Zero temperature for deterministic output
    top_p=1.0
)

# Access chat() via .module when using DataParallel
chat_method = model.module.chat if hasattr(model, 'module') else model.chat

response, history = chat_method(
    tokenizer,
    pixel_values,
    formatted_question,
    generation_config,
    history=None,
    return_history=True
)
```

## Why This Solution Works

### DataParallel vs device_map="auto"

| Aspect | device_map="auto" | DataParallel |
|--------|-------------------|--------------|
| **Distribution** | Splits layers across GPUs | Replicates full model on each GPU |
| **Memory per GPU** | Lower (layers divided) | Higher (full model) |
| **Forward Pass** | Sequential through devices | Parallel batch processing |
| **chat() Compatibility** | ❌ Breaks device assumptions | ✅ Maintains single-device semantics |
| **Multi-GPU Utilization** | ✅ Yes | ✅ Yes |

### Deterministic Generation via Temperature

Setting `temperature=0.0` with `do_sample=True`:
- Uses the sampling codepath (compatible with vision-language architecture)
- Temperature of 0.0 makes sampling deterministic (always picks highest probability token)
- More reliable than `do_sample=False` for multimodal models

### DataParallel Method Access

When using `torch.nn.DataParallel`, the actual model is wrapped:
- Original model: `model.chat()`
- After DataParallel: `model.module.chat()`

The code checks for this automatically:
```python
chat_method = model.module.chat if hasattr(model, 'module') else model.chat
```

## Multi-GPU Memory Considerations

### Memory Usage with DataParallel

For InternVL3-8B on 4x V100 (16GB each):

**With device_map="auto":**
- GPU 0: ~4GB (vision encoder)
- GPU 1: ~5GB (language model layers 1-8)
- GPU 2: ~5GB (language model layers 9-16)
- GPU 3: ~4GB (language model layers 17-24)

**With DataParallel:**
- GPU 0: ~8GB (full model replica + input batching)
- GPU 1: ~8GB (full model replica)
- GPU 2: ~8GB (full model replica)
- GPU 3: ~8GB (full model replica)

DataParallel requires more memory per GPU but provides better compatibility.

### If Memory is Insufficient

If running out of memory with DataParallel:

1. **Reduce image tiles**: Change `max_num=12` to `max_num=6`
2. **Use fewer GPUs**: `device_ids=[0, 1]` instead of `[0, 1, 2, 3]`
3. **Enable 8-bit quantization**:
   ```python
   model = AutoModel.from_pretrained(
       model_id,
       load_in_8bit=True,
       device_map="balanced"  # Only use device_map with quantization
   )
   ```

## Multi-Turn Conversation Support

The solution maintains multi-turn conversation capability via the `history` parameter:

```python
# First turn
response1, history = chat_method(
    tokenizer,
    pixel_values,
    '<image>\nWhat is in this image?',
    generation_config,
    history=None,
    return_history=True
)

# Second turn (uses history from first turn)
response2, history = chat_method(
    tokenizer,
    None,  # No new image
    'Can you describe it in more detail?',
    generation_config,
    history=history,  # Pass previous history
    return_history=True
)
```

## Verification Steps

After applying the fix, verify correct operation:

1. **Check GPU Distribution**:
   ```python
   if hasattr(model, 'module'):
       print(f"✅ Model replicated across {torch.cuda.device_count()} GPUs")
   ```

2. **Monitor Memory Usage**:
   ```python
   for i in range(torch.cuda.device_count()):
       allocated = torch.cuda.memory_allocated(i) / 1e9
       print(f"GPU {i}: {allocated:.2f}GB")
   ```

3. **Test Text Output**:
   - Response should contain coherent text
   - No repeated characters or gibberish
   - Content should be relevant to the image

## References

- **GitHub Issue #1015**: [Setting pad_token_id to eos_token_id warning](https://github.com/OpenGVLab/InternVL/issues/1015)
- **GitHub Issue #967**: [Attention mask all True due to mismatched pad_token_id](https://github.com/OpenGVLab/InternVL/issues/967)
- **InternVL3 Official Docs**: [Quick Start Guide](https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html)
- **Hugging Face Model Card**: [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)

## Summary

**Problem**: InternVL3-8B outputs exclamation marks instead of text when using `device_map="auto"`

**Root Cause**: device_map distributes layers across GPUs, breaking chat() method's device placement assumptions

**Solution**: Use `torch.nn.DataParallel` instead of `device_map="auto"` for proper multi-GPU utilization

**Key Changes**:
1. Replace `device_map="auto"` with `torch.nn.DataParallel` for 4x V100 GPU utilization
2. Move pixel_values to GPU with `.cuda()`
3. Use `temperature=0.0` with `do_sample=True` for deterministic generation
4. Access chat via `model.module.chat()` when using DataParallel
5. Remove explicit `pad_token_id` setting from generation_config

**Result**: Proper text generation with multi-GPU utilization and deterministic output