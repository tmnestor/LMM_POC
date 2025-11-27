# V100 to A10G GPU Migration Plan

**Date:** 2025-11-28
**Author:** Claude Code
**Status:** Draft for Review

---

## Overview

This document outlines the modifications required to migrate the LMM_POC codebase from V100 GPUs to A10G GPUs. The codebase contains V100-specific optimizations that need updating to take advantage of A10G's improved capabilities.

---

## Hardware Comparison

| Specification | V100 (Current) | A10G (Target) | Impact |
|---------------|----------------|---------------|--------|
| **Memory** | 16GB HBM2 | 24GB GDDR6 | +50% memory headroom |
| **Architecture** | Volta | Ampere | Modern features available |
| **Compute Capability** | SM 7.0 | SM 8.6 | Better kernel support |
| **Flash Attention** | Not supported | Supported | Faster attention ops |
| **bfloat16** | Limited support | Native support | Better precision handling |
| **TF32** | No | Yes | Faster matrix ops |
| **Memory Bandwidth** | 900 GB/s | 600 GB/s | Slightly lower |

### Key A10G Advantages
- Flash Attention 2 support (significant speedup for transformers)
- Native bfloat16 with tensor cores
- 50% more VRAM than 16GB V100
- Better memory efficiency with Ampere architecture

### A10G Limitations
- Lower memory bandwidth than V100 HBM2
- Still insufficient for Llama-3.2-11B in full precision (~28GB needed)

---

## Files Requiring Modification

### 1. `common/dynamic_gpu_config.py`

**Purpose:** Add A10G to explicit GPU architecture detection

**Current Code (lines 79-105):**
```python
def _detect_architecture(self, name: str, compute_capability: tuple) -> str:
    name_upper = name.upper()

    # High-memory datacenter GPUs
    if any(gpu in name_upper for gpu in ['H200', 'H100', 'A100']):
        return 'datacenter_high_memory'

    # Professional workstation GPUs
    elif any(gpu in name_upper for gpu in ['L40S', 'L40', 'RTX A6000']):
        return 'workstation_high_memory'

    # Legacy datacenter GPUs
    elif any(gpu in name_upper for gpu in ['V100', 'P100']):
        return 'legacy_datacenter'
```

**Proposed Change:**
```python
def _detect_architecture(self, name: str, compute_capability: tuple) -> str:
    name_upper = name.upper()

    # High-memory datacenter GPUs
    if any(gpu in name_upper for gpu in ['H200', 'H100', 'A100']):
        return 'datacenter_high_memory'

    # Professional workstation GPUs
    elif any(gpu in name_upper for gpu in ['L40S', 'L40', 'RTX A6000']):
        return 'workstation_high_memory'

    # Mid-tier datacenter GPUs (Ampere)
    elif any(gpu in name_upper for gpu in ['A10G', 'A10', 'A30']):
        return 'datacenter_mid_memory'

    # Legacy datacenter GPUs
    elif any(gpu in name_upper for gpu in ['V100', 'P100']):
        return 'legacy_datacenter'
```

**Additional Changes Required:**
- Add `'datacenter_mid_memory'` handling to `_calculate_memory_buffer()` (line 107-116)
- Update memory buffer calculation for A10G characteristics

**Proposed `_calculate_memory_buffer()` update:**
```python
def _calculate_memory_buffer(self, memory_gb: float, architecture: str) -> float:
    """Calculate appropriate memory buffer based on GPU characteristics."""
    if architecture == 'datacenter_high_memory':
        return min(20.0, memory_gb * 0.15)  # 15% buffer, max 20GB
    elif architecture == 'datacenter_mid_memory':
        return min(6.0, memory_gb * 0.20)   # 20% buffer, max 6GB (A10G optimized)
    elif architecture in ['workstation_high_memory', 'consumer_high_end']:
        return min(12.0, memory_gb * 0.20)  # 20% buffer, max 12GB
    elif architecture == 'legacy_datacenter':
        return min(4.0, memory_gb * 0.25)   # 25% buffer, max 4GB
    else:
        return min(8.0, memory_gb * 0.20)   # 20% buffer, max 8GB
```

---

### 2. `common/llama_model_loader_robust.py`

**Purpose:** Add A10G-specific memory detection and quantization logic

**Current Code (lines 163-195):** V100-specific detection only

**Proposed Change:** Add A10G detection block after V100 block (around line 195):

```python
# A10G-specific logic (after V100 block)
a10g_gpus = [gpu for gpu in memory_result.per_gpu_info
             if gpu.is_available and "A10G" in gpu.name.upper()]
a10g_detected = len(a10g_gpus) > 0

if a10g_detected:
    a10g_count = len(a10g_gpus)
    a10g_total_memory = sum(gpu.total_memory_gb for gpu in a10g_gpus)

    # A10G threshold: ~28GB for Llama-3.2-11B-Vision
    # Single A10G (24GB) needs quantization
    # 2x A10G (48GB) can run full precision
    a10g_threshold = estimated_memory_needed + memory_buffer  # ~28GB

    if a10g_total_memory >= a10g_threshold:
        if use_quantization:
            if verbose:
                rprint(f"[green]🚀 {a10g_count}x A10G setup ({a10g_total_memory:.0f}GB), disabling quantization[/green]")
            use_quantization = False
        else:
            if verbose:
                rprint(f"[green]✅ {a10g_count}x A10G with {a10g_total_memory:.0f}GB - full precision[/green]")
    else:
        if not use_quantization:
            if verbose:
                rprint(f"[yellow]⚠️ {a10g_count}x A10G ({a10g_total_memory:.0f}GB) needs quantization for Llama-3.2-11B[/yellow]")
            use_quantization = True
        else:
            if verbose:
                rprint(f"[yellow]⚠️ {a10g_count}x A10G - using quantization (24GB < 28GB needed)[/yellow]")
```

**Rationale:**
- Single A10G (24GB) cannot fit Llama-3.2-11B (~28GB with buffer)
- Must use 8-bit quantization for single A10G
- 2x A10G (48GB) can run full precision

---

### 3. `common/gpu_optimization.py`

**Purpose:** Update CUDA memory allocation for Ampere architecture

**Current Code (lines 51-68):**
```python
is_v100 = "V100" in gpu_name

if is_v100:
    cuda_alloc_config = "max_split_size_mb:32"
else:
    cuda_alloc_config = "max_split_size_mb:64"
```

**Proposed Change:**
```python
gpu_name_upper = gpu_name.upper()
is_v100 = "V100" in gpu_name_upper
is_ampere = any(gpu in gpu_name_upper for gpu in ['A10G', 'A10', 'A30', 'A100', 'RTX 30', 'RTX 40'])

if is_v100:
    # V100: Ultra-aggressive fragmentation prevention (older HBM2)
    cuda_alloc_config = "max_split_size_mb:32"
    if verbose:
        print("🎯 V100 detected: Using aggressive memory settings (32MB blocks)")
elif is_ampere:
    # Ampere: Better memory management, can use larger blocks
    cuda_alloc_config = "max_split_size_mb:128"
    if verbose:
        print("🎯 Ampere GPU detected: Using optimized memory settings (128MB blocks)")
else:
    # Default: Standard settings
    cuda_alloc_config = "max_split_size_mb:64"
    if verbose:
        print("🎯 Using standard memory settings (64MB blocks)")
```

**Rationale:**
- Ampere has better memory management than Volta
- Larger block sizes (128MB) reduce fragmentation overhead
- More efficient for the improved memory allocator

---

### 4. `ivl3_8b_bank_statement_batch_v2.py`

**Purpose:** Update MAX_TILES configuration for A10G

**Current Code (lines 95-96):**
```python
# V100 TILE CONFIGURATION
"MAX_TILES": 14,  # V100 optimized - InternVL3-8B config default
```

**Proposed Change:**
```python
# GPU TILE CONFIGURATION (dynamically adjusted by memory)
# V100 (16GB): 14 tiles, A10G (24GB): 18 tiles, H200 (80GB): 32 tiles
"MAX_TILES": 18,  # A10G optimized - increased from V100's 14
```

**Alternative (Dynamic):**
```python
# Import at top of file
from common.dynamic_gpu_config import get_gpu_config

# In CONFIG section
gpu_config = get_gpu_config()
"MAX_TILES": gpu_config.max_tiles if gpu_config else 14,
```

---

### 5. `ivl3_5_8b_bank_statement_batch_v2.py`

**Purpose:** Enable Flash Attention for A10G (currently H200-only)

**Current Code (lines 53-54):**
```python
# H200 precision settings
"TORCH_DTYPE": "bfloat16",
"USE_FLASH_ATTN": True,
```

**Proposed Change:** Make Flash Attention dynamic based on GPU capability

```python
# Precision and attention settings (dynamic based on GPU)
"TORCH_DTYPE": "bfloat16",
"USE_FLASH_ATTN": True,  # Supported on Ampere (A10G) and newer
```

**Additional Change in `load_internvl3_5_model()` (around line 210):**
```python
# Dynamic Flash Attention detection
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

# Flash Attention requires SM 8.0+ (Ampere and newer)
supports_flash_attn = compute_cap >= (8, 0)
use_flash_attn = CONFIG["USE_FLASH_ATTN"] and supports_flash_attn

if verbose:
    if use_flash_attn:
        print(f"  Flash Attention: Enabled (compute capability {compute_cap})")
    else:
        print(f"  Flash Attention: Disabled (compute capability {compute_cap} < 8.0)")
```

---

### 6. `llama_bank_statement_batch_v2.py`

**Purpose:** No direct V100 references, but uses `llama_model_loader_robust.py`

**Current Code (lines 958-966):**
```python
model, processor = load_llama_model_robust(
    model_path=CONFIG["MODEL_PATH"],
    use_quantization=False,  # Will be overridden based on GPU detection
    device_map="auto",
    max_new_tokens=4096,
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    verbose=True,
)
```

**Proposed Change:** Add explicit A10G guidance in comments

```python
# A10G Note: Single A10G (24GB) requires quantization for Llama-3.2-11B
# The loader will auto-detect and enable quantization if needed
model, processor = load_llama_model_robust(
    model_path=CONFIG["MODEL_PATH"],
    use_quantization=False,  # Auto-overridden: True for 1x A10G, False for 2x+ A10G
    device_map="auto",
    max_new_tokens=4096,
    torch_dtype="bfloat16",  # Native support on A10G Ampere
    low_cpu_mem_usage=True,
    verbose=True,
)
```

---

## Summary of Changes

| File | Change Type | Priority | Effort |
|------|-------------|----------|--------|
| `common/dynamic_gpu_config.py` | Add A10G architecture | HIGH | Low |
| `common/llama_model_loader_robust.py` | Add A10G detection | HIGH | Medium |
| `common/gpu_optimization.py` | Ampere memory settings | MEDIUM | Low |
| `ivl3_8b_bank_statement_batch_v2.py` | Update MAX_TILES | MEDIUM | Low |
| `ivl3_5_8b_bank_statement_batch_v2.py` | Dynamic Flash Attention | MEDIUM | Low |
| `llama_bank_statement_batch_v2.py` | Comments only | LOW | Trivial |

---

## Testing Checklist

After implementing changes, verify:

- [ ] A10G is correctly detected as `datacenter_mid_memory` architecture
- [ ] Single A10G automatically enables 8-bit quantization for Llama
- [ ] Flash Attention is enabled on A10G (compute capability 8.6)
- [ ] MAX_TILES is set appropriately (18 for A10G vs 14 for V100)
- [ ] Memory allocation uses 128MB blocks on A10G
- [ ] No V100-specific log messages appear when running on A10G
- [ ] InternVL3 models load successfully with Flash Attention
- [ ] Llama-3.2-Vision loads with quantization on single A10G

---

## Rollback Plan

If issues arise on A10G:

1. **Quick Fix:** Force quantization in `llama_bank_statement_batch_v2.py`:
   ```python
   use_quantization=True,  # Force for A10G
   ```

2. **Disable Flash Attention:**
   ```python
   "USE_FLASH_ATTN": False,
   ```

3. **Reduce MAX_TILES:**
   ```python
   "MAX_TILES": 12,  # Conservative
   ```

---

## Appendix: A10G Memory Budget

### Llama-3.2-11B-Vision on Single A10G (24GB)

| Component | Full Precision | 8-bit Quantized |
|-----------|----------------|-----------------|
| Model weights | ~22GB | ~11GB |
| KV cache (4096 tokens) | ~2GB | ~1GB |
| Activations | ~3GB | ~2GB |
| **Total** | **~27GB** | **~14GB** |
| A10G Available | 24GB | 24GB |
| **Fit?** | ❌ No | ✅ Yes |

### InternVL3-8B on Single A10G (24GB)

| Component | Full Precision | 8-bit Quantized |
|-----------|----------------|-----------------|
| Model weights | ~16GB | ~8GB |
| KV cache | ~1.5GB | ~0.8GB |
| Activations | ~2GB | ~1.5GB |
| **Total** | **~19.5GB** | **~10.3GB** |
| A10G Available | 24GB | 24GB |
| **Fit?** | ✅ Yes (tight) | ✅ Yes |

---

## Approval

- [ ] Reviewed by: ________________
- [ ] Approved for implementation: ________________
- [ ] Date: ________________
