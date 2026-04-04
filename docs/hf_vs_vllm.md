# HuggingFace vs vLLM: Accuracy and Throughput

InternVL3.5 models (8B, 14B, 38B) across three benchmarks, comparing HuggingFace (native) inference against vLLM tensor-parallel inference. Both run on 2x NVIDIA L40S (44 GB each).

## SROIE (626 scanned receipts, 4 fields, single-image)

| Model | HF F1 | vLLM F1 | Delta | HF (img/min) | vLLM (img/min) | Speedup |
|-------|-------|---------|-------|-------------|---------------|---------|
| 8B  | 0.7787 | 0.8214 | +4.3% | 14.6 | 37.5 | 2.6x |
| 14B | 0.8118 | 0.8426 | +3.1% | 12.0 | 25.0 | 2.1x |
| 38B | 0.8174 | 0.8482 | +3.1% | 5.8 | 10.7 | 1.9x |

## WildReceipt (472 in-the-wild receipts, 12 fields, single-image)

### InternVL3.5

| Model | HF F1 | vLLM F1 | Delta | HF (img/min) | vLLM (img/min) | Speedup |
|-------|-------|---------|-------|-------------|---------------|---------|
| 8B  | 0.7227 | 0.7319 | +0.9% | 4.3 | 15.0 | 3.5x |
| 14B | 0.7184 | 0.7231 | +0.5% | 3.6 | 9.2 | 2.5x |
| 38B | 0.7304 | 0.7308 | +0.0% | 1.8 | 4.1 | 2.3x |

### Other models — WildReceipt (vLLM only)

| Model | vLLM F1 | img/min | Notes |
|-------|---------|---------|-------|
| Gemma 4 31B-it | 0.7556 | 3.8 | 2x L40S, max_soft_tokens=560 |
| Qwen3.5-27B | 0.7386 | 4.4 | Early-fusion VLM, 2x L40S |

## Bank Statements (15 synthetic statements, 5 fields, multi-turn)

| Model | HF Acc | vLLM Acc | Delta | HF (img/min) | vLLM (img/min) | Speedup |
|-------|--------|----------|-------|-------------|---------------|---------|
| 8B  | 86.95% | 87.31% | +0.4% | 0.8 | 3.9 | 5.1x |
| 14B | 84.64% | 80.40% | -4.2% | 0.6 | 2.3 | 3.8x |
| 38B | 86.73% | 82.34% | -4.4% | 0.4 | 1.3 | 3.3x |

## Key Findings

### Throughput

- vLLM delivers **2-5x speedup** across all benchmarks and model sizes.
- Speedup is largest for smaller models (8B) and multi-turn workloads (bank statements).
- WildReceipt images are more complex than SROIE scans, resulting in lower throughput for both backends but a larger relative speedup for vLLM (3.5x vs 2.6x at 8B).
- Bank statements require ~10-20 turns per image, making throughput much lower for both backends. vLLM's KV cache reuse across turns gives it the largest advantage here (5.1x at 8B).

### Accuracy

- On single-image tasks (SROIE, WildReceipt), vLLM **matches or exceeds** HF accuracy.
- A ~4% accuracy drop appears only on bank statements (14B, 38B), which use multi-turn sequential extraction (10-20 turns per image).

### Likely cause of bank statement accuracy drop: KV cache reuse

vLLM's prefix caching — the same mechanism that delivers its largest speedup on bank statements — is the most likely source of the accuracy drop. Candidate mechanisms:

1. **KV cache precision**: vLLM may store cached KV values at reduced precision (e.g. FP8). Over 10-20 turns, small numerical errors accumulate. HF recomputes attention from scratch each turn.
2. **Chat template differences**: vLLM's tokenizer may format multi-turn conversation history differently than InternVL's native `.chat()` method, which has its own history formatting logic.
3. **Paged attention numerical drift**: Paged attention uses non-contiguous memory blocks. While mathematically equivalent for a single pass, floating-point differences can compound across many turns.

The pattern supports this hypothesis: **8B shows no accuracy drop** while **14B and 38B do** — larger models with more attention heads may amplify small numerical differences across turns.

### Suggested experiments

- Disable prefix caching (`enable_prefix_caching=False`) and re-run bank statements to isolate the KV cache effect.
- Compare the KV cache dtype vLLM uses vs HF's native dtype.
