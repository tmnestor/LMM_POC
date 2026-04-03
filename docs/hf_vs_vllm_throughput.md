# HuggingFace vs vLLM Throughput Comparison

InternVL3.5 models across three benchmarks, comparing HuggingFace (native) inference against vLLM tensor-parallel inference. Both run on 2x NVIDIA L40S (44 GB each).

## SROIE (626 images, single-image extraction)

| Model | HF (img/min) | vLLM (img/min) | Speedup | HF Total | vLLM Total |
|-------|-------------|---------------|---------|----------|------------|
| 8B  | 14.6 | 37.5 | 2.6x | 43 min | 17 min |
| 14B | 12.0 | 25.0 | 2.1x | 52 min | 25 min |
| 38B | 5.8 | 10.7 | 1.9x | 108 min | 58 min |

## WildReceipt (472 images, single-image extraction)

| Model | HF (img/min) | vLLM (img/min) | Speedup | HF Total | vLLM Total |
|-------|-------------|---------------|---------|----------|------------|
| 8B  | 4.3 | 15.0 | 3.5x | 111 min | 31 min |
| 14B | 3.6 | 9.2 | 2.5x | 130 min | 51 min |
| 38B | 1.8 | 4.1 | 2.3x | 261 min | 114 min |

## Bank Statements (15 images, multi-turn sequential extraction)

| Model | HF (img/min) | vLLM (img/min) | Speedup | HF Total | vLLM Total |
|-------|-------------|---------------|---------|----------|------------|
| 8B  | 0.8 | 3.9 | 5.1x | 20 min | 4 min |
| 14B | 0.6 | 2.3 | 3.8x | 24 min | 6 min |
| 38B | 0.4 | 1.3 | 3.3x | 39 min | 12 min |

## Key Findings

- vLLM delivers **2-5x speedup** across all benchmarks and model sizes.
- Speedup is largest for smaller models (8B) and multi-turn workloads (bank statements).
- WildReceipt images are more complex than SROIE scans, resulting in lower throughput for both backends but a larger relative speedup for vLLM (3.5x vs 2.6x at 8B).
- Bank statements require ~10-20 turns per image, making throughput much lower for both backends. vLLM's KV cache reuse across turns gives it the largest advantage here (5.1x at 8B).
- The throughput gains come at no accuracy cost for single-image tasks, but with a ~4% accuracy drop on multi-turn bank statement extraction for 14B and 38B models (see [hf_vs_vllm_accuracy.md](hf_vs_vllm_accuracy.md)).
