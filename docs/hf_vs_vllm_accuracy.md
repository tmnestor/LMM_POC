# HuggingFace vs vLLM Accuracy Comparison

InternVL3.5 models across three benchmarks, comparing HuggingFace (native) inference against vLLM tensor-parallel inference.

## SROIE (347 scanned receipts, 4 fields, single-image)

| Model | HF F1 | vLLM F1 | Delta |
|-------|-------|---------|-------|
| 8B  | 0.7787 | 0.8214 | +4.3% |
| 14B | 0.8118 | 0.8426 | +3.1% |
| 38B | 0.8174 | 0.8482 | +3.1% |

## WildReceipt (472 in-the-wild receipts, 12 fields, single-image)

| Model | HF F1 | vLLM F1 | Delta |
|-------|-------|---------|-------|
| 8B  | 0.7227 | 0.7319 | +0.9% |
| 14B | 0.7184 | 0.7231 | +0.5% |
| 38B | 0.7304 | 0.7308 | +0.0% |

## Bank Statements (15 synthetic statements, 5 fields, multi-turn)

| Model | HF Accuracy | vLLM Accuracy | Delta |
|-------|------------|---------------|-------|
| 8B  | 86.95% | 87.31% | +0.4% |
| 14B | 84.64% | 80.40% | -4.2% |
| 38B | 86.73% | 82.34% | -4.4% |

## Key Finding

On single-image tasks (SROIE, WildReceipt), vLLM matches or exceeds HF accuracy. The ~4% accuracy drop appears only on bank statements (14B, 38B), which use multi-turn sequential extraction. This suggests the degradation is related to how vLLM handles multi-turn conversation context rather than paged attention (which is mathematically equivalent for single forward passes).
