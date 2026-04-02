# Plan: VLM Model Evaluation — Synthetic Datasets

**Date**: 2026-03-31 (updated 2026-04-02)
**Remote hardware**: 2x L40S (48 GiB each, 96 GiB total)
**Local machine**: macOS (no GPU) — analysis, reporting, visualization only
**Branch**: `feature/multi-gpu`

---

## Problem Statement

Bank statement extraction is the hardest task in the pipeline. Unlike receipts (4 flat fields, single-turn), bank statements require:
- Multi-turn inference (header detection + strategy-specific extraction)
- Column layout interpretation (Balance, Debit/Credit, signed Amount)
- Row-by-row alignment of dates, descriptions, and amounts
- Filtering debits from credits and non-transaction rows (Opening/Closing Balance)

We need to systematically evaluate how model architecture and size affect bank statement extraction accuracy, identify failure modes, and determine which model gives the best cost/accuracy tradeoff.

---

## What Already Exists

| Component | File | Status |
|-----------|------|--------|
| Ground truth | `evaluation_data/bank/ground_truth_bank.csv` | 15 images, 5 evaluated fields |
| V2 bank extractor | `common/unified_bank_extractor.py` | Multi-turn, 4 strategies |
| Bank adapter | `common/bank_statement_adapter.py` | Bridges extractor to pipeline |
| V2 prompts | `config/bank_prompts.yaml` | Header detection + per-strategy Turn 1 |
| Evaluation metrics | `common/evaluation_metrics.py` | Position-aware F1, fuzzy matching |
| CLI orchestration | `cli.py` | `--model`, `bank_v2`, `balance_correction` |
| InternVL3.5-8B (HF) | Registry: `internvl3` | Complete, results collected |
| InternVL3.5-8B (vLLM) | Registry: `internvl3-vllm` | Registered, **pending evaluation** |
| InternVL3.5-14B | Registry: `internvl3-14b` | Complete, results collected |
| InternVL3.5-38B | Registry: `internvl3-38b` | Complete, results collected |
| Nemotron Nano 2 VL | Registry: `nemotron` | Complete, results collected |
| Qwen3.5-27B | Registry: `qwen35` | Complete, results collected |
| Llama 4 Scout W4A16 | Registry: `llama4scout-w4a16` | Complete, results collected (vLLM) |
| vLLM processor | `models/document_aware_vllm_processor.py` | Shared by `llama4scout-w4a16` and `internvl3-vllm` |

**All six original models have completed bank and SROIE runs. The seventh model (`internvl3-vllm`) is registered and ready for evaluation.**

---

## Environment Note

InternVL3.5 models, Nemotron, Qwen3.5, and Scout require **different conda environments** due to incompatible transformers versions:

| Models | Conda Environment | transformers | Inference Engine |
|--------|------------------|-------------|-----------------|
| InternVL3.5-8B, 14B, 38B (HF) | `LMM_POC_IVL3.5` | 4.57 | HuggingFace AutoModel |
| InternVL3.5-8B (vLLM) | `LMM_POC_LLAMA4SCOUT` | ≥4.51 + vLLM | vLLM offline engine |
| Nemotron Nano 2 VL | `LMM_POC_NEMOTRON` | 4.53.x | HuggingFace AutoModelForCausalLM |
| Qwen3.5-27B | `LMM_POC_QWEN35` | git main (bleeding edge) | HuggingFace Qwen3_5ForConditionalGeneration |
| Llama 4 Scout W4A16 | `LMM_POC_LLAMA4SCOUT` | ≥4.51 + vLLM | vLLM offline engine |

Switch environments between runs. The CLI commands are identical — only `conda activate` and `--model` change.

**vLLM models note**: Both `llama4scout-w4a16` and `internvl3-vllm` use the vLLM offline inference engine with tensor parallelism (`DocumentAwareVllmProcessor`). The vLLM engine handles PagedAttention memory management, continuous batching, and CUDA graph optimizations. On production (4x A10G) where flash-attn compilation fails, set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` to use vLLM's built-in Triton attention. On sandbox (2x L40S), flash-attn works natively.

**InternVL3.5-8B vLLM note**: Uses the same model weights as `internvl3` (`InternVL3_5-8B`) but runs through vLLM instead of HuggingFace `AutoModel.chat()`. Shares the `LMM_POC_LLAMA4SCOUT` conda environment (which has vLLM installed). Uses `internvl3_prompts.yaml` (same prompts as HF path). The `model_type_key="internvl3"` parameter selects the InternVL3 generation config (2000 base tokens, 50 per field) instead of the Scout config.

**Qwen3.5-27B note**: Uses `Qwen3_5ForConditionalGeneration` (early-fusion VLM, NOT the Qwen3-VL architecture). Requires transformers installed from git main branch. ~54 GB BF16, needs cross-GPU sharding via `split_model` or `device_map="auto"` on 2x L40S.

**Llama 4 Scout W4A16 note**: [`RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`](https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16) — INT4 weights / FP16 activations, quantized with llm-compressor. Loads via vLLM's `LLM()` engine with `tensor_parallel_size` auto-detected from `CUDA_VISIBLE_DEVICES`. Uses MarlinLinearKernel for W4A16 dequantization. ~55 GB across GPUs.

---

## Phase 1: Baseline Runs (All Seven Models)

Run the existing pipeline on the 15-image bank statement dataset with each model.

### Commands

```bash
# ============================================================
# InternVL3.5 models — HuggingFace (all three in one session)
# NOTE: no --document-types flag (it filters by filename, not content)
# ============================================================
conda activate LMM_POC_IVL3.5

# --- InternVL3.5-8B (~16 GB, fits single GPU) ---
python cli.py \
  --model internvl3 \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_8b

# --- InternVL3.5-14B (~30 GB, fits single GPU) ---
python cli.py \
  --model internvl3-14b \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_14b

# --- InternVL3.5-38B (~77 GB, auto-shards across 2x L40S via split_model) ---
# CLI auto-detects requires_sharding and bypasses the multi-GPU orchestrator.
# The loader's split_model function distributes LLM layers across both GPUs.
python cli.py \
  --model internvl3-38b \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_38b

# ============================================================
# InternVL3.5-8B via vLLM (same weights, different inference engine)
# Uses LMM_POC_LLAMA4SCOUT env (has vLLM installed)
# ============================================================
conda activate LMM_POC_LLAMA4SCOUT

# --- InternVL3.5-8B vLLM (tensor parallel across 2x L40S) ---
python cli.py \
  --model internvl3-vllm \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_8b_vllm

# ============================================================
# Nemotron (separate environment)
# ============================================================
conda activate LMM_POC_NEMOTRON

# --- Nemotron Nano 2 VL (~24 GB, fits single GPU) ---
python cli.py \
  --model nemotron \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_nemotron

# ============================================================
# Qwen3.5-27B (separate environment — transformers git main)
# ============================================================
conda activate LMM_POC_QWEN35

# --- Qwen3.5-27B (~54 GB, auto-shards across 2x L40S) ---
python cli.py \
  --model qwen35 \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_qwen35_27b

# ============================================================
# Llama 4 Scout W4A16 (vLLM tensor parallel)
# ============================================================
conda activate LMM_POC_LLAMA4SCOUT

# --- Llama 4 Scout W4A16 (~55 GB, tensor parallel across 2x L40S) ---
python cli.py \
  --model llama4scout-w4a16 \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_llama4scout_w4a16
```

### Expected Output Per Run

- `model_results.csv` — per-image extraction results
- `evaluation_report.txt` — per-field F1 scores
- Console output — Rich tables with accuracy breakdown

### What to Record

For each model, capture:
1. **Overall F1** (mean of per-field F1 across all images)
2. **Per-field F1**: DOCUMENT_TYPE, STATEMENT_DATE_RANGE, TRANSACTION_DATES, LINE_ITEM_DESCRIPTIONS, TRANSACTION_AMOUNTS_PAID
3. **Per-image F1** — identifies which images are hardest
4. **Strategy distribution** — which extraction strategy was selected per image (Balance/Amount/DebitCredit/Fallback)
5. **Elapsed time** and images/min (throughput vs model size)
6. **GPU memory peak** per model

---

## Phase 2: Failure Analysis

After all seven runs complete, compare results to answer:

### Key Questions

1. **Does model size help?** Compare 8B vs 14B vs 38B F1 across all fields. If 14B matches 38B, the extra 47 GB VRAM isn't justified.

2. **Does architecture matter?** Five architectures to compare:
   - InternVL3.5 (pure Transformer, ViT + Qwen3 LLM)
   - Nemotron (hybrid Transformer-Mamba, #1 OCRBench v2)
   - Qwen3.5-27B (early-fusion VLM, 262K context, dense 27B)
   - Llama 4 Scout W4A16 (109B MoE, INT4 quantized, 16 experts)
   Nemotron may excel at text recognition but struggle with multi-turn extraction logic. Qwen3.5 may benefit from early fusion and long context for dense bank statements. Llama 4 Scout tests whether MoE architecture + INT4 quantization retains extraction quality.

3. **Does inference engine matter?** Compare InternVL3.5-8B HF vs InternVL3.5-8B vLLM on identical weights:
   - Same prompts, same images, same generation config
   - Any accuracy difference is purely from the inference engine (tokenization, sampling, attention implementation)
   - Throughput comparison (images/min) quantifies the vLLM speed advantage

4. **Which field fails most?** Typically `LINE_ITEM_DESCRIPTIONS` and `TRANSACTION_AMOUNTS_PAID` are hardest (position-sensitive, many items per image).

5. **Which images fail?** Cross-reference per-image scores across models:
   - Do all models fail on the same images? (data problem or prompt problem)
   - Does the larger model rescue specific images? (capacity problem)

6. **Which strategy fails?** Check if `BALANCE_DESCRIPTION` outperforms `AMOUNT_DESCRIPTION` or vice versa. Does `TABLE_EXTRACTION` fallback ever fire?

7. **Does balance correction help?** Run the 8B model twice — once with `balance_correction: true`, once with `balance_correction: false` — and compare `TRANSACTION_AMOUNTS_PAID` F1.

### Balance Correction A/B Test

```bash
conda activate LMM_POC_IVL3.5

# With balance correction (default)
python cli.py --model internvl3 \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_8b_balcorr_on

# Without balance correction
python cli.py --model internvl3 \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_ivl35_8b_balcorr_off \
  --no-balance-correction
```

---

## Phase 3: Error Catalogue

For each model's worst-performing images, manually inspect:

1. **Raw model output** (Turn 0 headers + Turn 1 extraction) — saved in verbose mode
2. **Parsed vs ground truth** — from `model_results.csv`
3. **Misalignment patterns**:
   - Shifted rows (off-by-one in position-aware F1)
   - Missed transactions (model skipped a row)
   - Extra transactions (model hallucinated or included credits)
   - Amount formatting (missing `$`, wrong decimal, sign confusion)
   - Date formatting (DD/MM vs MM/DD, missing year)
   - Description truncation or merging of multi-line descriptions

Create a table:

| Image | Model | Strategy | Issue | Field | Example |
|-------|-------|----------|-------|-------|---------|
| cba_date_grouped.png | 8B | BALANCE | Row shift | AMOUNTS | GT: $45.00, Pred: $67.50 |
| ... | ... | ... | ... | ... | ... |

---

## Phase 4: Results Comparison Table

Compile into a single markdown table for the report:

| Metric | IVL3.5-8B | IVL3.5-8B vLLM | IVL3.5-14B | IVL3.5-38B | Nemotron 12B | Qwen3.5-27B | L4Scout W4A16 |
|--------|-----------|----------------|------------|------------|--------------|-------------|---------------|
| Overall F1 | ? | ? | ? | ? | ? | ? | ? |
| DOCUMENT_TYPE F1 | ? | ? | ? | ? | ? | ? | ? |
| STATEMENT_DATE_RANGE F1 | ? | ? | ? | ? | ? | ? | ? |
| TRANSACTION_DATES F1 | ? | ? | ? | ? | ? | ? | ? |
| LINE_ITEM_DESCRIPTIONS F1 | ? | ? | ? | ? | ? | ? | ? |
| TRANSACTION_AMOUNTS_PAID F1 | ? | ? | ? | ? | ? | ? | ? |
| Images/min | ? | ? | ? | ? | ? | ? | ? |
| Peak VRAM (GB) | ? | ? | ? | ? | ? | ? | ? |
| Fits single L40S? | Yes | Yes | Yes | No | Yes | No | No |
| Conda env | IVL3.5 | LLAMA4SCOUT | IVL3.5 | IVL3.5 | NEMOTRON | QWEN35 | LLAMA4SCOUT |
| Inference engine | HF AutoModel | vLLM | HF AutoModel | HF AutoModel | HF AutoModel | HF AutoModel | vLLM |
| OCRBench v2 rank | #6 | #6 | #2 | — | #1 | — | — |
| Architecture | ViT+Qwen3 | ViT+Qwen3 | ViT+Qwen3 | ViT+Qwen3 | CRadio+Mamba | Early fusion | MoE 16E W4A16 |
| Parameters | 8B | 8B | 14B | 38B | 12B | 27B | 109B (17Bx16E) |

---

## Phase 5: SROIE Benchmark (All Seven Models)

Run the standardized SROIE receipt extraction benchmark (ICDAR 2019 Task 3) on all seven models. This provides a controlled comparison on a simpler task (4 flat fields, single-turn, exact-match evaluation) to isolate OCR/extraction capability from the multi-turn complexity of bank statements.

**SROIE fields**: company, date, address, total (exact-match after normalization)

### Commands

```bash
# ============================================================
# InternVL3.5 models — HuggingFace
# ============================================================
conda activate LMM_POC_IVL3.5

# --- InternVL3.5-8B ---
python benchmark_sroie.py \
  --model internvl3 \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_ivl35_8b

# --- InternVL3.5-14B ---
python benchmark_sroie.py \
  --model internvl3-14b \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_ivl35_14b

# --- InternVL3.5-38B (auto-shards across 2x L40S) ---
python benchmark_sroie.py \
  --model internvl3-38b \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_ivl35_38b

# ============================================================
# InternVL3.5-8B via vLLM
# ============================================================
conda activate LMM_POC_LLAMA4SCOUT

python benchmark_sroie.py \
  --model internvl3-vllm \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_ivl35_8b_vllm

# ============================================================
# Nemotron (separate environment)
# ============================================================
conda activate LMM_POC_NEMOTRON

python benchmark_sroie.py \
  --model nemotron \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_nemotron

# ============================================================
# Qwen3.5-27B (separate environment)
# ============================================================
conda activate LMM_POC_QWEN35

python benchmark_sroie.py \
  --model qwen35 \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_qwen35_27b

# ============================================================
# Llama 4 Scout W4A16 (vLLM tensor parallel)
# ============================================================
conda activate LMM_POC_LLAMA4SCOUT

python benchmark_sroie.py \
  --model llama4scout-w4a16 \
  --data-dir data/sroie \
  --output-dir evaluation_data/output/sroie_llama4scout_w4a16
```

### SROIE Results Table

| Metric | IVL3.5-8B | IVL3.5-8B vLLM | IVL3.5-14B | IVL3.5-38B | Nemotron 12B | Qwen3.5-27B | L4Scout W4A16 |
|--------|-----------|----------------|------------|------------|--------------|-------------|---------------|
| Overall F1 | ? | ? | ? | ? | ? | ? | ? |
| company F1 | ? | ? | ? | ? | ? | ? | ? |
| date F1 | ? | ? | ? | ? | ? | ? | ? |
| address F1 | ? | ? | ? | ? | ? | ? | ? |
| total F1 | ? | ? | ? | ? | ? | ? | ? |
| Images/min | ? | ? | ? | ? | ? | ? | ? |

### SROIE vs Bank Statement Analysis

Compare each model's SROIE rank vs bank statement rank to answer:
- Does SROIE performance predict bank statement performance?
- Do models that excel at flat-field OCR also handle multi-turn structured extraction?
- Is there a model that ranks differently between the two tasks (suggesting task-specific strengths)?
- Does the vLLM inference engine produce the same accuracy as HuggingFace on identical weights?

---

## Phase 6: Recommendations

Based on results, recommend:

1. **Which model to use** for bank statements (cost/accuracy tradeoff)
2. **Prompt improvements** if specific failure patterns emerge
3. **Whether to add max_tiles tuning** — bank statements are dense; more tiles may help larger models but hurt smaller ones
4. **Whether the 20B MoE model** (`InternVL3_5-GPT-OSS-20B-A4B-Preview`, already downloaded) is worth testing as a middle ground
5. **Whether Nemotron's OCR strength translates** to structured multi-turn extraction or only benefits flat-field tasks like SROIE
6. **Whether Qwen3.5-27B's early fusion and long context** give it an edge on dense bank statements vs InternVL3.5's ViT+LLM pipeline approach
7. **Whether INT4 quantization degrades extraction quality** — compare Llama 4 Scout W4A16 accuracy against dense models to assess quantization's impact on structured document tasks
8. **Whether bank prompts are overfitted to InternVL3.5** — prompts were developed and iterated against InternVL3.5, so other models may underperform due to prompt style rather than capability. Key diagnostic: if a model scores well on SROIE (generic prompts) but poorly on bank extraction (InternVL3.5-tuned prompts), prompt overfitting is likely. Remedies: per-model prompt files, simpler model-agnostic prompts, or relaxing output format constraints
9. **Whether vLLM should replace HuggingFace for InternVL3.5** — if `internvl3-vllm` matches `internvl3` accuracy with better throughput, vLLM becomes the preferred inference path (especially on production where flash-attn is unavailable)

---

## Execution Order

### Bank Statement Runs

1. ~~Run IVL3.5-8B bank baseline~~ — **DONE** (`bank_ivl35_8b`)
2. ~~Run IVL3.5-14B bank baseline~~ — **DONE** (`bank_ivl35_14b`)
3. ~~Run IVL3.5-38B bank baseline~~ — **DONE** (`bank_ivl35_38b`)
4. ~~Run Nemotron bank baseline~~ — **DONE** (`bank_nemotron`)
5. ~~Run Qwen3.5-27B bank baseline~~ — **DONE** (`bank_qwen35_27b`)
6. ~~Run Llama 4 Scout W4A16 bank baseline~~ — **DONE** (`bank_llama4scout_w4a16`)
7. ~~Run 8B balance correction A/B~~ — **DONE** (`bank_ivl35_8b_balcorr_on`, `bank_ivl35_8b_balcorr_off`)
8. Run IVL3.5-8B vLLM bank baseline — **PENDING** (`bank_ivl35_8b_vllm`)

### SROIE Runs

9. ~~Run IVL3.5-8B SROIE~~ — **DONE** (`sroie_ivl35_8b`)
10. ~~Run IVL3.5-14B SROIE~~ — **DONE** (`sroie_ivl35_14b`)
11. ~~Run IVL3.5-38B SROIE~~ — **DONE** (`sroie_ivl35_38b`)
12. ~~Run Nemotron SROIE~~ — **DONE** (`sroie_nemotron`)
13. ~~Run Qwen3.5-27B SROIE~~ — **DONE** (`sroie_qwen35_27b`)
14. ~~Run Llama 4 Scout W4A16 SROIE~~ — **DONE** (`sroie_llama4scout_w4a16`)
15. Run IVL3.5-8B vLLM SROIE — **PENDING** (`sroie_ivl35_8b_vllm`)

### Analysis

16. Compile bank statement results table
17. Compile SROIE results table
18. Cross-task comparison (SROIE rank vs bank rank)
19. Failure analysis on worst bank images
20. Write findings to `plans/model_evaluation_results.md`

**Remaining wall time**: ~20 min (two `internvl3-vllm` runs) + analysis

---

## Prerequisites

- [x] InternVL3.5-8B downloaded
- [x] InternVL3.5-14B downloaded and registered
- [x] InternVL3.5-38B downloaded and registered
- [x] Nemotron Nano 2 VL downloaded and registered
- [x] Nemotron processor implemented (`models/document_aware_nemotron_processor.py`)
- [x] Verify `evaluation_data/bank/` directory exists on remote with all 15 images
- [x] Verify ground truth CSV paths match image filenames
- [x] Confirm `LMM_POC_IVL3.5` conda env is active (transformers 4.57)
- [x] Confirm `LMM_POC_NEMOTRON` conda env is active (transformers 4.53.x)
- [x] SROIE data downloaded to `data/sroie/`

### Qwen3.5-27B Prerequisites

- [x] Download model to NFS
- [x] Create conda env `LMM_POC_QWEN35` (`config/QWEN35_env.yml`)
- [x] Implement `models/document_aware_qwen35_processor.py` (uses `Qwen3_5ForConditionalGeneration` + `AutoProcessor`)
- [x] Register `qwen35` in `models/registry.py` with `requires_sharding=True` (~54 GB BF16)
- [x] Add `qwen35` default path to `config/run_config.yml`
- [x] Reuses `internvl3_prompts.yaml` (no separate prompts file needed)

### Llama 4 Scout W4A16 Prerequisites

- [x] Download model to NFS (`Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`)
- [x] Create conda env `LMM_POC_LLAMA4SCOUT` (`config/scout_env.yml`):
  - Python 3.12, transformers ≥4.51, compressed-tensors, vLLM
  - Post-install: torch cu124, flash-attn, vllm
- [x] Register `llama4scout-w4a16` in `models/registry.py` — uses vLLM `LLM()` engine with tensor parallelism (not HuggingFace `from_pretrained`)
- [x] Implement `DocumentAwareVllmProcessor` (`models/document_aware_vllm_processor.py`)
- [x] Add `llama4scout-w4a16` default path to `config/run_config.yml`
- [x] Add vLLM dispatch in `benchmark_sroie.py`

### InternVL3.5-8B vLLM Prerequisites

- [x] Register `internvl3-vllm` in `models/registry.py` — reuses vLLM `LLM()` engine loader pattern, same weights as `internvl3`
- [x] Add `model_type_key` parameter to `DocumentAwareVllmProcessor` (selects InternVL3 generation config)
- [x] Add `internvl3-vllm` default path to `config/run_config.yml`
- [x] Add `internvl3-vllm` dispatch in `benchmark_sroie.py`
- [ ] Run bank statement evaluation on remote (`bank_ivl35_8b_vllm`)
- [ ] Run SROIE benchmark on remote (`sroie_ivl35_8b_vllm`)

---

## Local Analysis (macOS — No GPU)

All inference runs on the remote 2x L40S machine. Outputs are committed to `evaluation_data/output/` and pulled locally via git. The following analysis is performed entirely on the local machine using Claude Code.

### Data Flow

```
Remote (2x L40S)                          Local (macOS)
─────────────────                         ─────────────
Run cli.py / benchmark_sroie.py           git pull
  -> model_results.csv                    Read CSVs + evaluation reports
  -> evaluation_report.txt                Run analysis scripts
  -> console logs                         Generate comparison tables
git add + commit + push ──────────────>   Write findings report
```

### Analysis Tasks

#### 1. Bank Statement Results Aggregation

- Read `evaluation_data/output/bank_*/model_results.csv` for all seven models
- Read `evaluation_data/output/bank_*/evaluation_report.txt` for per-field F1 scores
- Populate the Phase 4 results comparison table with actual numbers
- Identify the best model per field and overall

#### 2. SROIE Results Aggregation

- Read `evaluation_data/output/sroie_*/` output files for all seven models
- Populate the Phase 5 SROIE results table
- Compute per-field exact-match F1 (company, date, address, total)

#### 3. Cross-Task Correlation

- Rank models by bank statement F1 and by SROIE F1
- Identify rank inversions (models that do well on one task but not the other)
- Determine whether OCR capability (SROIE) predicts structured extraction capability (bank)
- Statistical correlation between SROIE overall F1 and bank overall F1

#### 4. HF vs vLLM Comparison (InternVL3.5-8B)

- Compare `bank_ivl35_8b` (HF) vs `bank_ivl35_8b_vllm` (vLLM) — same weights, different engine
- Per-field F1 delta to detect any accuracy regression from the inference engine change
- Throughput comparison (images/min) to quantify the vLLM speed advantage
- If accuracy is equivalent and throughput is better, recommend vLLM as default for production

#### 5. Per-Image Failure Analysis

- For each model's bottom-3 images (by bank F1), extract:
  - Which strategy was selected
  - Which fields failed
  - Raw model output vs ground truth
- Cross-reference failures across models (same image fails everywhere = data/prompt issue)
- Build the Phase 3 error catalogue table

#### 6. Balance Correction Impact

- Compare `bank_ivl35_8b_balcorr_on` vs `bank_ivl35_8b_balcorr_off` results
- Compute delta in `TRANSACTION_AMOUNTS_PAID` F1
- Determine if balance correction is worth the extra inference pass

#### 7. Cost/Accuracy Tradeoff Analysis

- Plot (or tabulate) F1 vs model size, F1 vs throughput (images/min), F1 vs peak VRAM
- Identify the Pareto-optimal models (highest accuracy for given resource budget)
- Factor in conda environment complexity (operational cost of maintaining separate envs)
- Factor in vLLM's operational advantage (no flash-attn compilation, Triton fallback)

#### 8. Final Report

- Write `plans/model_evaluation_results.md` with:
  - Executive summary (recommended model + rationale)
  - Bank statement results table (filled)
  - SROIE results table (filled)
  - HF vs vLLM comparison (accuracy + throughput)
  - Cross-task correlation analysis
  - Error catalogue with specific failure examples
  - Cost/accuracy tradeoff recommendation
  - Next steps (prompt tuning, max_tiles experiments, 20B MoE testing, vLLM migration)
