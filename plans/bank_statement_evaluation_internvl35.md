# Plan: Bank Statement Extraction Evaluation — VLM Model Comparison

**Date**: 2026-03-31
**Hardware**: 2x L40S (48 GiB each, 96 GiB total)
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
| InternVL3.5-8B | Registry: `internvl3` | Downloaded, tested |
| InternVL3.5-14B | Registry: `internvl3-14b` | Downloaded, registered |
| InternVL3.5-38B | Registry: `internvl3-38b` | Downloaded, registered |
| Nemotron Nano 2 VL | Registry: `nemotron` | Downloaded, processor implemented |

**The pipeline is fully functional for all four models.**

---

## Environment Note

InternVL3.5 models and Nemotron require **different conda environments** due to incompatible transformers versions:

| Models | Conda Environment | transformers |
|--------|------------------|-------------|
| InternVL3.5-8B, 14B, 38B | `LMM_POC_IVL3.5` | 4.57 |
| Nemotron Nano 2 VL | `LMM_POC_NEMOTRON` | 4.53.x |

Switch environments between runs. The CLI commands are identical — only `conda activate` and `--model` change.

---

## Phase 1: Baseline Runs (All Four Models)

Run the existing pipeline on the 15-image bank statement dataset with each model.

### Commands

```bash
# ============================================================
# InternVL3.5 models (all three in one session)
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
# Nemotron (separate environment)
# ============================================================
conda activate LMM_POC_NEMOTRON

# --- Nemotron Nano 2 VL (~24 GB, fits single GPU) ---
python cli.py \
  --model nemotron \
  --data-dir evaluation_data/bank \
  --ground-truth evaluation_data/bank/ground_truth_bank.csv \
  --output-dir evaluation_data/output/bank_nemotron
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

After all four runs complete, compare results to answer:

### Key Questions

1. **Does model size help?** Compare 8B vs 14B vs 38B F1 across all fields. If 14B matches 38B, the extra 47 GB VRAM isn't justified.

2. **Does architecture matter?** Nemotron (hybrid Transformer-Mamba, #1 OCRBench v2) vs InternVL3.5 (pure Transformer). Nemotron may excel at text recognition but struggle with multi-turn extraction logic.

3. **Which field fails most?** Typically `LINE_ITEM_DESCRIPTIONS` and `TRANSACTION_AMOUNTS_PAID` are hardest (position-sensitive, many items per image).

4. **Which images fail?** Cross-reference per-image scores across models:
   - Do all models fail on the same images? (data problem or prompt problem)
   - Does the larger model rescue specific images? (capacity problem)

5. **Which strategy fails?** Check if `BALANCE_DESCRIPTION` outperforms `AMOUNT_DESCRIPTION` or vice versa. Does `TABLE_EXTRACTION` fallback ever fire?

6. **Does balance correction help?** Run the 8B model twice — once with `balance_correction: true`, once with `balance_correction: false` — and compare `TRANSACTION_AMOUNTS_PAID` F1.

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

| Metric | IVL3.5-8B | IVL3.5-14B | IVL3.5-38B | Nemotron 12B |
|--------|-----------|------------|------------|--------------|
| Overall F1 | ? | ? | ? | ? |
| DOCUMENT_TYPE F1 | ? | ? | ? | ? |
| STATEMENT_DATE_RANGE F1 | ? | ? | ? | ? |
| TRANSACTION_DATES F1 | ? | ? | ? | ? |
| LINE_ITEM_DESCRIPTIONS F1 | ? | ? | ? | ? |
| TRANSACTION_AMOUNTS_PAID F1 | ? | ? | ? | ? |
| Images/min | ? | ? | ? | ? |
| Peak VRAM (GB) | ? | ? | ? | ? |
| Fits single L40S? | Yes | Yes | No | Yes |
| Conda env | IVL3.5 | IVL3.5 | IVL3.5 | NEMOTRON |
| OCRBench v2 rank | #6 | #2 | — | #1 |

---

## Phase 5: Recommendations

Based on results, recommend:

1. **Which model to use** for bank statements (cost/accuracy tradeoff)
2. **Prompt improvements** if specific failure patterns emerge
3. **Whether to add max_tiles tuning** — bank statements are dense; more tiles may help larger models but hurt smaller ones
4. **Whether the 20B MoE model** (`InternVL3_5-GPT-OSS-20B-A4B-Preview`, already downloaded) is worth testing as a middle ground
5. **Whether Nemotron's OCR strength translates** to structured multi-turn extraction or only benefits flat-field tasks like SROIE

---

## Execution Order

1. Run IVL3.5-8B baseline (fastest, validates pipeline works) — ~5 min
2. Run IVL3.5-14B baseline — ~8 min
3. Run IVL3.5-38B baseline — ~15 min
4. Switch to `LMM_POC_NEMOTRON` env, run Nemotron baseline — ~8 min
5. Run 8B balance correction A/B (back in IVL3.5 env) — ~5 min
6. Compile results table
7. Failure analysis on worst images
8. Write findings to `plans/bank_statement_evaluation_results.md`

**Total estimated wall time**: ~55 min (including model load/unload and env switching)

---

## Prerequisites

- [x] InternVL3.5-8B downloaded
- [x] InternVL3.5-14B downloaded and registered
- [x] InternVL3.5-38B downloaded and registered
- [x] Nemotron Nano 2 VL downloaded and registered
- [x] Nemotron processor implemented (`models/document_aware_nemotron_processor.py`)
- [ ] InternVL3.5-14B downloaded to NFS (`hf download OpenGVLab/InternVL3_5-14B --local-dir /home/jovyan/nfs_share/models/InternVL3_5-14B`)
- [ ] Verify `evaluation_data/bank/` directory exists on remote with all 15 images
- [ ] Verify ground truth CSV paths match image filenames
- [ ] Confirm `LMM_POC_IVL3.5` conda env is active (transformers 4.57)
- [ ] Confirm `LMM_POC_NEMOTRON` conda env is active (transformers 4.53.x)
