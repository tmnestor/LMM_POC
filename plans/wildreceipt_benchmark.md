# Plan: WildReceipt Benchmark

## Goal

Benchmark all registered models on the WildReceipt dataset (472 test images, 25
entity classes) to evaluate receipt entity extraction quality. This complements the
existing SROIE benchmark (4 fields) and bank statement evaluation (5 fields) with a
more challenging, fine-grained entity recognition task.

## Why WildReceipt

| Property | SROIE | WildReceipt | Bank Statements |
|----------|-------|-------------|-----------------|
| Images | 347 test | 472 test | 9 test |
| Entity classes | 4 | 25 (key/value pairs) | 5 |
| Source | Scanned | In-the-wild photos | Scanned |
| Complexity | Low | Medium | High (multi-turn) |
| Diversity | Single country | Multi-country | Single format |

WildReceipt tests **OCR robustness** (varied lighting, angles, occlusion) and
**entity disambiguation** (25 classes vs SROIE's 4). In-the-wild photos are
harder than scans.

---

## Dataset Structure

Downloaded via `python scripts/download_wildreceipt.py` to `../data/wildreceipt/`:

```
../data/wildreceipt/
├── image_files/          # 1,765 JPEG receipt images
├── train.txt             # 1,270 annotations (JSON-lines)
├── test.txt              # 472 annotations (JSON-lines)
└── class_list.txt        # 26 classes (0-25)
```

### Annotation Format (JSON-lines in train.txt / test.txt)

Each line is a JSON object:
```json
{
  "file_name": "image_files/Image_12/34/abc123.jpeg",
  "height": 348,
  "width": 348,
  "annotations": [
    {"box": [x1,y1,x2,y2,x3,y3,x4,y4], "text": "STORE NAME", "label": 1},
    {"box": [...], "text": "$12.99", "label": 15},
    ...
  ]
}
```

### 26 Entity Classes (class_list.txt)

| ID | Label | ID | Label |
|----|-------|----|-------|
| 0 | Ignore | 13 | Prod_quantity_value |
| 1 | Store_name_value | 14 | Prod_quantity_key |
| 2 | Store_name_key | 15 | Prod_price_value |
| 3 | Store_addr_value | 16 | Prod_price_key |
| 4 | Store_addr_key | 17 | Subtotal_value |
| 5 | Tel_value | 18 | Subtotal_key |
| 6 | Tel_key | 19 | Tax_value |
| 7 | Date_value | 20 | Tax_key |
| 8 | Date_key | 21 | Tips_value |
| 9 | Time_value | 22 | Tips_key |
| 10 | Time_key | 23 | Total_value |
| 11 | Prod_item_value | 24 | Total_key |
| 12 | Prod_item_key | 25 | Others |

**Key insight**: Classes come in key/value pairs (e.g., `Store_name_key` = the
label "Store:", `Store_name_value` = "SAFEWAY"). For VLM extraction we primarily
care about the **value** classes — the model should extract the values, not the
field labels.

### Value Classes (12 extraction targets)

These are the only classes we evaluate — key classes, Ignore (0), and Others (25)
are excluded from scoring.

| # | Group | Value class | Example | SROIE equivalent |
|---|-------|-------------|---------|------------------|
| 1 | Store | Store_name_value | "SAFEWAY" | `company` |
| 2 | Store | Store_addr_value | "123 Main St, VA 22030" | `address` |
| 3 | Contact | Tel_value | "703-777-5833" | -- |
| 4 | Date/Time | Date_value | "12/25/2023" | `date` |
| 5 | Date/Time | Time_value | "14:30" | -- |
| 6 | Line items | Prod_item_value | "MILK 2%" | -- |
| 7 | Line items | Prod_quantity_value | "2" | -- |
| 8 | Line items | Prod_price_value | "$3.99" | -- |
| 9 | Totals | Subtotal_value | "$45.67" | -- |
| 10 | Totals | Tax_value | "$3.20" | -- |
| 11 | Totals | Tips_value | "$5.00" | -- |
| 12 | Totals | Total_value | "$53.87" | `total` |

4 of the 12 classes overlap directly with SROIE's fields (company, address, date,
total), giving a direct cross-benchmark comparison on those fields. The remaining 8
classes (telephone, time, line items, subtotal, tax, tips) are unique to
WildReceipt and test finer-grained extraction capability.

### Excluded Classes

| ID | Class | Reason |
|----|-------|--------|
| 0 | Ignore | Background / non-entity text |
| 2 | Store_name_key | Field label ("Store:"), not data |
| 4 | Store_addr_key | Field label |
| 6 | Tel_key | Field label |
| 8 | Date_key | Field label |
| 10 | Time_key | Field label |
| 12 | Prod_item_key | Field label |
| 14 | Prod_quantity_key | Field label |
| 16 | Prod_price_key | Field label |
| 18 | Subtotal_key | Field label |
| 20 | Tax_key | Field label |
| 22 | Tips_key | Field label |
| 24 | Total_key | Field label |
| 25 | Others | Catch-all noise |

---

## Design Decisions

### 1. Evaluation Split

**Use test split only (472 images)** — matches SROIE approach. The train split is
available for few-shot prompt engineering if needed later.

### 2. Extraction Approach

**Single-turn unified prompt** — ask the model to extract all entity types at once
in structured JSON. This matches the SROIE benchmark's single-turn approach and
keeps inference cost manageable (472 images x 1 turn vs 472 x 25 turns).

### 3. Evaluation Metric

**Entity-level micro-average F1 with position-aware matching**, consistent with
the LMM_POC evaluation methodology.

#### Aggregation: Micro-Average F1

- For each value class, collect ground truth entities from annotations
- Match predicted entities to ground truth (method depends on class type — see below)
- Pool all TP / FP / FN across all 12 value classes into global counts
- Overall precision = total TP / (total TP + total FP)
- Overall recall = total TP / (total TP + total FN)
- Overall F1 = harmonic mean of precision and recall
- Also report **per-class F1 breakdown** alongside the overall score to identify
  which entity classes are weakest

#### GT Reading Order

WildReceipt annotations are bounding-box regions, not ordered text. Before
evaluation, annotations are sorted into **reading order** by bbox centroid
`(y, x)` — top-to-bottom, left-to-right. This ensures list fields (product
items) align positionally with the model's visual reading order.

#### NOT_FOUND Handling

Consistent with the LMM_POC evaluation methodology, the model is instructed to
return `"NOT_FOUND"` for fields not present on the receipt (rather than omitting
them).  When **both GT and prediction are absent** for a class, this counts as
a **TP** — the model correctly identified the field's absence.

| GT | Prediction | Outcome |
|----|------------|---------|
| absent | absent / `NOT_FOUND` | **TP** (correct absence) |
| absent | has value | **FP** (hallucination) |
| has value | absent / `NOT_FOUND` | **FN** (missed field) |
| has value | has value | match → **TP**, mismatch → **FP + FN** |

Sentinel values treated as absent: `NOT_FOUND`, `N/A`, `None`, `null`, empty.

#### Scalar Classes (9 fields)

For single-value fields (store_name, store_address, telephone, date, time,
subtotal, tax, tips, total), GT annotations are **concatenated** into one string
(some store names/addresses are split across multiple bounding boxes) and
compared against the model's single predicted value.

| Outcome | Count |
|---------|-------|
| Both absent (correct NOT_FOUND) | TP = 1 |
| Both present + match | TP = 1 |
| Both present + mismatch | FP = 1, FN = 1 |
| GT only (model missed) | FN = 1 |
| Pred only (hallucination) | FP = 1 |

#### List Classes (3 fields) — Position-Aware F1

For multi-value fields (prod_item, prod_quantity, prod_price), evaluation is
**position-aware**: pred[0] vs GT[0], pred[1] vs GT[1], etc.

| Position | Outcome | Count |
|----------|---------|-------|
| Both present + match | TP += 1 |
| Both present + mismatch | FN += 1 |
| GT only (under-extracted) | FN += 1 |
| Pred only (over-extracted) | FP += 1 |

This penalises ordering errors — if the model extracts correct items in the
wrong order, those positions count as misses.

#### Matching Primitives (by class type)

| Class type | Matcher | Threshold | Example |
|------------|---------|-----------|---------|
| **Text** (store_name, store_address, time) | ANLS (Levenshtein) | similarity >= 0.5 | `"safeway inc"` vs `"safeway in"` → 0.91 → match |
| **Currency** (total, subtotal, tax, tips, prod_price) | Exact after 2dp normalisation | `==` | `"$3.99"` → `"3.99"` vs `"3.99"` → match |
| **Phone** (telephone) | Exact digits-only | `==` | `"703-777-5833"` → `"7037775833"` |
| **Date** (date) | Exact after DD/MM/YYYY normalisation | `==` | `"2023-12-25"` → `"25/12/2023"` |
| **Quantity** (prod_quantity) | Exact digits-only | `==` | `"x2"` → `"2"` |
| **Item text** (prod_item) | Word-overlap Jaccard | overlap >= 0.75 | `"milk 2%"` vs `"milk 2% gallon"` → 0.67 → no match |

#### Normalisation Rules

Applied **before** matching, per class type:

- **Text**: lowercase, collapse whitespace, strip leading/trailing punctuation
- **Currency**: strip `$£€¥RM`, remove commas, format to 2 decimal places
- **Phone**: strip all non-digit characters
- **Date**: parse to DD/MM/YYYY (handles ISO, slash, dash, dot separators)
- **Quantity**: strip all non-digit characters

#### Worked Example

Image with GT annotations (after bbox sorting):

| Class | GT values |
|-------|-----------|
| store_name | `["QUICK", "MART"]` → concatenated: `"QUICK MART"` |
| prod_item | `["MILK 2%", "BREAD WW", "EGGS LG"]` |
| prod_price | `["$3.99", "$2.49", "$4.29"]` |
| total | `["$10.77"]` |

Model predicts:
```json
{
  "store_name": "Quick Mart",
  "items": [
    {"name": "MILK 2%", "price": "$3.99"},
    {"name": "WHOLE WHEAT BREAD", "price": "$2.49"},
    {"name": "EGGS LARGE", "price": "$4.29"}
  ],
  "total": "10.77"
}
```

**store_name** (scalar, ANLS): `"quick mart"` vs `"quick mart"` → 1.0 → **TP**

**prod_item** (list, position-aware, Jaccard):
- [0] `"milk 2%"` vs `"milk 2%"` → Jaccard 1.0 → **TP**
- [1] `"whole wheat bread"` vs `"bread ww"` → words `{whole,wheat,bread}` ∩
  `{bread,ww}` = `{bread}`, union size 4 → Jaccard 0.25 → **FN**
- [2] `"eggs large"` vs `"eggs lg"` → `{eggs,large}` ∩ `{eggs,lg}` = `{eggs}`,
  union 3 → Jaccard 0.33 → **FN**

**prod_price** (list, position-aware, exact 2dp):
- [0] `"3.99"` vs `"3.99"` → **TP**
- [1] `"2.49"` vs `"2.49"` → **TP**
- [2] `"4.29"` vs `"4.29"` → **TP**

**total** (scalar, exact 2dp): `"10.77"` vs `"10.77"` → **TP**

**Totals**: TP=6, FP=0, FN=2 → Precision=1.0, Recall=0.75, F1=0.857

### 4. Handling Key vs Value Classes

**Extract value classes only** — the model extracts the actual data (store name,
total amount), not the field labels. Key classes (Store_name_key, Date_key, etc.)
are ignored during evaluation. The `Ignore` (0) and `Others` (25) classes are also
excluded from scoring.

### 5. Prompt Design

Single unified prompt requesting structured JSON output:

```
Extract all information from this receipt image. Return a JSON object with these fields:
{
  "store_name": "...",
  "store_address": "...",
  "telephone": "...",
  "date": "...",
  "time": "...",
  "items": [{"name": "...", "quantity": "...", "price": "..."}],
  "subtotal": "...",
  "tax": "...",
  "tips": "...",
  "total": "..."
}
Rules:
- Include ALL fields listed above in every response.
- If a field is not visible on the receipt, set its value to "NOT_FOUND".
- For items, list each product as a separate entry. Use an empty list [] if no line items are visible.
```

This maps cleanly to WildReceipt's value classes:
- `store_name` -> Store_name_value
- `store_address` -> Store_addr_value
- `telephone` -> Tel_value
- `date` -> Date_value
- `time` -> Time_value
- `items[].name` -> Prod_item_value
- `items[].quantity` -> Prod_quantity_value
- `items[].price` -> Prod_price_value
- `subtotal` -> Subtotal_value
- `tax` -> Tax_value
- `tips` -> Tips_value
- `total` -> Total_value

---

## Models to Benchmark

All 9 registered models, matching the bank/SROIE evaluation matrix:

| # | Model | Type | Environment |
|---|-------|------|-------------|
| 1 | `internvl3` | HF 8B | `LMM_POC_IVL3.5` |
| 2 | `internvl3-14b` | HF 14B | `LMM_POC_IVL3.5` |
| 3 | `internvl3-38b` | HF 38B | `LMM_POC_IVL3.5` |
| 4 | `internvl3-vllm` | vLLM 8B | `LMM_POC_VLLM` |
| 5 | `internvl3-14b-vllm` | vLLM 14B | `LMM_POC_VLLM` |
| 6 | `internvl3-38b-vllm` | vLLM 38B | `LMM_POC_VLLM` |
| 7 | `nemotron` | HF | `LMM_POC_NEMOTRON` |
| 8 | `qwen35` | HF 27B | `LMM_POC_QWEN35` |
| 9 | `llama4scout-w4a16` | vLLM W4A16 | `LMM_POC_VLLM` |

---

## Implementation

### Files to Create

| File | Purpose |
|------|---------|
| `benchmark_wildreceipt.py` | Main benchmark script (modelled on `benchmark_sroie.py`) |

### Files Unchanged

| File | Why |
|------|-----|
| `models/registry.py` | Registry-driven — no model changes needed |
| `cli.py` | Benchmark is standalone, not part of the main pipeline |
| `benchmark_sroie.py` | Separate benchmark, untouched |

### `benchmark_wildreceipt.py` Structure

```python
# --- Data loading ---
def load_wildreceipt_ground_truth(test_txt: Path) -> dict[str, WildReceiptGT]:
    """Parse test.txt JSON-lines into ground truth dict keyed by image path."""

# --- Inference dispatch ---
def run_inference(model_type, model, processor, image, prompt, max_tokens) -> str:
    """Reuse the same per-model dispatch as benchmark_sroie.py."""

# --- Response parsing ---
def parse_wildreceipt_response(raw: str) -> dict:
    """Extract JSON from model response, handle markdown fences."""

# --- Evaluation ---
def evaluate_image(predicted: dict, ground_truth: WildReceiptGT) -> ImageResult:
    """Compare predicted fields to GT entities. Return per-class TP/FP/FN."""

def compute_metrics(results: list[ImageResult]) -> BenchmarkResult:
    """Aggregate per-class and overall F1 from image results."""

# --- Main benchmark ---
def run_benchmark(model_type, data_dir, ...) -> BenchmarkResult:
    """Load model via registry, iterate test images, run inference + eval."""

# --- CLI ---
def main():
    """Typer CLI: --model, --data-dir, --output-dir, --max-images, --max-tokens"""
```

### Output Structure

```
evaluation_data/output/wildreceipt_<model>/
├── wildreceipt_results.json       # Full results (per-class, per-image)
├── wildreceipt_summary.csv        # One row: model, overall_f1, per-class F1s
└── wildreceipt_per_image.csv      # One row per image with match details
```

---

## Execution Commands

```bash
# --- HuggingFace models (LMM_POC_IVL3.5 env) ---
conda activate LMM_POC_IVL3.5

python benchmark_wildreceipt.py \
  --model internvl3 --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_ivl35_8b

python benchmark_wildreceipt.py \
  --model internvl3-14b --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_ivl35_14b

python benchmark_wildreceipt.py \
  --model internvl3-38b --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_ivl35_38b

# --- vLLM models (LMM_POC_VLLM env) ---
conda activate LMM_POC_VLLM

VLLM_LOGGING_LEVEL=WARNING python benchmark_wildreceipt.py \
  --model internvl3-vllm --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_ivl35_8b_vllm

VLLM_LOGGING_LEVEL=WARNING python benchmark_wildreceipt.py \
  --model internvl3-14b-vllm --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_ivl35_14b_vllm

VLLM_LOGGING_LEVEL=WARNING python benchmark_wildreceipt.py \
  --model internvl3-38b-vllm --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_ivl35_38b_vllm

VLLM_LOGGING_LEVEL=WARNING python benchmark_wildreceipt.py \
  --model llama4scout-w4a16 --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_llama4scout

# --- Other HF models ---
conda activate LMM_POC_NEMOTRON
python benchmark_wildreceipt.py \
  --model nemotron --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_nemotron

conda activate LMM_POC_QWEN35
python benchmark_wildreceipt.py \
  --model qwen35 --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_qwen35
```

---

## Usage Examples

### Quick smoke test (5 images)

```bash
python benchmark_wildreceipt.py \
  --model internvl3-vllm -n 5 --data-dir ../data/wildreceipt
```

### Compare two models side-by-side

```bash
python benchmark_wildreceipt.py \
  --model internvl3-vllm --model internvl3-14b-vllm \
  --data-dir ../data/wildreceipt \
  --output-dir evaluation_data/output/wildreceipt_comparison
```

### Override model path

```bash
python benchmark_wildreceipt.py \
  --model internvl3 \
  --model-path /home/jovyan/nfs_share/models/InternVL3_5-8B \
  --data-dir ../data/wildreceipt
```

### Increase generation budget for verbose receipts

```bash
python benchmark_wildreceipt.py \
  --model internvl3-vllm --max-tokens 2048 \
  --data-dir ../data/wildreceipt
```

### CLI reference

```
python benchmark_wildreceipt.py --help

Options:
  -m, --model TEXT        Model type(s) to benchmark (repeatable)
  -d, --data-dir PATH     Path to WildReceipt data directory [../data/wildreceipt]
  -p, --model-path TEXT   Override model path (auto-detected if omitted)
  -n, --max-images INT    Maximum images to evaluate (all if omitted)
  -o, --output-dir PATH   Directory for results output [output/wildreceipt]
  --max-tokens INT        Maximum generation tokens per image [1024]
  --save-responses        Save raw model responses to JSONL in output dir
```

### Output files

Each run writes three files to `--output-dir`:

| File | Content |
|------|---------|
| `wildreceipt_results.json` | Full results: per-class TP/FP/FN, per-image detail |
| `wildreceipt_summary.csv` | One row per model: overall + per-class P/R/F1 |
| `wildreceipt_per_image.csv` | One row per model+image: per-class TP/FP/FN counts |
| `responses_<model>.jsonl` | Raw model responses (only with `--save-responses`) |

---

## Execution Order

Run fast models first to validate the benchmark, then scale up:

1. `internvl3-vllm` (8B, fast) — validate benchmark works end-to-end
2. `internvl3` (8B, HF) — compare HF vs vLLM accuracy parity
3. Remaining models in parallel where env allows

---

## Analysis

After all runs complete:

1. **Per-class accuracy heatmap** — which entity classes are hardest?
2. **Model ranking** — overall F1 across all 9 models
3. **HF vs vLLM parity** — do vLLM models match HF accuracy?
4. **Cross-benchmark comparison** — rank correlation with SROIE and bank results
5. **Error analysis** — common failure modes (OCR errors, entity confusion,
   hallucination)
6. **Class difficulty ranking** — which of the 12 value classes are most/least
   reliably extracted?
