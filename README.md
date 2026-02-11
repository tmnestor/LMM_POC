# InternVL3.5-8B Document Extraction Pipeline

A production-ready CLI for extracting structured fields from business document images using the InternVL3.5-8B vision-language model. Supports batched inference, automatic GPU memory management, and evaluation against ground truth.

## Project Structure

```
.
├── ivl3_cli.py                        # CLI entry point
├── config/
│   ├── run_config.yml                 # Single source of truth for all tunable parameters
│   ├── field_definitions.yaml         # Document types, fields, and evaluation settings
│   ├── bank_column_patterns.yaml      # Column header patterns for bank statement extraction
│   └── model_config.yaml              # Model configs for unified bank extraction
├── prompts/
│   ├── document_type_detection.yaml   # Detection prompts + type mappings
│   └── internvl3_prompts.yaml         # Per-type extraction prompts
├── common/
│   ├── __init__.py                    # Package exports (PipelineConfig, loaders, etc.)
│   ├── pipeline_config.py             # PipelineConfig dataclass, YAML/ENV/CLI merging
│   ├── config.py                      # Module-level constants (overridden by YAML)
│   ├── batch_processor.py             # Batch orchestration (detection → extraction)
│   ├── gpu_optimization.py            # CUDA memory management, OOM recovery
│   ├── robust_gpu_memory.py           # Reliable GPU memory detection across hardware
│   ├── bank_statement_adapter.py      # Bridges batch processor with multi-turn bank extraction
│   ├── bank_statement_calculator.py   # Derives transaction types and amounts from extracted data
│   ├── unified_bank_extractor.py      # Auto-selects optimal bank extraction strategy
│   ├── batch_analytics.py             # DataFrames and statistics from batch results
│   ├── batch_reporting.py             # Executive summaries and reports
│   ├── batch_visualizations.py        # Dashboards, heatmaps, and charts
│   ├── evaluation_metrics.py          # Ground truth comparison and accuracy calculation
│   ├── extraction_cleaner.py          # Normalisation and cleaning of extracted values
│   ├── extraction_parser.py           # Raw model output → structured data dicts
│   ├── field_definitions_loader.py    # Loads field definitions from YAML
│   ├── simple_model_evaluator.py      # Model performance comparison
│   └── simple_prompt_loader.py        # Prompt loading from YAML files
├── models/
│   ├── __init__.py                            # Package exports
│   ├── document_aware_internvl3_processor.py  # InternVL3 model wrapper
│   └── internvl3_image_preprocessor.py        # Image tiling and tensor preparation
└── environment_ivl35.yml              # Conda environment specification
```

## Quick Start

```bash
# 1. Create environment
conda env create -f environment_ivl35.yml
conda activate vision_notebooks

# 2. Run with config file (recommended)
python ivl3_cli.py --config config/run_config.yml

# 3. Or specify paths directly
python ivl3_cli.py -d ./images -o ./output

# 4. Evaluation mode (with ground truth)
python ivl3_cli.py -d ./images -o ./output -g ./ground_truth.csv
```

## Pipeline Flow

```
Document Images
      │
      ▼
  Detection ──► Classify document type (INVOICE, RECEIPT, BANK_STATEMENT, ...)
      │
      ▼
  Extraction ──► Type-specific prompt → structured field extraction
      │
      ▼
  Evaluation ──► Compare against ground truth CSV → F1 scores per field
      │
      ▼
  Reporting ──► CSV analytics, visualizations, markdown reports
```

**Batch processing**: Detection runs in batches (configurable). Standard documents extract in batches. Bank statements use sequential multi-turn extraction via `BankStatementAdapter`.

## CLI Reference

```
python ivl3_cli.py [OPTIONS]
```

### Data Options

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --data-dir` | *from YAML* | Directory containing document images |
| `-o, --output-dir` | *from YAML* | Output directory for results |
| `-g, --ground-truth` | `None` | Ground truth CSV (omit for inference-only) |
| `--max-images` | `None` (all) | Limit number of images to process |
| `--document-types` | `None` (all) | Filter by type, comma-separated: `INVOICE,RECEIPT` |

### Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model-path` | *auto-detected* | Path to InternVL3.5-8B model weights |
| `--max-tiles` | `18` | Image tiles (more = better OCR, more VRAM) |
| `--flash-attn / --no-flash-attn` | `true` | Flash Attention 2 (disable for V100) |
| `--dtype` | `bfloat16` | Torch dtype: `bfloat16`, `float16`, `float32` |

### Processing Options

| Flag | Default | Description |
|------|---------|-------------|
| `-b, --batch-size` | `None` (auto) | Images per batch (`null` = auto-detect from VRAM) |
| `--bank-v2 / --no-bank-v2` | `true` | Multi-turn bank statement extraction |
| `--balance-correction / --no-balance-correction` | `true` | Balance validation for bank statements |

### Output Options

| Flag | Default | Description |
|------|---------|-------------|
| `--no-viz / --viz` | `false` | Skip visualization generation |
| `--no-reports / --reports` | `false` | Skip report generation |
| `-v, --verbose / -q, --quiet` | `false` | Verbose debug output |
| `-V, --version` | | Show version and exit |

### Configuration

| Flag | Description |
|------|-------------|
| `-c, --config` | Path to YAML config file (defaults to `config/run_config.yml`) |

### Priority Order

CLI flags override YAML, which overrides environment variables, which override dataclass defaults:

```
CLI  >  YAML (run_config.yml)  >  ENV (IVL_*)  >  PipelineConfig defaults
```

`config/run_config.yml` is **always loaded** automatically. The `--config` flag overrides which YAML file is used.

### Environment Variables

All prefixed with `IVL_`:

```bash
IVL_DATA_DIR=/path/to/images
IVL_OUTPUT_DIR=/path/to/output
IVL_MODEL_PATH=/models/InternVL3_5-8B
IVL_BATCH_SIZE=4
IVL_MAX_TILES=14
IVL_FLASH_ATTN=false
IVL_DTYPE=float32
IVL_VERBOSE=true
```

## YAML Configuration

`config/run_config.yml` is the single source of truth for all tunable parameters. Every section is optional — missing keys fall back to Python defaults.

### `model` — Model identity

```yaml
model:
  path: /home/jovyan/nfs_share/models/InternVL3_5-8B
  max_tiles: 18          # Image tiles (H200: 18-36, V100: 14, L4: 18)
  flash_attn: true       # Flash Attention 2 (disable for V100)
  dtype: bfloat16        # bfloat16 | float16 | float32
  max_new_tokens: 2000   # Max generation tokens
```

### `data` — Input paths

```yaml
data:
  dir: ../evaluation_data/synthetic
  ground_truth: ../evaluation_data/synthetic/ground_truth_synthetic.csv
  max_images: null       # null = all
  document_types: null   # null = all, or: INVOICE,RECEIPT,BANK_STATEMENT
```

### `output` — Output paths and toggles

```yaml
output:
  dir: ../evaluation_data/output
  skip_visualizations: false
  skip_reports: false
```

### `processing` — Runtime behaviour

```yaml
processing:
  batch_size: null             # null = auto-detect from VRAM, 1 = sequential
  bank_v2: true                # Multi-turn bank statement extraction
  balance_correction: true     # Balance validation
  verbose: false
```

### `batch` — Batch processing tuning

Controls how images are grouped into batches for inference.

```yaml
batch:
  default_sizes: {internvl3: 4, internvl3-2b: 4, internvl3-8b: 4}
  max_sizes: {internvl3: 8, internvl3-2b: 8, internvl3-8b: 16}
  conservative_sizes: {internvl3: 1, internvl3-2b: 2, internvl3-8b: 1}
  min_size: 1
  strategy: balanced            # conservative | balanced | aggressive
  auto_detect: true             # Auto-select batch size from VRAM
  memory_safety_margin: 0.8     # Fraction of VRAM to use
  clear_cache_after_batch: true
  timeout_seconds: 300
  fallback_enabled: true
  fallback_steps: [8, 4, 2, 1] # OOM fallback: halve until success
```

**Strategies**: `conservative` uses minimum safe sizes, `balanced` uses defaults, `aggressive` uses maximum sizes. When `auto_detect` is enabled, VRAM is measured and the strategy is selected automatically based on `gpu.memory_thresholds`.

### `generation` — Token generation parameters

```yaml
generation:
  max_new_tokens_base: 2000     # Base token budget
  max_new_tokens_per_field: 50  # Extra tokens per extraction field
  do_sample: false              # Greedy decoding (deterministic)
  use_cache: true               # KV cache (required for quality)
  num_beams: 1                  # No beam search
  repetition_penalty: 1.0
  token_limits:
    2b: null                    # Use field-count calculation
    8b: 800                     # Hard cap for 8B model
```

### `gpu` — GPU memory management

```yaml
gpu:
  memory_thresholds:
    low: 8                      # GB — triggers conservative batching
    medium: 16                  # GB — triggers balanced batching
    high: 24                    # GB — triggers aggressive batching
    very_high: 64               # GB — triggers maximum batching
  cuda_max_split_size_mb: 128   # CUDA allocator block size
  fragmentation_threshold_gb: 0.5
  critical_fragmentation_threshold_gb: 1.0
  max_oom_retries: 3
  cudnn_benchmark: true
```

### `model_loading` — Model loading options

```yaml
model_loading:
  trust_remote_code: true       # Required for InternVL3 custom code
  use_fast_tokenizer: false     # Slow tokenizer for compatibility
  low_cpu_mem_usage: true       # Reduce CPU RAM during loading
  device_map: auto              # Automatic multi-GPU distribution
  default_paths:                # Auto-detection search order
    - /home/jovyan/nfs_share/models/InternVL3_5-8B
    - /models/InternVL3_5-8B
    - ./models/InternVL3_5-8B
```

### Hardware Presets

**L4 / A10G (24 GB VRAM)**:
```yaml
model:
  max_tiles: 18
  flash_attn: true
  dtype: bfloat16
batch:
  strategy: balanced
```

**L40S (48 GB VRAM)**:
```yaml
model:
  max_tiles: 24
  flash_attn: true
  dtype: bfloat16
batch:
  strategy: aggressive
```

## Configuring a New Document Type

Adding a new document type requires changes to **3 YAML files** — no Python code changes needed. Here's a worked example adding a **purchase order** document type.

### Step 1: Register the document type and its fields

**File**: `config/field_definitions.yaml`

Add the new type under `document_fields`, list it in `supported_document_types`, and add aliases:

```yaml
document_fields:
  # ... existing types ...

  purchase_order:
    count: 10
    fields:
      - DOCUMENT_TYPE
      - BUSINESS_ABN
      - SUPPLIER_NAME
      - BUSINESS_ADDRESS
      - PAYER_NAME
      - PAYER_ADDRESS
      - INVOICE_DATE          # PO date (reuse existing field)
      - LINE_ITEM_DESCRIPTIONS
      - LINE_ITEM_QUANTITIES
      - TOTAL_AMOUNT

  universal:
    count: 35                  # Update count to include any new fields
    fields:
      # ... add any NEW fields here (existing ones are already listed) ...
```

Add to the supported types list:

```yaml
supported_document_types:
  - invoice
  - receipt
  - bank_statement
  - travel_expense
  - vehicle_logbook
  - purchase_order              # <-- add
```

Add aliases so detection can map variations to the canonical name:

```yaml
document_type_aliases:
  # ... existing aliases ...

  purchase_order:
    - purchase order
    - po
    - purchase requisition
    - procurement order
```

If your new type introduces fields that aren't already in the universal set, add their descriptions:

```yaml
field_descriptions:
  # ... existing descriptions ...
  # (purchase_order reuses existing fields, so nothing to add here)
```

And classify them in `field_categories` and `evaluation.field_types` as appropriate.

### Step 2: Add detection support

**File**: `prompts/document_type_detection.yaml`

Add the new type to the detection prompt options:

```yaml
prompts:
  detection:
    prompt: |
      What type of business document is this?

      Answer with one of:
      - INVOICE
      - RECEIPT
      - BANK_STATEMENT
      - TRAVEL_EXPENSE
      - VEHICLE_LOGBOOK
      - PURCHASE_ORDER              # <-- add
```

Add type mappings so model responses get normalised:

```yaml
type_mappings:
  # ... existing mappings ...
  "purchase order": "PURCHASE_ORDER"
  "po": "PURCHASE_ORDER"
  "procurement order": "PURCHASE_ORDER"
```

Add fallback keywords (checked when type mappings don't match):

```yaml
fallback_keywords:
  # ... existing keywords ...
  # Put before INVOICE since POs can contain "invoice"-like language
  PURCHASE_ORDER:
    - purchase order
    - procurement
    - requisition
```

### Step 3: Write the extraction prompt

**File**: `prompts/internvl3_prompts.yaml`

Add an extraction prompt keyed to the document type name. The key must match the name in `field_definitions.yaml`:

```yaml
prompts:
  # ... existing prompts ...

  purchase_order:
    name: "Purchase Order Extraction"
    description: "Extract 10 purchase order fields"
    prompt: |
      Extract ALL data from this purchase order image.
      Respond in exact format below with actual values or NOT_FOUND.

      DOCUMENT_TYPE: PURCHASE_ORDER
      BUSINESS_ABN: NOT_FOUND
      SUPPLIER_NAME: NOT_FOUND
      BUSINESS_ADDRESS: NOT_FOUND
      PAYER_NAME: NOT_FOUND
      PAYER_ADDRESS: NOT_FOUND
      INVOICE_DATE: NOT_FOUND
      LINE_ITEM_DESCRIPTIONS: NOT_FOUND
      LINE_ITEM_QUANTITIES: NOT_FOUND
      TOTAL_AMOUNT: NOT_FOUND

      Instructions:
      - Find ABN: 11 digits like "12 345 678 901"
      - Find supplier: Company name at top
      - Find purchaser: "Ship To" or "Deliver To" section
      - Find date: Use DD/MM/YYYY format
      - Find line items: List with " | " separator
      - Find amounts: Include $ symbol
      - Replace NOT_FOUND with actual values
```

### Step 4: Prepare evaluation data (optional)

To evaluate extraction accuracy, create a ground truth CSV with columns matching your field names:

```
filename,DOCUMENT_TYPE,BUSINESS_ABN,SUPPLIER_NAME,...,TOTAL_AMOUNT
po_001.png,PURCHASE_ORDER,12 345 678 901,Acme Corp,...,$5000.00
```

Then run:

```bash
python ivl3_cli.py -d ./purchase_orders -o ./output -g ./ground_truth_po.csv
```

### Checklist

| File | What to add |
|------|-------------|
| `config/field_definitions.yaml` | `document_fields.<type>`, `supported_document_types`, `document_type_aliases`, field descriptions/categories |
| `prompts/document_type_detection.yaml` | Detection prompt options, `type_mappings`, `fallback_keywords` |
| `prompts/internvl3_prompts.yaml` | Extraction prompt with field template |
| Ground truth CSV *(optional)* | One row per image with expected field values |

The pipeline automatically discovers new document types from these YAML files — the prompt key in `internvl3_prompts.yaml` is matched against `supported_document_types` in `field_definitions.yaml` to build the extraction routing table. No Python code changes required.

### Layout Variants

If your document type has distinct visual layouts (like bank statements with flat vs. date-grouped formats), you can create layout-specific prompts by adding a suffix:

```yaml
# prompts/internvl3_prompts.yaml
prompts:
  purchase_order_domestic:
    prompt: |
      # Prompt optimised for domestic POs ...

  purchase_order_international:
    prompt: |
      # Prompt optimised for international POs with customs fields ...
```

Register the suffixes in the settings section:

```yaml
settings:
  structure_suffixes: ["_flat", "_date_grouped", "_domestic", "_international"]
```

The pipeline strips these suffixes to map back to the base `PURCHASE_ORDER` type for field validation and evaluation.
