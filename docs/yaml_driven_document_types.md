# YAML-Driven Document Types: How We Add New Document Types Without Code Changes

## The Problem

We extract structured fields from business documents — invoices, receipts, bank statements, travel expenses, vehicle logbooks. Each document type has different fields, different detection cues, and different extraction prompts. When the business needs a new document type (say, purchase orders), we don't want to touch Python code. We want a data scientist or prompt engineer to be able to add it by editing configuration files alone.

---

## The Solution: Three YAML Files

Adding a new document type requires changes to exactly **3 YAML files**. No Python code changes. No redeployment of the pipeline.

```
config/field_definitions.yaml       ← What fields to extract (and how to evaluate them)
prompts/document_type_detection.yaml ← How to recognise the document
prompts/internvl3_prompts.yaml       ← How to extract fields (one per model)
prompts/llama_prompts.yaml
```

The pipeline auto-discovers document types at startup by cross-referencing these files. Here's how each one works.

---

## File 1: `config/field_definitions.yaml` — The Schema

This file is the single source of truth for what fields each document type contains. It drives field validation, token budget calculation, and evaluation metrics.

```yaml
document_fields:
  invoice:
    count: 14
    fields:
      - DOCUMENT_TYPE
      - BUSINESS_ABN
      - SUPPLIER_NAME
      - INVOICE_DATE
      - LINE_ITEM_DESCRIPTIONS
      - GST_AMOUNT
      - TOTAL_AMOUNT
      # ... (14 fields total)

  bank_statement:
    count: 5
    min_tokens: 1500          # Override: bank statements need more output tokens
    fields:
      - DOCUMENT_TYPE
      - STATEMENT_DATE_RANGE
      - LINE_ITEM_DESCRIPTIONS
      - TRANSACTION_DATES
      - TRANSACTION_AMOUNTS_PAID

  travel_expense:
    count: 9
    fields:
      - DOCUMENT_TYPE
      - PASSENGER_NAME
      - TRAVEL_MODE
      - TRAVEL_ROUTE
      # ...
```

### What the pipeline reads from this file

| Section | Used by | Purpose |
|---|---|---|
| `document_fields.<type>.fields` | Processors, batch_processor | Which fields to extract and validate per type |
| `document_fields.<type>.count` | Model config | Token budget calculation (`count * tokens_per_field`) |
| `document_fields.<type>.min_tokens` | Model config | Override minimum token budget (bank statements) |
| `supported_document_types` | CLI startup | Validate which prompt keys are real document types |
| `document_type_aliases` | Detection parsing | Map variations ("tax invoice") to canonical names |
| `field_descriptions` | Documentation | Human-readable field definitions |
| `field_categories` | Evaluation reporting | Group fields for reporting (financial, temporal, etc.) |
| `evaluation.field_types` | Evaluation metrics | Type-aware comparison (monetary, date, list, boolean) |
| `field_usage.validation_only` | Evaluation | Exclude helper fields from accuracy scoring |

### The `universal` type

The `universal` document type is a superset of all fields across all document types. It's used for single-pass extraction mode (one prompt extracts everything, then the pipeline infers the document type from the results).

```yaml
  universal:
    count: 33
    fields:
      - DOCUMENT_TYPE
      - BUSINESS_ABN
      # ... every field from every document type
      - JOURNEY_PURPOSES
```

---

## File 2: `prompts/document_type_detection.yaml` — Recognition

This file tells the model how to classify a document image into one of the supported types.

### Detection prompt

```yaml
prompts:
  detection:
    prompt: |
      What type of business document is this?

      Answer with one of:
      - INVOICE (includes bills, quotes, estimates)
      - RECEIPT (includes purchase receipts)
      - BANK_STATEMENT (includes credit card statements)
      - TRAVEL_EXPENSE (includes boarding passes, airline tickets)
      - VEHICLE_LOGBOOK (includes motor vehicle logbooks, mileage logs)
```

When adding a new type, you add one line to this list. The model sees the option and can select it.

### Type mappings — normalising model responses

Models don't always respond with the exact canonical name. The `type_mappings` section handles variations:

```yaml
type_mappings:
  "invoice": "INVOICE"
  "tax invoice": "INVOICE"
  "bill": "INVOICE"
  "receipt": "RECEIPT"
  "bank statement": "BANK_STATEMENT"
  "boarding pass": "TRAVEL_EXPENSE"
  "logbook": "VEHICLE_LOGBOOK"
  # ... 40+ mappings
```

The pipeline checks the model's response against these mappings (case-insensitive substring match). First match wins.

### Fallback keywords — last resort

If no type mapping matches, the pipeline scans for keywords:

```yaml
fallback_keywords:
  RECEIPT:
    - receipt
    - purchase
    - payment
  BANK_STATEMENT:
    - bank
    - statement
    - account
  TRAVEL_EXPENSE:
    - travel
    - ticket
    - boarding
  VEHICLE_LOGBOOK:
    - logbook
    - odometer
    - business use
  INVOICE:              # Last — most general
    - invoice
    - bill
    - tax
```

Order matters — more specific types are checked first. If the model says "this appears to be a bank account statement for the period...", the keyword `bank` matches `BANK_STATEMENT` before `INVOICE` gets a chance.

If nothing matches at all, `settings.fallback_type: "INVOICE"` is returned.

---

## File 3: `prompts/<model>_prompts.yaml` — Extraction

Each model has its own extraction prompts file. The prompt key must match the document type name in `field_definitions.yaml`.

```yaml
# prompts/internvl3_prompts.yaml
prompts:
  invoice:
    name: "Invoice Extraction"
    prompt: |
      Extract ALL data from this invoice image.
      Respond in exact format below with actual values or NOT_FOUND.

      DOCUMENT_TYPE: INVOICE
      BUSINESS_ABN: NOT_FOUND
      SUPPLIER_NAME: NOT_FOUND
      ...
      TOTAL_AMOUNT: NOT_FOUND

  receipt:
    name: "Receipt Extraction"
    prompt: |
      Extract ALL data from this receipt image.
      ...

  bank_statement_flat:
    name: "Flat Table Bank Statement"
    prompt: |
      Extract ALL data from this bank statement...

  bank_statement_date_grouped:
    name: "Date-Grouped Bank Statement"
    prompt: |
      ...
```

### Layout variants

Some document types have visually distinct layouts. Bank statements, for example, come in flat-table and date-grouped formats. These are handled with **structure suffixes**:

```yaml
prompts:
  bank_statement_flat:        # Suffix: _flat
    prompt: |
      # Prompt optimised for flat table layout...

  bank_statement_date_grouped: # Suffix: _date_grouped
    prompt: |
      # Prompt optimised for date-grouped layout...

settings:
  structure_suffixes: ["_flat", "_date_grouped"]
```

The pipeline strips these suffixes to map back to the base type (`BANK_STATEMENT`) for field validation and evaluation, but uses the full key to select the right prompt. A vision-based classifier picks the layout variant at runtime.

### Per-model prompt tuning

Each model gets its own prompts file because different models respond better to different instruction styles:

```
prompts/internvl3_prompts.yaml   ← InternVL3 extraction prompts
prompts/llama_prompts.yaml       ← Llama extraction prompts
```

The model registry (`models/registry.py`) maps each model to its prompt file:

```python
register_model(ModelRegistration(
    model_type="internvl3",
    prompt_file="internvl3_prompts.yaml",   # ← here
    ...
))
```

---

## How Auto-Discovery Works at Startup

The CLI (`cli.py`) wires everything together at startup:

```
1. Load the model's extraction YAML (e.g. internvl3_prompts.yaml)
2. Read all prompt keys: {invoice, receipt, bank_statement_flat, bank_statement_date_grouped, ...}
3. Load supported_document_types from field_definitions.yaml: {invoice, receipt, bank_statement, ...}
4. Load structure_suffixes from extraction YAML settings: ["_flat", "_date_grouped"]
5. For each prompt key, strip suffixes → check if base type is in supported_document_types
     "bank_statement_flat" → strip "_flat" → "bank_statement" → YES, it's a doc type
     "invoice" → no suffix → "invoice" → YES
6. Build extraction_files mapping: {INVOICE: path, RECEIPT: path, BANK_STATEMENT: path, ...}
7. Pass prompt_config to the processor — it now knows which types it can extract
```

If a prompt key doesn't match any supported document type (after suffix stripping), it's silently skipped. This means you can have non-document prompts (like utility prompts) in the same YAML without confusing the pipeline.

### Fail-fast validation

If no document type prompts are found, the CLI exits immediately with a clear error:

```
FATAL: No document type prompts found in internvl3_prompts.yaml
Expected: prompts section with keys like 'invoice', 'receipt', etc.
```

---

## Runtime Flow: Detection to Extraction

```
Image arrives
    │
    ▼
Detection prompt (from detection YAML)
    │
    ▼
Model responds: "This is a bank statement"
    │
    ▼
type_mappings: "bank statement" → BANK_STATEMENT
    │
    ▼
Look up BANK_STATEMENT in extraction_files → found
    │
    ▼
Look up field list: field_definitions.yaml → document_fields.bank_statement.fields → 5 fields
    │
    ▼
Vision classifier picks layout: bank_statement_flat
    │
    ▼
Load prompt: extraction YAML → prompts.bank_statement_flat.prompt
    │
    ▼
Calculate token budget: max(base_tokens, 5 fields × tokens_per_field, min_tokens=1500)
    │
    ▼
Run extraction → parse response → validate against 5-field schema → return
```

---

## Worked Example: Adding a Purchase Order Type

### Step 1: Define fields — `config/field_definitions.yaml`

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
      - INVOICE_DATE
      - LINE_ITEM_DESCRIPTIONS
      - LINE_ITEM_QUANTITIES
      - TOTAL_AMOUNT

supported_document_types:
  - invoice
  - receipt
  - bank_statement
  - travel_expense
  - vehicle_logbook
  - purchase_order              # ← add

document_type_aliases:
  # ... existing aliases ...
  purchase_order:
    - purchase order
    - po
    - purchase requisition
```

### Step 2: Add detection support — `prompts/document_type_detection.yaml`

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
      - PURCHASE_ORDER            # ← add

type_mappings:
  # ... existing mappings ...
  "purchase order": "PURCHASE_ORDER"
  "po": "PURCHASE_ORDER"
  "procurement order": "PURCHASE_ORDER"

fallback_keywords:
  # ... existing keywords ...
  # Put BEFORE INVOICE since POs can contain invoice-like language
  PURCHASE_ORDER:
    - purchase order
    - procurement
    - requisition
```

### Step 3: Write extraction prompts — both model YAML files

```yaml
# In prompts/internvl3_prompts.yaml AND prompts/llama_prompts.yaml
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
```

### Step 4: Run it

```bash
python cli.py -d ./purchase_orders -o ./output -g ./ground_truth_po.csv
```

The pipeline auto-discovers `PURCHASE_ORDER` from the prompt key, maps it to the 10-field schema, and runs detection + extraction. Zero Python changes.

---

## Checklist

| File | What to add |
|---|---|
| `config/field_definitions.yaml` | `document_fields.<type>` with field list, add to `supported_document_types`, add `document_type_aliases` |
| `prompts/document_type_detection.yaml` | Add option to detection prompt, add `type_mappings`, add `fallback_keywords` |
| `prompts/internvl3_prompts.yaml` | Extraction prompt with field template |
| `prompts/llama_prompts.yaml` | Extraction prompt with field template |
| Ground truth CSV *(optional)* | One row per image with expected field values |

**Files you do NOT touch**: `cli.py`, `batch_processor.py`, `protocol.py`, `registry.py`, any processor, any evaluation/reporting code.

---

## Comparison: Model Extensibility vs Document Type Extensibility

| | New Model | New Document Type |
|---|---|---|
| **Pattern** | Protocol + Registry | YAML-driven configuration |
| **Mechanism** | Python classes + `register_model()` | YAML keys auto-discovered at startup |
| **Touch Python?** | Yes (3 files) | No |
| **Touch YAML?** | 1 prompts file | 3 YAML files |
| **Example** | Adding Qwen-VL | Adding Purchase Orders |
| **Who can do it** | ML engineer | Data scientist / prompt engineer |
| **Extends** | *How* we extract (different model APIs) | *What* we extract (different document schemas) |

These are two orthogonal extensibility axes. You can add a new model and a new document type independently — they compose without conflict.
