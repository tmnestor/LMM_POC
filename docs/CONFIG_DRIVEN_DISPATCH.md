# Config-Driven Dispatch Pattern

How the LMM_POC pipeline routes document images through detection, extraction, and evaluation using YAML configuration as the single source of truth — and how to extend it to new document types.

## Pattern Origins

Config-Driven Dispatch is not a single named pattern from a textbook — it is a composition of three established software design principles applied together:

| Principle | Origin | How we apply it |
|---|---|---|
| **Strategy Pattern** | Gamma et al., *Design Patterns* (GoF, 1994) | The detected document type selects which extraction prompt and field list to use at runtime, without changing the processing code itself. Each document type is effectively a different "strategy" for extraction. |
| **Table-Driven Design** | McConnell, *Code Complete* (2004, Ch. 18) | YAML lookup tables (`type_mappings`, `document_fields`, `prompts`) replace `if/elif` chains. Adding a new document type means adding rows to tables, not writing new branches. |
| **Convention over Configuration** | Ruby on Rails / DHH (2004); widely adopted | Prompt keys match document type names match ground truth directory names. The system resolves `"invoice"` to `field_definitions.yaml → invoice`, `internvl3_prompts.yaml → invoice`, and `evaluation_data/inv_rec/` by naming convention alone — no explicit wiring required. |

The combination is powerful: the Strategy Pattern provides the runtime polymorphism, Table-Driven Design makes that polymorphism data-driven rather than code-driven, and Convention over Configuration eliminates the glue code that would otherwise connect them. The result is a pipeline where new document types require only configuration changes.

## Architecture Overview

The pipeline processes document images in three phases: **detect** the document type, **extract** fields using a type-specific prompt, and **evaluate** against ground truth. Every phase is driven by YAML configuration — no code changes are needed for standard document types.

```mermaid
flowchart TB
    subgraph Input
        IMG["Document Image"]
    end

    subgraph Phase1["Phase 1: Detection"]
        DET_YAML["document_type_detection.yaml"]
        DET_PROMPT["Detection Prompt"]
        VLM_DET["VLM Inference"]
        PARSE["Response Parser + Type Mappings"]
    end

    subgraph Phase2["Phase 2: Extraction"]
        FD_YAML["field_definitions.yaml"]
        EXT_YAML["internvl3_prompts.yaml"]
        FIELD_SEL["Field List Selection"]
        EXT_PROMPT["Type-Specific Prompt"]
        VLM_EXT["VLM Inference"]
        PARSER["Extraction Parser + Cleaner"]
    end

    subgraph Phase3["Phase 3: Evaluation"]
        GT_CSV["ground_truth_*.csv"]
        EVAL["F1 / Precision / Recall"]
        REPORT["Per-Field Score Report"]
    end

    IMG --> DET_YAML --> DET_PROMPT --> VLM_DET --> PARSE
    PARSE -- "INVOICE / RECEIPT / BANK_STATEMENT / TRAVEL_EXPENSE / VEHICLE_LOGBOOK" --> FIELD_SEL
    FD_YAML --> FIELD_SEL
    EXT_YAML --> EXT_PROMPT
    FIELD_SEL --> EXT_PROMPT --> VLM_EXT --> PARSER
    PARSER --> EVAL
    GT_CSV --> EVAL --> REPORT
```

## The Five Configuration Files

All pipeline behaviour is defined in five YAML files and one CSV per document type. No Python code references document types by name — everything is looked up dynamically. `run_config.yml` controls all tunable runtime parameters (batch sizes, generation, GPU memory, model loading).

| File | Purpose | Keyed by |
|---|---|---|
| `config/run_config.yml` | Single source of truth for all tunable parameters (batch, generation, GPU, model loading) | YAML section |
| `config/field_definitions.yaml` | Field lists, types, categories, aliases | `document_type` (lowercase) |
| `prompts/document_type_detection.yaml` | Detection prompts, visual indicators, type mappings | prompt variant key |
| `prompts/internvl3_prompts.yaml` | Extraction prompts per document type | `document_type` (lowercase) |
| `evaluation_data/<type>/ground_truth_<type>.csv` | Ground truth values for evaluation | `filename` column |

```mermaid
graph LR
    subgraph "Single Source of Truth"
        RC["run_config.yml"]
        FD["field_definitions.yaml"]
    end

    subgraph "Consumers"
        BP["BatchDocumentProcessor"]
        IVL["InternVL3 Processor"]
        EM["Evaluation Metrics"]
        CLI["ivl3_cli.py"]
        GPU["GPU Optimization"]
    end

    RC --> CLI
    RC --> GPU
    RC --> BP
    FD --> BP
    FD --> IVL
    FD --> EM
    FD --> CLI
```

## Phase 1: Detection Dispatch

When an image enters the pipeline, the system classifies it using a VLM prompt defined in `document_type_detection.yaml`.

### Detection flow

```mermaid
sequenceDiagram
    participant BP as BatchDocumentProcessor
    participant VLM as Vision Language Model
    participant Parser as Type Parser

    note right of BP: Loads prompt + settings<br/>from detection.yaml
    BP->>VLM: Image + detection prompt
    VLM-->>BP: Raw response text
    BP->>Parser: Parse response (using type_mappings)
    Parser-->>BP: Normalized type (e.g. "INVOICE")
```

> **Reading sequence diagrams:** Solid arrows (&#x2192;) are calls/requests. Dashed arrows (&#x21E2;) are returns/responses. This is standard UML notation.

### Type mappings normalise VLM output

The VLM may respond with varied phrasing. The `type_mappings` section normalises these to canonical types:

```yaml
# prompts/document_type_detection.yaml
type_mappings:
  "invoice": "INVOICE"
  "tax invoice": "INVOICE"
  "bill": "INVOICE"
  "receipt": "RECEIPT"
  "bank statement": "BANK_STATEMENT"
  "boarding pass": "TRAVEL_EXPENSE"
  "e-ticket": "TRAVEL_EXPENSE"
  # ... every expected variation
```

The parser (`_parse_document_type_response`) walks these mappings first, then falls back to `fallback_keywords` matching, then to `settings.fallback_type`. All three layers are YAML-driven:

```yaml
# prompts/document_type_detection.yaml

# Layer 1: Exact phrase matching (checked first)
type_mappings:
  "tax invoice": "INVOICE"
  "boarding pass": "TRAVEL_EXPENSE"
  # ...

# Layer 2: Keyword matching (checked if type_mappings miss)
fallback_keywords:
  RECEIPT:
    - receipt
    - purchase
    - payment
  BANK_STATEMENT:
    - bank
    - statement
  # ...

# Layer 3: Final fallback
settings:
  fallback_type: "INVOICE"
```

## Phase 2: Extraction Dispatch

Once the document type is known, the pipeline selects the correct field list and extraction prompt — both looked up by the same document type key.

### Field selection from `field_definitions.yaml`

```yaml
# config/field_definitions.yaml
document_fields:
  invoice:
    count: 14
    fields:
      - DOCUMENT_TYPE
      - BUSINESS_ABN
      - SUPPLIER_NAME
      # ... 11 more fields

  bank_statement:
    count: 5
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
      # ... 6 more fields

  vehicle_logbook:
    count: 16
    fields:
      - DOCUMENT_TYPE
      - VEHICLE_MAKE
      - VEHICLE_MODEL
      # ... 13 more fields
```

This is the **single source of truth** for what fields exist per document type. The function `load_document_field_definitions()` in `common/batch_processor.py` loads this at runtime with fail-fast validation.

### Prompt selection from `internvl3_prompts.yaml`

The extraction prompt is loaded by key matching the document type:

```python
# Simplified dispatch logic (actual code in batch_processor.py and
# document_aware_internvl3_processor.py)

extraction_key = document_type.lower()   # e.g. "invoice"
prompt = SimplePromptLoader.load_prompt("internvl3_prompts.yaml", extraction_key)
field_list = load_document_field_definitions()[extraction_key]
```

Each prompt lists every expected field with `NOT_FOUND` defaults so the VLM knows the exact output schema:

```yaml
# prompts/internvl3_prompts.yaml
prompts:
  invoice:
    name: "Invoice Extraction"
    prompt: |
      Extract ALL data from this invoice image.
      DOCUMENT_TYPE: INVOICE
      BUSINESS_ABN: NOT_FOUND
      SUPPLIER_NAME: NOT_FOUND
      ...
```

### Dispatch flow

```mermaid
flowchart TD
    DT["Detected Type: INVOICE"]

    DT --> LOOKUP_FIELDS["field_definitions.yaml\n→ invoice.fields (14)"]
    DT --> LOOKUP_PROMPT["internvl3_prompts.yaml\n→ prompts.invoice.prompt"]

    LOOKUP_FIELDS --> VLM_CALL["VLM Inference\n(prompt + image)"]
    LOOKUP_PROMPT --> VLM_CALL

    VLM_CALL --> PARSE["Parse KEY: VALUE pairs"]
    PARSE --> FILTER["Filter to document-specific fields only"]
    FILTER --> RESULT["Cleaned extracted_data dict"]
```

### Document-aware field reduction

Instead of extracting all 33 universal fields for every document, the pipeline only asks for fields relevant to the detected type. This improves both accuracy and speed:

| Document Type | Fields | Reduction vs Universal |
|---|---|---|
| Vehicle Logbook | 16 | 52% fewer |
| Invoice / Receipt | 14 | 58% fewer |
| Travel Expense | 9 | 73% fewer |
| Bank Statement | 5 | 85% fewer |

## Phase 3: Evaluation Dispatch

Evaluation is also config-driven. Field types (monetary, date, boolean, list) determine comparison logic:

```yaml
# config/field_definitions.yaml
evaluation:
  field_types:
    monetary:
      - GST_AMOUNT
      - TOTAL_AMOUNT
      - TRANSACTION_AMOUNTS_PAID
    boolean:
      - IS_GST_INCLUDED
    date:
      - INVOICE_DATE
      - TRANSACTION_DATES
    list:
      - LINE_ITEM_DESCRIPTIONS
      - LINE_ITEM_QUANTITIES
```

The evaluation module (`common/evaluation_metrics.py`) reads these types to select the right comparison strategy per field — no hardcoded field names.

## Special Case: Structure Variants

Some document types have layout variants that require different extraction prompts. Bank statements are the current example:

```mermaid
flowchart TD
    DT["Detected: BANK_STATEMENT"]
    DT --> CLASSIFY["Vision-based structure classifier"]

    CLASSIFY -->|"flat"| FLAT["bank_statement_flat prompt"]
    CLASSIFY -->|"date_grouped"| GROUPED["bank_statement_date_grouped prompt"]

    FLAT --> VLM["VLM Extraction"]
    GROUPED --> VLM

    VLM --> MATH["Mathematical Enhancement\n(balance validation)"]
    MATH --> FILTER["Debit Transaction Filter"]
    FILTER --> EVAL["Evaluation"]
```

The structure classifier (in `models/document_aware_internvl3_processor.py`) uses the VLM itself to analyse layout, then appends a suffix to the prompt key:

```python
structure_type = classify_bank_statement_structure_vision(image_path, model)
extraction_key = f"bank_statement_{structure_type}"  # "bank_statement_flat" or "bank_statement_date_grouped"
```

Both variants share the same field list from `field_definitions.yaml` — only the prompt changes.

---

## Adding a New Document Type

### Zero Code Changes Required

All steps are **YAML/CSV only**. The pipeline derives supported types dynamically at runtime:

- `load_prompt_config()` reads extraction prompt keys from `internvl3_prompts.yaml`
- `load_document_field_definitions()` reads document types from `field_definitions.yaml`
- `_parse_document_type_response()` reads mappings and keywords from `document_type_detection.yaml`

### Checklist

```mermaid
flowchart LR
    A["1. Field Definitions\n(YAML)"] --> B["2. Detection Config\n(YAML)"]
    B --> C["3. Extraction Prompt\n(YAML)"]
    C --> D["4. Ground Truth\n(CSV)"]
    D --> E["Test"]

    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style E fill:#e8f5e9
```

The walkthrough below uses a hypothetical `PURCHASE_ORDER` type.

---

### Step 1: Define fields in `config/field_definitions.yaml`

Add the new type under `document_fields` with its field list:

```yaml
document_fields:
  # ... existing types ...

  purchase_order:
    count: 10
    fields:
      - DOCUMENT_TYPE
      - PO_NUMBER
      - VENDOR_NAME
      - VENDOR_ADDRESS
      - BUYER_NAME
      - BUYER_ADDRESS
      - ORDER_DATE
      - LINE_ITEM_DESCRIPTIONS
      - LINE_ITEM_QUANTITIES
      - TOTAL_AMOUNT
```

Then update the supporting sections:

```yaml
# Add descriptions for new fields
field_descriptions:
  PO_NUMBER: "Purchase order number"
  VENDOR_NAME: "Vendor/supplier company name"
  BUYER_NAME: "Buying organization name"
  BUYER_ADDRESS: "Buying organization address"
  ORDER_DATE: "Date the purchase order was issued (DD/MM/YYYY)"

# Categorise fields
field_categories:
  identification:
    - PO_NUMBER           # ADD
  business_details:
    - VENDOR_NAME         # already exists as SUPPLIER_NAME alias
    - VENDOR_ADDRESS      # ADD
  customer_details:
    - BUYER_NAME          # ADD
    - BUYER_ADDRESS       # ADD
  temporal:
    - ORDER_DATE          # ADD

# Classify field types for evaluation
evaluation:
  field_types:
    monetary:
      - TOTAL_AMOUNT      # already listed
    date:
      - ORDER_DATE        # ADD
    list:
      - LINE_ITEM_DESCRIPTIONS  # already listed
      - LINE_ITEM_QUANTITIES    # already listed

# Add aliases
document_type_aliases:
  purchase_order:
    - purchase order
    - po
    - p.o.

# Register as supported
supported_document_types:
  - invoice
  - receipt
  - bank_statement
  - travel_expense
  - vehicle_logbook
  - purchase_order        # ADD
```

### Step 2: Add detection rules in `prompts/document_type_detection.yaml`

Three sections to update: detection prompts, type mappings, and fallback keywords.

```yaml
# 2a. Add visual indicators to the detection prompt
prompts:
  detection:
    prompt: |
      What type of business document is this?
      Answer with one of:
      # ... existing types ...
      - PURCHASE_ORDER (includes purchase orders, P.O.s)

  detection_complex:
    prompt: |
      # ... existing indicators ...

      PURCHASE ORDER INDICATORS:
      - Header: "PURCHASE ORDER", "P.O.", "PO NUMBER"
      - Layout: Buyer header -> Vendor "Ship To" -> Line items -> Approval
      - Key fields: PO number, delivery date, line items with quantities
      - Visual: Company letterhead, approval signatures, delivery terms

      # ... update response options ...
      - PURCHASE_ORDER (for purchase orders, P.O.s)

# 2b. Add type mappings for response normalisation
type_mappings:
  "purchase order": "PURCHASE_ORDER"
  "purchase_order": "PURCHASE_ORDER"
  "purchaseorder": "PURCHASE_ORDER"
  "po": "PURCHASE_ORDER"
  "p.o.": "PURCHASE_ORDER"

# 2c. Add fallback keywords (used when type_mappings don't match)
fallback_keywords:
  # ... existing types ...
  PURCHASE_ORDER:
    - purchase order
    - po number
    - requisition
```

### Step 3: Create extraction prompt in `prompts/internvl3_prompts.yaml`

Add a prompt key matching the lowercase document type:

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
      PO_NUMBER: NOT_FOUND
      VENDOR_NAME: NOT_FOUND
      VENDOR_ADDRESS: NOT_FOUND
      BUYER_NAME: NOT_FOUND
      BUYER_ADDRESS: NOT_FOUND
      ORDER_DATE: NOT_FOUND
      LINE_ITEM_DESCRIPTIONS: NOT_FOUND
      LINE_ITEM_QUANTITIES: NOT_FOUND
      TOTAL_AMOUNT: NOT_FOUND

      Instructions:
      - Find PO number: Usually at top right, prefixed "PO" or "P.O."
      - Find vendor: Company fulfilling the order
      - Find buyer: Organization placing the order
      - Find date: Use DD/MM/YYYY format
      - Find line items: List with " | " separator
      - Find amounts: Include $ symbol
      - Replace NOT_FOUND with actual values
```

### Step 4: Create ground truth data

```
evaluation_data/
  purchase_order/
    ground_truth_purchase_order.csv
    po_001.png
    po_002.png
    ...
```

CSV structure — column names must match `field_definitions.yaml` exactly:

```csv
filename,DOCUMENT_TYPE,PO_NUMBER,VENDOR_NAME,VENDOR_ADDRESS,BUYER_NAME,BUYER_ADDRESS,ORDER_DATE,LINE_ITEM_DESCRIPTIONS,LINE_ITEM_QUANTITIES,TOTAL_AMOUNT
po_001.png,PURCHASE_ORDER,PO-2024-0042,Acme Supplies,"123 Industrial Rd, Sydney",TechCorp,"456 Business Ave, Melbourne",15/03/2024,Widget A | Widget B,100 | 50,$7500.00
```

That's it — no Python code changes needed. The pipeline will:

1. **`load_prompt_config()`** reads `internvl3_prompts.yaml`, finds the `purchase_order` key, and adds `PURCHASE_ORDER` to the extraction routing table automatically.
2. **`load_document_field_definitions()`** iterates all types under `document_fields` in the YAML (excluding `universal`), so `purchase_order` is validated and loaded automatically.
3. **`_parse_document_type_response()`** reads `type_mappings` and `fallback_keywords` from the detection YAML, so `PURCHASE_ORDER` is recognized automatically.

---

## Adding Structure Variants (Optional)

If your new document type has multiple layouts, follow the bank statement pattern:

```mermaid
flowchart TD
    A["Detect: PURCHASE_ORDER"] --> B{"Has layout variants?"}
    B -->|"No"| C["Use purchase_order prompt directly"]
    B -->|"Yes"| D["Run structure classifier"]
    D -->|"tabular"| E["purchase_order_tabular prompt"]
    D -->|"form"| F["purchase_order_form prompt"]
```

1. **Create a classifier** — write a function similar to `classify_bank_statement_structure_vision()` that uses the VLM to identify the layout variant
2. **Add variant prompts** — add `purchase_order_tabular` and `purchase_order_form` keys to the extraction prompt YAML
3. **Wire the suffix** — in `process_document_aware()`, append the structure suffix to the prompt key

The field list stays the same for all variants — only the prompt changes.

---

## File Reference

| File | Role |
|---|---|
| `config/run_config.yml` | Single source of truth for tunable parameters (batch, generation, GPU, model loading) |
| `config/field_definitions.yaml` | Field lists, types, categories, aliases (single source of truth for document types) |
| `prompts/document_type_detection.yaml` | Detection prompts, type mappings, fallback keywords, settings |
| `prompts/internvl3_prompts.yaml` | Extraction prompts per document type (keys define supported types) |
| `ivl3_cli.py` | CLI entry point; derives extraction routing from YAML prompt keys |
| `common/pipeline_config.py` | PipelineConfig dataclass; loads and merges YAML/ENV/CLI config |
| `common/config.py` | Module-level constants; `apply_yaml_overrides()` replaces defaults from YAML |
| `common/batch_processor.py` | Orchestrates detect -> extract -> evaluate pipeline |
| `common/gpu_optimization.py` | CUDA memory management, OOM recovery, resilient generation |
| `common/simple_prompt_loader.py` | Loads prompts from YAML by file + key |
| `common/evaluation_metrics.py` | Field-type-aware accuracy scoring |
| `models/document_aware_internvl3_processor.py` | InternVL3 detection + extraction + structure classification |
| `common/bank_statement_adapter.py` | Multi-turn extraction for complex tables |
| `evaluation_data/*/ground_truth_*.csv` | Ground truth per document type |

## Model Processor Interface

`BatchDocumentProcessor` expects a model handler object that implements a duck-typed interface. Any model processor that provides these methods can be plugged into the pipeline without changes to the batch processing code.

### Required methods

These methods must be implemented for the pipeline to function:

```python
def detect_and_classify_document(self, image_path: str, verbose: bool = False) -> dict:
    """Classify a document image into a type.

    Returns:
        dict with at least {"document_type": "INVOICE" | "RECEIPT" | "BANK_STATEMENT" | "TRAVEL_EXPENSE" | "VEHICLE_LOGBOOK" | ...}
    """

def process_document_aware(self, image_path: str, classification_info: dict, verbose: bool = False) -> dict:
    """Extract fields from a document given its classification.

    Returns:
        dict with at least {"extracted_data": {field: value, ...}}
    """
```

### Optional methods (enable batch mode)

When these methods are present and `batch_size > 1`, the pipeline uses batched inference for higher throughput:

```python
def batch_detect_documents(self, image_paths: list[str], verbose: bool = False) -> list[dict]:
    """Classify multiple documents in a single batched call.

    Returns:
        List of classification dicts, one per image (same format as detect_and_classify_document).
    """

def batch_extract_documents(
    self, image_paths: list[str], classification_infos: list[dict], verbose: bool = False
) -> list[dict]:
    """Extract fields from multiple documents in a single batched call.

    Returns:
        List of extraction dicts, one per image (same format as process_document_aware).
    """
```

If the optional methods are absent, the pipeline falls back to sequential processing using the required methods.

### Current implementation

`models/document_aware_internvl3_processor.py` (`DocumentAwareInternVL3HybridProcessor`) is the InternVL3 implementation of this interface.

## Design Principles

1. **YAML is the single source of truth** — field lists, prompt text, type mappings, fallback keywords, evaluation rules, and all tunable runtime parameters (`run_config.yml`) live in config files, not code
2. **Dynamic derivation over hardcoded lists** — supported document types are derived at runtime from YAML keys; adding a new type requires zero Python changes
3. **Fail fast with diagnostics** — `load_document_field_definitions()` validates structure at load time and raises clear errors with remediation steps
4. **Document-aware field reduction** — each type extracts only its relevant fields, improving accuracy and reducing token usage
5. **Model-agnostic pipeline** — `BatchDocumentProcessor` depends on a duck-typed interface, not a specific model implementation
6. **Convention over configuration** — prompt keys match document type names match ground truth directory names; the system resolves types by naming convention alone
