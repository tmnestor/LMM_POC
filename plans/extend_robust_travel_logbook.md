# Extend Robust Workflow: Travel, Logbook + JSONL Ground Truth

## Problem

1. **New document types**: TRAVEL (tickets + itineraries) and LOGBOOK (vehicle logbooks) need extraction support in the robust probe-based workflow.
2. **Ground truth format**: CSV forces a fixed column superset across all types. A receipt row has 15 empty logbook columns. This creates two problems:
   - **Noise**: Cross-schema NOT_FOUNDs inflate F1 with trivial free points.
   - **But**: Within-schema NOT_FOUND accuracy is critical — hallucination (model inventing values) must be punished.
3. **Solution**: JSONL ground truth where each record carries only its type's fields. The evaluator scores only fields present in the ground truth record.

---

## Part 1: JSONL Ground Truth

### Ground truth already converted

The combined JSONL file is already generated at:
```
../evaluation_data/synthetic/ground_truth.jsonl
```

29 records, 5 types, per-type field sets:

| Type | Count | Fields | Notes |
|------|-------|--------|-------|
| RECEIPT | 3 | 14 | Full receipt schema, cross-schema NOT_FOUNDs removed |
| INVOICE | 3 | 14 | Same field set as receipt |
| BANK_STATEMENT | 3 | 8 | Core + validation fields (AMOUNTS_RECEIVED, BALANCE) |
| TRAVEL | 10 | 9 | Tickets + itineraries |
| LOGBOOK | 10 | 15 | Vehicle logbooks |

Format:
```jsonl
{"filename": "image_001.png", "DOCUMENT_TYPE": "RECEIPT", "SUPPLIER_NAME": "Liberty Oil", "TOTAL_AMOUNT": "$94.87", "PAYER_NAME": "Robert Taylor", ...}
{"filename": "ticket_001_SX_SYD_HBA.png", "DOCUMENT_TYPE": "TRAVEL", "PASSENGER_NAME": "Martin/Olivia", "TRAVEL_ROUTE": "Sydney | Hobart", ...}
{"filename": "logbook_001_professional.png", "DOCUMENT_TYPE": "LOGBOOK", "VEHICLE_MAKE": "Toyota", "VEHICLE_MODEL": "Camry", ...}
```

Rules applied during conversion:
- Each record has `filename` + `DOCUMENT_TYPE` + only that type's schema fields
- Within-schema fields that are genuinely absent = `"NOT_FOUND"` (evaluated, catches hallucination)
- Cross-schema fields with NOT_FOUND values are dropped (no free points)
- Cross-schema fields with real values are kept (e.g., SUPPLIER_NAME on bank statements)

### Files to change

| File | Action | Purpose |
|------|--------|---------|
| `common/evaluation_metrics.py` | EXTEND | `load_ground_truth()` — detect `.jsonl` extension, load with per-record fields |
| `stages/evaluate.py` | MINOR | `--ground-truth` help text — note it accepts `.jsonl` |

No conversion script needed — already done.

### `load_ground_truth()` changes

```python
def load_ground_truth(gt_path: str, ...) -> dict[str, dict]:
    path = Path(gt_path)
    if path.suffix == ".jsonl":
        return _load_ground_truth_jsonl(path, ...)
    return _load_ground_truth_csv(path, ...)  # existing logic, renamed
```

JSONL loader:
```python
def _load_ground_truth_jsonl(path: Path, ...) -> dict[str, dict]:
    ground_truth_map = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            filename = record.get("filename") or record.get("image_name")
            if filename:
                ground_truth_map[filename] = record
    return ground_truth_map
```

### Evaluator field selection

Currently, `ExtractionEvaluator` filters fields by looking up `field_definitions.yaml[document_type].fields`. With JSONL, a simpler and more correct approach:

**Evaluate exactly the fields present in the ground truth record** (minus `filename`/`image_name`). This naturally handles:
- Within-type NOT_FOUNDs (present in record as `"NOT_FOUND"` -> evaluated)
- Cross-type fields (absent from record -> not evaluated)
- New types (no config change needed — fields come from the data)

This is a one-line change in the evaluator: instead of looking up field_definitions, use `set(gt_record.keys()) - {"filename", "image_name"}` as the evaluation field set. Only apply this when ground truth source is JSONL.

---

## Part 2: Extraction Templates

### Canonical type names

- **Router edges**: `is_travel`, `is_logbook` (short, clean)
- **Normalizer**: maps model output -> `TRAVEL` or `LOGBOOK`
- **field_definitions.yaml**: rename `travel_expense` -> `travel`, `vehicle_logbook` -> `logbook`
- **Ground truth DOCUMENT_TYPE**: `TRAVEL`, `LOGBOOK` (already in ground_truth.jsonl)

### `extract_travel` node

Template extracts the 8 non-DOCUMENT_TYPE fields:

```yaml
extract_travel:
  template: |
    Extract ALL data from this travel document. Respond in exact format below
    with actual values or NOT_FOUND.

    DOCUMENT_TYPE: TRAVEL
    PASSENGER_NAME: NOT_FOUND
    TRAVEL_MODE: NOT_FOUND
    TRAVEL_ROUTE: NOT_FOUND
    TRAVEL_DATES: NOT_FOUND
    INVOICE_DATE: NOT_FOUND
    GST_AMOUNT: NOT_FOUND
    TOTAL_AMOUNT: NOT_FOUND
    SUPPLIER_NAME: NOT_FOUND

    Instructions:
    - PASSENGER_NAME: Full passenger name in LASTNAME/FIRSTNAME format
    - TRAVEL_MODE: Mode of travel (plane, train, bus, taxi, ferry)
    - TRAVEL_ROUTE: All cities in order with " | " separator
      (e.g., "Sydney | Melbourne | Sydney" for a return trip)
    - TRAVEL_DATES: All travel/departure dates with " | " separator
      in DD Mon YYYY format (e.g., "16 Feb 2026 | 18 Feb 2026")
    - INVOICE_DATE: Booking/issue date in DD Mon YYYY format
    - GST_AMOUNT: GST amount with $ symbol
    - TOTAL_AMOUNT: Total fare amount with $ symbol
    - SUPPLIER_NAME: Airline or transport company name
    - Replace NOT_FOUND with actual values
  max_tokens: 1024
  parser: field_value
  edges:
    ok: done
```

### `extract_logbook` node

Template extracts the 14 non-DOCUMENT_TYPE fields:

```yaml
extract_logbook:
  template: |
    Extract ALL data from this vehicle logbook. Respond in exact format below
    with actual values or NOT_FOUND.

    DOCUMENT_TYPE: LOGBOOK
    VEHICLE_MAKE: NOT_FOUND
    VEHICLE_MODEL: NOT_FOUND
    VEHICLE_REGISTRATION: NOT_FOUND
    ENGINE_CAPACITY: NOT_FOUND
    LOGBOOK_PERIOD_START: NOT_FOUND
    LOGBOOK_PERIOD_END: NOT_FOUND
    ODOMETER_START: NOT_FOUND
    ODOMETER_END: NOT_FOUND
    TOTAL_KILOMETERS: NOT_FOUND
    BUSINESS_KILOMETERS: NOT_FOUND
    BUSINESS_USE_PERCENTAGE: NOT_FOUND
    JOURNEY_DATES: NOT_FOUND
    JOURNEY_DISTANCES: NOT_FOUND
    JOURNEY_PURPOSES: NOT_FOUND

    Instructions:
    - VEHICLE_MAKE: Manufacturer (e.g., Toyota, Mazda, Ford)
    - VEHICLE_MODEL: Model name (e.g., Camry, CX-5, Ranger)
    - VEHICLE_REGISTRATION: Registration plate number
    - ENGINE_CAPACITY: Engine size with L suffix (e.g., 2.5L)
    - LOGBOOK_PERIOD_START / LOGBOOK_PERIOD_END: DD Mon YYYY format
    - ODOMETER_START / ODOMETER_END: Numeric odometer readings
    - TOTAL_KILOMETERS: Total km for the period (numeric)
    - BUSINESS_KILOMETERS: Business-only km (numeric)
    - BUSINESS_USE_PERCENTAGE: Percentage with % symbol (e.g., 78%)
    - JOURNEY_DATES: All journey dates with " | " separator (DD Mon YYYY)
    - JOURNEY_DISTANCES: All distances in km with " | " separator (numeric)
    - JOURNEY_PURPOSES: All purposes with " | " separator
    - Extract the first 10 journeys only (evaluation uses first 10)
    - Replace NOT_FOUND with actual values
  max_tokens: 2048
  parser: field_value
  edges:
    ok: done
```

### Updated `robust_extract.yaml` graph

```
probe_document -> probe_bank_headers -> select_best_type -> route_best_type
  -> is_receipt        -> done  (2 calls, probe fields used)
  -> is_invoice        -> done  (2 calls, probe fields used)
  -> is_bank_statement -> select_bank_strategy -> ... -> bank_post_process -> done  (3-4 calls)
  -> is_travel         -> extract_travel -> done  (3 calls)
  -> is_logbook        -> extract_logbook -> done  (3 calls)
  -> default           -> done  (2 calls, probe fields as fallback)
```

### Validator changes

**`_normalize_doc_type()`** — add aliases:
```python
_DOC_TYPE_ALIASES: dict[str, str] = {
    # existing...
    "tax invoice": "INVOICE",
    # travel
    "travel": "TRAVEL",
    "travel expense": "TRAVEL",
    "itinerary": "TRAVEL",
    "boarding pass": "TRAVEL",
    "flight ticket": "TRAVEL",
    "airline ticket": "TRAVEL",
    "e-ticket": "TRAVEL",
    # logbook
    "logbook": "LOGBOOK",
    "vehicle logbook": "LOGBOOK",
    "vehicle_logbook": "LOGBOOK",
    "mileage log": "LOGBOOK",
    "motor vehicle logbook": "LOGBOOK",
}
```

Add `"TRAVEL"` and `"LOGBOOK"` to the canonical passthrough set.

**`run_select_best_type()`** — no logic change needed. The existing else branch already uses `_normalize_doc_type()` which will return TRAVEL/LOGBOOK, and the router will match `is_travel`/`is_logbook`.

**`probe_document` template** — add TRAVEL and LOGBOOK to the DOCUMENT_TYPE instructions:
```
- DOCUMENT_TYPE: Determine from document content:
  * RECEIPT if there is evidence of COMPLETED PAYMENT
  * INVOICE if this is a BILL requesting future payment
  * BANK_STATEMENT if this shows account transactions over a period
  * TRAVEL if this is a flight ticket, itinerary, or boarding pass
  * LOGBOOK if this is a vehicle/mileage logbook
```

### `field_definitions.yaml` changes

Rename types and add `TRAVEL`/`LOGBOOK` to canonical set:
- `travel_expense` -> `travel`
- `vehicle_logbook` -> `logbook`
- Update `supported_document_types`, `document_type_aliases`, `document_fields`

---

## Part 3: Pipeline Script

Update `scripts/run_graph_robust.sh` to use the combined dataset and JSONL ground truth:

```bash
DATA_DIR="../evaluation_data/synthetic"
GROUND_TRUTH="../evaluation_data/synthetic/ground_truth.jsonl"
```

Pass `--ground-truth` to the evaluate stage.

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `prompts/workflows/robust_extract.yaml` | EXTEND | Add `extract_travel`, `extract_logbook` nodes + router edges |
| `common/bank_post_process.py` | EXTEND | `_normalize_doc_type()` — add TRAVEL/LOGBOOK aliases + canonical values |
| `common/evaluation_metrics.py` | EXTEND | `load_ground_truth()` — JSONL support (detect `.jsonl`, load per-record) |
| `config/field_definitions.yaml` | EXTEND | Rename `travel_expense` -> `travel`, `vehicle_logbook` -> `logbook` |
| `scripts/run_graph_robust.sh` | UPDATE | Point to `ground_truth.jsonl` |
| `tests/test_robust_graph.py` | EXTEND | Travel/logbook path tests |
| `tests/test_ground_truth_jsonl.py` | NEW | JSONL loading tests |

---

## Implementation Order

1. **JSONL ground truth loader** (no GPU needed, testable locally)
   - `load_ground_truth()` JSONL branch in `evaluation_metrics.py`
   - Tests for JSONL loading
2. **Extraction templates** (testable locally with mock generate_fn)
   - `extract_travel` + `extract_logbook` nodes in YAML
   - Router edges in `robust_extract.yaml`
   - Normalizer aliases in `bank_post_process.py`
   - `probe_document` type instructions update
   - Tests with canned responses
3. **field_definitions.yaml** rename (coordinate with evaluation)
4. **Pipeline script** update + integration test on remote
