# Travel & Logbook extraction prompts are orphaned (key/type-name mismatch)

**Type:** Bug
**Component:** extraction routing (`classify` → `extract`)
**Severity:** High — `LOGBOOK` / `TRAVEL` classifications cannot be extracted end-to-end
**Affects:** `prompts/internvl3_prompts.yaml`, `common/prompt_catalog.py`, `cli.py`, `models/orchestrator.py`

---

## Summary

Dedicated extraction prompts for travel-expense and vehicle-logbook documents **exist**
in `prompts/internvl3_prompts.yaml`, but **nothing routes to them**. The prompts are keyed
`travel_expense` / `vehicle_logbook`, whereas the canonical document types are `travel` /
`logbook`. The routing builder only wires a prompt whose key matches a
`supported_document_types` entry, so both prompts are silently dropped.

As a result, a document the classifier labels `TRAVEL` or `LOGBOOK` does **not** reach its
dedicated prompt — it either raises a `ValueError` or falls back to the generic `universal`
prompt, depending on the code path.

This became material because the recently merged evidence-based classifier can now emit
`LOGBOOK` from column evidence (`odometer` / `distance` / `purpose`). Classification works;
extraction does not. The gap was invisible in smoke runs only because the dev set contains no
travel/logbook images.

---

## Background — how routing works

1. **Classification** (`stages/classify.py` → `ClassificationParser`) assigns a canonical
   `document_type` (e.g. `INVOICE`, `BANK_STATEMENT`, `LOGBOOK`).
2. **Routing** (`PromptCatalog.build_extraction_routing`, `common/prompt_catalog.py`) builds a
   `{DOC_TYPE -> prompt_key}` map. For each key in the model's prompt YAML it:
   - strips structure suffixes (`_flat`, `_date_grouped`),
   - checks whether the stripped base is in `supported_document_types`
     (`config/field_definitions.yaml`),
   - and only then adds `routing[base.upper()] = key`.
3. **Extraction** (`stages/extract.py`) resolves the prompt for a `document_type` via that
   routing. `cli.py:107` builds the orchestrator's `extraction_files` directly from the same
   routing dict.

The invariant the system *assumes* (but never enforces): **every extraction prompt key is a
canonical `supported_document_types` value** (modulo structure suffixes).

---

## The problem (verified)

The prompt keys violate that invariant:

| Prompt key (`prompts/internvl3_prompts.yaml`) | Canonical type (`supported_document_types`) | Match? |
|---|---|---|
| `invoice` | `invoice` | ✅ |
| `receipt` | `receipt` | ✅ |
| `bank_statement_flat` / `bank_statement_date_grouped` | `bank_statement` (after suffix strip) | ✅ |
| `universal` | `universal` | ✅ |
| **`travel_expense`** | **`travel`** | ❌ |
| **`vehicle_logbook`** | **`logbook`** | ❌ |

`supported_document_types` (`config/field_definitions.yaml`):
`invoice, receipt, bank_statement, travel, logbook, transaction_link, trust_distribution_link, universal`.

`travel_expense` and `vehicle_logbook` are **not** in that list, so step 2 drops them.

### Actual routing output

```
$ python -c "from common.prompt_catalog import PromptCatalog; \
print(PromptCatalog().build_extraction_routing('internvl3-vllm'))"

BANK_STATEMENT -> bank_statement_flat
INVOICE        -> invoice
RECEIPT        -> receipt
UNIVERSAL      -> universal
# TRAVEL  -> (absent)
# LOGBOOK -> (absent)
```

There is no `TRAVEL` or `LOGBOOK` entry, even though both prompts exist in the YAML.

---

## Impact at runtime

For a document classified `LOGBOOK` (or `TRAVEL`), behaviour depends on the resolution path:

- **Orchestrator path** (`models/orchestrator.py::_resolve_extraction_prompt`): `doc_type_upper`
  is not in `extraction_files` →
  `raise ValueError("No extraction file configured for 'LOGBOOK'. Available: [...]")`.
- **Duck-typed path** (`get_extraction_prompt`): `get_prompt(model, "logbook")` → key not found
  → **silently falls back to the `universal` prompt** (33-field superset).

Either way the dedicated 9-field travel / 16-field logbook prompt is **never used**. In the
silent-fallback case this also degrades accuracy (the universal superset prompt extracts each
field less precisely than a focused prompt — the same effect measured when `default: none`
sent ~21% of the dev set to `UNIVERSAL`, dropping F1 mean 0.650 → 0.626).

---

## Root cause

Legacy naming was never reconciled. The same `TRAVEL_EXPENSE` / `VEHICLE_LOGBOOK` vs
`TRAVEL` / `LOGBOOK` mismatch was already fixed on the **classifier keyword path**
(`type_mappings` values and `fallback_keywords` keys in `prompts/document_type_detection.yaml`
were changed to emit canonical types). The **extraction prompt keys** carry the identical
legacy names and were missed.

Compounding factor: `build_extraction_routing` drops a non-matching prompt key **silently**
(no log, no error), so an orphaned prompt looks like a configured one until a matching document
appears.

---

## Reproduction

```bash
# 1. Routing omits travel/logbook:
python -c "from common.prompt_catalog import PromptCatalog; \
r = PromptCatalog().build_extraction_routing('internvl3-vllm'); \
print('TRAVEL', 'TRAVEL' in r, '| LOGBOOK', 'LOGBOOK' in r)"
# -> TRAVEL False | LOGBOOK False

# 2. End-to-end: classify then extract a vehicle-logbook image.
#    classify -> LOGBOOK; extract -> ValueError or universal-prompt fallback.
#    (Run via the pipeline entrypoint, e.g. KFP_TASK=classify / KFP_TASK=extract.)
```

---

## Proposed solution

Rename the two prompt keys to the canonical type names — a YAML-only change, mirroring the
classifier-side canonical-naming fix already on `master`.

In `prompts/internvl3_prompts.yaml`:

```diff
-  travel_expense:
+  travel:
     name: "Travel Expense Extraction"
     ...
-  vehicle_logbook:
+  logbook:
     name: "Vehicle Logbook Extraction"
     ...
```

After the rename, routing resolves correctly:

```
TRAVEL  -> travel
LOGBOOK -> logbook
```

No Python change is required — `build_extraction_routing` and `cli.py:107` pick the keys up
automatically once they match `supported_document_types`.

### Apply to all model prompt files

Production is vLLM-only (`internvl3_prompts.yaml`), but if any other model prompt YAML defines
the same keys (e.g. a `llama_prompts.yaml`), rename them there too for consistency.

### Add a guard so this cannot silently recur

The underlying weakness is that `build_extraction_routing` drops unmatched keys silently. Add a
fail-fast check (and a unit test) so an orphaned prompt key is caught at load/CI time, e.g.:

- In `build_extraction_routing`, after building `routing`, compute the set of prompt keys that
  did **not** map to any supported type (excluding known non-extraction keys such as `settings`
  / generation blocks). If non-empty, raise a diagnostic error naming the orphaned keys, the
  file, and the allowed `supported_document_types`.
- Add a test asserting `build_extraction_routing('internvl3-vllm')` contains `TRAVEL` and
  `LOGBOOK` (and, more generally, that every prompt key resolves to a supported type).

---

## Acceptance criteria

- [ ] `build_extraction_routing('internvl3-vllm')` includes `TRAVEL -> travel` and
      `LOGBOOK -> logbook`.
- [ ] A document classified `TRAVEL` / `LOGBOOK` is extracted with its dedicated prompt
      (not `universal`, no `ValueError`).
- [ ] A guard/test fails loudly if any extraction prompt key does not map to a
      `supported_document_types` entry.
- [ ] Remote smoke on at least a few real travel and logbook images shows the dedicated
      prompts in use (`prompt_used = travel` / `logbook`) and sensible field extraction.

---

## Notes / out of scope

- **Field-count discrepancy:** the travel prompt's header comment says "8 Fields", but
  `field_definitions.yaml::document_fields.travel.count` is **9**. Confirm the prompt body
  lists all 9 travel fields so extraction and evaluation field sets agree. (Logbook is 16.)
- **`document_type_aliases`:** `config/field_definitions.yaml` already maps the variant phrases
  (`travel expense`, `vehicle logbook`, …) to canonical `travel` / `logbook`. That table is used
  elsewhere (e.g. `bank_post_process._normalize_doc_type`) and is **not** the mechanism behind
  extraction routing — so it does not need changing for this fix, but it confirms `travel` /
  `logbook` are the intended canonical names.
- **Related work:** the classifier-side canonical-naming fix (`type_mappings` /
  `fallback_keywords` now emit `TRAVEL` / `LOGBOOK`) and the evidence-based classifier that
  enables `LOGBOOK` detection are both already on `master`. This issue is the matching
  extraction-side fix.
