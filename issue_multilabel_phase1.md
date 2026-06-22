# Multi-label document classification — Phase 1: emit `document_labels` alongside the primary `document_type`

**Labels:** `enhancement` · `classification` · `good-first-task`
**Effort:** Small–Medium · **No prompt change, no GPU re-validation** (fully unit-testable locally)
**Owner:** _(you)_

---

## Summary

Extend the document classifier from **single-label** to **multi-label**, starting
with a backward-compatible first step: keep emitting the existing primary
`document_type`, and *additionally* emit a `document_labels: [...]` list capturing
**every** document type the evidence supports.

Phase 1 changes **no downstream behaviour** — extraction still routes on the
primary `document_type`. The label list is added as metadata/observability, and is
the foundation for true multi-label routing later (Phase 2).

## Background — why this is cheap

Our classifier is **evidence-based**: the model reports observable facts (which
table column roles are present, is there payment evidence, is it a travel
document), and YAML rules (`classification_evidence`) derive the type. The
evidence layer **already collects multiple, non-exclusive signals**; the rules
then *collapse* them to one label by first-match-wins precedence.

A **paid Tax Invoice** matches **two** rules at once on its evidence — `INVOICE`
(itemised columns: `gst` / `unit_price` / `quantity`) **and** `RECEIPT`
(`paid: true`). Today first-match-wins returns only `RECEIPT` and **silently
discards** the `INVOICE` match. The rule conditions themselves gate which
combinations are possible (a plain receipt with no itemised columns never
spuriously becomes an invoice), so collecting all matching rules yields a sensible
label set **for free**. This is *"stop throwing away the secondary matches,"* not
a rewrite.

> **Observability bonus:** had this existed, the recent bug where flight
> itineraries were mis-routed to `INVOICE` would have been visible at a glance.
> Surfacing the full label set exposes classification ambiguity for debugging and
> human review.

---

## Decisions for you to own

These are genuine choices — the plan below assumes a default, but they're yours:

1. **Field name** — `document_labels` (assumed) vs `candidate_types` / `labels`.
2. **Primary label for a paid Tax Invoice** — Phase 1 keeps the *current*
   first-match primary (`RECEIPT`) so routing is unchanged. Whether `INVOICE`
   should become primary is a **Phase 2** decision; flag it, don't change it here.
3. **Batch path scope** — see [Step 5](#step-5-decision--the-batch-detection-path).
   The evidence parser only runs on the sequential path today; decide whether to
   bring the batch path to parity now or defer.

---

## Implementation plan

Four required edits + one decision. Each step names the file, the current code,
and the change.

### Step 1 — collect *all* matching rule types

**File:** `common/turn_parsers.py`

Add a helper that returns the matched types in rule order (first = primary), and
refactor the existing `_evaluate_classification` to delegate to it so the
precedence logic lives in one place.

```python
def _evaluate_classification_labels(
    column_mapping: dict[str, str | None] | None,
    paid: bool,
    travel: bool,
    evidence: dict[str, Any],
) -> list[str]:
    """All document types whose rule matches the evidence, in rule order
    (first = primary). Empty when no rule matches (caller applies the default)."""
    present = _present_roles(column_mapping)
    return [
        rule["type"]
        for rule in evidence["rules"]
        if _when_matches(rule["when"], present, paid, travel)
    ]


def _evaluate_classification(
    column_mapping: dict[str, str | None] | None,
    paid: bool,
    travel: bool,
    evidence: dict[str, Any],
) -> str | None:
    """Primary label (first match), or the default — unchanged behaviour."""
    labels = _evaluate_classification_labels(column_mapping, paid, travel, evidence)
    if labels:
        return labels[0]
    default = evidence["default"]
    if isinstance(default, str) and default.lower() == "none":
        return None
    return default
```

`_when_matches` is **unchanged**. No new `when:` keys, so
`common/prompt_catalog.py` validation is **untouched**.

### Step 2 — surface the labels from the parser

**File:** `common/turn_parsers.py`, method `ClassificationParser._parse_enriched`

Current tail (after `paid` / `travel` / `column_mapping` are computed):

```python
evidence = PromptCatalog().get_classification_evidence()
doc_type = _evaluate_classification(column_mapping, paid, travel, evidence)
if doc_type is None:
    return None
complexity = _compute_complexity(row_count)

result: dict[str, Any] = {
    "DOCUMENT_TYPE": doc_type,
    "complexity": complexity,
    "row_count": row_count,
    "payment_evidence": paid,
    "travel_evidence": travel,
}
```

Change to derive the label set, preserving the `none`-default deferral:

```python
evidence = PromptCatalog().get_classification_evidence()
labels = _evaluate_classification_labels(column_mapping, paid, travel, evidence)
if labels:
    doc_type = labels[0]
    document_labels = labels
else:
    default = evidence["default"]
    if isinstance(default, str) and default.lower() == "none":
        return None          # no match + default none -> defer to legacy path
    doc_type = default
    document_labels = [doc_type]   # only the hard default applied

complexity = _compute_complexity(row_count)

result: dict[str, Any] = {
    "DOCUMENT_TYPE": doc_type,
    "DOCUMENT_LABELS": document_labels,   # NEW — primary first
    "complexity": complexity,
    "row_count": row_count,
    "payment_evidence": paid,
    "travel_evidence": travel,
}
```

`DOCUMENT_TYPE` is byte-for-byte the same as before in every case.

### Step 3 — carry labels through the orchestrator

**File:** `models/orchestrator.py`, `detect_and_classify_document()`

Current:

```python
enriched = parser._parse_enriched(response)
if enriched is not None:
    document_type = enriched["DOCUMENT_TYPE"]
else:
    document_type = self._parse_document_type_response(response, detection_config)
...
result = {
    "document_type": document_type,
    "confidence": 1.0,
    "raw_response": response,
    "prompt_used": detection_key,
}
```

Change — labels come from the enriched result, or `[document_type]` on the legacy
fallback path (no `COLUMNS` label at all):

```python
enriched = parser._parse_enriched(response)
if enriched is not None:
    document_type = enriched["DOCUMENT_TYPE"]
    document_labels = enriched["DOCUMENT_LABELS"]
else:
    document_type = self._parse_document_type_response(response, detection_config)
    document_labels = [document_type]
...
result = {
    "document_type": document_type,
    "document_labels": document_labels,   # NEW
    "confidence": 1.0,
    "raw_response": response,
    "prompt_used": detection_key,
}
```

### Step 4 — write labels into `classifications.jsonl`

**File:** `stages/classify.py` — **two** record-construction sites (the batch
loop ~L182 and the sequential loop ~L213). Add one line to each, with a safe
fallback so a record never breaks if a path omits it:

```python
records.append(
    {
        "image_path": image_path,
        "image_name": image_name,
        "document_type": result["document_type"],
        "document_labels": result.get("document_labels", [result["document_type"]]),  # NEW
        "confidence": result.get("confidence", 1.0),
        "raw_response": result.get("raw_response", ""),
        "prompt_used": result.get("prompt_used", "detection"),
    }
)
```

### Step 5 (decision) — the batch detection path

`models/orchestrator.py::detect_batch()` currently calls
`_parse_document_type_response` (legacy keyword matching) and **never runs the
evidence parser** — so it produces neither multi-labels nor evidence-based single
labels today. Two options:

- **(Recommended) Bring it to parity:** route `detect_batch` responses through
  `parser._parse_enriched(response)` (same as the sequential path) so batch and
  sequential classify behave identically and both emit `document_labels`. Small,
  and fixes a pre-existing inconsistency.
- **Defer:** scope Phase 1 to the sequential path; the Step-4 `.get(...)` fallback
  keeps batch records valid (`document_labels == [document_type]`). Note the gap
  in the issue.

Pick one and record the choice in the MR description.

---

## Test plan (local, no GPU)

Drive `ClassificationParser._parse_enriched` with synthetic `raw_response`
strings — the existing classification suite already uses this pattern
(`tests/common/test_classification_evidence.py`,
`tests/common/test_travel_classification.py`). Add:

| Case | `raw_response` (abridged) | Expected `DOCUMENT_TYPE` | Expected `DOCUMENT_LABELS` |
|---|---|---|---|
| Paid Tax Invoice (hybrid) | `COLUMNS: Description \| Qty \| Unit Price \| GST \| Total` · `PAID: YES` | `RECEIPT` | `["RECEIPT", "INVOICE"]` |
| Plain paid receipt | `COLUMNS: NONE` · `PAID: YES` | `RECEIPT` | `["RECEIPT"]` |
| Unpaid invoice | `COLUMNS: …\| Qty \| Unit Price \| GST` · `PAID: NO` | `INVOICE` | `["INVOICE"]` |
| Bank statement | `COLUMNS: Date \| Description \| Debit \| Credit \| Balance` · `PAID: NO` | `BANK_STATEMENT` | `["BANK_STATEMENT"]` |
| Travel | `COLUMNS: NONE` · `TRAVEL: YES` | `TRAVEL` | `["TRAVEL"]` |
| Unmatched + hard default | `COLUMNS: Foo \| Bar` · `PAID: NO` · `TRAVEL: NO` | `INVOICE` | `["INVOICE"]` |

Plus a **back-compat regression**: for every case, assert `DOCUMENT_TYPE` equals
the value the classifier produced *before* this change (i.e. `DOCUMENT_LABELS[0]`).

Run: `pytest tests/common/ -q` (note: `tests/` is local-only / gitignored).

---

## Acceptance criteria

- [ ] A paid, itemised Tax Invoice → `document_type: "RECEIPT"` (unchanged) **and**
      `document_labels: ["RECEIPT", "INVOICE"]`.
- [ ] Plain receipt → `["RECEIPT"]`; bank statement → `["BANK_STATEMENT"]`;
      travel → `["TRAVEL"]`; unpaid invoice → `["INVOICE"]`.
- [ ] Hard-default (unmatched, COLUMNS present) → `["INVOICE"]`.
- [ ] `document_type` is identical to the pre-change value in every case.
- [ ] `classifications.jsonl` records include `document_labels`.
- [ ] No change to extraction routing or to any extraction prompt.
- [ ] Batch-path decision (Step 5) made and recorded in the MR.
- [ ] Unit tests cover hybrid, single-label, default, and back-compat cases;
      `pytest tests/common/` green.

## Why this is low-risk

- **Backward-compatible:** every existing consumer reads `document_type`, which is
  unchanged; `document_labels` is purely additive.
- **No model/prompt change:** labels derive from evidence the model *already*
  returns — no thinking-drift risk, **no remote-GPU re-validation needed**.
- **Honours our conventions:** label set comes from the existing YAML rules — no
  new hardcoded config, fail-fast validation untouched.

## Follow-ups (separate issues — not this MR)

1. **Phase 2 — act on the labels:** choose the routing policy (union-extract +
   merge vs primary-routes / secondary-tags) and the primary-label rule for a Tax
   Invoice; add multi-label ground truth + metrics (per-label F1, subset accuracy,
   Hamming loss).
2. **Segmentation probe:** one detection question counting documents per image, run
   over the real production corpus, to decide *with data* whether image
   segmentation is worth building.
