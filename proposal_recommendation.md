# Proposal & Recommendation — *Expand `ClassificationParser` document types*

**Re:** GitLab issue *"Expand ClassificationParser document types"*
**Status:** Review response + recommended approach
**Scope:** the `classify` stage only (evidence-based document-type detection). Extraction is already YAML-only and is out of scope here.

---

## 1. Summary

The issue correctly identifies a real limitation: the **primary** classification path
(`ClassificationParser._parse_enriched`) derives the document type from a **hardcoded** Python
`if/elif` chain and can only ever emit `BANK_STATEMENT`, `RECEIPT`, or `INVOICE`. To support audited
documents that are none of these (e.g. vehicle logbooks), that logic must be expanded.

We **agree with the problem statement and the two target locations**. However, the example snippet in
*IDEAS* will not work as literally written — it is effectively a no-op and contains a routing bug
(details in §3). We recommend implementing the change as a **declarative, YAML-driven rule set**
(§4), which both fixes those defects and satisfies the project rule that *YAML is the single source
of truth*. This also makes the README's existing promise ("new document types are added in YAML
only") finally true for detection.

---

## 2. Correction to the issue text

> *IDEAS, item 2:* "Instead of defaulting to INVOICE, return None, which will pass to
> `ClassificationParser._parse_document_type_response`."

There is **no `_parse_document_type_response` method on `ClassificationParser`.** The actual fallback
chain is:

| Step | Location |
|---|---|
| `_parse_enriched` returns `None` | `common/turn_parsers.py:430` |
| → `ClassificationParser.parse()` calls `self._parse_legacy()` | `common/turn_parsers.py:525` |
| → which delegates to module-level `_match_document_type()` | `common/turn_parsers.py:542` |

`_parse_document_type_response` exists only on a **different** class, `DocumentOrchestrator`
(`models/orchestrator.py:237`). `_match_document_type` is documented as a *"standalone copy"* of that
method — almost certainly the source of the mix-up.

**The mechanism described is correct** (return `None` → fall through to keyword matching); only the
method name is wrong. The line should read:

> …return None, which will pass to `ClassificationParser._parse_legacy`.

---

## 3. Why the example snippet will not work as written

The *IDEAS* snippet, verified against the current code, has four issues. The first makes it a no-op;
the rest are latent bugs that survive even after the obvious follow-up edit.

### 3.1 The new `has_*_columns` checks are always `False` (no-op)

`column_mapping` is built by `HeaderListParser._match_columns` (`common/turn_parsers.py:144`), which
**only ever populates six fixed keys**: `date, description, debit, credit, balance, amount`.
Therefore:

```python
column_mapping.get("distance")   # -> always None
column_mapping.get("odometer")   # -> always None
column_mapping.get("GST")        # -> always None
```

`has_logbook_columns` and `has_invoice_columns` are permanently `False`, so those branches are dead
code. The snippet changes nothing except the final `else: return None`. The issue's *POSSIBLE
CHANGES* section does acknowledge this dependency ("Add more column mappings in `HeaderListParser`"),
but the snippet is not self-contained.

### 3.2 Case / name mismatch survives the follow-up edit

Mapping keys are **lowercase role names**. Even after a `gst` role is added,
`column_mapping.get("GST")` (uppercase) still returns `None`. Likewise `.get("price")` will miss a
role named `unit_price`. The lookup keys must match the registered role names exactly.

### 3.3 Precedence reorders receipts into invoices (behaviour regression)

The snippet evaluates `has_invoice_columns` **before** `paid`. The current order is
bank → `paid`→`RECEIPT` → `INVOICE` (default). A **paid receipt that includes an itemised
qty/price/GST table** (a common tax-invoice-style receipt) would now be classified `INVOICE` instead
of `RECEIPT`. This is a semantic change that should be a deliberate, reviewed decision.

### 3.4 `"VEHICLE_LOGBOOK"` will not route to extraction

The snippet returns `doc_type = "VEHICLE_LOGBOOK"`, but the **canonical key is `LOGBOOK`**
(`config/field_definitions.yaml`, `supported_document_types`). Neither routing path resolves the
alias on the classifier's output:

- `get_extraction_prompt()` → `get_prompt(..., "VEHICLE_LOGBOOK")` → key not found
  (`common/prompt_catalog.py:99`) → **silently falls back to the universal prompt.**
- `_resolve_extraction_prompt()` (`models/orchestrator.py:752`) uppercases and looks up
  `extraction_files["VEHICLE_LOGBOOK"]` → **`ValueError`, no file configured.**

The `document_type_aliases.logbook` table *does* list `vehicle_logbook`
(`config/field_definitions.yaml`), but that table is applied elsewhere (e.g.
`bank_post_process._normalize_doc_type`), **not** on the classifier output before routing. The rule
must therefore emit the canonical `"LOGBOOK"` directly. (The snippet also defines no `TRAVEL` rule.)

### 3.5 Note: the `return None` fallback is brittle by design-avoidance

When `_parse_enriched` returns `None`, the fallback keyword-matches the **raw COLUMNS/PAID/ROWS
response text**. It may *accidentally* catch a logbook because `"odometer"` is a `fallback_keyword`
that can appear as a column header — but this is exactly the fragile keyword-guessing the
evidence-based design exists to replace. It should not be relied upon as the detection mechanism for
a supported type.

---

## 4. Recommended approach — YAML-driven `classification_evidence`

Replace the hardcoded `COLUMN_PATTERNS` literal and the `if/elif` chain with two declarative blocks
in `prompts/document_type_detection.yaml`. Adding a detectable type then becomes a **YAML-only**
change, consistent with the keyword path and the README.

```yaml
# Semantic column roles: role -> header keywords that map to it.
# Replaces the hardcoded HeaderListParser.COLUMN_PATTERNS.
column_roles:
  date:        [date, trans date, transaction date, value date, posting date]
  description: [description, details, transaction, particulars, narrative, reference]
  debit:       [debit, withdrawal, withdrawals, dr, money out]
  credit:      [credit, deposit, deposits, cr, money in]
  balance:     [balance, running balance, closing balance]
  amount:      [amount, transaction amount]
  distance:    [distance, distance travelled, km, kilometres]
  odometer:    [odometer, odometer reading, start odometer, end odometer]
  purpose:     [purpose, purpose of trip, reason]
  quantity:    [quantity, qty, units]
  unit_price:  [unit price, price, rate]
  gst:         [gst, tax, vat]

# Evidence-based classification rules, evaluated TOP-DOWN, first match wins.
# This list IS the precedence that the if/elif chain currently hardcodes.
classification_evidence:
  rules:
    - type: BANK_STATEMENT
      when: { any_roles: [debit, credit, balance] }
    - type: LOGBOOK                 # canonical key — NOT "VEHICLE_LOGBOOK"
      when: { any_roles: [distance, odometer, purpose] }
    - type: INVOICE
      when: { any_roles: [gst, unit_price, quantity] }
    - type: RECEIPT
      when: { paid: true }
  # No rule matched:
  #   none   -> return None -> defer to the legacy keyword path (_parse_legacy)
  #   <TYPE> -> hard default (e.g. INVOICE = today's behaviour)
  default: none
```

**`when:` vocabulary** (intentionally minimal): `any_roles: [...]`, `all_roles: [...]`,
`paid: true|false`. Multiple keys within one `when:` are AND-ed.

**How the code consumes it:**

- `HeaderListParser._match_columns` iterates `column_roles` instead of the class literal, so new
  roles need no Python change.
- `_parse_enriched` builds `column_mapping`, then walks `rules` top-down and returns the first `type`
  whose `when:` is satisfied; otherwise applies `default` (`none` → `return None`).

This directly resolves §3: roles are data-driven (3.1), keys are lowercase YAML (3.2), precedence is
explicit and reviewable (3.3), and rule types are validated against the canonical type list (3.4).

---

## 5. Reconciliations to lock before coding

1. **Type-key naming.** The canonical keys are `TRAVEL` and `LOGBOOK`
   (`config/field_definitions.yaml` → `supported_document_types`), but `type_mappings` in
   `document_type_detection.yaml` currently emits `TRAVEL_EXPENSE` / `VEHICLE_LOGBOOK`. New rules
   **must** emit canonical keys, and the alias chain in `document_type_aliases` should be verified so
   the legacy and evidence paths agree. *(This is a pre-existing latent mismatch this work surfaces.)*

2. **Column-keyword consolidation (verified).** Column keyword lists already exist in **two**
   independently-maintained places that have **drifted apart**:
   - `prompts/bank_column_patterns.yaml` — YAML, shape `patterns.<role>.keywords` (+ `priority` /
     `required` / `strategy_selection`), consumed by `unified_bank_extractor.ColumnMatcher`
     (`common/unified_bank_extractor.py:93`).
   - `HeaderListParser.COLUMN_PATTERNS` — a **hardcoded** flat `<role>: [keywords]` literal
     (`common/turn_parsers.py:83`).

   These are not identical copies: e.g. the YAML maps `debit` to `paid / paid out / spent` and
   `credit` to `received`, whereas the hardcoded literal uses `money out` / `money in`. The proposed
   `column_roles` block would move the hardcoded literal into YAML (net: still two YAML sources, none
   hardcoded). Recommendation: co-locate `column_roles` in `document_type_detection.yaml` (next to
   the rules that consume it) and leave `bank_column_patterns.yaml` as-is for now; reconciling the
   two keyword sets onto one source can be a follow-up.

3. **`default:` value** — `none` (route unmatched docs to the keyword fallback, as the issue
   proposes) vs `INVOICE` (preserve today's exact behaviour). We recommend `none` but flag it as the
   one genuine behaviour change for explicit sign-off.

---

## 6. Validation & rollout (fail-fast)

Validate at config-load time, before any inference, with full diagnostics (what / where / expected /
how-to-fix):

- every role named in a `when:` clause exists in `column_roles`;
- every rule `type:` and the `default:` (unless `none`) is in `supported_document_types`;
- `rules` is non-empty and every `when:` uses only known keys.

The change should land on a dedicated branch, be developed test-first as a behaviour-preserving
refactor (read YAML as a no-op first, then enable the new types), with the remote classification
smoke run before merge to confirm no regression on existing bank/receipt/invoice cases.

---

## 7. Recommendation

Proceed with the issue's intent, implemented as the YAML-driven rule set in §4 rather than the
literal snippet. This removes the Python gate on detection, keeps configuration in one auditable
place, and avoids the four defects in §3. A detailed implementation plan is available on request.
