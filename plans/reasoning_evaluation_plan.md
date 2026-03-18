# Plan: Measure and Utilise Transaction Linking Reasoning

## Problem Statement

The transaction linking pipeline (`staged_transaction_linking.yaml`) produces a `REASONING` field
per receipt match, but this field is:
- Not captured in the model results CSV
- Not scored by `evaluate_transaction_linking.ipynb`
- Not leveraged for fuzzy/partial matching

Meanwhile, strict matching fails on real-world cases: currency differences, credit card surcharges,
and merchant acquirer names that differ from invoice supplier names.

---

## Phase 1: Capture REASONING in Output (prerequisite)

**Goal**: Store the model's reasoning so it can be measured and used.

### Changes

1. **`staged_transaction_linking.ipynb`** — update the Stage 2b response parser to extract
   `CONFIDENCE` and `REASONING` fields and write them to the model results CSV.

2. **`evaluation_data/transaction_link_model_results.csv`** — add `CONFIDENCE` and `REASONING`
   columns.

3. **`common/field_config.py` / `field_definitions.yaml`** — add `CONFIDENCE` and `REASONING`
   to the `transaction_link` field list (excluded from F1 scoring, but captured for analysis).

---

## Phase 2: Add PARTIAL_MATCH + Structured MISMATCH_TYPE to Prompt

**Goal**: Replace the binary FOUND/NOT_FOUND with a three-outcome model that explains *why*
matches fail, using machine-readable categories.

### Changes

1. **`prompts/staged_transaction_linking.yaml`** — update `match_to_statement` prompt:
   - Add `PARTIAL_MATCH` as a valid outcome for `MATCHED_TRANSACTION`
   - Define `PARTIAL_MATCH` criteria: at least 2 of 3 match criteria met, or 1 criterion
     fails in an explainable way (surcharge, currency, acquirer name)
   - Add `MISMATCH_TYPE` output field with enum values:
     `NONE | CURRENCY | SURCHARGE | NAME_MISMATCH | DATE_DELAY | AMOUNT_OTHER`
   - Keep `REASONING` as freetext for additional context

2. **Updated prompt output format**:
   ```
   --- RECEIPT 1 ---
   MATCHED_TRANSACTION: FOUND | PARTIAL_MATCH | NOT_FOUND
   TRANSACTION_DATE: ...
   TRANSACTION_AMOUNT: ...
   TRANSACTION_DESCRIPTION: ...
   RECEIPT_STORE: ...
   RECEIPT_TOTAL: ...
   CONFIDENCE: HIGH | MEDIUM | LOW
   MISMATCH_TYPE: NONE | CURRENCY | SURCHARGE | NAME_MISMATCH | DATE_DELAY | AMOUNT_OTHER
   REASONING: brief explanation
   ```

---

## Phase 3: Score Reasoning Quality in Evaluation

**Goal**: Measure whether the model's reasoning and partial matches are useful.

### Changes

1. **`evaluation_data/transaction_link_ground_truth.csv`** — add columns:
   - `EXPECTED_MATCH_STATUS`: `FOUND | PARTIAL_MATCH | NOT_FOUND`
   - `EXPECTED_MISMATCH_TYPE`: ground truth mismatch category
   - `REASONING_QUALITY`: human-annotated `USEFUL | PARTIALLY_USEFUL | NOT_USEFUL`

2. **`evaluate_transaction_linking.ipynb`** — add new scoring cells:
   - **Match status accuracy**: categorical match of `MATCHED_TRANSACTION` vs
     `EXPECTED_MATCH_STATUS`
   - **Mismatch type accuracy**: categorical match when status is `PARTIAL_MATCH` or `NOT_FOUND`
   - **Reasoning quality distribution**: bar chart of `REASONING_QUALITY` values
   - **Confidence calibration**: accuracy broken down by `CONFIDENCE` level
     (are HIGH-confidence matches more accurate than LOW?)
   - **Mismatch type confusion matrix**: predicted vs actual mismatch categories

---

## Phase 4 (Future): Two-Pass Fuzzy Re-matching

**Goal**: Feed NOT_FOUND results back for a second pass that accounts for known mismatch patterns.

This phase depends on Phase 2/3 data showing that PARTIAL_MATCH cases are real matches
the strict pass misses. Not implemented now — revisit once we have enough scored data
to validate the approach.

### Sketch

- Collect all `NOT_FOUND` and `PARTIAL_MATCH` results from pass 1
- Build a second prompt: "Re-examine these unmatched receipts against the bank statement,
  accounting for: currency conversion, surcharges (1-3%), merchant acquirer names,
  and extended processing delays"
- Score pass-2 matches separately to measure incremental value

---

## Implementation Order

1. Phase 2 first (prompt changes) — this is the core improvement
2. Phase 1 alongside (capture the new fields in CSV output)
3. Phase 3 after a model run produces data with the new fields
4. Phase 4 deferred — needs Phase 3 results to justify

## Files Touched

| File | Phase | Change |
|------|-------|--------|
| `prompts/staged_transaction_linking.yaml` | 2 | Add PARTIAL_MATCH, MISMATCH_TYPE |
| `staged_transaction_linking.ipynb` | 1 | Parse + store CONFIDENCE, REASONING, MISMATCH_TYPE |
| `evaluation_data/transaction_link_model_results.csv` | 1 | Add new columns |
| `evaluation_data/transaction_link_ground_truth.csv` | 3 | Add annotation columns |
| `evaluate_transaction_linking.ipynb` | 3 | New scoring + visualisation cells |
| `common/field_definitions.yaml` | 1 | Add new fields to transaction_link |
