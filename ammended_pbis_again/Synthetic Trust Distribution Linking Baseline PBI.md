PBI Title: Synthetic Trust Distribution Linking Baseline

Description: Establish a baseline for cross-document transaction linking over the
synthetic trust distribution quads using scalar fields. This is a one-week follow-on
to the Synthetic Trust Distribution Dataset Generation PBI, which produces the 50 quads
(200 documents) and the linking ground truth this work depends on. Each case is a quad
of four internally-consistent documents — Trust Tax Return (NAT 0660), Distribution
Statement (custom layout), Trust Income Schedule, and Beneficiary ITR (NAT 2541) —
joined by five scalar linking fields: beneficiary TFN, trust ABN, share of net income,
franking credit, and capital gain component. The existing linking module handles
two-document pairs only and must be extended to four-document quads before the baseline
can be measured. List fields are excluded — the LMM performs poorly on coherent list
extraction. Real taxpayer documents cannot be used; all data is synthetic.

Depends on: Synthetic Trust Distribution Dataset Generation PBI (synthetic quads and
`trust_distribution_links.yml` ground truth must exist before this work begins).

Acceptance Criteria:
- Linking evaluator extended from two-document pairs to four-document quads, with TFN
  normalization, and run end-to-end on the synthetic set
- Baseline metrics reported: per-field extraction F1, link accuracy (all five fields
  reconcile across all four documents), compliance detection rate, false positive rate,
  and discrepancy classification accuracy
- Per-field analysis identifying which fields extract reliably vs. which fail, to guide
  prompt tuning next sprint

---
Task 1: Extend Linking Evaluator, Run Baseline, and Report Metrics

Extend the linking module to handle four-document quads (the existing matcher handles
two-document pairs only): add TFN normalization to `linking/transaction_matcher.py`
and quad-linking support to `linking/link_validator.py`. Run the pipeline end-to-end
(`validate` → `generate` → `derive`), execute the linking evaluator over the synthetic
set, and produce a summary report covering per-field extraction F1, link accuracy
(all five fields reconcile across all four documents), compliance detection rate, false
positive rate, and discrepancy classification accuracy. Identify which fields extract
reliably vs. which fail, to guide prompt tuning next sprint.
