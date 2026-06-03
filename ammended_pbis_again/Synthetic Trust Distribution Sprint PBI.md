PBI Title: Synthetic Trust Distribution Dataset Generation

Description: Generate a synthetic document set for the trust distribution flow that
provides the ground-truth foundation for a downstream cross-document linking baseline
(see the Synthetic Trust Distribution Linking Baseline PBI).
Each case is a quad of four internally-consistent documents — Trust Tax Return
(NAT 0660), Distribution Statement (custom layout), Trust Income Schedule, and
Beneficiary ITR (NAT 2541) — joined by five scalar linking fields: beneficiary TFN,
trust ABN, share of net income, franking credit, and capital gain component. We
extend the existing `Synthetic_Doc_Generation` pipeline with four new NRO document
types, render them from scratch with PIL (NRO PDFs are design reference only), and
produce 50 quads (200 documents) on a 70/30 compliant/non-compliant split. List
fields are excluded — the LMM performs poorly on coherent list extraction. Real
taxpayer documents cannot be used; all data is synthetic.

Trusts are an explicit NRO compliance priority for 2025–26 (Areas of focus 2025–26,
QC 103062), which makes this the highest-value first flow for the baseline.

Acceptance Criteria:
- Four new document types registered in the pipeline with field definitions, layouts,
  and PIL renderers that produce clean PNGs
- 50 synthetic quads (200 ground-truth entries) generated: 35 compliant + 15
  non-compliant, with documented discrepancies
- Linking ground truth recording how each quad's four documents join on the five
  scalar fields, plus compliance status and discrepancy type

---
Task 1: Scaffold Config, Data, and Field Definitions

Lay the foundation the renderers depend on. Add TFN generation and validation to
`generators/common.py` using the NRO checksum algorithm (9 digits, weights
[1,4,3,7,5,8,6,9,10], sum mod 11 == 0), mirroring the existing ABN helpers. Add
trust-specific reference data to `config/data_pools.yml` (trust names, trustee names,
beneficiary names). Add the four new document types and their required fields to
`config/field_definitions.yml` — including the new `tfn` field type — and register
them in `config/generation_config.yml` and `generators/schema.py`. No rendering yet;
this task makes the new document types known to the pipeline and validatable.

Task 2: Build Layouts and PIL Renderers

Create the four layout YAML files in `config/layouts/` (trust_returns,
distribution_statements, trust_income_schedules, beneficiary_itrs) using the existing
section-based from-scratch pattern. Build the four matching renderer modules in
`generators/` following the `invoice.py` pattern (A4 PIL image at 300 DPI, section
iteration, reuse of `common.py` draw utilities). Use the downloaded NRO PDFs as design
reference for field labels and item numbers only. Register the renderers in
`generators/pipeline.py`. Outcome: each document type renders to a clean PNG from a
ground-truth entry.

Task 3: Seed the Synthetic Dataset and Linking Ground Truth

Build `scripts/seed_trust_distributions.py` to generate 50 quads (200 ground-truth
entries) on a 70/30 split: 35 compliant cases where all five fields reconcile across
all four documents, and 15 non-compliant cases with one injected discrepancy each —
under-reported income (~5), over-claimed franking (~4), missing capital gains (~3),
and Trust Return vs Distribution Statement mismatch (~3). Build
`scripts/seed_trust_distribution_links.py` to emit `trust_distribution_links.yml`
recording each quad's linked images, the five linking-field values, compliance status,
and discrepancy type. Use deterministic seeding (seed=42).
