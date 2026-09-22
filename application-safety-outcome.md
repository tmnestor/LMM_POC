# PTM application — "Has the PTM been scanned…? Provide outcome details"

**Model requested:** `google/gemma-4-31B-it-qat-w4a16-ct` (Apache 2.0)
**Role:** Information Extraction component of the production document-understanding pipeline
**Prepared:** 2026-09-22

Three strands, kept separate on purpose: what the vendor scanned, what we scanned,
and what nobody has. A reviewer should be able to see which is which.

---

## Paste-ready answer

**1. Vendor scanning (Google DeepMind).** The base model, Gemma 4 31B-it, was
evaluated by the vendor with automated and human safety testing across six
prohibited-content categories: child sexual abuse material and exploitation;
dangerous content; sexually explicit content; hate speech; harassment and
violence encouragement; and personal information disclosure (Gemma 4 Technical
Report, arXiv 2607.02770, §5.2–5.3; Hugging Face model card, "Ethics and
Safety"). The reported outcome is qualitative: "minimal policy violations" and
"major improvements in all categories of content safety relative to previous
Gemma models". The vendor publishes no per-category violation rates and no
named bias, toxicity or stereotype benchmark scores for this model. Training
data was "filtered for removal of certain personal information". **The
quantisation-aware-trained W4A16 checkpoint being requested was not separately
safety-evaluated by the vendor; its model card defers to the base model** and
states that quantisation "preserves similar quality to bfloat16" without
figures.

**2. Our scanning — inaccuracies, measured on the exact checkpoint.** On
2026-09-22 the requested checkpoint was benchmarked against 165 synthetic
documents (55 invoices, 55 receipts, 55 bank statements) using the production
17-field extraction schema and a per-field answer key: mean F1 0.987, median
0.989; twelve of seventeen fields at 1.000; document classification 165/165;
no whole-document failures. The residual error modes are enumerated rather
than averaged away: bank-statement transaction amounts 0.957 and dates 0.968
(partial misreads of dense tables on a small number of statements), business
address 0.970 (a punctuation policy in the scorer, not a misread), line-item
descriptions 0.971. Reasoning/"thinking" output is disabled and was verified
absent from all 165 model outputs. On identical conditions the checkpoint
outperformed the model currently in production (InternVL3.5-8B, mean F1 0.860)
by 12.7 points, and a same-size competitor (Qwen 3.8 27B, 0.937) by 5.0. This
benchmark is the only quality evidence that exists for the quantised weights;
the vendor provides none.

**3. Not scanned — stated gaps.** No bias, stereotype, toxicity or adversarial
(prompt-injection via document content) testing has been performed on this
checkpoint by us, and the vendor's testing is qualitative. Exposure is bounded
by the deployment design: the model receives a document image and a fixed
extraction prompt; its output is parsed into the 17-field schema and anything
outside that schema is discarded (verified: off-schema or deliberative text is
reduced to `NOT_FOUND`, never surfaced); no free-form model text reaches any
user. Residual risks that this design does not remove: (a) an adversarial
document could attempt to steer extracted field values — untested; (b)
recognition accuracy on personal names and addresses may vary with cultural
origin — untested on this checkpoint.

**4. Privacy.** The PTM runs on-premises in the air-gapped production
environment; no data leaves that environment. The weights are static — no
training or fine-tuning on production data occurs — so production data cannot
be memorised into the model. The documents processed contain personal
information by design (names, addresses, identifiers, transactions), and
extracting it is the pipeline's function; the PTM introduces no new data flow,
and the pipeline's existing controls on access and retention apply unchanged.

**Recommended before go-live (optional, low cost on the existing harness):** a
targeted name-diversity subset to address gap 3(b), and a small
document-injection test to address gap 3(a).

---

## Sources

| claim | source |
| --- | --- |
| six harm categories; "minimal policy violations"; no numbers | Gemma 4 Technical Report, arXiv 2607.02770 (Google DeepMind, 2026-06-19), §5.2 Policies, §5.3 Safety Evaluations, §5.4 Ethical Considerations |
| five categories, "major improvements… relative to previous Gemma", PII-filtered training data, mitigations | https://huggingface.co/google/gemma-4-31B-it — "Ethics and Safety", "Ethical Considerations and Risks" |
| QAT card defers to base; "similar quality to bfloat16", no figures; Apache 2.0 | https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct |
| our benchmark figures | `plans/2026-09-22-gemma4-31b-vs-qwen38-27b-RESULTS.md`; artefacts in `evaluation_data/output_gemma31b_20260921/` |
| thinking verified absent | 0 of 165 raw responses contain a think tag (checked 2026-09-22) |
| off-schema output discarded | observed on the Qwen run: deliberative prose parsed to `NOT_FOUND` |

## Things deliberately not claimed

- No downstream validation of field values beyond the parser — none was verified.
- No specific retention or access-control mechanism — "existing controls apply" only.
- No vendor number for any bias/toxicity benchmark — the report names none.
- The benchmark corpus is synthetic, and says so.
