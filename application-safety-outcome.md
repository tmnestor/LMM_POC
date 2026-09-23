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

---

# PTM application — "Has a reputational review been undertaken?"

**Prepared:** 2026-09-23. Every external figure below was read from its source
that day; download and vote counts move daily, so quote the date with them.

## Paste-ready answer

A reputational review was undertaken on 2026-09-23 against four questions.

**1. Is the developer organisation reputable?** Yes. The model is developed and
published by Google DeepMind. Gemma is Google's open-weights model family, now
in its fourth generation (Gemma 1, February 2024 → Gemma 4, June 2026), with a
published technical report (arXiv 2607.02770 — a preprint, not peer-reviewed)
and a documented safety-evaluation process (see the scanning answer above).
Both the base model and the requested quantised checkpoint are released under
the Apache 2.0 licence, verified on the model cards.

**2. Is a reputable third-party validation available?** Partially — for the
base model, not for the requested checkpoint.
- *Arena (arena.ai, formerly LMArena)* — crowd-sourced blind pairwise human
  preference. On the Vision leaderboard `gemma-4-31b` ranks **38th with a
  score of 1261 ± 6 from 35,713 votes**; the board's top score is 1310, held by
  proprietary frontier models. This is the largest independent human evaluation
  of the model, but it measures general vision-chat preference, not extraction.
- *Artificial Analysis* — independent evaluator. Lists Gemma 4 31B at
  Intelligence Index 19, "#13 of 142" models, with the index flagged as
  *estimated*; no document-benchmark scores are exposed.
- *OmniDocBench (OpenDataLab)* — an independent document-parsing benchmark.
  The vendor reports **0.131 average edit distance on v1.5** (Gemma 3 27B:
  0.365), i.e. a self-reported result on a third-party benchmark. The
  benchmark maintainers' own leaderboard lists no Gemma 4 model, so no
  independently run figure exists.
- *IDP Leaderboard* (OCR / table / key-information-extraction) — lists only
  the small Gemma 4 E4B (rank 23) and E2B (rank 26) variants; the 31B is absent.
- *Engineering validation* — the model has upstream support in vLLM (the
  serving engine used in production, version 0.29.0) and in Ollama/llama.cpp.
  This evidences community integration and testing, not accuracy.
- **No third-party evaluation of the quantised W4A16 checkpoint itself was
  found.** All figures above are for the bf16 base model; the vendor states
  the quantised weights "preserve similar quality" without figures. Our own
  165-document benchmark (mean F1 0.987) is the only measurement on the exact
  checkpoint requested.

**3. Has the model been adopted by other reputable users?** Yes, at scale.
- Hugging Face, base model `google/gemma-4-31B-it`: **9,158,522 downloads in
  the last 30 days**, 3.9k likes; 256 fine-tunes, 315 quantisations, 305
  adapters and 47 merges derived from it by the community.
- Hugging Face, requested checkpoint `google/gemma-4-31B-it-qat-w4a16-ct`:
  **4,841,066 downloads all-time** (480,780 in the last 30 days), 68 likes.
- Ollama `gemma4` library: 25.5M pulls across the family, with a `31b` tag.
- Arena: 35,713 human pairwise votes on the 31B alone.
Download counts are file requests, not distinct users, so they evidence breadth
of use rather than a user count.

**4. Does it hold a leaderboard position on an information-extraction
benchmark?** No. As of 2026-09-23 Gemma 4 31B appears on neither of the two
public document-extraction leaderboards checked (IDP Leaderboard; OmniDocBench
leaderboard). The vendor reports two document-understanding scores in the
technical report — InfographicVQA **92.0** (Gemma 3 27B: 70.6) and OmniDocBench
1.5 **0.131** edit distance — and the nearest independent placement is the
Arena Vision rank above, which is general-purpose. The information-extraction
evidence is therefore our own benchmark: on 165 documents with the production
17-field schema, the requested checkpoint scored mean F1 0.987 against 0.860
for the model currently in production and 0.937 for a same-size competitor
under identical conditions.

## Sources

| claim | source (read 2026-09-23) |
| --- | --- |
| developer, report, licence | arXiv 2607.02770 (Google DeepMind); https://huggingface.co/google/gemma-4-31B-it ; https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct — both `apache-2.0` |
| Arena Vision rank 38, 1261 ± 6, 35,713 votes; top 1310 | https://arena.ai/leaderboard/vision |
| Artificial Analysis index 19 (estimated), #13/142 | https://artificialanalysis.ai/models/gemma-4-31b |
| InfographicVQA 92.0; OmniDocBench 1.5 0.131; Gemma 3 27B 70.6 / 0.365 | arXiv 2607.02770 tech report tables; base model card benchmark table |
| OmniDocBench leaderboard has no Gemma 4 row | https://github.com/opendatalab/OmniDocBench README |
| IDP Leaderboard: E4B rank 23, E2B rank 26, 31B absent | https://idp-leaderboard.org/ |
| HF downloads / likes / model tree | Hub model pages and `https://huggingface.co/api/models/google/gemma-4-31B-it-qat-w4a16-ct?expand[]=downloadsAllTime` |
| Ollama 25.5M pulls, `31b` tag | https://ollama.com/library/gemma4 |
| vLLM support | production runs on vLLM 0.29.0; `conda_envs/vllm_env4.yaml` |

## Things deliberately not claimed

- No peer review of the technical report — it is an arXiv preprint.
- No third-party figure for the W4A16 checkpoint — none exists.
- No IE leaderboard position — the model is absent from both boards checked.
- The Artificial Analysis index is marked *estimated* by that site and is quoted as such.
