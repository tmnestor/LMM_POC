# Postmortem: InternVL3.5 "thinking mode" corrupted document classification

- **Date:** 2026-06-04
- **Status:** Resolved and verified (38 misclassifications → 0)
- **Component:** Detection / classify stage of `run_transaction_link` (receipt→bank linking)
- **Model:** InternVL3.5-8B (Qwen2.5 LLM backbone), served via the **offline vLLM `LLM` engine** in-process (`engine.chat()`), not `vllm serve`.
- **Fix commits:** `489638d` (made it worse) → `5e7914f` (made it worse a different way) → **`baa4b4e` (the fix)**. Parser/diagnostic support also in `489638d`/`5e7914f`.

---

## TL;DR

The detection step started emitting chain-of-thought wrapped in `<think>…</think>` and **ran out of tokens before producing the answer**, so the `COLUMNS/PAID/ROWS` classification fields were never parsed → bank statements and invoices were mislabelled (38 of 165 docs; `BANK_STATEMENT` count ballooned 55→77).

**Two things turned thinking ON, and both were self-inflicted:**

1. **Sending a `system` message.** On this InternVL3.5 build, *any* system prompt — even one literally saying "do not show your reasoning" — enables thinking. The clean path is a **bare user message (no system role)**.
2. **A "strict OUTPUT RULES" detection prompt.** Instructions like *"Do NOT explain your reasoning. Do NOT write 'let's analyze step by step'"* **negation-primed** the model into doing exactly that. The simple 3-question prompt does not.

The fix was to **undo both**: no system message + the simple prompt. Detection now returns terse, correct answers and classifies 55/55 bank statements with 0 misclassifications.

---

## Symptom

`classifications.jsonl` responses looked like:

```
<think>
Okay, let's tackle these questions one by one. First, the columns. The document
shows a transaction table with the following column headers:
- Date
- Transaction Description
- Debits
- Credits
- Balance
... (runs out of tokens here, never closes </think>, never answers)
```

- Every response opened with `<think>` and truncated mid-thought — no `1. COLUMNS:` answer ever appeared.
- The file grew from ~88 KB (terse) to ~151 KB (verbose reasoning).

## Impact

- **38/165 documents misclassified**, in both directions (bank statements → INVOICE/RECEIPT; invoices/receipts → BANK_STATEMENT). Offsetting errors (a swap within a case) hid some from a naive per-case count.
- The linking stage is bottlenecked by extraction, which is bottlenecked by classification — so the whole `run_transaction_link` result was untrustworthy until this was fixed.

## Root cause

InternVL3.5-8B is **thinking-capable** (`<think>`/`</think>` are special tokens in its `tokenizer_config.json`) but thinking is **OFF by default**; per the HF model card it is enabled **only** by setting the system prompt to the R1 "thinking" prompt. It does **not** honour `enable_thinking` / `/no_think` (those are Qwen3 / Nemotron mechanisms).

Empirically, on the served build:

| Condition (image + detection prompt) | Result |
|---|---|
| bare user message, **simple** prompt | clean answer, no `<think>` |
| bare user message, **strict OUTPUT-RULES** prompt | `<think>` (the 09:34 run) |
| **any** system message + simple prompt | `<think>` |

So both a system message and the negation-priming prompt independently trigger thinking. Two of our own changes had introduced exactly those:

- `489638d` rewrote the detection prompt with an elaborate "OUTPUT RULES — do NOT explain your reasoning…" block (intended to stop an earlier *markdown* drift). It triggered full `<think>` reasoning instead.
- `5e7914f` added a benign system message to `VllmBackend` intended to *suppress* thinking. It did the opposite.

A second, compounding bug: while thinking, the parser's recovery logic harvested column names from the reasoning prose (e.g. "no table with headers like Date, Description, **Debit**, etc.") and promoted receipts to `BANK_STATEMENT`.

### Why the docs misled us

The HF card says "thinking is off unless you set the R1 system prompt," which we read as "a benign system prompt is safe." On this build, providing **any** system role is enough to flip it on — the content doesn't matter. The text-only smoke probe ("reply OK", no image) also did **not** reproduce it; only the image + real detection prompt did.

## The fix (commit `baa4b4e`)

1. **`models/backends/vllm_backend.py`** — `_build_messages` sends a **bare user message** for all models (no system role). Removed the `_system_message()` / `_INTERNVL_SYSTEM_MESSAGE` injection from `5e7914f`.
2. **`prompts/document_type_detection.yaml`** — reverted the `detection` prompt to the **simple** 3-question `COLUMNS / PAID / ROWS` form (no anti-CoT "OUTPUT RULES").

### Kept as defense-in-depth (harmless, still useful)

- **Parser** (`common/turn_parsers.py` `ClassificationParser._parse_enriched`): strips `<think>…</think>` (and an unterminated tail) before parsing, and recovers headers **only** from a pipe-delimited line — never from prose. So a stray thinking response can't corrupt classification.
- **`model.chat_template` knob** (`config/run_config.yml` → `PipelineConfig` → every `VllmBackend`, fail-fast validated): override the serve template if a thinking-by-default template ever appears. Default `none`.

## The live path (confirmed the fix is on it)

```
classify stage / vLLM DP classify_worker
  → create_processor()                         (common/pipeline_ops.py)
  → build_vllm_processor_creator._creator()     (model_loader.py:521)
        VllmBackend(engine, chat_template=…) wrapped in DocumentOrchestrator
  → orchestrator.detect_and_classify_document() (models/orchestrator.py:349)
  → self.generate() → self._backend.generate()  (orchestrator.py:319)
  → VllmBackend.generate() → _build_messages()   ← the fix
```

`VllmBackend` is not a `BatchInference` implementation, so detection uses the sequential `generate()` path (no batched bypass). Nothing else injects a system prompt.

## Verification

- **Smoke test** (`scripts/check_thinking.py --image …`, loads the model and runs no-system vs system): default/no-system + simple prompt → clean `1. COLUMNS: Date | … | Balance`; adding a system message → `<think>`.
- **Post-fix run** (`classifications.jsonl`, 88 KB): 165 records, **0 errors, 0 `<think>`**, **55/55 BANK_STATEMENT**, **0 misclassifications** in either direction.

## How to diagnose this next time

1. Run `scripts/check_thinking.py --model <path> --tp <N> --image <a real doc>` — **always with `--image`**; text-only won't reproduce it. (Prepend `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` to dodge the libstdc++ load-order error; `entrypoint.sh` does the same.)
2. Read the verdict: if the bare-user case is clean and the system case thinks → don't send a system message.

## Lessons / guardrails

- **For InternVL3.5: send a bare user message and a plain prompt.** No system role; no "don't reason" instructions (they backfire via negation priming).
- **`enable_thinking: False` / `/no_think` are no-ops** for the InternVL family — don't rely on them.
- **Reproduce with the real modality.** A text-only probe missed a vision-triggered behaviour.
- **A parser that "recovers" from free text is dangerous** when the model emits prose — restrict recovery to structured lines only.
- **Trust an A/B probe over the model card.** The card's "off by default" was true in spirit but the practical trigger (any system role) was only visible by testing the served build.

## Minor follow-up (non-blocking)

Post-fix, receipts/invoices all classify as `INVOICE` (0 `RECEIPT`) because the simple prompt answers `PAID: NO`. This does **not** affect linking (`stages/transaction_link.py` treats `RECEIPT` and `INVOICE` identically). Only revisit if the RECEIPT-vs-INVOICE label is scored on its own.
