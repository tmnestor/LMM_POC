# Plan: Bank Statement Benchmark — All Models

## Overview

Create `scripts/run_bank_all.sh` to benchmark all models on the 15 synthetic
bank statements, with raw response saving for post-hoc analysis. Requires
adding `--save-responses` to `cli.py` first, since bank statements run through
the CLI pipeline (not a standalone benchmark script like WildReceipt).

## Background

- Bank statements use **multi-turn sequential extraction** via `BankStatementAdapter`
- The adapter takes a `generate_fn` callable (from each processor's `.generate()` method)
- `cli.py` orchestrates: model load -> processor creation -> `BankStatementAdapter` -> extraction -> evaluation
- Current bank results in `docs/hf_vs_vllm.md` only cover InternVL3 (8B/14B/38B) HF + vLLM
- Data lives on remote at `evaluation_data/bank/` with `ground_truth_bank.csv` (not in git)

## Changes Required

### 1. Add `--save-responses` to `cli.py`

Add a CLI flag that saves raw model responses to a JSONL file per image. This
requires hooking into the extraction pipeline to capture the raw text before
parsing.

```python
save_responses: bool = typer.Option(
    False,
    "--save-responses",
    help="Save raw model responses to JSONL in output dir.",
)
```

**Where to capture responses**: The `BatchDocumentProcessor.process_batch()`
method calls `processor.process_document_aware()` for standard documents and
`bank_adapter.extract_bank_statement()` for bank statements. Both ultimately
call `processor.generate()`.

**Implementation approach**: Add an optional response logger that wraps the
processor's `generate()` method. Each call gets logged with `(image_name,
prompt, raw_response)` to a JSONL file.

```python
# In cli.py, after processor creation:
if save_responses:
    responses_path = output_dirs["base"] / f"responses_{config.model_type}.jsonl"
    responses_file = responses_path.open("w")

    original_generate = processor.generate

    def _logging_generate(image, prompt, max_tokens=1024):
        response = original_generate(image, prompt, max_tokens)
        responses_file.write(
            json.dumps({
                "prompt": prompt[:200],  # truncate for readability
                "response": response,
            }, ensure_ascii=False) + "\n"
        )
        responses_file.flush()
        return response

    processor.generate = _logging_generate
```

This is non-invasive — wraps the existing `generate()` without modifying any
processor code. Works for all models (HF, vLLM, bank multi-turn).

### 2. Verify model compatibility with bank statements

Bank extraction requires `processor.generate()` which is the abstract method
on `BaseDocumentProcessor`. The `cli.py` creates processors via
`registration.processor_creator(...)`.

**Models with working processor_creator** (have DocumentProcessor subclass):
- `internvl3` / `internvl3-14b` / `internvl3-38b` — `DocumentAwareInternVL3Processor`
- `llama` — `DocumentAwareLlamaProcessor`
- `qwen3vl` — `DocumentAwareQwen3VLProcessor`
- `qwen35` — `DocumentAwareQwen35Processor`
- `nemotron` — `DocumentAwareNemotronProcessor`
- `internvl3-vllm` / `internvl3-14b-vllm` / `internvl3-38b-vllm` — `DocumentAwareVllmProcessor`
- `qwen3vl-vllm` / `qwen35-vllm` — `DocumentAwareVllmProcessor`
- `llama4scout` — `DocumentAwareLlama4Processor`
- `llama4scout-w4a16` — `DocumentAwareVllmProcessor`
- `gemma4` — `DocumentAwareVllmProcessor`

**Models WITHOUT working processor_creator** (returns None):
- `granite4` — `_granite4_vision_processor_creator` returns `None`

**Action**: Either skip `granite4` in the bank script, or implement a
`DocumentAwareGranite4Processor` class. Implementing the processor is the
correct long-term fix but adds scope. For v1, skip granite4 and add a TODO.

### 3. Create `scripts/run_bank_all.sh`

Same pattern as `run_wildreceipt_all.sh`: run each model sequentially, commit
and push results after each job.

```bash
#!/bin/bash
# =============================================================================
# Bank Statement Benchmark — Run All Models Unattended
# =============================================================================

set -o errexit
set -o pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="evaluation_data/bank"
GROUND_TRUTH="evaluation_data/bank/ground_truth_bank.csv"
OUTPUT_BASE="evaluation_data/output"

# ... log() and commit_results() same as wildreceipt script ...

run_job() {
  local env_name="$1"
  local model="$2"
  local output_dir="$3"
  shift 3
  local extra_env=("$@")

  # ... same conda activate + error handling pattern ...

  python cli.py \
    --model "$model" \
    --data-dir "$DATA_DIR" \
    --ground-truth "$GROUND_TRUTH" \
    --output-dir "$output_dir" \
    --bank-v2 \
    --save-responses

  # ... commit_results ...
}

# --- HuggingFace models ---
run_job LMM_POC_IVL3.5 internvl3 \
  "$OUTPUT_BASE/bank_ivl35_8b"

run_job LMM_POC_IVL3.5 internvl3-14b \
  "$OUTPUT_BASE/bank_ivl35_14b"

run_job LMM_POC_IVL3.5 internvl3-38b \
  "$OUTPUT_BASE/bank_ivl35_38b"

# --- vLLM models ---
run_job LMM_POC_VLLM internvl3-vllm \
  "$OUTPUT_BASE/bank_ivl35_8b_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM internvl3-14b-vllm \
  "$OUTPUT_BASE/bank_ivl35_14b_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM internvl3-38b-vllm \
  "$OUTPUT_BASE/bank_ivl35_38b_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM llama4scout-w4a16 \
  "$OUTPUT_BASE/bank_llama4scout" \
  env VLLM_LOGGING_LEVEL=WARNING

# --- Other HF models ---
run_job LMM_POC_NEMOTRON nemotron \
  "$OUTPUT_BASE/bank_nemotron"

run_job LMM_POC_QWEN35 qwen35 \
  "$OUTPUT_BASE/bank_qwen35"

run_job LMM_POC_VLLM qwen35-vllm \
  "$OUTPUT_BASE/bank_qwen35_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM gemma4 \
  "$OUTPUT_BASE/bank_gemma4" \
  env VLLM_LOGGING_LEVEL=WARNING

# NOTE: granite4 skipped — no DocumentProcessor (cli.py needs processor_creator)
```

### 4. Test ordering considerations

- Run InternVL3 8B HF first — known to work, validates the pipeline
- vLLM models after HF models — if vLLM env has issues, HF results are safe
- Gemma4 last among vLLM models — largest, most likely to OOM

## Risks

1. **Multi-turn context length**: Bank statements use 10-20 turns per image.
   Models with low `max_model_len` (vLLM) or limited context windows may
   truncate later turns. This was the suspected cause of the 14B/38B accuracy
   drop in existing results.

2. **Models not tested on bank statements**: Nemotron, Qwen3.5, Gemma 4,
   Llama 4 Scout have never been run on bank data. Their `generate()` method
   should work (it's the standard interface), but the multi-turn prompts were
   designed for InternVL3 — other models may produce different response
   formats that the bank extractor can't parse.

3. **`--save-responses` wrapper**: The `generate()` wrapper approach is clean
   but adds a function call overhead per inference. Negligible for bank
   statements (15 images, 10-20 turns each = ~300 calls total).

## Implementation Order

1. Add `--save-responses` to `cli.py` (with `generate()` wrapper)
2. Test manually: `python cli.py --model internvl3 --data-dir evaluation_data/bank --ground-truth evaluation_data/bank/ground_truth_bank.csv --output-dir /tmp/bank_test --bank-v2 --save-responses`
3. Create `scripts/run_bank_all.sh`
4. Run on sandbox with `-n 2` (2 images) to verify all models load and produce output
5. Full run on production
