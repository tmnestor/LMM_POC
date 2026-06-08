# Transaction-Linking Trace Diagnostics

Quick `jq` recipes for diagnosing the receipt→bank-statement **VLM fallback** by
reading its raw prompt/response trace. Used to investigate why the fallback's
recovery rate drops (e.g. the image-first / prefix-cache change regressing
linking recall).

## 1. Capture a trace

In `config/run_config.yml`:

```yaml
tracing:
  raw_prompts: true   # persist every VLM prompt+response to JSONL
  path: none          # none -> <output_dir>/raw_prompt_trace.jsonl
```

Then re-run the linking stage. Cheapest path (reuses existing
`cleaned_extractions.jsonl`; only the matcher + the queued fallback calls run):

```bash
python -m stages.transaction_link \
  --extractions ../synthetic_transaction_linking/output/cleaned_extractions.jsonl \
  --output      ../synthetic_transaction_linking/output/transaction_links.jsonl \
  --data-dir    ../synthetic_transaction_linking \
  --config      config/run_config.yml --model internvl3-vllm
```

Or via the orchestrator (re-runs classify — light — but **skips** the expensive
extract via resume): `KFP_TASK=run_transaction_link bash entrypoint.sh`.

Each trace line is `{image_name, pipeline, label, prompt, raw_response, tokens}`.
Linking-fallback calls are identified by the `RECEIPT TO LOCATE` marker in the
prompt (the per-receipt query suffix).

## 2. Inspect (`jq`)

> `cd` into the directory containing `raw_prompt_trace.jsonl` first.

**Counts — the fastest signal (run this first):**

```bash
jq -s '
  map(select(.prompt|contains("RECEIPT TO LOCATE"))) |
  {linking_calls: length,
   found:     (map(select(.raw_response|contains("MATCHED_TRANSACTION: FOUND"))) | length),
   not_found: (map(select((.raw_response|contains("MATCHED_TRANSACTION: FOUND"))|not)) | length),
   any_think: (map(select(.raw_response|contains("<think>"))) | length)}
' raw_prompt_trace.jsonl
```

**Failing calls — image + token count + first 600 chars of the response:**

```bash
jq -c 'select((.prompt|contains("RECEIPT TO LOCATE")) and ((.raw_response|contains("MATCHED_TRANSACTION: FOUND"))|not)) | {image_name, tokens, resp: .raw_response[0:600]}' raw_prompt_trace.jsonl
```

**One full prompt — confirm the prefix+query concatenation is well-formed:**

```bash
jq -r 'select(.prompt|contains("RECEIPT TO LOCATE")) | .prompt' raw_prompt_trace.jsonl | head -120
```

## 3. Interpreting the counts

| Observation | Likely cause |
|-------------|--------------|
| `any_think` high, or failing responses near the `max_tokens` cap | **Thinking/truncation** — `<think>` spend eats the output budget before the answer block lands ([POSTMORTEM_internvl35_thinking.md](../POSTMORTEM_internvl35_thinking.md)). Fix in prompt simplicity / `<think>` defense. |
| `linking_calls` ≠ the number the matcher queued | **Loop/queue bug** — the per-statement fallback is dropping receipts. |
| ~all failing as well-formed `MATCHED_TRANSACTION: NOT_FOUND`, no `<think>` | **Genuine recall loss** — the model isn't finding rows under the current ordering (image-first / key-as-suffix). Isolate by flipping one variable. |
| malformed / wrong block / `RECEIPT_STORE == TRANSACTION_DESCRIPTION` echo | **Construction/parse bug** — check `common/vlm_linker.py` build + parse. |
