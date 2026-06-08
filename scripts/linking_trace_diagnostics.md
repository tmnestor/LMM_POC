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

### ⚠️ Gotcha: the linking trace is NOT in the run's output dir

`transaction_link._build_processor` builds its config with `output_dir = data_dir`,
so the trace path defaults to **`<data_dir>/raw_prompt_trace.jsonl`** — one level
ABOVE the run's `output/` dir. A `run_transaction_link` produces **two** trace
files:

- `<output_dir>/raw_prompt_trace.jsonl` — the **extract/classify** calls
  (`pipeline`/`label` are null here; prompt looks like *"Look at this document
  image and answer each question…"*). On the synthetic set this is ~165 lines.
- `<data_dir>/raw_prompt_trace.jsonl` — the **linking fallback** calls (~33 lines).
  **This is the one you want.**

Locate them by size:

```bash
find <data_dir> -name raw_prompt_trace.jsonl -exec wc -l {} \;
```

The small (~matcher-queued count) file is the linking trace. The queries below
assume you point them at THAT file.

## 2. Trace record schema

Each line is a JSON object with these keys (confirmed via `jq 'keys'`):

```
image_name, model, pipeline, label, prompt, raw_response, prompt_tokens, completion_tokens
```

Note: `pipeline` and `label` are populated inconsistently (null on the extract
path), so **isolate the linking calls by file, not by `.pipeline`/`.label`**.
Token counts are `.prompt_tokens` / `.completion_tokens` (there is no `.tokens`).

## 3. Inspect (`jq`)

Point these at the linking trace file (here `../raw_prompt_trace.jsonl` when you
are inside `output/`).

**Counts + truncation signal — run this first:**

```bash
jq -s '{calls: length,
  found:     (map(select(.raw_response|contains("MATCHED_TRANSACTION: FOUND")))|length),
  not_found: (map(select((.raw_response|contains("MATCHED_TRANSACTION: FOUND"))|not))|length),
  any_think: (map(select(.raw_response|contains("<think>")))|length),
  max_completion_tokens: (map(.completion_tokens)|max)}' ../raw_prompt_trace.jsonl
```

**Failing calls — image, completion length, first 500 chars of the response:**

```bash
jq -c 'select((.raw_response|contains("MATCHED_TRANSACTION: FOUND"))|not) | {image_name, completion_tokens, resp: .raw_response[0:500]}' ../raw_prompt_trace.jsonl | head -3
```

**One full prompt — confirm the prefix+query concatenation is well-formed:**

```bash
jq -r '.prompt' ../raw_prompt_trace.jsonl | head -60
```

**Full prompt+response pairs for failing calls — the decisive artifact** (shows
whether receipt details actually substituted and whether the model says "no row
matches that amount" (recall loss) vs "no amount given / no table" (construction
bug)):

```bash
jq -r 'select((.raw_response|contains("MATCHED_TRANSACTION: FOUND"))|not) | "########## " + .image_name + "  (" + (.completion_tokens|tostring) + " tok)\n--- PROMPT ---\n" + .prompt + "\n--- RESPONSE ---\n" + .raw_response + "\n"' ../raw_prompt_trace.jsonl | head -140
```

## 4. Interpreting the counts

`vlm_max_tokens` for linking is 4096 (`run_config.yml` → `linking.vlm_max_tokens`).

| Observation | Likely cause |
|-------------|--------------|
| `any_think > 0`, or `max_completion_tokens` near 4096 | **Thinking/truncation** — `<think>` spend eats the output budget before the answer block lands ([POSTMORTEM_internvl35_thinking.md](../POSTMORTEM_internvl35_thinking.md)). Fix in prompt simplicity / `<think>` defense. |
| `calls` ≠ the number the matcher queued | **Loop/queue bug** — the per-statement fallback is dropping receipts. |
| ~all failing as well-formed `MATCHED_TRANSACTION: NOT_FOUND`, no `<think>` | **Genuine recall loss** — the model isn't finding rows under the current ordering (image-first / key-as-suffix). Isolate by flipping one variable. |
| malformed / wrong block / `RECEIPT_STORE == TRANSACTION_DESCRIPTION` echo | **Construction/parse bug** — check `common/vlm_linker.py` build + parse. |
