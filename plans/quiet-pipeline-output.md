# Plan: Split verbose-output into Progress / Verbose / Debug tiers

## Context — what went wrong twice

I have made two bad attempts at this:

1. **Attempt 1 — change `--verbose` default from `True` to `None`.**
   Intent: let YAML's `processing.verbose: false` win. Effect: killed **all** output
   between "Model loaded!" and pipeline completion, because the codebase uses a
   single `debug=config.verbose` switch that gates *both* noisy dev prints
   (PARSING DEBUG, CONFIG DEBUG, TENSOR_DTYPE) *and* the only per-image progress
   prints the user has. Result: looks hung. Reverted in `ad702ff`.

2. **Attempt 2 — suggest adding a new `--debug` tier on top.**
   User rejected — "DEVISE A PROPER PLAN" — so I'm writing this plan first and
   NOT editing code until it's approved.

The user's verbatim requirement:
> "I do not want this verbose output"

What they explicitly do NOT want (from the screenshot they pasted):
- `DocumentOrchestrator initialized: 42 fields, batch_size=1, model_type=internvl3`
- `Auto-detected batch size: 1 (GPU Memory: 6.1GB)`
- `Generation config: max_new_tokens=2100, do_sample=False`
- `CONFIG DEBUG - detection_key='detection'`
- `Using document detection prompt: detection`
- `Prompt: What type of business document is this?\nAnswer with one of:\n- INVOICE (includes bills, quotes, esti...`
- `TENSOR_DTYPE: Using vision model dtype torch.bfloat16`
- `Model response: This is an INVOICE.`
- `PARSING DEBUG - Raw response: 'This is an INVOICE.'`
- `PARSING DEBUG - Cleaned response: 'this is an invoice.'`
- `PARSING DEBUG - Found mapping: 'invoice' -> 'INVOICE'`
- `Detected document type: INVOICE`

What they DO want kept (inferred — they *still need to see progress*):
- Model load banner + GPU status table (already always-on, not affected)
- Per-image progress showing which image is being processed and its result
- Phase headings and totals/timing
- Any error message

## Root-cause analysis: why silencing verbose also killed progress

There are **two** debug switches in `models/orchestrator.py`, both ultimately driven
by the same `config.verbose` value:

| Switch                                      | Source                                                                                            | Gates                                                                                                                                          |
|---------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `self.debug` (on the instance)              | `DocumentOrchestrator(debug=config.verbose)` from `models/model_loader.py:289, 307, 315, 456, 464` | init-time prints (lines 127-132, 159-172, 187-191), per-image prints in `process_single_image` (lines 550-555, 561-566, 572, 595), PARSING DEBUG (247-281), DETECTION ERROR traceback (405-408) |
| `verbose: bool` param passed per-call       | `processor.detect_and_classify_document(image_path, verbose=verbose)` in `stages/classify.py:131`, and similar in `extract.py` | CONFIG DEBUG (366), detection prompt echo (375-378), model response echo (383-385), detected type echo (391-393), "Processing X document" (427-428), vision-classify messages (296-308), "Using X prompt: N chars" (475-478), "Extracted N/M fields" (508-513), "Batch detecting N images" (630-632), error messages (435-437, 518-519) |

Plus `models/internvl3_image_preprocessor.py` has its own `self.debug`, set from
the same source, gating all `TENSOR_DTYPE` / `DEVICE_MOVE` / `DTYPE_CHECK` prints.

**Crucially**: the ONLY per-image progress indicators the user has today are
the `verbose=True` prints inside `detect_and_classify_document` and
`process_document_aware`. When verbose is false, both `self.debug` and the
per-call `verbose` param go quiet, and stages/classify.py's non-batch loop
(`classify.py:128-143`) has **no `logger.info` per image** — only a summary at
the end. That's why Attempt 1 looked like a hang: the user sees nothing for
9 × ~2-5s of silent inference.

## Goal

Make the default run produce **quiet, meaningful progress output** — no dev
noise, but always a visible indication that inference is progressing — and keep
two opt-in flags for engineers who need more detail.

## Target output (default run, no flags)

```
Loading internvl3 from: /efs/shared/PTM/InternVL3_5-8B
[transformers INFO lines — library's own logging, out of scope]
∴ Model loaded!

GPU Status
[table]

Phase 1/4: classify — detecting document types (9 images)...
  [1/9] receipt_001.jpg → INVOICE
  [2/9] bank_stmt_01.jpg → BANK_STATEMENT
  ...
  [9/9] travel_001.jpg → TRAVEL_EXPENSE
Phase 1/4: classify complete (12.3s).

Phase 2/4: extract — extracting fields (9 images)...
  [1/9] receipt_001.jpg (INVOICE) — 18/20 fields in 2.1s
  ...
Phase 2/4: extract complete (24.7s).

Phase 3/4: clean (CPU, no GPU)...
Phase 3/4: clean complete.

Phase 4/4: evaluate (CPU, no GPU)...
Evaluation: F1 = 0.95 (9/9 images scored)
Pipeline completed successfully.
```

## Design: three tiers of output

| Tier        | Flag                  | Contents                                                                                                                         | Mechanism                            |
|-------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| **Progress** (default, always on) | none                  | Per-image progress lines, phase start/end, timing totals, errors                                                                 | `logger.info(...)` at stage level     |
| **Verbose** | `--verbose`           | Batch-size auto-detection, generation config dump, orchestrator init line, "Processing X document", "Using X prompt", field counts, "Loading X prompt", vision-classify messages | `if config.verbose:` in orchestrator |
| **Debug**   | `--debug` *(new flag)* | PARSING DEBUG, CONFIG DEBUG, TENSOR_DTYPE, DEVICE_MOVE, detection prompt dump, raw model response dump, error tracebacks         | `if config.debug:` (new config field) |

Key property: **removing `--verbose` must never produce a silent run**. Per-image
progress is always visible because it comes from `logger.info` at the *stage*
level, not from the orchestrator's `debug` prints.

## Implementation plan

### Step 1 — Add `debug` config field (separate from `verbose`)

**File: `common/pipeline_config.py`**
- Add `debug: bool = False` alongside existing `verbose: bool = True`.
- Add `flat_config["debug"] = raw_config["processing"].get("debug")` on line ~194.
- Add `f"{ENV_PREFIX}DEBUG": ("debug", lambda x: x.lower() == "true")` on line ~233.
- Flip `verbose: bool = True` default to `verbose: bool = False` so YAML-absent
  case is quiet. (The whole point of this plan — we no longer *need* verbose on
  by default because progress is now `logger.info`, not print.)

**File: `config/run_config.yml`**
- Add `debug: false` under `processing:` next to existing `verbose: false`.

### Step 2 — Add per-image progress at the stage level (Tier A: Progress)

**File: `stages/classify.py`**
- In the non-batch loop (currently lines 128-143), add
  `logger.info("[%d/%d] %s → %s", idx+1, len(image_paths), Path(image_path).name, result["document_type"])`
  after each `detect_and_classify_document()` call.
- Similar for batch loop (line 113-127).

**File: `stages/extract.py`**
- Already has `logger.info("[%d/%d] %s: %s (%.1fs)", ...)` at lines 201-210 — keep.
- Consider enriching with field count from result if cheap.

These `logger.info` calls will render because `logging.basicConfig(level=logging.INFO)`
is set at the top of each stage's `main()`. They're independent of the
orchestrator's `debug` prints.

### Step 3 — Retarget the orchestrator's `debug` switch (Tier C: Debug)

**File: `models/model_loader.py`**
- Change `debug=config.verbose` → `debug=config.debug` on lines 289, 307, 315, 456, 464.
- Semantics: the `DocumentOrchestrator.debug` flag now means "dev-noise debug",
  not "general verbose".

**File: `models/orchestrator.py`**
- No changes to existing `if self.debug:` guards — just the upstream source shifts.
- Rename `verbose` parameter in `detect_and_classify_document`, `process_document_aware`,
  `detect_batch`, `_classify_bank_structure` for consistency... actually leave it
  alone — see Step 4 for how the call-sites change instead.

**File: `models/internvl3_image_preprocessor.py`**
- No changes — already gated by `self.debug`, which is fed from
  `InternVL3Backend(debug=...)` which comes from `HFChatTemplateBackend(debug=config.verbose)`
  in `model_loader.py`. Point the backend's debug arg at `config.debug` too.

**File: `models/backends/internvl3.py`, `models/backend.py` (HFChatTemplateBackend)**
- `debug: bool` param already exists — no signature change needed. Just upstream
  source (`model_loader.py`) is now `config.debug`.

### Step 4 — Pipe `config.verbose` through for Tier B messages only

**Goal**: keep the per-call `verbose` parameter in orchestrator methods, but feed
it from `config.verbose` (not the stage's `--verbose` CLI flag default, which was
`True`).

**File: `stages/classify.py`**
- After `AppConfig.load(...)`:
  ```python
  effective_verbose = config.verbose  # Tier B gate
  ```
- Pass `verbose=effective_verbose` to `processor.detect_batch(...)` and
  `processor.detect_and_classify_document(...)` (not the stage's CLI arg, which
  may be None now).
- Change the stage's CLI signature:
  ```python
  verbose: bool | None = typer.Option(None, "--verbose/--no-verbose",
      help="Override YAML processing.verbose"),
  debug: bool | None = typer.Option(None, "--debug/--no-debug",
      help="Enable dev debug output (PARSING DEBUG, TENSOR_DTYPE, etc.)"),
  ```
- Only inject into `cli_args` when not None (so YAML is the source of truth when
  flag omitted).

**File: `stages/extract.py`**
- Same changes.

### Step 5 — Move the 3 orchestrator "Tier B init" prints behind `config.verbose`

Currently gated by `self.debug` (line 127-132, 159-172, 187-191) — which now
means Tier C. They're actually Tier B (useful but not noisy), so promote:

**File: `models/orchestrator.py`**
- Add `self._verbose = verbose` in `__init__` (new param, default False).
- Replace `if self.debug:` at lines 127, 159, 168, 187 with `if self._verbose:`.
- Leave PARSING DEBUG (247-281), process_single_image prompt/response dumps
  (550-566), extracted count (572), error traceback (595-597) on `self.debug`
  (Tier C).
- Pass `verbose=config.verbose` alongside `debug=config.debug` from
  `model_loader.py`.

### Step 6 — entrypoint.sh stays as-is

No changes needed. The stage CLIs with `None` defaults already let YAML win.
`config/run_config.yml` will ship with `verbose: false` and `debug: false`, so
default run is quiet.

### Step 7 — Verification

1. **Default run** (no flags, YAML says `verbose: false, debug: false`):
   - Expected: only phase headings + per-image progress + timing.
   - No PARSING DEBUG, no CONFIG DEBUG, no TENSOR_DTYPE, no prompt dumps, no
     model-response dumps.
   - Critically: **never silent for more than 2-3s** during inference.

2. **`--verbose`** only:
   - Adds: orchestrator init line, auto-batch-size line, generation config line,
     "Processing X document", "Using X prompt: N chars", "Extracted 18/20 fields".
   - Still no PARSING DEBUG, CONFIG DEBUG, TENSOR_DTYPE.

3. **`--debug`** only:
   - Adds: PARSING DEBUG, CONFIG DEBUG, TENSOR_DTYPE, prompt dump, response dump,
     error tracebacks.
   - Implicitly also shows Tier A (progress). Tier B is not enabled by --debug
     alone — independent switches.

4. **`--verbose --debug`**:
   - All three tiers.

5. **Fragility test**: add a commented-out alternative under `processing:` in
   YAML — helper from previous commit still works (unrelated sanity check).

## Files to modify (summary)

| File                                      | Change                                                                                                  |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `common/pipeline_config.py`               | Add `debug` field; flip `verbose` default to False; add YAML + ENV mapping                              |
| `config/run_config.yml`                   | Add `processing.debug: false`                                                                           |
| `stages/classify.py`                      | Add per-image `logger.info`; add `--debug` flag; `verbose`/`debug` default None; conditional cli_args   |
| `stages/extract.py`                       | Add `--debug` flag; `verbose`/`debug` default None; conditional cli_args                                |
| `models/model_loader.py`                  | `debug=config.debug`; pass `verbose=config.verbose` to DocumentOrchestrator                              |
| `models/orchestrator.py`                  | Split Tier B init prints onto new `self._verbose`; keep Tier C on `self.debug`                          |
| `models/internvl3_image_preprocessor.py`  | No changes (debug gate already in place; source now `config.debug`)                                     |
| `models/backends/internvl3.py`            | No changes (debug arg already exists; source now `config.debug`)                                        |

**Not touched**: `entrypoint.sh`, `cli.py`, other stages.

## Explicitly out of scope

- Refactoring the existing `sys.stdout.write` vs `print` inconsistency in
  orchestrator.py — separate concern.
- Touching bank-statement extractors' own verbosity — they have their own
  `verbose` param, wired from `config.verbose` in extract.py:148. That stays
  Tier B, correct.
- Removing or moving the "Patched eager attention -> SDPA" banner — that's in
  `models/registry.py` and is arguably useful startup info, leaving alone.
- Suppressing transformers library's own INFO logs — out of scope.

## Risk assessment

- **Low**: Step 2 (add `logger.info` per image) — purely additive.
- **Low**: Step 1 (new `debug` field + YAML + ENV wiring) — additive, defaults to False.
- **Medium**: Step 3 (flip `debug=config.verbose` → `debug=config.debug` in 5 places
  in model_loader.py) — if we miss one, old noise leaks through. Mitigation: grep
  for `debug=config.verbose` after the change; should return 0 results.
- **Medium**: Step 5 (split `self.debug` → `self._verbose` for 3 init prints).
  Mitigation: verify by eye in a diff, and run the three verification scenarios.
- **Low**: Step 4 (CLI tri-state) — I already implemented this once; understand
  the pitfall (silent run due to no Tier A progress), which Step 2 fixes.

## What I'm explicitly NOT going to do again

- Ship a change that silences all output between "Model loaded" and phase end.
  Verification step 1 (no silence >3s during inference) catches this class of
  regression before commit.
- Conflate "make the default quieter" with "remove all signals of progress".
  The fix is to *add* progress at the right level, not to *silence* debug at
  the wrong level.
