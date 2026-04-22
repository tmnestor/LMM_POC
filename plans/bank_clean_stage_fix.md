# Fix: Bank Statement Data Flow Through Clean Stage

## Problem

The current fix in `clean.py` is ugly for three reasons:

1. **Private member access** — `handler._cleaner.clean()` and `handler._validator.validate()` bypass the public `handle()` API
2. **Two code paths** — clean stage has a bank-specific branch with a fragile heuristic (`any(v != "NOT_FOUND" ...)`)
3. **Redundant data** — extract stage saves both `raw_response` (unparseable multi-turn JSON) and `extracted_data` (already-parsed dict)

## Root Cause

The bank adapter (`UnifiedBankExtractor`) does **extraction + parsing + structuring** in one shot, returning structured `schema_fields`. But the pipeline contract says extract produces a raw response string and clean parses it. We broke this contract by smuggling pre-parsed data through a side channel.

## Solution: Synthetic Raw Response

Have the extract stage serialize `schema_fields` into a flat key-value string that the standard parser already understands. This respects the stage contract, eliminates the special case in clean, and drops the `extracted_data` side channel.

### Changes

#### 1. `stages/extract.py` — `_extract_bank_with_adapter()`

Replace:
```python
raw_responses = metadata.get("raw_responses", {})
raw_response_str = json.dumps(raw_responses) if raw_responses else ""
...
writer.write({
    ...
    "raw_response": raw_response_str,
    "extracted_data": schema_fields,
    ...
})
```

With:
```python
# Build a flat key-value string the standard parser can handle.
# This respects the stage contract: extract produces parseable text,
# clean parses it.  The multi-turn raw responses are debug-only.
raw_response_str = "\n".join(
    f"{field}: {value}" for field, value in schema_fields.items()
)

writer.write({
    ...
    "raw_response": raw_response_str,
    # No extracted_data — raw_response is now parseable
    ...
})
```

The `HybridParserAdapter` already handles `FIELD_NAME: value` format via `hybrid_parse_response()` — this is its plain-text fallback path, already well-tested for standard documents.

#### 2. `stages/clean.py` — `run()`

Revert the bank-specific branch. The entire block:
```python
pre_parsed = record.get("extracted_data")
if pre_parsed and any(v != "NOT_FOUND" for v in pre_parsed.values()):
    extracted_data = {
        field_name: handler._cleaner.clean(...)
        ...
    }
    extracted_data = handler._validator.validate(extracted_data)
else:
    extracted_data = handler.handle(raw_response, expected_fields)
```

Back to:
```python
extracted_data = handler.handle(raw_response, expected_fields)
```

One code path. No private member access. No heuristics.

#### 3. No other changes needed

- `ResponseHandler.handle()` — unchanged
- `hybrid_parse_response()` — already parses `FIELD: value` format
- `ExtractionCleaner` — already handles bank array fields (pipe-delimited)
- `BusinessKnowledgeAdapter` — already validates bank transaction count matching

### What About the Multi-Turn Debug Artifact?

The original multi-turn JSON (`{"turn0": "...", "turn1": "..."}`) is useful for debugging but shouldn't be `raw_response`. Two options:

- **Option A (simple)**: Drop it. The synthetic flat response is the artifact. If you need turn-level debugging, add `--debug` logging in the extract stage.
- **Option B (preserve)**: Add a `raw_turns` field to the JSONL record for debugging only. Clean stage ignores it.

Recommend **Option A** — keep it simple. Turn-level debugging belongs in extract stage logs, not the inter-stage JSONL.

### Verification

After applying, run the pipeline on bank statements and confirm:
1. `raw_extractions.jsonl` contains flat `FIELD: value` text in `raw_response` (no JSON, no `extracted_data` key)
2. `cleaned_extractions.jsonl` has correct field values (not NOT_FOUND)
3. Evaluation scores unchanged
