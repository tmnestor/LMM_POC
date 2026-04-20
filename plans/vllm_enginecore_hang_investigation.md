# vLLM EngineCore Hang Investigation

**Date**: 2026-04-20
**Branch**: `kfp`
**Model**: InternVL3.5-8B via vLLM 0.19.0
**Infrastructure**: KFP pipeline on G5 node (4x A10G, 24 GiB each)
**Task**: Classify stage -- sequential image classification of 195 document images

---

## Problem Statement

The vLLM engine consistently dies after processing exactly 11 images during the classify stage. The 12th image request is sent by the Python client but never reaches the EngineCore, which reports "waiting for work" and eventually self-terminates. This occurs regardless of engine version (V0/V1) or memory configuration.

---

## Timeline of Debugging Attempts

### Run 53 (Pipeline Version 53) -- Baseline

- **KFP_TASK**: `run_batch_inference` (wrong -- should be individual stages)
- **Root cause**: Old manifest deployed; `workflow_definition` still set to monolithic task
- **Fix**: Updated `kfp_manifest` to `workflow_definition: classify, filter, extract, clean, evaluate`
- **Outcome**: Subsequent runs correctly dispatch individual stages

### Run 54 -- Staged Pipeline, Step Deadline

- **Symptom**: Classify stage ran for 21 minutes, then `Step exceeded its deadline`
- **Observation**: vLLM startup (model load + CUDA graph compilation) consumed most of the 21-minute window
- **Fix**: Added `enforce_eager=True` to `LLM()` constructor to skip CUDA graph warmup
- **Commit**: `7a5df3e`
- **Outcome**: Startup reduced from ~55 min to ~4 min

### Run 55 -- enforce_eager, First Hang at Image 12

- **Start**: 10:20:56, Image 11 completed ~10:25
- **Symptom**: "Sending image 12/195" logged, then "EngineCore waiting for work" at 10:25:09
- Engine waited 5 minutes idle, then died at 10:30:11
- **Observation**: Python client entered `detect_and_classify_document()` but the request never reached the engine
- **Diagnostic added**: Pre-request logging in `stages/classify.py` (`Sending image N/195 to engine: <filename>`)
- **Commit**: `50c7817`

### Run 56 -- Confirmed: Client Sends, Engine Never Receives

- **Symptom**: Identical -- "Sending image 12/195" appears, engine says "waiting for work", dies after 5 min
- **Hypothesis**: vLLM output objects holding shared memory buffer references, preventing new requests
- **Fix**: Added explicit cleanup in `VllmBackend.generate()`:
  - `buf.close()` after BytesIO usage
  - `del outputs, messages, data_uri` after extracting text
- **Commit**: `d5f1cc8`
- **Outcome**: No change -- still dies at image 12

### Run 57 -- Skipped Suspect Image

- **Test**: Removed `1264488654_3_10.jpeg` (the image that was always image 12) to rule out a corrupt file
- **Outcome**: Died at 11/194 -- the NEW image 12 (`1264488864_3_9.jpeg`) triggered the same hang
- **Conclusion**: Not image-specific; cumulative issue after ~11 requests

### Run 58 -- V0 Engine Fallback

- **Hypothesis**: V1 engine multimodal IPC bug (known issues in vLLM GitHub)
- **Fix**: Set `VLLM_USE_V1=0` in `entrypoint.sh`
- **Commit**: `7b2fc84`
- **Outcome**: Worse -- engine died immediately after 11 images (no 5-min grace period)
- **Conclusion**: Not a V1-specific bug; affects both engine versions

### Run 59 -- Reduced Memory Pressure

- **Hypothesis**: GPU OOM from multimodal cache / KV cache pre-allocation
- **Fix**: Reverted to V1; lowered `gpu_memory_utilization` from 0.85 to 0.70; reduced `max_num_seqs` from 8 to 1
- **Commit**: `6c3a64d`
- **Outcome**: Still dies at image 12
- **Conclusion**: Not a GPU memory utilization issue

---

## What We Know

1. **Exactly 11 images** process successfully every time, regardless of which images they are
2. The **Python client** calls `detect_and_classify_document()` for image 12, which enters the vLLM backend, but the request **never reaches the EngineCore**
3. The hang occurs in the **client-side request pipeline** (between `llm.chat()` being called and the EngineCore receiving work)
4. Both **V0 and V1** engines exhibit the same wall at 11 images
5. **Lowering `gpu_memory_utilization`** (0.85 -> 0.70) does not help
6. **`max_num_seqs: 1`** does not help
7. **Explicit cleanup** of vLLM outputs does not help
8. **Image content** is irrelevant -- skipping images shifts the wall to the same count
9. Each image is base64-encoded into a data URI (~1-5 MB per image) and passed via OpenAI-compatible chat messages
10. The HF code path (non-vLLM) processes all 195 images successfully (Run 51, 1:31:11)

## What We Don't Know

1. **What the client is doing** during the hang -- no traceback, no error, just silence
2. **GPU memory state** at the time of hang -- no `nvidia-smi` output captured (runs inside KFP pod, no `dmesg` access)
3. **Whether it's a vLLM 0.19 regression** -- we haven't tested older vLLM versions
4. **Whether tensor parallelism is the trigger** -- haven't tested with `tp=1` (single GPU)
5. **Whether the base64 data URI approach is the bottleneck** -- haven't tried passing images via file path

---

## Relevant vLLM GitHub Issues

| Issue | Description | Relevance |
|-------|-------------|-----------|
| [#37602](https://github.com/vllm-project/vllm/issues/37602) | EngineCore crash on concurrent image requests (Qwen3.5) | Same symptom: EngineCore dies during multimodal inference |
| [#31404](https://github.com/vllm-project/vllm/issues/31404) | MM cache AssertionError crashes engine | Multimodal cache eviction bug -- fixed in PR #34749 |
| [#27249](https://github.com/vllm-project/vllm/issues/27249) | Multi-node multimodal inference fails with V1 | V1 + distributed + multimodal = failure |
| [#17972](https://github.com/vllm-project/vllm/issues/17972) | Server hangs after initial requests | Similar symptom: zero throughput after N requests |
| [#27557](https://github.com/vllm-project/vllm/issues/27557) | EngineCore died unexpectedly (exit code 0) | Matches our error message exactly |

---

## Next Steps to Try

### High Priority

1. **Single GPU (`tp=1`)**: Test whether tensor parallelism is the trigger. InternVL3.5-8B (~16 GB BF16) fits on one A10G (24 GB). Set `num_gpus: 1` in `run_config.yml` or `CUDA_VISIBLE_DEVICES=0`.

2. **Add `nvidia-smi` monitoring**: Run `nvidia-smi` in a background loop during inference to capture GPU memory at the moment of hang:
   ```bash
   while true; do nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >> /tmp/gpu_monitor.csv; sleep 2; done &
   ```

3. **Capture Python-level traceback**: Add a `signal` handler or `faulthandler` to dump the Python stack when the process gets stuck:
   ```python
   import faulthandler
   faulthandler.enable()
   faulthandler.dump_traceback_later(300, repeat=True)  # dump every 5 min
   ```

4. **Try file-path image input** instead of base64 data URIs -- the repeated base64 encoding of large images may be exhausting client-side memory or IPC buffers.

### Medium Priority

5. **Downgrade vLLM** to 0.17 or 0.18 and retest -- isolate whether this is a 0.19 regression.

6. **Reduce `max_tiles`** from 18 to 6 -- fewer vision tiles = smaller intermediate activations per image.

7. **Test with `max_model_len: 4096`** (halved) -- reduces KV cache pre-allocation, leaving more VRAM for multimodal processing.

### Lower Priority

8. **File a vLLM GitHub issue** with the full reproduction details -- the consistent 11-image wall is a clean, reproducible bug report.

9. **Consider alternative backends**: The HF code path works reliably. vLLM's throughput advantage may not justify the reliability cost for 195 images.

---

## Current Configuration

```yaml
# VllmSpec defaults (model_loader.py)
gpu_memory_utilization: 0.70
max_model_len: 8192
max_num_seqs: 1
enforce_eager: true

# entrypoint.sh env vars
VLLM_ATTENTION_BACKEND: FLASHINFER
VLLM_NO_USAGE_STATS: 1
VLLM_LOGGING_LEVEL: DEBUG  (temporarily)

# run_config.yml
model.type: internvl3-vllm
model.max_tiles: 18
processing.num_gpus: 0  (auto-detect = 4)
```

## Commits on `kfp` Branch (This Investigation)

| Commit | Description |
|--------|-------------|
| `7a5df3e` | `enforce_eager=True` -- skip CUDA graph compilation |
| `50c7817` | Pre-request logging to diagnose hang location |
| `d5f1cc8` | Explicit cleanup of vLLM outputs and BytesIO buffers |
| `7b2fc84` | V0 engine fallback (reverted) |
| `6c3a64d` | Lower `gpu_memory_utilization` to 0.70, `max_num_seqs` to 1 |
