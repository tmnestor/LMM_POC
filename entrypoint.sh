#!/bin/bash
# =============================================================================
# LMM POC - KFP Pipeline Entrypoint
# =============================================================================
#
# This is the first script that runs when a KFP pipeline job starts.
# Sets up environment (conda, CUDA, logging), then dispatches to the
# requested KFP_TASK which calls stages/*.py modules directly.
#
# Flow:
#   KFP Pipeline → Container starts → entrypoint.sh → stages.{classify,extract,clean,evaluate}
#
# How KFP passes configuration:
#   The pipeline YAML defines `input_params` (model, image_dir, output, etc.)
#   which users can fill in via the KFP UI. KFP injects these as environment
#   variables into the container. This script reads those env vars and
#   translates them into stages.* command-line flags.
#
#   Example: if a user sets model=llama and num_gpus=4 in the KFP UI:
#     python3 -m stages.extract --model llama --num-gpus 4 ...
#
# Available KFP_TASK values: the dispatcher's `case` near the bottom is the
# source of truth, and `_print_task_help` renders the human-readable list at
# runtime (run with KFP_TASK unset to print it). To avoid a third copy drifting
# out of sync, the full list is NOT duplicated here — read `_print_task_help`.
# Note: `filter` is deprecated — kept as a no-op for KFP manifest compatibility.
#
# Local examples:
#   KFP_TASK=run_batch_inference image_dir=../evaluation_data/synthetic bash entrypoint.sh
#   KFP_TASK=run_graph_robust image_dir=../evaluation_data/synthetic ground_truth=../evaluation_data/synthetic/ground_truth.jsonl bash entrypoint.sh
#   KFP_TASK=run_trust_pipeline trust_data_dir=../evaluation_data/trust trust_ground_truth=../evaluation_data/trust/ground_truth.yaml bash entrypoint.sh
#   KFP_TASK=run_transaction_link bash entrypoint.sh                                  # paths from run_config.yml linking:
#   KFP_TASK=run_transaction_link linking_data_dir=../evaluation_data/linking bash entrypoint.sh
#
# =============================================================================

# #############################################################################
#  SETUP: Shell Safety, CUDA, Logging (pre-conda)
# #############################################################################

# ---- Shell Safety Settings ---- #
# errexit:  Exit immediately if any command fails (non-zero exit code)
# nounset:  Treat unset variables as an error (catches typos in var names)
# pipefail: A pipeline fails if ANY command in the pipe fails, not just the last
set -o errexit
set -o nounset
set -o pipefail

# ---- CUDA Environment ---- #
# Deterministic GPU indexing: ensures cuda:0 always maps to the same physical
# GPU regardless of driver enumeration order. Critical for multi-GPU so that
# log messages ("GPU 0 failed") match nvidia-smi output.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# vLLM attention backend: prefer VllmSpec.attention_backend in model_loader.py
# (passed to LLM() constructor). Env var is a fallback for envs with
# pre-compiled FlashInfer kernels. Omitted by default so vLLM auto-selects.
# export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# Suppress vLLM usage telemetry (avoids TLS cert errors in air-gapped envs).
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

# NCCL shared memory: KFP pods may default /dev/shm to 64 MB, which is
# too small for NCCL's SHM transport under tensor parallelism — after ~11
# images the SHM region fills and NCCL silently deadlocks. The fix is to
# increase /dev/shm in the KFP pod spec (emptyDir medium=Memory, 8Gi+).
# Setting NCCL_SHM_DISABLE=1 does NOT work — on G5 instances without
# NVLink, disabling SHM leaves NCCL with no viable intra-node transport
# and it fails immediately with "unhandled system error".

# ---- Log Configuration ---- #
# All output (stdout + stderr) is captured to a timestamped log file on EFS,
# while still being printed to the console (so KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
#
# Priority: LMM_LOG_DIR env var > run_config.yml bootstrap.logging.log_dir > fail
# No silent fallback — in KFP, pod-local writes are ephemeral/forbidden.
CONFIG_FILE="./config/run_config.yml"

# ---- Resolve ALL YAML defaults up front, with the conda env's own python ----
# A SINGLE PyYAML resolver (scripts/resolve_yaml_defaults.py) supplies every
# YAML_* variable used below — log dirs, data/model paths, trust + linking
# paths. It must run BEFORE the `exec`/`tee` redirect (which needs the log dir),
# but PyYAML lives only inside the conda env, never in system/base python (the
# base env genuinely has no `yaml` on DEV/PROD). So we run the resolver with the
# conda env's OWN interpreter, addressed by path — no `conda activate` needed
# just to launch an interpreter — and activate the env properly further down.
#
# The conda env itself cannot be read from run_config.yml: parsing that YAML
# needs a PyYAML-capable python, which only exists INSIDE this env (chicken-and-
# egg). So CONDA_ENV is bootstrapped from LMM_CONDA_ENV or the default below.
# CONDA_ENV="${LMM_CONDA_ENV:-/efs/shared/.conda/envs/vllm_env}"
CONDA_ENV="${LMM_CONDA_ENV:-/home/jovyan/.conda/envs/vllm_env2}"
CONDA_PY="${CONDA_ENV}/bin/python"
if [[ ! -x "$CONDA_PY" ]]; then
  echo "FATAL: bootstrap interpreter not found: $CONDA_PY"
  echo "  What:  the python used to parse $CONFIG_FILE (it needs PyYAML) is missing."
  echo "  Where: CONDA_ENV='$CONDA_ENV' — from LMM_CONDA_ENV env var, or the default in entrypoint.sh."
  echo "  Fix:   point LMM_CONDA_ENV at a real conda env dir so \$LMM_CONDA_ENV/bin/python exists, e.g."
  echo "           export LMM_CONDA_ENV=/home/jovyan/.conda/envs/vllm_env2"
  exit 1
fi
# Emits YAML_* assignments (every key, unconditionally; missing keys -> ''),
# all read below as ${YAML_*:-} so `set -o nounset` is satisfied.
eval "$("$CONDA_PY" scripts/resolve_yaml_defaults.py "$CONFIG_FILE")"

# Select this task's log dir (env var > YAML > fail). Trust tasks use
# pipeline.trust.log_dir; transaction-linking uses pipeline.linking.log_dir;
# everything else uses bootstrap.logging.log_dir.
case "${KFP_TASK:-}" in
  trust_classify|trust_extract|trust_clean|trust_evaluate|run_trust_pipeline)
    LOG_DIR="${LMM_TRUST_LOG_DIR:-${YAML_TRUST_LOG_DIR:-${LMM_LOG_DIR:-${YAML_LOG_DIR:-}}}}"
    ;;
  run_transaction_link)
    LOG_DIR="${LMM_LINKING_LOG_DIR:-${YAML_LINKING_LOG_DIR:-${LMM_LOG_DIR:-${YAML_LOG_DIR:-}}}}"
    ;;
  *)
    LOG_DIR="${LMM_LOG_DIR:-${YAML_LOG_DIR:-}}"
    ;;
esac
if [[ -z "$LOG_DIR" ]]; then
  echo "FATAL: No log directory configured. Set LMM_LOG_DIR env var or bootstrap.logging.log_dir in $CONFIG_FILE"
  exit 1
fi
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"
# Export so Python (cli.py) can display it in the startup Configuration panel.
export LMM_LOG_FILE="$LOG_FILE"

# `exec` redirects ALL subsequent output through `tee`, which writes to
# both the console (for KFP) and the log file (for persistent debugging).
# The `2>&1` merges stderr into stdout so errors are captured too.
# NOTE: Process substitution with tee means the subshell can outlive this
# script — the final log line(s) may flush slightly after exit. This is a
# known bash nuance and is harmless in practice.
exec > >(tee -a "$LOG_FILE") 2>&1

# Timestamped logging function — prefixes every message with a timestamp
# so you can correlate events with KFP logs and identify slow steps.
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Prominent stage banner — mirrors the top-level run banner so each phase of a
# multi-stage pipeline stands out in the logs. Leads with a blank line for
# separation; pass the stage label (e.g. "Phase 1/5: classify (GPU)").
_banner() {
  log ""
  log "================================================================="
  log "    $1"
  log "================================================================="
}

# ---- Cleanup Trap ---- #
# `trap ... EXIT` runs this code whenever the script exits — whether it
# succeeds, fails, or gets killed (e.g. OOM). This guarantees you always
# see the exit code and how long the run took, even on crashes.
# $SECONDS is a built-in bash variable that counts elapsed seconds.
SECONDS=0
trap 'rc=$?; echo ""; log "Exited with code $rc after ${SECONDS}s"; log "Log file: $LOG_FILE"' EXIT

# #############################################################################
#  SETUP: Conda Activation & Environment
# #############################################################################

# ---- Banner ---- #
log "================================================================="
log "    Running LMM for Information Extraction"
log "================================================================="
log ""

# ---- Conda Activation ---- #
# Assumes EFS is mounted via KFP volume spec at /efs/shared/. If the
# volume mount is missing, conda activate will fail below. Check the
# KFP pipeline YAML volume definitions if this step errors.
#
# KFP containers start with a bare shell. We need to initialise conda
# (the `eval` line) and then activate our environment which has all
# the Python dependencies (torch, transformers, etc.) pre-installed.
log "Activating conda environment..."
eval "$(conda shell.bash hook)"
# CONDA_ENV was bootstrapped (and its python validated) during the YAML
# resolution above, before the tee redirect.
log "Conda env: $CONDA_ENV"
# Temporarily allow unbound variables — conda activation scripts (e.g. MKL)
# reference variables that may not be set yet.
set +o nounset
conda activate "$CONDA_ENV" || { set -o nounset; log "FATAL: conda activate failed"; exit 1; }
set -o nounset
# Ensure conda's libstdc++ is found before the (older) system copy.
# The vllm_env ships libstdcxx-ng>=12 which provides GLIBCXX_3.4.30
# needed by libzmq; without this, the linker finds /usr/lib64/libstdc++
# first and fails with "version GLIBCXX_3.4.30 not found".
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Suppress verbose INFO logging from vLLM engine, transformers, and tokenizers.
# Override by setting these env vars before running entrypoint.sh.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Log environment details for debugging failed runs —
# knowing the Python version, conda env, and GPU type is critical
# when something works locally but fails in the pipeline.
log "---------------------------------------"
log "Python:  $(which python3)"
log "Version: $(python3 --version 2>&1)"
log "Conda:   $(conda info --envs | grep '*' || echo 'unknown')"
log "Log dir: $LOG_DIR (source: ${LMM_LOG_DIR:+env}${LMM_LOG_DIR:-${YAML_LOG_DIR:+yaml}})"
log "---------------------------------------"
log ""

# ---- Route the fix_mistral_regex tokenizer cache to a writable dir ---- #
# vLLM bakes a fix_mistral_regex-corrected tokenizer copy to disk (see
# models/model_loader.ensure_corrected_tokenizer). Its default is ~/.cache,
# which is NOT writable in the KFP prod pod — only the run_config.yml output
# directory (output.dir -> YAML_OUTPUT_DIR) is. Point the cache there so the
# pre-warm below AND the vLLM workers (which inherit this exported env, incl.
# vLLM's spawned EngineCore children) can write it. An operator-set
# LMM_TOKENIZER_CACHE always wins.
if [[ -z "${LMM_TOKENIZER_CACHE:-}" && -n "${YAML_OUTPUT_DIR:-}" ]]; then
  export LMM_TOKENIZER_CACHE="${YAML_OUTPUT_DIR%/}/tokenizer_cache"
  log "Tokenizer cache dir: ${LMM_TOKENIZER_CACHE} (under run_config.yml output.dir)"
fi

# ---- GPU Health Check ---- #
# Verify GPUs are accessible and healthy before loading ~16GB models onto them.
# Catches ECC errors, fallen-off-bus GPUs, and driver mismatches early —
# much cheaper than discovering mid-inference after a 60s model load.
log "GPU environment:"
log "  CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES:-<not set — all GPUs visible>}"
log "  CUDA_DEVICE_ORDER:     ${CUDA_DEVICE_ORDER}"
log "  NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-<not set>}"
log ""

if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
  log "Detected $GPU_COUNT GPU(s):"
  log ""
  # Per-GPU detail: index, name, VRAM, temperature, ECC errors
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name mem_total mem_free temp ecc; do
    log "  GPU $idx: $name |$(echo "$mem_total" | xargs) total |$(echo "$mem_free" | xargs) free | ${temp}C | ECC errors: $(echo "$ecc" | xargs)"
  done
  log ""

  # Check for GPUs in error state — ERR or "Unknown Error" in nvidia-smi
  # means the GPU has fallen off the bus or has a hardware fault.
  if nvidia-smi --query-gpu=index,pstate --format=csv,noheader 2>/dev/null | grep -qi "err"; then
    log "WARNING: One or more GPUs report error state. Run may fail."
    log "$(nvidia-smi --query-gpu=index,pstate,ecc.errors.uncorrected.volatile.total --format=csv 2>/dev/null)"
    log ""
  fi
else
  log "WARNING: nvidia-smi not found — cannot verify GPU health"
  log ""
fi

# ---- Pre-warm fix_mistral_regex-corrected tokenizer (once, before workers) ---- #
# InternVL3.5's Mistral tokenizer ships a buggy whitespace/digit regex that
# corrupts amount tokenization on dense bank statements. vLLM loads its tokenizer
# in the front-end AND every spawned EngineCore child, so a load-time patch can't
# reach the child; instead we bake a fix_mistral_regex-corrected copy to disk
# (models/model_loader.ensure_corrected_tokenizer) and hand vLLM that path.
# Doing it HERE — once, before the DP workers spawn — avoids the parallel workers
# racing to build the shared cache. Idempotent (instant after first build) and
# best-effort: a failure only means the engine falls back to the model's own
# tokenizer, i.e. the prior functional behaviour.
if [[ "${YAML_MODEL_TYPE:-}" == internvl*-vllm && -n "${YAML_MODEL_PATH:-}" ]]; then
  log "Pre-warming fix_mistral_regex-corrected tokenizer (${YAML_MODEL_TYPE})..."
  if _tok_path=$(python3 -c "from models.model_loader import ensure_corrected_tokenizer as e; print(e('${YAML_MODEL_PATH}'))" 2>&1); then
    log "  corrected tokenizer: ${_tok_path}"
  else
    log "  WARNING: tokenizer pre-warm failed — engine load falls back to the model's own tokenizer:"
    log "  ${_tok_path}"
  fi
fi

# #############################################################################
#  HELPERS: Shared Functions
# #############################################################################

# True if $1 is set, non-empty, and not the literal "None".
# KFP stringifies an unset/blank `input_param` as the literal string "None"
# (not ""), so both must be rejected when deciding whether a value was provided.
# Used throughout to gate optional flags and YAML fallbacks under `set -o nounset`.
_is_set() { [[ -n "${1:-}" && "${1}" != "None" ]]; }

_print_task_help() {
  # Single source of truth for the KFP_TASK list, shared by the unset ("")
  # and unknown (*) dispatch branches.  Keep the header comment at the top of
  # this file in sync with this list.
  log "  Available tasks:"
  log "    run_batch_inference   — 4-stage classic pipeline (classify/extract/clean/evaluate)  [local dev]"
  log "    run_graph_robust      — 3-stage probe-based pipeline (extract --graph-robust/clean/evaluate)  [local dev]"
  log "    run_trust_pipeline    — 4-stage trust distribution compliance pipeline  [local dev]"
  log "    run_transaction_link  — 5-stage receipt->bank transaction linking (matcher-first, VLM fallback)  [local dev]"
  log "    classify              — Stage 1: document type detection (GPU)"
  log "    extract               — Stage 2: field extraction (GPU)"
  log "    clean                 — Stage 3: parse/clean responses (CPU)"
  log "    evaluate              — Stage 4: evaluation (CPU)"
  log "    trust_classify        — Trust document type classification (GPU)"
  log "    trust_extract         — Trust distribution field extraction (GPU)"
  log "    trust_clean           — Trust compliance validation (CPU)"
  log "    trust_evaluate        — Trust compliance evaluation (CPU)"
}

_clear_prev_output() {
  # CLEAR_PREV_OUTPUT=true: delete the listed output artifacts so the stage
  # recomputes from scratch.  Unset/false (default): no-op — the stage resumes
  # and skips already-processed images.
  #
  # Why resume-by-default: production drips new images into the image directory
  # over time, so a re-run should process only the NEW arrivals, not reprocess
  # the whole directory.  A full clean-slate run is opt-in via
  # CLEAR_PREV_OUTPUT=true.
  #
  # Only explicit artifact FILES are passed in — never a directory, never a log
  # path — so logs are ALWAYS preserved (trust logs live under $TRUST_OUT/logs).
  # Each stage must pass only its OWN outputs, never its inputs.
  [[ "${CLEAR_PREV_OUTPUT}" == "true" ]] || return 0
  local f
  for f in "$@"; do
    [[ -n "$f" && -e "$f" ]] || continue
    rm -f "$f"
    log "  CLEAR_PREV_OUTPUT: removed $f"
  done
}

_resolve_trust_vars() {
  # Resolve TRUST_OUT and TRUST_LOG_DIR from env vars / YAML fallbacks.
  # Called by every trust_* case branch.
  TRUST_OUT="${trust_output:-${output:-./outputs}}"
  TRUST_LOG_DIR="${trust_log_dir:-${TRUST_OUT}/logs}"
  mkdir -p "$TRUST_OUT" "$TRUST_LOG_DIR"
}

_resolve_linking_vars() {
  # Resolve the transaction-linking paths from env vars / YAML fallbacks, then
  # point the SHARED classify/extract/clean path globals at the linking output
  # root so the existing _run_classify/_run_extract runners write there.
  # linking_output is a FILE (transaction_links.jsonl); its parent dir is the
  # output root for the intermediate JSONL artefacts.
  LINK_OUT="$(dirname "${linking_output:-./outputs/transaction_links.jsonl}")"
  LINK_LOG_DIR="${linking_log_dir:-${LINK_OUT}/logs}"
  mkdir -p "$LINK_OUT" "$LINK_LOG_DIR"
  # Drive the GPU stages off the linking dataset + output root.
  image_dir="${linking_data_dir:-${image_dir:-}}"
  CLASSIFICATIONS="${LINK_OUT}/classifications.jsonl"
  RAW_EXTRACTIONS="${LINK_OUT}/raw_extractions.jsonl"
  CLEAN_EXTRACTIONS="${LINK_OUT}/cleaned_extractions.jsonl"
}

_ensure_trust_quads() {
  # Auto-generate quads CSV from ground truth YAML if no quads file exists.
  # Requires TRUST_OUT to be set (call _resolve_trust_vars first).
  if [[ ! -f "${trust_quads:-}" ]] && _is_set "${trust_ground_truth:-}"; then
    trust_quads="${TRUST_OUT}/trust_quads.csv"
    log "  Generating quads CSV from ground truth → ${trust_quads}"
    python3 scripts/generate_trust_manifest.py \
      --ground-truth "$trust_ground_truth" \
      --output "$trust_quads" || exit $?
  fi
}

_read_inference_elapsed() {
  # Read and sum GPU inference elapsed times from a file.
  # Sets INFERENCE_ARGS=() or INFERENCE_ARGS=(--inference-seconds N).
  local elapsed_file="${1:?usage: _read_inference_elapsed <file>}"
  INFERENCE_ARGS=()
  if [[ -f "$elapsed_file" ]]; then
    local total=0
    while IFS= read -r line; do
      total=$((total + line))
    done < "$elapsed_file"
    INFERENCE_ARGS=(--inference-seconds "$total")
    log "GPU inference elapsed: ${total}s (from $elapsed_file)"
  else
    log "WARNING: $elapsed_file not found — throughput will use sum of processing_time."
  fi
}

_run_cpu_stages() {
  # Run the clean + evaluate CPU tail shared by run_batch_inference and
  # run_graph_robust.  Expects INFERENCE_ELAPSED, RAW_EXTRACTIONS,
  # CLEAN_EXTRACTIONS, EVAL_DIR to be set by the caller.
  local clean_label="${1:?usage: _run_cpu_stages <clean_label> <eval_label>}"
  local eval_label="${2:?usage: _run_cpu_stages <clean_label> <eval_label>}"

  log ""
  log "${clean_label}: clean (CPU, no GPU)..."
  python3 -m stages.clean \
    --input      "$RAW_EXTRACTIONS" \
    --output-dir "$CLEAN_EXTRACTIONS" || exit $?
  log "${clean_label}: clean complete."

  if _is_set "${ground_truth:-}"; then
    log ""
    log "${eval_label}: evaluate (CPU, no GPU)..."
    python3 -m stages.evaluate \
      --input              "$CLEAN_EXTRACTIONS" \
      --ground-truth       "$ground_truth" \
      --output-dir         "$EVAL_DIR" \
      --inference-seconds  "$INFERENCE_ELAPSED" || exit $?
    log "${eval_label}: evaluate complete."
  else
    log ""
    log "${eval_label}: evaluate skipped — ground_truth not provided."
  fi
}

# ---- Per-stage runners (shared by KFP pods and local orchestrators) ---- #
# Each runner is the single source of truth for one stage's `python3 -m stages.*`
# invocation, called by BOTH its standalone KFP pod branch and the local
# orchestrator that chains stages. This keeps the two paths from drifting (which
# is how the .inference_elapsed >/>> inconsistency originally crept in).
#
# Convention: runners hold ONLY the invocation (+ its required-var diagnostics).
# Orchestration policy — logging labels, CLEAR_PREV_OUTPUT clearing, elapsed-file
# writes, and INFERENCE_ARGS construction — stays in the caller, because it
# differs between the per-pod path (per-stage .inference_elapsed file) and the
# orchestrator path (single wall-clock).  Runners read globals set by the caller
# (OPT_MODEL, OUT_ROOT-derived paths, TRUST_OUT, trust_* vars).

_run_classify() {
  # GPU. Document type detection. Used by the `classify` pod and
  # run_batch_inference Phase 1 (identical invocation).
  python3 -m stages.classify \
    --data-dir   "${image_dir:?image_dir env var required}" \
    --output-dir "$CLASSIFICATIONS" \
    "${OPT_MODEL[@]}" || exit $?
}

_run_extract() {
  # GPU. Field extraction. $1 = mode:
  #   auto         — classified if $CLASSIFICATIONS exists, else graph-robust
  #                  (the `extract` pod auto-detects per the KFP DAG shape)
  #   classified   — type-specific extraction from $CLASSIFICATIONS
  #                  (run_batch_inference, where classify always ran first)
  #   graph_robust — force probe-based extraction, ignoring any (stale)
  #                  classifications file (run_graph_robust)
  local mode="${1:?usage: _run_extract <auto|classified|graph_robust>}"
  local use_classified=false
  case "$mode" in
    classified) use_classified=true ;;
    graph_robust) use_classified=false ;;
    auto)
      if [[ -f "$CLASSIFICATIONS" ]]; then
        use_classified=true
        log "Found $CLASSIFICATIONS — using classified extraction."
      else
        log "No classifications found — using graph-robust probes."
      fi
      ;;
    *) log "FATAL: _run_extract: unknown mode '$mode'"; exit 1 ;;
  esac
  if $use_classified; then
    python3 -m stages.extract \
      --classifications "$CLASSIFICATIONS" \
      --data-dir        "${image_dir:?image_dir env var required}" \
      --output-dir      "$RAW_EXTRACTIONS" \
      "${OPT_MODEL[@]}" || exit $?
  else
    python3 -m stages.extract \
      --data-dir   "${image_dir:?image_dir env var required}" \
      --output-dir "$RAW_EXTRACTIONS" \
      --graph-robust \
      "${OPT_MODEL[@]}" || exit $?
  fi
}

_run_trust_classify() {
  # GPU. Builds OPT_CLASSIFY_PATHS from the optional trust path vars, then runs
  # stages.trust_classify. Used by the `trust_classify` pod and run_trust_pipeline.
  OPT_CLASSIFY_PATHS=()
  if _is_set "${trust_classifications:-}"; then
    OPT_CLASSIFY_PATHS+=(--classifications "$trust_classifications")
  fi
  if _is_set "${trust_quads:-}"; then
    OPT_CLASSIFY_PATHS+=(--quads "$trust_quads")
  fi
  if _is_set "${trust_quads_incomplete:-}"; then
    OPT_CLASSIFY_PATHS+=(--quads-incomplete "$trust_quads_incomplete")
  fi
  python3 -m stages.trust_classify \
    --data-dir   "${trust_data_dir:?trust_data_dir is required — set via pipeline.trust.data_dir in run_config.yml or trust_data_dir env var}" \
    --output-dir "${TRUST_OUT}" \
    "${OPT_CLASSIFY_PATHS[@]}" \
    "${OPT_MODEL[@]}" || exit $?
}

_run_trust_extract() {
  # GPU. Runs the trust-link extraction. Used by the `trust_extract` pod and
  # run_trust_pipeline. Expects trust_quads to exist (trust_classify or
  # _ensure_trust_quads produced it).
  python3 -m stages.link trust-link \
    --quads    "${trust_quads:?trust_quads is required — set via pipeline.trust.quads in run_config.yml or trust_quads env var}" \
    --data-dir "${trust_data_dir:?trust_data_dir is required — set via pipeline.trust.data_dir in run_config.yml or trust_data_dir env var}" \
    --output   "${trust_raw_extractions:?trust_raw_extractions is required — set via pipeline.trust.raw_extractions in run_config.yml or trust_raw_extractions env var}" \
    "${OPT_MODEL[@]}" || exit $?
}

_run_trust_clean() {
  # CPU. Runs trust compliance validation. Used by the `trust_clean` pod and
  # run_trust_pipeline.
  python3 -m stages.trust_clean \
    --input  "${trust_raw_extractions:?trust_raw_extractions is required — set via pipeline.trust.raw_extractions in run_config.yml or trust_raw_extractions env var}" \
    --output "${trust_compliance_results:?trust_compliance_results is required — set via pipeline.trust.compliance_results in run_config.yml or trust_compliance_results env var}" || exit $?
}

_run_trust_evaluate() {
  # CPU. Builds optional flags and runs stages.evaluate_trust. Used by the
  # `trust_evaluate` pod and run_trust_pipeline. The caller MUST set:
  #   TRUST_EVAL_DIR  — output dir (already created)
  #   INFERENCE_ARGS  — (--inference-seconds N) or () — from _read_inference_elapsed
  #                     (pod path) or the orchestrator's wall-clock.
  OPT_CLASSIFICATIONS=()
  if _is_set "${trust_classifications:-}"; then
    OPT_CLASSIFICATIONS=(--classifications "$trust_classifications")
  fi
  OPT_CLASSIFICATION_GT=()
  if _is_set "${trust_classification_gt:-}"; then
    OPT_CLASSIFICATION_GT=(--classification-gt "$trust_classification_gt")
  fi
  python3 -m stages.evaluate_trust \
    --input        "${trust_compliance_results:?trust_compliance_results is required — set via pipeline.trust.compliance_results in run_config.yml or trust_compliance_results env var}" \
    --ground-truth "${trust_ground_truth:?trust_ground_truth is required — set via pipeline.trust.ground_truth in run_config.yml or trust_ground_truth env var}" \
    --output-dir   "$TRUST_EVAL_DIR" \
    "${OPT_CLASSIFICATIONS[@]}" \
    "${OPT_CLASSIFICATION_GT[@]}" \
    "${INFERENCE_ARGS[@]}" || exit $?
}

# #############################################################################
#  CONFIG RESOLUTION (env vars, YAML fallbacks, CLI args)
# #############################################################################

# ---- YAML fallbacks for missing env vars ---- #
# cli.py internally cascades env → YAML → defaults via AppConfig.load.
# But stages/*.py mark their CLI flags as required via typer, so they
# fail before AppConfig runs. Apply the YAML fallback HERE so the stage
# commands below receive a concrete --data-dir / --output-dir value.
# Env var always wins when explicitly set (matches cli.py semantics).
# Assign the YAML fallback ($2) to the named env var ($1) only when the env
# var was not already provided (env always wins, matching cli.py). Uses
# indirect read (${!name}) + `printf -v` write — both available in macOS
# bash 3.2. The `||` short-circuit keeps a false `_is_set` from tripping
# `set -o errexit`.
_default_from_yaml() {
  local name="$1"
  _is_set "${!name:-}" || printf -v "$name" '%s' "$2"
}
_default_from_yaml model                    "${YAML_MODEL_TYPE:-}"
_default_from_yaml image_dir                "${YAML_DATA_DIR:-}"
_default_from_yaml ground_truth             "${YAML_GROUND_TRUTH:-}"
_default_from_yaml output                   "${YAML_OUTPUT_DIR:-}"
_default_from_yaml trust_data_dir           "${YAML_TRUST_DATA_DIR:-}"
_default_from_yaml trust_quads              "${YAML_TRUST_QUADS:-}"
_default_from_yaml trust_quads_incomplete   "${YAML_TRUST_QUADS_INCOMPLETE:-}"
_default_from_yaml trust_ground_truth       "${YAML_TRUST_GROUND_TRUTH:-}"
_default_from_yaml trust_classification_gt  "${YAML_TRUST_CLASSIFICATION_GT:-}"
_default_from_yaml trust_classifications    "${YAML_TRUST_CLASSIFICATIONS:-}"
_default_from_yaml trust_raw_extractions    "${YAML_TRUST_RAW_EXTRACTIONS:-}"
_default_from_yaml trust_compliance_results "${YAML_TRUST_COMPLIANCE_RESULTS:-}"
_default_from_yaml trust_output             "${YAML_TRUST_OUTPUT_DIR:-}"
_default_from_yaml trust_evaluation_dir     "${YAML_TRUST_EVALUATION_DIR:-}"
_default_from_yaml trust_log_dir            "${YAML_TRUST_LOG_DIR:-}"
_default_from_yaml linking_data_dir         "${YAML_LINKING_DATA_DIR:-}"
_default_from_yaml linking_output           "${YAML_LINKING_OUTPUT:-}"
_default_from_yaml linking_ground_truth     "${YAML_LINKING_GROUND_TRUTH:-}"
_default_from_yaml linking_evaluation_dir   "${YAML_LINKING_EVALUATION_DIR:-}"
_default_from_yaml linking_log_dir          "${YAML_LINKING_LOG_DIR:-}"

# ---- CLEAR_PREV_OUTPUT toggle (validated at startup, before any work) ---- #
# Controls whether stages start from a clean slate or resume:
#   true           → delete previous OUTPUT artifacts (never logs), full recompute
#   false / unset  → resume: skip already-processed images (production default)
# Resume is the default because prod drips new images into the image directory
# over time — a re-run should process only the new arrivals.
#
# Normalize case before validating: an unquoted YAML boolean in the KFP
# manifest (CLEAR_PREV_OUTPUT: true) is often injected into the container env
# as the Python string "True"/"False" — and KFP stringifies unset params as
# the literal "None".  Lowercase so true/True/TRUE (and none/None) all work.
# Use tr, not bash ${x,,}, for macOS bash 3.2 compatibility.
CLEAR_PREV_OUTPUT="${CLEAR_PREV_OUTPUT:-false}"
CLEAR_PREV_OUTPUT="$(printf '%s' "$CLEAR_PREV_OUTPUT" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$CLEAR_PREV_OUTPUT" || "$CLEAR_PREV_OUTPUT" == "none" ]]; then
  CLEAR_PREV_OUTPUT="false"
fi
case "$CLEAR_PREV_OUTPUT" in
  true)  log "CLEAR_PREV_OUTPUT=true — stages will DELETE previous output artifacts (logs preserved) and recompute." ;;
  false) log "CLEAR_PREV_OUTPUT=false — stages will RESUME (already-processed images are skipped)." ;;
  *)
    log "FATAL: CLEAR_PREV_OUTPUT must be 'true' or 'false' (got '${CLEAR_PREV_OUTPUT}')."
    log "  Where: CLEAR_PREV_OUTPUT environment variable (KFP input_param or shell export)."
    log "  Fix:   set CLEAR_PREV_OUTPUT=true for a clean-slate re-run, or leave it unset/false to resume."
    exit 1
    ;;
esac

CLI_ARGS=()

# model → --model (e.g. "internvl3", "llama")
if _is_set "${model:-}"; then
  CLI_ARGS+=(--model "$model")
fi

# image_dir → --data-dir (path to folder of images to process)
if _is_set "${image_dir:-}"; then
  CLI_ARGS+=(--data-dir "$image_dir")
fi

# output → --output-dir (where results, CSVs, and reports are saved)
if _is_set "${output:-}"; then
  CLI_ARGS+=(--output-dir "$output")
fi

# num_gpus → --num-gpus (0 = auto-detect all, 1 = single GPU, N = use N GPUs)
if _is_set "${num_gpus:-}"; then
  CLI_ARGS+=(--num-gpus "$num_gpus")
fi

# ground_truth → --ground-truth (CSV for evaluation; optional)
if _is_set "${ground_truth:-}"; then
  CLI_ARGS+=(--ground-truth "$ground_truth")
fi

# Append any direct command-line arguments passed to this script.
# This allows local dev usage: bash entrypoint.sh --model llama --verbose
# These are added AFTER KFP params, so they take precedence (last wins).
CLI_ARGS+=("$@")

# Log what we received from KFP and what we're about to pass to cli.py.
# <not set> means KFP left the param blank — cli.py will use its defaults.
log "KFP input_params (env var > YAML > unset):"
log "  model:          ${model:-<not set>}"
log "  image_dir:      ${image_dir:-<not set>}"
log "  output:         ${output:-<not set>}"
log "  num_gpus:       ${num_gpus:-<not set>}"
log "  ground_truth:   ${ground_truth:-<not set>}"
log "  clear_prev_out: ${CLEAR_PREV_OUTPUT} (true=delete prev outputs/recompute, false=resume)"
# metadata, system_message, and prompt are KFP input_params reserved for
# future use. They are logged here for visibility but not yet translated
# into CLI_ARGS — cli.py does not currently consume them.
log "  metadata:       ${metadata:-<not set>}"
log "  system_message: ${system_message:-<not set>}"
log "  prompt:         ${prompt:-<not set>}"
log ""
log "Resolved CLI args: ${CLI_ARGS[*]:-<none>}"
log ""

# ---- Per-stage flag builders ---- #
# Each stages.X CLI expects different flag names than cli.py. Build
# stage-specific argument arrays from the same env vars so both the
# orchestrated `run_batch_inference` path and the standalone KFP stage
# pods produce identical invocations.
#
# Optional flags use bash arrays so they expand to nothing when unset
# (safer than string interpolation under `set -o nounset`).
OPT_MODEL=()
if _is_set "${model:-}"; then
  OPT_MODEL=(--model "$model")
fi
# Resolve the output root. Every intermediate JSONL lives here so that
# re-runs of a single stage can read the upstream artefacts written by
# a previous run (this is also what the KFP pod volume mount sees).
OUT_ROOT="${output:-./outputs}"
CLASSIFICATIONS="${OUT_ROOT}/classifications.jsonl"
RAW_EXTRACTIONS="${OUT_ROOT}/raw_extractions.jsonl"
CLEAN_EXTRACTIONS="${OUT_ROOT}/cleaned_extractions.jsonl"
EVAL_DIR="${OUT_ROOT}/evaluation"
# GPU inference elapsed time (seconds) — written by GPU stages, read by
# evaluate. Each GPU stage appends its elapsed seconds to this file (one
# line per stage). The evaluate pod sums the values to compute true GPU
# throughput without including CPU phases (clean, evaluate).
INFERENCE_ELAPSED_FILE="${OUT_ROOT}/.inference_elapsed"

# #############################################################################
#  TASK DISPATCH
# #############################################################################

# ---- Task Dispatch ---- #
# KFP sets $KFP_TASK to the current stage name from workflow_definition.
# Each case must match a task name defined in the kfp_manifest.
# Fail fast if the task is unknown — never silently skip work.
log "KFP_TASK: ${KFP_TASK:-<not set>}"
log ""

case "${KFP_TASK:-}" in
  # ========================================================================
  # LOCAL DEV ONLY — NOT used by the KFP pipeline.
  # ========================================================================
  # In production, KFP runs each stage in its own pod by setting
  # KFP_TASK=classify / extract / clean / evaluate (see branches below).
  # The `run_batch_inference` branch chains all stages in a single shell
  # for sandbox/laptop iteration — it does NOT appear in the KFP DAG and
  # should never be set by the KFP manifest. Keep it for local smoke tests.
  # ========================================================================
  run_batch_inference)
    # Local simulation of the KFP pipeline.
    # Each phase is a FRESH python3 process: model loads, runs, exits,
    # CUDA context is torn down and GPU memory fully released before the
    # next phase starts. This mirrors pod-per-stage KFP deployment and
    # isolates GPU state between phases (no fragmentation leak across
    # classify → extract → clean → evaluate).
    log "Mode: run_batch_inference — simulating 4-stage KFP pipeline locally."
    log "Output root: $OUT_ROOT"
    mkdir -p "$OUT_ROOT"

    # Clear previous outputs only when CLEAR_PREV_OUTPUT=true; otherwise the
    # extract stage resumes and skips already-processed images from a prior run.
    _clear_prev_output "$CLASSIFICATIONS" "$RAW_EXTRACTIONS" "$CLEAN_EXTRACTIONS" \
      "${EVAL_DIR}/evaluation_results.jsonl" "$INFERENCE_ELAPSED_FILE"

    # Track GPU inference wall-clock (classify + extract only, not clean/evaluate).
    INFERENCE_START=$(date +%s)

    _banner "Phase 1/4: classify (fresh process, model reload)"
    _run_classify
    log "Phase 1/4: classify complete."

    _banner "Phase 2/4: extract (fresh process, model reload)"
    _run_extract classified
    log "Phase 2/4: extract complete."

    INFERENCE_ELAPSED=$(($(date +%s) - INFERENCE_START))
    log "GPU inference elapsed: ${INFERENCE_ELAPSED}s"

    _run_cpu_stages "Phase 3/4" "Phase 4/4"

    log ""
    log "Pipeline completed successfully."
    ;;

  # ========================================================================
  # LOCAL DEV — Robust probe-based pipeline (3 stages, 1 GPU process).
  # ========================================================================
  # Skips the separate classify stage entirely. The extract stage
  # with --graph-robust runs two probes per image (document + bank headers)
  # and picks the best type by field count. One GPU process, no wasted
  # classification call.
  #
  # Model calls per type: receipt/invoice=2, travel/logbook=3, bank=4.
  # ========================================================================
  run_graph_robust)
    log "Mode: graph_robust — probe-based classification (3-stage pipeline)."
    log "Output root: $OUT_ROOT"
    mkdir -p "$OUT_ROOT"

    # Clear previous outputs only when CLEAR_PREV_OUTPUT=true; otherwise the
    # extract stage resumes and skips already-processed images from a prior run.
    _clear_prev_output "$RAW_EXTRACTIONS" "$CLEAN_EXTRACTIONS" \
      "${EVAL_DIR}/evaluation_results.jsonl" "$INFERENCE_ELAPSED_FILE"

    # Track GPU inference wall-clock (extract only, not clean/evaluate).
    INFERENCE_START=$(date +%s)

    _banner "Phase 1/3: extract --graph-robust (probe + extract, single GPU process)"
    _run_extract graph_robust
    log "Phase 1/3: extract complete."

    INFERENCE_ELAPSED=$(($(date +%s) - INFERENCE_START))
    log "GPU inference elapsed: ${INFERENCE_ELAPSED}s"

    _run_cpu_stages "Phase 2/3" "Phase 3/3"

    log ""
    log "Pipeline completed successfully."
    ;;

  # ========================================================================
  # KFP PRODUCTION BRANCHES — one per pod in the 4-stage DAG.
  # ========================================================================
  # These are the branches the KFP manifest dispatches to. Each pod sets
  # KFP_TASK=<stage> via its container env and entrypoint.sh routes here.
  # ========================================================================
  # -- Staged pipeline (GPU stages) ------------------------------------------
  classify)
    # Stage 1: Document type detection (GPU).
    # Writes classifications.jsonl — one record per image.
    _banner "Stage 1: classify — detecting document types (GPU)"
    mkdir -p "$OUT_ROOT"
    _clear_prev_output "$CLASSIFICATIONS" "$INFERENCE_ELAPSED_FILE"
    CLASSIFY_START=$(date +%s)
    _run_classify
    # Write classify elapsed to the shared file. The extract pod will
    # append its own elapsed time; evaluate sums all lines.
    echo $(($(date +%s) - CLASSIFY_START)) > "$INFERENCE_ELAPSED_FILE"
    log "Classification complete ($(cat "$INFERENCE_ELAPSED_FILE")s)."
    ;;
  extract)
    # Stage 2: Field extraction (GPU).
    # Writes raw_extractions.jsonl.
    #
    # Two modes, auto-detected:
    #   - Classified: if classifications.jsonl exists (from a prior classify
    #     pod), reads it and runs type-specific extraction per image.
    #   - Graph-robust: if no classifications.jsonl, runs probe-based
    #     classification + extraction in one pass (--graph-robust).
    #     This is the default for the 3-stage KFP DAG (extract/clean/evaluate).
    _banner "Stage 2: extract — extracting fields (GPU)"
    mkdir -p "$OUT_ROOT"
    _clear_prev_output "$RAW_EXTRACTIONS"
    EXTRACT_START=$(date +%s)
    _run_extract auto
    # Append extract elapsed to the shared file (classify may have written
    # the first line). The evaluate pod sums all lines.
    echo $(($(date +%s) - EXTRACT_START)) >> "$INFERENCE_ELAPSED_FILE"
    log "Extraction complete."
    ;;

  # -- Deprecated stages (kept as no-ops for KFP manifest compatibility) ------
  filter)
    # The filter stage was removed — cleaning now handles everything.
    # Keep as a no-op so existing KFP DAGs that include a filter pod
    # don't fail. Remove this case once the KFP manifest is updated.
    log "Stage filter: skipped (deprecated — no longer needed)."
    ;;

  # -- Staged pipeline (CPU stages) ------------------------------------------
  clean)
    # Stage 3: Parse and clean raw responses (CPU only, no GPU needed).
    # Reads raw_extractions.jsonl, writes cleaned_extractions.jsonl.
    _banner "Stage 3: clean — parsing and cleaning raw responses (CPU)"
    mkdir -p "$OUT_ROOT"
    _clear_prev_output "$CLEAN_EXTRACTIONS"
    python3 -m stages.clean \
      --input      "$RAW_EXTRACTIONS" \
      --output-dir "$CLEAN_EXTRACTIONS" || exit $?
    log "Cleaning complete."
    ;;
  evaluate)
    # Stage 4: Evaluation against ground truth (CPU only, no GPU needed).
    # Reads cleaned_extractions.jsonl + ground truth CSV/JSONL, writes evaluation_results.jsonl.
    _banner "Stage 4: evaluate — scoring against ground truth (CPU)"
    mkdir -p "$EVAL_DIR"
    _clear_prev_output "${EVAL_DIR}/evaluation_results.jsonl"
    _read_inference_elapsed "$INFERENCE_ELAPSED_FILE"
    python3 -m stages.evaluate \
      --input        "$CLEAN_EXTRACTIONS" \
      --ground-truth "${ground_truth:?ground_truth env var required}" \
      --output-dir   "$EVAL_DIR" \
      "${INFERENCE_ARGS[@]}" || exit $?
    log "Evaluation complete."
    ;;

  # ========================================================================
  # TRUST DISTRIBUTION LINKING — NRO Private Wealth compliance pipeline.
  # ========================================================================
  trust_classify)
    # Trust document type classification (GPU).
    # Scans a flat directory for CASEXXX_* documents, classifies each
    # via VLM inference, and assembles a quads CSV for trust_extract.
    _banner "Stage 1: trust_classify — classifying trust documents (GPU)"
    _resolve_trust_vars
    log "  trust_data_dir: ${trust_data_dir:-<not set>}  trust_output: ${TRUST_OUT}"
    _clear_prev_output "${trust_classifications:-}" "${trust_quads:-}" \
      "${trust_quads_incomplete:-}" "${TRUST_OUT}/.inference_elapsed"
    CLASSIFY_START=$(date +%s)
    _run_trust_classify
    # Truncate (>) not append (>>): trust_classify is the first GPU stage,
    # mirroring the `classify` branch's elapsed-file write. On a reused KFP
    # volume this resets stale timing from a prior run so trust_evaluate's
    # throughput sum (_read_inference_elapsed) only includes this run.
    echo $(($(date +%s) - CLASSIFY_START)) > "${TRUST_OUT}/.inference_elapsed"
    log "Trust classification complete."
    ;;
  trust_extract)
    # Trust distribution field extraction (GPU-only).
    # Reads a quads CSV (4 documents per case), extracts linking fields
    # via VLM calls. Compliance validation is deferred to trust_clean.
    _banner "Stage 2: trust_extract — extracting trust distribution fields (GPU)"
    _resolve_trust_vars
    log "  trust_output: ${TRUST_OUT}  trust_log_dir: ${TRUST_LOG_DIR}"
    _ensure_trust_quads
    _clear_prev_output "${trust_raw_extractions:-}"
    TRUST_START=$(date +%s)
    _run_trust_extract
    echo $(($(date +%s) - TRUST_START)) >> "${TRUST_OUT}/.inference_elapsed"
    log "Trust extraction complete."
    ;;
  trust_clean)
    # Trust distribution compliance cleaning (CPU-only).
    # Re-parses per-node raw_responses and runs compliance validation.
    _banner "Stage 3: trust_clean — running trust compliance validation (CPU)"
    _resolve_trust_vars
    _clear_prev_output "${trust_compliance_results:-}"
    _run_trust_clean
    log "Trust compliance cleaning complete."
    ;;
  trust_evaluate)
    # Trust distribution evaluation (CPU).
    # Reads trust_compliance_results.jsonl + ground truth YAML, computes compliance metrics.
    _banner "Stage 4: trust_evaluate — scoring trust compliance detection (CPU)"
    _resolve_trust_vars
    log "  trust_output: ${TRUST_OUT}"
    TRUST_EVAL_DIR="${trust_evaluation_dir:?trust_evaluation_dir is required — set via pipeline.trust.evaluation_dir in run_config.yml or trust_evaluation_dir env var}"
    mkdir -p "$TRUST_EVAL_DIR"
    _clear_prev_output "${TRUST_EVAL_DIR}/trust_evaluation_results.jsonl"
    _read_inference_elapsed "${TRUST_OUT}/.inference_elapsed"
    _run_trust_evaluate
    log "Trust evaluation complete."
    ;;
  run_trust_pipeline)
    # Local dev: chain trust_classify + trust_extract + trust_clean + trust_evaluate in one shell.
    log "Mode: run_trust_pipeline — trust distribution compliance pipeline (4-stage)."
    _resolve_trust_vars
    log "  trust_data_dir:      ${trust_data_dir:-<not set>}"
    log "  trust_quads:         ${trust_quads:-<not set>}"
    log "  trust_ground_truth:  ${trust_ground_truth:-<not set>}"
    log "  trust_output:        ${TRUST_OUT}"
    log "  trust_log_dir:       ${TRUST_LOG_DIR}"
    # Clear previous outputs only when CLEAR_PREV_OUTPUT=true; otherwise the
    # GPU stages resume and skip already-processed cases from a prior run.
    _clear_prev_output "${trust_classifications:-}" "${trust_quads:-}" \
      "${trust_quads_incomplete:-}" "${trust_raw_extractions:-}" \
      "${trust_compliance_results:-}" "${TRUST_OUT}/.inference_elapsed"

    TRUST_START=$(date +%s)

    _banner "Phase 1/4: trust_classify (GPU)"
    _run_trust_classify
    log "Phase 1/4: trust_classify complete."

    _banner "Phase 2/4: trust_extract (GPU)"
    _run_trust_extract
    log "Phase 2/4: trust_extract complete."

    INFERENCE_ELAPSED=$(($(date +%s) - TRUST_START))
    log "GPU inference elapsed: ${INFERENCE_ELAPSED}s"

    _banner "Phase 3/4: trust_clean (CPU)"
    _run_trust_clean
    log "Phase 3/4: trust_clean complete."

    if _is_set "${trust_ground_truth:-}"; then
      _banner "Phase 4/4: trust_evaluate (CPU)"
      TRUST_EVAL_DIR="${trust_evaluation_dir:?trust_evaluation_dir is required — set via pipeline.trust.evaluation_dir in run_config.yml or trust_evaluation_dir env var}"
      mkdir -p "$TRUST_EVAL_DIR"
      # Orchestrator path: pass the single wall-clock as INFERENCE_ARGS so
      # _run_trust_evaluate uses the same --inference-seconds flag the per-pod
      # path builds from _read_inference_elapsed.
      INFERENCE_ARGS=(--inference-seconds "$INFERENCE_ELAPSED")
      _run_trust_evaluate
      log "Phase 4/4: trust_evaluate complete."
    else
      log ""
      log "Phase 4/4: trust_evaluate skipped — trust_ground_truth not provided."
    fi

    log ""
    log "Trust pipeline completed successfully."
    ;;

  # ========================================================================
  # TRANSACTION LINKING — receipt/invoice -> bank-statement debit matching.
  # ========================================================================
  # Local dev: chain classify + extract + clean + transaction_link (+ optional
  # evaluate_linking) in one shell. Matcher-first: the algorithmic matcher links
  # against the rows extract already pulled; receipts it can't confidently match
  # fall through to a targeted single-image VLM lookup. Does NOT touch the trust
  # pipeline.
  run_transaction_link)
    log "Mode: run_transaction_link — receipt->bank transaction linking (matcher-first, VLM fallback)."
    _resolve_linking_vars
    log "  linking_data_dir:     ${linking_data_dir:-<not set>}"
    log "  linking_output:       ${linking_output:-<not set>}"
    log "  linking_ground_truth: ${linking_ground_truth:-<not set>}"
    log "  link_out:             ${LINK_OUT}"
    log "  link_log_dir:         ${LINK_LOG_DIR}"
    # Clear previous outputs only when CLEAR_PREV_OUTPUT=true; otherwise the
    # GPU stages resume and skip already-processed images from a prior run.
    _clear_prev_output "$CLASSIFICATIONS" "$RAW_EXTRACTIONS" "$CLEAN_EXTRACTIONS" \
      "${linking_output:-}" "${LINK_OUT}/.inference_elapsed"

    INFERENCE_START=$(date +%s)

    _banner "Phase 1/5: classify (GPU)"
    _run_classify
    log "Phase 1/5: classify complete."

    _banner "Phase 2/5: extract (GPU)"
    _run_extract classified
    log "Phase 2/5: extract complete."

    INFERENCE_ELAPSED=$(($(date +%s) - INFERENCE_START))
    log "GPU inference elapsed: ${INFERENCE_ELAPSED}s"

    _banner "Phase 3/5: clean (CPU)"
    python3 -m stages.clean \
      --input      "$RAW_EXTRACTIONS" \
      --output-dir "$CLEAN_EXTRACTIONS" || exit $?
    log "Phase 3/5: clean complete."

    _banner "Phase 4/5: transaction_link (matcher-first + VLM fallback)"
    python3 -m stages.transaction_link \
      --extractions "$CLEAN_EXTRACTIONS" \
      --output      "${linking_output:?linking_output is required — set via pipeline.linking.output in run_config.yml or linking_output env var}" \
      --data-dir    "${image_dir:?image_dir is required — set via pipeline.linking.data_dir in run_config.yml or linking_data_dir env var}" \
      --config      "$CONFIG_FILE" \
      "${OPT_MODEL[@]}" || exit $?
    log "Phase 4/5: transaction_link complete."

    if _is_set "${linking_ground_truth:-}"; then
      _banner "Phase 5/5: evaluate_linking (CPU)"
      LINK_EVAL_DIR="${linking_evaluation_dir:?linking_evaluation_dir is required — set via pipeline.linking.evaluation_dir in run_config.yml or linking_evaluation_dir env var}"
      mkdir -p "$LINK_EVAL_DIR"
      python3 -m stages.evaluate_linking \
        --input        "${linking_output}" \
        --ground-truth "$linking_ground_truth" \
        --output-dir   "$LINK_EVAL_DIR" || exit $?
      log "Phase 5/5: evaluate_linking complete."
    else
      log ""
      log "Phase 5/5: evaluate_linking skipped — linking_ground_truth not provided."
    fi

    log ""
    log "Transaction linking pipeline completed successfully."
    ;;

  "")
    log "FATAL: KFP_TASK is not set. This script must be run by the KFP pipeline."
    log "  For local dev, set KFP_TASK explicitly:"
    log "  KFP_TASK=run_batch_inference bash entrypoint.sh --model internvl3"
    log ""
    _print_task_help
    exit 1
    ;;
  *)
    log "FATAL: Unknown KFP_TASK '${KFP_TASK}'"
    log ""
    _print_task_help
    exit 1
    ;;
esac