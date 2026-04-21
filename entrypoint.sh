#!/bin/bash
# =============================================================================
# LMM POC - KFP Pipeline Entrypoint
# =============================================================================
#
# This is the first script that runs when a KFP pipeline job starts.
# Its job is simple: set up the environment, then hand off to run_batch_inference KFP Task which calls cli.py to do the actual work.
#
# Flow:
#   KFP Pipeline → Container starts → entrypoint.sh → KFP Task (classify/filter/extract/clean/evaluate) → cli.py
#
# How KFP passes configuration:
#   The pipeline YAML defines `input_params` (model, image_dir, output, etc.)
#   which users can fill in via the KFP UI. KFP injects these as environment
#   variables into the container. This script reads those env vars and
#   translates them into cli.py command-line flags.
#
#   Example: if a user sets model=llama and num_gpus=4 in the KFP UI:
#     python3 ./cli.py --model llama --num-gpus 4
#
# Three ways to run:
#   1. GitLab CI/CD (standard): git tag triggers pipeline → builds container → deploys to KFP
#   2. Via KFP UI (ad-hoc):     input_params set in UI → injected as env vars
#   3. Locally (dev/debug):     KFP_TASK=run_batch_inference bash entrypoint.sh --model llama
# Local example:
#   KFP_TASK=run_batch_inference num_gpus=4 model=internvl3-vllm bash entrypoint.sh
#
# =============================================================================

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

# vLLM attention backend: use FlashInfer when available (installed in vllm_env).
# Bypasses the legacy SDPA patch in registry.py used by the HF code path.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

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
# Priority: LMM_LOG_DIR env var > run_config.yml logging.log_dir > fail
# No silent fallback — in KFP, pod-local writes are ephemeral/forbidden.
CONFIG_FILE="./config/run_config.yml"
YAML_LOG_DIR=""
YAML_MODEL_TYPE=""
YAML_DATA_DIR=""
YAML_GROUND_TRUTH=""
YAML_OUTPUT_DIR=""
# Only the log_dir is resolved here (pre-conda) because we need it before
# conda is on PATH so we can redirect output to the log file. It's a single
# uncluttered key inside `logging:` with no commented alternatives, so the
# narrow `-A5` window is safe. The other YAML defaults (data/output) are
# resolved in Python AFTER conda activation — see the call to
# scripts/resolve_yaml_defaults.py below — which is immune to comment churn.
if [[ -f "$CONFIG_FILE" ]]; then
  YAML_LOG_DIR=$(grep -A5 '^logging:' "$CONFIG_FILE" | grep -E '^[[:space:]]+log_dir:' | head -1 | sed 's/.*log_dir:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | tr -d "'" | tr -d '"')
fi
LOG_DIR="${LMM_LOG_DIR:-${YAML_LOG_DIR:-}}"
if [[ -z "$LOG_DIR" ]]; then
  echo "FATAL: No log directory configured. Set LMM_LOG_DIR env var or logging.log_dir in $CONFIG_FILE"
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

# ---- Cleanup Trap ---- #
# `trap ... EXIT` runs this code whenever the script exits — whether it
# succeeds, fails, or gets killed (e.g. OOM). This guarantees you always
# see the exit code and how long the run took, even on crashes.
# $SECONDS is a built-in bash variable that counts elapsed seconds.
SECONDS=0
trap 'rc=$?; echo ""; log "Exited with code $rc after ${SECONDS}s"; log "Log file: $LOG_FILE"' EXIT

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
# CONDA_ENV="${LMM_CONDA_ENV:-/efs/shared/.conda/envs/lmm_poc_env}"
# CONDA_ENV="${LMM_CONDA_ENV:-/efs/shared/.conda/envs/vllm_env}"
CONDA_ENV="${LMM_CONDA_ENV:-/home/jovyan/.conda/envs/vllm_env2}"
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

# ---- Resolve YAML defaults (data_dir / ground_truth / output_dir) ---- #
# Uses yaml.safe_load via a tiny Python helper — immune to comment churn
# and whitespace changes in run_config.yml. Must run AFTER conda activate
# so `import yaml` resolves against the project environment. Missing keys
# yield empty strings (safe under `set -u` with `${var:-}` later on).
eval "$(python3 scripts/resolve_yaml_defaults.py "$CONFIG_FILE")"

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

# ---- Build CLI args from KFP input_params ---- #
# KFP injects `input_params` from the pipeline YAML as environment variables.
# We check each one: if it's set and non-empty, add the corresponding
# cli.py flag. The `${var:-}` syntax means "use empty string if unset"
# which prevents `set -o nounset` from erroring on blank KFP params.
#
# IMPORTANT: KFP stringifies Python None as the literal string "None" for
# unset input_params. We must reject both empty AND "None" values.
_is_set() { [[ -n "${1:-}" && "${1}" != "None" ]]; }

# ---- YAML fallbacks for missing env vars ---- #
# cli.py internally cascades env → YAML → defaults via AppConfig.load.
# But stages/*.py mark their CLI flags as required via typer, so they
# fail before AppConfig runs. Apply the YAML fallback HERE so the stage
# commands below receive a concrete --data-dir / --output-dir value.
# Env var always wins when explicitly set (matches cli.py semantics).
if ! _is_set "${model:-}"; then
  model="${YAML_MODEL_TYPE:-}"
fi
if ! _is_set "${image_dir:-}"; then
  image_dir="${YAML_DATA_DIR:-}"
fi
if ! _is_set "${ground_truth:-}"; then
  ground_truth="${YAML_GROUND_TRUTH:-}"
fi
if ! _is_set "${output:-}"; then
  output="${YAML_OUTPUT_DIR:-}"
fi

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
FILTERED_CLASSIFICATIONS="${OUT_ROOT}/classifications_filtered.jsonl"
RAW_EXTRACTIONS="${OUT_ROOT}/raw_extractions.jsonl"
CLEAN_EXTRACTIONS="${OUT_ROOT}/cleaned_extractions.jsonl"
EVAL_DIR="${OUT_ROOT}/evaluation"
# End-to-end wall-clock accountability across the 4 KFP pods.
# The classify pod writes `date +%s` here at start; the evaluate pod reads
# it so the Execution Summary can report true DAG-wide wall-clock rather
# than stage-4-local timing. Lives on EFS so both pods share it.
PIPELINE_START_FILE="${OUT_ROOT}/.pipeline_start"

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
  # The `run_batch_inference` branch chains all 4 stages in a single shell
  # for sandbox/laptop iteration — it does NOT appear in the KFP DAG and
  # should never be set by the KFP manifest. Keep it for local smoke tests.
  # ========================================================================
  run_batch_inference)
    # Local simulation of the 4-pod KFP pipeline.
    # Each phase is a FRESH python3 process: model loads, runs, exits,
    # CUDA context is torn down and GPU memory fully released before the
    # next phase starts. This mirrors pod-per-stage KFP deployment and
    # isolates GPU state between phases (no fragmentation leak across
    # detect → extract → clean → evaluate).
    log "Mode: run_batch_inference — simulating 4-stage KFP pipeline locally."
    log "Output root: $OUT_ROOT"
    mkdir -p "$OUT_ROOT"

    # Pipeline-wide start timestamp (epoch seconds). Written to EFS via
    # PIPELINE_START_FILE so the exact same read-by-evaluate mechanism
    # works in local dev AND in KFP (classify pod writes, evaluate reads).
    # One pattern, one code path.
    date +%s > "$PIPELINE_START_FILE"
    PIPELINE_START=$(cat "$PIPELINE_START_FILE")

    log ""
    log "Phase 1/4: classify (fresh process, model reload)..."
    python3 -m stages.classify \
      --data-dir   "${image_dir:?image_dir env var required}" \
      --output-dir "$CLASSIFICATIONS" \
      "${OPT_MODEL[@]}" || exit $?
    log "Phase 1/4: classify complete."

    # -- Filter disabled for dev testing (all images are bank statements) --
    # log ""
    # log "Phase 1.5/4: filter — removing bank statements..."
    # TOTAL=$(wc -l < "$CLASSIFICATIONS")
    # # jq -c 'select(.document_type | ascii_upcase != "BANK_STATEMENT")' \
    # #   "$CLASSIFICATIONS" > "$FILTERED_CLASSIFICATIONS"
    # python3 -c "
# import json, sys
# with open(sys.argv[1]) as f:
#     lines = [l for l in f if json.loads(l).get('document_type','').upper() != 'BANK_STATEMENT']
# with open(sys.argv[2], 'w') as f:
#     f.writelines(lines)
# " "$CLASSIFICATIONS" "$FILTERED_CLASSIFICATIONS"
    # KEPT=$(wc -l < "$FILTERED_CLASSIFICATIONS")
    # log "Filter: kept $KEPT, dropped $((TOTAL - KEPT)) bank statement(s)"
    # log "Phase 1.5/4: filter complete."

    log ""
    log "Phase 2/4: extract (fresh process, model reload)..."
    python3 -m stages.extract \
      --classifications "$CLASSIFICATIONS" \
      --data-dir        "$image_dir" \
      --output-dir      "$RAW_EXTRACTIONS" \
      "${OPT_MODEL[@]}" || exit $?
    log "Phase 2/4: extract complete."

    log ""
    log "Phase 3/4: clean (CPU, no GPU)..."
    python3 -m stages.clean \
      --input      "$RAW_EXTRACTIONS" \
      --output-dir "$CLEAN_EXTRACTIONS" || exit $?
    log "Phase 3/4: clean complete."

    if _is_set "${ground_truth:-}"; then
      log ""
      log "Phase 4/4: evaluate (CPU, no GPU)..."
      python3 -m stages.evaluate \
        --input             "$CLEAN_EXTRACTIONS" \
        --ground-truth      "$ground_truth" \
        --output-dir        "$EVAL_DIR" \
        --wall-clock-start  "$PIPELINE_START" || exit $?
      log "Phase 4/4: evaluate complete."
    else
      log ""
      log "Phase 4/4: evaluate skipped — ground_truth not provided."
    fi

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
    log "Stage 1: classify — detecting document types..."
    mkdir -p "$OUT_ROOT"
    # Record DAG-wide start epoch on EFS for the evaluate pod to read.
    # Overwrites on every classify invocation so re-runs always use a
    # fresh start time. See PIPELINE_START_FILE definition above.
    date +%s > "$PIPELINE_START_FILE"
    log "Pipeline start epoch written: $(cat "$PIPELINE_START_FILE") -> $PIPELINE_START_FILE"
    python3 -m stages.classify \
      --data-dir   "${image_dir:?image_dir env var required}" \
      --output-dir "$CLASSIFICATIONS" \
      "${OPT_MODEL[@]}" || exit $?
    log "Classification complete."
    ;;
  filter)
    # Stage 1.5: Remove bank statements from classifications (CPU, no GPU).
    # Reads classifications.jsonl, writes classifications_filtered.jsonl.
    log "Stage 1.5: filter — removing bank statements..."
    mkdir -p "$OUT_ROOT"
    TOTAL=$(wc -l < "$CLASSIFICATIONS")
    # jq -c 'select(.document_type | ascii_upcase != "BANK_STATEMENT")' \
    #   "$CLASSIFICATIONS" > "$FILTERED_CLASSIFICATIONS"
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    lines = [l for l in f if json.loads(l).get('document_type','').upper() != 'BANK_STATEMENT']
with open(sys.argv[2], 'w') as f:
    f.writelines(lines)
" "$CLASSIFICATIONS" "$FILTERED_CLASSIFICATIONS"
    KEPT=$(wc -l < "$FILTERED_CLASSIFICATIONS")
    log "Filter: kept $KEPT, dropped $((TOTAL - KEPT)) bank statement(s)"
    log "Filter complete."
    ;;
  extract)
    # Stage 2: Field extraction (GPU).
    # Reads classifications_filtered.jsonl, writes raw_extractions.jsonl.
    # Supports resumption from partial output after crash.
    log "Stage 2: extract — extracting fields (raw responses)..."
    mkdir -p "$OUT_ROOT"
    python3 -m stages.extract \
      --classifications "$FILTERED_CLASSIFICATIONS" \
      --data-dir        "${image_dir:?image_dir env var required}" \
      --output-dir      "$RAW_EXTRACTIONS" \
      "${OPT_MODEL[@]}" || exit $?
    log "Extraction complete."
    ;;

  # -- Staged pipeline (CPU stages) ------------------------------------------
  clean)
    # Stage 3: Parse and clean raw responses (CPU only, no GPU needed).
    # Reads raw_extractions.jsonl, writes cleaned_extractions.jsonl.
    log "Stage 3: clean — parsing and cleaning raw responses..."
    mkdir -p "$OUT_ROOT"
    python3 -m stages.clean \
      --input      "$RAW_EXTRACTIONS" \
      --output-dir "$CLEAN_EXTRACTIONS" || exit $?
    log "Cleaning complete."
    ;;
  evaluate)
    # Stage 4: Evaluation against ground truth (CPU only, no GPU needed).
    # Reads cleaned_extractions.jsonl + ground truth CSV, writes evaluation_results.jsonl.
    log "Stage 4: evaluate — scoring against ground truth..."
    mkdir -p "$EVAL_DIR"
    # End-to-end wall-clock: read the pipeline start epoch written by the
    # classify pod (if present) so the Execution Summary reports true
    # DAG-wide wall-clock. Missing file degrades gracefully — the table
    # simply omits the Wall Clock row.
    WALL_CLOCK_ARGS=()
    if [[ -f "$PIPELINE_START_FILE" ]]; then
      PIPELINE_START=$(cat "$PIPELINE_START_FILE")
      WALL_CLOCK_ARGS=(--wall-clock-start "$PIPELINE_START")
      NOW=$(date +%s)
      log "Pipeline start epoch: $PIPELINE_START (DAG wall-clock so far: $((NOW - PIPELINE_START))s)"
    else
      log "WARNING: $PIPELINE_START_FILE not found — Wall Clock row will be omitted."
      log "         (This is expected when evaluate runs without a prior classify.)"
    fi
    python3 -m stages.evaluate \
      --input        "$CLEAN_EXTRACTIONS" \
      --ground-truth "${ground_truth:?ground_truth env var required}" \
      --output-dir   "$EVAL_DIR" \
      "${WALL_CLOCK_ARGS[@]}" || exit $?
    log "Evaluation complete."
    ;;

  "")
    log "FATAL: KFP_TASK is not set. This script must be run by the KFP pipeline."
    log "  For local dev, set KFP_TASK explicitly:"
    log "  KFP_TASK=run_batch_inference bash entrypoint.sh --model internvl3"
    log ""
    log "  Available tasks:"
    log "    run_batch_inference  — 5-stage simulation (fresh process per phase)"
    log "    classify             — Stage 1: document type detection (GPU)"
    log "    filter               — Stage 1.5: remove bank statements (CPU)"
    log "    extract              — Stage 2: field extraction (GPU)"
    log "    clean                — Stage 3: parse/clean responses (CPU)"
    log "    evaluate             — Stage 4: evaluation (CPU)"
    exit 1
    ;;
  *)
    log "FATAL: Unknown KFP_TASK '${KFP_TASK}'"
    log "  Expected one of: run_batch_inference, classify, filter, extract, clean, evaluate"
    exit 1
    ;;
esac