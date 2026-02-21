#!/bin/bash
# =============================================================================
# LMM POC - KFP Pipeline Entrypoint
# =============================================================================
#
# This is the first script that runs when a KFP pipeline job starts.
# Its job is simple: set up the environment, then dispatch to the correct
# cli.py subcommand based on $KFP_TASK.
#
# Flow:
#   KFP Pipeline → Container starts → entrypoint.sh → $KFP_TASK dispatch → cli.py [subcommand]
#
# KFP Tasks:
#   run_batch_inference   Full pipeline (backward compat) — GPU required
#   classify_documents    Classification only → CSV — GPU required
#   extract_documents     Extraction from CSV → JSON — GPU required
#   evaluate_extractions  Evaluation from JSON — CPU only
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
#   3. Locally (dev/debug):     KFP_TASK=<task> bash entrypoint.sh [extra flags]
#
# Local examples:
#   KFP_TASK=run_batch_inference num_gpus=4 batch_size=4 model=internvl3 bash entrypoint.sh
#   KFP_TASK=classify_documents bash entrypoint.sh -d ./images -o ./output --model internvl3
#   KFP_TASK=extract_documents classifications_csv=./output/csv/batch_*_classifications.csv bash entrypoint.sh -o ./output
#   KFP_TASK=evaluate_extractions extractions_json=./output/batch_results/batch_*_extractions.json ground_truth=./gt.csv bash entrypoint.sh -o ./output
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

# ---- Log Configuration ---- #
# All output (stdout + stderr) is captured to a timestamped log file on EFS,
# while still being printed to the console (so KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
#
# Priority: LMM_LOG_DIR env var > run_config.yml logging.log_dir > fail
# No silent fallback — in KFP, pod-local writes are ephemeral/forbidden.
CONFIG_FILE="./config/run_config.yml"
YAML_LOG_DIR=""
if [[ -f "$CONFIG_FILE" ]]; then
  YAML_LOG_DIR=$(grep -A1 '^logging:' "$CONFIG_FILE" | grep 'log_dir:' | sed 's/.*log_dir:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | tr -d "'" | tr -d '"')
fi
LOG_DIR="${LMM_LOG_DIR:-${YAML_LOG_DIR:-}}"
if [[ -z "$LOG_DIR" ]]; then
  echo "FATAL: No log directory configured. Set LMM_LOG_DIR env var or logging.log_dir in $CONFIG_FILE"
  exit 1
fi
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"

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
conda activate /efs/shared/.conda/envs/lmm_poc_env || { log "FATAL: conda activate failed"; exit 1; }

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

# ---- Helper Functions: Build CLI Args from KFP input_params ---- #
# Each helper appends flags for a specific concern so task cases
# compose only the args their subcommand accepts.
#
# KFP injects `input_params` from the pipeline YAML as environment variables.
# The `${var:-}` syntax means "use empty string if unset" which prevents
# `set -o nounset` from erroring on blank KFP params.
#
# IMPORTANT: KFP stringifies Python None as the literal string "None" for
# unset input_params. We must reject both empty AND "None" values.
_is_set() { [[ -n "${1:-}" && "${1}" != "None" ]]; }

_add_common_args() {
  # --model (e.g. "internvl3", "llama")
  # --output-dir (where results, CSVs, and reports are saved)
  if _is_set "${model:-}"; then
    CLI_ARGS+=(--model "$model")
  fi
  if _is_set "${output:-}"; then
    CLI_ARGS+=(--output-dir "$output")
  fi
}

_add_gpu_args() {
  # --num-gpus (0 = auto-detect all, 1 = single GPU, N = use N GPUs)
  # --batch-size (images per batch per GPU; omit for auto-detect)
  if _is_set "${num_gpus:-}"; then
    CLI_ARGS+=(--num-gpus "$num_gpus")
  fi
  if _is_set "${batch_size:-}"; then
    CLI_ARGS+=(--batch-size "$batch_size")
  fi
}

_add_data_args() {
  # --data-dir (path to folder of images to process)
  # --document-types (comma-separated filter)
  # --max-images (cap on number of images)
  if _is_set "${image_dir:-}"; then
    CLI_ARGS+=(--data-dir "$image_dir")
  fi
  if _is_set "${document_types:-}"; then
    CLI_ARGS+=(--document-types "$document_types")
  fi
  if _is_set "${max_images:-}"; then
    CLI_ARGS+=(--max-images "$max_images")
  fi
}

_add_bank_args() {
  # --bank-v2/--no-bank-v2 (V2 bank statement extraction)
  # --balance-correction/--no-balance-correction (balance validation)
  if _is_set "${bank_v2:-}"; then
    if [[ "${bank_v2}" == "true" ]]; then
      CLI_ARGS+=(--bank-v2)
    else
      CLI_ARGS+=(--no-bank-v2)
    fi
  fi
  if _is_set "${balance_correction:-}"; then
    if [[ "${balance_correction}" == "true" ]]; then
      CLI_ARGS+=(--balance-correction)
    else
      CLI_ARGS+=(--no-balance-correction)
    fi
  fi
}

# Log what we received from KFP.
# <not set> means KFP left the param blank — cli.py will use its defaults.
log "KFP input_params:"
log "  model:               ${model:-<not set>}"
log "  image_dir:           ${image_dir:-<not set>}"
log "  output:              ${output:-<not set>}"
log "  num_gpus:            ${num_gpus:-<not set>}"
log "  batch_size:          ${batch_size:-<not set>}"
log "  ground_truth:        ${ground_truth:-<not set>}"
log "  classifications_csv: ${classifications_csv:-<not set>}"
log "  extractions_json:    ${extractions_json:-<not set>}"
log "  bank_v2:             ${bank_v2:-<not set>}"
log "  balance_correction:  ${balance_correction:-<not set>}"
log "  document_types:      ${document_types:-<not set>}"
log "  max_images:          ${max_images:-<not set>}"
# metadata, system_message, and prompt are KFP input_params reserved for
# future use. They are logged here for visibility but not yet translated
# into CLI_ARGS — cli.py does not currently consume them.
log "  metadata:            ${metadata:-<not set>}"
log "  system_message:      ${system_message:-<not set>}"
log "  prompt:              ${prompt:-<not set>}"
log ""

# ---- Task Dispatch ---- #
# KFP sets $KFP_TASK to the current stage name from workflow_definition.
# Each case must match a task name defined in the kfp_manifest.
# Fail fast if the task is unknown — never silently skip work.
log "KFP_TASK: ${KFP_TASK:-<not set>}"
log ""

case "${KFP_TASK:-}" in
  run_batch_inference)
    # Full pipeline (backward compat) — no subcommand.
    # cli.py exit codes — propagated via `|| exit $?` for KFP:
    #   0 = success
    #   1 = config error (bad YAML, missing paths, invalid params)
    #   2 = model loading error (OOM, corrupt weights, missing checkpoint)
    #   3 = processing error (all images failed, fatal crash)
    #   4 = partial success (some images succeeded, some failed)
    CLI_ARGS=()
    _add_common_args; _add_gpu_args; _add_data_args; _add_bank_args
    if _is_set "${ground_truth:-}"; then
      CLI_ARGS+=(--ground-truth "$ground_truth")
    fi
    CLI_ARGS+=("$@")
    log "Resolved CLI args: ${CLI_ARGS[*]:-<none>}"
    log ""
    log "Starting cli.py..."
    python3 ./cli.py "${CLI_ARGS[@]}" || exit $?
    log "Pipeline completed successfully."
    ;;

  classify_documents)
    # GPU: classification only → CSV
    CLI_ARGS=()
    _add_common_args; _add_gpu_args; _add_data_args
    CLI_ARGS+=("$@")
    log "Resolved CLI args: classify ${CLI_ARGS[*]:-<none>}"
    log ""
    log "Starting cli.py classify..."
    python3 ./cli.py classify "${CLI_ARGS[@]}" || exit $?
    log "Classification completed successfully."
    ;;

  extract_documents)
    # GPU: extraction from classification CSV → JSON
    CLI_ARGS=()
    _add_common_args; _add_gpu_args; _add_bank_args
    if _is_set "${classifications_csv:-}"; then
      CLI_ARGS+=(--classifications "$classifications_csv")
    fi
    CLI_ARGS+=("$@")
    log "Resolved CLI args: extract ${CLI_ARGS[*]:-<none>}"
    log ""
    log "Starting cli.py extract..."
    python3 ./cli.py extract "${CLI_ARGS[@]}" || exit $?
    log "Extraction completed successfully."
    ;;

  evaluate_extractions)
    # CPU-ONLY: evaluation from extraction JSON — no model needed
    CLI_ARGS=()
    if _is_set "${output:-}"; then
      CLI_ARGS+=(--output-dir "$output")
    fi
    if _is_set "${extractions_json:-}"; then
      CLI_ARGS+=(--extractions "$extractions_json")
    fi
    if _is_set "${ground_truth:-}"; then
      CLI_ARGS+=(--ground-truth "$ground_truth")
    fi
    CLI_ARGS+=("$@")
    log "Resolved CLI args: evaluate ${CLI_ARGS[*]:-<none>}"
    log ""
    log "Starting cli.py evaluate..."
    python3 ./cli.py evaluate "${CLI_ARGS[@]}" || exit $?
    log "Evaluation completed successfully."
    ;;

  "")
    log "FATAL: KFP_TASK is not set. This script must be run by the KFP pipeline."
    log "  For local dev, set KFP_TASK explicitly:"
    log "  KFP_TASK=run_batch_inference bash entrypoint.sh --model internvl3"
    log "  KFP_TASK=classify_documents bash entrypoint.sh -d ./images -o ./output"
    log "  KFP_TASK=extract_documents classifications_csv=./out/classifications.csv bash entrypoint.sh -o ./output"
    log "  KFP_TASK=evaluate_extractions extractions_json=./out/extractions.json ground_truth=./gt.csv bash entrypoint.sh -o ./output"
    exit 1
    ;;

  *)
    log "FATAL: Unknown KFP_TASK '${KFP_TASK}'"
    log "  Expected one of: run_batch_inference, classify_documents, extract_documents, evaluate_extractions"
    exit 1
    ;;
esac