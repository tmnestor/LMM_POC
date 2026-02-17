#!/bin/bash
# =============================================================================
# LMM POC - KFP Pipeline Entrypoint
# =============================================================================
#
# This is the first script that runs when a KFP pipeline job starts.
# Its job is simple: set up the environment, then hand off to cli.py.
#
# Flow:
#   KFP Pipeline → Container starts → entrypoint.sh → cli.py
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
# Two ways to run:
#   1. Via KFP (production):  input_params set in UI → injected as env vars
#   2. Locally (dev/debug):   bash entrypoint.sh --model llama --num-gpus 2
# For example:
#   num_gpus=4 batch_size=4 model=internvl3 image_dir=../evaluation_data/synthetic bash entrypoint.sh
#   Both work — KFP env vars and direct CLI args are merged together.
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

# ---- Build CLI args from KFP input_params ---- #
# KFP injects `input_params` from the pipeline YAML as environment variables.
# We check each one: if it's set and non-empty, add the corresponding
# cli.py flag. The `${var:-}` syntax means "use empty string if unset"
# which prevents `set -o nounset` from erroring on blank KFP params.
CLI_ARGS=()

# model → --model (e.g. "internvl3", "llama")
if [[ -n "${model:-}" ]]; then
  CLI_ARGS+=(--model "$model")
fi

# image_dir → --data-dir (path to folder of images to process)
if [[ -n "${image_dir:-}" ]]; then
  CLI_ARGS+=(--data-dir "$image_dir")
fi

# output → --output-dir (where results, CSVs, and reports are saved)
if [[ -n "${output:-}" ]]; then
  CLI_ARGS+=(--output-dir "$output")
fi

# num_gpus → --num-gpus (0 = auto-detect all, 1 = single GPU, N = use N GPUs)
if [[ -n "${num_gpus:-}" ]]; then
  CLI_ARGS+=(--num-gpus "$num_gpus")
fi

# batch_size → --batch-size (images per batch per GPU; omit for auto-detect)
if [[ -n "${batch_size:-}" ]]; then
  CLI_ARGS+=(--batch-size "$batch_size")
fi

# Append any direct command-line arguments passed to this script.
# This allows local dev usage: bash entrypoint.sh --model llama --verbose
# These are added AFTER KFP params, so they take precedence (last wins).
CLI_ARGS+=("$@")

# Log what we received from KFP and what we're about to pass to cli.py.
# <not set> means KFP left the param blank — cli.py will use its defaults.
log "KFP input_params:"
log "  model:          ${model:-<not set>}"
log "  image_dir:      ${image_dir:-<not set>}"
log "  output:         ${output:-<not set>}"
log "  num_gpus:       ${num_gpus:-<not set>}"
log "  batch_size:     ${batch_size:-<not set>}"
log "  metadata:       ${metadata:-<not set>}"
log "  system_message: ${system_message:-<not set>}"
log "  prompt:         ${prompt:-<not set>}"
log ""
log "Resolved CLI args: ${CLI_ARGS[*]:-<none>}"
log ""

# ---- Run Pipeline ---- #
# Hand off to cli.py which handles all the actual work:
# model loading, image processing, extraction, evaluation, and reporting.
# If cli.py fails (non-zero exit), the `|| exit 1` ensures we exit too,
# which triggers the cleanup trap above to log the failure.
log "Starting cli.py..."
python3 ./cli.py "${CLI_ARGS[@]}" || exit 1
log "Pipeline completed successfully."
