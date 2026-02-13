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
#   Example: if a user sets model=llama in the KFP UI, this script runs:
#     python3 ./cli.py --model llama
#
# Two ways to run:
#   1. Via KFP (production):  input_params set in UI → injected as env vars
#   2. Locally (dev/debug):   bash entrypoint.sh --model llama --verbose
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

# ---- Log Configuration ---- #
# All output (stdout + stderr) is captured to a timestamped log file on EFS,
# while still being printed to the console (so KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
#
# LMM_LOG_DIR is set in the KFP manifest under `environment_variables:`.
# Falls back to ./logs if not set (e.g. local dev usage).
LOG_DIR="${LMM_LOG_DIR:-./logs}"
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
log "GPU:     $(python3 -c 'import torch; print(f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No CUDA")' 2>/dev/null || echo 'torch not available')"
log "---------------------------------------"
log ""

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
