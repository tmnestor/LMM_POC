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
# Path Configuration (single source of truth):
#   All operational paths (DATA_DIR, OUTPUT_DIR, GROUND_TRUTH, LOG_DIR) are
#   defined once at the top of this script with KFP override support. They are
#   passed as CLI flags to cli.py, which gives them highest priority (CLI > YAML
#   > ENV > defaults). run_config.yml still has path fields as local-dev
#   fallbacks for direct `python cli.py` runs, but entrypoint.sh never parses
#   them — no grep/sed YAML parsing for paths.
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

# ---- Path Configuration (single source of truth) ---- #
# Canonical paths for this deployment. KFP input_params override defaults.
# These are passed as CLI flags to cli.py, giving them highest priority.
# run_config.yml path fields are only used for direct `python cli.py` runs.
#
# IMPORTANT: KFP stringifies Python None as the literal string "None" for
# unset input_params. _kfp_or() filters both empty and "None" values.
_kfp_or() { [[ -n "${1:-}" && "${1}" != "None" ]] && echo "$1" || echo "$2"; }

DATA_DIR=$(_kfp_or "${image_dir:-}" "/efs/shared/PoC_data/evaluation_data_Feb/bank")
OUTPUT_DIR=$(_kfp_or "${output:-}" "/efs/shared/PoC_data/output")
GROUND_TRUTH=$(_kfp_or "${ground_truth:-}" "/efs/shared/PoC_data/evaluation_data_Feb/bank/ground_truth_bank.csv")
LOG_DIR=$(_kfp_or "${LMM_LOG_DIR:-}" "/efs/shared/PoC_data/output/logs")

# ---- Log Setup ---- #
# All output (stdout + stderr) is captured to a timestamped log file on EFS,
# while still being printed to the console (so KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
CONFIG_FILE="./config/run_config.yml"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"

# ---- Run ID ---- #
# Shared identifier for all stages in this pipeline run. Used to construct
# predictable inter-stage file paths (e.g. batch_{RUN_ID}_classifications.csv).
# KFP can override via run_id env var; otherwise auto-generated from timestamp.
RUN_ID="${run_id:-$(date +'%Y%m%d_%H%M%S')}"

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
log "Log dir: $LOG_DIR"
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
  # --output-dir is always set from resolved OUTPUT_DIR
  # --model (e.g. "internvl3", "llama") is optional — cli.py has defaults
  CLI_ARGS+=(--output-dir "$OUTPUT_DIR")
  if _is_set "${model:-}"; then
    CLI_ARGS+=(--model "$model")
  fi
}

_add_gpu_args() {
  # --num-gpus: always default to 0 (auto-detect all GPUs for data parallelism).
  # KFP can override via num_gpus env var.
  # --batch-size (images per batch per GPU; omit for auto-detect)
  CLI_ARGS+=(--num-gpus "${num_gpus:-0}")
  if _is_set "${batch_size:-}"; then
    CLI_ARGS+=(--batch-size "$batch_size")
  fi
}

_add_data_args() {
  # --data-dir is always set from resolved DATA_DIR
  # --document-types and --max-images are optional filters
  CLI_ARGS+=(--data-dir "$DATA_DIR")
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

log "Run ID: ${RUN_ID}"
log "Resolved paths:"
log "  DATA_DIR:      $DATA_DIR"
log "  OUTPUT_DIR:    $OUTPUT_DIR"
log "  GROUND_TRUTH:  $GROUND_TRUTH"
log "  LOG_DIR:       $LOG_DIR"
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
    CLI_ARGS=(--run-id "$RUN_ID")
    _add_common_args; _add_gpu_args; _add_data_args; _add_bank_args
    if _is_set "$GROUND_TRUTH"; then
      CLI_ARGS+=(--ground-truth "$GROUND_TRUTH")
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
    CLI_ARGS=(--run-id "$RUN_ID")
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
    # Find the latest classifications CSV when not explicitly set.
    # Each KFP stage runs in its own container, so RUN_ID differs per stage.
    # Glob for the most recent file instead of constructing a path.
    if ! _is_set "${classifications_csv:-}"; then
      classifications_csv=$(ls -t "${OUTPUT_DIR}/csv/"*_classifications.csv 2>/dev/null | head -1)
      if [[ -z "${classifications_csv}" ]]; then
        log "FATAL: No classifications CSV found in ${OUTPUT_DIR}/csv/"
        log "  Ensure classify_documents ran successfully, or set classifications_csv explicitly"
        exit 1
      fi
      log "Using latest classifications_csv: ${classifications_csv}"
    fi
    CLI_ARGS=(--run-id "$RUN_ID")
    _add_common_args; _add_gpu_args; _add_bank_args
    CLI_ARGS+=(--classifications "$classifications_csv")
    CLI_ARGS+=("$@")
    log "Resolved CLI args: extract ${CLI_ARGS[*]:-<none>}"
    log ""
    log "Starting cli.py extract..."
    python3 ./cli.py extract "${CLI_ARGS[@]}" || exit $?
    log "Extraction completed successfully."
    ;;

  evaluate_extractions)
    # CPU-ONLY: evaluation from extraction JSON — no model needed
    # Find the latest extractions JSON when not explicitly set.
    # Each KFP stage runs in its own container, so RUN_ID differs per stage.
    if ! _is_set "${extractions_json:-}"; then
      extractions_json=$(ls -t "${OUTPUT_DIR}/batch_results/"*_extractions.json 2>/dev/null | head -1)
      if [[ -z "${extractions_json}" ]]; then
        log "FATAL: No extractions JSON found in ${OUTPUT_DIR}/batch_results/"
        log "  Ensure extract_documents ran successfully, or set extractions_json explicitly"
        exit 1
      fi
      log "Using latest extractions_json: ${extractions_json}"
    fi
    CLI_ARGS=(--run-id "$RUN_ID")
    CLI_ARGS+=(--output-dir "$OUTPUT_DIR")
    CLI_ARGS+=(--extractions "$extractions_json")
    CLI_ARGS+=(--ground-truth "$GROUND_TRUTH")
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