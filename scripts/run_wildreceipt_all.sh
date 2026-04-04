#!/bin/bash
# =============================================================================
# WildReceipt Benchmark — Run All Models Unattended
# =============================================================================
#
# Runs each model sequentially, committing and pushing results after each job.
# Safe to run overnight — if the cluster shuts down mid-run, all completed
# jobs will already be committed and pushed.
#
# Usage:
#   bash scripts/run_wildreceipt_all.sh
#   nohup bash scripts/run_wildreceipt_all.sh > wildreceipt_run.log 2>&1 &
#
# =============================================================================

set -o errexit
set -o pipefail

# Unbuffered Python output so logs stream in real time via nohup/tee
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="../data/wildreceipt"
OUTPUT_BASE="evaluation_data/output"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Commit and push results for a completed job
commit_results() {
  local model_name="$1"
  local output_dir="$2"

  log "Committing results for $model_name..."
  git add "$output_dir/" || true
  git commit -m "🗃️ data: add WildReceipt results for $model_name" || {
    log "Nothing to commit for $model_name (no new files)"
    return 0
  }
  git push || {
    log "Push failed for $model_name — trying rebase"
    git pull --rebase && git push
  }
  log "Results for $model_name committed and pushed."
}

# Run a single benchmark job
run_job() {
  local env_name="$1"
  local model="$2"
  local output_dir="$3"
  shift 3
  local extra_env=("$@")

  log "================================================================="
  log "Model: $model | Env: $env_name | Output: $output_dir"
  log "================================================================="

  eval "$(conda shell.bash hook)"
  conda activate "$env_name" || {
    log "FATAL: conda activate $env_name failed — skipping $model"
    return 1
  }

  local rc=0
  if [[ ${#extra_env[@]} -gt 0 ]]; then
    "${extra_env[@]}" python benchmark_wildreceipt.py \
      --model "$model" \
      --data-dir "$DATA_DIR" \
      --output-dir "$output_dir" \
      --save-responses || rc=$?
  else
    python benchmark_wildreceipt.py \
      --model "$model" \
      --data-dir "$DATA_DIR" \
      --output-dir "$output_dir" \
      --save-responses || rc=$?
  fi

  if [[ $rc -eq 0 ]]; then
    log "Benchmark completed: $model"
    commit_results "$model" "$output_dir"
  else
    log "ERROR: Benchmark failed for $model (exit code $rc)"
  fi

  log ""
}

# =============================================================================
# Jobs
# =============================================================================

log "Starting WildReceipt benchmark suite"
log "Data dir: $DATA_DIR"
log ""

# --- HuggingFace models ---
run_job LMM_POC_IVL3.5 internvl3 \
  "$OUTPUT_BASE/wildreceipt_ivl35_8b"

run_job LMM_POC_IVL3.5 internvl3-14b \
  "$OUTPUT_BASE/wildreceipt_ivl35_14b"

run_job LMM_POC_IVL3.5 internvl3-38b \
  "$OUTPUT_BASE/wildreceipt_ivl35_38b"

# --- vLLM models ---
run_job LMM_POC_VLLM internvl3-vllm \
  "$OUTPUT_BASE/wildreceipt_ivl35_8b_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM internvl3-14b-vllm \
  "$OUTPUT_BASE/wildreceipt_ivl35_14b_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM internvl3-38b-vllm \
  "$OUTPUT_BASE/wildreceipt_ivl35_38b_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM llama4scout-w4a16 \
  "$OUTPUT_BASE/wildreceipt_llama4scout" \
  env VLLM_LOGGING_LEVEL=WARNING

# --- Other HF models ---
run_job LMM_POC_NEMOTRON nemotron \
  "$OUTPUT_BASE/wildreceipt_nemotron"

run_job LMM_POC_QWEN35 qwen35 \
  "$OUTPUT_BASE/wildreceipt_qwen35"

run_job LMM_POC_VLLM qwen35-vllm \
  "$OUTPUT_BASE/wildreceipt_qwen35_vllm" \
  env VLLM_LOGGING_LEVEL=WARNING

run_job LMM_POC_VLLM gemma4 \
  "$OUTPUT_BASE/wildreceipt_gemma4" \
  env VLLM_LOGGING_LEVEL=WARNING

log "================================================================="
log "All jobs complete."
log "================================================================="
