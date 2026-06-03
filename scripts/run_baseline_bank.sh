#!/usr/bin/env bash
# Run the 4-stage bank extraction pipeline using the baseline UnifiedBankExtractor.
# Usage: bash scripts/run_baseline_bank.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_LOGGING_LEVEL=WARNING
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="../evaluation_data/bank"
ARTIFACTS="../evaluation_data/artifacts/baseline_bank"
CLASSIFICATIONS="${ARTIFACTS}/classifications.jsonl"
GROUND_TRUTH="../evaluation_data/bank/ground_truth_bank.csv"

# ---------------------------------------------------------------------------
# Clean previous outputs
# ---------------------------------------------------------------------------
echo "=== Cleaning previous baseline_bank outputs ==="
rm -f "${CLASSIFICATIONS}"
rm -f "${ARTIFACTS}/raw_extractions.jsonl"
rm -f "${ARTIFACTS}/cleaned_extractions.jsonl"
rm -f "${ARTIFACTS}/evaluation_results.jsonl"
mkdir -p "${ARTIFACTS}"

# ---------------------------------------------------------------------------
# Stage 1: Classify (GPU)
# ---------------------------------------------------------------------------
echo "=== Stage 1: Classify ==="
python -m stages.classify \
    --data-dir "${DATA_DIR}" \
    --output-dir "${CLASSIFICATIONS}"

# ---------------------------------------------------------------------------
# Stage 2: Extract (GPU) -- baseline path
# ---------------------------------------------------------------------------
echo "=== Stage 2: Extract (baseline) ==="
python -m stages.extract \
    --classifications "${CLASSIFICATIONS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ARTIFACTS}/raw_extractions.jsonl" \
    --no-graph-bank

# ---------------------------------------------------------------------------
# Stage 3: Clean (CPU)
# ---------------------------------------------------------------------------
echo "=== Stage 3: Clean ==="
python -m stages.clean \
    --input "${ARTIFACTS}/raw_extractions.jsonl" \
    --output-dir "${ARTIFACTS}/cleaned_extractions.jsonl"

# ---------------------------------------------------------------------------
# Stage 4: Evaluate (CPU)
# ---------------------------------------------------------------------------
echo "=== Stage 4: Evaluate ==="
python -m stages.evaluate \
    --input "${ARTIFACTS}/cleaned_extractions.jsonl" \
    --ground-truth "${GROUND_TRUTH}" \
    --output-dir "${ARTIFACTS}"

echo "=== Baseline pipeline complete ==="
