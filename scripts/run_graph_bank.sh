#!/usr/bin/env bash
# Run the 4-stage bank extraction pipeline using the graph-based workflow.
# Usage: bash scripts/run_graph_bank.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_LOGGING_LEVEL=WARNING
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="../evaluation_data/bank"
ARTIFACTS="../evaluation_data/artifacts/graph_bank"
CLASSIFICATIONS="../evaluation_data/artifacts/classifications_bank.jsonl"
GROUND_TRUTH="../evaluation_data/bank/ground_truth_bank.csv"

# ---------------------------------------------------------------------------
# Clean previous outputs
# ---------------------------------------------------------------------------
echo "=== Cleaning previous graph_bank outputs ==="
rm -f "${ARTIFACTS}/raw_extractions.jsonl"
rm -f "${ARTIFACTS}/cleaned_extractions.jsonl"
rm -f "${ARTIFACTS}/evaluation_results.jsonl"
mkdir -p "${ARTIFACTS}"

# ---------------------------------------------------------------------------
# Stage 0: Generate classifications (all bank statements)
# ---------------------------------------------------------------------------
if [ ! -f "${CLASSIFICATIONS}" ]; then
    echo "=== Stage 0: Generating classifications ==="
    mkdir -p "$(dirname "${CLASSIFICATIONS}")"
    for img in ${DATA_DIR}/*.png; do
        name=$(basename "$img")
        echo "{\"image_path\": \"$(realpath "$img")\", \"image_name\": \"$name\", \"document_type\": \"BANK_STATEMENT\"}"
    done > "${CLASSIFICATIONS}"
    echo "Wrote $(wc -l < "${CLASSIFICATIONS}") classifications"
fi

# ---------------------------------------------------------------------------
# Stage 1: Extract (GPU) -- graph-bank path
# ---------------------------------------------------------------------------
echo "=== Stage 1: Extract (graph-bank) ==="
python -m stages.extract \
    --classifications "${CLASSIFICATIONS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ARTIFACTS}/raw_extractions.jsonl" \
    --graph-bank

# ---------------------------------------------------------------------------
# Stage 2: Clean (CPU)
# ---------------------------------------------------------------------------
echo "=== Stage 2: Clean ==="
python -m stages.clean \
    --input "${ARTIFACTS}/raw_extractions.jsonl" \
    --output-dir "${ARTIFACTS}/cleaned_extractions.jsonl"

# ---------------------------------------------------------------------------
# Stage 3: Evaluate (CPU)
# ---------------------------------------------------------------------------
echo "=== Stage 3: Evaluate ==="
python -m stages.evaluate \
    --input "${ARTIFACTS}/cleaned_extractions.jsonl" \
    --ground-truth "${GROUND_TRUTH}" \
    --output-dir "${ARTIFACTS}"

echo "=== Graph-bank pipeline complete ==="
