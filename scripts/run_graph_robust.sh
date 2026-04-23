#!/usr/bin/env bash
# Run the 3-stage robust extraction pipeline (probe-based classification).
# No separate classification stage -- extraction probes determine doc type.
# Usage: bash scripts/run_graph_robust.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_LOGGING_LEVEL=WARNING
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DATA_DIR="../evaluation_data/synthetic"
ARTIFACTS="../evaluation_data/artifacts/graph_robust"
GROUND_TRUTH="../evaluation_data/synthetic/ground_truth_synthetic.csv"

# ---------------------------------------------------------------------------
# Clean previous outputs
# ---------------------------------------------------------------------------
echo "=== Cleaning previous graph_robust outputs ==="
rm -f "${ARTIFACTS}/raw_extractions.jsonl"
rm -f "${ARTIFACTS}/cleaned_extractions.jsonl"
rm -f "${ARTIFACTS}/evaluation_results.jsonl"
mkdir -p "${ARTIFACTS}"

# ---------------------------------------------------------------------------
# Stage 1+2: Probe + Extract (GPU) -- robust graph
# ---------------------------------------------------------------------------
echo "=== Stage 1+2: Robust probe + extract ==="
python -m stages.extract \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ARTIFACTS}/raw_extractions.jsonl" \
    --graph-robust

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

echo "=== Robust pipeline complete ==="
