#!/bin/bash
# Trace per-GPU memory over time while a workload runs.
#
# Writes a CSV: timestamp,gpu_id,mem_used_mib,mem_free_mib per sample.
# Used to validate the MultiGPU device-pinning fix -- pre-fix runs should
# show GPU 0 memory growing monotonically while GPU 1 stays clean;
# post-fix both GPUs should show similar bounded patterns.
#
# Usage:
#   ./scripts/trace_gpu_memory.sh /tmp/trace_pre.csv &
#   TRACE_PID=$!
#   python3 cli.py --model internvl3 -d evaluation_data/images_5 -o /tmp/out --num-gpus 0
#   kill $TRACE_PID
#
# Then inspect /tmp/trace_pre.csv -- look for GPU 0 with no downward steps
# (fragmentation never released) vs GPU 1 with periodic clean-up dips.

set -o errexit
set -o nounset
set -o pipefail

OUTPUT_CSV="${1:-/tmp/gpu_trace.csv}"
INTERVAL_SECONDS="${2:-2}"

echo "timestamp,gpu_id,mem_used_mib,mem_free_mib" > "$OUTPUT_CSV"

while true; do
  TS=$(date +'%Y-%m-%dT%H:%M:%S')
  nvidia-smi \
    --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader,nounits \
    | while IFS=',' read -r idx used free; do
        printf '%s,%s,%s,%s\n' "$TS" "$(echo "$idx" | xargs)" "$(echo "$used" | xargs)" "$(echo "$free" | xargs)" >> "$OUTPUT_CSV"
      done
  sleep "$INTERVAL_SECONDS"
done
