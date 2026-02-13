#!/bin/bash
# =============================================================================
# LMM POC - KFP Pipeline Entrypoint
# Activates conda environment and delegates to cli.py
# =============================================================================
set -o errexit
set -o nounset
set -o pipefail

# ---- Log Configuration ---- #
LMM_LOG_DIR="/efs/shared/data/lmm_poc/logs"
LOG_DIR="$LMM_LOG_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"

# Tee all stdout and stderr to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Timestamped logging function
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# ---- Cleanup trap ---- #
SECONDS=0
trap 'rc=$?; echo ""; log "Exited with code $rc after ${SECONDS}s"; log "Log file: $LOG_FILE"' EXIT

# ---- Banner ---- #
log "================================================================="
log "    Running LMM for Information Extraction"
log "================================================================="
log "Arguments: $*"
log ""

# ---- Conda Activation ---- #
log "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /efs/shared/.conda/envs/lmm_poc_env || { log "FATAL: conda activate failed"; exit 1; }
log "---------------------------------------"
log "Python:  $(which python3)"
log "Version: $(python3 --version 2>&1)"
log "Conda:   $(conda info --envs | grep '*' || echo 'unknown')"
log "GPU:     $(python3 -c 'import torch; print(f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No CUDA")' 2>/dev/null || echo 'torch not available')"
log "---------------------------------------"
log ""

# ---- Run Pipeline ---- #
log "Starting cli.py..."
python3 ./cli.py "$@" || exit 1
log "Pipeline completed successfully."
