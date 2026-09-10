#!/bin/bash
# =============================================================================
# LMM POC - KFP Pipeline Entrypoint
# =============================================================================
#
# This is the first script that runs when a KFP pipeline job starts.
# Sets up environment (conda, CUDA, logging), then dispatches to the
# requested KFP_TASK which calls stages/*.py modules directly.
#
# Flow:
#   KFP Pipeline → Container starts → entrypoint.sh → stages.{quality_screen,evaluate_quality_screen}
#
# How KFP passes configuration:
#   The pipeline YAML defines `input_params` (model, image_dir, output, etc.)
#   which users can fill in via the KFP UI. KFP injects these as environment
#   variables into the container. This script reads those env vars and
#   translates them into stages.* command-line flags.
#
#   Example: if a user sets model=llama and num_gpus=4 in the KFP UI:
#     python3 -m stages.quality_screen --model internvl3-vllm ...
#
# TWO TASKS, and `_print_task_help` renders them at runtime (run with KFP_TASK
# unset). The dispatcher's `case` near the bottom is the source of truth.
#
#   classify   screen image quality (GPU; shards across every GPU it is given)
#   evaluate   score the screen against ground truth (CPU ONLY -- no model)
#
# There is no clean stage between them. `clean` existed to normalise free-text
# field values before comparison; the screen's answers are fixed tokens, so
# there is nothing to normalise.
#
# `classify` no longer classifies document types -- it screens image quality.
# The name is kept because the production DAG dispatches it, and renaming would
# require this repo and the KFP manifest to land in lockstep.
#
# Local examples:
#   KFP_TASK=classify image_dir=<corpus> output=<run-dir> bash entrypoint.sh
#   KFP_TASK=evaluate ground_truth=<corpus>/quality_ground_truth.jsonl output=<run-dir> bash entrypoint.sh
#
# Optional overrides for comparison runs, all env vars:
#   screen_variant     prompt variant, overriding run_config
#   screen_min_tiles   tile floor (the lever for small images)
#   screen_max_tiles   tile ceiling
#
# =============================================================================

# #############################################################################
#  SETUP: Shell Safety, CUDA, Logging (pre-conda)
# #############################################################################

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

# vLLM attention backend: prefer VllmSpec.attention_backend in model_loader.py
# (passed to LLM() constructor). Env var is a fallback for envs with
# pre-compiled FlashInfer kernels. Omitted by default so vLLM auto-selects.
# export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# Suppress vLLM usage telemetry (avoids TLS cert errors in air-gapped envs).
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

# NCCL shared memory: KFP pods may default /dev/shm to 64 MB, which is
# too small for NCCL's SHM transport under tensor parallelism — after ~11
# images the SHM region fills and NCCL silently deadlocks. The fix is to
# increase /dev/shm in the KFP pod spec (emptyDir medium=Memory, 8Gi+).
# Setting NCCL_SHM_DISABLE=1 does NOT work — on G5 instances without
# NVLink, disabling SHM leaves NCCL with no viable intra-node transport
# and it fails immediately with "unhandled system error".

# ---- Log Configuration ---- #
# All output (stdout + stderr) is captured to a timestamped log file on EFS,
# while still being printed to the console (so KFP UI shows it too).
# Each run creates its own log file, e.g. entrypoint_20260213_143022.log
#
# Priority: LMM_LOG_DIR env var > run_config.yml bootstrap.logging.log_dir > fail
# No silent fallback — in KFP, pod-local writes are ephemeral/forbidden.
CONFIG_FILE="./config/run_config.yml"

# ---- Resolve ALL YAML defaults up front, with the conda env's own python ----
# A SINGLE PyYAML resolver (scripts/resolve_yaml_defaults.py) supplies every
# YAML_* variable used below — the log dir and the data/model paths. It still
# emits keys for the removed flows, harmlessly, until scripts/
# resolve_yaml_defaults.py is trimmed alongside run_config.yml.
# It must run BEFORE the `exec`/`tee` redirect (which needs the log dir),
# but PyYAML lives only inside the conda env, never in system/base python (the
# base env genuinely has no `yaml` on DEV/PROD). So we run the resolver with the
# conda env's OWN interpreter, addressed by path — no `conda activate` needed
# just to launch an interpreter — and activate the env properly further down.
#
# The conda env itself cannot be read from run_config.yml: parsing that YAML
# needs a PyYAML-capable python, which only exists INSIDE this env (chicken-and-
# egg). So CONDA_ENV is bootstrapped from LMM_CONDA_ENV or the default below.
# CONDA_ENV="${LMM_CONDA_ENV:-/efs/shared/.conda/envs/vllm_env}"
CONDA_ENV="${LMM_CONDA_ENV:-/home/jovyan/.conda/envs/vllm_env2}"
CONDA_PY="${CONDA_ENV}/bin/python"
if [[ ! -x "$CONDA_PY" ]]; then
  echo "FATAL: bootstrap interpreter not found: $CONDA_PY"
  echo "  What:  the python used to parse $CONFIG_FILE (it needs PyYAML) is missing."
  echo "  Where: CONDA_ENV='$CONDA_ENV' — from LMM_CONDA_ENV env var, or the default in entrypoint.sh."
  echo "  Fix:   point LMM_CONDA_ENV at a real conda env dir so \$LMM_CONDA_ENV/bin/python exists, e.g."
  echo "           export LMM_CONDA_ENV=/home/jovyan/.conda/envs/vllm_env2"
  exit 1
fi
# Emits YAML_* assignments (every key, unconditionally; missing keys -> ''),
# all read below as ${YAML_*:-} so `set -o nounset` is satisfied.
eval "$("$CONDA_PY" scripts/resolve_yaml_defaults.py "$CONFIG_FILE")"

# Log dir: env var > YAML > fail. One source now that the per-flow log
# directories are gone with their flows.
LOG_DIR="${LMM_LOG_DIR:-${YAML_LOG_DIR:-}}"
if [[ -z "$LOG_DIR" ]]; then
  echo "FATAL: No log directory configured. Set LMM_LOG_DIR env var or bootstrap.logging.log_dir in $CONFIG_FILE"
  exit 1
fi
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/entrypoint_$(date +'%Y%m%d_%H%M%S').log"
# Export so Python (cli.py) can display it in the startup Configuration panel.
export LMM_LOG_FILE="$LOG_FILE"

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

# Prominent stage banner — mirrors the top-level run banner so each phase of a
# multi-stage pipeline stands out in the logs. Leads with a blank line for
# separation; pass the stage label (e.g. "Phase 1/5: classify (GPU)").
_banner() {
  log ""
  log "================================================================="
  log "    $1"
  log "================================================================="
}

# ---- Cleanup Trap ---- #
# `trap ... EXIT` runs this code whenever the script exits — whether it
# succeeds, fails, or gets killed (e.g. OOM). This guarantees you always
# see the exit code and how long the run took, even on crashes.
# $SECONDS is a built-in bash variable that counts elapsed seconds.
SECONDS=0
trap 'rc=$?; echo ""; log "Exited with code $rc after ${SECONDS}s"; log "Log file: $LOG_FILE"' EXIT

# #############################################################################
#  SETUP: Conda Activation & Environment
# #############################################################################

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
# CONDA_ENV was bootstrapped (and its python validated) during the YAML
# resolution above, before the tee redirect.
log "Conda env: $CONDA_ENV"
# Temporarily allow unbound variables — conda activation scripts (e.g. MKL)
# reference variables that may not be set yet.
set +o nounset
conda activate "$CONDA_ENV" || { set -o nounset; log "FATAL: conda activate failed"; exit 1; }
set -o nounset
# Ensure conda's libstdc++ is found before the (older) system copy.
# The vllm_env ships libstdcxx-ng>=12 which provides GLIBCXX_3.4.30
# needed by libzmq; without this, the linker finds /usr/lib64/libstdc++
# first and fails with "version GLIBCXX_3.4.30 not found".
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Suppress verbose INFO logging from vLLM engine, transformers, and tokenizers.
# Override by setting these env vars before running entrypoint.sh.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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

# ---- Route the fix_mistral_regex tokenizer cache to a writable dir ---- #
# vLLM bakes a fix_mistral_regex-corrected tokenizer copy to disk (see
# models/model_loader.ensure_corrected_tokenizer). Its default is ~/.cache,
# which is NOT writable in the KFP prod pod — only the run_config.yml output
# directory (output.dir -> YAML_OUTPUT_DIR) is. Point the cache there so the
# pre-warm below AND the vLLM workers (which inherit this exported env, incl.
# vLLM's spawned EngineCore children) can write it. An operator-set
# LMM_TOKENIZER_CACHE always wins.
if [[ -z "${LMM_TOKENIZER_CACHE:-}" && -n "${YAML_OUTPUT_DIR:-}" ]]; then
  export LMM_TOKENIZER_CACHE="${YAML_OUTPUT_DIR%/}/tokenizer_cache"
  log "Tokenizer cache dir: ${LMM_TOKENIZER_CACHE} (under run_config.yml output.dir)"
fi

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

# ---- Pre-warm fix_mistral_regex-corrected tokenizer (once, before workers) ---- #
# InternVL3.5's Mistral tokenizer ships a buggy whitespace/digit regex that
# corrupts amount tokenization on dense bank statements. vLLM loads its tokenizer
# in the front-end AND every spawned EngineCore child, so a load-time patch can't
# reach the child; instead we bake a fix_mistral_regex-corrected copy to disk
# (models/model_loader.ensure_corrected_tokenizer) and hand vLLM that path.
# Doing it HERE — once, before the DP workers spawn — avoids the parallel workers
# racing to build the shared cache. Idempotent (instant after first build) and
# best-effort: a failure only means the engine falls back to the model's own
# tokenizer, i.e. the prior functional behaviour.
if [[ "${YAML_MODEL_TYPE:-}" == internvl*-vllm && -n "${YAML_MODEL_PATH:-}" ]]; then
  log "Pre-warming fix_mistral_regex-corrected tokenizer (${YAML_MODEL_TYPE})..."
  if _tok_path=$(python3 -c "from models.model_loader import ensure_corrected_tokenizer as e; print(e('${YAML_MODEL_PATH}'))" 2>&1); then
    log "  corrected tokenizer: ${_tok_path}"
  else
    log "  WARNING: tokenizer pre-warm failed — engine load falls back to the model's own tokenizer:"
    log "  ${_tok_path}"
  fi
fi

# #############################################################################
#  HELPERS: Shared Functions
# #############################################################################

# True if $1 is set, non-empty, and not the literal "None".
# KFP stringifies an unset/blank `input_param` as the literal string "None"
# (not ""), so both must be rejected when deciding whether a value was provided.
# Used throughout to gate optional flags and YAML fallbacks under `set -o nounset`.
_is_set() { [[ -n "${1:-}" && "${1}" != "None" ]]; }

_print_task_help() {
  # Single source of truth for the KFP_TASK list, shared by the unset ("")
  # and unknown (*) dispatch branches.  Keep the header comment at the top of
  # this file in sync with this list.
  log "  Available tasks:"
  log "    classify   — Stage 1: screen image quality (GPU; shards across all available)"
  log "    evaluate   — Stage 2: score the screen against ground truth (CPU only)"
  log ""
  log "  There is no clean stage between them. The screen's answers are fixed"
  log "  tokens, so there is nothing to normalise."
  log ""
  log "  Example:"
  log "    KFP_TASK=classify image_dir=<corpus> output=<run-dir> bash entrypoint.sh"
  log "    KFP_TASK=evaluate ground_truth=<corpus>/quality_ground_truth.jsonl \\"
  log "      output=<run-dir> bash entrypoint.sh"
}

_clear_prev_output() {
  # CLEAR_PREV_OUTPUT=true: delete the listed output artifacts so the stage
  # recomputes from scratch.  Unset/false (default): no-op — the stage resumes
  # and skips already-processed images.
  #
  # Why resume-by-default: production drips new images into the image directory
  # over time, so a re-run should process only the NEW arrivals, not reprocess
  # the whole directory.  A full clean-slate run is opt-in via
  # CLEAR_PREV_OUTPUT=true.
  #
  # Only explicit artifact FILES are passed in — never a directory, never a log
  # path — so logs are ALWAYS preserved.
  # Each stage must pass only its OWN outputs, never its inputs.
  [[ "${CLEAR_PREV_OUTPUT}" == "true" ]] || return 0
  local f
  for f in "$@"; do
    [[ -n "$f" && -e "$f" ]] || continue
    rm -f "$f"
    log "  CLEAR_PREV_OUTPUT: removed $f"
  done
}

_read_inference_elapsed() {
  # Read and sum GPU inference elapsed times from a file.
  # Sets INFERENCE_ARGS=() or INFERENCE_ARGS=(--inference-seconds N).
  local elapsed_file="${1:?usage: _read_inference_elapsed <file>}"
  INFERENCE_ARGS=()
  if [[ -f "$elapsed_file" ]]; then
    local total=0
    while IFS= read -r line; do
      total=$((total + line))
    done < "$elapsed_file"
    INFERENCE_ARGS=(--inference-seconds "$total")
    log "GPU inference elapsed: ${total}s (from $elapsed_file)"
  else
    log "WARNING: $elapsed_file not found — throughput will use sum of processing_time."
  fi
}

# ---- Per-stage runners (shared by KFP pods and local orchestrators) ---- #
# Each runner is the single source of truth for one stage's `python3 -m stages.*`
# invocation, called by BOTH its standalone KFP pod branch and the local
# orchestrator that chains stages. This keeps the two paths from drifting (which
# is how the .inference_elapsed >/>> inconsistency originally crept in).
#
# Convention: runners hold ONLY the invocation (+ its required-var diagnostics).
# Orchestration policy — logging labels, CLEAR_PREV_OUTPUT clearing, elapsed-file
# writes — stays in the caller. Runners read globals set by the caller
# (OPT_MODEL and the OUT_ROOT-derived paths).

_run_quality_screen() {
  # GPU. Image-quality screen: six YES/NO defect questions and one graded
  # OVERALL per image, written to $QUALITY_SCREEN for the evaluate stage.
  #
  # There is no clean stage after this one. `clean` normalises free-text field
  # values before comparison, and these answers are already canonical tokens,
  # so the path is classify -> evaluate.
  # Optional overrides, so a diagnostic sweep never requires editing config
  # between runs -- which is how you lose track of which settings produced
  # which output.
  #
  #   screen_variant    prompt variant, overriding run_config
  #   screen_min_tiles  tile floor. The lever for small images: the
  #                     aspect-ratio match settles a small receipt on about one
  #                     tile, at which resolution heavy damage reads as none.
  #   screen_max_tiles  tile ceiling
  local screen_args=()
  if [ -n "${screen_variant:-}" ]; then
    screen_args+=(--variant "$screen_variant")
  fi
  if [ -n "${screen_min_tiles:-}" ]; then
    screen_args+=(--min-tiles "$screen_min_tiles")
  fi
  if [ -n "${screen_max_tiles:-}" ]; then
    screen_args+=(--max-tiles "$screen_max_tiles")
  fi
  python3 -m stages.quality_screen \
    --data-dir "${image_dir:?image_dir env var required}" \
    --output   "$QUALITY_SCREEN" \
    "${screen_args[@]}" \
    "${OPT_MODEL[@]}" || exit $?
}

# #############################################################################
#  CONFIG RESOLUTION (env vars, YAML fallbacks, CLI args)
# #############################################################################

# ---- YAML fallbacks for missing env vars ---- #
# cli.py internally cascades env → YAML → defaults via AppConfig.load.
# But stages/*.py mark their CLI flags as required via typer, so they
# fail before AppConfig runs. Apply the YAML fallback HERE so the stage
# commands below receive a concrete --data-dir / --output-dir value.
# Env var always wins when explicitly set (matches cli.py semantics).
# Assign the YAML fallback ($2) to the named env var ($1) only when the env
# var was not already provided (env always wins, matching cli.py). Uses
# indirect read (${!name}) + `printf -v` write — both available in macOS
# bash 3.2. The `||` short-circuit keeps a false `_is_set` from tripping
# `set -o errexit`.
_default_from_yaml() {
  local name="$1"
  _is_set "${!name:-}" || printf -v "$name" '%s' "$2"
}
_default_from_yaml model                    "${YAML_MODEL_TYPE:-}"
_default_from_yaml image_dir                "${YAML_DATA_DIR:-}"
_default_from_yaml ground_truth             "${YAML_GROUND_TRUTH:-}"
_default_from_yaml output                   "${YAML_OUTPUT_DIR:-}"

# ---- CLEAR_PREV_OUTPUT toggle (validated at startup, before any work) ---- #
# Controls whether stages start from a clean slate or resume:
#   true           → delete previous OUTPUT artifacts (never logs), full recompute
#   false / unset  → resume: skip already-processed images (production default)
# Resume is the default because prod drips new images into the image directory
# over time — a re-run should process only the new arrivals.
#
# Normalize case before validating: an unquoted YAML boolean in the KFP
# manifest (CLEAR_PREV_OUTPUT: true) is often injected into the container env
# as the Python string "True"/"False" — and KFP stringifies unset params as
# the literal "None".  Lowercase so true/True/TRUE (and none/None) all work.
# Use tr, not bash ${x,,}, for macOS bash 3.2 compatibility.
CLEAR_PREV_OUTPUT="${CLEAR_PREV_OUTPUT:-false}"
CLEAR_PREV_OUTPUT="$(printf '%s' "$CLEAR_PREV_OUTPUT" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$CLEAR_PREV_OUTPUT" || "$CLEAR_PREV_OUTPUT" == "none" ]]; then
  CLEAR_PREV_OUTPUT="false"
fi
case "$CLEAR_PREV_OUTPUT" in
  true)  log "CLEAR_PREV_OUTPUT=true — stages will DELETE previous output artifacts (logs preserved) and recompute." ;;
  false) log "CLEAR_PREV_OUTPUT=false — stages will RESUME (already-processed images are skipped)." ;;
  *)
    log "FATAL: CLEAR_PREV_OUTPUT must be 'true' or 'false' (got '${CLEAR_PREV_OUTPUT}')."
    log "  Where: CLEAR_PREV_OUTPUT environment variable (KFP input_param or shell export)."
    log "  Fix:   set CLEAR_PREV_OUTPUT=true for a clean-slate re-run, or leave it unset/false to resume."
    exit 1
    ;;
esac

CLI_ARGS=()

# model → --model (e.g. "internvl3", "llama")
if _is_set "${model:-}"; then
  CLI_ARGS+=(--model "$model")
fi

# image_dir → --data-dir (path to folder of images to process)
if _is_set "${image_dir:-}"; then
  CLI_ARGS+=(--data-dir "$image_dir")
fi

# output → --output-dir (where results, CSVs, and reports are saved)
if _is_set "${output:-}"; then
  CLI_ARGS+=(--output-dir "$output")
fi

# num_gpus → --num-gpus (0 = auto-detect all, 1 = single GPU, N = use N GPUs)
if _is_set "${num_gpus:-}"; then
  CLI_ARGS+=(--num-gpus "$num_gpus")
fi

# ground_truth → --ground-truth (CSV for evaluation; optional)
if _is_set "${ground_truth:-}"; then
  CLI_ARGS+=(--ground-truth "$ground_truth")
fi

# Append any direct command-line arguments passed to this script.
# This allows local dev usage: bash entrypoint.sh --model llama --verbose
# These are added AFTER KFP params, so they take precedence (last wins).
CLI_ARGS+=("$@")

# Log what we received from KFP and what we're about to pass to cli.py.
# <not set> means KFP left the param blank — cli.py will use its defaults.
log "KFP input_params (env var > YAML > unset):"
log "  model:          ${model:-<not set>}"
log "  image_dir:      ${image_dir:-<not set>}"
log "  output:         ${output:-<not set>}"
log "  num_gpus:       ${num_gpus:-<not set>}"
log "  ground_truth:   ${ground_truth:-<not set>}"
log "  clear_prev_out: ${CLEAR_PREV_OUTPUT} (true=delete prev outputs/recompute, false=resume)"
# metadata, system_message, and prompt are KFP input_params reserved for
# future use. They are logged here for visibility but not yet translated
# into CLI_ARGS — cli.py does not currently consume them.
log "  metadata:       ${metadata:-<not set>}"
log "  system_message: ${system_message:-<not set>}"
log "  prompt:         ${prompt:-<not set>}"
log ""
log "Resolved CLI args: ${CLI_ARGS[*]:-<none>}"
log ""

# ---- Per-stage flag builders ---- #
# Each stages.X CLI expects different flag names than cli.py. Build
# stage-specific argument arrays from the same env vars so both the
# orchestrated `run_info_extract` path and the standalone KFP stage
# pods produce identical invocations.
#
# Optional flags use bash arrays so they expand to nothing when unset
# (safer than string interpolation under `set -o nounset`).
OPT_MODEL=()
if _is_set "${model:-}"; then
  OPT_MODEL=(--model "$model")
fi
# Resolve the output root. Every intermediate JSONL lives here so that
# re-runs of a single stage can read the upstream artefacts written by
# a previous run (this is also what the KFP pod volume mount sees).
OUT_ROOT="${output:-./outputs}"
# Written by classify, read by evaluate. Holds one record per image: its
# answers, the raw model response, and the prompt variant that produced them.
QUALITY_SCREEN="${OUT_ROOT}/quality_screen.jsonl"
EVAL_DIR="${OUT_ROOT}/evaluation"
# GPU inference elapsed time (seconds) — written by GPU stages, read by
# evaluate. Each GPU stage appends its elapsed seconds to this file (one
# line per stage). The evaluate pod sums the values to compute true GPU
# throughput without including CPU phases (clean, evaluate).
INFERENCE_ELAPSED_FILE="${OUT_ROOT}/.inference_elapsed"

# #############################################################################
#  TASK DISPATCH
# #############################################################################

# ---- Task Dispatch ---- #
# KFP sets $KFP_TASK to the current stage name from workflow_definition.
# Each case must match a task name defined in the kfp_manifest.
# Fail fast if the task is unknown — never silently skip work.
log "KFP_TASK: ${KFP_TASK:-<not set>}"
log ""

case "${KFP_TASK:-}" in
  # ========================================================================
  # LOCAL DEV ONLY — NOT used by the KFP pipeline.
  # ========================================================================
  # In production, KFP runs each stage in its own pod by setting
  # KFP_TASK=classify / extract / clean / evaluate (see branches below).
  # The `run_info_extract` branch chains all stages in a single shell
  # for sandbox/laptop iteration — it does NOT appear in the KFP DAG and
  # should never be set by the KFP manifest. Keep it for local smoke tests.
  # ========================================================================
  # ========================================================================
  # LOCAL DEV — Robust probe-based pipeline (3 stages, 1 GPU process).
  # ========================================================================
  # Skips the separate classify stage entirely. The extract stage
  # with --graph-robust runs two probes per image (document + bank headers)
  # and picks the best type by field count. One GPU process, no wasted
  # classification call.
  #
  # Model calls per type: receipt/invoice=2, travel/logbook=3, bank=4.
  # ========================================================================
  # ========================================================================
  # KFP PRODUCTION BRANCHES — one per pod in the 4-stage DAG.
  # ========================================================================
  # These are the branches the KFP manifest dispatches to. Each pod sets
  # KFP_TASK=<stage> via its container env and entrypoint.sh routes here.
  # ========================================================================
  # -- Staged pipeline (GPU stages) ------------------------------------------
  classify)
    # Stage 1: image-quality screen (GPU). Writes quality_screen.jsonl -- one
    # record per image carrying six defect answers, a graded verdict, the raw
    # model response for audit, and the prompt variant that produced it.
    #
    # The task keeps the name `classify` because the production DAG dispatches
    # it; it screens image quality rather than classifying document types.
    _banner "Stage 1: classify — screening image quality (GPU)"
    mkdir -p "$OUT_ROOT"
    _clear_prev_output "$QUALITY_SCREEN" "$INFERENCE_ELAPSED_FILE"
    CLASSIFY_START=$(date +%s)
    _run_quality_screen
    # Elapsed GPU seconds, read by evaluate so it can report inference time
    # separately from wall clock (which includes engine startup).
    echo $(($(date +%s) - CLASSIFY_START)) > "$INFERENCE_ELAPSED_FILE"
    log "Screening complete ($(cat "$INFERENCE_ELAPSED_FILE")s)."
    ;;

  evaluate)
    # Stage 2: score the screen against the corpus labels. CPU ONLY -- the KFP
    # manifest gives this pod no GPU, so nothing here may load a model. The
    # stage reads its config directly rather than through AppConfig, which
    # would validate a model path this pod cannot see.
    _banner "Stage 2: evaluate — scoring the image-quality screen (CPU)"
    mkdir -p "$EVAL_DIR"
    _clear_prev_output "${EVAL_DIR}/quality_screen_report.json"
    _read_inference_elapsed "$INFERENCE_ELAPSED_FILE"
    python3 -m stages.evaluate_quality_screen \
      --input        "$QUALITY_SCREEN" \
      --ground-truth "${ground_truth:?ground_truth env var required}" \
      --output-dir   "$EVAL_DIR" || exit $?
    log "Evaluation complete."
    ;;

  "")
    log "FATAL: KFP_TASK is not set. This script must be run by the KFP pipeline."
    log "  For local dev, set KFP_TASK explicitly:"
    log "  KFP_TASK=classify image_dir=<dir> output=<dir> bash entrypoint.sh"
    log ""
    _print_task_help
    exit 1
    ;;
  *)
    log "FATAL: Unknown KFP_TASK '${KFP_TASK}'"
    log ""
    _print_task_help
    exit 1
    ;;
esac