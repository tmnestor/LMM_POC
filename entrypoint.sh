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
# THREE TASKS, and `_print_task_help` renders them at runtime (run with
# KFP_TASK unset). The dispatcher's `case` near the bottom is the source of
# truth.
#
#   classify   screen image quality (GPU; shards across every GPU it is given)
#   evaluate   score the screen against ground truth (CPU ONLY -- no model)
#   screen     both of the above, in one shell. SANDBOX ONLY.
#
# The first two are the KFP pods. They are separate because they want opposite
# hardware -- the manifest gives classify every GPU and evaluate none.
#
# `screen` exists because the sandbox is one box rather than a DAG: there is no
# pod boundary to hand artifacts across, so two commands buy nothing and cost a
# chance to mistype the second one's paths. It must NEVER be set by the KFP
# manifest, where it would run the CPU-only scoring inside the GPU pod. It
# calls the same two functions the KFP branches call, so the sandbox exercises
# what production runs rather than a second spelling of it.
#
# There is no clean stage between them. `clean` existed to normalise free-text
# field values before comparison; the screen's answers are fixed tokens, so
# there is nothing to normalise.
#
# `classify` no longer classifies document types -- it screens image quality.
# The name is kept because the production DAG dispatches it, and renaming would
# require this repo and the KFP manifest to land in lockstep.
#
# Sandbox -- one command:
#   KFP_TASK=screen image_dir=<corpus> ground_truth=<corpus>/quality_ground_truth.jsonl \
#     output=<run-dir> bash entrypoint.sh
#
# KFP -- two pods:
#   KFP_TASK=classify image_dir=<corpus> output=<run-dir> bash entrypoint.sh
#   KFP_TASK=evaluate ground_truth=<corpus>/quality_ground_truth.jsonl output=<run-dir> bash entrypoint.sh
#
# Optional overrides for comparison runs, all env vars:
#   screen_variant     prompt variant, overriding run_config
#   screen_min_tiles   tile floor (the lever for small images)
#   screen_max_tiles   tile ceiling
#   screen_max_images  screen only the first N images by filename (smoke tests)
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
# YAML_* variable used below — the log dir and the data/model paths. It emits
# exactly the six names read here and no others; a test asserts that
# correspondence in both directions.
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
  #
  # The example commands print RESOLVED paths, not <placeholders>. This runs
  # after config resolution, so image_dir/ground_truth/output already hold
  # whatever run_config.yml supplied -- and the whole point of reaching this
  # function is that the operator got the invocation wrong, which is the worst
  # moment to hand them a command they still have to fill in by hand. A
  # placeholder is shown only for a value the YAML genuinely does not carry,
  # and then it is marked so, rather than being printed as though it were real.
  local ex_image_dir ex_ground_truth ex_output
  ex_image_dir="${image_dir:-}"
  ex_ground_truth="${ground_truth:-}"
  ex_output="${output:-}"
  _is_set "$ex_image_dir"    || ex_image_dir="<SET-ME: image dir>"
  _is_set "$ex_ground_truth" || ex_ground_truth="<SET-ME: labels .jsonl>"
  _is_set "$ex_output"       || ex_output="<SET-ME: run dir>"

  log "  KFP tasks (one pod each):"
  log "    classify   — Stage 1: screen image quality (GPU; shards across all available)"
  log "    evaluate   — Stage 2: score the screen against ground truth (CPU only)"
  log ""
  log "  Sandbox / local task (one box, both stages, NOT in the KFP manifest):"
  log "    screen     — classify then evaluate, in a single shell"
  log ""
  log "  Preflight (no GPU, no work, exits 0):"
  log "    check      — print the resolved configuration and stop"
  log ""
  log "  There is no clean stage between them. The screen's answers are fixed"
  log "  tokens, so there is nothing to normalise."
  log ""
  log "  Values below come from ${CONFIG_FILE:-run_config.yml} unless overridden by env."
  log ""
  log "  Sandbox — one command:"
  log "    KFP_TASK=screen \\"
  log "      image_dir=${ex_image_dir} \\"
  log "      ground_truth=${ex_ground_truth} \\"
  log "      output=${ex_output} bash entrypoint.sh"
  log ""
  log "    Add screen_max_images=30 for a smoke test (first 30 images by name;"
  log "    the report then counts the rest as missing, so it cannot be mistaken"
  log "    for a full run)."
  log ""
  log "  KFP — two pods:"
  log "    KFP_TASK=classify image_dir=${ex_image_dir} output=${ex_output} bash entrypoint.sh"
  log "    KFP_TASK=evaluate ground_truth=${ex_ground_truth} \\"
  log "      output=${ex_output} bash entrypoint.sh"
}

_clear_prev_output() {
  # CLEAR_PREV_OUTPUT=true (the DEFAULT): delete the listed output artifacts so
  # the stage rescreens from scratch. Explicitly false: the classify stage
  # resumes, screening only images with no record yet.
  #
  # Resume is worth turning on for the recurring production run -- this is a
  # pipeline, new images arrive over time, and rescreening the whole directory
  # is waste that grows with the corpus. It is opt-in rather than default so
  # that a run never silently inherits an earlier run's output.
  #
  # Resume is safe because a kept record must carry the SAME prompt variant and
  # the SAME tile budget as the run doing the resuming -- both change the
  # answers, so a file mixing two of either is not one run. When the settings
  # have moved, the stage discards everything and rescreens by itself, loudly.
  # That check is in stages/quality_screen.py:partition_for_resume, not here.
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
# Orchestration policy — logging labels, elapsed-file
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
  #   screen_variant     prompt variant, overriding run_config
  #   screen_min_tiles   tile floor. The lever for small images: the
  #                      aspect-ratio match settles a small receipt on about one
  #                      tile, at which resolution heavy damage reads as none.
  #   screen_max_tiles   tile ceiling
  #   screen_max_images  screen only the first N images by filename. For smoke
  #                      tests: a wiring change can be proved on 30 images
  #                      rather than 330. evaluate still scores against the
  #                      whole ground truth and reports the rest as missing,
  #                      so a short run cannot be mistaken for a full one.
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
  if [ -n "${screen_max_images:-}" ]; then
    screen_args+=(--max-images "$screen_max_images")
  fi
  python3 -m stages.quality_screen \
    --data-dir "${image_dir:?image_dir env var required}" \
    --output   "$QUALITY_SCREEN" \
    "${screen_args[@]}" \
    "${OPT_MODEL[@]}" || exit $?
}

# ---- The two stages, as functions ---- #
# Lifted out of the dispatch `case` so the sandbox's chained `screen` task can
# call BOTH without restating either. Restating them is how the chained path
# drifts from the production one: a flag added to the classify pod and not to
# the local chain means the sandbox stops testing what production runs, and
# nothing says so -- the run completes and reports a number either way.

_stage_classify() {
  # Stage 1: image-quality screen (GPU). Writes quality_screen.jsonl -- one
  # record per image carrying six defect answers, a graded verdict, the raw
  # model response for audit, and the prompt variant that produced it.
  #
  # The task keeps the name `classify` because the production DAG dispatches
  # it; it screens image quality rather than classifying document types.
  _banner "Stage 1: classify — screening image quality (GPU)"
  mkdir -p "$OUT_ROOT"
  # Deleting quality_screen.jsonl is what forces a full rescreen: with the file
  # gone the stage has nothing to resume from. The elapsed file goes with it so
  # evaluate does not report the previous run's GPU seconds.
  _clear_prev_output "$QUALITY_SCREEN" "$INFERENCE_ELAPSED_FILE"
  local classify_start
  classify_start=$(date +%s)
  _run_quality_screen
  # Elapsed GPU seconds, read by evaluate so it can report inference time
  # separately from wall clock (which includes engine startup).
  echo $(($(date +%s) - classify_start)) > "$INFERENCE_ELAPSED_FILE"
  log "Screening complete ($(cat "$INFERENCE_ELAPSED_FILE")s)."
}

_stage_evaluate() {
  # Stage 2: score the screen against the corpus labels. CPU ONLY -- the KFP
  # manifest gives this pod no GPU, so nothing here may load a model. The
  # stage reads its config directly rather than through AppConfig, which
  # would validate a model path this pod cannot see.
  _banner "Stage 2: evaluate — scoring the image-quality screen (CPU)"
  mkdir -p "$EVAL_DIR"
  # The report is rewritten wholesale every run regardless -- it is derived
  # from quality_screen.jsonl, so there is nothing in it worth resuming.
  _clear_prev_output "${EVAL_DIR}/quality_screen_report.json"
  _read_inference_elapsed "$INFERENCE_ELAPSED_FILE"
  python3 -m stages.evaluate_quality_screen \
    --input        "$QUALITY_SCREEN" \
    --ground-truth "${ground_truth:?ground_truth env var required}" \
    --output-dir   "$EVAL_DIR" || exit $?
  log "Evaluation complete."
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
# Controls whether the classify stage starts from a clean slate or resumes:
#   true / unset   → delete previous OUTPUT artifacts (never logs), rescreen all
#   false          → resume: screen only images with no record yet
#
# A CLEAN SLATE IS THE DEFAULT. Resume is the cheaper path and the reason it
# exists is real — this is a pipeline, images arrive over time, and rescreening
# the whole directory is waste that grows with the corpus — but it is opt-in.
# Defaulting to it would mean a run silently inherits whatever an earlier run
# left in the output directory, which is the wrong surprise to hand someone who
# has just changed something and wants to see its effect.
#
# So the deployment that wants the saving asks for it: set
# CLEAR_PREV_OUTPUT=false in the KFP manifest for the recurring production run,
# and leave it alone everywhere else. See _clear_prev_output above for what
# makes resume safe when it is turned on.
#
# Normalize case before validating: an unquoted YAML boolean in the KFP
# manifest (CLEAR_PREV_OUTPUT: true) is often injected into the container env
# as the Python string "True"/"False" — and KFP stringifies unset params as
# the literal "None".  Lowercase so true/True/TRUE (and none/None) all work.
# Use tr, not bash ${x,,}, for macOS bash 3.2 compatibility.
CLEAR_PREV_OUTPUT="${CLEAR_PREV_OUTPUT:-true}"
CLEAR_PREV_OUTPUT="$(printf '%s' "$CLEAR_PREV_OUTPUT" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$CLEAR_PREV_OUTPUT" || "$CLEAR_PREV_OUTPUT" == "none" ]]; then
  # "none" is KFP's spelling of an unset input_param, so it must land on the
  # SAME value as the `:-` default above. Leaving it on the opposite value
  # would mean the toggle behaves one way from a shell and the other way from
  # KFP, which is the kind of difference that is only ever discovered in
  # production.
  CLEAR_PREV_OUTPUT="true"
fi
case "$CLEAR_PREV_OUTPUT" in
  true)  log "CLEAR_PREV_OUTPUT=true — previous output artifacts are DELETED (logs preserved); every image is rescreened." ;;
  false) log "CLEAR_PREV_OUTPUT=false — resuming: only images with no record yet are screened." ;;
  *)
    log "FATAL: CLEAR_PREV_OUTPUT must be 'true' or 'false' (got '${CLEAR_PREV_OUTPUT}')."
    log "  Where: CLEAR_PREV_OUTPUT environment variable (KFP input_param or shell export)."
    log "  Fix:   leave it unset (or true) to rescreen everything, or set it to false to resume."
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
log "  clear_prev_out: ${CLEAR_PREV_OUTPUT} (true=rescreen everything [default], false=resume)"
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
  # KFP PRODUCTION BRANCHES — one per pod.
  # ========================================================================
  # These are the branches the KFP manifest dispatches to. Each pod sets
  # KFP_TASK=<stage> via its container env and entrypoint.sh routes here.
  # The split exists because the two stages want opposite hardware: the
  # screen takes every available GPU, evaluate is given none.
  # ========================================================================
  classify)
    _stage_classify
    ;;

  evaluate)
    _stage_evaluate
    ;;

  # ========================================================================
  # LOCAL DEV / SANDBOX ONLY — NOT used by the KFP pipeline.
  # ========================================================================
  # The sandbox is one box, not a DAG: there are no pods to hand artifacts
  # between, so splitting the run across two commands buys nothing and costs
  # a chance to mistype the second one's paths. This branch runs both stages
  # in a single shell against a single $output dir.
  #
  # It must NEVER appear in the KFP manifest. Under KFP it would put the
  # CPU-only evaluate work inside the GPU pod, holding four cards idle while
  # it scores a JSONL file.
  #
  # It calls the same two functions the production branches above call, so
  # what the sandbox exercises is what production runs -- not a second
  # spelling of it that can drift.
  # ========================================================================
  check)
    # Preflight. Prints the resolved configuration and exits 0, touching no
    # GPU and writing nothing but its own log.
    #
    # It exists because "what will this run against?" had no answer that was
    # not spelled FATAL: the natural way to ask -- running entrypoint.sh with
    # no task -- is indistinguishable from a misconfigured pod, so it exits 1.
    # Everything worth seeing has already been logged above by the time
    # dispatch is reached, so this branch adds nothing but a successful exit.
    _banner "check — resolved configuration only, nothing was run"
    log "Config file:  ${CONFIG_FILE}"
    log "Image dir:    ${image_dir:-<not set>}"
    log "Ground truth: ${ground_truth:-<not set>}"
    log "Output dir:   ${output:-<not set>}"
    log "Log dir:      ${LOG_DIR}"
    log ""
    log "Paths are read from ${CONFIG_FILE} unless overridden by env. Set KFP_TASK"
    log "to one of classify / evaluate / screen to actually run something."
    ;;

  screen)
    # Check evaluate's input BEFORE the GPU run, not after it. Both stages
    # take their required paths from env, and evaluate's `${ground_truth:?}`
    # would otherwise fire once the screening had already finished -- losing
    # a full model load and every image of inference to a missing variable.
    : "${image_dir:?image_dir env var required}"
    : "${ground_truth:?ground_truth env var required (evaluate scores against it)}"
    if [[ ! -f "$ground_truth" ]]; then
      log "FATAL: ground_truth file not found: $ground_truth"
      log "  What:       the labels evaluate scores the screen against are missing."
      log "  Where:      the ground_truth env var passed to this script."
      log "  Expected:   a JSONL file, e.g."
      log "              ground_truth=<corpus>/quality_ground_truth.jsonl"
      log "  How to fix: point ground_truth at the file the corpus generator wrote"
      log "              beside the images, or regenerate the corpus."
      exit 1
    fi
    _stage_classify
    _stage_evaluate
    _banner "Screen complete — $EVAL_DIR"
    ;;

  "")
    log "FATAL: KFP_TASK is not set — this run does not know which stage to perform."
    log ""
    log "  KFP_TASK is NOT a run_config.yml setting, and cannot be. It names which"
    log "  of the stages below THIS pod runs, and both pods read the same"
    log "  run_config.yml — so a value in the YAML would make the classify pod and"
    log "  the evaluate pod do the same work. It comes from the pod's environment,"
    log "  set per-pod by the KFP manifest."
    log ""
    log "  Everything else above resolved correctly; only the stage is missing."
    log "  There is no default: guessing would mean a misconfigured pod finishing"
    log "  successfully having done something other than what was asked."
    log ""
    log "  To see the resolved configuration without running anything:"
    log "    KFP_TASK=check bash entrypoint.sh"
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