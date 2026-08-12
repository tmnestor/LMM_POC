#!/bin/bash
# =============================================================================
# Full 4-stage pipeline over BOTH evaluation datasets, for ONE model.
# =============================================================================
#
# Runs classify -> extract -> clean -> evaluate over the clean set and then the
# degraded set, for whichever model the CURRENT BRANCH selects.
#
#   git checkout feature/gemma4 && git pull && bash scripts/run_full_comparison.sh
#   git checkout main           && git pull && bash scripts/run_full_comparison.sh
#
# It deliberately does NOT switch branches itself: this file is tracked, so a
# checkout mid-run would replace the script while it is executing.
#
# Everything is passed as environment variables, which entrypoint.sh prefers
# over run_config.yml (see _default_from_yaml). No config edits, so a run leaves
# no working-tree changes to stash before the next checkout.
#
# Why the guards below exist -- each has already cost a run:
#   * model/env mismatch is a QUIET failure. InternVL LOADS in vllm_env3; it
#     just runs on transformers 5.x where ensure_corrected_tokenizer()'s
#     fix_mistral_regex path is unverified, risking a degraded tokenizer on the
#     digit-dense amounts the accuracy baseline rests on. No error, wrong number.
#   * a wrong output directory silently RESUMES: read_completed_images() skips
#     every image whose name is already present, so the run "succeeds" in
#     seconds having done nothing.
#   * log_dir does NOT follow the `output` env var -- it resolves separately, so
#     without LMM_LOG_DIR every run's logs land in one directory.
# =============================================================================

set -o errexit
set -o nounset
set -o pipefail

DATA_ROOT="${LMM_DATA_ROOT:-/home/jovyan/nfs_share/tod_2026/evaluation_data}"
# Regenerated 2026-08-12 (the box stamps these with the UTC date, so a set built
# on the morning of the 13th AEST is named ..._20260812). The answer key differs
# from the set it replaced: LINE_ITEM_PRICES no longer carries a unit price the
# receipt never printed. Earlier dated sets have been deleted, so this is the
# only pair on the box — a wrong date here fails the dataset check below rather
# than scoring against stale truth.
CLEAN_SET="${LMM_CLEAN_SET:-synthetic_20260812}"
DEGRADED_SET="${LMM_DEGRADED_SET:-degraded_20260812}"

cd "$(dirname "$0")/.."

# ---- Which model does this branch select? --------------------------------- #
CONDA_ENV_LINE="$(grep -E '^CONDA_ENV=' entrypoint.sh | head -1)"
CONDA_ENV_PATH="${LMM_CONDA_ENV:-$(printf '%s' "$CONDA_ENV_LINE" | sed -E 's/.*:-([^}]*)\}.*/\1/')}"
CONDA_PY="${CONDA_ENV_PATH}/bin/python"

if [[ ! -x "$CONDA_PY" ]]; then
  echo "FATAL: interpreter not found: $CONDA_PY"
  echo "  What:  the conda env this branch selects does not exist."
  echo "  Where: entrypoint.sh -> CONDA_ENV, or the LMM_CONDA_ENV env var."
  echo "  Expected: a real conda env dir, e.g. /home/jovyan/.conda/envs/vllm_env3"
  echo "  How to fix: create the env, or export LMM_CONDA_ENV to point at one."
  exit 1
fi

if ! MODEL="$("$CONDA_PY" -c "
import yaml
print(yaml.safe_load(open('config/run_config.yml'))['bootstrap']['model']['type'])
" 2>/dev/null)"; then
  echo "FATAL: could not read the selected model."
  echo "  What:  '$CONDA_PY' could not parse config/run_config.yml — most often"
  echo "         because that interpreter has no PyYAML."
  echo "  Where: the conda env at $CONDA_ENV_PATH, and config/run_config.yml"
  echo "         -> bootstrap.model.type"
  echo "  Expected: an interpreter with PyYAML, i.e. the env this branch selects"
  echo "            (vllm_env2 on main, vllm_env3 on feature/gemma4)."
  echo "  How to fix: rebuild the env from conda_envs/, or export LMM_CONDA_ENV"
  echo "              to point at a complete one."
  exit 1
fi

# ---- Guard: the model and the conda env must agree ------------------------- #
case "$MODEL" in
  internvl3*) EXPECTED_ENV="vllm_env2"; SLUG="internvl" ;;
  gemma4*)    EXPECTED_ENV="vllm_env3"; SLUG="gemma" ;;
  *)
    echo "FATAL: unrecognised model '$MODEL'"
    echo "  What:  this script only knows the InternVL and Gemma 4 pairings."
    echo "  Where: config/run_config.yml -> bootstrap.model.type"
    echo "  Expected: a type starting 'internvl3' or 'gemma4'."
    echo "  How to fix: add the model's conda-env pairing to the case block in"
    echo "              scripts/run_full_comparison.sh before running it."
    exit 1
    ;;
esac

if [[ "$CONDA_ENV_PATH" != *"$EXPECTED_ENV"* ]]; then
  echo "FATAL: model/env mismatch — refusing to run."
  echo "  What:  model '$MODEL' needs $EXPECTED_ENV but this branch selects"
  echo "         '$CONDA_ENV_PATH'. This is a QUIET failure if allowed to run:"
  echo "         the engine loads and the numbers are subtly wrong."
  echo "  Where: entrypoint.sh -> CONDA_ENV, and config/run_config.yml ->"
  echo "         bootstrap.model.type. They are branch-bound and must agree."
  echo "  Expected: internvl3* with vllm_env2 (on main), gemma4* with vllm_env3"
  echo "            (on feature/gemma4)."
  echo "  How to fix: check out the branch whose model you meant to run, or"
  echo "              export LMM_CONDA_ENV to the matching env deliberately."
  exit 1
fi

run_one() {
  local label="$1" dataset="$2"
  local out="${DATA_ROOT}/output_${SLUG}_${label}"

  if [[ ! -d "${DATA_ROOT}/${dataset}" ]]; then
    echo "FATAL: dataset not found: ${DATA_ROOT}/${dataset}"
    echo "  What:  the image directory for the '${label}' run is missing."
    echo "  Where: LMM_DATA_ROOT / LMM_${label^^}_SET, currently '${dataset}'."
    echo "  Expected: a directory of images plus its own ground_truth.jsonl."
    echo "  How to fix: regenerate the eval set, or point LMM_DATA_ROOT at the"
    echo "              tree that holds it."
    exit 1
  fi

  echo
  echo "=============================================================="
  echo "  ${MODEL}  |  ${label}  ->  ${out}"
  echo "=============================================================="

  # CLEAR_PREV_OUTPUT: these dirs may hold a previous run, and resume would
  # skip every image rather than recompute. Logs are preserved either way.
  image_dir="${DATA_ROOT}/${dataset}" \
  ground_truth="${DATA_ROOT}/${dataset}/ground_truth.jsonl" \
  output="${out}" \
  LMM_LOG_DIR="${out}/logs" \
  CLEAR_PREV_OUTPUT=true \
  KFP_TASK=run_info_extract \
    bash entrypoint.sh
}

echo "model:      ${MODEL}"
echo "conda env:  ${CONDA_ENV_PATH}"
echo "data root:  ${DATA_ROOT}"
echo "commit:     $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

run_one synthetic "${CLEAN_SET}"
run_one degraded  "${DEGRADED_SET}"

echo
echo "Done. Both runs for ${MODEL} are complete."
echo "Run the other branch to produce the comparison pair."
