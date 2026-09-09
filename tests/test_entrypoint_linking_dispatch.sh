#!/bin/bash
# Local-only dispatch smoke test for the per-pod transaction-linking KFP stages.
#
# Stubs the conda bootstrap + YAML resolver + python3 so the REAL entrypoint.sh
# runs end-to-end on CPU, then asserts each `KFP_TASK=link_*` pod emits the right
# `python3 -m stages.<x> ...` invocation with linking-dataset paths (proving
# _resolve_linking_vars repointed the shared globals onto LINK_OUT), and that
# link_evaluate passes --inference-seconds summed from .inference_elapsed.
#
# Mirrors the existing classic/trust dispatch smoke checks. Run from anywhere:
#   bash tests/test_entrypoint_linking_dispatch.sh
set -o errexit
set -o nounset
set -o pipefail

# --- Locate repo root (entrypoint.sh lives there) ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

DATA="$TMP/data"
OUT="$TMP/out"                       # LINK_OUT (dirname of linking_output)
OUTPUT="$OUT/transaction_links.jsonl"
GT="$TMP/gt.yml"
EVAL="$TMP/eval"
LOGS="$TMP/logs"
mkdir -p "$DATA" "$OUT" "$LOGS"

# --- Fake conda env interpreter (CONDA_PY) — emits YAML_* on resolver call ---
FAKEENV="$TMP/fakeenv"
mkdir -p "$FAKEENV/bin"
cat > "$FAKEENV/bin/python" <<EOF
#!/bin/bash
# Ignores args; emits the YAML_* assignments entrypoint.sh eval's. EMIT_NO_GT=1
# omits the linking ground truth (to exercise link_evaluate's fail-fast path).
echo "YAML_LOG_DIR='$LOGS'"
echo "YAML_LINKING_LOG_DIR='$LOGS'"
echo "YAML_LINKING_DATA_DIR='$DATA'"
echo "YAML_LINKING_OUTPUT='$OUTPUT'"
echo "YAML_LINKING_EVALUATION_DIR='$EVAL'"
echo "YAML_MODEL_TYPE='internvl3'"   # non-vllm -> skip tokenizer pre-warm
if [[ "\${EMIT_NO_GT:-0}" != "1" ]]; then
  echo "YAML_LINKING_GROUND_TRUTH='$GT'"
fi
EOF
chmod +x "$FAKEENV/bin/python"

# --- Stub bin: conda + python3 ----------------------------------------------
STUB="$TMP/stub"
mkdir -p "$STUB/bin"

cat > "$STUB/bin/conda" <<'EOF'
#!/bin/bash
case "$1" in
  info) echo "* base" ;;     # `conda info --envs | grep '*'`
  *) : ;;                    # shell hook / activate / anything else -> no-op
esac
exit 0
EOF
chmod +x "$STUB/bin/conda"

cat > "$STUB/bin/python3" <<'EOF'
#!/bin/bash
# Record each `python3 -m stages.*` invocation; answer --version for the banner.
if [[ "${1:-}" == "--version" ]]; then
  echo "Python 3.12.5"
  exit 0
fi
printf '%s\n' "$*" >> "$STAGE_CAPTURE"
exit 0
EOF
chmod +x "$STUB/bin/python3"

# --- Shared env for every entrypoint run ------------------------------------
run_task() {
  local task="$1"
  : > "$STAGE_CAPTURE"   # reset capture for this task
  (
    cd "$REPO_ROOT"
    PATH="$STUB/bin:$PATH" \
    LMM_CONDA_ENV="$FAKEENV" \
    CONDA_PREFIX="$FAKEENV" \
    CLEAR_PREV_OUTPUT="false" \
    STAGE_CAPTURE="$STAGE_CAPTURE" \
    KFP_TASK="$task" \
      bash entrypoint.sh
  ) > "$TMP/run_${task}.out" 2>&1
}

FAILS=0
assert_contains() {
  local hay="$1" needle="$2" label="$3"
  if grep -Fq -- "$needle" "$hay"; then
    echo "  PASS: $label"
  else
    echo "  FAIL: $label"
    echo "        expected to find: $needle"
    echo "        in:"
    sed 's/^/          /' "$hay"
    FAILS=$((FAILS + 1))
  fi
}

export STAGE_CAPTURE="$TMP/capture.txt"

echo "== link_classify =="
run_task link_classify
assert_contains "$STAGE_CAPTURE" "-m stages.classify" "classify module invoked"
assert_contains "$STAGE_CAPTURE" "--data-dir $DATA" "classify reads linking data dir"
assert_contains "$STAGE_CAPTURE" "--output-dir $OUT/classifications.jsonl" "classify writes under LINK_OUT"

echo "== link_extract =="
run_task link_extract
assert_contains "$STAGE_CAPTURE" "-m stages.extract" "extract module invoked"
assert_contains "$STAGE_CAPTURE" "--classifications $OUT/classifications.jsonl" "extract reads classifications"
assert_contains "$STAGE_CAPTURE" "--output-dir $OUT/raw_extractions.jsonl" "extract writes under LINK_OUT"

echo "== link_clean =="
run_task link_clean
assert_contains "$STAGE_CAPTURE" "-m stages.clean" "clean module invoked"
assert_contains "$STAGE_CAPTURE" "--input $OUT/raw_extractions.jsonl" "clean reads raw_extractions"
assert_contains "$STAGE_CAPTURE" "--output-dir $OUT/cleaned_extractions.jsonl" "clean writes cleaned under LINK_OUT"

echo "== link =="
run_task link
assert_contains "$STAGE_CAPTURE" "-m stages.transaction_link" "transaction_link module invoked"
assert_contains "$STAGE_CAPTURE" "--extractions $OUT/cleaned_extractions.jsonl" "link reads cleaned_extractions"
assert_contains "$STAGE_CAPTURE" "--output $OUTPUT" "link writes linking_output"
assert_contains "$STAGE_CAPTURE" "--data-dir $DATA" "link reads linking data dir"

echo "== link_evaluate (sums .inference_elapsed) =="
printf '10\n20\n5\n' > "$OUT/.inference_elapsed"   # classify+extract+link = 35
run_task link_evaluate
assert_contains "$STAGE_CAPTURE" "-m stages.evaluate_linking" "evaluate_linking module invoked"
assert_contains "$STAGE_CAPTURE" "--input $OUTPUT" "evaluate reads linking_output"
assert_contains "$STAGE_CAPTURE" "--ground-truth $GT" "evaluate reads linking ground truth"
assert_contains "$STAGE_CAPTURE" "--output-dir $EVAL" "evaluate writes to evaluation dir"
assert_contains "$STAGE_CAPTURE" "--inference-seconds 35" "evaluate passes summed GPU elapsed (35s)"

echo "== link_evaluate fail-fast on missing ground truth =="
rc=0
( : > "$STAGE_CAPTURE"
  cd "$REPO_ROOT"
  PATH="$STUB/bin:$PATH" LMM_CONDA_ENV="$FAKEENV" CONDA_PREFIX="$FAKEENV" \
  CLEAR_PREV_OUTPUT="false" STAGE_CAPTURE="$STAGE_CAPTURE" EMIT_NO_GT=1 \
  KFP_TASK="link_evaluate" bash entrypoint.sh
) > "$TMP/run_noGT.out" 2>&1 || rc=$?
if [[ "$rc" -ne 0 ]]; then
  echo "  PASS: link_evaluate exits non-zero without ground truth (rc=$rc)"
else
  echo "  FAIL: link_evaluate should fail fast without ground truth"
  FAILS=$((FAILS + 1))
fi
assert_contains "$TMP/run_noGT.out" "linking_ground_truth is required" "fail-fast diagnostic shown"

echo ""
if [[ "$FAILS" -eq 0 ]]; then
  echo "ALL DISPATCH CHECKS PASSED"
  exit 0
else
  echo "$FAILS DISPATCH CHECK(S) FAILED"
  exit 1
fi
