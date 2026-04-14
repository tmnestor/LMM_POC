# Testing the batch-processor-decompose refactor

Branch: `refactor/batch-processor-decompose`

This is a behavior-preserving refactor. Verification = identical F1 scores before/after.

## 1. Local (macOS) -- structural checks only

```bash
# Switch to the branch
git checkout refactor/batch-processor-decompose

# Lint all changed files
conda run -n du ruff check --fix common/batch_types.py common/extraction_evaluator.py common/batch_processor.py cli.py --ignore ARG001,ARG002,F841
conda run -n du ruff format common/batch_types.py common/extraction_evaluator.py common/batch_processor.py cli.py

# Type-check (filter to only our files -- transitive errors are pre-existing)
conda run -n du mypy common/batch_types.py common/extraction_evaluator.py common/batch_processor.py cli.py --ignore-missing-imports 2>&1 | grep -E "^(common/batch_types|common/extraction_evaluator|common/batch_processor|cli)\.py:"
# Expected: no output (zero errors in changed files)

# Verify imports resolve
conda run -n du python -c "from common.batch_types import BatchResult, ImageResult, DetectionResult, ExtractionOutput, BatchStats; print('batch_types OK')"
conda run -n du python -c "from common.extraction_evaluator import ExtractionEvaluator; print('extraction_evaluator OK')"
conda run -n du python -c "from common.batch_processor import create_batch_pipeline, BatchDocumentProcessor, load_document_field_definitions; print('batch_processor OK')"
```

## 2. Remote (GPU) -- behavioral verification

### 2a. Baseline: run on main branch first

```bash
# On the remote machine, capture baseline scores
git checkout feature/multi-gpu

# Bank statements (small set, fast)
python cli.py -c config/run_config.yml --max-images 5 2>&1 | tee /tmp/baseline_bank_5.log

# Full bank eval
python cli.py -c config/run_config.yml 2>&1 | tee /tmp/baseline_bank_full.log

# If you have other data dirs configured:
# python cli.py -c config/run_config.yml --data-dir evaluation_data/synthetic --ground-truth evaluation_data/synthetic/ground_truth_synthetic.csv 2>&1 | tee /tmp/baseline_synthetic.log
```

### 2b. Run on refactored branch

```bash
git checkout refactor/batch-processor-decompose

# Same commands -- compare scores
python cli.py -c config/run_config.yml --max-images 5 2>&1 | tee /tmp/refactor_bank_5.log

# Full bank eval
python cli.py -c config/run_config.yml 2>&1 | tee /tmp/refactor_bank_full.log
```

### 2c. Compare results

```bash
# Quick diff of accuracy lines
diff <(grep -E "Median F1|Mean F1|Weighted" /tmp/baseline_bank_5.log) \
     <(grep -E "Median F1|Mean F1|Weighted" /tmp/refactor_bank_5.log)
# Expected: identical

diff <(grep -E "Median F1|Mean F1|Weighted" /tmp/baseline_bank_full.log) \
     <(grep -E "Median F1|Mean F1|Weighted" /tmp/refactor_bank_full.log)
# Expected: identical
```

## 3. Edge cases to verify

### Sequential mode (batch_size=1)

```bash
python cli.py -c config/run_config.yml --batch-size 1 --max-images 3 2>&1 | tee /tmp/refactor_seq.log
```

### Inference-only (no ground truth)

```bash
python cli.py -c config/run_config.yml --max-images 3 --ground-truth "" 2>&1 | tee /tmp/refactor_infer.log
# Should run without errors, no evaluation output
```

### Llama model (sequential-only processor)

```bash
python cli.py -c config/run_config.yml --model llama --max-images 3 2>&1 | tee /tmp/refactor_llama.log
```

### Multi-GPU (if available)

```bash
python cli.py -c config/run_config.yml --num-gpus 0 2>&1 | tee /tmp/refactor_multigpu.log
```

## 4. What to look for

- **F1 scores must match** between baseline and refactor runs
- **No Python errors** -- especially ImportError or AttributeError
- **batch_stats in summary table** -- "Batch Size (configured)" and "Avg Batch Size (actual)" should appear
- **Bank statements** route correctly -- look for "BANK STATEMENT (sequential):" in logs
- **Progress bar** renders correctly during processing

## 5. New API smoke test (optional)

```python
# In a Python REPL on the remote machine
from common.batch_processor import create_batch_pipeline
from common.batch_types import BatchResult

# The new run() method returns typed BatchResult
# (process_batch() still returns the legacy 3-tuple)
pipeline = create_batch_pipeline(
    model=processor,  # your loaded processor
    prompt_config={},
    ground_truth_csv="evaluation_data/bank/ground_truth_bank.csv",
    batch_size=1,
)
result = pipeline.run(["path/to/image.jpg"])
assert isinstance(result, BatchResult)
assert isinstance(result.results[0].evaluation, dict)
print(f"F1: {result.results[0].evaluation.get('median_f1', 0):.1%}")
```
