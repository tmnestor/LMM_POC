# Autonomous Prompt Tuning — Autoresearch Loop for Document Extraction

## Context

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the
[Claude Code skill port](https://github.com/uditgoenka/autoresearch).  Adapt the
autonomous Modify → Verify → Keep/Discard → Repeat loop to optimise extraction prompts
rather than model weights.

**Goal**: Maximise overall F1 score on the synthetic bank statement evaluation set by
iterating on prompt YAML files, with no human in the loop.

**Hardware**: Remote sandbox (1x L40S) with internet access.
Claude Code runs on the sandbox, orchestrates the loop, and never sees production data.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  program.md                      │
│  (goal, scope, metric, verify cmd, constraints) │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    Autonomous Loop      │
          │  (Claude Code agent)    │
          └────────────┬────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌────────┐      ┌───────────┐     ┌────────────┐
│ Modify │      │  Verify   │     │  Decide    │
│ prompt │ ───► │ cli.py    │ ──► │ keep /     │
│ YAML   │      │ --eval    │     │ discard    │
└────────┘      └───────────┘     └────────────┘
                       │
                       ▼
              results.tsv (gitignored)
```

### Separation of Concerns

| Layer | Who writes | What it does |
|-------|-----------|--------------|
| **program.md** | Human | Goal, scope, metric, constraints, verify command |
| **Autonomous loop** | Agent (Claude Code) | Phase 1–8 protocol (see below) |
| **Evaluation pipeline** | Existing code (read-only) | `cli.py` + F1 scoring — agent NEVER modifies |
| **Prompt YAMLs** | Agent (the ONLY files modified) | Extraction & detection prompts |

---

## Phase 0: Setup & Preconditions

**New files to create:**

| File | Purpose |
|------|---------|
| `autoresearch/program.md` | Goal definition, scope, metric, constraints |
| `autoresearch/run.sh` | Verify script — runs eval, extracts F1 |
| `.gitignore` entry | `autoresearch/results.tsv` |

**program.md contents:**

```markdown
# Autonomous Prompt Tuning — Bank Statement Extraction

## Goal
Maximise overall F1 score on synthetic bank statement evaluation.

## Scope (files the agent MAY edit)
- prompts/internvl3_prompts.yaml
- prompts/llama_prompts.yaml
- prompts/qwen3vl_prompts.yaml
- prompts/llama4scout_prompts.yaml
- (per-model prompt files)

## Scope (files the agent MUST NOT edit)
- cli.py, common/*, models/*, evaluation code
- Ground truth CSVs
- This file (program.md)

## Metric
Overall F1 score (higher is better).  Extracted from cli.py output.

## Verify Command
bash autoresearch/run.sh <model_type>

## Constraints
- Prompts must be GENERIC — no image-specific data (store names, amounts, etc.)
- Examples in prompts must use FICTITIOUS merchant names only
- One atomic change per iteration
- If F1 drops, revert immediately

## Iterations
Unbounded (run overnight) or bounded (e.g., 30 iterations)
```

**run.sh** (the verify command):

```bash
#!/usr/bin/env bash
# Usage: bash autoresearch/run.sh internvl3
# Runs evaluation and extracts the overall F1 score as a single number.
set -euo pipefail

MODEL="${1:?Usage: run.sh <model_type>}"
DATA_DIR="evaluation_data/synthetic"
GT="evaluation_data/synthetic/ground_truth_synthetic.csv"

# Run evaluation, capture output
OUTPUT=$(python cli.py \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --ground-truth "$GT" \
    --skip-visualizations \
    --skip-reports \
    2>&1)

# Extract overall F1 (last occurrence of "F1: X.XXXX" or similar)
F1=$(echo "$OUTPUT" | grep -oP 'Overall F1[:\s]+\K[0-9]+\.[0-9]+' | tail -1)

echo "metric:${F1:-0.0}"
echo "model:${MODEL}"
```

---

## Phase 1–8: The Autonomous Loop

Adapted from the 8-phase protocol.  Each iteration takes ~3–5 minutes
(dominated by GPU inference on 15 synthetic images).

### Phase 1: Review (every iteration)
- Read the current prompt YAML file
- Read last 10 entries from `autoresearch/results.tsv`
- `git log --oneline -10` to see experiment history
- `git diff HEAD~1` if last iteration was "keep"
- Identify: what worked, what failed, what is untried

### Phase 2: Ideate
Priority order:
1. **Fix crashes** from previous iteration
2. **Exploit successes** — variant of a kept change (e.g., if reordering fields helped, try different orderings)
3. **Explore new approaches**:
   - Instruction clarity (more specific wording)
   - Output format constraints (stricter JSON/field templates)
   - Few-shot examples (add/remove/improve fictitious examples)
   - Negative examples ("do NOT include...", "if missing, return NOT_FOUND")
   - Field ordering and grouping
   - Chain-of-thought hints ("first identify the document type, then...")
   - Contextual hints for ambiguous fields
4. **Combine near-misses** — merge two discarded ideas that were close
5. **Simplify** — remove prompt complexity while maintaining F1
6. **Radical experiments** when stuck (>5 consecutive discards)

### Phase 3: Modify (one atomic change)
- Edit ONE prompt YAML file
- ONE focused change, explainable in one sentence
- "One-sentence test": if you need "and" to describe it, split into two iterations

### Phase 4: Commit (before verification)
- `git add prompts/<model>_prompts.yaml`
- `git commit -m "experiment(prompts): <one-sentence description>"`
- NEVER `git add -A`

### Phase 5: Verify
- `bash autoresearch/run.sh <model_type>`
- Parse `metric:` line from output
- Timeout: if >10 minutes, kill and treat as crash

### Phase 6: Decide

| Condition | Action |
|-----------|--------|
| F1 improved (delta > +0.005) | **keep** |
| F1 same (delta within +/-0.005) but simpler prompt | **keep** |
| F1 same or worse | **discard** → `git revert HEAD --no-edit` |
| Crashed | attempt fix (max 3 tries), else **crash** → revert |

**Minimum delta threshold**: 0.005 (0.5%) to avoid noise.  Bank statement extraction
has some non-determinism from field ordering and date formats.

### Phase 7: Log Results
Append to `autoresearch/results.tsv` (gitignored):

```
iteration	commit	f1	delta	status	model	description
0	a1b2c3d	0.945	0.000	baseline	internvl3	initial prompts
1	b2c3d4e	0.952	+0.007	keep	internvl3	add explicit date format hint
2	c3d4e5f	0.948	-0.004	discard	internvl3	remove few-shot example
```

### Phase 8: Repeat
- Go to Phase 1
- If bounded: check `current_iteration < max_iterations`, stop with summary
- If stuck (>5 consecutive discards): re-read ALL prompt files, review entire
  results log, try the OPPOSITE of recent attempts, try radical restructuring

---

## Multi-Model Strategy

Two approaches (choose one per session):

### Option A: Single-Model Deep Dive
- Pick one model (e.g., `internvl3`)
- Run 20–30 iterations on its prompts
- Then apply learnings to other models' prompts manually

### Option B: Round-Robin
- Rotate through models each iteration
- Slower per-model progress but catches cross-model regressions
- Better for discovering universally good prompt patterns

**Recommendation**: Start with Option A on the model with the lowest F1 score,
then apply winning patterns to others.

---

## Guard Rails

### Prompt Authoring Rules (enforced every iteration)
- No image-specific data in prompts (store names, amounts, dates from test images)
- All examples must use fictitious/generic merchant names
- Prompts must work for ANY document, not just the evaluation set

### Overfitting Detection
- If F1 on synthetic set improves dramatically but real-data performance is unknown,
  flag for human review
- Periodically run against the bank evaluation set as a cross-validation check:
  ```bash
  bash autoresearch/run.sh internvl3  # synthetic (primary metric)
  # Every 10 iterations, also run:
  python cli.py --model internvl3 --data-dir evaluation_data/bank \
      --ground-truth evaluation_data/bank/ground_truth_bank.csv
  ```

### Complexity Budget
- If a prompt grows beyond 2x its original length with <2% F1 improvement, simplify
- Track prompt length (chars) alongside F1 in the results log

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `autoresearch/program.md` | Create | Goal, scope, constraints |
| `autoresearch/run.sh` | Create | Verify script (runs eval, extracts F1) |
| `autoresearch/results.tsv` | Auto-generated | Experiment log (gitignored) |
| `.gitignore` | Modify | Add `autoresearch/results.tsv` |
| `prompts/*.yaml` | Modified by agent | The only files the loop edits |

---

## Sandbox Setup

Claude Code runs on the remote sandbox machine (1x L40S, internet access).
Synthetic evaluation data only — no production/private data involved.

### Prerequisites

```bash
# 1. Install Node.js (if not already present)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# 2. Install Claude Code
npm install -g @anthropic-ai/claude-code

# 3. Set API key (add to ~/.bashrc or conda env activate script)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Verify installation
claude --version
```

### Project Setup on Sandbox

```bash
# Clone the repo (or pull latest)
cd /home/jovyan/LMM_POC
git pull origin feature/multi-gpu

# Ensure synthetic evaluation data exists
ls evaluation_data/synthetic/
# Expected: ground_truth_synthetic.csv + synthetic bank statement images

# Activate the model's conda environment
conda activate LMM_POC_IVL3.5   # or whichever model you're tuning

# Create the autoresearch directory
mkdir -p autoresearch

# Make run.sh executable
chmod +x autoresearch/run.sh

# Create a dedicated branch for experiments
git checkout -b autoresearch/prompt-tuning-$(date +%Y%m%d)
```

### Running Claude Code in Headless Mode

For overnight/unattended runs, use `--dangerously-skip-permissions` so Claude Code
doesn't block on permission prompts:

```bash
# Start a tmux/screen session (persists after SSH disconnect)
tmux new -s autoresearch

# Launch Claude Code with the autoresearch prompt
claude --dangerously-skip-permissions \
  -p "Read autoresearch/program.md and start the autonomous prompt tuning loop. \
      Model: internvl3. Max iterations: 30."
```

Alternatively, use `--allowedTools` for more granular control:

```bash
claude --dangerously-skip-permissions \
  --allowedTools "Edit,Read,Bash,Glob,Grep,Write" \
  -p "Read autoresearch/program.md and start the autonomous prompt tuning loop. \
      Model: internvl3. Run until stopped."
```

### After the Run

```bash
# Review results on the sandbox
column -t autoresearch/results.tsv

# Push winning prompt changes to remote
git push origin autoresearch/prompt-tuning-$(date +%Y%m%d)

# On local machine: pull and review
git fetch origin
git log --oneline origin/autoresearch/prompt-tuning-*
git diff main..origin/autoresearch/prompt-tuning-YYYYMMDD -- prompts/

# Cherry-pick or merge winning prompts into feature branch
```

### Data Flow (Privacy Safe)

```
Sandbox (internet + GPU)          Production (no internet)
┌──────────────────────┐          ┌──────────────────────┐
│ Claude Code (API)    │          │                      │
│   ↕                  │          │ Private eval data    │
│ Prompt YAMLs (edit)  │          │ Production models    │
│ Synthetic data (eval)│  ──git──►│ Apply winning        │
│ F1 scores (metric)   │  push    │ prompts manually     │
│ results.tsv (log)    │          │                      │
└──────────────────────┘          └──────────────────────┘
```

- Claude Code only sees: prompt YAML files, synthetic data F1 scores, git history
- Claude Code never sees: production images, bank statements, private data
- Winning prompts are transferred via git push/pull

---

## Running It

### Bounded (recommended for first run)
```
Start the autoresearch loop per autoresearch/program.md.
Model: internvl3
Max iterations: 20
```

### Unbounded (overnight)
```
Start the autoresearch loop per autoresearch/program.md.
Model: internvl3
Run until stopped.
```

### Reviewing Results
```bash
# See experiment history
column -t autoresearch/results.tsv

# See kept experiments only
grep 'keep' autoresearch/results.tsv | column -t

# See git history of prompt changes
git log --oneline -- prompts/
```

---

## Success Criteria

- F1 improvement of 2–5% over baseline on synthetic bank statements
- Prompt changes are generic (no overfitting to evaluation data)
- Learnings transferable across models
- Clear experiment trail in git + results.tsv
