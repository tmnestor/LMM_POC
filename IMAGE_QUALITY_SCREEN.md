# Image-quality screen — summary

Measured 2026-09-09. Branch `feature/quality-screen`, paired with
`feature/quality-screen-corpus` in Synthetic_Doc_Generation.

---

## What it is

An automated check that looks at a photographed receipt or invoice and decides
whether the image is good enough to process, or should be sent back for
re-capture. It runs on our existing model and GPU infrastructure, as a stage in
the current pipeline.

## Does it work

**Yes, for the pass/fail decision.** Tested on 330 images with known defects:

| | result |
|---|---|
| Poor-quality images correctly flagged | **209 of 220 (95%)** |
| Poor-quality images missed | 11 of 220 (5%) |
| Good images flagged unnecessarily | **1 of 110 (1%)** |

Of the eleven missed, ten were mildly degraded and one was heavily degraded.

**Reproduced independently on production hardware.** An earlier version of the
prompt was run on the production GPUs against a separately generated test set
and gave the same result within noise, on different hardware and different
images. The two checks that do not work (below) failed again in the same way,
confirming those are real limitations rather than quirks of one test set. 330
images take about 11 minutes on four production GPUs.

**Grading *how* bad an image is works reasonably well** — 86% correct across
three levels (good / fair / poor). It separates damaged from undamaged
reliably, and is now also fairly good at telling mild damage from severe.

**It also reports whether the photograph contains more than one receipt.**
Taxpayers commonly place several receipts on a table and photograph them
together, and downstream extraction handles those badly. This is reported
separately from the quality verdict, because a photograph of four receipts is
often perfectly sharp and well lit — nothing is wrong with the *picture* — and
the remedy is different: split it, rather than re-photograph it.

That check is **not yet measured**. The test images contain no such
photographs, so there is nothing for it to find; what we can say is that it
raised no false alarms on any of the 330 single-receipt images. Measuring it
needs test data that contains them, which is the next piece of work.

## What it cannot do yet

**Two of its six checks do not work.** It almost never detects creasing (3
detections out of 110 creased images), and it never detects shadow on receipts
though it detects shadow on invoices perfectly. In both cases we believe the
test images themselves are at fault rather than the model, and we can confirm
that cheaply. The other four checks are sound.

**It has only been tested on generated images.** The test set is synthetic —
damage applied deliberately so we know the right answer for every image. The
screen has never seen a real photograph of a real receipt. This is the main
gap before any production use.

## What it cost

No new infrastructure. It reuses the existing model, GPU setup and pipeline
stages, and runs across both available GPUs. The work was prompt design and
building a labelled test set, not new machine learning.

## One design question, answered

The prompt asks six specific questions (is it blurry, is there a shadow, is the
paper creased…) before asking for an overall verdict. We tested whether those
questions are worth their cost by asking for the verdict alone.

They are: **overall accuracy is 80% with the questions and 48% without.**
Without them the model calls 18 of every 20 *clean* images "moderately
degraded" — the questions give it a reference for what an undamaged document
looks like. The stepwise design was the right call and is now backed by
measurement.

## Recommended next steps

1. **Confirm the two broken checks are test-data problems.** Cheap, and decides
   whether to fix them or drop them to four checks.
2. **Test against real photographs.** Until then we know the screen works on
   images we generated, which is not the same as images users take.
3. **Decide what the screen is for.** If it is "send this back for
   re-capture", the pass/fail decision is ready. If it needs to route by
   severity, the three-level grading needs more work.

## Caveats worth stating

- All results are from one model (InternVL3.5-8B). Not tested on alternatives.
- 330 images from 55 source documents, so fewer independent documents than
  images.
- The screen reports *image quality*, not whether extraction will succeed. We
  have not yet measured whether a flagged image actually extracts worse — that
  is a separate and worthwhile question.

---

## Running this on PROD

Two repositories, both on their own branch. The test images are generated
rather than collected, so any environment can rebuild them from source.
Nothing below is environment-specific: substitute your own paths.

### Part A — build the test corpus

**A1. Clone the corpus generator at the matching branch.**

```bash
git clone --branch feature/quality-screen-corpus \
    https://github.com/tmnestor/Synthetic_Doc_Generation.git
cd Synthetic_Doc_Generation
```

The branch matters. The corpus on `main` is a different set — three document
types, three severity tiers, receipts only degraded, and no quality labels.

**A2. Build the environment.**

```bash
conda env create -f environment.yml
conda activate synthetic
```

**A3. Verify the image libraries before generating.**

```bash
python -c 'import augraphy, cv2, numpy; print(augraphy.__version__, cv2.__version__, numpy.__version__)'
```

Expect augraphy 8.2.6 and OpenCV from the **headless** build. Augraphy declares
the full GUI OpenCV as a dependency, which silently displaces the pinned
headless one and changes rendering. If the wrong build is installed:

```bash
pip uninstall -y opencv-python && pip install --no-deps augraphy==8.2.6
```

**A4. Generate.**

```bash
python -m generators.pipeline eval-set --out <writable-output-parent>
```

Choose an output path the process can actually write to — on an orchestrated
run that generally means the job's own output directory rather than a home or
cache path.

Takes roughly 15 minutes and produces three dated directories totalling ~1.6 GB:

| directory | contents | purpose |
|---|---|---|
| `synthetic_<date>/` | 110 clean images | clean-only comparison runs |
| `degraded_<date>/` | 220 degraded images | degraded-only comparison runs |
| `quality_<date>/` | all 330 | **the quality screen reads this one** |

Each carries `ground_truth.jsonl` (what the document says) and
`quality_ground_truth.jsonl` (which defects each image actually has, plus the
values drawn to produce them).

Generation is deterministic given the ground-truth seeds, so a rebuild produces
the same images. Labels may differ by one or two on criteria whose drawn value
sits within floating-point distance of a threshold, which varies by CPU
architecture; the labels always describe the images actually produced.

### Part B — run the screen

**B1. Clone this repository at the matching branch.**

```bash
git clone --branch feature/quality-screen-standalone \
    https://github.com/tmnestor/LMM_POC.git
cd LMM_POC
```

**B2. Point the config at the local model.** In `config/run_config.yml`, the
model location appears in **three** places and all three must agree:

```yaml
bootstrap:
  model:
    path: <local model checkpoint>          # 1
    default_paths:
      internvl3:      <local model checkpoint>   # 2
      internvl3-vllm: <local model checkpoint>   # 3
```

Startup validates the path exists and fails with a diagnostic naming it, so a
missed one is caught immediately rather than part-way into a run.

Everything else that governs the screen is already set to the measured
configuration and needs no change: prompt variant `quality_screen_v13`, tile
budget `min_tiles: 12 / max_tiles: 12`, token budget 400.

**B3. Check what it resolved to**, before spending any GPU time. Prints the
paths it will use and exits without running anything:

```bash
KFP_TASK=check bash entrypoint.sh
```

**B4. Run it.** Two stages, no clean stage between them — the screen's answers
are fixed tokens with nothing to normalise.

```bash
# GPU. Writes quality_screen.jsonl: one record per image with its answers,
# the raw model response, and the settings that produced it.
KFP_TASK=classify \
    image_dir=<corpus>/quality_<date> \
    output=<run-output-dir> \
    bash entrypoint.sh

# CPU. Scores it and prints the report.
KFP_TASK=evaluate \
    ground_truth=<corpus>/quality_<date>/quality_ground_truth.jsonl \
    output=<run-output-dir> \
    bash entrypoint.sh
```

Always through `entrypoint.sh` — it sets up the environment the stages expect,
and invoking the modules directly does not.

On one box rather than a pipeline, `KFP_TASK=screen` runs both in a single
shell. It must never be set in the KFP manifest, where it would run the
CPU-only scoring inside the GPU pod.

Roughly 20 minutes for 330 images, halving with each additional GPU: the
classify stage shards across every GPU it is given.

**Re-runs.** By default every run rescreens the whole directory. Where images
arrive over time and only the new ones need screening, set
`CLEAR_PREV_OUTPUT=false` — the stage then screens only images with no record
yet. It resumes only when the prompt variant and tile budget are unchanged;
if either has moved it discards the previous records and rescreens, saying so,
because a file mixing two prompts is not one run and the report cannot tell.

**B4. Read the result.** `evaluate` prints the per-criterion table, a
per-document-type split, the severity confusion matrix, and — first, before
the scores — the counts:

```
images 330   scored 330   malformed 0   missing 0   reasoning drift 0
```

Check that line first. Any image not scored means the rates below it describe a
subset rather than the corpus, and the report says so explicitly when it
happens. The full report is also written to
`<run-output-dir>/quality_screen_report.json`.

### Comparing prompt variants

Three environment variables override the config without editing it, so a
comparison run cannot silently become the default:

```bash
KFP_TASK=classify image_dir=... output=./out_v6 \
    screen_variant=quality_screen_v6 \
    screen_min_tiles=12 screen_max_tiles=12 \
    bash entrypoint.sh
```

Each run records which variant produced it, and `evaluate` scores against that
rather than against config — so a run screened with one prompt can never be
scored with another's criteria or answer polarity.
