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
| Poor-quality images correctly flagged | **215 of 220 (98%)** |
| Poor-quality images missed | 5 of 220 (2%) |
| Good images flagged unnecessarily | 15 of 110 (14%) |

The errors fall on the safer side: it rarely lets a bad image through, and its
main cost is asking for a second look at an image that was fine.

**Grading *how* bad an image is works less well** — 72% correct across three
levels (good / fair / poor). It reliably separates *damaged from undamaged*; it
is less reliable at telling mild damage from severe.

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

## Reproducing the test corpus on PROD

The 330 test images are generated, not collected, so any environment can
rebuild them from source. Nothing here is environment-specific: substitute your
own working directory and output path.

**1. Clone the corpus generator at the matching branch.**

```bash
git clone --branch feature/quality-screen-corpus \
    https://github.com/tmnestor/Synthetic_Doc_Generation.git
cd Synthetic_Doc_Generation
```

The branch matters. The corpus on `main` is a different set — three document
types, three severity tiers, receipts only degraded, and no quality labels.

**2. Build the environment.**

```bash
conda env create -f environment.yml
conda activate synthetic
```

**3. Verify the image libraries before generating.**

```bash
python -c 'import augraphy, cv2, numpy; print(augraphy.__version__, cv2.__version__, numpy.__version__)'
```

Expect augraphy 8.2.6 and OpenCV from the **headless** build. Augraphy declares
the full GUI OpenCV as a dependency, which silently displaces the pinned
headless one and changes rendering. If the wrong build is installed:

```bash
pip uninstall -y opencv-python && pip install --no-deps augraphy==8.2.6
```

**4. Generate.**

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

**5. Run the screen against it.**

```bash
KFP_TASK=classify image_dir=<...>/quality_<date> output=<run-output-dir> bash entrypoint.sh
KFP_TASK=evaluate ground_truth=<...>/quality_<date>/quality_ground_truth.jsonl \
    output=<run-output-dir> bash entrypoint.sh
```

Prompt variant and tile budget come from `config/run_config.yml`; both defaults
are the measured configuration reported above. `screen_variant`,
`screen_min_tiles` and `screen_max_tiles` override them for comparison runs
without editing config.

Generation is deterministic given the ground-truth seeds, so a rebuild produces
the same images. Labels may differ by one or two on criteria whose drawn value
sits within floating-point distance of a threshold, which varies by CPU
architecture; the labels always describe the images actually produced.
