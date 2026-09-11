# Image-quality screen — summary

Measured 2026-09-09 on a 330-image corpus. Branch
`feature/quality-screen-standalone`, paired with `feature/quality-screen-corpus`
in Synthetic_Doc_Generation.

Re-measured 2026-09-11 on 450 images — the same 330 plus 90 receipt collages
and 30 folded receipts, added to measure the collage check. Where a figure
differs between the two corpora both are given, because they are different
populations rather than a before and after: the 450-image set is balanced
150/150/150 across the three severity levels, the 330-image set 110/110/110.
Part A builds the 450-image corpus.

**In one line:** on the decision the pipeline makes — send this image on to
extraction, or send it back — the screen scores precision 0.997 and recall
0.961, sending back 317 of the 330 unusable images while wrongly returning 1 of
120 usable ones.

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

**On the larger 450-image corpus, judged on severity alone, precision appears
to fall** — 0.903 against 0.995, with 31 of 150 good images flagged where the
330-image corpus flagged 1 of 110. Recall held (0.957 against 0.950).

**That reading is wrong, and the reason matters more than the number.** It
scores the screen against "is the photograph damaged", which is not the
decision the pipeline makes. Downstream extraction does not read a photograph
of several receipts reliably *whether or not the photograph is any good*, so a
collage must be sent back regardless of its severity. A clean collage graded
POOR is therefore a correct decision, not a false alarm — and the 330-image
corpus contained no collages at all, which is why the question never arose.

The corpus is built so that most of those 31 flags fall on the 40 *clean*
collage and folded images, and every one of the 90 collages was detected on the
composition axis. So the screen is catching them; only the yardstick was wrong.

`evaluate` now scores the gate directly:

```yaml
pipeline:
  quality_screen:
    routing:
      pass_levels: [GOOD]
      multiple_documents: reject
```

An image passes only if its severity is in `pass_levels` **and** it holds a
single document. Scored that way, on the same 450 images:

| | result |
|---|---|
| Unusable images correctly sent back | **317 of 330 (96%)** |
| Usable images wrongly sent back | **1 of 120 (0.8%)** |
| Unusable images passed through | 13 of 330 (4%) |
| Precision / recall / F1 | **0.997 / 0.961 / 0.978** |

The false-alarm rate is 1 in 120, against 1 in 110 on the 330-image corpus —
unchanged. All 13 misses are *degraded single documents* graded GOOD; **no
collage was missed**.

The split confirms the mechanism. Severity is exact on 86.4% of single-document
photographs — statistically unchanged from the 85.5% measured on the smaller
corpus — and on only 34.4% of collages, because **all 30 clean collages were
graded degraded**. Not one was passed as GOOD. Against severity those are 30
errors; against the routing gate they are 30 correct rejections.

**Reproduced independently on production hardware.** An earlier version of the
prompt was run on the production GPUs against a separately generated test set
and gave the same result within noise, on different hardware and different
images. The two checks that do not work (below) failed again in the same way,
confirming those are real limitations rather than quirks of one test set. 330
images take about 11 minutes on four production GPUs.

**Grading *how* bad an image is works reasonably well** — 86% correct across
three levels (good / fair / poor) on the 330-image corpus, 76% on the 450-image
one. It separates damaged from undamaged reliably, and is now also fairly good
at telling mild damage from severe. The lower figure is the same artefact as
above: it counts a clean collage graded POOR as an error, when for routing
purposes that verdict sends the image exactly where it should go.

**It also reports whether the photograph contains more than one receipt.**
Taxpayers commonly place several receipts on a table and photograph them
together, and downstream extraction does not read those reliably — clean or
damaged. So **every** collage is sent back, however good the photograph is.

It is reported on its own axis rather than folded into the quality verdict,
because a photograph of four receipts is often perfectly sharp and well lit —
nothing is wrong with the *picture* — and the remedy differs: split it, rather
than ask the taxpayer to photograph it again. Two different messages to send,
so two separate answers, combined into one decision by the routing gate.

**That check is now measured, and it works.** Scored 2026-09-11 on the
450-image corpus, which adds 90 photographs of several receipts laid out on one
plate, and 30 photographs of a single long receipt folded — the hard negative,
because a folded receipt looks like two receipts and must still answer
"one document".

| | result |
|---|---|
| Collages correctly identified | **90 of 90 (100%)** |
| Single documents wrongly called collages | **2 of 360 (0.6%)** |
| Accuracy | 0.996 |

This is the strongest of the screen's checks, and the only one measured at
ceiling. Both mistakes were folded receipts at the heaviest degradation tier —
the hard negative built because a folded receipt looks like two receipts. **No
ordinary document was ever called a collage.** The only two errors fell on the
case designed to be hard, which is where errors should fall.

## What it cannot do yet

**Two of its six checks do not work**, and the 450-image run confirms both as
persistent rather than accidental. It almost never detects creasing (11 of 150
creased images, recall 0.07), and it barely detects shadow on receipts (recall
0.02) though it detects shadow on invoices perfectly (recall 1.00). Both are
*precise* when they do fire — 1.000 for each, so they never cry wolf; they
simply stay silent. In both cases we believe the test images themselves are at
fault rather than the model, and we can confirm that cheaply.

The other four checks are sound but not symmetric: blur and tilt never miss a
defect (recall 1.000) at the cost of firing on roughly four in ten undamaged
images. That is deliberate — the per-criterion answers prime the overall
verdict rather than informing it, and making the criteria individually accurate
has twice made the screen as a whole worse. The per-criterion table is not the
deliverable.

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

1. **Key the taxpayer's message on composition, not severity.** The screen
   reaches the right decision on a collage by the wrong stated reason: it calls
   the *photograph* poor when the photograph is fine. A message driven from the
   severity verdict would tell someone their sharp, well-lit picture was
   blurry, and ask them to take it again — when what they need to do is
   photograph the receipts separately. The two remedies are different, so the
   message must read the composition answer. Cheap, and it is the only finding
   here that is visible to a user.
2. **Confirm the two broken checks are test-data problems.** Cheap, and decides
   whether to fix them or drop them to four checks.
3. **Test against real photographs.** Until then we know the screen works on
   images we generated, which is not the same as images users take.
4. **Decide what the screen is for.** If it is "send this back for
   re-capture", the pass/fail decision is ready. If it needs to route by
   severity, the three-level grading needs more work.

## Caveats worth stating

- All results are from one model (InternVL3.5-8B). Not tested on alternatives.
- 450 images from 55 source documents, so far fewer independent documents than
  images. The collages are built from those same receipts, so a model that
  learned one receipt's quirks sees them again on a plate.
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
python -m generators.pipeline eval-set --out <writable-output-parent> --force
```

Not `generate` — that renders clean pages and per-field geometry for the
extraction exports. `eval-set` renders the evaluation corpus straight from the
committed ground truth, and is the only command needed here. `--force` replaces
an existing dated directory; without it a second run refuses to overwrite.

Choose an output path the process can actually write to — on an orchestrated
run that generally means the job's own output directory rather than a home or
cache path.

Takes roughly 15 minutes and produces three dated directories totalling ~1.8 GB:

| directory | contents | size | purpose |
|---|---|---|---|
| `synthetic_<date>/` | 110 clean images | 294 MB | clean-only comparison runs |
| `degraded_<date>/` | 220 degraded images | 497 MB | degraded-only comparison runs |
| `quality_<date>/` | all 450 | 962 MB | **the quality screen reads this one** |

`quality_<date>/` holds three families of filename:

| prefix | count | composition | what it is |
|---|---|---|---|
| `CASE*` | 330 | SINGLE | one document per photograph, clean and degraded |
| `COLLAGE*` | 90 | MULTIPLE | several receipts on one plate, photographed together |
| `FOLDED*` | 30 | SINGLE | one long receipt folded — the collage check's hard negative |

Each directory carries `ground_truth.jsonl` (what the document says),
`ground_truth.csv` (the same, flat) and `quality_ground_truth.jsonl` (which
defects each image actually has, the values drawn to produce them, and its
composition).

**A5. The directory name follows the generating machine's clock.** The stamp is
today's date where the command runs, so a UTC host and an AEST laptop can
disagree by a day. Check rather than assume, and do not rename the directory to
match another machine — point the screen at the name you got.

```bash
ls -d <writable-output-parent>/quality_*
```

**A6. Confirm two machines built the same corpus.** Renders are byte-identical
across machines only because `pillow`, `numpy`, `opencv-python-headless` and
`augraphy` are pinned exactly — pillow's bundled FreeType drives font metrics,
which drive every fit decision and therefore the pixels. Equal digests mean the
pins held:

```bash
cd <writable-output-parent>/quality_<date>
ls *.png | sort | xargs sha256sum | sha256sum      # shasum -a 256 on macOS
sha256sum quality_ground_truth.jsonl
```

Reference, from the build of 2026-09-11 (450 images):

```
PNG manifest                24297c1e3a16a2b5719a0fe75a8395542c97cc25b2e1f2790ec00d3edb78208c
quality_ground_truth.jsonl  ee026133c63a0fe68c94a853d657384b75fbd94d6ef7b68b7c018cd7ba1224aa
```

A mismatch is an environment problem before it is anything else. Check the
OpenCV build (A3) first.

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

**B4. Set the paths once.** Everything below overrides `run_config.yml` from
the environment, so a diagnostic run cannot silently become the default.

```bash
QDIR=<corpus>/quality_<date>
```

Four overrides matter, and the fourth is the one people forget:

| variable | why |
|---|---|
| `image_dir` | the corpus to screen |
| `ground_truth` | its labels, beside the images |
| `output` | where results are written |
| `LMM_LOG_DIR` | **easy to forget** — see below |

`log_dir` comes from `run_config.yml` while `output` comes from the
environment. Override one without the other and the YAML stays perfectly
self-consistent while the two point at different corpora. `entrypoint.sh`
compares the *resolved* paths and exits before the model loads, rather than
failing on first write half an hour in. Setting `LMM_LOG_DIR` under `output` is
the fix.

Note also that `run_config.yml` is committed, so it names whatever corpus it
last named. A run without these overrides will screen that older set quite
happily and report nothing amiss — `KFP_TASK=check` (B3) is how you find out
before spending GPU time.

**B5. Smoke test first — 30 images.** Prove the wiring before committing to a
full run:

```bash
KFP_TASK=screen \
    image_dir="$QDIR" \
    ground_truth="$QDIR/quality_ground_truth.jsonl" \
    output="$QDIR/output" \
    LMM_LOG_DIR="$QDIR/output/logs" \
    screen_max_images=30 \
    bash entrypoint.sh
```

A good smoke shows `malformed 0`, `reasoning drift 0`, and a severity diagonal
in the same region as a full run. Two results are expected rather than wrong:

- **`missing 420`.** Evaluate always scores against the whole ground truth and
  reports the rest as missing, so a short run cannot be mistaken for a full one.
- **Composition accuracy of 1.000, on SINGLE only.** `screen_max_images` takes
  the first N images *by filename*, and `CASE*` sorts before `COLLAGE*` — the
  first MULTIPLE image is at index 330. A 30-image smoke therefore contains no
  collage at all. It shows the screen does not false-positive on ordinary
  documents, and says nothing about detection.

**B6. Run it.** Two stages, no clean stage between them — the screen's answers
are fixed tokens with nothing to normalise.

```bash
# GPU. Writes quality_screen.jsonl: one record per image with its answers,
# the raw model response, and the settings that produced it.
KFP_TASK=classify \
    image_dir="$QDIR" \
    output="$QDIR/output" \
    LMM_LOG_DIR="$QDIR/output/logs" \
    bash entrypoint.sh

# CPU. Scores it and prints the report.
KFP_TASK=evaluate \
    ground_truth="$QDIR/quality_ground_truth.jsonl" \
    output="$QDIR/output" \
    LMM_LOG_DIR="$QDIR/output/logs" \
    bash entrypoint.sh
```

Always through `entrypoint.sh` — it sets up the environment the stages expect,
and invoking the modules directly does not.

On one box rather than a pipeline, `KFP_TASK=screen` runs both in a single
shell. It must never be set in the KFP manifest, where it would run the
CPU-only scoring inside the GPU pod, holding every card idle throughout.

330 images took roughly 20 minutes on two GPUs, which extrapolates to about 28
minutes for 450. It halves with each additional GPU: the classify stage shards
across every GPU it is given, and 330 images took about 11 minutes on four
production GPUs.

**Re-runs.** By default every run rescreens the whole directory. Where images
arrive over time and only the new ones need screening, set
`CLEAR_PREV_OUTPUT=false` — the stage then screens only images with no record
yet. This is also how to follow a smoke test with a full run without paying
twice for the first 30 images. It resumes only when the prompt variant and tile
budget are unchanged; if either has moved it discards the previous records and
rescreens, saying so, because a file mixing two prompts is not one run and the
report cannot tell.

**B7. Read the result.** `evaluate` prints the per-criterion table, a
per-document-type split, the severity confusion matrix, the composition tally,
and — first, before the scores — the counts:

```
images 450   scored 450   malformed 0   missing 0   reasoning drift 0
```

Check that line first. Any image not scored means the rates below it describe a
subset rather than the corpus, and the report says so explicitly when it
happens. The full report is also written to
`<run-output-dir>/quality_screen_report.json`.

**Read the ROUTING block first.** It is the only section that scores the
decision the pipeline makes — send this image on to extraction, or send it
back — and it states the gate it used, so the number can be interpreted months
later. The criterion table below it is a diagnostic, not an outcome: the six
questions prime the model's overall verdict rather than being deliverables in
their own right.

For the composition block, read the `MULTIPLE->` row: that is collage
detection. The `SINGLE->` row is the false-positive rate on ordinary documents,
and the 30 `FOLDED*` images are the hard negative inside it — a folded receipt
looks like two receipts and must still answer SINGLE. Any mistakes are named
individually, so you can see which they were.

A clean collage graded POOR in the severity matrix is **not** a false alarm.
It is an unprocessable image sent back for the wrong stated reason, and the
ROUTING block scores it as the correct decision it is.

A criterion showing `n/a` precision with `0.000` recall was never predicted
present on any image. That is a real result, not a missing measurement.

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
