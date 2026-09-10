"""Stage: score the image-quality screen against the corpus's defect labels.

The evaluate half of `classify -> evaluate`. There is no clean stage between
them: `clean` normalises free-text field values before comparison, and the
screen's answers are already canonical tokens.

CPU only. The GPU pass already happened and its raw responses are on disk,
which is why the two stages are separate -- moving a threshold or fixing the
reader re-scores 330 images in a second instead of re-running inference.
"""

import json
import logging
from pathlib import Path

import typer
import yaml

from common.quality_screen_parser import QualityResponse, load_screen_vocabulary
from common.quality_screen_scorer import QualityScore, score_quality_screen

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

# What an undefined rate renders as. Not "0.00": a criterion the model never
# predicted has no precision, and printing a number would read as a measured
# failure rather than an absence of measurement.
_UNDEFINED = "n/a"


def load_screen_records(path: Path) -> dict[str, QualityResponse]:
    """Read the classify stage's output back into scoreable responses.

    Args:
        path: The quality_screen.jsonl written by the classify stage.

    Returns:
        Image name -> its response.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        record["image_name"]: QualityResponse(
            answers=record["answers"],
            overall=record["overall"],
            malformed=record["malformed"],
            malformed_reason=record["malformed_reason"],
            think_drift=record["think_drift"],
            # .get, not [], because files written before v13 have no such key.
            # Absent reads as None -- "this run did not ask" -- which is what
            # the report then says, rather than inventing a SINGLE.
            composition=record.get("composition"),
        )
        for record in records
    }


_DEFAULT_CONFIG = Path("config/run_config.yml")


def load_screen_config(config_path: Path | None = None) -> dict:
    """Read the screen's config section without validating the whole pipeline.

    Deliberately does NOT go through `AppConfig.load`. That validates the model
    path exists, and this stage runs on a CPU-only pod which may have no model
    volume mounted -- it would crash on a check for a model it never touches.
    Scoring needs three values (the prompt file, the variant, and the condition
    mapping) and no model at all.

    Args:
        config_path: Path to run_config.yml, or None for the default.

    Returns:
        The validated `pipeline.quality_screen` block.

    Raises:
        ConfigError: The block is missing or malformed.
    """
    from common.app_config import AppConfig

    path = config_path or _DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text())
    return AppConfig._validate_quality_screen(raw, str(path))  # noqa: SLF001


def variant_of_run(path: Path) -> str | None:
    """Read which prompt variant produced a screen file.

    The records carry it because the two stages must agree on the prompt. A
    run screened with one variant and scored against another's vocabulary
    fails loudly when the criteria differ -- and silently when only the
    POLARITY differs, producing a full report of inverted numbers that look
    entirely plausible.

    Args:
        path: The quality_screen.jsonl written by the classify stage.

    Returns:
        The variant name, or None for a file written before it was recorded.
    """
    for line in path.read_text().splitlines():
        if line.strip():
            return json.loads(line).get("variant")
    return None


def screening_runs(path: Path) -> list[str]:
    """The distinct screening runs whose records are in this file.

    A resumed file is built by several runs, and the report otherwise cannot
    say so: `330 scored / missing 0` is what a SUCCESSFUL resume produces --
    30 carried plus 300 new -- and also what a full rescreen produces. Reading
    the total alone, the two are indistinguishable, which is exactly how a
    resume that worked gets reported as one that did not.

    Args:
        path: The quality_screen.jsonl written by the classify stage.

    Returns:
        Sorted timestamps, one per run that contributed records. Empty for a
        file written before runs were stamped.
    """
    stamps = set()
    for line in path.read_text().splitlines():
        if line.strip():
            stamp = json.loads(line).get("screened_at")
            if stamp:
                stamps.add(stamp)
    return sorted(stamps)


def _criterion_dict(criterion) -> dict:
    """One criterion's counts and rates, `None` where a rate is undefined."""
    return {
        "true_positives": criterion.true_positives,
        "false_positives": criterion.false_positives,
        "false_negatives": criterion.false_negatives,
        "precision": criterion.precision,
        "recall": criterion.recall,
        "f1": criterion.f1,
    }


def build_report(score: QualityScore, responses: dict[str, QualityResponse]) -> dict:
    """Assemble a serialisable report.

    Args:
        score: The scorer's result.
        responses: The loaded responses, for rates the scorer does not carry.

    Returns:
        A JSON-serialisable report. The confusion matrix is re-keyed from
        `(truth, prediction)` tuples to `"truth->prediction"` strings, since
        JSON has no tuple keys and a report that cannot be written is worse
        than one that is awkward to read.
    """
    return {
        "counts": {
            "total": score.total,
            "scored": score.scored,
            "malformed": score.malformed,
            "missing": score.missing,
            "think_drift": sum(1 for response in responses.values() if response.think_drift),
        },
        "per_criterion": {
            name: _criterion_dict(criterion) for name, criterion in score.per_criterion.items()
        },
        "by_document_type": {
            doc_type: {name: _criterion_dict(c) for name, c in criteria.items()}
            for doc_type, criteria in score.by_document_type.items()
        },
        "overall_confusion": {
            f"{truth}->{predicted}": count
            for (truth, predicted), count in sorted(score.overall_confusion.items())
        },
    }


def _rate(value: float | None) -> str:
    """Render a rate, distinguishing undefined from zero."""
    return _UNDEFINED if value is None else f"{value:.3f}"


def format_report(report: dict) -> str:
    """Render a report as readable text.

    Args:
        report: A report from `build_report`.

    Returns:
        The report as lines of text, counts first. The counts lead because a
        run with a high malformed rate has scores that describe a subset, and
        reading the F1 numbers before that fact is how a partial run gets
        mistaken for a whole one.
    """
    counts = report["counts"]
    lines = [
        f"Image-quality screen -- {report.get('variant', 'unknown variant')}",
        "=" * 60,
        f"images {counts['total']}   scored {counts['scored']}   "
        f"malformed {counts['malformed']}   missing {counts['missing']}   "
        f"reasoning drift {counts['think_drift']}",
    ]
    runs = report.get("screening_runs") or []
    if len(runs) > 1:
        # Said before the numbers, because "scored 330" reads as one run and a
        # resumed file is several. Without this the only way to tell a working
        # resume from a full rescreen is to go and read the classify log.
        lines.append(
            f"NOTE: these records were screened across {len(runs)} runs "
            f"({runs[0]} .. {runs[-1]}) -- the classify stage resumed rather than "
            f"rescreening. Same prompt and tile budget throughout; a change in "
            f"either would have forced a full rescreen."
        )

    if counts["malformed"] or counts["missing"]:
        unscored = counts["malformed"] + counts["missing"]
        share = 100.0 * unscored / counts["total"] if counts["total"] else 0.0
        lines.append(
            f"NOTE: {unscored} of {counts['total']} images ({share:.1f}%) were not scored. "
            f"The rates below describe the remainder, not the corpus."
        )

    lines += [
        "",
        f"{'CRITERION':<12}{'PREC':>8}{'RECALL':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}",
        "-" * 54,
    ]
    for name, criterion in report["per_criterion"].items():
        lines.append(
            f"{name.upper():<12}{_rate(criterion['precision']):>8}{_rate(criterion['recall']):>8}"
            f"{_rate(criterion['f1']):>8}{criterion['true_positives']:>6}"
            f"{criterion['false_positives']:>6}{criterion['false_negatives']:>6}"
        )

    for doc_type, criteria in report["by_document_type"].items():
        lines += ["", f"-- {doc_type} --"]
        for name, criterion in criteria.items():
            lines.append(
                f"{name.upper():<12}{_rate(criterion['precision']):>8}"
                f"{_rate(criterion['recall']):>8}{_rate(criterion['f1']):>8}"
            )

    lines += ["", "OVERALL severity (truth -> predicted)", "-" * 40]
    lines += [f"  {key:<24}{count:>6}" for key, count in report["overall_confusion"].items()]

    lines += _composition_lines(report)
    return "\n".join(lines)


def _composition_lines(report: dict) -> list[str]:
    """The composition section, printed separately from the severity table.

    Its own heading because it is its own axis. A reader skimming the severity
    matrix must not take a MULTIPLE for a severity, or a GOOD as evidence the
    image is processable.
    """
    tally = report.get("composition_tally") or {}
    scored = report.get("composition")

    if not tally and not scored:
        return []  # a variant that does not ask; say nothing rather than "0".

    lines = ["", "COMPOSITION — how many documents in the picture", "-" * 46]
    for value, count in tally.items():
        lines.append(f"  answered {value:<16}{count:>6}")

    if scored is None:
        lines += [
            "  NOT SCORED: the ground truth carries no composition label, so this",
            "  column is the model's answers with nothing to check them against.",
            "  On a corpus known to hold no collages, any MULTIPLE is a false positive.",
        ]
        return lines

    accuracy = scored["accuracy"]
    lines.append(
        f"  scored {scored['scored']} of {scored['labelled']} labelled   "
        f"correct {scored['correct']}   "
        f"accuracy {'n/a' if accuracy is None else f'{accuracy:.3f}'}"
    )
    lines += [f"  {key:<24}{count:>6}" for key, count in scored["confusion"].items()]
    return lines


def score_composition(responses: dict[str, QualityResponse], truths: list[dict]) -> dict | None:
    """Score the SINGLE/MULTIPLE answer, when there is ground truth for it.

    Kept apart from the criterion scorer on purpose. The six criteria and the
    severity level all answer "how bad is this photograph"; composition answers
    "how many documents are in it", which is a different question with a
    different remedy -- re-photograph versus split. Scoring them together would
    average two unrelated things into one number.

    Args:
        responses: Image name -> its response.
        truths: Ground-truth records. A record carrying no `composition` key
            contributes nothing, so a corpus that has never been labelled for
            collages yields None rather than a score of zero.

    Returns:
        `{"labelled", "correct", "accuracy", "confusion"}`, or None when no
        truth record carries a composition label. None means UNMEASURED, and
        the report says so -- an unmeasured criterion reported as 0.0 reads as
        a broken one, and reported as 1.0 reads as a working one.
    """
    labelled = [t for t in truths if t.get("composition")]
    if not labelled:
        return None

    confusion: dict[str, int] = {}
    correct = 0
    scored = 0
    for truth in labelled:
        response = responses.get(truth["image_name"])
        if response is None or response.malformed or response.composition is None:
            continue
        scored += 1
        key = f"{truth['composition']}->{response.composition}"
        confusion[key] = confusion.get(key, 0) + 1
        if truth["composition"] == response.composition:
            correct += 1

    return {
        "labelled": len(labelled),
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored) if scored else None,
        "confusion": dict(sorted(confusion.items())),
    }


def composition_tally(responses: dict[str, QualityResponse]) -> dict[str, int]:
    """Count the composition answers, for a run with no labels to score against.

    Not a score. It is the only thing that can honestly be said about an
    unlabelled run: how many images the model called MULTIPLE. Worth printing
    anyway -- on a corpus known to hold no collages, any MULTIPLE at all is a
    false positive, and that is a measurement even without a label file.
    """
    tally: dict[str, int] = {}
    for response in responses.values():
        if response.composition:
            tally[response.composition] = tally.get(response.composition, 0) + 1
    return dict(sorted(tally.items()))


def load_truths(path: Path) -> list[dict]:
    """Read the corpus's quality ground truth.

    Args:
        path: The corpus's quality_ground_truth.jsonl.

    Returns:
        One record per image.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(
    screen_path: Path,
    ground_truth: Path,
    output_dir: Path,
    *,
    prompt_file: Path,
    variant: str,
    condition_to_level: dict[str, str],
) -> Path:
    """Score a screen run and write the report.

    Args:
        screen_path: quality_screen.jsonl from the classify stage.
        ground_truth: The corpus's quality_ground_truth.jsonl.
        output_dir: Directory to write the report into.
        prompt_file: The prompt config, for the criterion vocabulary.
        variant: Which prompt variant the run used.
        condition_to_level: Corpus condition -> prompt severity level.

    Returns:
        Path to the written JSON report.
    """
    # The run says which prompt made it; config only supplies a fallback for
    # files written before that was recorded. Trusting config over the records
    # is how a run gets scored against another variant's criteria and polarity.
    recorded = variant_of_run(screen_path)
    if recorded and recorded != variant:
        logger.info("Scoring against the variant the run recorded: %s (config says %s)", recorded, variant)
    elif not recorded:
        logger.warning(
            "%s records no variant, so it predates variant stamping. Scoring against the "
            "configured variant %r -- if the run used a different prompt, the criteria and "
            "polarity below are wrong.",
            screen_path,
            variant,
        )
    resolved_variant = recorded or variant

    vocabulary = load_screen_vocabulary(prompt_file, variant=resolved_variant)
    responses = load_screen_records(screen_path)
    truths = load_truths(ground_truth)

    # A variant that renames its severity levels brings its own mapping; config
    # supplies the default. Scoring GOOD/FAIR/POOR answers against a
    # NONE/MODERATE/HEAVY mapping would fail every severity call.
    if vocabulary.condition_to_level:
        condition_to_level = vocabulary.condition_to_level
        logger.info("Using the variant's own condition mapping: %s", condition_to_level)

    score = score_quality_screen(
        responses,
        truths,
        criteria=vocabulary.criteria,
        condition_to_level=condition_to_level,
        # From the variant's own `evidence:` block, so a prompt asking "is it
        # perfectly sharp?" has its NO scored as a detection. Reading it from
        # the prompt that produced the answers is the only way this stays right
        # when variants differ.
        polarity=vocabulary.polarity,
    )
    report = build_report(score, responses)
    report["variant"] = resolved_variant
    report["screening_runs"] = screening_runs(screen_path)
    report["composition"] = score_composition(responses, truths)
    report["composition_tally"] = composition_tally(responses)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "quality_screen_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    # Printed, not just written: the operator reading the pod log should not
    # have to go and open a file to learn whether the run is usable.
    print(format_report(report))  # noqa: T201

    return report_path


@app.command()
def main(
    screen: Path = typer.Option(..., "--input", "-i", help="quality_screen.jsonl from classify"),
    ground_truth: Path = typer.Option(..., "--ground-truth", "-g", help="quality_ground_truth.jsonl"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for the report"),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration file"),
) -> None:
    """Stage 2: score the image-quality screen against the corpus labels."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    screen_cfg = load_screen_config(config)

    run(
        screen,
        ground_truth,
        output_dir,
        prompt_file=Path(screen_cfg["prompt_file"]),
        variant=screen_cfg["variant"],
        condition_to_level=screen_cfg["condition_to_level"],
    )


if __name__ == "__main__":
    app()
