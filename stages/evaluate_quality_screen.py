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
    return "\n".join(lines)


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
