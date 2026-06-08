# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Trust distribution compliance evaluation (CPU).

Reads raw_extractions.jsonl from the trust linking stage,
compares against trust_distribution_links.yml ground truth,
and computes:
  - Per-field extraction accuracy (exact match for IDs, numeric for amounts)
  - Compliance detection metrics (precision, recall, F1, confusion matrix)
  - Discrepancy classification accuracy
  - Throughput (cases/min, avg seconds/case)

No GPU needed -- runs on CPU nodes.

Usage:
    python -m stages.evaluate_trust \
        --input /artifacts/raw_extractions.jsonl \
        --ground-truth /data/trust_distribution_links.yml \
        --output-dir /artifacts/trust_evaluation
"""

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from .io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


# ---------------------------------------------------------------------------
# Configuration (fail-fast, 4-element diagnostics)
# ---------------------------------------------------------------------------


def _diagnostic(what: str, where: str, example: str, fix: str) -> str:
    """Assemble a 4-element diagnostic error message."""
    return f"What: {what}\nWhere: {where}\nExpected: {example}\nHow to fix: {fix}"


def _load_trust_eval_config(config_path: Path | None) -> dict[str, Any]:
    """Load and validate the ``pipeline.trust`` eval settings from run_config.yml.

    Reads the amount-match tolerance and the id/amount field split used by the
    per-field accuracy loop. Every key is REQUIRED; a missing or invalid value
    fails fast with a 4-element diagnostic (what / where / valid example /
    remediation) before any work begins.

    Args:
        config_path: Path to run_config.yml. Defaults to the repo config.

    Returns:
        ``{"amount_tolerance": float, "id_fields": set[str],
        "amount_fields": set[str]}``.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "run_config.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            _diagnostic(
                what="the configuration file is missing.",
                where=str(config_path),
                example="a YAML file with a 'pipeline.trust:' section.",
                fix=f"create {config_path} or pass --config with a valid path.",
            )
        )

    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    pipeline_section = raw.get("pipeline", {})
    if "trust" not in pipeline_section:
        raise ValueError(
            _diagnostic(
                what="the key 'pipeline.trust' is absent.",
                where=f"{config_path} -> pipeline.trust",
                example=(
                    "pipeline:\n"
                    "    trust:\n"
                    "        amount_tolerance: 0.01\n"
                    "        linking_fields:\n"
                    "            id_fields: [trust_abn, beneficiary_tfn]\n"
                    "            amount_fields: [share_of_net_income, franking_credit]"
                ),
                fix=f"add a 'pipeline.trust:' section to {config_path}.",
            )
        )

    trust = pipeline_section["trust"]

    if "amount_tolerance" not in trust:
        raise ValueError(
            _diagnostic(
                what="required key 'pipeline.trust.amount_tolerance' is missing.",
                where=f"{config_path} -> pipeline.trust.amount_tolerance",
                example="a relative tolerance float, e.g.: 0.01",
                fix=f"add 'amount_tolerance: 0.01' under 'pipeline.trust:' in {config_path}.",
            )
        )
    try:
        amount_tolerance = float(trust["amount_tolerance"])
    except (TypeError, ValueError):
        raise ValueError(
            _diagnostic(
                what=(
                    f"'pipeline.trust.amount_tolerance' is {trust['amount_tolerance']!r}, "
                    "which is not a number."
                ),
                where=f"{config_path} -> pipeline.trust.amount_tolerance",
                example="a relative tolerance float, e.g.: 0.01",
                fix=f"set 'pipeline.trust.amount_tolerance' to a float in {config_path}.",
            )
        ) from None

    if "linking_fields" not in trust or not isinstance(trust["linking_fields"], dict):
        raise ValueError(
            _diagnostic(
                what=(
                    "'pipeline.trust.linking_fields' is missing or is not a nested "
                    "mapping of 'id_fields'/'amount_fields'."
                ),
                where=f"{config_path} -> pipeline.trust.linking_fields",
                example=(
                    "linking_fields:\n"
                    "    id_fields: [trust_abn, beneficiary_tfn]\n"
                    "    amount_fields: [share_of_net_income, franking_credit]"
                ),
                fix="replace any flat 'linking_fields' list with the nested "
                "'id_fields'/'amount_fields' mapping in "
                f"{config_path}.",
            )
        )
    linking_fields = trust["linking_fields"]

    id_fields = _require_field_list(linking_fields, "id_fields", config_path)
    amount_fields = _require_field_list(linking_fields, "amount_fields", config_path)

    return {
        "amount_tolerance": amount_tolerance,
        "id_fields": id_fields,
        "amount_fields": amount_fields,
    }


def _require_field_list(linking_fields: dict[str, Any], key: str, config_path: Path) -> set[str]:
    """Return ``linking_fields[key]`` as a non-empty ``set[str]`` or fail fast."""
    value = linking_fields.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError(
            _diagnostic(
                what=(
                    f"'pipeline.trust.linking_fields.{key}' is missing, empty, or not a "
                    "non-empty list of strings."
                ),
                where=f"{config_path} -> pipeline.trust.linking_fields.{key}",
                example=f"{key}: [field_a, field_b]",
                fix=f"add a non-empty list under 'pipeline.trust.linking_fields.{key}' in {config_path}.",
            )
        )
    return set(value)


def _load_classification_gt(path: Path) -> dict[str, str]:
    """Load classification ground truth YAML (filename -> document_type)."""
    with path.open() as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        msg = f"Classification ground truth is empty or invalid: {path}"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in data.items()}


def _evaluate_classifications(
    classifications: list[dict[str, Any]],
    classification_gt: dict[str, str],
) -> dict[str, Any]:
    """Compare predicted document types against ground truth.

    Args:
        classifications: Records from trust_classifications.jsonl.
        classification_gt: Mapping of filename -> expected document_type.

    Returns:
        Dict with: total, correct, accuracy, per_type, confusion_matrix.
    """
    all_types = sorted(set(classification_gt.values()))
    confusion: dict[str, dict[str, int]] = {t: {t2: 0 for t2 in all_types} for t in all_types}

    if not classifications:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "per_type": {},
            "confusion_matrix": confusion,
        }

    df = pd.DataFrame(classifications)
    df["actual"] = df["image_name"].map(classification_gt)

    # Drop records not in ground truth (with warning)
    unmatched = df[df["actual"].isna()]
    for _, row in unmatched.iterrows():
        logger.warning("Classification for '%s' not in ground truth — skipped", row["image_name"])
    matched = df[df["actual"].notna()].copy()

    total = len(matched)
    matched["is_correct"] = matched["document_type"] == matched["actual"]
    correct = int(matched["is_correct"].sum())

    # Per-type metrics via groupby
    per_type_agg = matched.groupby("actual")["is_correct"].agg(correct="sum", total="count")
    per_type_agg["accuracy"] = (per_type_agg["correct"] / per_type_agg["total"]).round(3)
    per_type_result = {
        str(doc_type): {
            "correct": int(row["correct"]),
            "total": int(row["total"]),
            "accuracy": float(row["accuracy"]),
        }
        for doc_type, row in per_type_agg.iterrows()
    }

    # Confusion matrix via pd.crosstab, then merge into initialised dict
    if not matched.empty:
        ct = pd.crosstab(matched["actual"], matched["document_type"])
        for actual_type in ct.index:
            for predicted_type in ct.columns:
                if actual_type in confusion:
                    confusion[actual_type][predicted_type] = int(ct.loc[actual_type, predicted_type])

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total > 0 else 0.0,
        "per_type": per_type_result,
        "confusion_matrix": confusion,
    }


def _normalise_id(s: str) -> str:
    """Strip all whitespace from an ABN/TFN string."""
    return re.sub(r"\s+", "", s.strip()) if s else ""


def _parse_amount(s: str) -> float | None:
    """Parse a numeric string, stripping $, commas, spaces."""
    if not s or s.upper() in ("NOT_FOUND", "N/A", "NONE", "NULL"):
        return None
    cleaned = re.sub(r"[$,\\s]", "", s.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _amounts_match(extracted: str, expected: str, tolerance: float = 0.01) -> bool:
    """Check if two amount strings are numerically equal within tolerance."""
    e = _parse_amount(extracted)
    g = _parse_amount(expected)
    if e is None or g is None:
        return False
    denom = max(abs(e), abs(g))
    if denom == 0:
        return e == g
    return abs(e - g) / denom <= tolerance


def _ids_match(extracted: str, expected: str) -> bool:
    """Check if two ABN/TFN strings match after space normalisation."""
    return _normalise_id(extracted) == _normalise_id(expected)


def run(
    extractions_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    *,
    classifications_path: Path | None = None,
    classification_gt_path: Path | None = None,
    inference_seconds: float | None = None,
    config_path: Path | None = None,
) -> Path:
    """Evaluate trust distribution linking results.

    Args:
        extractions_path: Path to raw_extractions.jsonl from trust-link stage.
        ground_truth_path: Path to trust_distribution_links.yml.
        output_dir: Directory for evaluation output files.
        classifications_path: Optional path to trust_classifications.jsonl.
        classification_gt_path: Optional path to trust_classification_gt.yml.
        inference_seconds: GPU inference wall-clock seconds for throughput.
        config_path: Path to run_config.yml supplying ``pipeline.trust`` eval
            settings (amount_tolerance, id/amount field split). Defaults to the
            repo config.

    Returns:
        Path to the written trust_evaluation_results.jsonl.
    """
    trust_eval_cfg = _load_trust_eval_config(config_path)
    id_fields = trust_eval_cfg["id_fields"]
    amount_fields = trust_eval_cfg["amount_fields"]
    amount_tolerance = trust_eval_cfg["amount_tolerance"]
    linking_fields = id_fields | amount_fields
    records = read_jsonl(extractions_path)
    if not records:
        msg = f"No records found in {extractions_path}"
        raise FileNotFoundError(msg)

    with ground_truth_path.open() as f:
        ground_truth = yaml.safe_load(f)

    if not ground_truth:
        msg = f"No ground truth found in {ground_truth_path}"
        raise FileNotFoundError(msg)

    logger.info("Read %d extractions and %d ground truth entries", len(records), len(ground_truth))

    # Build case_id -> ground truth lookup
    gt_by_case: dict[str, dict[str, Any]] = {}
    for dist_file, entry in ground_truth.items():
        case_match = re.match(r"(CASE\d+)", dist_file)
        if case_match:
            gt_by_case[case_match.group(1)] = {
                "linking_fields": entry["linking_fields"],
                "compliance_status": entry["compliance_status"],
                "discrepancy_type": entry.get("discrepancy_type"),
            }

    # --- Evaluate each record ---
    eval_results: list[dict[str, Any]] = []

    for record in records:
        case_id = record.get("image_name", "")
        error = record.get("error")
        processing_time = record.get("processing_time", 0.0)

        if error or case_id not in gt_by_case:
            actual_status = None
            actual_discrepancy = None
            if case_id in gt_by_case:
                actual_status = gt_by_case[case_id]["compliance_status"]
                actual_discrepancy = gt_by_case[case_id].get("discrepancy_type")
            eval_results.append(
                {
                    "case_id": case_id,
                    "error": error or f"No ground truth for {case_id}",
                    "field_accuracy": {},
                    "compliance_correct": False,
                    "processing_time": processing_time,
                    "actual_status": actual_status,
                    "predicted_status": "compliant",
                    "predicted_discrepancy": None,
                    "actual_discrepancy": actual_discrepancy,
                }
            )
            continue

        gt = gt_by_case[case_id]
        gt_fields = gt["linking_fields"]
        gt_status = gt["compliance_status"]
        gt_disc_type = gt.get("discrepancy_type")

        extracted_data = record.get("extracted_data", {})

        # Map extracted field names to ground truth field names
        field_map = {
            "trust_abn": extracted_data.get("TRUST_ABN", ""),
            "beneficiary_tfn": extracted_data.get("BENEFICIARY_TFN", ""),
            "share_of_net_income": extracted_data.get("SHARE_OF_NET_INCOME", ""),
            "franking_credit": extracted_data.get("FRANKING_CREDIT", ""),
            "capital_gain_component": extracted_data.get("CAPITAL_GAIN_COMPONENT", ""),
        }

        # Per-field accuracy
        case_field_accuracy: dict[str, bool] = {}
        for field_name in linking_fields:
            extracted_val = field_map.get(field_name, "")
            expected_val = str(gt_fields.get(field_name, ""))

            if field_name in id_fields:
                correct = _ids_match(extracted_val, expected_val)
            else:
                correct = _amounts_match(extracted_val, expected_val, amount_tolerance)

            case_field_accuracy[field_name] = correct
            if not correct:
                logger.warning(
                    "%s field %s mismatch: extracted=%r expected=%r",
                    case_id,
                    field_name,
                    extracted_val,
                    expected_val,
                )

        # Compliance detection
        pred_status = extracted_data.get("COMPLIANCE_STATUS", "compliant")
        compliance_correct = (pred_status == "non_compliant") == (gt_status == "non_compliant")

        eval_results.append(
            {
                "case_id": case_id,
                "field_accuracy": case_field_accuracy,
                "compliance_correct": compliance_correct,
                "predicted_status": pred_status,
                "actual_status": gt_status,
                "predicted_discrepancy": extracted_data.get("DISCREPANCY_TYPE"),
                "actual_discrepancy": gt_disc_type,
                "processing_time": processing_time,
            }
        )

    # --- Compute aggregate metrics (pandas) ---
    results_df = pd.DataFrame(eval_results)
    total_processing = float(results_df["processing_time"].sum())

    # Field accuracy: expand field_accuracy dicts → DataFrame, compute column means
    field_dicts = results_df["field_accuracy"].tolist()
    field_rows = [d for d in field_dicts if d]
    if field_rows:
        field_df = pd.DataFrame(field_rows)
        per_field_accuracy = {
            field: float(field_df[field].mean()) if field in field_df.columns else 0.0
            for field in linking_fields
        }
    else:
        per_field_accuracy = {field: 0.0 for field in linking_fields}

    # Compliance confusion matrix via boolean masks
    has_gt = results_df["actual_status"].notna()
    gt_df = results_df[has_gt]
    gt_nc = gt_df["actual_status"] == "non_compliant"
    pred_nc = gt_df["predicted_status"] == "non_compliant"
    tp = int((pred_nc & gt_nc).sum())
    fp = int((pred_nc & ~gt_nc).sum())
    fn = int((~pred_nc & gt_nc).sum())
    tn = int((~pred_nc & ~gt_nc).sum())

    # Discrepancy classification
    disc_mask = pred_nc & gt_nc
    disc_df = gt_df[disc_mask]
    discrepancy_total = len(disc_df)
    discrepancy_correct = (
        int((disc_df["predicted_discrepancy"] == disc_df["actual_discrepancy"]).sum())
        if discrepancy_total > 0
        else 0
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    disc_accuracy = discrepancy_correct / discrepancy_total if discrepancy_total > 0 else 0.0

    inference_time = (
        inference_seconds if inference_seconds is not None and inference_seconds > 0 else total_processing
    )
    num_cases = len(records)
    throughput = (num_cases / inference_time * 60.0) if inference_time > 0 else 0.0
    avg_seconds = inference_time / num_cases if num_cases > 0 else 0.0

    summary = {
        "total_cases": num_cases,
        "field_accuracy": per_field_accuracy,
        "compliance": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
        },
        "discrepancy_classification": {
            "correct": discrepancy_correct,
            "total": discrepancy_total,
            "accuracy": disc_accuracy,
        },
        "throughput": {
            "cases_per_min": throughput,
            "avg_seconds_per_case": avg_seconds,
            "inference_seconds": inference_time,
        },
    }

    # --- Optional classification evaluation ---
    if classifications_path is not None and classifications_path.is_file():
        classification_records = read_jsonl(classifications_path)
        if classification_gt_path is not None and classification_gt_path.is_file():
            classification_gt = _load_classification_gt(classification_gt_path)
        else:
            logger.warning(
                "No classification_ground_truth YAML provided — "
                "set pipeline.trust.classification_ground_truth in run_config.yml"
            )
            classification_gt = {}
        if classification_gt:
            cls_summary = _evaluate_classifications(classification_records, classification_gt)
            summary["classification"] = cls_summary
            logger.info(
                "Classification accuracy: %d/%d (%.1f%%)",
                cls_summary["correct"],
                cls_summary["total"],
                cls_summary["accuracy"] * 100,
            )

    # Write results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trust_evaluation_results.jsonl"
    all_output = eval_results + [{"_summary": summary}]
    write_jsonl(output_path, all_output)
    logger.info("Wrote %d evaluation results to %s", len(all_output), output_path)

    # Rich console summary
    _print_summary(summary, output_dir)

    return output_path


def _print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Render Rich tables with evaluation metrics."""
    console = Console()

    # Field extraction accuracy
    field_table = Table(title="Field Extraction Accuracy", show_header=True, header_style="bold")
    field_table.add_column("Field", style="cyan")
    field_table.add_column("Accuracy", style="green")
    for field, acc in sorted(summary["field_accuracy"].items()):
        style = "green" if acc >= 0.80 else "yellow" if acc >= 0.60 else "red"
        field_table.add_row(field, f"[{style}]{acc:.1%}[/{style}]")
    console.print()
    console.print(field_table)

    # Compliance detection
    comp = summary["compliance"]
    comp_table = Table(title="Compliance Detection", show_header=True, header_style="bold")
    comp_table.add_column("Metric", style="cyan")
    comp_table.add_column("Value", style="green")
    comp_table.add_row("True Positives", str(comp["true_positives"]))
    comp_table.add_row("False Positives", str(comp["false_positives"]))
    comp_table.add_row("True Negatives", str(comp["true_negatives"]))
    comp_table.add_row("False Negatives", str(comp["false_negatives"]))
    comp_table.add_row("Precision", f"{comp['precision']:.3f}")
    comp_table.add_row("Recall (Detection Rate)", f"{comp['recall']:.3f}")
    comp_table.add_row("F1 Score", f"{comp['f1']:.3f}")
    comp_table.add_row("False Positive Rate", f"{comp['false_positive_rate']:.3f}")
    console.print()
    console.print(comp_table)

    # Discrepancy classification
    disc = summary["discrepancy_classification"]
    disc_table = Table(title="Discrepancy Classification", show_header=True, header_style="bold")
    disc_table.add_column("Metric", style="cyan")
    disc_table.add_column("Value", style="green")
    disc_table.add_row("Correct", str(disc["correct"]))
    disc_table.add_row("Total Detected", str(disc["total"]))
    disc_table.add_row("Classification Accuracy", f"{disc['accuracy']:.1%}")
    console.print()
    console.print(disc_table)

    # Document classification (only when classifications were evaluated)
    if "classification" in summary:
        cls = summary["classification"]
        cls_table = Table(title="Document Classification", show_header=True, header_style="bold")
        cls_table.add_column("Type", style="cyan")
        cls_table.add_column("Accuracy", style="green")
        for doc_type, metrics in sorted(cls["per_type"].items()):
            acc = metrics["accuracy"]
            style = "green" if acc >= 0.80 else "yellow" if acc >= 0.60 else "red"
            cls_table.add_row(doc_type, f"[{style}]{acc:.1%}[/{style}]")
        overall_acc = cls["accuracy"]
        overall_style = "green" if overall_acc >= 0.80 else "yellow" if overall_acc >= 0.60 else "red"
        cls_table.add_section()
        cls_table.add_row("Overall", f"[{overall_style}]{overall_acc:.1%}[/{overall_style}]")
        console.print()
        console.print(cls_table)

    # Throughput
    tp = summary["throughput"]
    tp_table = Table(title="Throughput", show_header=True, header_style="bold")
    tp_table.add_column("Metric", style="cyan")
    tp_table.add_column("Value", style="green")
    tp_table.add_row("Cases/min", f"{tp['cases_per_min']:.2f}")
    tp_table.add_row("Avg seconds/case", f"{tp['avg_seconds_per_case']:.1f}")
    tp_table.add_row("Total inference time", f"{tp['inference_seconds']:.1f}s")
    tp_table.add_row("Output directory", str(output_dir))
    console.print()
    console.print(tp_table)


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to raw_extractions.jsonl from trust-link stage"
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g", help="Path to trust_distribution_links.yml"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for evaluation output"),
    classifications: Path | None = typer.Option(
        None,
        "--classifications",
        "-c",
        help="Path to trust_classifications.jsonl (from pipeline.trust.classifications in run_config.yml)",
    ),
    classification_gt: Path | None = typer.Option(
        None,
        "--classification-gt",
        help="Path to trust_classification_gt.yml (from pipeline.trust.classification_ground_truth in run_config.yml)",
    ),
    inference_seconds: float | None = typer.Option(
        None,
        "--inference-seconds",
        help="GPU inference wall-clock seconds for throughput computation",
    ),
    config: Path | None = typer.Option(
        None, "--config", help="YAML configuration file (for pipeline.trust eval settings)"
    ),
) -> None:
    """Evaluate trust distribution compliance detection (CPU only)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        input_path,
        ground_truth,
        output_dir,
        classifications_path=classifications,
        classification_gt_path=classification_gt,
        inference_seconds=inference_seconds,
        config_path=config,
    )


if __name__ == "__main__":
    app()
