"""Evaluation stage: score extraction results against ground truth.

CPU-only — no model needed.

Reads extraction output (JSON or in-memory ExtractionOutput) and
ground truth CSV. Produces:
    - CSV summary (one row per image with aggregate metrics)
    - JSON field-level detail (nested per-field F1/precision/recall)
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

from .io_schemas import (
    EvaluationOutput,
    ExtractionOutput,
    FieldEvaluation,
    ImageEvaluation,
)

logger = logging.getLogger(__name__)


def evaluate_extractions(
    extractions: ExtractionOutput,
    ground_truth_csv: Path,
    field_definitions: dict[str, list[str]],
    evaluation_method: str | None = None,
    enable_math_enhancement: bool = True,
    verbose: bool = False,
) -> EvaluationOutput:
    """Evaluate extraction results against ground truth.

    CPU-only. No model needed.

    Args:
        extractions: Output from extract_documents() or read_extraction_json().
        ground_truth_csv: Path to ground truth CSV file.
        field_definitions: Document type → field list mapping from
            field_definitions.yaml.
        evaluation_method: Scoring method. None reads from EVALUATION_METHOD
            env var, defaulting to "order_aware_f1".
        enable_math_enhancement: Apply bank statement math correction.
        verbose: Enable debug output.

    Returns:
        EvaluationOutput with per-image evaluations and aggregate summary.
    """
    from common.evaluation_metrics import (
        calculate_correlation_aware_f1,
        calculate_field_accuracy_with_method,
        load_ground_truth,
    )
    from common.simple_model_evaluator import SimpleModelEvaluator

    if evaluation_method is None:
        evaluation_method = os.environ.get("EVALUATION_METHOD", "order_aware_f1")

    ground_truth_csv = Path(ground_truth_csv)
    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv}")

    gt_data = load_ground_truth(str(ground_truth_csv), verbose=verbose)
    logger.info("Loaded ground truth for %d images", len(gt_data))

    model_evaluator = SimpleModelEvaluator()
    image_evaluations: list[ImageEvaluation] = []

    for record in extractions.records:
        if record.error:
            image_evaluations.append(
                ImageEvaluation(
                    image_name=record.image_name,
                    image_path=record.image_path,
                    document_type=record.document_type,
                    overall_f1=0.0,
                    median_f1=0.0,
                    precision=0.0,
                    recall=0.0,
                    total_fields=0,
                    correct_fields=0,
                    fields_extracted=0,
                )
            )
            continue

        ground_truth = _lookup_ground_truth(gt_data, record.image_path)
        if not ground_truth:
            logger.debug("No ground truth for %s", record.image_name)
            image_evaluations.append(
                ImageEvaluation(
                    image_name=record.image_name,
                    image_path=record.image_path,
                    document_type=record.document_type,
                    overall_f1=0.0,
                    median_f1=0.0,
                    precision=0.0,
                    recall=0.0,
                    total_fields=0,
                    correct_fields=0,
                    fields_extracted=0,
                )
            )
            continue

        extracted_data = dict(record.extracted_data)

        # Bank statement math enhancement
        if (
            record.document_type.upper() == "BANK_STATEMENT"
            and enable_math_enhancement
            and not record.skip_math_enhancement
        ):
            extracted_data = _apply_bank_math_enhancement(extracted_data, verbose)

        # Filter ground truth to document-specific fields
        doc_type_lower = record.document_type.lower()
        eval_fields = field_definitions.get(
            doc_type_lower, field_definitions.get("invoice", [])
        )
        filtered_gt = {f: ground_truth[f] for f in eval_fields if f in ground_truth}

        # Run SimpleModelEvaluator for missing/incorrect field counts
        eval_result = model_evaluator.evaluate_extraction(
            extracted_data,
            filtered_gt,
            record.image_name,
        )

        # Build field-level F1 scores
        field_evals, overall_f1, median_f1, precision, recall, correct_fields = (
            _compute_field_scores(
                extracted_data,
                filtered_gt,
                record.document_type,
                evaluation_method,
                verbose,
                calculate_field_accuracy_with_method,
                calculate_correlation_aware_f1,
            )
        )

        fields_extracted = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

        image_evaluations.append(
            ImageEvaluation(
                image_name=record.image_name,
                image_path=record.image_path,
                document_type=record.document_type,
                overall_f1=overall_f1,
                median_f1=median_f1,
                precision=precision,
                recall=recall,
                total_fields=len(filtered_gt),
                correct_fields=correct_fields,
                fields_extracted=fields_extracted,
                field_evaluations=field_evals,
            )
        )

        logger.debug(
            "%s: Median F1 %.1f%% | Mean F1 %.1f%%",
            record.image_name,
            median_f1 * 100,
            overall_f1 * 100,
        )

    summary = _build_summary(image_evaluations)
    return EvaluationOutput(image_evaluations=image_evaluations, summary=summary)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lookup_ground_truth(gt_data: dict, image_path: str) -> dict:
    """Look up ground truth for an image with fuzzy matching."""
    image_name = Path(image_path).name
    image_stem = Path(image_path).stem

    # Exact match
    if image_name in gt_data:
        return gt_data[image_name]

    # Stem match
    for gt_key in gt_data:
        if Path(gt_key).stem == image_stem or gt_key == image_stem:
            return gt_data[gt_key]

    return {}


def _apply_bank_math_enhancement(extracted_data: dict, verbose: bool) -> dict:
    """Apply mathematical enhancement and debit filtering for bank statements."""
    from common.bank_statement_calculator import enhance_bank_statement_extraction

    logger.debug("Applying mathematical enhancement for bank statement")
    enhanced = enhance_bank_statement_extraction(extracted_data, verbose=verbose)

    result = {k: v for k, v in enhanced.items() if k != "_mathematical_analysis"}

    # Log math analysis results
    math_analysis = enhanced.get("_mathematical_analysis", {})
    if math_analysis.get("calculation_success"):
        logger.debug(
            "Math analysis: %d transactions",
            math_analysis.get("transaction_count", 0),
        )

    # Filter to debit-only transactions
    result = _filter_debit_transactions(result, verbose)
    return result


def _filter_debit_transactions(extracted_data: dict, verbose: bool = False) -> dict:
    """Filter bank statement data to debit-only transactions.

    Extracted from BatchDocumentProcessor._filter_debit_transactions().
    """
    if extracted_data.get("DOCUMENT_TYPE") != "BANK_STATEMENT":
        return extracted_data

    try:
        import pandas as pd

        descriptions = extracted_data.get("LINE_ITEM_DESCRIPTIONS", "")
        dates = extracted_data.get("TRANSACTION_DATES", "")
        paid = extracted_data.get("TRANSACTION_AMOUNTS_PAID", "")
        received = extracted_data.get("TRANSACTION_AMOUNTS_RECEIVED", "")
        balances = extracted_data.get("ACCOUNT_BALANCE", "")

        if any(
            field == "" or field == "NOT_FOUND" for field in [descriptions, dates, paid]
        ):
            logger.warning("Missing transaction data - skipping debit filtering")
            return extracted_data

        balance_values = [b.strip() for b in balances.split(" | ")] if balances else []
        all_balances_missing = all(b in ("NOT_FOUND", "") for b in balance_values)

        if balances in ("", "NOT_FOUND") or all_balances_missing:
            logger.warning("No balance data available - skipping debit filtering")
            return extracted_data

        desc_list = descriptions.split(" | ")
        date_list = dates.split(" | ")
        paid_list = paid.split(" | ")
        balance_list = balances.split(" | ")
        received_list = (
            received.split(" | ") if received and received != "NOT_FOUND" else None
        )

        lengths = [len(desc_list), len(date_list), len(paid_list), len(balance_list)]
        if len(set(lengths)) > 1:
            logger.warning(
                "Array length mismatch: %s - skipping debit filtering", lengths
            )
            return extracted_data

        transactions_df = pd.DataFrame(
            {
                "description": desc_list,
                "date": date_list,
                "paid": paid_list,
                "received": received_list,
                "balance": balance_list,
            }
        )

        debit_df = transactions_df[transactions_df["paid"] != "NOT_FOUND"].copy()

        logger.debug(
            "Debit transactions: %d/%d",
            len(debit_df),
            len(transactions_df),
        )

        filtered = extracted_data.copy()
        filtered["LINE_ITEM_DESCRIPTIONS"] = " | ".join(
            debit_df["description"].tolist()
        )
        filtered["TRANSACTION_DATES"] = " | ".join(debit_df["date"].tolist())
        filtered["TRANSACTION_AMOUNTS_PAID"] = " | ".join(debit_df["paid"].tolist())
        filtered["TRANSACTION_AMOUNTS_RECEIVED"] = "NOT_FOUND"
        filtered["ACCOUNT_BALANCE"] = " | ".join(debit_df["balance"].tolist())

        return filtered

    except Exception as e:
        logger.error("Debit filtering failed: %s", e)
        return extracted_data


def _compute_field_scores(
    extracted_data: dict,
    filtered_gt: dict,
    document_type: str,
    evaluation_method: str,
    verbose: bool,
    calc_field_accuracy_fn,
    calc_correlation_fn,
) -> tuple[list[FieldEvaluation], float, float, float, float, int]:
    """Compute per-field F1 scores and aggregate metrics.

    Returns:
        (field_evals, overall_f1, median_f1, precision, recall, correct_fields)
    """
    field_evals: list[FieldEvaluation] = []
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    if evaluation_method in ("correlation", "correlation_aware_f1"):
        corr = calc_correlation_fn(
            extracted_data,
            filtered_gt,
            document_type,
            debug=False,
        )
        for field_name in filtered_gt:
            field_evals.append(
                FieldEvaluation(
                    field_name=field_name,
                    f1_score=corr["f1_score"],
                    precision=corr["precision"],
                    recall=corr["recall"],
                    extracted_value=str(extracted_data.get(field_name, "NOT_FOUND")),
                    ground_truth_value=str(filtered_gt.get(field_name, "")),
                )
            )
            total_f1 += corr["f1_score"]
            total_precision += corr["precision"]
            total_recall += corr["recall"]
    else:
        for field_name in filtered_gt:
            extracted_val = extracted_data.get(field_name, "NOT_FOUND")
            ground_val = filtered_gt.get(field_name, "NOT_FOUND")

            is_debug = field_name == "IS_GST_INCLUDED" and verbose
            metrics = calc_field_accuracy_fn(
                extracted_val,
                ground_val,
                field_name,
                method=evaluation_method,
                debug=is_debug,
            )

            field_evals.append(
                FieldEvaluation(
                    field_name=field_name,
                    f1_score=metrics["f1_score"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    extracted_value=str(extracted_val),
                    ground_truth_value=str(ground_val),
                )
            )
            total_f1 += metrics["f1_score"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]

    num_fields = len(field_evals)
    overall_f1 = total_f1 / num_fields if num_fields else 0.0
    precision = total_precision / num_fields if num_fields else 0.0
    recall = total_recall / num_fields if num_fields else 0.0

    # Median F1
    f1_values = sorted(fe.f1_score for fe in field_evals)
    if f1_values:
        mid = len(f1_values) // 2
        if len(f1_values) % 2 == 0:
            median_f1 = (f1_values[mid - 1] + f1_values[mid]) / 2
        else:
            median_f1 = f1_values[mid]
    else:
        median_f1 = 0.0

    correct_fields = sum(1 for fe in field_evals if fe.f1_score == 1.0)

    return field_evals, overall_f1, median_f1, precision, recall, correct_fields


def _build_summary(evaluations: list[ImageEvaluation]) -> dict[str, Any]:
    """Build aggregate summary metrics from per-image evaluations."""
    valid = [e for e in evaluations if e.total_fields > 0]
    total = len(evaluations)
    evaluated = len(valid)

    if not valid:
        return {
            "total_images": total,
            "evaluated_images": 0,
            "avg_overall_f1": 0.0,
            "avg_median_f1": 0.0,
        }

    avg_f1 = sum(e.overall_f1 for e in valid) / evaluated
    avg_median_f1 = sum(e.median_f1 for e in valid) / evaluated

    # By document type
    by_type: dict[str, list[ImageEvaluation]] = {}
    for e in valid:
        by_type.setdefault(e.document_type, []).append(e)

    type_summary = {}
    for doc_type, evals in sorted(by_type.items()):
        n = len(evals)
        type_summary[doc_type] = {
            "count": n,
            "avg_overall_f1": sum(e.overall_f1 for e in evals) / n,
            "avg_median_f1": sum(e.median_f1 for e in evals) / n,
        }

    return {
        "total_images": total,
        "evaluated_images": evaluated,
        "avg_overall_f1": avg_f1,
        "avg_median_f1": avg_median_f1,
        "by_document_type": type_summary,
    }


# ---------------------------------------------------------------------------
# CSV / JSON serialization
# ---------------------------------------------------------------------------

_EVAL_CSV_COLUMNS = [
    "image_name",
    "image_path",
    "document_type",
    "overall_f1",
    "median_f1",
    "precision",
    "recall",
    "total_fields",
    "correct_fields",
    "fields_extracted",
]


def write_evaluation_csv(output: EvaluationOutput, output_path: Path) -> Path:
    """Write evaluation summary CSV (one row per image).

    Args:
        output: EvaluationOutput from evaluate_extractions().
        output_path: Path to write the CSV file.

    Returns:
        The path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_CSV_COLUMNS)
        writer.writeheader()
        for ie in output.image_evaluations:
            writer.writerow(
                {
                    "image_name": ie.image_name,
                    "image_path": ie.image_path,
                    "document_type": ie.document_type,
                    "overall_f1": f"{ie.overall_f1:.4f}",
                    "median_f1": f"{ie.median_f1:.4f}",
                    "precision": f"{ie.precision:.4f}",
                    "recall": f"{ie.recall:.4f}",
                    "total_fields": ie.total_fields,
                    "correct_fields": ie.correct_fields,
                    "fields_extracted": ie.fields_extracted,
                }
            )

    logger.info(
        "Wrote evaluation CSV: %s (%d rows)",
        output_path,
        len(output.image_evaluations),
    )
    return output_path


def write_evaluation_json(output: EvaluationOutput, output_path: Path) -> Path:
    """Write field-level evaluation detail JSON.

    Args:
        output: EvaluationOutput from evaluate_extractions().
        output_path: Path to write the JSON file.

    Returns:
        The path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": output.summary,
        "image_evaluations": [
            {
                "image_name": ie.image_name,
                "image_path": ie.image_path,
                "document_type": ie.document_type,
                "overall_f1": ie.overall_f1,
                "median_f1": ie.median_f1,
                "precision": ie.precision,
                "recall": ie.recall,
                "total_fields": ie.total_fields,
                "correct_fields": ie.correct_fields,
                "fields_extracted": ie.fields_extracted,
                "field_evaluations": [
                    {
                        "field_name": fe.field_name,
                        "f1_score": fe.f1_score,
                        "precision": fe.precision,
                        "recall": fe.recall,
                        "extracted_value": fe.extracted_value,
                        "ground_truth_value": fe.ground_truth_value,
                    }
                    for fe in ie.field_evaluations
                ],
            }
            for ie in output.image_evaluations
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(
        "Wrote evaluation JSON: %s (%d images)",
        output_path,
        len(output.image_evaluations),
    )
    return output_path
