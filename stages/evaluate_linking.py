# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage: Evaluate transaction linking results against ground truth.

Reads transaction_links.jsonl (link stage output), compares against
ground truth (.csv, .yml, or .yaml), and reports match accuracy,
amount/date/description correctness, and confidence calibration.

No GPU needed -- this stage runs on CPU nodes.

Usage:
    python -m stages.evaluate_linking \
        --input /artifacts/transaction_links.jsonl \
        --ground-truth /data/transaction_link_ground_truth.yml \
        --output-dir /artifacts/evaluation
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


def load_linking_ground_truth(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load ground truth, dispatching on file extension.

    Supported formats:
    - ``.csv``: Flat CSV with pipe-delimited multi-receipt fields.
    - ``.jsonl``: One JSON object per line, same field names as CSV.
    - ``.yml`` / ``.yaml``: Human-friendly YAML with per-receipt entries
      grouped by image key.

    Args:
        path: Path to ground truth file (.csv, .jsonl, .yml, or .yaml).

    Returns:
        Mapping of image_file -> list of per-receipt ground truth dicts.
        Each dict has: expected_match, receipt_total, bank_date,
        bank_description, bank_amount, mismatch_type.

    Raises:
        ValueError: If file extension is not supported.
    """
    if path.suffix in (".yml", ".yaml"):
        return _load_linking_ground_truth_yaml(path)
    if path.suffix == ".csv":
        return _load_linking_ground_truth_csv(path)
    if path.suffix == ".jsonl":
        return _load_linking_ground_truth_jsonl(path)
    msg = (
        f"Unsupported ground truth format: {path.suffix!r}. "
        f"Where: {path}. "
        f"Expected: .csv, .jsonl, .yml, or .yaml. "
        f"How to fix: use one of the supported file extensions."
    )
    raise ValueError(msg)


def _load_linking_ground_truth_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load ground truth from CSV with pipe-delimited multi-receipt fields."""
    result: dict[str, list[dict[str, Any]]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_file = row["image_file"].strip()

            # Parse pipe-delimited fields
            match_statuses = _split_pipe(row.get("EXPECTED_MATCH_STATUS", ""))
            receipt_totals = _split_pipe(row.get("RECEIPT_TOTAL", ""))
            bank_dates = _split_pipe(row.get("BANK_TRANSACTION_DATE", ""))
            bank_descriptions = _split_pipe(row.get("BANK_TRANSACTION_DESCRIPTION", ""))
            bank_amounts = _split_pipe(row.get("BANK_TRANSACTION_DEBIT", ""))
            mismatch_types = _split_pipe(row.get("EXPECTED_MISMATCH_TYPE", ""))

            # Determine receipt count from the field with the most entries
            n = max(
                len(match_statuses),
                len(receipt_totals),
                1,
            )

            entries: list[dict[str, Any]] = []
            for i in range(n):
                total_str = receipt_totals[i] if i < len(receipt_totals) else ""
                amount_str = bank_amounts[i] if i < len(bank_amounts) else ""
                entries.append(
                    {
                        "expected_match": (
                            match_statuses[i].upper() if i < len(match_statuses) else "FOUND"
                        ),
                        "receipt_total": _parse_float(total_str),
                        "bank_date": bank_dates[i] if i < len(bank_dates) else "",
                        "bank_description": bank_descriptions[i] if i < len(bank_descriptions) else "",
                        "bank_amount": _parse_float(amount_str),
                        "mismatch_type": (mismatch_types[i].upper() if i < len(mismatch_types) else "NONE"),
                    }
                )

            result[image_file] = entries

    logger.info("Loaded ground truth for %d images from %s", len(result), path)
    return result


def _load_linking_ground_truth_jsonl(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load ground truth from JSONL (one JSON object per line).

    Same field names as CSV: ``image_file``, ``RECEIPT_TOTAL``, etc.
    Pipe-delimited multi-receipt values are split the same way.
    """
    result: dict[str, list[dict[str, Any]]] = {}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # Support both "image_file" and "filename" keys
            image_file = row.get("image_file", row.get("filename", "")).strip()
            if not image_file:
                continue

            match_statuses = _split_pipe(row.get("EXPECTED_MATCH_STATUS", ""))
            receipt_totals = _split_pipe(row.get("RECEIPT_TOTAL", ""))
            bank_dates = _split_pipe(row.get("BANK_TRANSACTION_DATE", ""))
            bank_descriptions = _split_pipe(row.get("BANK_TRANSACTION_DESCRIPTION", ""))
            bank_amounts = _split_pipe(row.get("BANK_TRANSACTION_DEBIT", ""))
            mismatch_types = _split_pipe(row.get("EXPECTED_MISMATCH_TYPE", ""))

            n = max(len(match_statuses), len(receipt_totals), 1)

            entries: list[dict[str, Any]] = []
            for i in range(n):
                total_str = receipt_totals[i] if i < len(receipt_totals) else ""
                amount_str = bank_amounts[i] if i < len(bank_amounts) else ""
                entries.append(
                    {
                        "expected_match": (
                            match_statuses[i].upper() if i < len(match_statuses) else "FOUND"
                        ),
                        "receipt_total": _parse_float(total_str),
                        "bank_date": bank_dates[i] if i < len(bank_dates) else "",
                        "bank_description": bank_descriptions[i] if i < len(bank_descriptions) else "",
                        "bank_amount": _parse_float(amount_str),
                        "mismatch_type": (mismatch_types[i].upper() if i < len(mismatch_types) else "NONE"),
                    }
                )

            result[image_file] = entries

    logger.info("Loaded ground truth for %d images from %s", len(result), path)
    return result


def _load_linking_ground_truth_yaml(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load ground truth from YAML with per-receipt entries grouped by image.

    Expected YAML structure::

        CASE001_receipt.png:
          - supplier: Office Supplies Plus
            receipt_total: 83.48
            match_status: FOUND
            bank_date: 15/01/2024
            bank_description: OFFICE SUPPLIES PLU
            bank_amount: 83.48
    """
    import yaml

    with path.open() as f:
        raw: dict[str, list[dict[str, Any]]] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = (
            f"Invalid YAML ground truth structure in {path}: "
            f"expected top-level mapping of image_file -> list of entries, "
            f"got {type(raw).__name__}."
        )
        raise ValueError(msg)

    result: dict[str, list[dict[str, Any]]] = {}
    for image_file, entries in raw.items():
        converted: list[dict[str, Any]] = []
        for entry in entries:
            converted.append(
                {
                    "expected_match": str(entry.get("match_status", "FOUND")).upper(),
                    "receipt_total": float(entry["receipt_total"])
                    if entry.get("receipt_total") not in (None, "")
                    else None,
                    "bank_date": str(entry.get("bank_date", "")),
                    "bank_description": str(entry.get("bank_description", "")),
                    "bank_amount": float(entry["bank_amount"])
                    if entry.get("bank_amount") not in (None, "")
                    else None,
                    "mismatch_type": str(entry.get("mismatch_type", "NONE")).upper(),
                }
            )
        result[image_file] = converted

    logger.info("Loaded ground truth for %d images from %s", len(result), path)
    return result


def evaluate_linking(
    links: list[dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
    *,
    amount_tolerance: float,
) -> list[dict[str, Any]]:
    """Compare link results against ground truth.

    For multi-receipt images, matches link records to ground truth entries
    by amount (since amounts are unique identifiers within a case).
    Falls back to positional matching if amounts don't disambiguate.
    Dates are compared exactly (DD/MM/YYYY string match).

    Args:
        links: Records from transaction_links.jsonl.
        ground_truth: Output of load_linking_ground_truth().
        amount_tolerance: Tolerance for amount comparisons. Required — the
            stage sources it from pipeline.linking.hybrid_amount_tolerance so
            the evaluation gate always measures what the matcher enforces.

    Returns:
        Per-receipt evaluation records.
    """
    # Group link records by image_name
    links_by_image: dict[str, list[dict[str, Any]]] = {}
    for lnk in links:
        name = lnk.get("image_name", "")
        links_by_image.setdefault(name, []).append(lnk)

    results: list[dict[str, Any]] = []

    # Evaluate each image that appears in ground truth or link output
    all_images = set(ground_truth.keys()) | set(links_by_image.keys())
    for image_name in sorted(all_images):
        gt_entries = ground_truth.get(image_name, [])
        link_entries = links_by_image.get(image_name, [])

        if not gt_entries:
            # Link output exists but no ground truth — log and skip
            for lnk in link_entries:
                results.append(
                    {
                        "image_name": image_name,
                        "receipt_index": 0,
                        "expected_match": "UNKNOWN",
                        "actual_match": lnk.get("matched", False),
                        "correct": None,
                        "confidence": lnk.get("confidence", "NONE"),
                        "amount_correct": None,
                        "date_correct": None,
                        "description_correct": None,
                        "error": "no_ground_truth",
                    }
                )
            continue

        if not link_entries:
            # Ground truth exists but no link output — all missing
            for idx, gt_entry in enumerate(gt_entries):
                results.append(
                    {
                        "image_name": image_name,
                        "receipt_index": idx,
                        "expected_match": gt_entry["expected_match"],
                        "actual_match": None,
                        "correct": False,
                        "confidence": "NONE",
                        "amount_correct": False,
                        "date_correct": False,
                        "description_correct": False,
                        "error": "missing_link_output",
                    }
                )
            continue

        # Match link records to ground truth entries
        paired = _pair_by_amount(link_entries, gt_entries, amount_tolerance)

        for idx, pair in enumerate(paired):
            link_rec = pair[0]
            gt_rec = pair[1]
            if link_rec is None:
                # Ground truth entry with no matching link record
                assert gt_rec is not None  # noqa: S101
                results.append(
                    {
                        "image_name": image_name,
                        "receipt_index": idx,
                        "expected_match": gt_rec["expected_match"],
                        "actual_match": None,
                        "correct": False,
                        "confidence": "NONE",
                        "amount_correct": False,
                        "date_correct": False,
                        "description_correct": False,
                        "error": "unmatched_ground_truth",
                    }
                )
                continue

            if gt_rec is None:
                # Link record with no matching ground truth
                results.append(
                    {
                        "image_name": image_name,
                        "receipt_index": idx,
                        "expected_match": "UNKNOWN",
                        "actual_match": link_rec.get("matched", False),
                        "correct": None,
                        "confidence": link_rec.get("confidence", "NONE"),
                        "amount_correct": None,
                        "date_correct": None,
                        "description_correct": None,
                        "error": "unmatched_link_output",
                    }
                )
                continue

            expected_found = gt_rec["expected_match"] == "FOUND"
            actual_found = bool(link_rec.get("matched", False))
            match_correct = expected_found == actual_found

            # Field-level accuracy (only meaningful when both expect and got FOUND)
            amt_correct = False
            dt_correct = False
            desc_correct = False

            if expected_found and actual_found:
                # Amount check
                bank_amount = link_rec.get("bank_transaction_amount")
                gt_amount = gt_rec.get("bank_amount")
                if bank_amount is not None and gt_amount is not None:
                    amt_correct = abs(float(bank_amount) - float(gt_amount)) <= amount_tolerance

                # Date check
                bank_date = (link_rec.get("bank_transaction_date") or "").strip()
                gt_date = (gt_rec.get("bank_date") or "").strip()
                dt_correct = _dates_match(bank_date, gt_date)

                # Description check (case-insensitive substring match)
                bank_desc = (link_rec.get("bank_transaction_description") or "").strip().upper()
                gt_desc = (gt_rec.get("bank_description") or "").strip().upper()
                desc_correct = _descriptions_match(bank_desc, gt_desc)

            results.append(
                {
                    "image_name": image_name,
                    "receipt_index": idx,
                    "expected_match": gt_rec["expected_match"],
                    "actual_match": actual_found,
                    "correct": match_correct,
                    "confidence": link_rec.get("confidence", "NONE"),
                    "amount_correct": amt_correct,
                    "date_correct": dt_correct,
                    "description_correct": desc_correct,
                }
            )

    return results


def _compute_summary_metrics(
    results: list[dict[str, Any]],
    *,
    inference_seconds: float | None = None,
    fallback_seconds: float = 0.0,
) -> dict[str, Any]:
    """Aggregate per-receipt evaluation records into headline + throughput metrics.

    The denominator for both the match-accuracy table and the throughput figure
    is the number of *scored* receipts (records with a non-``None`` ``correct``),
    matching the summary table's "Total Receipts Evaluated" row.

    Throughput uses ``inference_seconds`` (GPU wall-clock from the link stage)
    when provided and positive, else falls back to the summed per-record
    ``processing_time`` (``fallback_seconds``). When neither is available the
    figure is reported as ``0.0`` and rendered as N/A — never a fake number.

    Args:
        results: Per-receipt evaluation records from ``evaluate_linking``.
        inference_seconds: GPU inference wall-clock seconds, or None.
        fallback_seconds: Sum of per-record ``processing_time`` to use when no
            GPU timing was supplied.

    Returns:
        Summary dict with headline metrics (match accuracy, FOUND/NOT_FOUND
        counts, precision/recall, per-field accuracy), the confidence
        distribution, and a ``throughput`` block.
    """
    scored = [r for r in results if r.get("correct") is not None]
    total = len(scored)
    correct = sum(1 for r in scored if r["correct"])
    accuracy = correct / total * 100 if total else 0.0

    expected_found = [r for r in scored if r["expected_match"] == "FOUND"]
    expected_not_found = [r for r in scored if r["expected_match"] == "NOT_FOUND"]

    # False positives: expected NOT_FOUND but got FOUND.
    false_positives = sum(1 for r in expected_not_found if r["actual_match"])
    # False negatives: expected FOUND but got NOT_FOUND.
    false_negatives = sum(1 for r in expected_found if not r["actual_match"])

    # Precision/recall on the "predicted FOUND" positive class.
    tp = sum(1 for r in expected_found if r["actual_match"])
    precision = tp / (tp + false_positives) if (tp + false_positives) > 0 else 0.0
    recall = tp / (tp + false_negatives) if (tp + false_negatives) > 0 else 0.0

    # Field accuracy (only for correctly matched FOUND results).
    found_correct = [r for r in expected_found if r["actual_match"] and r.get("correct")]
    n_found = len(found_correct)
    amount_correct = sum(1 for r in found_correct if r.get("amount_correct"))
    date_correct = sum(1 for r in found_correct if r.get("date_correct"))
    desc_correct = sum(1 for r in found_correct if r.get("description_correct"))

    def _field(count: int) -> dict[str, Any]:
        return {
            "correct": count,
            "total": n_found,
            "accuracy": count / n_found if n_found else 0.0,
        }

    # Confidence distribution (all scored results).
    confidence_counts: dict[str, int] = {}
    for r in scored:
        conf = r.get("confidence", "NONE")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    # Throughput — prefer the GPU wall-clock, fall back to summed processing_time.
    inference_time = (
        inference_seconds if inference_seconds is not None and inference_seconds > 0 else fallback_seconds
    )
    receipts_per_min = total / inference_time * 60.0 if inference_time > 0 else 0.0
    avg_seconds = inference_time / total if (inference_time > 0 and total) else 0.0

    return {
        "total_receipts_evaluated": total,
        "correct": correct,
        "match_accuracy": accuracy,
        "expected_found": len(expected_found),
        "expected_not_found": len(expected_not_found),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "field_accuracy": {
            "amount": _field(amount_correct),
            "date": _field(date_correct),
            "description": _field(desc_correct),
        },
        "confidence_distribution": confidence_counts,
        "throughput": {
            "receipts_per_min": receipts_per_min,
            "avg_seconds_per_receipt": avg_seconds,
            "inference_seconds": inference_time,
        },
    }


def print_summary(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Print summary + throughput tables to console (same style as evaluate_trust)."""
    console = Console()

    total = summary["total_receipts_evaluated"]
    if total == 0:
        console.print("\n[yellow]No scored results — check ground truth alignment.[/yellow]")
        return

    # Summary table
    table = Table(title="Linking Evaluation Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Receipts Evaluated", str(total))
    table.add_row("Match Accuracy", f"{summary['match_accuracy']:.1f}% ({summary['correct']}/{total})")
    table.add_row("Expected FOUND", str(summary["expected_found"]))
    table.add_row("Expected NOT_FOUND", str(summary["expected_not_found"]))
    table.add_row("False Positives", str(summary["false_positives"]))
    table.add_row("False Negatives", str(summary["false_negatives"]))
    table.add_row("Precision", f"{summary['precision']:.3f}")
    table.add_row("Recall", f"{summary['recall']:.3f}")

    field_accuracy = summary["field_accuracy"]
    n_found = field_accuracy["amount"]["total"]
    if n_found > 0:
        table.add_row("", "")
        for label, key in (
            ("Amount Match (FOUND)", "amount"),
            ("Date Match (FOUND)", "date"),
            ("Description Match (FOUND)", "description"),
        ):
            fa = field_accuracy[key]
            table.add_row(label, f"{fa['correct']}/{n_found} ({fa['accuracy'] * 100:.1f}%)")

    console.print()
    console.print(table)

    # Confidence distribution
    confidence_counts = summary["confidence_distribution"]
    if confidence_counts:
        console.print("\n[bold]Confidence Distribution:[/bold]")
        for level in ("HIGH", "MEDIUM", "LOW", "NONE"):
            count = confidence_counts.get(level, 0)
            if count:
                console.print(f"  {level}: {count}")

    # Throughput table (cyan/green, same style as evaluate_trust._print's tp_table).
    tp = summary["throughput"]
    tp_table = Table(title="Throughput", show_header=True, header_style="bold")
    tp_table.add_column("Metric", style="cyan")
    tp_table.add_column("Value", style="green")
    if tp["inference_seconds"] > 0:
        tp_table.add_row("Receipts/min", f"{tp['receipts_per_min']:.2f}")
        tp_table.add_row("Avg seconds/receipt", f"{tp['avg_seconds_per_receipt']:.1f}")
        tp_table.add_row("Total inference time", f"{tp['inference_seconds']:.1f}s")
    else:
        # No GPU wall-clock and no per-record processing_time — report N/A, not 0.
        tp_table.add_row("Receipts/min", "N/A")
        tp_table.add_row("Avg seconds/receipt", "N/A")
        tp_table.add_row("Total inference time", "N/A (no GPU timing provided)")
    console.print()
    console.print(tp_table)

    # Error summary
    errors = [r for r in results if r.get("error")]
    if errors:
        console.print(f"\n[yellow]Warnings: {len(errors)} records with issues[/yellow]")
        error_types: dict[str, int] = {}
        for r in errors:
            err = r["error"]
            error_types[err] = error_types.get(err, 0) + 1
        for err_type, count in sorted(error_types.items()):
            console.print(f"  {err_type}: {count}")


def run(
    input_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    inference_seconds: float | None = None,
    config_path: Path | None = None,
) -> Path:
    """Orchestration: load, evaluate, write results + summary, print summary.

    Args:
        input_path: Path to transaction_links.jsonl from the link stage.
        ground_truth_path: Path to ground truth file (.csv, .jsonl, .yml, or .yaml).
        output_dir: Directory for evaluation output files.
        inference_seconds: GPU inference wall-clock seconds for throughput
            computation (summed classify + extract + link pod durations). When
            None, throughput falls back to the summed per-record
            ``processing_time``, else reports N/A.
        config_path: Path to run_config.yml. None = the repo default. The
            amount tolerance comes from pipeline.linking.hybrid_amount_tolerance
            (fail-fast if absent) so the evaluator and the link stage can never
            drift apart.

    Note:
        The GPU ``inference_seconds`` spans classify + extract + link over **all**
        images (receipts + bank statements), while the throughput denominator is
        **receipts evaluated** — so the figure is "receipts linked per minute of
        GPU time", not images/min.

    Returns:
        Path to the written linking_evaluation_results.jsonl.
    """
    # Read link results
    links = read_jsonl(input_path)
    if not links:
        msg = f"No records found in {input_path}"
        raise FileNotFoundError(msg)

    logger.info("Read %d linking results from %s", len(links), input_path)

    # Load ground truth
    ground_truth = load_linking_ground_truth(ground_truth_path)

    # Tolerance comes from the SAME validated config block the link stage
    # reads — the evaluation gate must measure what the matcher enforces.
    from stages.transaction_link import _load_linking_config

    linking_cfg = _load_linking_config(config_path)
    amount_tolerance = float(linking_cfg["hybrid_amount_tolerance"])

    # Evaluate
    results = evaluate_linking(links, ground_truth, amount_tolerance=amount_tolerance)

    # Write per-receipt results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "linking_evaluation_results.jsonl"
    count = write_jsonl(output_path, results)
    logger.info("Wrote %d linking evaluation results to %s", count, output_path)

    # Compute summary (headline metrics + throughput). Fallback timing = sum of
    # any per-record processing_time in the link results (else 0.0 -> N/A).
    fallback_seconds = sum(float(lnk.get("processing_time", 0.0) or 0.0) for lnk in links)
    summary = _compute_summary_metrics(
        results, inference_seconds=inference_seconds, fallback_seconds=fallback_seconds
    )

    # Auditability artifact: one JSON record per pod run with throughput.
    summary_path = output_dir / "linking_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote linking evaluation summary to %s", summary_path)

    # Print summary
    print_summary(results, summary)

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_pipe(value: str) -> list[str]:
    """Split a pipe-delimited string into stripped parts.

    Returns empty list for empty/blank input.
    """
    if not value or not value.strip():
        return []
    return [part.strip() for part in value.split("|")]


def _parse_float(value: str) -> float | None:
    """Parse a float from a string, returning None on failure."""
    value = value.strip()
    if not value:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _pair_by_amount(
    link_entries: list[dict[str, Any]],
    gt_entries: list[dict[str, Any]],
    tolerance: float,
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Pair link records with ground truth entries by receipt amount.

    Strategy:
    1. Try amount-based matching (receipt_total from link vs receipt_total
       from ground truth).
    2. Fall back to positional matching for any unpaired entries.

    Returns list of (link_record, gt_entry) tuples. Either may be None
    for unpaired entries.
    """
    used_links: set[int] = set()
    used_gt: set[int] = set()
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []

    # Phase 1: Match by receipt_total amount
    for gi, gt in enumerate(gt_entries):
        gt_total = gt.get("receipt_total")
        if gt_total is None:
            continue
        for li, link in enumerate(link_entries):
            if li in used_links:
                continue
            link_total = link.get("receipt_total")
            if link_total is not None and abs(float(link_total) - float(gt_total)) <= tolerance:
                pairs.append((link, gt))
                used_links.add(li)
                used_gt.add(gi)
                break

    # Phase 2: Positional fallback for unpaired entries
    remaining_links = [link for i, link in enumerate(link_entries) if i not in used_links]
    remaining_gt = [gt for i, gt in enumerate(gt_entries) if i not in used_gt]

    for link, gt in zip(remaining_links, remaining_gt):
        pairs.append((link, gt))

    # Leftover unpaired ground truth
    extra_gt_start = len(remaining_links)
    for gt in remaining_gt[extra_gt_start:]:
        pairs.append((None, gt))

    # Leftover unpaired link records
    extra_link_start = len(remaining_gt)
    for link in remaining_links[extra_link_start:]:
        pairs.append((link, None))

    return pairs


def _dates_match(date1: str, date2: str, tolerance_days: int = 0) -> bool:
    """Check if two date strings match within a tolerance.

    Both dates should be in DD/MM/YYYY format.
    With tolerance_days=0, requires exact string match.
    """
    if not date1 or not date2:
        return False

    if tolerance_days == 0:
        return date1 == date2

    # Parse dates for tolerance comparison
    from datetime import datetime

    try:
        d1 = datetime.strptime(date1, "%d/%m/%Y")
        d2 = datetime.strptime(date2, "%d/%m/%Y")
        return abs((d1 - d2).days) <= tolerance_days
    except ValueError:
        return date1 == date2


def _descriptions_match(actual: str, expected: str) -> bool:
    """Check if bank descriptions match.

    Uses bidirectional substring: either must contain the other,
    or they share enough keyword overlap (at least 50% of expected words).
    """
    if not actual or not expected:
        return False

    if actual == expected:
        return True

    # Substring match (either direction)
    if expected in actual or actual in expected:
        return True

    # Keyword overlap: at least 50% of expected words appear in actual
    expected_words = set(expected.split())
    actual_words = set(actual.split())
    if expected_words:
        overlap = len(expected_words & actual_words)
        return overlap >= len(expected_words) * 0.5

    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to transaction_links.jsonl from link stage"
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g", help="Path to ground truth file (.csv, .jsonl, .yml, or .yaml)"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for evaluation output"),
    inference_seconds: float | None = typer.Option(
        None,
        "--inference-seconds",
        help="GPU inference wall-clock seconds for throughput computation",
    ),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Evaluate transaction linking results against ground truth."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    run(
        input_path,
        ground_truth,
        output_dir,
        inference_seconds=inference_seconds,
        config_path=config,
    )


if __name__ == "__main__":
    app()
