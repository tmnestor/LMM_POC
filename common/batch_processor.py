"""Batch processing utilities.

print_accuracy_by_document_type() — accuracy summary for CLI output

The main pipeline class is now DocumentPipeline in document_pipeline.py.
Field definitions are now in common/field_schema.py (FieldSchema + get_field_schema).
"""

import logging

from rich.console import Console

logger = logging.getLogger(__name__)


def print_accuracy_by_document_type(
    batch_results: list[dict],
    console: Console | None = None,
) -> dict:
    """Print accuracy summary grouped dynamically by document type.

    Groups results by the actual document types found in batch_results
    rather than using a hardcoded type list.
    """
    if console is None:
        console = Console()

    # Group results dynamically by document type
    doc_type_results: dict[str, list[dict]] = {}

    for result in batch_results:
        if "error" in result:
            continue

        doc_type = result.get("document_type", "UNKNOWN").upper()
        evaluation = result.get("evaluation", {})

        if not evaluation or "overall_accuracy" not in evaluation:
            continue

        doc_type_results.setdefault(doc_type, []).append(result)

    # Calculate and display metrics for each document type
    console.rule("[bold cyan]Accuracy by Document Type[/bold cyan]")

    summary = {}

    for doc_type_key, results in sorted(doc_type_results.items()):
        if not results:
            continue

        # Extract metrics
        mean_f1_scores = []
        median_f1_scores = []

        for r in results:
            eval_data = r.get("evaluation", {})
            mean_f1_scores.append(eval_data.get("overall_accuracy", 0))
            median_f1_scores.append(eval_data.get("median_f1", 0))

        # Calculate aggregates
        n_docs = len(results)
        avg_mean_f1 = sum(mean_f1_scores) / n_docs if n_docs else 0
        avg_median_f1 = sum(median_f1_scores) / n_docs if n_docs else 0

        # Calculate median of medians (most robust)
        sorted_medians = sorted(median_f1_scores)
        mid = len(sorted_medians) // 2
        if len(sorted_medians) % 2 == 0 and len(sorted_medians) > 0:
            median_of_medians = (sorted_medians[mid - 1] + sorted_medians[mid]) / 2
        elif len(sorted_medians) > 0:
            median_of_medians = sorted_medians[mid]
        else:
            median_of_medians = 0

        display_name = doc_type_key.replace("_", " ").title()

        logger.info("%s", display_name)
        logger.info("  Documents: %d", n_docs)
        logger.info(
            "  Median F1 (avg): %.1f%% - typical field performance",
            avg_median_f1 * 100,
        )
        logger.info("  Mean F1 (avg): %.1f%%", avg_mean_f1 * 100)
        logger.info(
            "  Median of Medians: %.1f%% - most robust", median_of_medians * 100
        )

        summary[doc_type_key] = {
            "count": n_docs,
            "avg_mean_f1": avg_mean_f1,
            "avg_median_f1": avg_median_f1,
            "median_of_medians": median_of_medians,
        }

    # Overall summary (weighted by document count)
    total_docs = sum(s["count"] for s in summary.values())
    if total_docs > 0:
        weighted_median = (
            sum(s["avg_median_f1"] * s["count"] for s in summary.values()) / total_docs
        )
        weighted_mean = (
            sum(s["avg_mean_f1"] * s["count"] for s in summary.values()) / total_docs
        )

        logger.info("Overall (weighted by document count)")
        logger.info("  Total Documents: %d", total_docs)
        logger.info("  Weighted Median F1: %.1f%%", weighted_median * 100)
        logger.info("  Weighted Mean F1: %.1f%%", weighted_mean * 100)

        summary["overall"] = {
            "count": total_docs,
            "weighted_median_f1": weighted_median,
            "weighted_mean_f1": weighted_mean,
        }

    console.rule()

    return summary
