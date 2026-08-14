"""Per-field F1 dispersion for the evaluate stage.

Addresses the gap named in plans/2026-08-13-field-f1-reporting.md: the
pipeline publishes one number per field with no dispersion, no interval and
no sample size, so a field scored on 12 documents reads the same as one
scored on 165.

The inputs are the per-(document, field) F1 values the evaluator already
computes and stores in each record's ``field_scores`` — no scorer change is
needed to report this.

NOTE ON THE INTERVAL. Production per-(document, field) F1 is CONTINUOUS:
list fields (transaction dates, line items) are scored position-agnostically
and earn partial credit. So the interval here is the normal approximation on
a mean, ``1.96 * sd / sqrt(n)``. It is deliberately NOT the Wilson interval
used for the SROIE benchmark, where a field holds a single value and each
document scores 1 or 0 — a proportion interval would be the wrong tool for
continuous scores.
"""

import math
import statistics
from dataclasses import dataclass

_Z_95 = 1.96


@dataclass(frozen=True)
class FieldDistribution:
    """One field's F1 across the documents that were scored for it."""

    field: str
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    n: int


def field_distributions(eval_results: list[dict]) -> list[FieldDistribution]:
    """Summarise per-field F1 across evaluated documents.

    Args:
        eval_results: Evaluation records. Each may carry ``field_scores``
            mapping field name to a dict with ``f1_score``. Records without
            it (errored documents) are skipped rather than counted as
            zeros, which would understate every field.

    Returns:
        One entry per field, WEAKEST FIRST — the table is read to find what
        to fix. ``n`` differs between fields because field sets differ by
        document type, and reporting it is what stops a 12-document field
        being read like a 165-document one.
    """
    by_field: dict[str, list[float]] = {}
    for record in eval_results:
        scores = record.get("field_scores")
        if not isinstance(scores, dict):
            continue
        for name, payload in scores.items():
            if isinstance(payload, dict) and "f1_score" in payload:
                by_field.setdefault(name, []).append(float(payload["f1_score"]))

    rows = []
    for name, values in by_field.items():
        mean = statistics.fmean(values)
        # Population SD: these are all the documents scored for this field,
        # not a sample drawn from a larger pool.
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        half_width = _Z_95 * sd / math.sqrt(len(values)) if values else 0.0
        # Full precision here; rounding happens at render. A rounded field
        # would stop the interval agreeing with its own formula, and any
        # JSON consumer would inherit the loss.
        rows.append(
            FieldDistribution(
                field=name,
                mean=mean,
                sd=sd,
                # F1 cannot leave [0, 1], so an interval must not claim it might.
                ci_low=max(0.0, mean - half_width),
                ci_high=min(1.0, mean + half_width),
                n=len(values),
            )
        )

    return sorted(rows, key=lambda r: (r.mean, r.field))


def render_field_table(rows: list[FieldDistribution]) -> None:
    """Print the per-field table, matching the evaluate stage's style."""
    if not rows:
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title="Per-Field F1", show_header=True, header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Mean", style="green", justify="right")
    table.add_column("SD", style="green", justify="right")
    table.add_column("95% CI", style="green", justify="right")
    table.add_column("n", style="green", justify="right")

    for row in rows:
        table.add_row(
            row.field,
            f"{row.mean:.4f}",
            f"{row.sd:.4f}",
            f"{row.ci_low:.3f}–{row.ci_high:.3f}",
            str(row.n),
        )

    console = Console()
    console.print()
    console.print(table)
