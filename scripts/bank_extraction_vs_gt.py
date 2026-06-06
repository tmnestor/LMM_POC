"""Compare PROD bank-statement extraction against synthetic ground truth.

Quantifies dense bank-table extraction quality. For each bank statement it
classifies every ground-truth ``TRANSACTION_AMOUNTS_PAID`` value as:

* **matched** — present in the extraction (within tolerance),
* **missing** — in GT but not extracted (a *dropped* row),
* **spurious** — extracted but not in GT (a *misread*/hallucinated amount),

then reports per-statement and per-layout recall so the worst dense layouts
(e.g. ``cba_date_grouped``) surface for targeted extraction work.

This exists because the residual linking false negatives were traced to
dense-table extraction errors (mis-read amounts + dropped tail rows), NOT a
post-processing bug — see memory ``thousands-comma-cleaner-bug`` (CORRECTION).


Usage::

    python scripts/bank_extraction_vs_gt.py \
        --extracted /home/jovyan/synthetic_clean/output/cleaned_extractions.jsonl \
        --gt /path/to/ground_truth/bank_statements.yml
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

_CASE_RE = re.compile(r"(CASE\d+)")
_NOT_FOUND = "NOT_FOUND"


@dataclass
class GtStmt:
    """Ground-truth bank statement: its layout and paid (debit) amounts."""

    layout: str
    paid: list[float]


@dataclass
class StmtResult:
    """Per-statement comparison result."""

    case: str
    layout: str
    gt: int
    match: int
    miss: int
    spur: int


@dataclass
class LayoutStat:
    """Aggregated counts for one layout."""

    gt: int = 0
    match: int = 0
    miss: int = 0
    spur: int = 0


def summarize_by_layout(results: list[StmtResult]) -> dict[str, LayoutStat]:
    """Sum gt/match/miss/spur across statements, grouped by layout."""
    out: dict[str, LayoutStat] = defaultdict(LayoutStat)
    for r in results:
        stat = out[r.layout]
        stat.gt += r.gt
        stat.match += r.match
        stat.miss += r.miss
        stat.spur += r.spur
    return dict(out)


def _parse_one(token: str) -> float | None:
    """Parse a single currency token to a non-negative float, or None."""
    s = token.strip()
    if not s or s.upper() == _NOT_FOUND:
        return None
    s = re.sub(r"[$,\s]", "", s)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    try:
        return abs(float(s))
    except ValueError:
        return None


def parse_amount_list(value: str) -> list[float]:
    """Parse a pipe-delimited amount string to floats, skipping NOT_FOUND/blank.

    Handles ``$`` prefixes and thousands separators (``$1,234.56`` -> 1234.56).
    """
    if not value:
        return []
    out: list[float] = []
    for token in value.split("|"):
        parsed = _parse_one(token)
        if parsed is not None:
            out.append(parsed)
    return out


def extract_case_id(image_name: str) -> str | None:
    """Return the ``CASE<NNN>`` prefix from an image name, or None."""
    match = _CASE_RE.match(image_name or "")
    return match.group(1) if match else None


def diff_amounts(
    gt: list[float], got: list[float], tol: float = 0.01
) -> tuple[list[float], list[float], list[float]]:
    """Multiset-diff ground-truth amounts against extracted amounts.

    Greedy match within ``tol`` (dollars). Returns ``(matched, missing,
    spurious)`` where matched/missing preserve GT order and spurious is the
    leftover extracted amounts (misreads/hallucinations).
    """
    remaining = list(got)
    matched: list[float] = []
    missing: list[float] = []
    for g in gt:
        idx = next((i for i, x in enumerate(remaining) if abs(x - g) <= tol), None)
        if idx is None:
            missing.append(g)
        else:
            matched.append(g)
            remaining.pop(idx)
    return matched, missing, remaining


def load_gt(path: Path) -> dict[str, GtStmt]:
    """Load bank-statement GT keyed by case id."""
    data = yaml.safe_load(path.read_text())
    out: dict[str, GtStmt] = {}
    for case, block in (data or {}).items():
        if not isinstance(block, dict):
            continue
        fields = block.get("fields", {})
        out[case] = GtStmt(
            layout=str(block.get("layout", "?")),
            paid=parse_amount_list(fields.get("TRANSACTION_AMOUNTS_PAID", "")),
        )
    return out


def load_extracted(path: Path) -> dict[str, list[float]]:
    """Load extracted bank amounts keyed by case id from cleaned_extractions.jsonl."""
    out: dict[str, list[float]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "BANK" not in str(rec.get("document_type", "")).upper():
            continue
        case = extract_case_id(rec.get("image_name", ""))
        if not case:
            continue
        amounts = (rec.get("extracted_data") or {}).get("TRANSACTION_AMOUNTS_PAID", "")
        out[case].extend(parse_amount_list(amounts))
    return dict(out)


def main() -> None:
    """CLI: print per-statement and per-layout extraction recall vs GT."""
    parser = argparse.ArgumentParser(description="Bank extraction vs ground-truth diff")
    parser.add_argument("--extracted", required=True, type=Path, help="cleaned_extractions.jsonl")
    parser.add_argument("--gt", required=True, type=Path, help="ground_truth/bank_statements.yml")
    parser.add_argument("--tol", type=float, default=0.01, help="match tolerance in dollars")
    args = parser.parse_args()

    gt = load_gt(args.gt)
    extracted = load_extracted(args.extracted)

    results: list[StmtResult] = []
    for case, info in sorted(gt.items()):
        if case not in extracted:
            continue
        matched, missing, spurious = diff_amounts(info.paid, extracted[case], args.tol)
        results.append(
            StmtResult(case, info.layout, len(info.paid), len(matched), len(missing), len(spurious))
        )

    print(f"\n{'CASE':<10} {'LAYOUT':<22} {'GT':>4} {'MATCH':>6} {'MISS':>5} {'SPUR':>5} {'RECALL':>7}")
    print("-" * 64)
    for r in sorted(results, key=lambda s: s.match / max(s.gt, 1)):
        recall = r.match / r.gt if r.gt else 1.0
        print(f"{r.case:<10} {r.layout:<22} {r.gt:>4} {r.match:>6} {r.miss:>5} {r.spur:>5} {recall:>6.0%}")

    print(f"\n{'LAYOUT':<24} {'RECALL':>13} {'MISS':>6} {'SPUR':>6}  dominant")
    print("-" * 60)
    by_layout = summarize_by_layout(results)
    for layout, s in sorted(by_layout.items(), key=lambda kv: kv[1].match / max(kv[1].gt, 1)):
        recall_str = f"{s.match}/{s.gt} ({s.match / s.gt:.0%})" if s.gt else "0/0"
        dominant = "dropped" if s.miss > s.spur else "misread" if s.spur > s.miss else "mixed"
        print(f"  {layout:<22} {recall_str:>13} {s.miss:>6} {s.spur:>6}  {dominant}")

    tot = summarize_by_layout(results)
    tot_gt = sum(s.gt for s in tot.values())
    tot_match = sum(s.match for s in tot.values())
    tot_spur = sum(s.spur for s in tot.values())
    overall = tot_match / tot_gt if tot_gt else 1.0
    print(f"\nOVERALL paid-amount recall: {tot_match}/{tot_gt} ({overall:.1%})")
    print(f"Spurious (misread) extracted amounts: {tot_spur}")
    print(f"Statements compared: {len(results)} (GT cases: {len(gt)}, extracted cases: {len(extracted)})")


if __name__ == "__main__":
    main()
