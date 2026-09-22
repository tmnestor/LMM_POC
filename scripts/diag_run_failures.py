"""Diagnose a scored extraction run before believing its headline number.

Prints four sections from a run's own artefacts, read-only:

1. Misclassifications: predicted ``document_type`` against the ground truth.
2. Whole-document failures: records whose ``overall_accuracy`` is below 0.5,
   with how many extracted fields came back empty or ``NOT_FOUND``.
3. One field's failures as predicted-vs-truth value pairs, most common first.
   A format gap (the model says ``Yes``, the key says ``true``) shows up as a
   single dominant pair; a real reading problem is spread out.
4. The score distribution of one list field on one document type. Scores of
   exactly 0.286 / 0.333 / 0.400 are the signature of the pre-2026-09-22
   scorer that punished a correctly predicted ``NOT_FOUND``; seeing them means
   the run was scored before that fix and needs re-scoring.

Every surprising number this project has produced turned out to be a scoring or
parsing artefact rather than a model weakness. Run this before writing anything
up.

Usage:
    python scripts/diag_run_failures.py <run_dir> <ground_truth.jsonl> \
        [--field IS_GST_INCLUDED] [--list-field LINE_ITEM_PRICES] [--doc-type RECEIPT]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SENTINEL_SIGNATURE = {0.286, 0.333, 0.4}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_one(run_dir: Path, name: str) -> Path:
    """Locate a single artefact under the run directory, failing clearly if absent."""
    matches = list(run_dir.rglob(name))
    if len(matches) != 1:
        sys.exit(
            f"Expected exactly one {name} under {run_dir}, found {len(matches)}.\n"
            "  What:  the run directory does not hold a single scored run.\n"
            f"  Where: {run_dir}\n"
            "  Expected: <run_dir>/.../evaluation_results.jsonl and cleaned_extractions.jsonl\n"
            "  How to fix: point at the directory of ONE run (its output.dir in run_config.yml)."
        )
    return matches[0]


def field_f1(entry: dict[str, Any] | None) -> float | None:
    """Return a per-field F1 from a score entry, whichever key the scorer used."""
    if entry is None:
        return None
    if "f1_score" in entry:
        return float(entry["f1_score"])
    if "f1" in entry:
        return float(entry["f1"])
    precision, recall = entry.get("precision", 0.0), entry.get("recall", 0.0)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def is_blank(value: Any) -> bool:
    """True for an empty string or the NOT_FOUND placeholder, case-insensitively."""
    return str(value).strip().upper() in ("", "NOT_FOUND")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--field", default="IS_GST_INCLUDED", help="field for the predicted-vs-truth table")
    parser.add_argument(
        "--list-field", default="LINE_ITEM_PRICES", help="list field for the score distribution"
    )
    parser.add_argument("--doc-type", default="RECEIPT", help="document type for the score distribution")
    args = parser.parse_args()

    evaluations = load_jsonl(find_one(args.run_dir, "evaluation_results.jsonl"))
    predictions = {
        r["image_name"]: r for r in load_jsonl(find_one(args.run_dir, "cleaned_extractions.jsonl"))
    }
    truth = {r.get("filename") or r.get("image_name"): r for r in load_jsonl(args.ground_truth)}

    print(f"{len(evaluations)} scored records\n")

    print("=== 1. MISCLASSIFIED: predicted document_type vs ground truth ===")
    for record in evaluations:
        expected = truth.get(record["image_name"], {}).get("DOCUMENT_TYPE")
        if expected and record["document_type"] != expected:
            print(f"  {record['image_name']:<34} pred={record['document_type']:<15} gt={expected}")

    print("\n=== 2. WHOLE-DOCUMENT FAILURES: overall_accuracy below 0.5 ===")
    for record in sorted(evaluations, key=lambda r: r.get("overall_accuracy", 1.0)):
        accuracy = record.get("overall_accuracy", 1.0)
        if accuracy >= 0.5:
            break
        extracted = predictions.get(record["image_name"], {}).get("extracted_data", {})
        blanks = sum(1 for value in extracted.values() if is_blank(value))
        print(
            f"  {record['image_name']:<34} type={record['document_type']:<15} "
            f"acc={accuracy:.3f}  fields={len(extracted)}  blank/NOT_FOUND={blanks}"
        )

    print(f"\n=== 3. {args.field} failures: what the model SAID vs the answer key ===")
    failures = [
        r
        for r in evaluations
        if args.field in r.get("field_scores", {})
        and (field_f1(r["field_scores"][args.field]) or 0.0) < 1.0
    ]
    print(f"  {len(failures)} failing document(s)")
    pairs = Counter(
        (
            repr(predictions.get(r["image_name"], {}).get("extracted_data", {}).get(args.field)),
            repr(truth.get(r["image_name"], {}).get(args.field)),
        )
        for r in failures
    )
    for (predicted, expected), count in pairs.most_common():
        print(f"  {count:3}x  PRED={predicted:<24} GT={expected}")

    print(f"\n=== 4. {args.list_field} score distribution on {args.doc_type} (scorer-fix check) ===")
    distribution = Counter(
        round(field_f1(r["field_scores"][args.list_field]) or 0.0, 3)
        for r in evaluations
        if r["document_type"] == args.doc_type and args.list_field in r.get("field_scores", {})
    )
    for score, count in sorted(distribution.items()):
        flag = "   <-- pre-fix scorer signature" if score in SENTINEL_SIGNATURE else ""
        print(f"  f1={score}: {count}{flag}")
    if SENTINEL_SIGNATURE & set(distribution):
        print("  => this run was scored BEFORE the NOT_FOUND sentinel fix; re-score it.")


if __name__ == "__main__":
    main()
