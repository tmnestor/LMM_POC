#!/usr/bin/env python3
"""Measure bank-statement row under-extraction (band-split regression gauge).

Compares extracted transaction-row count vs ground-truth row count per bank
statement on the transaction-linking set. The headline number — % of rows
extracted — is the gauge for the band-split work: today it's ~76%; band-split
should push it toward ~95%+.

Phase 0 of plans/2026-06-05-band-split-bank-extraction.md. Pure CPU; reads the
GT CSV (TRANSACTION_DATES) and the run's cleaned_extractions.jsonl.

Usage:
    python scripts/measure_bank_row_extraction.py <linking_dir>
    # <linking_dir> contains ground_truth_extraction.csv and output/cleaned_extractions.jsonl
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _rows(field: object) -> list[str]:
    return [e.strip() for e in str(field or "").split("|") if e.strip()]


def _layout(name: str) -> str:
    m = re.match(r"CASE\d+_(.+)\.png", name or "")
    return m.group(1) if m else (name or "?")


def _bank_row_count(record: dict) -> int:
    """Transaction-row count from a bank record — cleaned (extracted_data) or
    raw (raw_response) form. extract-only runs produce raw_extractions.jsonl,
    which carries the rows as a 'TRANSACTION_DATES: a | b | c' line."""
    ed = record.get("extracted_data") or {}
    if "TRANSACTION_DATES" in ed:
        return len(_rows(ed.get("TRANSACTION_DATES")))
    for line in str(record.get("raw_response", "")).splitlines():
        if line.strip().upper().startswith("TRANSACTION_DATES"):
            return len(_rows(line.partition(":")[2]))
    return 0


def main(linking_dir: Path) -> int:
    gt_csv = linking_dir / "ground_truth_extraction.csv"
    # Prefer the direct extract output (raw_extractions) so extract-only runs can
    # be measured without running the clean stage; fall back to cleaned.
    out = linking_dir / "output"
    ext_jsonl = out / "raw_extractions.jsonl"
    if not ext_jsonl.exists():
        ext_jsonl = out / "cleaned_extractions.jsonl"
    print(f"reading: {ext_jsonl.name}\n")

    gt: dict[str, int] = {}
    with gt_csv.open() as f:
        for r in csv.DictReader(f):
            if (r.get("DOCUMENT_TYPE") or "").upper() != "BANK_STATEMENT":
                continue
            img = r.get("image_file") or r.get("image_name") or ""
            gt[img] = len(_rows(r.get("TRANSACTION_DATES")))

    ext: dict[str, int] = {}
    for line in ext_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("document_type", "")).upper() != "BANK_STATEMENT":
            continue
        ext[r.get("image_name")] = _bank_row_count(r)

    common = sorted(set(gt) & set(ext))
    tot_e = sum(gt[n] for n in common)
    tot_x = sum(ext[n] for n in common)
    print(f"bank statements: {len(common)}")
    print(
        f"TOTAL rows: expected={tot_e} extracted={tot_x} -> {tot_x / max(tot_e, 1):.1%} "
        f"({tot_e - tot_x} dropped)\n"
    )

    per: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for n in common:
        lay = _layout(n)
        per[lay][0] += gt[n]
        per[lay][1] += ext[n]
        per[lay][2] += 1
    print("per-layout (expected -> extracted, %):")
    for lay, (e, x, k) in sorted(per.items(), key=lambda kv: kv[1][1] / max(kv[1][0], 1)):
        print(f"  {lay:22s} n={k:2d}  {e:4d} -> {x:4d}  ({x / max(e, 1):5.0%})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
