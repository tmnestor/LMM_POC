#!/usr/bin/env python3
"""Measure bank-statement row under-extraction (extraction-quality gauge).

Compares extracted transaction-row count vs ground-truth row count per bank
statement on the transaction-linking set. The headline number — % of rows
extracted — is the diagnostic for dense bank-table under-extraction (the
established root cause of transaction-linking misses).

Pure CPU; reads the GT CSV (TRANSACTION_DATES) and the run's
cleaned_extractions.jsonl.

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


def _debits(field: object) -> list[str]:
    """Debit cells only — the extraction's TRANSACTION_DATES is debit-filtered, so
    we must compare against GT *debits* (non-NOT_FOUND amounts), NOT all rows
    (which include credits/deposits). Comparing debit-vs-all is apples-to-oranges."""
    return [e for e in _rows(field) if e.upper() != "NOT_FOUND"]


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

    # GT debits = non-NOT_FOUND amounts (the extraction outputs debit rows only).
    gt: dict[str, int] = {}
    with gt_csv.open() as f:
        for r in csv.DictReader(f):
            if (r.get("DOCUMENT_TYPE") or "").upper() != "BANK_STATEMENT":
                continue
            img = r.get("image_file") or r.get("image_name") or ""
            gt[img] = len(_debits(r.get("TRANSACTION_AMOUNTS_PAID")))

    ext: dict[str, int] = {}
    for line in ext_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("document_type", "")).upper() != "BANK_STATEMENT":
            continue
        ext[r.get("image_name")] = _bank_row_count(r)  # debit-filtered TRANSACTION_DATES

    common = sorted(set(gt) & set(ext))
    tot_e = sum(gt[n] for n in common)
    tot_x = sum(ext[n] for n in common)
    print(f"bank statements: {len(common)}")
    print(f"DEBIT recovery: extracted={tot_x} / GT_debits={tot_e} = {tot_x / max(tot_e, 1):.1%}")
    print("  (>100% on a layout = over-extraction: seam/header de-dup leak)\n")

    per: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for n in common:
        lay = _layout(n)
        per[lay][0] += gt[n]
        per[lay][1] += ext[n]
        per[lay][2] += 1
    print("per-layout (GT debits -> extracted debits, %):")
    for lay, (e, x, k) in sorted(per.items(), key=lambda kv: kv[1][1] / max(kv[1][0], 1)):
        flag = "  <-- OVER" if x > e else ("  <-- under" if x < e * 0.9 else "")
        print(f"  {lay:22s} n={k:2d}  {e:4d} -> {x:4d}  ({x / max(e, 1):5.0%}){flag}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
