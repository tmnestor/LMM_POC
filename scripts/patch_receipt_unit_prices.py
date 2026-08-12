#!/usr/bin/env python3
"""Blank receipt unit prices the receipt never printed, in existing ground truth.

A receipt line carries ONE amount, and that amount is the line total::

    3x <item name>                          9.72

The unit price (3.24) is nowhere on the page -- the generator knows it because
it composed the line, but no reader can extract it. Ground truth carrying that
derived value makes ``LINE_ITEM_PRICES`` unscoreable in the honest sense: a
model reporting exactly what is printed is marked wrong, and the only route to a
good score is division, which is arithmetic rather than extraction.

Where the quantity is 1 the unit price and the line total coincide, so the
printed amount IS the unit price and is kept. Everywhere else the position
becomes NOT_FOUND.

INVOICES ARE NOT TOUCHED. Their layout has an explicit ``Unit Price`` column, so
the value is on the page and their answer key is already correct. Applying this
to invoices would destroy correct ground truth, which is why every rewrite is
gated on DOCUMENT_TYPE == RECEIPT.

The same projection is in the generator (``generators/eval_set.py``,
``_blank_unprinted_unit_prices``) so a freshly generated corpus is born correct.
This script exists for corpora already on disk, which must be PATCHED IN PLACE
rather than regenerated -- the Synthetic_Doc_Generation YAMLs have drifted from
the images these datasets contain, so regenerating would replace correct ground
truth with ground truth for different documents.

Idempotent: running it twice changes nothing the second time.

Usage:
    # See what would change, touching nothing:
    python3 scripts/patch_receipt_unit_prices.py --data-dir ~/Desktop/evaluation_data --dry-run

    # Apply, writing a .bak beside every file it rewrites:
    python3 scripts/patch_receipt_unit_prices.py --data-dir ~/Desktop/evaluation_data
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

NOT_FOUND = "NOT_FOUND"
RECEIPT = "RECEIPT"
DOC_TYPE = "DOCUMENT_TYPE"
UNIT_PRICES = "LINE_ITEM_PRICES"
QUANTITIES = "LINE_ITEM_QUANTITIES"


def _diagnostic(what: str, where: str, expected: str, fix: str) -> str:
    return f"What: {what}\nWhere: {where}\nExpected: {expected}\nHow to fix: {fix}"


def blank_unprinted(prices_raw: str, quantities_raw: str) -> str:
    """Return LINE_ITEM_PRICES with unprinted positions blanked to NOT_FOUND.

    Returns the input unchanged when the rule cannot be applied safely: either
    field absent, or the two columns not the same length. Guessing in an answer
    key is worse than leaving it as generated.
    """
    prices_raw, quantities_raw = str(prices_raw).strip(), str(quantities_raw).strip()
    if prices_raw.upper() == NOT_FOUND or quantities_raw.upper() == NOT_FOUND:
        return prices_raw
    if not prices_raw or not quantities_raw:
        return prices_raw

    prices = [item.strip() for item in prices_raw.split("|")]
    quantities = [item.strip() for item in quantities_raw.split("|")]
    if len(prices) != len(quantities):
        return prices_raw

    kept = []
    for price, quantity in zip(prices, quantities):
        try:
            shown = float(quantity) == 1.0
        except ValueError:
            shown = True  # cannot establish it is unprinted; leave it alone
        kept.append(price if shown else NOT_FOUND)

    if all(value == NOT_FOUND for value in kept):
        return NOT_FOUND
    separator = " | " if " | " in prices_raw else "|"
    return separator.join(kept)


def patch_jsonl(path: Path, dry_run: bool) -> tuple[int, int]:
    """Rewrite one ground_truth.jsonl. Returns (receipts seen, records changed)."""
    records = [json.loads(line) for line in path.open() if line.strip()]
    seen = changed = 0
    for record in records:
        if record.get(DOC_TYPE) != RECEIPT:
            continue
        seen += 1
        before = str(record.get(UNIT_PRICES, NOT_FOUND))
        after = blank_unprinted(before, record.get(QUANTITIES, NOT_FOUND))
        if after != before:
            record[UNIT_PRICES] = after
            changed += 1

    if changed and not dry_run:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return seen, changed


def patch_csv(path: Path, dry_run: bool) -> tuple[int, int]:
    """Rewrite one ground_truth.csv. Returns (receipts seen, rows changed)."""
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing = [c for c in (DOC_TYPE, UNIT_PRICES, QUANTITIES) if c not in fieldnames]
    if missing:
        raise SystemExit(
            _diagnostic(
                what=f"{path.name} has no {', '.join(missing)} column(s), so receipts "
                f"cannot be identified or their prices located.",
                where=str(path),
                expected=f"a header row containing {DOC_TYPE}, {UNIT_PRICES} and {QUANTITIES}.",
                fix="point --data-dir at the evaluation_data tree whose ground_truth.csv "
                "came from the extraction ground-truth generator.",
            )
        )

    seen = changed = 0
    for row in rows:
        if row.get(DOC_TYPE) != RECEIPT:
            continue
        seen += 1
        before = str(row.get(UNIT_PRICES, NOT_FOUND))
        after = blank_unprinted(before, row.get(QUANTITIES, NOT_FOUND))
        if after != before:
            row[UNIT_PRICES] = after
            changed += 1

    if changed and not dry_run:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return seen, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="evaluation_data tree holding the dataset directories.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset directory name; repeatable. Defaults to every directory that "
        "has a ground_truth file with receipts in it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes, write nothing.")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(
            _diagnostic(
                what=f"--data-dir '{args.data_dir}' is not a directory.",
                where="the --data-dir argument",
                expected="a path such as ~/Desktop/evaluation_data containing dataset "
                "directories like synthetic_20260811/ and degraded_20260811/.",
                fix="correct --data-dir.",
            )
        )

    names = args.dataset or sorted(d.name for d in args.data_dir.iterdir() if d.is_dir())

    total_changed = 0
    touched = False
    for name in names:
        directory = args.data_dir / name
        for filename, patch in (("ground_truth.jsonl", patch_jsonl), ("ground_truth.csv", patch_csv)):
            path = directory / filename
            if not path.is_file():
                continue
            seen, changed = patch(path, args.dry_run)
            if not seen:
                continue
            touched = True
            total_changed += changed
            verb = "would change" if args.dry_run else "changed"
            print(f"{path}\n    receipts {seen:4d}   {verb} {changed:4d}")

    if not touched:
        raise SystemExit(
            _diagnostic(
                what=f"no ground-truth file with receipts was found under {args.data_dir}.",
                where=str(args.data_dir),
                expected="dataset directories containing ground_truth.jsonl or "
                "ground_truth.csv with DOCUMENT_TYPE == RECEIPT rows.",
                fix="check --data-dir, or name the datasets explicitly with --dataset.",
            )
        )

    print()
    if args.dry_run:
        print(f"DRY RUN -- nothing written. {total_changed} record(s) would change.")
    elif total_changed:
        print(f"Patched {total_changed} record(s). Originals kept alongside as .bak")
        print("Re-run KFP_TASK=evaluate: every LINE_ITEM_PRICES score is now against new truth.")
    else:
        print("Nothing to change -- already patched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
