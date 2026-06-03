#!/usr/bin/env python3
"""Generate a quads manifest CSV from trust_distribution_links.yml ground truth.

Reads the ground truth YAML and produces a CSV with columns:
  case_id, trust_return, distribution_stmt, income_schedule, beneficiary_itr

Each row represents one 4-document case for trust distribution linking.

Usage:
    python scripts/generate_trust_manifest.py \
        --ground-truth /path/to/trust_distribution_links.yml \
        --output trust_quads.csv
"""

import csv
import logging
import re
from pathlib import Path

import typer
import yaml

logger = logging.getLogger(__name__)
app = typer.Typer()


def _extract_case_id(filename: str) -> str:
    """Extract CASE### from a filename like CASE201_trust_return_standard.png."""
    match = re.match(r"(CASE\d+)", filename)
    if not match:
        msg = f"Cannot extract case ID from filename: {filename}"
        raise ValueError(msg)
    return match.group(1)


def generate(ground_truth_path: Path, output_path: Path) -> int:
    """Generate quads CSV from ground truth YAML.

    Args:
        ground_truth_path: Path to trust_distribution_links.yml.
        output_path: Path to write the output CSV.

    Returns:
        Number of rows written.
    """
    with ground_truth_path.open() as f:
        data = yaml.safe_load(f)

    if not data:
        msg = f"No data found in {ground_truth_path}"
        raise ValueError(msg)

    rows: list[dict[str, str]] = []
    for dist_stmt_file, entry in data.items():
        case_id = _extract_case_id(dist_stmt_file)
        rows.append(
            {
                "case_id": case_id,
                "trust_return": entry["trust_return"],
                "distribution_stmt": dist_stmt_file,
                "income_schedule": entry["trust_income_schedule"],
                "beneficiary_itr": entry["beneficiary_itr"],
            }
        )

    # Sort by case ID for deterministic output
    rows.sort(key=lambda r: r["case_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "trust_return",
                "distribution_stmt",
                "income_schedule",
                "beneficiary_itr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d quads to %s", len(rows), output_path)
    return len(rows)


@app.command()
def main(
    ground_truth: Path = typer.Option(..., "--ground-truth", help="Path to trust_distribution_links.yml"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to write quads CSV"),
) -> None:
    """Generate trust distribution quads manifest CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    count = generate(ground_truth, output)
    typer.echo(f"Generated {count} quads in {output}")


if __name__ == "__main__":
    app()
