"""Benchmark artefacts: a per-image CSV and a summary JSON.

The per-image file is the one a disputed score gets checked in, so it
carries both sides of every comparison and both policies' verdicts. The
summary keeps raw tp/fp/fn counts alongside the ratios — dropping them
would make any pooled metric uncomputable without re-running the model.
"""

import csv
import json
from pathlib import Path

from common.sroie.ground_truth import SROIE_FIELDS, SroieRecord
from common.sroie.scoring import MatchPolicy, PolicyScore, field_matches


def write_per_image_csv(
    path: Path,
    records: list[SroieRecord],
    predictions: dict[str, dict[str, str]],
) -> Path:
    """Write one row per receipt, with both policies' verdicts per field.

    Args:
        path: Destination CSV path.
        records: Ground-truth records for the split.
        predictions: Image id to parsed field values.

    Returns:
        The path written.
    """
    columns = ["image_id"]
    for field_name in SROIE_FIELDS:
        columns += [
            f"{field_name}_gt",
            f"{field_name}_pred",
            f"{field_name}_strict",
            f"{field_name}_lenient",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for record in records:
            answers = predictions.get(record.image_id, {})
            row: dict[str, object] = {"image_id": record.image_id}
            for field_name in SROIE_FIELDS:
                expected = getattr(record, field_name)
                actual = answers.get(field_name, "")
                row[f"{field_name}_gt"] = expected
                row[f"{field_name}_pred"] = actual
                for policy in MatchPolicy:
                    matched = bool(
                        actual
                        and field_matches(
                            field_name,
                            expected,
                            actual,
                            policy=policy,
                            source=record.image_id,
                        )
                    )
                    row[f"{field_name}_{policy.value}"] = matched
            writer.writerow(row)

    return path


def write_summary_json(
    path: Path,
    *,
    model_name: str,
    scores: dict[MatchPolicy, PolicyScore],
    image_count: int,
    elapsed_seconds: float,
) -> Path:
    """Write the headline numbers for both policies.

    Args:
        path: Destination JSON path.
        model_name: Model type as configured, recorded for comparability.
        scores: One PolicyScore per match policy.
        image_count: Number of receipts scored.
        elapsed_seconds: Wall-clock inference time.

    Returns:
        The path written.
    """
    summary = {
        "model": model_name,
        "total_images": image_count,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "policies": {
            policy.value: {
                "overall_f1": round(score.overall_f1, 4),
                "per_field": {
                    field_name: {
                        "precision": round(counts.precision, 4),
                        "recall": round(counts.recall, 4),
                        "f1": round(counts.f1, 4),
                        "true_positives": counts.true_positives,
                        "false_positives": counts.false_positives,
                        "false_negatives": counts.false_negatives,
                    }
                    for field_name, counts in score.per_field.items()
                },
            }
            for policy, score in scores.items()
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path
