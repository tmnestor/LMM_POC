"""Generate SYNTHETIC batch-result CSVs for MODEL1/MODEL2/MODEL3.

The numbers produced here are FABRICATED. They exist so the comparison dashboard
in notebooks/LMMPOC_comparison.ipynb can be exercised end to end before real
evaluation runs land. Predictions are derived by corrupting the ground truth at a
controlled rate, so the F1/accuracy pipeline itself stays genuine -- only the
model outputs are invented.

Replace these files with real batch results before presenting any of it.

Usage:
    python scripts/generate_synthetic_model_runs.py
"""

import random
import re
from pathlib import Path

import pandas as pd
import yaml

EVAL_DIR = Path("/Users/tod/Desktop/evaluation_data/synthetic_20260728")
SOURCE_YAML = EVAL_DIR / "synthetic.yml"
PROJECT_ROOT = Path(__file__).parents[1]
GT_OUT = PROJECT_ROOT / "evaluation_data" / "synthetic_20260728" / "ground_truth.csv"
CSV_OUT = PROJECT_ROOT / "output" / "csv"

PROVENANCE = "SYNTHETIC_DEMO_DATA_NOT_REAL_MEASUREMENTS"
RUN_STAMP = "20260728_150000"

# The 17 columns the notebook scores on.
FIELD_COLUMNS = [
    "DOCUMENT_TYPE",
    "BUSINESS_ABN",
    "SUPPLIER_NAME",
    "BUSINESS_ADDRESS",
    "PAYER_NAME",
    "PAYER_ADDRESS",
    "INVOICE_DATE",
    "LINE_ITEM_DESCRIPTIONS",
    "LINE_ITEM_QUANTITIES",
    "LINE_ITEM_PRICES",
    "LINE_ITEM_TOTAL_PRICES",
    "IS_GST_INCLUDED",
    "GST_AMOUNT",
    "TOTAL_AMOUNT",
    "STATEMENT_DATE_RANGE",
    "TRANSACTION_DATES",
    "TRANSACTION_AMOUNTS_PAID",
]

DOC_TYPE_MAP = {
    "BANK_STATEMENT": "bank_statement",
    "INVOICE": "invoice",
    "RECEIPT": "receipt",
}

# Fabricated model profiles. field_error/item_error drive extraction quality;
# base_seconds drives the speed panels.
MODEL_PROFILES = {
    "MODEL1": {"field_error": 0.10, "item_error": 0.06, "base_seconds": 40.0},
    "MODEL2": {"field_error": 0.18, "item_error": 0.12, "base_seconds": 22.0},
    "MODEL3": {"field_error": 0.32, "item_error": 0.22, "base_seconds": 9.0},
}

# Dense tables are harder and slower than receipts, mirroring the real runs.
DOC_DIFFICULTY = {"bank_statement": 1.6, "invoice": 1.0, "receipt": 0.7}
DOC_TIME_FACTOR = {"bank_statement": 3.5, "invoice": 1.1, "receipt": 0.9}

DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
MONEY_RE = re.compile(r"^\d[\d,]*\.\d{2}$")


def load_ground_truth() -> list[dict]:
    """Parse the YAML preserving duplicate CASE keys (3 layouts share each id)."""
    text = SOURCE_YAML.read_text()
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    loader = yaml.SafeLoader(text)

    records = []
    for key_node, value_node in root.value:
        case_id = key_node.value
        block = loader.construct_document(value_node)
        fields = block["fields"]
        records.append(
            {
                "image_file": f"{case_id}_{block['layout']}.png",
                "document_type": DOC_TYPE_MAP[fields["DOCUMENT_TYPE"]],
                "fields": {name: str(fields.get(name, "NOT_FOUND")) for name in FIELD_COLUMNS},
            }
        )
    return records


def perturb_scalar(value: str, rng: random.Random) -> str:
    """Return a plausible-but-wrong version of a single value."""
    if value == "NOT_FOUND":
        return "NOT_FOUND"

    roll = rng.random()
    if roll < 0.35:
        return "NOT_FOUND"  # missed the field entirely

    date_match = DATE_RE.match(value)
    if date_match:
        day, month, year = date_match.groups()
        shifted = (int(day) % 28) + 1
        return f"{shifted:02d}/{month}/{year}"

    if MONEY_RE.match(value):
        amount = float(value.replace(",", ""))
        return f"{amount * rng.choice([0.1, 1.1, 10.0]):.2f}"

    words = value.split()
    if len(words) > 1:
        return " ".join(words[:-1])  # truncated read
    return value[:-1] if len(value) > 3 else "UNKNOWN"


def perturb_list(value: str, item_error: float, rng: random.Random) -> str:
    """Corrupt individual items in a pipe-delimited list field."""
    items = value.split("|")
    out = []
    for item in items:
        if rng.random() >= item_error:
            out.append(item)
            continue
        if rng.random() < 0.15:
            continue  # dropped row -- shifts positions, as a real miss would
        out.append(perturb_scalar(item, rng))
    return "|".join(out) if out else "NOT_FOUND"


def build_prediction(record: dict, profile: dict, rng: random.Random) -> dict:
    """Corrupt one document's ground truth into one model's prediction."""
    difficulty = DOC_DIFFICULTY[record["document_type"]]
    field_error = min(profile["field_error"] * difficulty, 0.9)
    item_error = min(profile["item_error"] * difficulty, 0.9)

    predicted = {}
    matched = 0
    scored = 0

    for name, truth in record["fields"].items():
        if truth == "NOT_FOUND":
            predicted[name] = "NOT_FOUND"
            continue

        scored += 1
        if "|" in truth:
            value = perturb_list(truth, item_error, rng)
        elif rng.random() < field_error:
            value = perturb_scalar(truth, rng)
        else:
            value = truth

        predicted[name] = value
        if value == truth:
            matched += 1

    accuracy = (matched / scored * 100) if scored else 0.0
    return {
        "predicted": predicted,
        "overall_accuracy": accuracy,
        "fields_matched": matched,
        "fields_scored": scored,
    }


def main() -> None:
    records = load_ground_truth()
    print(f"loaded {len(records)} ground-truth documents from {SOURCE_YAML}")

    # --- Ground truth CSV the notebook scores against ---
    GT_OUT.parent.mkdir(parents=True, exist_ok=True)
    gt_rows = [{"image_file": r["image_file"], **r["fields"]} for r in records]
    pd.DataFrame(gt_rows).to_csv(GT_OUT, index=False)
    print(f"wrote {GT_OUT} ({len(gt_rows)} rows)")

    # --- One fabricated result CSV per model ---
    for index, (model, profile) in enumerate(MODEL_PROFILES.items()):
        rows = []
        for doc_index, record in enumerate(records):
            # Deterministic per (model, document) so reruns are reproducible.
            rng = random.Random(f"{model}:{record['image_file']}")
            result = build_prediction(record, profile, rng)

            doc_type = record["document_type"]
            seconds = profile["base_seconds"] * DOC_TIME_FACTOR[doc_type] * rng.uniform(0.75, 1.25)

            rows.append(
                {
                    "image_file": record["image_file"],
                    "image_name": record["image_file"],
                    "document_type": doc_type,
                    "processing_time": round(seconds, 6),
                    "field_count": result["fields_scored"],
                    "found_fields": sum(1 for v in result["predicted"].values() if v != "NOT_FOUND"),
                    "field_coverage": round(result["fields_scored"] / len(FIELD_COLUMNS) * 100, 6),
                    "prompt_used": f"{model}_synthetic_demo_prompt",
                    "timestamp": f"2026-07-28T15:{doc_index % 60:02d}:00",
                    "overall_accuracy": round(result["overall_accuracy"], 6),
                    "fields_extracted": result["fields_scored"],
                    "fields_matched": result["fields_matched"],
                    "total_fields": len(FIELD_COLUMNS),
                    "inference_only": False,
                    "data_provenance": PROVENANCE,
                    **result["predicted"],
                }
            )

        out_path = CSV_OUT / f"{model}_SYNTHETIC_batch_results_{RUN_STAMP}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out_path, index=False)
        mean_acc = frame["overall_accuracy"].mean()
        mean_time = frame["processing_time"].mean()
        print(
            f"wrote {out_path.name}: {len(frame)} rows | "
            f"mean exact-match {mean_acc:.1f}% | mean {mean_time:.1f}s"
        )


if __name__ == "__main__":
    main()
