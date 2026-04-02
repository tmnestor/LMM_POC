"""Download the WildReceipt dataset from HuggingFace.

Creates data/wildreceipt/ with:
    images/       — all receipt images (JPEG)
    train.json    — train split annotations
    test.json     — test split annotations

Each JSON file contains a list of records:
    {
        "id": "0",
        "image_path": "images/train_0000.jpeg",
        "words": ["SAFEWAY", "TM", ...],
        "bboxes": [[x1, y1, x2, y2], ...],
        "ner_tags": [1, 25, ...],
        "ner_labels": ["Store_name_value", "Others", ...]
    }

Usage:
    python scripts/download_wildreceipt.py
    python scripts/download_wildreceipt.py --output-dir /path/to/data/wildreceipt
"""

import argparse
import json
from pathlib import Path

NER_LABELS = [
    "Ignore",
    "Store_name_value",
    "Store_name_key",
    "Store_addr_value",
    "Store_addr_key",
    "Tel_value",
    "Tel_key",
    "Date_value",
    "Date_key",
    "Time_value",
    "Time_key",
    "Prod_item_value",
    "Prod_item_key",
    "Prod_quantity_value",
    "Prod_quantity_key",
    "Prod_price_value",
    "Prod_price_key",
    "Subtotal_value",
    "Subtotal_key",
    "Tax_value",
    "Tax_key",
    "Tips_value",
    "Tips_key",
    "Total_value",
    "Total_key",
    "Others",
]


def download(output_dir: Path) -> None:
    from datasets import load_dataset

    print("Downloading WildReceipt from HuggingFace...")
    # Load from the auto-converted parquet files (datasets 4.x dropped script support)
    ds = load_dataset(
        "Theivaprakasham/wildreceipt",
        revision="refs/convert/parquet",
    )

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        split = ds[split_name]
        records = []

        for i, row in enumerate(split):
            # Save image
            fname = f"{split_name}_{i:04d}.jpeg"
            img_path = img_dir / fname

            if hasattr(row.get("image", None), "save"):
                row["image"].save(img_path)
            elif "image_path" in row and Path(row["image_path"]).exists():
                import shutil

                shutil.copy2(row["image_path"], img_path)

            # Build annotation record
            ner_tags = row.get("ner_tags", [])
            record = {
                "id": row.get("id", str(i)),
                "image_path": f"images/{fname}",
                "words": row.get("words", []),
                "bboxes": row.get("bboxes", []),
                "ner_tags": ner_tags,
                "ner_labels": [
                    NER_LABELS[t] if t < len(NER_LABELS) else "Unknown"
                    for t in ner_tags
                ],
            }
            records.append(record)

        out_file = output_dir / f"{split_name}.json"
        out_file.write_text(json.dumps(records, indent=2))
        print(f"  {split_name}: {len(records)} images -> {out_file}")

    # Write label map
    label_map = output_dir / "label_map.json"
    label_map.write_text(
        json.dumps({i: lbl for i, lbl in enumerate(NER_LABELS)}, indent=2)
    )

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"  Images: {len(list(img_dir.glob('*.jpeg')))} files")
    print(f"  Labels: {len(NER_LABELS)} classes")


def main():
    parser = argparse.ArgumentParser(description="Download WildReceipt dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wildreceipt"),
        help="Output directory (default: data/wildreceipt)",
    )
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
