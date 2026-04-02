"""Download the WildReceipt dataset from OpenMMLab.

Downloads wildreceipt.tar from the MMOCR project and extracts it.

Creates data/wildreceipt/ with:
    image_files/   — 1,765 receipt images (JPEG)
    train.txt      — train split (1,270 images, MMOCR JSON-lines format)
    test.txt       — test split (472 images)
    class_list.txt — 25 entity classes + Ignore + Others

Usage:
    python scripts/download_wildreceipt.py
    python scripts/download_wildreceipt.py --output-dir /path/to/data/wildreceipt

Source: https://download.openmmlab.com/mmocr/data/wildreceipt.tar
Paper:  https://arxiv.org/abs/2103.14470
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

TAR_URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"


def download(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "wildreceipt.tar"

    # Download
    if tar_path.exists():
        print(f"Archive already exists: {tar_path}")
    else:
        print(f"Downloading {TAR_URL} ...")
        urllib.request.urlretrieve(TAR_URL, tar_path, _progress_hook)
        print()

    # Extract
    print(f"Extracting to {output_dir} ...")
    with tarfile.open(tar_path) as tf:
        # Strip the top-level 'wildreceipt/' prefix so files go directly into output_dir
        members = tf.getmembers()
        for member in members:
            # wildreceipt/image_files/... -> image_files/...
            parts = Path(member.name).parts
            if len(parts) > 1:
                member.name = str(Path(*parts[1:]))
            else:
                continue  # skip the top-level directory itself
            tf.extract(member, output_dir)

    # Report
    images = list((output_dir / "image_files").rglob("*.jpeg")) + list(
        (output_dir / "image_files").rglob("*.jpg")
    )
    train_txt = output_dir / "train.txt"
    test_txt = output_dir / "test.txt"
    class_list = output_dir / "class_list.txt"

    print(f"\nDone! Dataset saved to {output_dir}")
    print(f"  Images:     {len(images)} files")
    if train_txt.exists():
        train_count = sum(1 for _ in train_txt.open())
        print(f"  Train:      {train_count} annotations ({train_txt.name})")
    if test_txt.exists():
        test_count = sum(1 for _ in test_txt.open())
        print(f"  Test:       {test_count} annotations ({test_txt.name})")
    if class_list.exists():
        class_count = sum(1 for _ in class_list.open())
        print(f"  Classes:    {class_count} ({class_list.name})")

    # Clean up tar
    tar_path.unlink()
    print(f"  Removed:    {tar_path.name}")


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


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
