"""Verify a downloaded Hugging Face checkpoint against the repo's own crc32.txt.

Qwen ships a CRC32 for every file in the repository. Checking against that is
stronger than comparing a byte count: a resumed download that stitched a shard
back together incorrectly can land on exactly the right size and still be junk,
and the failure would only surface later as garbage weights at engine load.

Usage:
    python verify_checkpoint_crc32.py /home/jovyan/nfs_share/models/Qwen3.8-27B-FP8

Exits non-zero and names every bad file if anything does not match.
"""

import sys
import zlib
from pathlib import Path

# 8 MiB at a time, so a multi-gigabyte shard never sits in memory all at once.
CHUNK_BYTES = 8 * 1024 * 1024


def crc32_of_file(path: Path) -> str:
    """Return the CRC32 of a file as a lowercase 8-character hex string."""
    digest = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest = zlib.crc32(chunk, digest)
    return f"{digest:08x}"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {Path(sys.argv[0]).name} <model-directory>")

    model_dir = Path(sys.argv[1])
    manifest = model_dir / "crc32.txt"

    if not manifest.is_file():
        sys.exit(
            f"No crc32.txt in {model_dir}.\n"
            "  What:  the per-file checksum manifest is missing, so the download is "
            "either incomplete or pointed at the wrong directory.\n"
            f"  Where: {model_dir}\n"
            "  Expected: a crc32.txt listing one '<crc32>  <filename>' pair per file.\n"
            "  How to fix: re-run the hf download command for this repo; it resumes."
        )

    missing: list[str] = []
    bad_weights: list[str] = []
    bad_metadata: list[str] = []
    checked = 0

    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        expected, name = line.split(None, 1)
        name = name.strip()
        path = model_dir / name

        if not path.is_file():
            missing.append(f"MISSING  {name}")
            continue

        actual = crc32_of_file(path)
        checked += 1
        if actual == expected.lower():
            continue

        detail = f"{name}  expected {expected.lower()}, got {actual}"
        # Weights and metadata fail for different reasons and need different
        # responses, so they are never reported as one undifferentiated count.
        if name.endswith(".safetensors"):
            bad_weights.append(f"WEIGHTS   {detail}")
        else:
            bad_metadata.append(f"METADATA  {detail}")

    print(f"Checked {checked} file(s) against {manifest}")

    if not (missing or bad_weights or bad_metadata):
        print("OK - every file matches its recorded CRC32.")
        return

    print("\n".join(missing + bad_weights + bad_metadata))

    if missing or bad_weights:
        sys.exit(
            f"FAILED: {len(missing)} missing, {len(bad_weights)} corrupt shard(s).\n"
            "  What:  weight data does not match the manifest, so the checkpoint would "
            "load garbage tensors or fail outright.\n"
            f"  Where: {model_dir}\n"
            "  How to fix: re-run the same `hf download` command; it resumes and repairs "
            "only the files that differ."
        )

    # Metadata-only mismatches are usually a STALE MANIFEST, not a bad download:
    # crc32.txt is written once at publication and is not regenerated when the
    # publisher later patches a chat template, tokenizer config or generation
    # config. Re-downloading refetches the identical bytes and fails the same way.
    sys.exit(
        f"REVIEW: every weight shard verified, but {len(bad_metadata)} metadata file(s) differ.\n"
        "  What:  this is most likely a stale crc32.txt - the publisher patched these "
        "files after writing the manifest. It is NOT evidence of a bad download.\n"
        f"  Where: {model_dir}\n"
        "  Expected: confirm by fetching each file from the Hub and comparing its CRC32 "
        "to the local copy; matching values mean the local copy is correct and the "
        "manifest is out of date.\n"
        "  How to fix: if they match the Hub, accept the files and record the deviation. "
        "Do NOT re-run `hf download` - it will refetch identical bytes and fail again."
    )


if __name__ == "__main__":
    main()
