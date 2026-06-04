#!/usr/bin/env bash
#
# zip_dir.sh — recursively zip a directory using Python's shutil.make_archive.
#
# Usage:
#   ./zip_dir.sh                          # zips ./my_complex_dir  -> ./output/archive.zip
#   ./zip_dir.sh SRC_DIR                  # zips SRC_DIR           -> ./output/archive.zip
#   ./zip_dir.sh SRC_DIR OUT_BASENAME     # zips SRC_DIR           -> OUT_BASENAME.zip
#
# NOTE: the executable bit does not travel with the file (copy/download/git may
# drop it). If you get "Permission denied", restore it once with:
#   chmod +x zip_dir.sh
# Or just run it without the bit:
#   bash zip_dir.sh
#
set -euo pipefail

SRC="${1:-my_complex_dir}"
OUT_BASE="${2:-output/archive}"

python3 - "$SRC" "$OUT_BASE" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_base = Path(sys.argv[2])

if not src.is_dir():
    sys.exit(f"error: source directory does not exist: {src}")

out_base.parent.mkdir(parents=True, exist_ok=True)

archive = shutil.make_archive(
    base_name=str(out_base),  # -> output/archive.zip
    format="zip",             # or "gztar", "bztar", "xztar", "tar"
    root_dir=src.parent,
    base_dir=src.name,
)
print(f"created {archive}")
PY
