#!/usr/bin/env python3
"""Drop-in replacement for `diff -u` using only the standard library.

Usage:
    pydiff file1 file2          # unified diff (default)
    pydiff -s file1 file2       # silent — exit code only (0=identical, 1=different)
    pydiff -c file1 file2       # context diff
    pydiff --html file1 file2   # HTML side-by-side diff
"""

import argparse
import difflib
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two files (no coreutils required)"
    )
    parser.add_argument("file1", type=Path, help="First file")
    parser.add_argument("file2", type=Path, help="Second file")
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Exit code only, no output"
    )
    parser.add_argument(
        "-c", "--context", action="store_true", help="Context diff format"
    )
    parser.add_argument("--html", action="store_true", help="HTML side-by-side diff")
    parser.add_argument(
        "-n", "--lines", type=int, default=3, help="Context lines (default: 3)"
    )
    args = parser.parse_args()

    try:
        a = args.file1.read_text().splitlines(keepends=True)
        b = args.file2.read_text().splitlines(keepends=True)
    except FileNotFoundError as e:
        print(f"pydiff: {e}", file=sys.stderr)
        return 2

    if a == b:
        if not args.silent:
            print("identical")
        return 0

    if args.silent:
        return 1

    if args.html:
        diff = difflib.HtmlDiff().make_file(a, b, str(args.file1), str(args.file2))
    elif args.context:
        diff = "".join(
            difflib.context_diff(a, b, str(args.file1), str(args.file2), n=args.lines)
        )
    else:
        diff = "".join(
            difflib.unified_diff(a, b, str(args.file1), str(args.file2), n=args.lines)
        )

    print(diff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
