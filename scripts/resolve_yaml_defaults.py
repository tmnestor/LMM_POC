#!/usr/bin/env python3
"""Resolve YAML defaults for entrypoint.sh.

Prints bash-eval-able assignments:
    YAML_DATA_DIR=...
    YAML_GROUND_TRUTH=...
    YAML_OUTPUT_DIR=...

Usage:
    eval "$(python3 scripts/resolve_yaml_defaults.py config/run_config.yml)"

Missing sections/keys yield empty strings (safe under `set -u` with `${var:-}`).
"""

import shlex
import sys
from pathlib import Path

import yaml


def _emit(key: str, value: str | None) -> None:
    # shlex.quote handles paths with spaces/quotes safely.
    print(f"{key}={shlex.quote(value or '')}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: resolve_yaml_defaults.py <config.yml>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        # No config file → all fallbacks empty. Not an error for local dev.
        _emit("YAML_DATA_DIR", "")
        _emit("YAML_GROUND_TRUTH", "")
        _emit("YAML_OUTPUT_DIR", "")
        return 0

    cfg = yaml.safe_load(path.read_text()) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}

    _emit("YAML_DATA_DIR", data.get("dir"))
    _emit("YAML_GROUND_TRUTH", data.get("ground_truth"))
    _emit("YAML_OUTPUT_DIR", output.get("dir"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
