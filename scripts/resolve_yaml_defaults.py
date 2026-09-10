#!/usr/bin/env python3
"""Resolve YAML defaults for entrypoint.sh.

Prints bash-eval-able assignments:
    YAML_MODEL_TYPE=...
    YAML_MODEL_PATH=...
    YAML_DATA_DIR=...
    YAML_GROUND_TRUTH=...
    YAML_OUTPUT_DIR=...
    YAML_LOG_DIR=...

Every name here is read by entrypoint.sh, and every name entrypoint.sh reads
is emitted here. That correspondence is the point of the file: a var emitted
but never read is dead weight, and one read but never emitted resolves to the
empty string under `${VAR:-}` rather than failing, so the run starts against
whatever the CLI default happens to be.

Usage:
    eval "$(python3 scripts/resolve_yaml_defaults.py config/run_config.yml)"

Missing sections/keys yield empty strings (safe under `set -u` with `${var:-}`).
"""

import shlex
import sys
from pathlib import Path

import yaml

# Emitted unconditionally, in this order. Named once so the no-config branch
# and the resolved branch cannot drift apart -- the earlier version listed them
# twice by hand, which is exactly how a var ends up emitted on one path only.
_KEYS = (
    "YAML_MODEL_TYPE",
    "YAML_MODEL_PATH",
    "YAML_DATA_DIR",
    "YAML_GROUND_TRUTH",
    "YAML_OUTPUT_DIR",
    "YAML_LOG_DIR",
)


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
        for key in _KEYS:
            _emit(key, "")
        return 0

    cfg = yaml.safe_load(path.read_text()) or {}
    bootstrap = cfg.get("bootstrap", {}) or {}
    model = bootstrap.get("model", {}) or {}
    log_cfg = bootstrap.get("logging", {}) or {}
    pipeline = cfg.get("pipeline", {}) or {}
    # The shared image-source and output paths live under
    # pipeline.information_extraction.* (moved there from top-level io.* on
    # 2026-06-10). The emitted names stay UNPREFIXED so the entrypoint.sh
    # contract -- and the PROD run_config files edited against it -- are
    # unchanged; the YAML_INFORMATION_EXTRACTION_* rename is deferred.
    info_extract = pipeline.get("information_extraction", {}) or {}
    data = info_extract.get("input", {}) or {}
    output = info_extract.get("output", {}) or {}

    resolved = {
        "YAML_MODEL_TYPE": model.get("type"),
        "YAML_MODEL_PATH": model.get("path"),
        "YAML_DATA_DIR": data.get("dir"),
        "YAML_GROUND_TRUTH": data.get("ground_truth"),
        "YAML_OUTPUT_DIR": output.get("dir"),
        "YAML_LOG_DIR": log_cfg.get("log_dir"),
    }
    for key in _KEYS:
        _emit(key, resolved[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
