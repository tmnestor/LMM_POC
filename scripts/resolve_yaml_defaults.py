#!/usr/bin/env python3
"""Resolve YAML defaults for entrypoint.sh.

Prints bash-eval-able assignments:
    YAML_MODEL_TYPE=...
    YAML_DATA_DIR=...
    YAML_GROUND_TRUTH=...
    YAML_OUTPUT_DIR=...
    YAML_TRUST_DATA_DIR=...
    YAML_TRUST_QUADS=...
    YAML_TRUST_QUADS_INCOMPLETE=...
    YAML_TRUST_GROUND_TRUTH=...
    YAML_TRUST_CLASSIFICATION_GT=...
    YAML_TRUST_CLASSIFICATIONS=...
    YAML_TRUST_RAW_EXTRACTIONS=...
    YAML_TRUST_COMPLIANCE_RESULTS=...
    YAML_TRUST_OUTPUT_DIR=...
    YAML_TRUST_EVALUATION_DIR=...
    YAML_TRUST_LOG_DIR=...
    YAML_LINKING_DATA_DIR=...
    YAML_LINKING_OUTPUT=...
    YAML_LINKING_GROUND_TRUTH=...
    YAML_LINKING_EVALUATION_DIR=...
    YAML_LINKING_LOG_DIR=...

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
        _emit("YAML_MODEL_TYPE", "")
        _emit("YAML_DATA_DIR", "")
        _emit("YAML_GROUND_TRUTH", "")
        _emit("YAML_OUTPUT_DIR", "")
        _emit("YAML_TRUST_DATA_DIR", "")
        _emit("YAML_TRUST_QUADS", "")
        _emit("YAML_TRUST_QUADS_INCOMPLETE", "")
        _emit("YAML_TRUST_GROUND_TRUTH", "")
        _emit("YAML_TRUST_CLASSIFICATION_GT", "")
        _emit("YAML_TRUST_CLASSIFICATIONS", "")
        _emit("YAML_TRUST_RAW_EXTRACTIONS", "")
        _emit("YAML_TRUST_COMPLIANCE_RESULTS", "")
        _emit("YAML_TRUST_OUTPUT_DIR", "")
        _emit("YAML_TRUST_EVALUATION_DIR", "")
        _emit("YAML_TRUST_LOG_DIR", "")
        _emit("YAML_LINKING_DATA_DIR", "")
        _emit("YAML_LINKING_OUTPUT", "")
        _emit("YAML_LINKING_GROUND_TRUTH", "")
        _emit("YAML_LINKING_EVALUATION_DIR", "")
        _emit("YAML_LINKING_LOG_DIR", "")
        return 0

    cfg = yaml.safe_load(path.read_text()) or {}
    model = cfg.get("model", {}) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}
    trust = cfg.get("trust_distribution", {}) or {}
    linking = cfg.get("linking", {}) or {}

    _emit("YAML_MODEL_TYPE", model.get("type"))
    _emit("YAML_DATA_DIR", data.get("dir"))
    _emit("YAML_GROUND_TRUTH", data.get("ground_truth"))
    _emit("YAML_OUTPUT_DIR", output.get("dir"))
    _emit("YAML_TRUST_DATA_DIR", trust.get("data_dir"))
    _emit("YAML_TRUST_QUADS", trust.get("quads"))
    _emit("YAML_TRUST_QUADS_INCOMPLETE", trust.get("quads_incomplete"))
    _emit("YAML_TRUST_GROUND_TRUTH", trust.get("ground_truth"))
    _emit("YAML_TRUST_CLASSIFICATION_GT", trust.get("classification_ground_truth"))
    _emit("YAML_TRUST_CLASSIFICATIONS", trust.get("classifications"))
    _emit("YAML_TRUST_RAW_EXTRACTIONS", trust.get("raw_extractions"))
    _emit("YAML_TRUST_COMPLIANCE_RESULTS", trust.get("compliance_results"))
    _emit("YAML_TRUST_OUTPUT_DIR", trust.get("output_dir"))
    _emit("YAML_TRUST_EVALUATION_DIR", trust.get("evaluation_dir"))
    _emit("YAML_TRUST_LOG_DIR", trust.get("log_dir"))
    _emit("YAML_LINKING_DATA_DIR", linking.get("data_dir"))
    _emit("YAML_LINKING_OUTPUT", linking.get("output"))
    _emit("YAML_LINKING_GROUND_TRUTH", linking.get("ground_truth"))
    _emit("YAML_LINKING_EVALUATION_DIR", linking.get("evaluation_dir"))
    _emit("YAML_LINKING_LOG_DIR", linking.get("log_dir"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
