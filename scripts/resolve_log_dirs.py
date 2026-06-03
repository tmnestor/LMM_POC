#!/usr/bin/env python3
"""Extract log directory paths from run_config.yml (stdlib only, no PyYAML).

Runs BEFORE conda activation to determine where tee should write logs.
Only uses Python stdlib -- no third-party imports.

Outputs bash-eval-able assignments:
    YAML_LOG_DIR='...'
    YAML_TRUST_LOG_DIR_EARLY='...'
    YAML_LINKING_LOG_DIR_EARLY='...'

Usage:
    eval "$(python3 scripts/resolve_log_dirs.py config/run_config.yml)"
"""

import re
import shlex
import sys
from pathlib import Path


def _extract_nested_value(text: str, section: str, key: str) -> str:
    """Extract a scalar value from section.key in YAML-like text.

    Handles inline comments, quoted values, and stops at the next
    top-level section.  Not a full YAML parser -- only handles the
    simple ``key: value`` patterns used in run_config.yml.
    """
    section_re = re.compile(rf"^{re.escape(section)}:\s*(?:#.*)?$", re.MULTILINE)
    match = section_re.search(text)
    if not match:
        return ""

    for line in text[match.end() :].split("\n"):
        stripped = line.lstrip()
        # Stop at next top-level key (non-indented, non-comment, non-empty)
        if line and not line[0].isspace() and stripped and not stripped.startswith("#"):
            break
        # Match the target key (must be indented)
        key_match = re.match(rf"^\s+{re.escape(key)}:\s+(.*)", line)
        if key_match:
            value = key_match.group(1).strip()
            # Strip inline comments
            value = re.sub(r"\s+#.*$", "", value)
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value
    return ""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: resolve_log_dirs.py <config.yml>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        # No config file -> all values empty (safe under set -u with ${var:-}).
        print(f"YAML_LOG_DIR={shlex.quote('')}")
        print(f"YAML_TRUST_LOG_DIR_EARLY={shlex.quote('')}")
        print(f"YAML_LINKING_LOG_DIR_EARLY={shlex.quote('')}")
        return 0

    text = path.read_text()

    log_dir = _extract_nested_value(text, "logging", "log_dir")
    trust_log_dir = _extract_nested_value(text, "trust_distribution", "log_dir")
    linking_log_dir = _extract_nested_value(text, "linking", "log_dir")

    print(f"YAML_LOG_DIR={shlex.quote(log_dir)}")
    print(f"YAML_TRUST_LOG_DIR_EARLY={shlex.quote(trust_log_dir)}")
    print(f"YAML_LINKING_LOG_DIR_EARLY={shlex.quote(linking_log_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
