"""Shared pytest fixtures for the transaction-linking test suite.

tests/ is gitignored — these are local-only tests for the matcher-first
transaction-linking restoration.
"""

import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable (common.*, stages.*) regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Required by the project's engineering standards: fail-fast diagnostics must
# carry all four elements.
_DIAGNOSTIC_ELEMENTS = ("What:", "Where:", "Expected:", "How to fix:")


@pytest.fixture
def assert_diagnostic_error():
    """Return a helper that asserts an error message has all four elements.

    The fail-fast convention requires every config/validation error to state
    What is wrong, Where to fix it, what it should look like (Expected), and
    How to recover.
    """

    def _assert(message: str) -> None:
        missing = [label for label in _DIAGNOSTIC_ELEMENTS if label not in message]
        assert not missing, f"diagnostic message missing element(s) {missing}.\n--- message ---\n{message}"

    return _assert
