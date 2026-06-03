"""Per-bank column-mapping cache for standalone bank workflow.

Caches column mappings by institution identifier to skip the
detect_headers call on subsequent documents from the same bank.

Cache lifetime is per-run (in-memory, not persisted) to avoid
stale-state bugs across runs.
"""

import re
from pathlib import Path
from typing import Any


class BankHeaderCache:
    """In-memory cache for per-institution column mappings.

    Args:
        key_pattern: Regex to extract institution identifier from filenames.
            Must contain a named group ``institution``.
        enabled: When False, all operations are no-ops.
    """

    def __init__(self, key_pattern: str, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._pattern = re.compile(key_pattern) if enabled else None
        self._cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BankHeaderCache":
        """Build from the ``bank_header_cache`` section of run_config.yml."""
        return cls(
            key_pattern=config["key_pattern"],
            enabled=config["enabled"],
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def extract_key(self, image_path: str) -> str | None:
        """Extract institution key from filename, or None if no match."""
        if not self._enabled or self._pattern is None:
            return None
        filename = Path(image_path).name
        match = self._pattern.search(filename)
        if match:
            try:
                return match.group("institution")
            except IndexError:
                return None
        return None

    def get(self, institution_key: str) -> dict[str, Any] | None:
        """Return cached column mapping for institution, or None on miss."""
        if not self._enabled:
            return None
        return self._cache.get(institution_key)

    def put(self, institution_key: str, column_mapping: dict[str, Any]) -> None:
        """Store column mapping for institution."""
        if not self._enabled:
            return
        self._cache[institution_key] = dict(column_mapping)

    def lookup(self, image_path: str) -> dict[str, Any] | None:
        """Convenience: extract key and look up in one call."""
        key = self.extract_key(image_path)
        if key is None:
            return None
        return self.get(key)

    def store(self, image_path: str, column_mapping: dict[str, Any]) -> None:
        """Convenience: extract key and store in one call."""
        key = self.extract_key(image_path)
        if key is not None:
            self.put(key, column_mapping)
