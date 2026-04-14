"""JSONL I/O helpers for stage artifact files.

Each stage writes one JSON object per line. Writers flush per-record
so that partial output survives crashes (critical at 10K images).
"""

import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write records to a JSONL file (atomic: writes all at once).

    Returns number of records written.
    """
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    return len(records)


class StreamingJsonlWriter:
    """Append-mode JSONL writer that flushes per record.

    Use this for long-running stages (extract) where crash recovery matters.

    Usage:
        with StreamingJsonlWriter(path) as writer:
            for image in images:
                result = process(image)
                writer.write(result)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = None
        self._count = 0

    def __enter__(self):
        self._file = self._path.open("a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: dict[str, Any]) -> None:
        """Write a single record and flush to disk immediately."""
        if self._file is None:
            msg = "Writer not open. Use as context manager."
            raise RuntimeError(msg)
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        self._count += 1

    @property
    def count(self) -> int:
        """Number of records written so far."""
        return self._count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_jsonl(path: Path) -> int:
    """Count records in a JSONL file without loading them all into memory."""
    if not path.exists():
        return 0
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_completed_images(path: Path) -> set[str]:
    """Read image names from a partial JSONL output for resumption.

    Returns a set of image_name values already processed.
    """
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    name = record.get("image_name", "")
                    if name:
                        completed.add(name)
                except json.JSONDecodeError:
                    pass  # Skip corrupted trailing line from crash
    return completed
