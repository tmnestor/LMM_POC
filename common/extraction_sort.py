"""Sort utilities for extraction batch ordering.

Sorts classification records by doc_type (primary) and optionally by
image dimensions (secondary) to maximize vLLM prefix cache reuse.
"""

from pathlib import Path
from typing import Any

from PIL import Image


def sort_for_extraction(
    records: list[dict[str, Any]],
    type_order: list[str],
    *,
    secondary_sort: str = "none",
) -> list[dict[str, Any]]:
    """Sort classification records for optimal extraction batching.

    Primary key: doc_type position in *type_order* (stable sort preserves
    input order within each type). Unknown types land at the end.

    Secondary key (within each type group):
    - ``none``: no secondary sort (input order preserved).
    - ``image_area_asc``: ascending by image width * height.
    - ``image_area_desc``: descending by image width * height.

    Args:
        records: Classification dicts, each with at least ``document_type``.
        type_order: Ordered list of doc_type strings (first = highest priority).
        secondary_sort: One of ``none``, ``image_area_asc``, ``image_area_desc``.

    Returns:
        A new list in the desired order (does not mutate *records*).
    """
    sentinel = len(type_order)
    order_map = {t.lower(): i for i, t in enumerate(type_order)}

    def _primary_key(rec: dict[str, Any]) -> int:
        return order_map.get(rec.get("document_type", "").lower(), sentinel)

    if secondary_sort == "none":
        return sorted(records, key=_primary_key)

    # For image_area sorts, compute area lazily per record
    area_cache: dict[str, int] = {}

    def _get_area(rec: dict[str, Any]) -> int:
        image_path = rec.get("image_path", "")
        if image_path in area_cache:
            return area_cache[image_path]
        area = _read_image_area(image_path)
        area_cache[image_path] = area
        return area

    ascending = secondary_sort == "image_area_asc"

    def _composite_key(rec: dict[str, Any]) -> tuple[int, int]:
        area = _get_area(rec)
        return (_primary_key(rec), area if ascending else -area)

    return sorted(records, key=_composite_key)


def _read_image_area(image_path: str) -> int:
    """Read image dimensions and return width * height.

    Returns 0 if the file cannot be opened (graceful degradation).
    """
    try:
        p = Path(image_path)
        if p.exists():
            with Image.open(p) as img:
                w, h = img.size
                return w * h
    except Exception:
        pass
    return 0


def filter_skip_labels(
    records: list[dict[str, Any]],
    skip_labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition records into (to_extract, skipped) based on skip labels.

    Args:
        records: Classification dicts with ``document_type`` key.
        skip_labels: Labels that should bypass extraction.

    Returns:
        Tuple of (records to process, records to skip).
    """
    if not skip_labels:
        return records, []

    skip_set = {label.lower() for label in skip_labels}
    to_extract: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for rec in records:
        doc_type = rec.get("document_type", "")
        if doc_type.lower() in skip_set:
            skipped.append(rec)
        else:
            to_extract.append(rec)

    return to_extract, skipped
