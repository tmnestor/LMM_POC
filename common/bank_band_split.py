"""Band-split primitives for dense bank-statement extraction.

The VLM under-reads long transaction tables: shown a 40-row statement it returns a
*complete* answer covering only ~20 rows (attention decay, not token truncation —
measured at 76% row recovery, see plans/2026-06-05-band-split-bank-extraction.md).
The fix is to slice the statement into overlapping horizontal **bands**, extract
each (short → fully enumerated), then stitch the per-band rows back with
overlap-scoped de-duplication.

This module holds the two pure, testable pieces — band geometry and row stitching.
The VLM calls and per-strategy parsing stay in ``UnifiedBankExtractor``; it crops
with ``band_boxes``/``split_into_bands`` and reassembles with ``merge_rows``.
"""

from collections.abc import Callable, Hashable, Sequence
from typing import Any

from PIL import Image


def band_boxes(
    width: int, height: int, n_bands: int, overlap_frac: float
) -> list[tuple[int, int, int, int]]:
    """Full-width horizontal band crop boxes covering ``height``, with overlap.

    Each band spans ``height / n_bands`` plus an ``overlap_frac`` margin on each
    side (clamped to the image), so a row straddling a cut appears whole in one of
    the two adjacent bands. Returns ``(left, top, right, bottom)`` boxes top→bottom.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if not 0.0 <= overlap_frac < 0.5:
        raise ValueError(f"overlap_frac must be in [0, 0.5), got {overlap_frac}")
    band_h = height / n_bands
    margin = band_h * overlap_frac
    boxes = []
    for i in range(n_bands):
        top = max(0, int(round(i * band_h - margin)))
        bottom = min(height, int(round((i + 1) * band_h + margin)))
        boxes.append((0, top, width, bottom))
    return boxes


def split_into_bands(image: Image.Image, n_bands: int, overlap_frac: float) -> list[Image.Image]:
    """Crop *image* into ``n_bands`` overlapping full-width horizontal strips."""
    if n_bands <= 1:
        return [image]
    w, h = image.size
    return [image.crop(box) for box in band_boxes(w, h, n_bands, overlap_frac)]


def plan_band_count(*, expected_rows: int | None, target_rows_per_band: int, max_bands: int) -> int:
    """Adaptive band count: ~``target_rows_per_band`` rows per band, capped.

    Returns 1 (single-pass, no behaviour change) when the row estimate is unknown
    or small. ``expected_rows`` is a hint (e.g. from a quick count pass); None or
    <= target keeps the statement single-pass so short docs aren't over-split.
    """
    if not expected_rows or expected_rows <= target_rows_per_band:
        return 1
    import math

    return min(max_bands, math.ceil(expected_rows / target_rows_per_band))


def prepend_header(header: Image.Image, band: Image.Image) -> Image.Image:
    """Vertically stack a column-header strip atop a band.

    Header-less band strips (everything below band 0) lose the column-header row,
    so the model mis-assigns debit/credit/balance and rows fail the debit filter.
    Prepending the statement's header strip restores that visual column context.
    """
    width = max(header.width, band.width)
    out = Image.new("RGB", (width, header.height + band.height), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(band, (0, header.height))
    return out


def dedup_rows(
    rows: Sequence[dict[str, Any]], row_key: Callable[[dict[str, Any]], Hashable]
) -> list[dict[str, Any]]:
    """Global, order-preserving de-dup (first occurrence wins).

    Used when bands share content — overlapping seams AND the header strip
    prepended to every band repeat the top rows. Safe for bank statements because
    the running balance makes ``(date, amount, balance)`` unique per real
    transaction, so only true duplicates are removed; two genuine same-day,
    same-amount transactions differ in balance and both survive.
    """
    seen: set[Hashable] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        k = row_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def band_count_for_height(image_height: int, target_band_height: int, max_bands: int) -> int:
    """Height-based band count: ``ceil(image_height / target_band_height)``, capped.

    Decided over a model "how many rows?" pass, which would suffer the same
    attention-decay failure as full-table extraction. Returns 1 (single-pass) for
    images at or below ``target_band_height``.
    """
    if target_band_height <= 0:
        raise ValueError(f"target_band_height must be > 0, got {target_band_height}")
    if image_height <= target_band_height:
        return 1
    import math

    return min(max_bands, math.ceil(image_height / target_band_height))


def merge_rows(
    band_rows: Sequence[Sequence[dict[str, Any]]],
    row_key: Callable[[dict[str, Any]], Hashable],
) -> list[dict[str, Any]]:
    """Stitch per-band row lists into one, de-duplicating the overlap seams.

    For each adjacent pair, finds the largest k where the last k rows of the
    accumulated list equal (by ``row_key``) the first k rows of the next band, and
    drops that duplicated prefix — so overlap rows appear once. De-dup is
    **seam-scoped only**: a legitimately repeated transaction elsewhere in the
    statement (same date+amount on a different line) is preserved, because it is
    not at a band boundary.

    Rows whose ``row_key`` differs across the seam (e.g. one band misread the
    overlap row) are NOT merged — both are kept. Choose a forgiving ``row_key``
    (normalised date + rounded amount, not raw description) to minimise that.
    """
    bands = [list(b) for b in band_rows if b]
    if not bands:
        return []
    merged: list[dict[str, Any]] = list(bands[0])
    for nxt in bands[1:]:
        max_k = min(len(merged), len(nxt))
        overlap = 0
        for k in range(max_k, 0, -1):
            if [row_key(r) for r in merged[-k:]] == [row_key(r) for r in nxt[:k]]:
                overlap = k
                break
        merged.extend(nxt[overlap:])
    return merged
