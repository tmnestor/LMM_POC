"""Field-specific canonicalisation for SROIE benchmark scoring.

Every function here either returns a canonical value or raises. Nothing
falls back to comparing raw strings: a fallback of that kind is what made
the previous SROIE run report 0.74 for a date field that was actually
scoring 0.92 — the model was right and the scorer could not tell.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

# Slash, hyphen, dot and space all appear as date separators in the corpus.
_SEPARATORS = re.compile(r"[-/.,\s]+")

_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Shown in diagnostics so the reader sees what an accepted value looks like
# without opening the source.
_ACCEPTED_DATE_EXAMPLES = "15/01/2019, 22-03-2018, 03/03/18, 07 MAR 2018, 2016-07-31"
_ACCEPTED_TOTAL_EXAMPLES = "193.00, 8.20, $8.20, RM 10.60, 1,234.50, -9.99"

# Anchored currency tokens, NOT a character class. A bare [RM] class would
# strip every R and M from the string, which silently mangles any value it
# is pointed at.
_CURRENCY_MARKERS = re.compile(r"^(?:(?:RM|MYR|USD|[$£€¥])\s*)+|\s*(?:RM|MYR|USD)$", re.IGNORECASE)
_THOUSANDS_SEPARATOR = re.compile(r"(\d),(\d)")

_EIGHT_DIGIT_DATE = re.compile(r"\d{8}")

_WHITESPACE = re.compile(r"\s+")
# Everything that is not alphanumeric becomes a space in lenient mode.
_PUNCTUATION = re.compile(r"[^a-z0-9]+")


class SroieNormalisationError(ValueError):
    """A SROIE field value could not be canonicalised."""


def _raise_unparseable(kind: str, raw: str, source: str | None, examples: str) -> None:
    """Raise a diagnostic error naming the value, its origin and a fix."""
    origin = f" (from {source})" if source else ""
    raise SroieNormalisationError(
        f"What: cannot parse {kind} value {raw!r}{origin}.\n"
        f"Where: the {kind} field of that SROIE record.\n"
        f"Expected: one of the corpus formats — {examples}.\n"
        f"How to fix: if this is a real corpus format, add it to "
        f"common/sroie/normalise.py and cover it in "
        f"tests/common/test_sroie_normalise.py. Do NOT make this a silent "
        f"fallback — an unscoreable value must stop the run, not become a "
        f"false negative."
    )


def normalise_total(raw: str, *, source: str | None = None) -> Decimal:
    """Parse a SROIE total into a 2dp decimal amount.

    Args:
        raw: Total exactly as it appears in the ground truth or the model
            response, with or without a currency marker.
        source: Image id or file the value came from, quoted in errors.

    Returns:
        The amount, quantized to two decimal places, sign preserved.

    Raises:
        SroieNormalisationError: If the value holds no parseable amount.
    """
    text = _CURRENCY_MARKERS.sub("", raw.strip())
    text = _THOUSANDS_SEPARATOR.sub(r"\1\2", text).strip()

    try:
        return Decimal(text).quantize(Decimal("0.01"))
    except (InvalidOperation, ArithmeticError):
        _raise_unparseable("total", raw, source, _ACCEPTED_TOTAL_EXAMPLES)
        raise  # unreachable; keeps the return type honest


def normalise_text_strict(raw: str) -> str:
    """Canonicalise company/address for the official exact-match protocol.

    Case and whitespace run-length are forgiven; punctuation is not.
    """
    return _WHITESPACE.sub(" ", raw.strip().lower())


def normalise_text_lenient(raw: str) -> str:
    """Canonicalise company/address for read-the-receipt scoring.

    Additionally forgives punctuation, so '27, JALAN' and '27,JALAN' agree.
    Digits and letters are preserved, so distinct values stay distinct.
    """
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", raw.lower())).strip()


def _month_number(part: str) -> int | None:
    """Return the month number for an alphabetic month name, else None."""
    return _MONTH_NAMES.get(part[:3].lower())


def normalise_date(raw: str, *, source: str | None = None) -> date:
    """Parse a SROIE date string into a calendar date.

    Args:
        raw: Date exactly as it appears in the ground truth or the model
            response.
        source: Image id or file the value came from, quoted in errors.

    Returns:
        The parsed calendar date.

    Raises:
        SroieNormalisationError: If the value matches no known corpus format.
    """
    # Brackets and stray punctuation wrap the value without changing it.
    text = raw.strip().strip("()[]{}<>\"' \t.,")

    # Unseparated 8-digit dates appear in both orderings. A leading 19xx or
    # 20xx can only be a year, which settles it without guessing.
    if _EIGHT_DIGIT_DATE.fullmatch(text):
        if text[:2] in ("19", "20"):
            parts = [text[:4], text[4:6], text[6:]]
        else:
            parts = [text[:2], text[2:4], text[4:]]
    else:
        parts = _SEPARATORS.split(text)

    if len(parts) != 3:
        _raise_unparseable("date", raw, source, _ACCEPTED_DATE_EXAMPLES)

    # An alphabetic month names itself, wherever it sits. The remaining two
    # fields are then day and year, distinguished by width.
    month_index, named_month = next(
        (
            (index, number)
            for index, part in enumerate(parts)
            if (number := _month_number(part)) is not None
        ),
        (-1, None),
    )
    if named_month is not None:
        month = named_month
        rest = [part for index, part in enumerate(parts) if index != month_index]
        # A 4-digit field can only be the year.
        if len(rest[0]) == 4:
            year_part, day_part = rest
        else:
            day_part, year_part = rest
    else:
        # A 4-digit leading field can only be a year, so the order is ISO.
        if len(parts[0]) == 4 and parts[0].isdigit():
            year_part, month_part, day_part = parts
        else:
            day_part, month_part, year_part = parts

        if not month_part.isdigit():
            _raise_unparseable("date", raw, source, _ACCEPTED_DATE_EXAMPLES)
        month = int(month_part)

    if not (day_part.isdigit() and year_part.isdigit()):
        _raise_unparseable("date", raw, source, _ACCEPTED_DATE_EXAMPLES)

    day = int(day_part)
    year = int(year_part)
    if year < 100:
        year += 2000

    # Numeric fields are ambiguous. Malaysian receipts write DD/MM, so
    # day-first is the default; swap only when the month field holds a
    # value no month can take, which makes MM/DD the only reading.
    if month > 12 and day <= 12:
        day, month = month, day

    try:
        return date(year, month, day)
    except ValueError:
        _raise_unparseable("date", raw, source, _ACCEPTED_DATE_EXAMPLES)
        raise  # unreachable; keeps the return type honest
