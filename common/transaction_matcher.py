"""Transaction matching engine: receipt-to-bank-statement linking.

Pure algorithmic matching on already-extracted fields. No model inference.
Decoupled from stage I/O for testability.
"""

import logging
import re
import string
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BankTransaction:
    """Single transaction row from a bank statement."""

    date: date | None
    description: str
    amount: float
    source_image: str
    row_index: int


@dataclass(frozen=True, slots=True)
class ReceiptSummary:
    """Key fields from a receipt/invoice for matching."""

    image_name: str
    supplier_name: str
    date: date | None
    total: float | None
    document_type: str


@dataclass(frozen=True, slots=True)
class LinkResult:
    """Result of matching one receipt to the transaction index."""

    receipt: ReceiptSummary
    matched: bool
    transaction: BankTransaction | None
    confidence: str  # HIGH / MEDIUM / LOW / NONE
    match_scores: dict[str, float]
    reasoning: str


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------


def parse_amount(amount_str: str) -> float | None:
    """Parse '$127.35', '127.35', '-$127.35', '1,234.56' → float.

    Returns None if the string cannot be parsed or is a NOT_FOUND sentinel.
    """
    if not amount_str or amount_str.strip().upper() == "NOT_FOUND":
        return None
    cleaned = amount_str.strip()
    # Remove currency symbols and whitespace
    cleaned = cleaned.replace("$", "").replace("€", "").replace("£", "").replace(" ", "")
    # Track sign
    negative = cleaned.startswith("-") or cleaned.startswith("(")
    cleaned = cleaned.strip("-()").strip()
    # Remove thousands separators
    cleaned = cleaned.replace(",", "")
    try:
        value = float(cleaned)
        return -value if negative else value
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
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
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def normalize_date(date_str: str) -> date | None:
    """Normalize '18 Mar 2024', '22 Dec 23', '18/03/2024', '2024-03-18' → date object.

    Returns None if the string cannot be parsed or is a NOT_FOUND sentinel.
    """
    if not date_str or date_str.strip().upper() == "NOT_FOUND":
        return None
    cleaned = date_str.strip()

    # ISO format: 2024-03-18
    if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        parts = cleaned.split("-")
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", cleaned)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None

    # DD/MM/YY or DD-MM-YY (2-digit year)
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})$", cleaned)
    if m:
        yy = int(m.group(3))
        year = 2000 + yy if yy <= 68 else 1900 + yy
        try:
            return date(year, int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None

    # DD Mon YYYY (e.g., "18 Mar 2024", "18 March 2024")
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", cleaned)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        year = int(m.group(3))
        month = _MONTH_MAP.get(month_str)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None

    # DD Mon YY (e.g., "22 Dec 23", "18 Mar 24")
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{2})$", cleaned)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        yy = int(m.group(3))
        year = 2000 + yy if yy <= 68 else 1900 + yy
        month = _MONTH_MAP.get(month_str)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None

    return None


# ---------------------------------------------------------------------------
# Description matching
# ---------------------------------------------------------------------------

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(text: str) -> list[str]:
    """Normalize text for comparison: uppercase, strip punctuation, tokenize."""
    upper = text.upper()
    upper = upper.replace("&", "AND")
    upper = upper.translate(_PUNCTUATION_TABLE)
    return upper.split()


def description_score(supplier_name: str, bank_description: str) -> float:
    """Score how well supplier_name matches bank_description.

    Strategy:
    1. Normalize both: uppercase, strip punctuation, & → AND
    2. Tokenize supplier name
    3. Check if supplier tokens appear in bank description tokens
    4. Return fraction of supplier tokens found (1.0 = all found)

    Returns 0.0 if supplier_name is empty/NOT_FOUND.
    """
    if not supplier_name or supplier_name.strip().upper() == "NOT_FOUND":
        return 0.0
    if not bank_description:
        return 0.0

    supplier_tokens = _normalize_text(supplier_name)
    if not supplier_tokens:
        return 0.0

    bank_tokens = _normalize_text(bank_description)
    if not bank_tokens:
        return 0.0

    # Check how many supplier tokens appear in bank description
    found = sum(1 for token in supplier_tokens if token in bank_tokens)
    return found / len(supplier_tokens)


# ---------------------------------------------------------------------------
# Case grouping
# ---------------------------------------------------------------------------


def extract_case_id(image_name: str, pattern: re.Pattern[str]) -> str | None:
    """Extract case ID from filename using configured regex.

    Uses a named capture group 'case' (e.g., '^(?P<case>[^_]+)_').
    Returns None if the pattern doesn't match.
    """
    m = pattern.match(image_name)
    if m:
        try:
            return m.group("case")
        except IndexError:
            return None
    return None


def group_by_case(
    records: list[dict[str, Any]], pattern: re.Pattern[str]
) -> dict[str, list[dict[str, Any]]]:
    """Group extraction records by case ID.

    Records whose filenames don't match the pattern are collected
    under a special '_ungrouped' key and logged as warnings.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        image_name = record.get("image_name", "")
        case_id = extract_case_id(image_name, pattern)
        if case_id is None:
            logger.warning("Filename does not match case pattern: %s", image_name)
            groups.setdefault("_ungrouped", []).append(record)
        else:
            groups.setdefault(case_id, []).append(record)
    return groups


# ---------------------------------------------------------------------------
# Transaction index builder
# ---------------------------------------------------------------------------


def build_transaction_index(bank_records: list[dict[str, Any]]) -> list[BankTransaction]:
    """Parse bank statement records for ONE case into a flat transaction list.

    Each bank statement has pipe-delimited parallel lists in extracted_data:
    - TRANSACTION_DATES: "18 Mar 2024 | 19 Mar 2024 | ..."
    - LINE_ITEM_DESCRIPTIONS: "WOOLWORTHS 2847 | TELSTRA | ..."
    - TRANSACTION_AMOUNTS_PAID: "127.35 | 48.50 | ..."

    Returns flat list of all transactions across all bank statements in this case.
    """
    transactions: list[BankTransaction] = []

    for record in bank_records:
        image_name = record.get("image_name", "")
        extracted = record.get("extracted_data", {})

        dates_raw = extracted.get("TRANSACTION_DATES", "")
        descriptions_raw = extracted.get("LINE_ITEM_DESCRIPTIONS", "")
        amounts_raw = extracted.get("TRANSACTION_AMOUNTS_PAID", "")

        # Skip empty / NOT_FOUND fields
        if not dates_raw or dates_raw.strip().upper() == "NOT_FOUND":
            dates_raw = ""
        if not descriptions_raw or descriptions_raw.strip().upper() == "NOT_FOUND":
            descriptions_raw = ""
        if not amounts_raw or amounts_raw.strip().upper() == "NOT_FOUND":
            continue  # No amounts = no usable transactions

        # Split pipe-delimited fields
        dates = [d.strip() for d in dates_raw.split("|")] if dates_raw else []
        descriptions = [d.strip() for d in descriptions_raw.split("|")] if descriptions_raw else []
        amounts = [a.strip() for a in amounts_raw.split("|")]

        # Build transaction for each amount entry
        for idx, amount_str in enumerate(amounts):
            amount = parse_amount(amount_str)
            if amount is None:
                continue

            txn_date = normalize_date(dates[idx]) if idx < len(dates) else None
            txn_desc = descriptions[idx] if idx < len(descriptions) else ""

            transactions.append(
                BankTransaction(
                    date=txn_date,
                    description=txn_desc,
                    amount=abs(amount),
                    source_image=image_name,
                    row_index=idx,
                )
            )

    return transactions


# ---------------------------------------------------------------------------
# Receipt summary builder
# ---------------------------------------------------------------------------


def _is_pipe_delimited(value: str) -> bool:
    """Check if a field value contains pipe-delimited multiple entries."""
    return "|" in value and value.strip().upper() != "NOT_FOUND"


def build_receipt_summaries(record: dict[str, Any]) -> list[ReceiptSummary]:
    """Build ReceiptSummary list from a cleaned extraction record.

    Handles multi-receipt images where TOTAL_AMOUNT, SUPPLIER_NAME, and
    INVOICE_DATE are pipe-delimited (e.g., "$83.48 | $39.70 | $142.80").
    Returns one ReceiptSummary per sub-receipt found.
    """
    extracted = record.get("extracted_data", {})
    image_name = record.get("image_name", "")
    document_type = record.get("document_type", "")

    total_raw = extracted.get("TOTAL_AMOUNT", "")
    supplier_raw = extracted.get("SUPPLIER_NAME", "NOT_FOUND")
    date_raw = extracted.get("INVOICE_DATE", "")

    # Detect multi-receipt: if TOTAL_AMOUNT is pipe-delimited, split all fields
    if _is_pipe_delimited(total_raw):
        totals = [t.strip() for t in total_raw.split("|")]
        suppliers = (
            [s.strip() for s in supplier_raw.split("|")]
            if _is_pipe_delimited(supplier_raw)
            else [supplier_raw] * len(totals)
        )
        dates = (
            [d.strip() for d in date_raw.split("|")]
            if _is_pipe_delimited(date_raw)
            else [date_raw] * len(totals)
        )

        summaries: list[ReceiptSummary] = []
        for idx, amount_str in enumerate(totals):
            supplier = suppliers[idx] if idx < len(suppliers) else suppliers[-1]
            date_str = dates[idx] if idx < len(dates) else dates[-1]
            summaries.append(
                ReceiptSummary(
                    image_name=image_name,
                    supplier_name=supplier,
                    date=normalize_date(date_str),
                    total=parse_amount(amount_str),
                    document_type=document_type,
                )
            )
        return summaries

    # Single receipt
    return [
        ReceiptSummary(
            image_name=image_name,
            supplier_name=supplier_raw,
            date=normalize_date(date_raw),
            total=parse_amount(total_raw),
            document_type=document_type,
        )
    ]


# ---------------------------------------------------------------------------
# Date scoring
# ---------------------------------------------------------------------------


def _business_days_between(d1: date, d2: date) -> int:
    """Count business days between two dates (absolute difference)."""
    if d1 > d2:
        d1, d2 = d2, d1
    days = 0
    current = d1
    while current < d2:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days += 1
    return days


def _date_score(receipt_date: date | None, txn_date: date | None, window_days: int) -> float:
    """Score date proximity: 1.0 for same day, decaying over window."""
    if receipt_date is None or txn_date is None:
        return 0.0
    bdays = _business_days_between(receipt_date, txn_date)
    if bdays == 0:
        return 1.0
    if bdays > window_days:
        return 0.0
    # Linear decay: +1 day = 0.9, +2 = 0.8, ..., +window = slightly above 0
    return max(0.0, 1.0 - (bdays / (window_days + 1)))


# ---------------------------------------------------------------------------
# Matching algorithm
# ---------------------------------------------------------------------------


def match_receipt(
    receipt: ReceiptSummary,
    index: list[BankTransaction],
    *,
    amount_tolerance: float = 0.01,
    date_window_days: int = 5,
    description_threshold: float = 0.5,
) -> LinkResult:
    """Match a single receipt against the transaction index.

    Three-gate matching (in priority order):
    1. Amount gate (hard): |receipt.total - txn.amount| <= tolerance
    2. Date window (soft): scoring based on business-day proximity
    3. Description match (soft): token-based supplier name containment

    Confidence assignment:
    - HIGH: amount exact + date match + description match
    - MEDIUM: amount exact + (date within window OR description match)
    - LOW: amount exact only (no date/description support)
    - NONE: no amount match found
    """
    if receipt.total is None:
        return LinkResult(
            receipt=receipt,
            matched=False,
            transaction=None,
            confidence="NONE",
            match_scores={"amount": 0.0, "date": 0.0, "description": 0.0},
            reasoning="Receipt has no parseable total amount",
        )

    # Gate 1: Filter by amount
    candidates: list[tuple[BankTransaction, float]] = []
    for txn in index:
        diff = abs(receipt.total - txn.amount)
        if diff <= amount_tolerance:
            candidates.append((txn, diff))

    if not candidates:
        return LinkResult(
            receipt=receipt,
            matched=False,
            transaction=None,
            confidence="NONE",
            match_scores={"amount": 0.0, "date": 0.0, "description": 0.0},
            reasoning=f"No bank transaction found with amount ${receipt.total:.2f}",
        )

    # Score all candidates on date + description
    best_txn: BankTransaction | None = None
    best_score = -1.0
    best_date_score = 0.0
    best_desc_score = 0.0

    for txn, amount_diff in candidates:
        d_score = _date_score(receipt.date, txn.date, date_window_days)
        desc_sc = description_score(receipt.supplier_name, txn.description)
        combined = d_score + desc_sc  # Equal weight, max 2.0
        if combined > best_score or (
            combined == best_score and best_txn is not None and txn.row_index < best_txn.row_index
        ):
            best_score = combined
            best_txn = txn
            best_date_score = d_score
            best_desc_score = desc_sc

    assert best_txn is not None  # noqa: S101 — we have at least one candidate

    # Determine confidence
    amount_score = 1.0  # We passed the gate
    has_date = best_date_score > 0.0
    has_desc = best_desc_score >= description_threshold

    if has_date and has_desc:
        confidence = "HIGH"
    elif has_date or has_desc:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Build reasoning
    parts: list[str] = [f"Exact amount (${receipt.total:.2f})"]
    if has_date:
        parts.append(f"date match (score={best_date_score:.2f})")
    if has_desc:
        parts.append(f"supplier '{receipt.supplier_name}' found in description")
    reasoning = ", ".join(parts)

    return LinkResult(
        receipt=receipt,
        matched=True,
        transaction=best_txn,
        confidence=confidence,
        match_scores={
            "amount": amount_score,
            "date": best_date_score,
            "description": best_desc_score,
        },
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Batch matching with one-to-one constraint
# ---------------------------------------------------------------------------


def match_all_receipts(
    receipts: list[ReceiptSummary],
    index: list[BankTransaction],
    *,
    amount_tolerance: float = 0.01,
    date_window_days: int = 5,
    description_threshold: float = 0.5,
) -> list[LinkResult]:
    """Match all receipts against the transaction index with one-to-one constraint.

    Each bank transaction can only match ONE receipt. Processes receipts in
    order that maximizes match quality (best potential matches first).
    """
    # First pass: score all receipts to determine processing order
    scored: list[tuple[int, LinkResult]] = []
    for i, receipt in enumerate(receipts):
        result = match_receipt(
            receipt,
            index,
            amount_tolerance=amount_tolerance,
            date_window_days=date_window_days,
            description_threshold=description_threshold,
        )
        scored.append((i, result))

    # Sort by confidence (HIGH first) then by total score (descending)
    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    scored.sort(
        key=lambda x: (
            confidence_order.get(x[1].confidence, 4),
            -(x[1].match_scores.get("date", 0) + x[1].match_scores.get("description", 0)),
        )
    )

    # Second pass: greedy assignment with consumed tracking
    available_index = list(index)
    final_results: list[tuple[int, LinkResult]] = []

    for orig_idx, preliminary_result in scored:
        receipt = preliminary_result.receipt
        if not preliminary_result.matched:
            # No candidates at all
            final_results.append((orig_idx, preliminary_result))
            continue

        # Re-match against remaining available transactions
        result = match_receipt(
            receipt,
            available_index,
            amount_tolerance=amount_tolerance,
            date_window_days=date_window_days,
            description_threshold=description_threshold,
        )
        final_results.append((orig_idx, result))

        # Remove matched transaction from available pool
        if result.matched and result.transaction is not None:
            available_index = [
                t
                for t in available_index
                if not (
                    t.source_image == result.transaction.source_image
                    and t.row_index == result.transaction.row_index
                )
            ]

    # Restore original order
    final_results.sort(key=lambda x: x[0])
    return [r for _, r in final_results]
