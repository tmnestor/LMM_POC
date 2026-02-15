"""Extract structured fields from raw OCR text.

Used by GLM-OCR processor which returns raw text instead of structured
FIELD_NAME: value pairs.  Applies regex patterns and heuristics to
parse OCR output into the same field dictionary format used by other models.

Also handles markdown table parsing for bank statement extraction
via GLM-OCR's "Table Recognition:" mode.
"""

import re

# ============================================================================
# Public API
# ============================================================================


def extract_fields_from_ocr(
    ocr_text: str,
    document_type: str,
    expected_fields: list[str],
) -> dict[str, str]:
    """Parse raw OCR text into structured extraction fields.

    Args:
        ocr_text: Raw OCR text from GLM-OCR "OCR:" prompt.
        document_type: Detected document type (e.g. "INVOICE", "RECEIPT").
        expected_fields: List of field names to extract.

    Returns:
        Dict mapping field names to extracted values (or "NOT_FOUND").
    """
    data: dict[str, str] = {f: "NOT_FOUND" for f in expected_fields}

    if not ocr_text or not ocr_text.strip():
        return data

    lines = ocr_text.strip().split("\n")

    # DOCUMENT_TYPE — use the already-detected type
    if "DOCUMENT_TYPE" in data:
        data["DOCUMENT_TYPE"] = document_type

    # BUSINESS_ABN
    if "BUSINESS_ABN" in data:
        abn = _extract_abn(ocr_text)
        if abn:
            data["BUSINESS_ABN"] = abn

    # INVOICE_DATE
    if "INVOICE_DATE" in data:
        date = _extract_primary_date(ocr_text)
        if date:
            data["INVOICE_DATE"] = date

    # Monetary amounts
    if "GST_AMOUNT" in data:
        gst = _extract_gst_amount(ocr_text)
        if gst:
            data["GST_AMOUNT"] = gst

    if "TOTAL_AMOUNT" in data:
        total = _extract_total_amount(ocr_text)
        if total:
            data["TOTAL_AMOUNT"] = total

    if "IS_GST_INCLUDED" in data:
        has_gst = bool(
            re.search(r"\bgst\b|goods\s+and\s+services\s+tax", ocr_text, re.IGNORECASE)
        )
        data["IS_GST_INCLUDED"] = "true" if has_gst else "false"

    # Names
    if "SUPPLIER_NAME" in data:
        name = _extract_supplier_name(lines)
        if name:
            data["SUPPLIER_NAME"] = name

    if "PAYER_NAME" in data:
        name = _extract_payer_name(ocr_text)
        if name:
            data["PAYER_NAME"] = name

    if "PASSENGER_NAME" in data:
        name = _extract_payer_name(ocr_text)
        if name:
            data["PASSENGER_NAME"] = name

    # Addresses
    if "BUSINESS_ADDRESS" in data:
        addr = _extract_address(ocr_text)
        if addr:
            data["BUSINESS_ADDRESS"] = addr

    if "PAYER_ADDRESS" in data:
        addr = _extract_payer_address(ocr_text)
        if addr:
            data["PAYER_ADDRESS"] = addr

    # Line items
    items = _extract_line_items(lines)
    if items:
        if "LINE_ITEM_DESCRIPTIONS" in data:
            data["LINE_ITEM_DESCRIPTIONS"] = " | ".join(i["description"] for i in items)
        if "LINE_ITEM_QUANTITIES" in data:
            data["LINE_ITEM_QUANTITIES"] = " | ".join(
                i.get("quantity", "1") for i in items
            )
        if "LINE_ITEM_PRICES" in data:
            data["LINE_ITEM_PRICES"] = " | ".join(
                i.get("price", "NOT_FOUND") for i in items
            )
        if "LINE_ITEM_TOTAL_PRICES" in data:
            data["LINE_ITEM_TOTAL_PRICES"] = " | ".join(
                i.get("total", "NOT_FOUND") for i in items
            )

    # Travel-specific
    if "TRAVEL_MODE" in data:
        mode = _extract_travel_mode(ocr_text)
        if mode:
            data["TRAVEL_MODE"] = mode

    return data


def extract_bank_fields_from_table(
    table_text: str,
    expected_fields: list[str],
) -> dict[str, str]:
    """Parse markdown table from GLM-OCR "Table Recognition:" into bank fields.

    Args:
        table_text: Markdown table text (or raw OCR as fallback).
        expected_fields: Expected bank statement field names.

    Returns:
        Dict mapping field names to extracted values.
    """
    data: dict[str, str] = {f: "NOT_FOUND" for f in expected_fields}
    data["DOCUMENT_TYPE"] = "BANK_STATEMENT"

    rows = _parse_markdown_table(table_text)
    if not rows:
        # Fallback: try to extract from raw OCR text
        return _extract_bank_from_ocr(table_text, expected_fields)

    # Identify columns by header keywords
    headers = rows[0] if rows else {}
    date_col = _find_column(
        headers, ["date", "value date", "transaction date", "effective date"]
    )
    desc_col = _find_column(
        headers,
        ["description", "details", "transaction", "particulars", "narration"],
    )
    debit_col = _find_column(headers, ["debit", "withdrawal", "dr", "money out"])
    credit_col = _find_column(headers, ["credit", "deposit", "cr", "money in"])
    balance_col = _find_column(headers, ["balance", "running balance", "closing"])
    # Fallback: single "amount" column when no explicit debit column
    amount_col = _find_column(headers, ["amount"]) if not debit_col else None

    dates: list[str] = []
    descriptions: list[str] = []
    amounts_paid: list[str] = []

    for row in rows[1:]:  # Skip header row
        date_val = row.get(date_col, "").strip() if date_col else ""
        desc_val = row.get(desc_col, "").strip() if desc_col else ""
        debit_val = row.get(debit_col, "").strip() if debit_col else ""
        amount_val = row.get(amount_col, "").strip() if amount_col else ""

        # Skip empty rows
        if not desc_val and not debit_val and not amount_val:
            continue

        # Determine if this is a debit transaction
        is_debit = False
        raw_amount = ""

        if debit_col and debit_val and debit_val not in ("", "-", "\u2014", " "):
            is_debit = True
            raw_amount = debit_val
        elif amount_col and amount_val:
            # Single amount column: negatives are debits
            if _is_debit_amount(amount_val):
                is_debit = True
                raw_amount = amount_val

        if is_debit:
            amount = _normalize_amount(raw_amount)
            if amount:
                dates.append(_normalize_date_str(date_val) if date_val else "NOT_FOUND")
                descriptions.append(desc_val or "NOT_FOUND")
                amounts_paid.append(amount)

    if dates:
        data["TRANSACTION_DATES"] = " | ".join(dates)
        data["LINE_ITEM_DESCRIPTIONS"] = " | ".join(descriptions)
        data["TRANSACTION_AMOUNTS_PAID"] = " | ".join(amounts_paid)

        # Statement date range
        valid_dates = [d for d in dates if d != "NOT_FOUND"]
        if len(valid_dates) >= 2:
            data["STATEMENT_DATE_RANGE"] = f"{valid_dates[0]} - {valid_dates[-1]}"
        elif len(valid_dates) == 1:
            data["STATEMENT_DATE_RANGE"] = valid_dates[0]

    return data


# ============================================================================
# ABN extraction
# ============================================================================

_ABN_PATTERN = re.compile(r"(?:ABN|A\.B\.N\.?)[\s:]*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})")
_ABN_BARE = re.compile(r"\b(\d{2}\s\d{3}\s\d{3}\s\d{3})\b")


def _extract_abn(text: str) -> str | None:
    """Extract ABN (11-digit Australian Business Number)."""
    m = _ABN_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    m = _ABN_BARE.search(text)
    if m:
        return m.group(1).strip()
    return None


# ============================================================================
# Date extraction
# ============================================================================

_DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"),
    # DD Month YYYY
    re.compile(
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Month DD, YYYY
    re.compile(
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # YYYY-MM-DD
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
]

_DATE_LABEL = re.compile(
    r"(?:date|dated|invoice\s+date|tax\s+date|receipt\s+date)[\s:]*",
    re.IGNORECASE,
)


def _extract_primary_date(text: str) -> str | None:
    """Extract the primary document date (preferring labeled dates)."""
    # Try: date near a label like "Date:" or "Invoice Date:"
    for line in text.split("\n"):
        if _DATE_LABEL.search(line):
            for pat in _DATE_PATTERNS:
                m = pat.search(line)
                if m:
                    return _normalize_date_str(m.group(1))
    # Fallback: first date found in text
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            return _normalize_date_str(m.group(1))
    return None


# ============================================================================
# Amount extraction
# ============================================================================

_AMOUNT_PATTERN = re.compile(r"\$([\d,]+\.?\d*)")


def _extract_total_amount(text: str) -> str | None:
    """Extract total amount (near 'total' keyword, or largest amount)."""
    # Look for amount near "total" keyword
    for line in text.split("\n"):
        if re.search(r"\btotal\b", line, re.IGNORECASE):
            m = _AMOUNT_PATTERN.search(line)
            if m:
                return f"${m.group(1)}"

    # Fallback: largest dollar amount in text
    all_amounts = _AMOUNT_PATTERN.findall(text)
    if all_amounts:
        parsed: list[tuple[float, str]] = []
        for a in all_amounts:
            try:
                parsed.append((float(a.replace(",", "")), f"${a}"))
            except ValueError:
                continue
        if parsed:
            parsed.sort(key=lambda x: x[0], reverse=True)
            return parsed[0][1]
    return None


def _extract_gst_amount(text: str) -> str | None:
    """Extract GST amount."""
    for line in text.split("\n"):
        if re.search(r"\bgst\b", line, re.IGNORECASE):
            m = _AMOUNT_PATTERN.search(line)
            if m:
                return f"${m.group(1)}"
    return None


# ============================================================================
# Name / address extraction
# ============================================================================


def _extract_supplier_name(lines: list[str]) -> str | None:
    """Extract supplier name (first substantial text line at top of document)."""
    skip_words = {
        "receipt",
        "invoice",
        "tax invoice",
        "statement",
        "bank statement",
        "abn",
        "date",
        "total",
        "subtotal",
        "gst",
        "payment",
        "table",
    }

    for line in lines[:10]:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        lower = stripped.lower()
        if lower in skip_words:
            continue
        # Skip dates and amounts
        if re.match(r"^[\d/$.\-]", stripped):
            continue
        # Skip ABN lines
        if lower.startswith("abn"):
            continue
        # Skip lines that are all digits/symbols
        if re.match(r"^[\d\s\-/.$%]+$", stripped):
            continue
        return stripped
    return None


def _extract_payer_name(text: str) -> str | None:
    """Extract payer/customer name from OCR text."""
    patterns = [
        re.compile(
            r"(?:bill\s+to|customer|client|payer|sold\s+to|ship\s+to)[\s:]*(.+)",
            re.IGNORECASE,
        ),
        re.compile(r"(?:passenger|traveller|name)[\s:]*(.+)", re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            name = m.group(1).strip().split("\n")[0].strip()
            if name and len(name) > 1:
                return name
    return None


_ADDR_PATTERN = re.compile(
    r"(\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|"
    r"Court|Ct|Place|Pl|Way|Boulevard|Blvd|Parade|Pde|"
    r"Highway|Hwy|Crescent|Cres|Terrace|Tce)"
    r"[^,\n]*)",
    re.IGNORECASE,
)

_STATE_POSTCODE = re.compile(r"(?:VIC|NSW|QLD|SA|WA|TAS|NT|ACT)\s+\d{4}", re.IGNORECASE)


def _extract_address(text: str) -> str | None:
    """Extract business address (street address + optional state/postcode)."""
    m = _ADDR_PATTERN.search(text)
    if m:
        addr = m.group(1).strip()
        # Try to append state/postcode from nearby text
        remaining = text[m.end() : m.end() + 100]
        state_m = _STATE_POSTCODE.search(remaining)
        if state_m:
            addr += " " + remaining[: state_m.end()].strip().lstrip(",").strip()
        return addr
    return None


def _extract_payer_address(text: str) -> str | None:
    """Extract payer address (after payer-related keywords)."""
    # Find text after "Bill To" / "Ship To" section, then look for address
    payer_section = re.search(
        r"(?:bill\s+to|ship\s+to|customer|client)[\s:]*\n?(.{0,300})",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if payer_section:
        section = payer_section.group(1)
        return _extract_address(section)
    return None


# ============================================================================
# Line item extraction
# ============================================================================


def _extract_line_items(lines: list[str]) -> list[dict[str, str]]:
    """Extract line items from OCR text.

    Looks for lines with description + dollar amount patterns.
    """
    items: list[dict[str, str]] = []

    # Pattern: text followed by dollar amount at end of line
    item_pattern = re.compile(
        r"^(?:\d+[.)\s]+)?"  # optional number prefix
        r"(.+?)\s+"  # description
        r"\$?([\d,]+\.\d{2})\s*$"  # amount at end
    )

    skip_labels = {
        "total",
        "subtotal",
        "gst",
        "tax",
        "balance",
        "amount",
        "payment",
        "change",
        "discount",
        "credit",
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        m = item_pattern.match(stripped)
        if m:
            desc = m.group(1).strip()
            amount = m.group(2).strip()

            if desc.lower() in skip_labels:
                continue
            if len(desc) < 3:
                continue

            items.append(
                {
                    "description": desc,
                    "quantity": "1",
                    "price": f"${amount}",
                    "total": f"${amount}",
                }
            )

    return items


# ============================================================================
# Travel-specific extraction
# ============================================================================


def _extract_travel_mode(text: str) -> str | None:
    """Extract travel mode from document text."""
    mode_keywords = {
        "air": ["flight", "airline", "boarding pass", "air"],
        "rail": ["train", "rail", "railway"],
        "bus": ["bus", "coach"],
        "taxi": ["taxi", "uber", "rideshare", "lyft"],
        "ferry": ["ferry", "boat"],
    }
    text_lower = text.lower()
    for mode, keywords in mode_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return mode
    return None


# ============================================================================
# Markdown table parsing (for bank statements)
# ============================================================================


def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    """Parse a markdown table into list of row dicts.

    Expected format:
        | Header1 | Header2 | Header3 |
        |---------|---------|---------|
        | value1  | value2  | value3  |

    Returns:
        List of dicts where first element maps headers to themselves
        (for column identification), followed by data row dicts.
        Empty list if no valid table found.
    """
    table_lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    # Find header row (first line with pipes and non-separator content)
    header_idx = None
    for i, line in enumerate(table_lines):
        if "|" in line and not re.match(r"^[\s|:\-]+$", line):
            header_idx = i
            break

    if header_idx is None:
        return []

    # Parse header
    raw_header = table_lines[header_idx].split("|")
    headers = [h.strip() for h in raw_header if h.strip()]

    if not headers:
        return []

    # First element: header → header mapping for column identification
    rows: list[dict[str, str]] = [{h: h for h in headers}]

    # Parse data rows
    for line in table_lines[header_idx + 1 :]:
        # Skip separator lines (e.g. |---|---|---|)
        if re.match(r"^[\s|:\-]+$", line):
            continue

        raw_cells = line.split("|")
        # Strip empty first/last cells from leading/trailing pipes
        if raw_cells and raw_cells[0].strip() == "":
            raw_cells = raw_cells[1:]
        if raw_cells and raw_cells[-1].strip() == "":
            raw_cells = raw_cells[:-1]

        cells = [c.strip() for c in raw_cells]
        if cells:
            row: dict[str, str] = {}
            for j, header in enumerate(headers):
                row[header] = cells[j] if j < len(cells) else ""
            rows.append(row)

    return rows


def _find_column(headers: dict[str, str], keywords: list[str]) -> str | None:
    """Find a column name that contains any of the given keywords."""
    for col_name in headers:
        col_lower = col_name.lower().strip()
        for kw in keywords:
            if kw in col_lower:
                return col_name
    return None


# ============================================================================
# Bank statement fallback (raw OCR → transactions)
# ============================================================================


def _extract_bank_from_ocr(ocr_text: str, expected_fields: list[str]) -> dict[str, str]:
    """Fallback: extract bank fields from raw OCR text (no table found)."""
    data: dict[str, str] = {f: "NOT_FOUND" for f in expected_fields}
    data["DOCUMENT_TYPE"] = "BANK_STATEMENT"

    # Try to find date-description-amount patterns per line
    date_amount_pattern = re.compile(
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+"  # date
        r"(.+?)\s+"  # description
        r"\$?([\d,]+\.\d{2})"  # amount
    )

    dates: list[str] = []
    descriptions: list[str] = []
    amounts: list[str] = []

    for line in ocr_text.strip().split("\n"):
        m = date_amount_pattern.search(line)
        if m:
            dates.append(_normalize_date_str(m.group(1)))
            descriptions.append(m.group(2).strip())
            amounts.append(f"${m.group(3)}")

    if dates:
        data["TRANSACTION_DATES"] = " | ".join(dates)
        data["LINE_ITEM_DESCRIPTIONS"] = " | ".join(descriptions)
        data["TRANSACTION_AMOUNTS_PAID"] = " | ".join(amounts)

        valid_dates = [d for d in dates if d != "NOT_FOUND"]
        if len(valid_dates) >= 2:
            data["STATEMENT_DATE_RANGE"] = f"{valid_dates[0]} - {valid_dates[-1]}"

    return data


# ============================================================================
# Shared utilities
# ============================================================================


def _normalize_amount(text: str) -> str | None:
    """Normalize amount string to $XX.XX format."""
    if not text:
        return None
    clean = text.strip().lstrip("-").strip("()").strip()
    clean = re.sub(r"[$\s]", "", clean)
    clean = clean.replace(",", "")
    try:
        val = float(clean)
        if val > 0:
            return f"${val:.2f}"
    except ValueError:
        pass
    return None


def _is_debit_amount(text: str) -> bool:
    """Check if an amount represents a debit (negative/withdrawal)."""
    text = text.strip()
    return text.startswith("-") or text.startswith("(") or "DR" in text.upper()


def _normalize_date_str(date_str: str) -> str:
    """Normalize date string to DD/MM/YYYY."""
    if not date_str:
        return "NOT_FOUND"
    try:
        from dateutil import parser as date_parser

        parsed = date_parser.parse(date_str, dayfirst=True)
        return parsed.strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return date_str
