"""Turn parsers for the agentic extraction engine.

Each parser implements the TurnParser protocol: ``parse(raw_response, context) -> dict``.
Parsers are standalone classes that duplicate algorithms from the notebook
prototype and unified_bank_extractor.py -- no existing code is modified.
"""

import re
from typing import Any, Protocol

from common.extraction_types import WorkflowState


class TurnParser(Protocol):
    """Protocol for parsing a single model turn's output."""

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]: ...


class ParseError(Exception):
    """Raised when a parser cannot extract structured data."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RECEIPT_HEADER_RE = re.compile(r"(?:---\s*)?RECEIPT\s+(\d+)(?:\s*---)?", re.IGNORECASE)


def _strip_bullet(line: str) -> str:
    """Remove numbering, bullets, and markdown formatting from a line."""
    cleaned = line.lstrip("0123456789.-\u2022* ").strip()
    cleaned = cleaned.replace("**", "").replace("__", "")
    return cleaned


def _parse_amount(s: str) -> float | None:
    """Parse a currency string to float, returning None on failure."""
    if not s or s.upper() == "NOT_FOUND":
        return None
    cleaned = re.sub(r"[$,\s]", "", s.strip())
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Concrete Parsers
# ---------------------------------------------------------------------------


class HeaderListParser:
    """Parse numbered/comma/pipe column headers, then map to semantic roles.

    Duplicates logic from ``unified_bank_extractor.ResponseParser.parse_headers``
    and ``ColumnMatcher.match``.
    """

    COLUMN_PATTERNS: dict[str, list[str]] = {
        "date": [
            "date",
            "trans date",
            "transaction date",
            "value date",
            "posting date",
        ],
        "description": [
            "description",
            "details",
            "transaction",
            "particulars",
            "narrative",
            "transaction details",
            "reference",
        ],
        "debit": ["debit", "withdrawal", "withdrawals", "dr", "money out"],
        "credit": ["credit", "deposit", "deposits", "cr", "money in"],
        "balance": ["balance", "running balance", "closing balance"],
        "amount": ["amount", "transaction amount"],
    }

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        """Parse column headers and produce column_mapping."""
        headers = self._parse_headers(raw_response)
        if not headers:
            msg = f"No column headers found in response: {raw_response[:200]}"
            raise ParseError(msg)
        column_mapping = self._match_columns(headers)
        return {
            "headers": headers,
            "column_mapping": column_mapping,
        }

    def _parse_headers(self, response: str) -> list[str]:
        """Extract header strings from various formats."""
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        headers: list[str] = []

        for line in lines:
            cleaned = _strip_bullet(line)
            if cleaned.endswith(":"):
                continue

            # Multi-value line: commas or pipes
            if len(cleaned) > 20 and (
                cleaned.count(",") >= 2 or cleaned.count("|") >= 2
            ):
                sep = "|" if cleaned.count("|") >= 2 else ","
                for part in cleaned.split(sep):
                    part = part.strip()
                    if part and len(part) > 1:
                        headers.append(part)
                continue

            if len(cleaned) > 40:
                continue
            if cleaned and len(cleaned) > 2:
                headers.append(cleaned)

        return headers

    def _match_columns(self, headers: list[str]) -> dict[str, str | None]:
        """Map detected headers to semantic column types."""
        mapping: dict[str, str | None] = {
            "date": None,
            "description": None,
            "debit": None,
            "credit": None,
            "balance": None,
            "amount": None,
        }
        headers_lower = [h.lower() for h in headers]

        for col_type, keywords in self.COLUMN_PATTERNS.items():
            matched = self._find_match(headers, headers_lower, keywords)
            if matched:
                mapping[col_type] = matched

        return mapping

    @staticmethod
    def _find_match(
        headers: list[str],
        headers_lower: list[str],
        keywords: list[str],
    ) -> str | None:
        """Find matching header using keywords."""
        for keyword in keywords:
            for i, header_lower in enumerate(headers_lower):
                if keyword == header_lower:
                    return headers[i]
        for keyword in keywords:
            if len(keyword) > 2:
                for i, header_lower in enumerate(headers_lower):
                    if len(header_lower) <= 20 and keyword in header_lower:
                        return headers[i]
        return None


class ReceiptListParser:
    """Parse ``--- RECEIPT N ---`` blocks into receipt list + formatted_text.

    Duplicates logic from the transaction-linking notebook's
    ``parse_stage1_response()``.
    """

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        """Parse receipt blocks, return receipts list and formatted_text."""
        receipts = self._parse_blocks(raw_response)
        if not receipts:
            msg = f"No receipt blocks found in response: {raw_response[:200]}"
            raise ParseError(msg)

        formatted_parts: list[str] = []
        for i, r in enumerate(receipts, 1):
            store = r.get("STORE", "UNKNOWN")
            date = r.get("DATE", "UNKNOWN")
            total = r.get("TOTAL", "UNKNOWN")
            formatted_parts.append(f"Purchase {i}: {store}, date {date}, total {total}")
        formatted_text = "\n".join(formatted_parts)

        return {
            "receipts": receipts,
            "formatted_text": formatted_text,
            "receipt_count": len(receipts),
        }

    def _parse_blocks(self, response: str) -> list[dict[str, str]]:
        """Split response into receipt blocks and parse each."""
        blocks: list[dict[str, str]] = []
        current_block: dict[str, str] = {}
        in_block = False

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            if RECEIPT_HEADER_RE.match(line):
                if in_block and current_block:
                    blocks.append(current_block)
                current_block = {}
                in_block = True
                continue

            if in_block and ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().upper()
                value = value.strip()
                if key in ("STORE", "DATE", "TOTAL"):
                    current_block[key] = value

        if in_block and current_block:
            blocks.append(current_block)

        return blocks


class TransactionMatchParser:
    """Parse match response blocks into per-receipt match results.

    Duplicates logic from the transaction-linking notebook's match parsing.
    """

    MATCH_FIELDS = frozenset(
        {
            "MATCHED_TRANSACTION",
            "TRANSACTION_DATE",
            "TRANSACTION_AMOUNT",
            "TRANSACTION_DESCRIPTION",
            "RECEIPT_STORE",
            "RECEIPT_TOTAL",
            "AMOUNT_CHECK",
            "NAME_CHECK",
            "CONFIDENCE",
            "MISMATCH_TYPE",
            "REASONING",
        }
    )

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        """Parse match blocks, return matches list."""
        matches = self._parse_blocks(raw_response)
        if not matches:
            msg = f"No match blocks found in response: {raw_response[:200]}"
            raise ParseError(msg)
        return {"matches": matches, "match_count": len(matches)}

    def _parse_blocks(self, response: str) -> list[dict[str, str]]:
        """Split response into receipt match blocks."""
        blocks: list[dict[str, str]] = []
        current_block: dict[str, str] = {}
        in_block = False

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            if RECEIPT_HEADER_RE.match(line):
                if in_block and current_block:
                    blocks.append(current_block)
                current_block = {}
                in_block = True
                continue

            if in_block and ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().upper()
                value = value.strip()
                if key in self.MATCH_FIELDS:
                    current_block[key] = value

        if in_block and current_block:
            blocks.append(current_block)

        return blocks


class BalanceDescriptionParser:
    """Parse balance-description Turn 1 via ``ResponseParser.parse_balance_description()``.

    Reads column names from ``detect_headers.column_mapping`` in the
    accumulated workflow state.  Returns ``rows``, ``row_count``, and
    per-column name metadata for downstream post-processing.
    """

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        from common.unified_bank_extractor import ResponseParser

        mapping = context.get("detect_headers.column_mapping")
        date_col = mapping.get("date") or "Date"
        desc_col = mapping.get("description") or "Description"
        debit_col = mapping.get("debit") or "Debit"
        credit_col = mapping.get("credit") or "Credit"
        balance_col = mapping.get("balance") or "Balance"

        rows = ResponseParser.parse_balance_description(
            raw_response,
            date_col=date_col,
            desc_col=desc_col,
            debit_col=debit_col,
            credit_col=credit_col,
            balance_col=balance_col,
        )

        if not rows:
            msg = f"No transaction rows parsed from response: {raw_response[:200]}"
            raise ParseError(msg)

        return {
            "rows": rows,
            "row_count": len(rows),
            "date_col": date_col,
            "desc_col": desc_col,
            "debit_col": debit_col,
            "credit_col": credit_col,
            "balance_col": balance_col,
        }


class AmountDescriptionParser:
    """Parse amount-description Turn 1 via ``ResponseParser.parse_amount_description()``.

    For statements with a signed Amount column (negative = withdrawal).
    Reads column names from ``detect_headers.column_mapping``.
    """

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        from common.unified_bank_extractor import ResponseParser

        mapping = context.get("detect_headers.column_mapping")
        date_col = mapping.get("date") or "Date"
        desc_col = mapping.get("description") or "Description"
        amount_col = mapping.get("amount") or "Amount"
        balance_col = mapping.get("balance")  # May be None

        rows = ResponseParser.parse_amount_description(
            raw_response,
            date_col=date_col,
            desc_col=desc_col,
            amount_col=amount_col,
            balance_col=balance_col,
        )

        if not rows:
            msg = f"No transaction rows parsed from response: {raw_response[:200]}"
            raise ParseError(msg)

        return {
            "rows": rows,
            "row_count": len(rows),
            "date_col": date_col,
            "desc_col": desc_col,
            "amount_col": amount_col,
            "balance_col": balance_col,
        }


class ClassificationParser:
    """Parse document classification response using detection YAML config.

    Reuses the ``type_mappings`` and ``fallback_keywords`` from
    ``document_type_detection.yaml`` via ``PromptCatalog``.
    """

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        from common.prompt_catalog import PromptCatalog

        catalog = PromptCatalog()
        detection_config = catalog.get_detection_config()

        doc_type = _match_document_type(
            raw_response,
            detection_config.get("type_mappings", {}),
            detection_config.get("fallback_keywords", {}),
            detection_config.get("settings", {}).get("fallback_type", "RECEIPT"),
        )

        return {"DOCUMENT_TYPE": doc_type, "_raw_classification": raw_response}


def _match_document_type(
    response: str,
    type_mappings: dict[str, str],
    fallback_keywords: dict[str, list[str]],
    fallback_type: str,
) -> str:
    """Match a model response to a canonical document type.

    Standalone copy of the matching logic from
    ``DocumentOrchestrator._parse_document_type_response()``.
    """
    cleaned = response.strip().lower()

    # 1. Direct mapping (case-insensitive substring)
    for variant, canonical in type_mappings.items():
        if variant.lower() in cleaned:
            return canonical

    # 2. Fallback keyword matching (first match wins)
    for doc_type, keywords in fallback_keywords.items():
        for keyword in keywords:
            if keyword.lower() in cleaned:
                return doc_type

    # 3. Ultimate fallback
    return fallback_type


class FieldValueParser:
    """Parse ``FIELD: value`` lines (standard document extraction)."""

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        """Parse FIELD: value lines into a flat dict."""
        fields: dict[str, str] = {}
        for line in raw_response.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if key and value:
                fields[key] = value

        if not fields:
            msg = f"No FIELD: value pairs found in response: {raw_response[:200]}"
            raise ParseError(msg)
        return fields


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def enforce_amount_gate(
    receipts: list[dict[str, str]],
    matches: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Validate that receipt totals match transaction amounts.

    For each match marked FOUND, compare the numeric amounts.
    Override to NOT_FOUND if they differ by more than 1%.
    """
    validated: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        match_copy = dict(match)
        if match_copy.get("MATCHED_TRANSACTION") == "FOUND":
            receipt_total = _parse_amount(
                receipts[i].get("TOTAL", "") if i < len(receipts) else ""
            )
            tx_amount = _parse_amount(match_copy.get("TRANSACTION_AMOUNT", ""))

            if receipt_total is not None and tx_amount is not None:
                if abs(receipt_total) > 0 and (
                    abs(abs(tx_amount) - abs(receipt_total)) / abs(receipt_total) > 0.01
                ):
                    match_copy["MATCHED_TRANSACTION"] = "NOT_FOUND"
                    match_copy["REASONING"] = (
                        f"Amount gate: receipt {receipt_total} "
                        f"!= transaction {tx_amount}"
                    )
        validated.append(match_copy)
    return validated


def dedup_by_field(
    records: list[dict[str, str]],
    field_name: str,
) -> list[dict[str, str]]:
    """Deduplicate records by a specified field, keeping first occurrence."""
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for record in records:
        key = record.get(field_name, "")
        if key and key not in seen:
            seen.add(key)
            result.append(record)
        elif not key:
            result.append(record)
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_parser_registry() -> dict[str, TurnParser]:
    """Build the default parser registry."""
    return {
        "header_list": HeaderListParser(),
        "receipt_list": ReceiptListParser(),
        "transaction_match": TransactionMatchParser(),
        "field_value": FieldValueParser(),
        "classification": ClassificationParser(),
        "balance_description": BalanceDescriptionParser(),
        "amount_description": AmountDescriptionParser(),
    }
