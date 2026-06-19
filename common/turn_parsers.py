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


def _resolve_column_mapping(context: WorkflowState) -> dict[str, str | None]:
    """Resolve column_mapping from whichever upstream node produced it.

    Checks ``detect_headers`` (standalone bank workflow) first, then
    ``classify_document`` (unified workflow).  Raises ``KeyError`` with
    diagnostics if neither source has a column_mapping.
    """
    for source in ("detect_headers", "classify_document"):
        mapping = context.get(f"{source}.column_mapping", None)
        if isinstance(mapping, dict):
            return mapping
    sources_present = [k for k in ("detect_headers", "classify_document") if context.has(k)]
    msg = (
        f"No column_mapping found in state. "
        f"Checked: detect_headers, classify_document. "
        f"Nodes present: {sources_present or 'none'}"
    )
    raise KeyError(msg)


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

    Column-role keyword lists are read from ``column_roles`` in
    ``document_type_detection.yaml`` (single source of truth) — see
    ``PromptCatalog.get_column_roles``. Duplicates the parsing logic from
    ``unified_bank_extractor.ResponseParser.parse_headers``.
    """

    def __init__(self, column_roles: dict[str, list[str]] | None = None) -> None:
        # Lazy-loaded from YAML on first use; injectable for tests.
        self._column_roles = column_roles

    def _get_column_roles(self) -> dict[str, list[str]]:
        if self._column_roles is None:
            from common.prompt_catalog import PromptCatalog

            self._column_roles = PromptCatalog().get_column_roles()
        return self._column_roles

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
            if len(cleaned) > 20 and (cleaned.count(",") >= 2 or cleaned.count("|") >= 2):
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
        roles = self._get_column_roles()
        mapping: dict[str, str | None] = dict.fromkeys(roles)
        headers_lower = [h.lower() for h in headers]

        for col_type, keywords in roles.items():
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

    Reads column names from ``column_mapping`` in the accumulated workflow
    state.  Checks ``detect_headers`` (standalone bank workflow) then
    ``classify_document`` (unified workflow).  Returns ``rows``,
    ``row_count``, and per-column name metadata for downstream
    post-processing.
    """

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        from common.unified_bank_extractor import ResponseParser

        mapping = _resolve_column_mapping(context)
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
    Reads column names from ``column_mapping`` in the accumulated workflow
    state (``detect_headers`` or ``classify_document``).
    """

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        from common.unified_bank_extractor import ResponseParser

        mapping = _resolve_column_mapping(context)
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
    """Parse enriched classification response with evidence-based disambiguation.

    Parses structured COLUMNS / PAID / ROWS fields from the model response,
    derives document type from evidence rather than model guesswork, and
    builds a ``column_mapping`` dict for bank statements.

    Falls back to ``type_mappings`` / ``fallback_keywords`` from
    ``document_type_detection.yaml`` when structured parsing fails.
    ``fallback_type`` is read from ``run_config.yml`` classification section.
    """

    # Regexes for structured field extraction.
    # A response may arrive terse ("1. COLUMNS: a | b | c") or — when the model
    # ignores the format and emits chain-of-thought / markdown — with the label
    # as a heading ("### 1. COLUMNS") and the value on a LATER line inside a code
    # fence. The prefix therefore tolerates leading markdown / number / bullet
    # noise (#, >, -, *, digits), and _parse_enriched recovers the value from the
    # whole response when it is not present inline.
    # See prompts/document_type_detection.yaml for the format the prompt requests.
    _LABEL_PREFIX = r"^[#>\-*\d.\)\s]*"
    _COLUMNS_RE = re.compile(rf"{_LABEL_PREFIX}COLUMNS\**\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    _PAID_RE = re.compile(rf"{_LABEL_PREFIX}PAID\**\s*:\s*(YES|NO)\b", re.MULTILINE | re.IGNORECASE)
    _ROWS_RE = re.compile(rf"{_LABEL_PREFIX}ROWS\**\s*:\s*(\d+)", re.MULTILINE | re.IGNORECASE)
    # Bare COLUMNS label, even with no inline value (e.g. a "### 1. COLUMNS"
    # heading). Presence of the label marks the response as an enriched answer.
    _COLUMNS_LABEL_RE = re.compile(r"\bCOLUMNS\b", re.IGNORECASE)
    # Reasoning models (InternVL3.5 thinking mode) wrap their chain-of-thought
    # in <think>...</think>; a truncated response leaves an unterminated <think>
    # tail. Strip both before parsing so reasoning prose can never be mistaken
    # for the answer (e.g. "no table with headers like Date, Description, Debit"
    # must not yield a Debit column). See prompts/document_type_detection.yaml.
    _THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
    _STRAY_THINK_RE = re.compile(r"</?think>", re.IGNORECASE)
    # A structured, pipe-delimited header line (>= 2 pipes), e.g.
    # "Date | Description | Debit | Credit | Balance".
    _MIN_PIPES = 2

    def __init__(self, fallback_type: str = "UNIVERSAL") -> None:
        self._fallback_type = fallback_type
        self._header_parser = HeaderListParser()

    def parse(self, raw_response: str, context: WorkflowState) -> dict[str, Any]:
        result = self._parse_enriched(raw_response)
        if result is not None:
            result["_raw_classification"] = raw_response
            return result

        # Fallback: structured parsing failed, use legacy matching
        return self._parse_legacy(raw_response)

    @staticmethod
    def _clean_value(value: str) -> str:
        """Strip surrounding markdown emphasis / code-fence noise from a value."""
        return value.strip().strip("*`").strip()

    @classmethod
    def _strip_think(cls, raw_response: str) -> str:
        """Remove <think>...</think> blocks and any unterminated <think> tail."""
        text = cls._THINK_CLOSED_RE.sub("", raw_response)
        text = cls._THINK_OPEN_RE.sub("", text)
        text = cls._STRAY_THINK_RE.sub("", text)
        return text.strip()

    def _recover_pipe_headers(self, text: str) -> list[str]:
        """Recover headers from the first structured pipe-delimited line.

        Used only when the COLUMNS value is not inline (e.g. a markdown heading
        with the value on a later line / in a code fence). Restricted to lines
        with >= 2 pipes so prose sentences can never be harvested as headers —
        that prose harvesting previously promoted receipts to BANK_STATEMENT.
        """
        for line in text.splitlines():
            if line.count("|") >= self._MIN_PIPES:
                return self._header_parser._parse_headers(line.strip())
        return []

    def _parse_enriched(self, raw_response: str) -> dict[str, Any] | None:
        """Parse COLUMNS / PAID / ROWS fields and derive document type.

        Reasoning-model ``<think>`` blocks are stripped first. If the COLUMNS
        label is present but its value is not inline (a heading with the value
        on a later line / in a code fence), headers are recovered only from a
        structured pipe-delimited line — never from prose. The
        debit/credit/balance guard below then decides BANK_STATEMENT.
        """
        text = self._strip_think(raw_response)

        # An enriched response must at least mention the COLUMNS label; without
        # it, defer to legacy keyword matching.
        if self._COLUMNS_LABEL_RE.search(text) is None:
            return None

        columns_match = self._COLUMNS_RE.search(text)
        columns_inline = self._clean_value(columns_match.group(1)) if columns_match else ""
        paid_match = self._PAID_RE.search(text)
        rows_match = self._ROWS_RE.search(text)

        if columns_inline.upper() == "NONE":
            headers: list[str] = []
        elif columns_inline:
            headers = self._header_parser._parse_headers(columns_inline)
        else:
            # Label present but value not inline — recover only from a
            # structured pipe-delimited header line (no prose harvesting).
            headers = self._recover_pipe_headers(text)

        paid = paid_match.group(1).upper() == "YES" if paid_match else False
        row_count = int(rows_match.group(1)) if rows_match else 0

        # Derive document type from evidence using the YAML-driven rule set
        # (classification_evidence in document_type_detection.yaml). The rules
        # encode the precedence that used to be a hardcoded if/elif chain.
        column_mapping: dict[str, str | None] | None = None
        if headers:
            column_mapping = self._header_parser._match_columns(headers)

        from common.prompt_catalog import PromptCatalog

        evidence = PromptCatalog().get_classification_evidence()
        doc_type = _evaluate_classification(column_mapping, paid, evidence)
        if doc_type is None:
            # No rule matched and default is `none` -> defer to the legacy
            # keyword path (_parse_legacy).
            return None

        complexity = _compute_complexity(row_count)

        result: dict[str, Any] = {
            "DOCUMENT_TYPE": doc_type,
            "complexity": complexity,
            "row_count": row_count,
            "payment_evidence": paid,
        }
        if column_mapping is not None:
            result["column_mapping"] = column_mapping

        return result

    def _parse_legacy(self, raw_response: str) -> dict[str, Any]:
        """Fallback to type_mappings / fallback_keywords matching."""
        from common.prompt_catalog import PromptCatalog

        catalog = PromptCatalog()
        detection_config = catalog.get_detection_config()

        doc_type = _match_document_type(
            raw_response,
            detection_config.get("type_mappings", {}),
            detection_config.get("fallback_keywords", {}),
            self._fallback_type,
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


def _present_roles(column_mapping: dict[str, str | None] | None) -> set[str]:
    """Roles with a matched header in ``column_mapping`` (empty if None)."""
    if not column_mapping:
        return set()
    return {role for role, col in column_mapping.items() if col}


def _when_matches(when: dict[str, Any], present_roles: set[str], paid: bool) -> bool:
    """True if a rule's ``when:`` clause holds for the parsed evidence.

    Supported keys (all AND-ed when present): ``any_roles`` (at least one role
    present), ``all_roles`` (every role present), ``paid`` (matches the PAID
    flag).
    """
    if "any_roles" in when and not (set(when["any_roles"]) & present_roles):
        return False
    if "all_roles" in when and not set(when["all_roles"]).issubset(present_roles):
        return False
    return not ("paid" in when and bool(when["paid"]) != paid)


def _evaluate_classification(
    column_mapping: dict[str, str | None] | None,
    paid: bool,
    evidence: dict[str, Any],
) -> str | None:
    """Derive the document type from evidence using ``classification_evidence``.

    Walks ``rules`` top-down (first match wins). When no rule matches, returns
    the ``default`` — or ``None`` when ``default`` is ``"none"``, signalling the
    caller to defer to the legacy keyword path.
    """
    present = _present_roles(column_mapping)
    for rule in evidence["rules"]:
        if _when_matches(rule["when"], present, paid):
            return rule["type"]
    default = evidence["default"]
    if isinstance(default, str) and default.lower() == "none":
        return None
    return default


def _compute_complexity(row_count: int) -> str:
    """Derive complexity tier from row/item count.

    Thresholds: LOW (1-5), MEDIUM (6-20), HIGH (21+).
    """
    if row_count <= 5:
        return "LOW"
    if row_count <= 20:
        return "MEDIUM"
    return "HIGH"


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
            receipt_total = _parse_amount(receipts[i].get("TOTAL", "") if i < len(receipts) else "")
            tx_amount = _parse_amount(match_copy.get("TRANSACTION_AMOUNT", ""))

            if receipt_total is not None and tx_amount is not None:
                if abs(receipt_total) > 0 and (
                    abs(abs(tx_amount) - abs(receipt_total)) / abs(receipt_total) > 0.01
                ):
                    match_copy["MATCHED_TRANSACTION"] = "NOT_FOUND"
                    match_copy["REASONING"] = (
                        f"Amount gate: receipt {receipt_total} != transaction {tx_amount}"
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


def build_parser_registry(*, fallback_type: str = "UNIVERSAL") -> dict[str, TurnParser]:
    """Build the default parser registry.

    Args:
        fallback_type: Default document type when classification can't parse
            the model response. Reads from ``run_config.yml`` classification
            section at call sites that have ``AppConfig``.
    """
    return {
        "header_list": HeaderListParser(),
        "receipt_list": ReceiptListParser(),
        "transaction_match": TransactionMatchParser(),
        "field_value": FieldValueParser(),
        "classification": ClassificationParser(fallback_type=fallback_type),
        "balance_description": BalanceDescriptionParser(),
        "amount_description": AmountDescriptionParser(),
    }
