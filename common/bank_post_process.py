"""Bank statement post-processing for the graph-based extraction workflow.

Validator logic: balance correction, debit filtering, date range computation,
and final field assembly.  Reuses ``BalanceCorrector`` and ``TransactionFilter``
from ``bank_corrector.py`` -- no duplication of correction arithmetic.

Only ``_compute_date_range`` is duplicated (~40 lines) from
``UnifiedBankExtractor`` to avoid importing the full class.
"""

import logging
import re
from datetime import datetime
from typing import Any

from common.extraction_types import WorkflowState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Date range helper (standalone copy from UnifiedBankExtractor)
# ------------------------------------------------------------------

_DAY_PREFIX_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+", re.IGNORECASE)

_DATE_FORMATS = [
    "%d/%m/%Y",  # 03/05/2025
    "%d %b %Y",  # 04 Sep 2025
    "%d %B %Y",  # 04 September 2025
    "%Y-%m-%d",  # 2025-09-04
    "%m/%d/%Y",  # 05/03/2025 (US format)
]


def _compute_date_range(dates: list[str]) -> str:
    """Compute date range string, always oldest - newest."""
    if len(dates) < 2:
        return dates[0] if dates else "NOT_FOUND"

    first_str = _DAY_PREFIX_RE.sub("", dates[0].strip())
    last_str = _DAY_PREFIX_RE.sub("", dates[-1].strip())

    first_date = None
    last_date = None

    for fmt in _DATE_FORMATS:
        if first_date is None:
            try:
                first_date = datetime.strptime(first_str, fmt)
            except ValueError:
                pass
        if last_date is None:
            try:
                last_date = datetime.strptime(last_str, fmt)
            except ValueError:
                pass

    if first_date is None or last_date is None:
        return f"{first_str} - {last_str}"

    if first_date <= last_date:
        return f"{first_str} - {last_str}"
    return f"{last_str} - {first_str}"


# ------------------------------------------------------------------
# Amount formatting helper (from UnifiedBankExtractor._format_debit_amount)
# ------------------------------------------------------------------


def _format_debit_amount(amount_str: str) -> str:
    """Format debit amount, preserving negative sign for Amount strategy."""
    if not amount_str:
        return ""
    amount = amount_str.strip()

    is_negative = False
    if amount.startswith("-"):
        is_negative = True
        amount = amount[1:]
    elif amount.startswith("(") and amount.endswith(")"):
        is_negative = True
        amount = amount[1:-1]

    if amount and not amount.startswith("$"):
        amount = "$" + amount

    if is_negative:
        amount = "-" + amount

    return amount


# ------------------------------------------------------------------
# Array alignment helpers
# ------------------------------------------------------------------


def _align_balance_arrays(
    debit_rows: list[dict[str, str]],
    date_col: str,
    desc_col: str,
    debit_col: str,
    balance_col: str | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Extract aligned arrays from debit-filtered rows (balance/debit_credit strategies)."""
    dates: list[str] = []
    descriptions: list[str] = []
    amounts: list[str] = []
    balances: list[str] = []

    for r in debit_rows:
        date_val = r.get(date_col, "")
        desc_val = r.get(desc_col, "")
        debit_val = r.get(debit_col, "")

        if date_val and desc_val and debit_val:
            dates.append(date_val)
            descriptions.append(desc_val)
            amounts.append(debit_val)
            bal = r.get(balance_col, "") if balance_col else ""
            balances.append(bal if bal else "NOT_FOUND")

    return dates, descriptions, amounts, balances


def _align_amount_arrays(
    debit_rows: list[dict[str, str]],
    date_col: str,
    desc_col: str,
    amount_col: str,
    balance_col: str | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Extract aligned arrays from negative-amount-filtered rows."""
    dates: list[str] = []
    descriptions: list[str] = []
    amounts: list[str] = []
    balances: list[str] = []

    for r in debit_rows:
        date_val = r.get(date_col, "")
        desc_val = r.get(desc_col, "")
        amount_val = r.get(amount_col, "")

        if date_val and desc_val and amount_val:
            dates.append(date_val)
            descriptions.append(desc_val)
            amounts.append(_format_debit_amount(amount_val))
            bal = r.get(balance_col, "") if balance_col else ""
            balances.append(bal if bal else "NOT_FOUND")

    return dates, descriptions, amounts, balances


# ------------------------------------------------------------------
# Main entry point (called by GraphExecutor._run_validator)
# ------------------------------------------------------------------


def run_bank_post_process(
    state: WorkflowState,
) -> tuple[bool, dict[str, Any]]:
    """Post-process bank extraction results into pipe-delimited schema fields.

    Detects which extraction node ran, applies balance correction (if
    applicable), filters to debit transactions, computes date range, and
    builds the final ``FIELD: value`` dict.

    Always returns ``(True, fields_dict)`` -- the validator never fails,
    it only transforms data.
    """
    from common.bank_corrector import BalanceCorrector, TransactionFilter

    # Detect which extraction node ran
    strategy = None
    parsed: dict[str, Any] | None = None
    for node_key in ("extract_balance", "extract_debit_credit", "extract_amount"):
        if state.has(node_key):
            strategy = node_key
            parsed = state.node_results[node_key].parsed
            break

    if strategy is None or parsed is None:
        logger.warning("bank_post_process: no extraction node found in state")
        return True, {"DOCUMENT_TYPE": "BANK_STATEMENT"}

    rows: list[dict[str, str]] = parsed["rows"]
    date_col: str = parsed["date_col"]
    desc_col: str = parsed["desc_col"]

    dates: list[str]
    descriptions: list[str]
    amounts: list[str]
    balances: list[str]

    if strategy == "extract_balance":
        debit_col: str = parsed["debit_col"]
        credit_col: str = parsed["credit_col"]
        balance_col: str = parsed["balance_col"]

        # Sort to chronological order if needed
        is_chrono, _ = BalanceCorrector.is_chronological_order(rows, date_col)
        sorted_rows = (
            rows if is_chrono else BalanceCorrector.sort_by_date(rows, date_col)
        )

        # Apply balance correction
        corrector = BalanceCorrector()
        corrected_rows, stats = corrector.correct_transactions(
            sorted_rows,
            balance_col=balance_col,
            debit_col=debit_col,
            credit_col=credit_col,
            desc_col=desc_col,
        )
        logger.debug("Balance correction: %s", stats)

        # Filter debits
        debit_rows = TransactionFilter.filter_debits(
            corrected_rows, debit_col=debit_col, desc_col=desc_col
        )

        dates, descriptions, amounts, balances = _align_balance_arrays(
            debit_rows, date_col, desc_col, debit_col, balance_col
        )

    elif strategy == "extract_debit_credit":
        debit_col = parsed["debit_col"]

        debit_rows = TransactionFilter.filter_debits(
            rows, debit_col=debit_col, desc_col=desc_col
        )

        dates, descriptions, amounts, balances = _align_balance_arrays(
            debit_rows, date_col, desc_col, debit_col, None
        )

    elif strategy == "extract_amount":
        amount_col: str = parsed["amount_col"]
        amt_balance_col: str | None = parsed.get("balance_col")

        debit_rows = TransactionFilter.filter_negative_amounts(
            rows, amount_col=amount_col, desc_col=desc_col
        )

        dates, descriptions, amounts, balances = _align_amount_arrays(
            debit_rows, date_col, desc_col, amount_col, amt_balance_col
        )

    else:
        return True, {"DOCUMENT_TYPE": "BANK_STATEMENT"}

    # Date range from ALL parsed rows (not just debits)
    all_dates = [r.get(date_col, "") for r in rows if r.get(date_col)]
    date_range = _compute_date_range(all_dates) if all_dates else "NOT_FOUND"

    fields: dict[str, Any] = {
        "DOCUMENT_TYPE": "BANK_STATEMENT",
        "STATEMENT_DATE_RANGE": date_range,
        "TRANSACTION_DATES": " | ".join(dates) if dates else "NOT_FOUND",
        "LINE_ITEM_DESCRIPTIONS": " | ".join(descriptions)
        if descriptions
        else "NOT_FOUND",
        "TRANSACTION_AMOUNTS_PAID": " | ".join(amounts) if amounts else "NOT_FOUND",
        "ACCOUNT_BALANCE": " | ".join(balances) if balances else "NOT_FOUND",
    }

    logger.debug(
        "bank_post_process: strategy=%s, debits=%d, date_range=%s",
        strategy,
        len(dates),
        date_range,
    )

    return True, fields


# ------------------------------------------------------------------
# Best-type selector (called by GraphExecutor._run_validator)
# ------------------------------------------------------------------


def run_select_best_type(
    state: WorkflowState,
) -> tuple[bool, dict[str, Any]]:
    """Pick best document type by comparing probe results.

    Scoring:
    - Document probe: count non-NOT_FOUND uppercase fields (max ~15)
    - Bank probe: count non-None column mappings (max ~6)
    - Bank wins if >=3 real columns AND document probe score < 6
    - Otherwise document probe wins, DOCUMENT_TYPE from its output
    - PAYMENT_DATE present -> override to RECEIPT (evidence of completed payment)

    When bank wins, ``probe_bank_headers`` is renamed to ``detect_headers``
    in state so the downstream bank subgraph's inject references work.
    """
    # Score document probe
    doc_fields = state.node_results["probe_document"].parsed
    doc_score = sum(
        1
        for k, v in doc_fields.items()
        if isinstance(v, str)
        and v != "NOT_FOUND"
        and not k.startswith("_")
        and k.isupper()
    )

    # Score bank probe (may have parse error -> no column_mapping)
    bank_score = 0
    if state.has("probe_bank_headers"):
        mapping = state.get("probe_bank_headers.column_mapping", None)
        if isinstance(mapping, dict):
            bank_score = sum(1 for v in mapping.values() if v is not None)

    logger.debug(
        "select_best_type: doc_score=%d, bank_score=%d",
        doc_score,
        bank_score,
    )

    # Decision
    if bank_score >= 3 and doc_score < 6:
        best_type = "BANK_STATEMENT"
        # Remove document probe -- bank subgraph will produce fields
        del state.node_results["probe_document"]
        # Rename probe_bank_headers -> detect_headers so bank subgraph
        # inject references (detect_headers.column_mapping.*) resolve
        if "probe_bank_headers" in state.node_results:
            state.node_results["detect_headers"] = state.node_results.pop(
                "probe_bank_headers"
            )
    else:
        best_type = doc_fields.get("DOCUMENT_TYPE", "RECEIPT")
        if best_type == "NOT_FOUND":
            best_type = "RECEIPT"
        # PAYMENT_DATE present -> strong receipt signal (overrides model guess)
        payment_date = doc_fields.get("PAYMENT_DATE", "NOT_FOUND")
        if isinstance(payment_date, str) and payment_date != "NOT_FOUND":
            best_type = "RECEIPT"
        # Clean up bank probe from state
        if "probe_bank_headers" in state.node_results:
            del state.node_results["probe_bank_headers"]

    logger.debug("select_best_type: winner=%s", best_type)
    return True, {"DOCUMENT_TYPE": best_type}
