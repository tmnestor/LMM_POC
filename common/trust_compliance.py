"""Trust distribution compliance validator.

Cross-document comparison of linking fields across 4 trust distribution
documents (trust return, distribution statement, income schedule,
beneficiary ITR).  Detects NRO Private Wealth discrepancy types:

- under_reported_income: ITR income < distribution statement income (>1%)
- over_claimed_franking: ITR franking > distribution statement franking (>1%)
- missing_cgt: distribution CGT > 0 but income schedule CGT = 0
- trust_return_mismatch: trust return share != distribution share (>1%)
"""

import logging
import re
from collections import Counter
from typing import Any

from common.extraction_types import WorkflowState

logger = logging.getLogger(__name__)

_ABN_WEIGHTS = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def validate_abn(abn: str) -> bool:
    """Validate an Australian Business Number using the modulus-89 check digit algorithm.

    Args:
        abn: ABN string (may contain spaces).

    Returns:
        True if the ABN passes the check digit validation.
    """
    digits = _normalise_id(abn)
    if len(digits) != 11 or not digits.isdigit():
        return False
    nums = [int(d) for d in digits]
    nums[0] -= 1  # subtract 1 from first digit per NRO spec
    return sum(n * w for n, w in zip(nums, _ABN_WEIGHTS)) % 89 == 0


def _resolve_abn(abn_values: dict[str, str]) -> str:
    """Pick canonical ABN using validation + majority voting.

    Strategy:
    1. Filter to non-empty ABNs, validate each with modulus-89.
    2. Among valid ABNs, pick the majority value (2+ sources agree).
    3. If no majority among valid, pick first valid (distribution_stmt preferred).
    4. If none valid, fall back to first non-empty (current behaviour).

    Args:
        abn_values: Mapping of source name to normalised ABN string.

    Returns:
        Best canonical ABN string.
    """
    non_empty = {src: abn for src, abn in abn_values.items() if abn}
    if not non_empty:
        return ""

    valid = {src: abn for src, abn in non_empty.items() if validate_abn(abn)}
    invalid = {src: abn for src, abn in non_empty.items() if not validate_abn(abn)}

    for src, abn in invalid.items():
        logger.warning("ABN validation failed for %s: %r", src, abn)

    if valid:
        # Count occurrences of each valid ABN value
        counts = Counter(valid.values())
        majority = [abn for abn, count in counts.items() if count >= 2]

        if majority:
            chosen = majority[0]
            ds_abn = valid.get("distribution_stmt")
            if ds_abn and ds_abn != chosen:
                logger.warning(
                    "Majority voting overrode distribution_stmt ABN: distribution_stmt=%r, majority=%r",
                    ds_abn,
                    chosen,
                )
            return chosen

        # No majority — prefer distribution_stmt if valid, else first valid
        preferred_order = ("distribution_stmt", "trust_return", "income_schedule")
        for src in preferred_order:
            if src in valid:
                return valid[src]
        return next(iter(valid.values()))

    # No valid ABNs — fall back to first non-empty (distribution_stmt preferred)
    preferred_order = ("distribution_stmt", "trust_return", "income_schedule")
    for src in preferred_order:
        if src in non_empty:
            return non_empty[src]
    return next(iter(non_empty.values()))


def _parse_amount(s: str) -> float | None:
    """Convert a currency/numeric string to float.

    Strips $, commas, spaces.  Returns None if unparseable.
    """
    if not s or s.upper() in ("NOT_FOUND", "N/A", "NONE", "NULL"):
        return None
    cleaned = re.sub(r"[$,\s]", "", s.strip())
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalise_id(s: str) -> str:
    """Normalise ABN/TFN by stripping all spaces."""
    return re.sub(r"\s+", "", s.strip()) if s else ""


def _pct_diff(a: float, b: float) -> float:
    """Percentage difference relative to max(abs(a), abs(b))."""
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.0
    return abs(a - b) / denom


def run_trust_compliance(
    state: WorkflowState,
    *,
    tolerance: float = 0.01,
) -> tuple[bool, dict[str, Any]]:
    """Compare linking fields across 4 extraction nodes.

    Args:
        state: Accumulated workflow state with node results from
            extract_trust_return, extract_distribution_stmt,
            extract_income_schedule, extract_beneficiary_itr.
        tolerance: Fractional tolerance for amount comparisons (default 1%).

    Returns:
        ``(all_match, result_dict)`` where ``result_dict`` contains
        compliance_status, discrepancy_type, discrepancy_details,
        field_comparisons, and canonical linking field values.
    """
    # Gather parsed fields from each extraction node
    tr = state.node_results["extract_trust_return"].parsed if state.has("extract_trust_return") else {}
    ds = (
        state.node_results["extract_distribution_stmt"].parsed
        if state.has("extract_distribution_stmt")
        else {}
    )
    isc = (
        state.node_results["extract_income_schedule"].parsed if state.has("extract_income_schedule") else {}
    )
    itr = (
        state.node_results["extract_beneficiary_itr"].parsed if state.has("extract_beneficiary_itr") else {}
    )

    # --- Normalise identifiers ---
    abn_tr = _normalise_id(tr.get("TRUST_ABN", ""))
    abn_ds = _normalise_id(ds.get("TRUST_ABN", ""))
    abn_isc = _normalise_id(isc.get("TRUST_ABN", ""))

    tfn_tr = _normalise_id(tr.get("BENEFICIARY_TFN", ""))
    tfn_ds = _normalise_id(ds.get("BENEFICIARY_TFN", ""))
    tfn_isc = _normalise_id(isc.get("BENEFICIARY_TFN", ""))
    tfn_itr = _normalise_id(itr.get("BENEFICIARY_TFN", ""))

    # --- Parse amounts ---
    income_tr = _parse_amount(tr.get("SHARE_OF_NET_INCOME", ""))
    income_ds = _parse_amount(ds.get("SHARE_OF_NET_INCOME", ""))
    income_isc = _parse_amount(isc.get("SHARE_OF_NET_INCOME", ""))
    income_itr = _parse_amount(itr.get("TOTAL_TRUST_INCOME", ""))

    franking_tr = _parse_amount(tr.get("FRANKING_CREDIT", ""))
    franking_ds = _parse_amount(ds.get("FRANKING_CREDIT", ""))
    franking_isc = _parse_amount(isc.get("FRANKING_CREDIT", ""))
    franking_itr = _parse_amount(itr.get("TRUST_FRANKING_CREDIT", ""))

    cgt_tr = _parse_amount(tr.get("CAPITAL_GAIN_COMPONENT", ""))
    cgt_ds = _parse_amount(ds.get("CAPITAL_GAIN_COMPONENT", ""))
    cgt_isc = _parse_amount(isc.get("CAPITAL_GAIN_COMPONENT", ""))

    # --- Build field comparisons ---
    comparisons: dict[str, dict[str, Any]] = {}

    # Trust ABN: trust_return, distribution_stmt, income_schedule
    abn_values = {"trust_return": abn_tr, "distribution_stmt": abn_ds, "income_schedule": abn_isc}
    abn_unique = {v for v in abn_values.values() if v}
    comparisons["trust_abn"] = {"values": abn_values, "match": len(abn_unique) <= 1}

    # Beneficiary TFN: all 4 documents
    tfn_values = {
        "trust_return": tfn_tr,
        "distribution_stmt": tfn_ds,
        "income_schedule": tfn_isc,
        "beneficiary_itr": tfn_itr,
    }
    tfn_unique = {v for v in tfn_values.values() if v}
    comparisons["beneficiary_tfn"] = {"values": tfn_values, "match": len(tfn_unique) <= 1}

    # Share of net income: all 4 (ITR = TOTAL_TRUST_INCOME)
    income_values = {
        "trust_return": income_tr,
        "distribution_stmt": income_ds,
        "income_schedule": income_isc,
        "beneficiary_itr": income_itr,
    }
    _check_amount_match(comparisons, "share_of_net_income", income_values, tolerance)

    # Franking credit: all 4 (ITR = TRUST_FRANKING_CREDIT)
    franking_values = {
        "trust_return": franking_tr,
        "distribution_stmt": franking_ds,
        "income_schedule": franking_isc,
        "beneficiary_itr": franking_itr,
    }
    _check_amount_match(comparisons, "franking_credit", franking_values, tolerance)

    # Capital gain component: trust_return, distribution_stmt, income_schedule
    cgt_values = {"trust_return": cgt_tr, "distribution_stmt": cgt_ds, "income_schedule": cgt_isc}
    _check_amount_match(comparisons, "capital_gain_component", cgt_values, tolerance)

    # --- Classify discrepancy (first match wins) ---
    discrepancy_type: str | None = None
    discrepancy_details: str | None = None

    # 1. under_reported_income: ITR < distribution (>tolerance difference)
    if income_itr is not None and income_ds is not None:
        if income_itr < income_ds and _pct_diff(income_itr, income_ds) > tolerance:
            discrepancy_type = "under_reported_income"
            discrepancy_details = (
                f"ITR reports ${income_itr:.2f} trust income but "
                f"Distribution Statement shows ${income_ds:.2f}"
            )

    # 2. over_claimed_franking: ITR > distribution (>tolerance difference)
    if discrepancy_type is None and franking_itr is not None and franking_ds is not None:
        if franking_itr > franking_ds and _pct_diff(franking_itr, franking_ds) > tolerance:
            discrepancy_type = "over_claimed_franking"
            discrepancy_details = (
                f"ITR claims ${franking_itr:.2f} franking credit but "
                f"Distribution Statement shows ${franking_ds:.2f}"
            )

    # 3. missing_cgt: distribution CGT > 0 but schedule = 0
    if discrepancy_type is None and cgt_ds is not None and cgt_isc is not None:
        if cgt_ds > 0 and cgt_isc == 0:
            discrepancy_type = "missing_cgt"
            discrepancy_details = (
                f"Distribution Statement shows ${cgt_ds:.2f} capital gain but Income Schedule reports $0.00"
            )

    # 4. trust_return_mismatch: trust return share != distribution share (>tolerance)
    if discrepancy_type is None and income_tr is not None and income_ds is not None:
        if _pct_diff(income_tr, income_ds) > tolerance:
            discrepancy_type = "trust_return_mismatch"
            discrepancy_details = (
                f"Trust Return shows ${income_tr:.2f} share but "
                f"Distribution Statement shows ${income_ds:.2f}"
            )

    all_match = discrepancy_type is None
    compliance_status = "compliant" if all_match else "non_compliant"

    # Canonical linking field values (validated + majority voted)
    canonical_abn = _resolve_abn(abn_values)
    canonical_tfn = tfn_ds or tfn_tr or tfn_isc or tfn_itr

    result = {
        "TRUST_ABN": canonical_abn,
        "BENEFICIARY_TFN": canonical_tfn,
        "SHARE_OF_NET_INCOME": f"{income_ds:.2f}" if income_ds is not None else "NOT_FOUND",
        "FRANKING_CREDIT": f"{franking_ds:.2f}" if franking_ds is not None else "NOT_FOUND",
        "CAPITAL_GAIN_COMPONENT": f"{cgt_ds:.2f}" if cgt_ds is not None else "NOT_FOUND",
        "COMPLIANCE_STATUS": compliance_status,
        "DISCREPANCY_TYPE": discrepancy_type or "none",
        "DISCREPANCY_DETAILS": discrepancy_details or "none",
        "field_comparisons": comparisons,
    }

    logger.info(
        "Trust compliance: %s (discrepancy=%s)",
        compliance_status,
        discrepancy_type or "none",
    )

    return all_match, result


def _check_amount_match(
    comparisons: dict[str, dict[str, Any]],
    field_name: str,
    values: dict[str, float | None],
    tolerance: float,
) -> None:
    """Check if all non-None amounts agree within tolerance."""
    non_none = [v for v in values.values() if v is not None]
    if len(non_none) < 2:
        comparisons[field_name] = {"values": values, "match": True}
        return

    ref = non_none[0]
    match = all(_pct_diff(ref, v) <= tolerance for v in non_none[1:])
    comparisons[field_name] = {"values": values, "match": match}
