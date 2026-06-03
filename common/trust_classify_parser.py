"""Evidence-based trust document type parser.

Extracts structured evidence fields from a VLM response and derives
the trust document type using priority-ordered rules. Mirrors the
ClassificationParser pattern from common/turn_parsers.py.

The prompt asks the VLM to report observable features (HEADER, HAS_ITEM_13,
HAS_ABN, HAS_DISTRIBUTION_TABLE, ADDRESSED_TO), then this parser derives
the type from evidence — more robust and auditable than asking the model
to name the type directly.
"""

import re
from pathlib import Path
from typing import Any

import yaml

# Prompt YAML path (for fallback_keywords lookup)
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "trust_document_type_detection.yaml"


def _load_fallback_keywords() -> dict[str, list[str]]:
    """Load fallback_keywords from the trust detection prompt YAML."""
    with _PROMPT_PATH.open() as f:
        data = yaml.safe_load(f)
    return data.get("fallback_keywords", {})


def _extract_evidence(raw_response: str) -> dict[str, str]:
    """Extract structured evidence fields from VLM response.

    Handles two response formats:

    Compact (answer on same line as field):
        1. HEADER: Trust Income Tax Return
        2. HAS_ABN: YES

    Verbose (model echoes the question, answer on a ``- `` prefixed line):
        1. HEADER: What is the main title or header of this document?
           Write the exact text of the primary heading.
           - Trust Income Tax Return 2024
        2. HAS_ABN: Does this document show an Australian Business Number?
           Answer YES or NO.
           - **YES**
    """
    # Split response into per-field blocks using the numbered field anchors.
    # Each block runs from one field header to the next (or end of string).
    field_order = ["HEADER", "HAS_ABN", "HAS_TFN", "HAS_DISTRIBUTION_TABLE", "HAS_ITEM_13", "ADDRESSED_TO"]
    block_pattern = re.compile(
        r"(?:(?:\d\.\s*)?(?:" + "|".join(field_order) + r")\s*:)",
        re.IGNORECASE,
    )
    splits = list(block_pattern.finditer(raw_response))

    blocks: dict[str, str] = {}
    for i, m in enumerate(splits):
        # Identify which field this block belongs to
        header_text = m.group(0).upper()
        for field in field_order:
            if field in header_text:
                start = m.end()
                end = splits[i + 1].start() if i + 1 < len(splits) else len(raw_response)
                blocks[field] = raw_response[start:end]
                break

    evidence: dict[str, str] = {}

    # --- HEADER: extract the document title ---
    if "HEADER" in blocks:
        block = blocks["HEADER"]
        # Prefer a ``- <answer>`` line (verbose format, strip optional markdown bold)
        bullet = re.search(r"[-–]\s*\*{0,2}(.+?)\*{0,2}\s*$", block, re.MULTILINE)
        if bullet:
            evidence["HEADER"] = bullet.group(1).strip()
        else:
            # Compact format: answer is the rest of the first line
            first_line = block.strip().split("\n")[0].strip()
            if first_line:
                evidence["HEADER"] = first_line

    # --- YES/NO fields ---
    for field in field_order[1:]:  # skip HEADER
        if field not in blocks:
            continue
        block = blocks[field]
        # Prefer a ``- <answer>`` bullet line (verbose format) — the question
        # text often contains "Answer YES or NO" which would false-match if we
        # searched the entire block naively.
        bullet_yn = re.search(r"[-–]\s*\*{0,2}(YES|NO)\*{0,2}", block, re.IGNORECASE)
        if bullet_yn:
            evidence[field] = bullet_yn.group(1).upper()
        else:
            # Compact format: answer is on the same line as the field name
            first_line = block.strip().split("\n")[0]
            yn = re.search(r"\*{0,2}(YES|NO)\*{0,2}", first_line, re.IGNORECASE)
            if yn:
                evidence[field] = yn.group(1).upper()

    return evidence


def _derive_type_from_evidence(evidence: dict[str, str]) -> tuple[str, float]:
    """Apply priority-ordered rules to derive document type from evidence.

    Returns:
        Tuple of (document_type, confidence).
    """
    header = evidence.get("HEADER", "").lower()
    has_item_13 = evidence.get("HAS_ITEM_13", "").upper() == "YES"
    has_distribution_table = evidence.get("HAS_DISTRIBUTION_TABLE", "").upper() == "YES"
    addressed_to = evidence.get("ADDRESSED_TO", "").upper() == "YES"
    has_abn = evidence.get("HAS_ABN", "").upper() == "YES"

    # Priority 1: HAS_ITEM_13 = YES -> BENEFICIARY_ITR
    if has_item_13:
        return "BENEFICIARY_ITR", 0.95

    # --- Header-based rules (priorities 2-4) ---
    # The header is the most reliable signal from the VLM. YES/NO fields
    # (ADDRESSED_TO, HAS_DISTRIBUTION_TABLE) are noisy — the model
    # hallucinates them on trust returns and misses them on distribution
    # statements. Header keywords must fire before YES/NO-based rules.

    # Priority 2: Header contains "schedule" -> INCOME_SCHEDULE
    if "schedule" in header:
        return "INCOME_SCHEDULE", 0.85

    # Priority 3: Header contains "trust" + "return" -> TRUST_RETURN
    if "trust" in header and "return" in header:
        return "TRUST_RETURN", 0.85

    # Priority 4: Header contains "distribution" -> DISTRIBUTION_STMT
    if "distribution" in header:
        return "DISTRIBUTION_STMT", 0.85

    # --- YES/NO fallback rules (priorities 5-6) ---
    # Only reached when the header is ambiguous or missing.

    # Priority 5: ADDRESSED_TO + HAS_DISTRIBUTION_TABLE -> DISTRIBUTION_STMT
    if addressed_to and has_distribution_table:
        return "DISTRIBUTION_STMT", 0.9

    # Priority 6: HAS_ABN + header mentions "trust" or "return" -> TRUST_RETURN
    if has_abn and ("trust" in header or "return" in header):
        return "TRUST_RETURN", 0.85

    return "", 0.0


def _keyword_fallback(raw_response: str) -> tuple[str, float]:
    """Fall back to keyword matching on the raw response text.

    Returns:
        Tuple of (document_type, confidence).
    """
    fallback_keywords = _load_fallback_keywords()
    text_lower = raw_response.lower()

    # Check each type's keywords; return first match
    # Order: BENEFICIARY_ITR first (most distinctive keywords)
    check_order = ["BENEFICIARY_ITR", "DISTRIBUTION_STMT", "INCOME_SCHEDULE", "TRUST_RETURN"]
    for doc_type in check_order:
        keywords = fallback_keywords.get(doc_type, [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return doc_type, 0.5

    return "UNKNOWN_TRUST_DOC", 0.3


def parse_trust_classification(raw_response: str) -> dict[str, Any]:
    """Parse VLM response to derive trust document type.

    Extracts structured evidence fields, applies priority-ordered derivation
    rules, and falls back to keyword matching if no structured evidence.

    Args:
        raw_response: Raw text response from the VLM.

    Returns:
        Dict with keys: document_type, confidence, evidence, raw_response.
    """
    evidence = _extract_evidence(raw_response)

    # Try evidence-based derivation first
    doc_type, confidence = _derive_type_from_evidence(evidence)

    # Fall back to keyword matching if evidence-based derivation fails
    if not doc_type:
        doc_type, confidence = _keyword_fallback(raw_response)

    return {
        "document_type": doc_type,
        "confidence": confidence,
        "evidence": evidence,
        "raw_response": raw_response,
    }
