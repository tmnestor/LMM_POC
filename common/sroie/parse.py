"""Parsing of model responses into SROIE field values.

The parser is deliberately forgiving about packaging — casing, code
fences, surrounding prose — and strict about content: only the four SROIE
labels are read, and a declined or blank answer is recorded as absent
rather than as a value. That distinction matters at scoring time, where an
absent answer is a miss and a present one can be a wrong answer.
"""

import re

from common.sroie.ground_truth import SROIE_FIELDS

_FIELD_LINE = re.compile(
    rf"^\s*[-*]?\s*\**({'|'.join(SROIE_FIELDS)})\**\s*[:=]\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_CODE_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)

# A model that declines to answer must not be scored as having answered.
_DECLINED = {"not_found", "n/a", "na", "none", "null", "-", "unknown"}


def parse_sroie_response(raw_response: str) -> dict[str, str]:
    """Extract SROIE field values from a model response.

    Args:
        raw_response: The model's reply, verbatim.

    Returns:
        Mapping of SROIE field name to value, containing only fields the
        model actually answered. Absent and declined fields are omitted.
    """
    text = _CODE_FENCE.sub("", raw_response)

    fields: dict[str, str] = {}
    for match in _FIELD_LINE.finditer(text):
        name = match.group(1).lower()
        value = match.group(2).strip().strip("*").strip()
        if not value or value.lower() in _DECLINED:
            continue
        # Models sometimes restate a field while reasoning; the first
        # answer is the one the requested format asked for.
        fields.setdefault(name, value)
    return fields
