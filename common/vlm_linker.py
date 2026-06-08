"""Single-image VLM linker module.

Targeted fallback for receipts the algorithmic matcher could not confidently
link.  Sends ONE image — the bank statement — to the model together with the
receipt's already-extracted details (supplier, date, total) embedded as text
in the prompt, then parses the structured ``KEY: value`` response into a match
record.

This inverts the original multi-image design (receipt-image + bank-image): the
receipt has already been read by the extract stage, so re-sending its pixels is
wasteful.  Embedding the extracted fields as text focuses the model on the one
task it does better than the matcher — scanning the long, attention-decayed
tail of the bank-statement table for the matching debit row.

Public API
----------
load_link_prompt()    -- Load the template from prompts/<name>.yaml
build_link_prompt()   -- Substitute receipt details + bank column context
parse_link_response() -- Parse --- RECEIPT N --- blocks from VLM output
call_vlm_linker()     -- Full round-trip: build -> generate -> parse
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image

from common.transaction_matcher import ReceiptSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class LinkPrompt(NamedTuple):
    """Two-part linking prompt: a statement-only ``prefix`` shared across all
    receipts for one statement (cacheable), and a per-receipt ``query`` suffix."""

    prefix: str
    query: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_LINK_PROMPT = "single_receipt_link"

_LINK_FIELDS = frozenset(
    {
        "RECEIPT_STORE",
        "RECEIPT_DATE",
        "RECEIPT_TOTAL",
        "MATCHED_TRANSACTION",
        "TRANSACTION_DATE",
        "TRANSACTION_AMOUNT",
        "TRANSACTION_DESCRIPTION",
        "CONFIDENCE",
        "REASONING",
    }
)

_RECEIPT_HEADER_RE = re.compile(r"(?:---\s*)?RECEIPT\s+(\d+)(?:\s*---)?", re.IGNORECASE)

# Values that indicate the VLM emitted a placeholder rather than a real answer.
_PLACEHOLDER_LOWER = frozenset({"not specified", "unknown", "n/a", "none", "next purchase"})


# ---------------------------------------------------------------------------
# Fallback text (matches the original hardcoded prompt description)
# ---------------------------------------------------------------------------

_GENERIC_COLUMN_CONTEXT = (
    "The bank statement contains a table with columns for Date, Description,\n"
    "  and Amount (Debit/Credit or a single Amount column)."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_bank_column_context(bank_columns: dict[str, Any]) -> str:
    """Build a human-readable column description from bank column metadata.

    Args:
        bank_columns: Dict with "headers" (list[str]) and "mapping"
            (dict[str, str|None]) keys.

    Returns:
        Multi-line string describing the table structure for the VLM.
    """
    headers = bank_columns.get("headers", [])
    mapping = bank_columns.get("mapping", {})

    parts: list[str] = []

    if headers:
        parts.append(
            f"The bank statement contains a table with columns (left to right): "
            f"{', '.join(repr(h) for h in headers)}."
        )
    else:
        parts.append("The bank statement contains a table of transactions.")

    # Describe semantic roles
    role_descriptions: list[str] = []

    debit_col = mapping.get("debit")
    credit_col = mapping.get("credit")
    amount_col = mapping.get("amount")
    balance_col = mapping.get("balance")

    if debit_col and credit_col:
        role_descriptions.append(
            f"Withdrawals/debits appear in the '{debit_col}' column; "
            f"deposits/credits appear in the '{credit_col}' column."
        )
    elif amount_col:
        role_descriptions.append(
            f"Amounts appear in the '{amount_col}' column "
            f"(negative values are debits, positive are credits)."
        )

    if balance_col:
        role_descriptions.append(f"Running balance is in the '{balance_col}' column.")

    if role_descriptions:
        parts.append(" ".join(role_descriptions))

    return "\n  ".join(parts)


def _format_receipt_date(receipt: ReceiptSummary) -> str:
    """Render the receipt date as DD/MM/YYYY, or NOT_FOUND when absent."""
    if receipt.date is None:
        return "NOT_FOUND"
    return receipt.date.strftime("%d/%m/%Y")


def _format_receipt_total(receipt: ReceiptSummary) -> str:
    """Render the receipt total as a 2dp decimal, or NOT_FOUND when absent."""
    if receipt.total is None:
        return "NOT_FOUND"
    return f"{receipt.total:.2f}"


def _format_receipt_store(receipt: ReceiptSummary) -> str:
    """Render the supplier/store name, or NOT_FOUND when absent."""
    supplier = (receipt.supplier_name or "").strip()
    if not supplier or supplier.upper() == "NOT_FOUND":
        return "NOT_FOUND"
    return supplier


def _load_image(path: Path) -> Image.Image:
    """Load an image file as an RGB PIL Image for the model seam."""
    with Image.open(path) as raw_img:
        return raw_img.convert("RGB")


def _is_placeholder(block: dict[str, str]) -> bool:
    """Return True if the block looks like a VLM template artifact.

    Detects:
    - Empty RECEIPT_STORE.
    - RECEIPT_STORE that starts with ``[`` (e.g. ``[Next Purchase]``).
    - RECEIPT_STORE whose value (case-insensitive) is a known placeholder word
      (e.g. "unknown", "not specified", "n/a", "none", "next purchase").

    Only RECEIPT_STORE is inspected for placeholder content — other fields
    (like CONFIDENCE: LOW) use those words legitimately.

    Args:
        block: Parsed key/value dict for one receipt block.

    Returns:
        True if the block should be filtered out.
    """
    store = block.get("RECEIPT_STORE", "")
    if not store:
        return True
    if store.startswith("["):
        return True
    if store.lower() in _PLACEHOLDER_LOWER:
        return True
    return False


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_link_prompt(prompt_name: str | None = None) -> LinkPrompt:
    """Load a two-part linking prompt (prefix + query) from YAML.

    Args:
        prompt_name: Base name of the prompt file (without ``.yaml`` extension).
            E.g. ``"single_receipt_link"`` loads ``prompts/single_receipt_link.yaml``.
            If None, loads the default ``single_receipt_link`` prompt.

    Returns:
        A :class:`LinkPrompt` with ``prefix`` (statement instructions, cacheable)
        and ``query`` (per-receipt key + answer block).

    Raises:
        FileNotFoundError: With a diagnostic error message if the YAML file is
            missing.
        KeyError: With a diagnostic error message if either ``prefix_template``
            or ``query_template`` key is absent from the YAML.
    """
    import yaml

    name = prompt_name or _DEFAULT_LINK_PROMPT
    prompt_path = _PROMPTS_DIR / f"{name}.yaml"

    if not prompt_path.exists():
        msg = (
            f"What: linking prompt YAML file is missing.\n"
            f"Where: {prompt_path}\n"
            f"Expected: a YAML file with 'prefix_template' and 'query_template' keys.\n"
            f"How to fix: create {prompt_path} with both keys, e.g.:\n"
            f"  prefix_template: |\n"
            f"    You are given ONE document image: a BANK STATEMENT ...\n"
            f"  query_template: |\n"
            f"    RECEIPT TO LOCATE: ...\n"
            f"    --- RECEIPT 1 ---\n"
            f"    RECEIPT_STORE: ..."
        )
        raise FileNotFoundError(msg)

    with prompt_path.open() as fh:
        data = yaml.safe_load(fh)

    missing = [k for k in ("prefix_template", "query_template") if k not in data]
    if missing:
        msg = (
            f"What: linking prompt YAML is missing key(s): {', '.join(missing)}.\n"
            f"Where: {prompt_path} — top-level keys 'prefix_template' and 'query_template'\n"
            f"Expected: both a 'prefix_template:' (statement instructions, no receipt "
            f"data) and a 'query_template:' (per-receipt key + answer block).\n"
            f"How to fix: add the missing key(s) to {prompt_path}, e.g.:\n"
            f"  prefix_template: |\n"
            f"    You are given ONE document image: a BANK STATEMENT ...\n"
            f"  query_template: |\n"
            f"    RECEIPT TO LOCATE: ...\n"
            f"    --- RECEIPT 1 ---\n"
            f"    RECEIPT_STORE: ..."
        )
        raise KeyError(msg)

    return LinkPrompt(prefix=data["prefix_template"], query=data["query_template"])


def build_link_prompt(
    receipt: ReceiptSummary,
    prompt: LinkPrompt | None = None,
    bank_columns: dict[str, Any] | None = None,
) -> str:
    """Render prefix (with bank column context) + query (with receipt key).

    Substitutes ``{bank_column_context}`` in the prefix and
    ``{receipt_store}`` / ``{receipt_date}`` / ``{receipt_total}``
    in the query, then concatenates them into a single prompt string.

    Returns ``prefix + query`` — prefix first, so the statement prefix stays a
    stable shared prefix for vLLM's cache.

    Args:
        receipt: The receipt summary (supplier, date, total) to look up.
        prompt: Two-part :class:`LinkPrompt`. If None, calls
            :func:`load_link_prompt`.
        bank_columns: Bank column metadata (headers + mapping) from extraction.
            If provided, injects specific column names into the prefix.
            If None, uses generic fallback text.

    Returns:
        The fully-substituted prompt string ready for the model.
    """
    if prompt is None:
        prompt = load_link_prompt()

    if bank_columns is not None:
        context = _format_bank_column_context(bank_columns)
    else:
        context = _GENERIC_COLUMN_CONTEXT

    prefix = prompt.prefix.replace("{bank_column_context}", context)
    query = (
        prompt.query.replace("{receipt_store}", _format_receipt_store(receipt))
        .replace("{receipt_date}", _format_receipt_date(receipt))
        .replace("{receipt_total}", _format_receipt_total(receipt))
    )
    return prefix + query


# Greedy match of the outermost {...} object (the model sometimes wraps the
# answer in a ```json ... ``` fence instead of the --- RECEIPT N --- block).
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_link_block(raw_response: str) -> dict[str, str] | None:
    """Parse a JSON-object linking answer into a stringified field dict.

    InternVL3.5 may reply with a ``{...}`` object (often inside a ```json
    fence) rather than the ``--- RECEIPT N --- / KEY: value`` block. Extract the
    object, keep only ``_LINK_FIELDS`` keys, and stringify every value so the
    downstream string handling (``_parse_amount_str``, ``_is_bank_row_echo``)
    works unchanged. Returns None when no JSON object carrying
    ``MATCHED_TRANSACTION`` is present, so a real KEY: value block falls through
    to the block parser.
    """
    match = _JSON_OBJ_RE.search(raw_response)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    block: dict[str, str] = {}
    for key, value in obj.items():
        upper = str(key).strip().upper()
        if upper in _LINK_FIELDS:
            block[upper] = "" if value is None else str(value).strip()
    # Require the decision field, else this isn't a link answer (guards against
    # stray braces appearing inside a KEY: value response).
    return block if "MATCHED_TRANSACTION" in block else None


def parse_link_response(raw_response: str) -> list[dict[str, str]]:
    """Parse a VLM linking response: a ```json object OR ``--- RECEIPT N ---`` blocks.

    Each block is parsed into a dict of ``_LINK_FIELDS`` key/value pairs.
    Blocks that look like VLM template placeholders are filtered out via
    :func:`_is_placeholder`.

    Args:
        raw_response: Raw text from the VLM (may be empty or malformed).

    Returns:
        List of dicts, one per valid receipt block. Empty list if no blocks
        are found or all blocks are placeholders.
    """
    if not raw_response or not raw_response.strip():
        return []

    # The model may answer with a ```json {...}``` object instead of the
    # --- RECEIPT N --- / KEY: value block. Try JSON first; fall back to blocks.
    json_block = _parse_json_link_block(raw_response)
    if json_block is not None:
        return [] if _is_placeholder(json_block) else [json_block]

    blocks: list[dict[str, str]] = []
    current_block: dict[str, str] = {}
    in_block = False

    for raw_line in raw_response.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if _RECEIPT_HEADER_RE.match(line):
            if in_block and current_block:
                blocks.append(current_block)
            current_block = {}
            in_block = True
            continue

        if in_block:
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if key in _LINK_FIELDS:
                current_block[key] = value

    if in_block and current_block:
        blocks.append(current_block)

    return [b for b in blocks if not _is_placeholder(b)]


def call_vlm_linker(
    generate_fn: Any,
    bank_path: Path,
    receipt: ReceiptSummary,
    *,
    max_tokens: int = 1024,
    prompt: LinkPrompt | None = None,
    bank_columns: dict[str, Any] | None = None,
    max_tiles: int | None = None,
    min_tiles: int | None = None,
) -> list[dict[str, str]]:
    """Send the bank-statement image + receipt details to the model and parse.

    Builds the single-image prompt (bank statement pixels + receipt fields as
    text), calls the shared extract seam
    ``generate_fn(image, prompt, max_tokens=..., extra={"image_first": True})``
    (deterministic decoding — temperature is fixed at 0 by the backend), and
    returns parsed receipt match records.

    When a tile budget is supplied (``max_tiles`` / ``min_tiles``), it is added
    to the ``extra`` dict so the backend pre-tiles the dense bank-statement
    image at the configured ceiling instead of vLLM's internal default. Both
    flow from the ``bank_statement`` image budget in run_config.yml.

    Args:
        generate_fn: The processor's ``generate`` callable, i.e.
            ``processor.generate(image, prompt, max_tokens, extra) -> str``.
        bank_path: Path to the bank-statement image file.
        receipt: Receipt summary to locate in the statement.
        max_tokens: Maximum tokens to generate.
        prompt: Two-part :class:`LinkPrompt`. If None, loads from YAML via
            :func:`load_link_prompt`.
        bank_columns: Bank column metadata from the extraction stage. If
            provided, injects specific column names into the linking prompt.
        max_tiles: Pre-tiling ceiling for the bank image. Forwarded to the
            backend only when not None.
        min_tiles: Pre-tiling floor for the bank image. Forwarded to the
            backend only when not None.

    Returns:
        Parsed list of receipt match dicts. Empty list if the VLM returns
        nothing parseable.
    """
    full_prompt = build_link_prompt(receipt, prompt=prompt, bank_columns=bank_columns)
    image = _load_image(bank_path)

    extra: dict[str, Any] = {"image_first": True}
    if max_tiles is not None:
        extra["max_tiles"] = max_tiles
    if min_tiles is not None:
        extra["min_tiles"] = min_tiles
    text = generate_fn(image, full_prompt, max_tokens=max_tokens, extra=extra)
    text = (text or "").strip()

    logger.info(
        "vlm_linker: receipt=%s (store=%s total=%s) bank=%s response_chars=%d",
        receipt.image_name,
        _format_receipt_store(receipt),
        _format_receipt_total(receipt),
        bank_path.name,
        len(text),
    )

    return parse_link_response(text)
