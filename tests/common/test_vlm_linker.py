"""Unit tests for common.vlm_linker (single-image + text-details adaptation)."""

from datetime import date

from PIL import Image

from common import vlm_linker
from common.transaction_matcher import ReceiptSummary
from common.vlm_linker import LinkPrompt, build_link_prompt, load_link_prompt

_RECEIPT = ReceiptSummary(
    image_name="CASE001_receipt.png",
    supplier_name="Woolworths",
    date=date(2024, 3, 18),
    total=127.35,
    document_type="RECEIPT",
)


# ---------------------------------------------------------------------------
# parse_link_response
# ---------------------------------------------------------------------------


def test_parse_link_response_found_block():
    raw = """
    --- RECEIPT 1 ---
    RECEIPT_STORE: Woolworths
    RECEIPT_DATE: 18/03/2024
    RECEIPT_TOTAL: 127.35
    MATCHED_TRANSACTION: FOUND
    TRANSACTION_DATE: 18/03/2024
    TRANSACTION_AMOUNT: 127.35
    TRANSACTION_DESCRIPTION: WOOLWORTHS 2847
    CONFIDENCE: HIGH
    REASONING: amount and store match
    """
    blocks = vlm_linker.parse_link_response(raw)
    assert len(blocks) == 1
    assert blocks[0]["MATCHED_TRANSACTION"] == "FOUND"
    assert blocks[0]["TRANSACTION_AMOUNT"] == "127.35"


def test_parse_link_response_empty():
    assert vlm_linker.parse_link_response("") == []
    assert vlm_linker.parse_link_response("   ") == []


def test_parse_link_response_filters_placeholder():
    raw = """
    --- RECEIPT 1 ---
    RECEIPT_STORE: [Next Purchase]
    MATCHED_TRANSACTION: NOT_FOUND
    """
    assert vlm_linker.parse_link_response(raw) == []


# ---------------------------------------------------------------------------
# build_link_prompt (using LinkPrompt)
# ---------------------------------------------------------------------------


def test_build_link_prompt_substitutes_receipt_details():
    prompt = LinkPrompt(
        prefix="COLS: {bank_column_context}\n",
        query="STORE: {receipt_store}\nDATE: {receipt_date}\nTOTAL: {receipt_total}\n",
    )
    out = vlm_linker.build_link_prompt(_RECEIPT, prompt=prompt)
    assert "STORE: Woolworths" in out
    assert "DATE: 18/03/2024" in out
    assert "TOTAL: 127.35" in out
    # generic fallback context substituted when no bank_columns given
    assert "{bank_column_context}" not in out


def test_build_link_prompt_not_found_fields():
    receipt = ReceiptSummary("r.png", "NOT_FOUND", None, None, "RECEIPT")
    prompt = LinkPrompt(
        prefix="{bank_column_context}|",
        query="{receipt_store}|{receipt_date}|{receipt_total}",
    )
    out = vlm_linker.build_link_prompt(receipt, prompt=prompt)
    # prefix has the bank_column_context replaced; query has NOT_FOUND values
    assert "|NOT_FOUND|NOT_FOUND|NOT_FOUND" in out


def test_build_link_prompt_with_bank_columns():
    prompt = LinkPrompt(prefix="{bank_column_context}", query="")
    cols = {"headers": ["Date", "Description", "Debit"], "mapping": {"debit": "Debit"}}
    out = vlm_linker.build_link_prompt(_RECEIPT, prompt=prompt, bank_columns=cols)
    assert "Debit" in out


# ---------------------------------------------------------------------------
# load_link_prompt — returns LinkPrompt
# ---------------------------------------------------------------------------


def test_load_link_prompt_returns_link_prompt():
    prompt = load_link_prompt("single_receipt_link")
    assert isinstance(prompt, LinkPrompt)
    assert isinstance(prompt.prefix, str)
    assert isinstance(prompt.query, str)
    assert len(prompt.prefix) > 0
    assert len(prompt.query) > 0


# ---------------------------------------------------------------------------
# Two-part prefix/query invariants
# ---------------------------------------------------------------------------


def _receipt() -> ReceiptSummary:
    return ReceiptSummary(
        image_name="CASE1_receipt.png",
        supplier_name="Bunnings Warehouse",
        date=date(2024, 2, 1),
        total=56.30,
        document_type="RECEIPT",
    )


def test_prefix_template_has_no_receipt_placeholder():
    prompt = load_link_prompt("single_receipt_link")
    assert "{receipt_" not in prompt.prefix  # invariant: prefix is receipt-independent


def test_build_link_prompt_puts_statement_before_receipt_key():
    full = build_link_prompt(_receipt(), prompt=load_link_prompt("single_receipt_link"))
    assert "BANK STATEMENT" in full
    assert "RECEIPT TO LOCATE" in full
    assert full.index("BANK STATEMENT") < full.index("RECEIPT TO LOCATE")
    assert "Bunnings Warehouse" in full
    assert "{receipt_" not in full and "{bank_column_context}" not in full  # all substituted


# ---------------------------------------------------------------------------
# call_vlm_linker (mocked generate_fn)
# ---------------------------------------------------------------------------


def test_call_vlm_linker_round_trip(tmp_path):
    bank_path = tmp_path / "CASE001_bank.png"
    Image.new("RGB", (8, 8), "white").save(bank_path)

    captured = {}

    def fake_generate(image, prompt, max_tokens, extra=None):
        captured["max_tokens"] = max_tokens
        captured["prompt"] = prompt
        captured["image_mode"] = image.mode
        captured["extra"] = extra
        return (
            "--- RECEIPT 1 ---\n"
            "RECEIPT_STORE: Woolworths\n"
            "MATCHED_TRANSACTION: FOUND\n"
            "TRANSACTION_AMOUNT: 127.35\n"
            "CONFIDENCE: MEDIUM\n"
        )

    # Use a LinkPrompt so the call uses the correct type
    link_prompt = LinkPrompt(
        prefix="find {bank_column_context} ",
        query="{receipt_total}",
    )
    blocks = vlm_linker.call_vlm_linker(
        fake_generate,
        bank_path,
        _RECEIPT,
        max_tokens=512,
        prompt=link_prompt,
    )
    assert blocks[0]["MATCHED_TRANSACTION"] == "FOUND"
    assert captured["max_tokens"] == 512
    assert "127.35" in captured["prompt"]
    assert captured["image_mode"] == "RGB"
    assert captured["extra"] == {"image_first": True}


def test_call_vlm_linker_handles_empty_response(tmp_path):
    bank_path = tmp_path / "CASE001_bank.png"
    Image.new("RGB", (8, 8), "white").save(bank_path)

    link_prompt = LinkPrompt(prefix="x {bank_column_context} ", query="{receipt_total}")
    blocks = vlm_linker.call_vlm_linker(
        lambda image, prompt, max_tokens, extra=None: None,
        bank_path,
        _RECEIPT,
        prompt=link_prompt,
    )
    assert blocks == []


# ---------------------------------------------------------------------------
# parse_link_response: JSON-format answers (regression 2026-06-08)
# The restructured prompt elicits a ```json {...}``` object instead of the
# --- RECEIPT N --- / KEY: value block; the parser must accept both.
# ---------------------------------------------------------------------------

_JSON_FOUND = """```json
{
  "RECEIPT_STORE": "Smith & Associates Accounting",
  "RECEIPT_DATE": "17/04/2023",
  "RECEIPT_TOTAL": 6493.84,
  "MATCHED_TRANSACTION": "FOUND",
  "TRANSACTION_DATE": "17/04/2023",
  "TRANSACTION_AMOUNT": 6493.84,
  "TRANSACTION_DESCRIPTION": "BPAY SMITH ASSOCIATES CRN 890779946",
  "CONFIDENCE": "HIGH",
  "REASONING": "amount matches the receipt total exactly"
}
```"""


def test_parse_link_response_accepts_json_object():
    blocks = vlm_linker.parse_link_response(_JSON_FOUND)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["MATCHED_TRANSACTION"] == "FOUND"
    assert b["RECEIPT_STORE"] == "Smith & Associates Accounting"
    # numeric JSON values must be stringified so downstream str ops work
    assert b["TRANSACTION_AMOUNT"] == "6493.84"
    assert b["TRANSACTION_DESCRIPTION"] == "BPAY SMITH ASSOCIATES CRN 890779946"


def test_parse_link_response_json_not_found():
    raw = '```json\n{"RECEIPT_STORE": "Acme Co", "MATCHED_TRANSACTION": "NOT_FOUND"}\n```'
    blocks = vlm_linker.parse_link_response(raw)
    assert len(blocks) == 1
    assert blocks[0]["MATCHED_TRANSACTION"] == "NOT_FOUND"


def test_parse_link_response_still_parses_block_format():
    raw = (
        "--- RECEIPT 1 ---\nRECEIPT_STORE: Acme Co\nMATCHED_TRANSACTION: FOUND\nTRANSACTION_AMOUNT: 12.50\n"
    )
    blocks = vlm_linker.parse_link_response(raw)
    assert len(blocks) == 1
    assert blocks[0]["MATCHED_TRANSACTION"] == "FOUND"
    assert blocks[0]["TRANSACTION_AMOUNT"] == "12.50"


# ---------------------------------------------------------------------------
# call_vlm_linker: tile budget threading (bank-18-tiles)
# ---------------------------------------------------------------------------


def test_call_vlm_linker_no_tile_budget_keeps_image_first_only(tmp_path):
    bank_path = tmp_path / "CASE001_bank.png"
    Image.new("RGB", (8, 8), "white").save(bank_path)

    captured = {}

    def fake_generate(image, prompt, max_tokens, extra=None):
        captured["extra"] = extra
        return "--- RECEIPT 1 ---\nRECEIPT_STORE: Woolworths\nMATCHED_TRANSACTION: FOUND\n"

    link_prompt = LinkPrompt(prefix="find {bank_column_context} ", query="{receipt_total}")
    vlm_linker.call_vlm_linker(fake_generate, bank_path, _RECEIPT, prompt=link_prompt)
    assert captured["extra"] == {"image_first": True}
