"""Tests for the shared SROIE prompt."""

from common.sroie.ground_truth import SROIE_FIELDS
from common.sroie.parse import parse_sroie_response
from common.sroie.prompt import SROIE_PROMPT


def test_prompt_requests_every_scored_field() -> None:
    """A field dropped from the prompt scores zero for every document and
    reads as a model failure."""
    for field in SROIE_FIELDS:
        assert f"{field}:" in SROIE_PROMPT


def test_prompt_shows_a_worked_decimal_example_for_the_total() -> None:
    """Asking for a 'plain number' with no example made InternVL3.5 write
    '490' for 4.90 on 196 of 347 receipts — every digit right, the decimal
    point dropped. The example is what stops that."""
    assert "4.95" in SROIE_PROMPT
    assert "decimal point" in SROIE_PROMPT


def test_prompt_excludes_the_company_registration_number() -> None:
    """Gemma 4 appended '(519537-X)' on 34 of its 120 company misses;
    InternVL3 did it once in 46. The number IS printed on the receipt, so
    without an instruction either answer is defensible and the score
    measures which convention a model happens to prefer."""
    assert "registration number" in SROIE_PROMPT


def test_the_parser_understands_the_format_the_prompt_asks_for() -> None:
    """The prompt and the parser must not drift apart. A model that obeys
    the prompt exactly must parse cleanly."""
    obedient_response = (
        "company: OJC MARKETING SDN BHD\ndate: 15/01/2019\naddress: NO 2 & 4, JALAN BAYU 4\ntotal: 193.00"
    )

    assert set(parse_sroie_response(obedient_response)) == set(SROIE_FIELDS)
