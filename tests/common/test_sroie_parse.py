"""Tests for parsing model responses into SROIE field values."""

from common.sroie.parse import parse_sroie_response


def test_parses_the_four_fields_from_a_clean_response() -> None:
    response = (
        "company: OJC MARKETING SDN BHD\ndate: 15/01/2019\naddress: NO 2 & 4, JALAN BAYU 4\ntotal: 193.00"
    )

    assert parse_sroie_response(response) == {
        "company": "OJC MARKETING SDN BHD",
        "date": "15/01/2019",
        "address": "NO 2 & 4, JALAN BAYU 4",
        "total": "193.00",
    }


def test_field_labels_are_case_insensitive() -> None:
    """Models vary the label casing regardless of how the prompt writes it."""
    assert parse_sroie_response("COMPANY: ACME\nDate:  01/02/2018 ") == {
        "company": "ACME",
        "date": "01/02/2018",
    }


def test_ignores_prose_around_the_answer_block() -> None:
    """A preamble is a formatting quirk, not a wrong answer."""
    response = "Sure! Here are the fields I extracted:\n\ncompany: ACME\ntotal: 9.00\n\nHope this helps."

    assert parse_sroie_response(response) == {"company": "ACME", "total": "9.00"}


def test_strips_markdown_code_fences() -> None:
    response = "```\ncompany: ACME\ntotal: 9.00\n```"

    assert parse_sroie_response(response) == {"company": "ACME", "total": "9.00"}


def test_not_found_is_absent_rather_than_a_value() -> None:
    """NOT_FOUND is the model declining to answer. Recording it as a value
    would score a blank as a wrong answer instead of a miss."""
    assert parse_sroie_response("company: ACME\naddress: NOT_FOUND") == {"company": "ACME"}


def test_empty_value_is_absent() -> None:
    assert parse_sroie_response("company: ACME\naddress:   ") == {"company": "ACME"}


def test_first_occurrence_wins_when_a_field_repeats() -> None:
    """Models sometimes restate a field while reasoning. The first answer
    is the one the format asked for."""
    assert parse_sroie_response("total: 9.00\ntotal: 90.00") == {"total": "9.00"}


def test_unrecognised_labels_are_ignored() -> None:
    """Only the four SROIE fields are scored."""
    assert parse_sroie_response("company: ACME\ngst: 0.54") == {"company": "ACME"}


def test_empty_response_yields_no_fields() -> None:
    assert parse_sroie_response("") == {}
