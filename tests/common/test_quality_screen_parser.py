"""Reading InternVL3.5's answers to the quality-screen prompt.

The prompt (prompts/quality_screen.yaml -> quality_screen_v5) asks eight
numbered questions: six YES/NO defect criteria and one graded OVERALL. This
module's job is to turn the model's text back into those answers, or to say
clearly that it could not.

The malformed cases are the reason this is a separate, tested unit. A
constrained answer space does not guarantee a constrained response, and a
reader that quietly turns an unreadable answer into a plausible one produces a
score that looks fine and means nothing.
"""

from pathlib import Path

import pytest

from common.quality_screen_parser import (
    ScreenVocabularyError,
    load_screen_vocabulary,
    parse_quality_response,
)

PROMPT_CONFIG = Path("prompts/quality_screen.yaml")
VARIANT = "quality_screen_v5"

CRITERIA = ["blur", "shadow", "crease", "faded", "tilt", "speckle"]
OVERALL_LEVELS = ["NONE", "MODERATE", "HEAVY"]

WELL_FORMED = """1. BLUR: YES
2. SHADOW: YES
3. CREASE: NO
4. FADED: YES
5. TILT: NO
6. SPECKLE: YES
7. OVERALL: MODERATE"""


def parse(text: str):
    return parse_quality_response(text, criteria=CRITERIA, overall_levels=OVERALL_LEVELS)


def test_reads_a_well_formed_response():
    result = parse(WELL_FORMED)

    assert result.malformed is False
    assert result.answers == {
        "blur": True,
        "shadow": True,
        "crease": False,
        "faded": True,
        "tilt": False,
        "speckle": True,
    }
    assert result.overall == "MODERATE"


# --------------------------------------------------------------------------
# Malformed responses
# --------------------------------------------------------------------------
# Each of these must be REPORTED as unreadable, never coerced into a plausible
# answer. A reader that turns a refusal into six NOs produces a score that
# looks fine and means nothing.


def test_a_missing_slot_makes_the_whole_record_malformed():
    """Not a per-slot default: if the model skipped a question we do not know
    what it would have said, and guessing NO invents data."""
    missing = "\n".join(line for line in WELL_FORMED.splitlines() if "CREASE" not in line)
    result = parse(missing)

    assert result.malformed is True
    assert "crease" in result.malformed_reason.lower()


def test_an_answer_outside_yes_no_is_malformed():
    """`MAYBE` is not `NO`. Mapping any non-YES token to False would score a
    hedge as a confident negative."""
    hedged = WELL_FORMED.replace("1. BLUR: YES", "1. BLUR: MAYBE")
    result = parse(hedged)

    assert result.malformed is True
    assert "blur" in result.malformed_reason.lower()


def test_an_overall_outside_the_declared_levels_is_malformed():
    result = parse(WELL_FORMED.replace("OVERALL: MODERATE", "OVERALL: QUITE BAD"))

    assert result.malformed is True
    assert "overall" in result.malformed_reason.lower()


def test_an_unexpected_criterion_name_is_malformed():
    """The model inventing a criterion means it is not answering this prompt."""
    result = parse(WELL_FORMED.replace("3. CREASE:", "3. WRINKLE:"))

    assert result.malformed is True


def test_positional_answers_without_labels_are_malformed():
    """The UNIVERSAL failure: the model answered a numbered prompt positionally.
    Reading by position would silently accept it and assign the answers to
    whichever criteria happened to line up."""
    positional = "1. YES\n2. YES\n3. NO\n4. YES\n5. NO\n6. YES\n7. MODERATE"
    result = parse(positional)

    assert result.malformed is True


def test_correctly_labelled_answers_in_the_wrong_slots_are_malformed():
    """The subtler half of the positional failure: the labels are right but the
    numbering disagrees with the prompt, so the model is answering a different
    question order from the one asked."""
    shuffled = "\n".join(
        [
            "1. SHADOW: YES",
            "2. BLUR: YES",
            "3. CREASE: NO",
            "4. FADED: YES",
            "5. TILT: NO",
            "6. SPECKLE: YES",
            "7. OVERALL: MODERATE",
        ]
    )
    result = parse(shuffled)

    assert result.malformed is True
    assert "numbered" in result.malformed_reason


def test_a_refusal_is_malformed():
    result = parse("I'm sorry, I can't help with analysing this image.")

    assert result.malformed is True


def test_an_empty_response_is_malformed():
    result = parse("")

    assert result.malformed is True


def test_a_malformed_response_carries_no_answers():
    """`None`, not an empty dict. An empty dict compares as six wrong answers
    instead of one unreadable response, quietly inflating the error count."""
    result = parse("nonsense")

    assert result.malformed is True
    assert result.answers is None
    assert result.overall is None


# --------------------------------------------------------------------------
# Response shapes seen in practice
# --------------------------------------------------------------------------
# Formatting noise should not be reported as unreadable -- a run whose
# malformed rate is really a markdown-asterisk rate tells us nothing about the
# model's judgement. Hedging is a different matter and stays malformed.


def test_reasoning_before_the_answers_is_flagged_but_still_read():
    """v5 was written to avoid the two known <think> triggers, but that is
    untested against the real model. Measure the rate rather than tolerate it
    silently -- and do not throw away answers that are present."""
    drifted = f"<think>\nThe image looks tilted and a bit dark.\n</think>\n{WELL_FORMED}"
    result = parse(drifted)

    assert result.malformed is False
    assert result.think_drift is True
    assert result.answers["blur"] is True


def test_a_clean_response_is_not_flagged_as_drifting():
    assert parse(WELL_FORMED).think_drift is False


def test_a_trailing_full_stop_is_tolerated():
    result = parse(WELL_FORMED.replace("1. BLUR: YES", "1. BLUR: YES."))

    assert result.malformed is False
    assert result.answers["blur"] is True


def test_markdown_emphasis_is_tolerated():
    result = parse(WELL_FORMED.replace("1. BLUR: YES", "1. BLUR: **YES**"))

    assert result.malformed is False
    assert result.answers["blur"] is True


def test_a_hedged_answer_is_malformed_not_read_as_its_first_word():
    """Guards the tolerance above from going too far. "YES, but the top is
    unclear" is not a YES -- reading the first token and discarding the rest
    would turn every hedge into a confident answer."""
    hedged = WELL_FORMED.replace("1. BLUR: YES", "1. BLUR: YES, but the top is unclear")
    result = parse(hedged)

    assert result.malformed is True


def test_surrounding_prose_does_not_prevent_reading_the_template():
    result = parse(f"Here are my answers:\n\n{WELL_FORMED}\n\nLet me know if you need more.")

    assert result.malformed is False
    assert result.overall == "MODERATE"


# --------------------------------------------------------------------------
# Vocabulary comes from the prompt, never from the caller
# --------------------------------------------------------------------------
# The six criterion names are shared with the corpus generator's defect labels
# and nothing across the two repos enforces that. Hardcoding them at a call
# site would add a third place to keep in step.


def test_vocabulary_is_read_from_the_prompt_config():
    vocabulary = load_screen_vocabulary(PROMPT_CONFIG, variant=VARIANT)

    assert vocabulary.criteria == CRITERIA
    assert vocabulary.overall_levels == OVERALL_LEVELS


def test_vocabulary_exposes_the_answer_that_means_the_defect_is_present():
    """The `evidence:` block records polarity per criterion: the answer value
    that means the defect IS there. It is only meaningful if a consumer can
    read it -- the scorer compared booleans directly and ignored it, which is
    correct only while every question is defect-phrased.
    """
    vocabulary = load_screen_vocabulary(PROMPT_CONFIG, variant=VARIANT)

    assert vocabulary.polarity == dict.fromkeys(CRITERIA, True)


def test_a_good_phrased_criterion_declares_false_polarity(tmp_path):
    """A question asking "is it good?" is answered NO when the defect is
    present, so its evidence value is false."""
    import yaml as _yaml

    path = tmp_path / "mixed.yaml"
    path.write_text(
        _yaml.safe_dump(
            {
                "prompts": {
                    "mixed": {
                        "evidence": {"blur": False, "shadow": True},
                        "overall_levels": OVERALL_LEVELS,
                        "prompt": "x",
                    }
                }
            }
        )
    )

    vocabulary = load_screen_vocabulary(path, variant="mixed")

    assert vocabulary.criteria == ["blur", "shadow"]
    assert vocabulary.polarity == {"blur": False, "shadow": True}


def test_vocabulary_order_matches_the_prompts_question_order():
    """The reader checks slot numbers against this order, so if it disagreed
    with the prompt every well-formed response would be reported unreadable."""
    vocabulary = load_screen_vocabulary(PROMPT_CONFIG, variant=VARIANT)
    prompt_text = vocabulary.prompt

    positions = [prompt_text.upper().index(f"{name.upper()}:") for name in vocabulary.criteria]
    assert positions == sorted(positions), "evidence keys are not in the prompt's question order"


def test_the_shipped_prompt_and_reader_agree_end_to_end():
    """Guards the seam the unit tests cannot: a response built from the real
    prompt's own template must read cleanly."""
    vocabulary = load_screen_vocabulary(PROMPT_CONFIG, variant=VARIANT)
    answers = [f"{i}. {name.upper()}: NO" for i, name in enumerate(vocabulary.criteria, start=1)]
    answers.append(f"{len(vocabulary.criteria) + 1}. OVERALL: {vocabulary.overall_levels[0]}")

    result = parse_quality_response(
        "\n".join(answers),
        criteria=vocabulary.criteria,
        overall_levels=vocabulary.overall_levels,
    )

    assert result.malformed is False, result.malformed_reason
    assert set(result.answers) == set(vocabulary.criteria)


def test_an_unknown_variant_fails_with_a_diagnostic(assert_diagnostic_error):
    with pytest.raises(ScreenVocabularyError) as exc_info:
        load_screen_vocabulary(PROMPT_CONFIG, variant="quality_screen_v99")

    message = str(exc_info.value)
    assert_diagnostic_error(message)
    assert "quality_screen_v99" in message
    assert VARIANT in message, "the error should name the variants that do exist"


def test_a_missing_config_fails_with_a_diagnostic(assert_diagnostic_error, tmp_path):
    with pytest.raises(ScreenVocabularyError) as exc_info:
        load_screen_vocabulary(tmp_path / "absent.yaml", variant=VARIANT)

    assert_diagnostic_error(str(exc_info.value))


@pytest.mark.parametrize("missing_key", ["evidence", "overall_levels", "prompt"])
def test_a_variant_missing_a_required_key_fails_with_a_diagnostic(
    assert_diagnostic_error, tmp_path, missing_key
):
    """A variant without `evidence:` has no vocabulary, and defaulting to the
    six names we happen to expect would let the prompt and the reader disagree
    silently."""
    import yaml as _yaml

    block = {
        "evidence": {name: True for name in CRITERIA},
        "overall_levels": OVERALL_LEVELS,
        "prompt": WELL_FORMED,
    }
    del block[missing_key]
    path = tmp_path / "partial.yaml"
    path.write_text(_yaml.safe_dump({"prompts": {VARIANT: block}}))

    with pytest.raises(ScreenVocabularyError) as exc_info:
        load_screen_vocabulary(path, variant=VARIANT)

    message = str(exc_info.value)
    assert_diagnostic_error(message)
    assert missing_key in message
