"""Read a model's answers to the image-quality screen prompt.

The prompt asks eight numbered questions -- six YES/NO defect criteria and one
graded OVERALL -- and this turns the response text back into those answers.

The reader is deliberately strict. A closed answer space does not guarantee a
closed response: the model can refuse, hedge, reason aloud, skip a question, or
answer positionally without labels. Every one of those is reported as
unreadable rather than coerced into a plausible answer, because a reader that
turns a refusal into six NOs produces a score that looks fine and means
nothing.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Captures the whole remainder of the line, not just its first token. Reading
# only the first word would turn "YES, but the top is unclear" into a confident
# YES -- a hedge is not an answer, and must stay unreadable.
_ANSWER_LINE = re.compile(r"^\s*(\d+)\.\s*([A-Za-z]+)\s*:\s*(.+?)\s*$", re.MULTILINE)
_YES_NO = {"YES": True, "NO": False}

_THINK = re.compile(r"<think>|</think>", re.IGNORECASE)


@dataclass(frozen=True)
class QualityResponse:
    """One image's screen result.

    Attributes:
        answers: Criterion name -> whether the model said the defect is
            present. `None` when the response could not be read -- not an empty
            dict, which a scorer would compare as six wrong answers rather than
            one unreadable response.
        overall: The graded OVERALL answer, or `None` when unreadable.
        malformed: Whether the response could not be read.
        malformed_reason: What made it unreadable, for auditing without
            re-running inference.
        think_drift: Whether the response carried reasoning markers. Recorded
            rather than rejected: the answers may still be present and
            readable, and the rate is what tells us whether the prompt is
            holding.
    """

    answers: dict[str, bool] | None
    overall: str | None
    malformed: bool
    malformed_reason: str | None = None
    think_drift: bool = False


class ScreenVocabularyError(RuntimeError):
    """Raised when the screen's prompt config cannot supply a vocabulary."""


@dataclass(frozen=True)
class ScreenVocabulary:
    """What the prompt asks, read from the prompt itself.

    Attributes:
        criteria: Criterion names in the prompt's question order. The reader
            checks slot numbers against this order.
        polarity: Criterion name -> the answer that means the defect IS
            present. True for a defect-phrased question ("is it blurry?"),
            False for a good-phrased one ("is it perfectly sharp?"). A scorer
            comparing the model's boolean straight against the corpus label is
            correct only while every question is defect-phrased.
        overall_levels: Permitted OVERALL answers.
        condition_to_level: Corpus condition -> this variant's severity level,
            or None to fall back to run_config. A variant that renames its
            levels must bring its own mapping, or the scorer compares two
            different vocabularies and every severity call reads as wrong.
        prompt: The prompt text, so a caller can send it and a test can check
            it against the vocabulary.
    """

    criteria: list[str]
    polarity: dict[str, bool]
    overall_levels: list[str]
    prompt: str
    condition_to_level: dict[str, str] | None = None


def load_screen_vocabulary(config_path: Path, *, variant: str) -> ScreenVocabulary:
    """Read one screen variant's vocabulary from the prompt config.

    The criterion names are shared with the corpus generator's defect labels,
    and nothing across the two repositories enforces that. Reading them from
    the prompt keeps this side to one source instead of two.

    Args:
        config_path: Path to the quality-screen prompt YAML.
        variant: Key under `prompts:` to load, e.g. "quality_screen_v5".

    Returns:
        The variant's criteria, permitted OVERALL levels, and prompt text.

    Raises:
        ScreenVocabularyError: The file is missing, or the variant is absent or
            incomplete.
    """
    if not config_path.exists():
        raise ScreenVocabularyError(
            f"Quality-screen prompt config not found.\n"
            f"  What:        no file at the configured prompt path.\n"
            f"  Where:       {config_path.resolve()}\n"
            f"  Expected:    a YAML file with a 'prompts:' block declaring '{variant}'.\n"
            f"  How to fix:  point the prompt path at prompts/quality_screen.yaml, or create it."
        )

    data = yaml.safe_load(config_path.read_text()) or {}
    prompts = data.get("prompts") or {}

    if variant not in prompts:
        available = sorted(prompts)
        raise ScreenVocabularyError(
            f"Quality-screen prompt variant not found.\n"
            f"  What:        '{variant}' is not declared in the prompt config.\n"
            f"  Where:       {config_path.resolve()} -> prompts.{variant}\n"
            f"  Expected:    one of the declared variants: {available}.\n"
            f"  How to fix:  set the variant to one of those, or add a "
            f"'{variant}:' block under 'prompts:'."
        )

    block = prompts[variant]
    for key in ("evidence", "overall_levels", "prompt"):
        if key not in block:
            raise ScreenVocabularyError(
                f"Quality-screen prompt variant is incomplete.\n"
                f"  What:        '{variant}' has no '{key}:' key, so its vocabulary is unknown.\n"
                f"  Where:       {config_path.resolve()} -> prompts.{variant}.{key}\n"
                f"  Expected:    'evidence:' (criterion names), 'overall_levels:' and 'prompt:'.\n"
                f"  How to fix:  add '{key}:' to prompts.{variant}."
            )

    return ScreenVocabulary(
        criteria=list(block["evidence"]),
        polarity={name: bool(value) for name, value in block["evidence"].items()},
        overall_levels=list(block["overall_levels"]),
        prompt=str(block["prompt"]),
        condition_to_level=(dict(block["condition_to_level"]) if block.get("condition_to_level") else None),
    )


def _unreadable(reason: str, *, think_drift: bool) -> QualityResponse:
    """Build the single shape every failure returns."""
    return QualityResponse(
        answers=None,
        overall=None,
        malformed=True,
        malformed_reason=reason,
        think_drift=think_drift,
    )


def parse_quality_response(text: str, *, criteria: list[str], overall_levels: list[str]) -> QualityResponse:
    """Read one response.

    Args:
        text: The model's raw response.
        criteria: Expected criterion names, lowercase, in prompt order. Taken
            from the prompt config rather than hardcoded, so the vocabulary has
            one source.
        overall_levels: Permitted OVERALL values.

    Returns:
        The answers read from the response, or an unreadable result naming what
        failed. Never a partial answer set.
    """
    expected = [name.upper() for name in criteria] + ["OVERALL"]
    think_drift = bool(_THINK.search(text))

    found: dict[str, tuple[int, str]] = {}
    for number, label, value in _ANSWER_LINE.findall(text):
        # Case-folded, not rewritten. Comparing YES to yes is matching the same
        # word; anything else the model adds -- punctuation, markdown, a
        # qualifier -- is reported rather than tidied away.
        found[label.upper()] = (int(number), value.upper())

    if not found:
        return _unreadable("no numbered answer lines found in the response", think_drift=think_drift)

    unknown = sorted(set(found) - set(expected))
    if unknown:
        return _unreadable(
            f"response answers criteria this prompt does not ask: {unknown}", think_drift=think_drift
        )

    missing = [label for label in expected if label not in found]
    if missing:
        return _unreadable(
            f"response is missing an answer for: {[m.lower() for m in missing]}",
            think_drift=think_drift,
        )

    # Slot numbers must match the prompt's order. Reading by position alone is
    # how a positionally-answered numbered prompt gets silently accepted, with
    # answers assigned to whichever criteria happened to line up.
    for position, label in enumerate(expected, start=1):
        number, _value = found[label]
        if number != position:
            return _unreadable(
                f"{label.lower()} is numbered {number} but this prompt asks it as {position}",
                think_drift=think_drift,
            )

    answers: dict[str, bool] = {}
    for name in criteria:
        _number, value = found[name.upper()]
        if value not in _YES_NO:
            return _unreadable(
                f"{name} answered {value!r}, which is neither YES nor NO", think_drift=think_drift
            )
        answers[name] = _YES_NO[value]

    _number, overall = found["OVERALL"]
    if overall not in overall_levels:
        return _unreadable(
            f"overall answered {overall!r}, which is not one of {overall_levels}",
            think_drift=think_drift,
        )

    return QualityResponse(answers=answers, overall=overall, malformed=False, think_drift=think_drift)
