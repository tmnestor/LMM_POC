"""The screen must attribute every traced VLM call to its image.

tests/ is gitignored — local-only.

Locks in the 2026-08-11 finding: ``prompt_trace.trace_context()`` existed and
worked, but had NO production caller, so every trace line carried
``image_name: null`` / ``label: null`` / ``pipeline: null``. A trace you cannot
attribute to an image is close to useless once there is more than one image.

It regressed once already. The finding's original fix put the context in the
DP workers, and deleting those workers in the strip to the standalone screen
took the only callers with them — leaving a trace that still wrote a line per
call, still looked healthy, and named no image. So the guard now sits on
``screen_batch``, which is where the per-image loop actually lives on this
branch, and the last test here asserts a caller exists at all.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from common import prompt_trace
from models.backend import GenerationParams
from models.orchestrator import DocumentOrchestrator


@pytest.fixture(autouse=True)
def _sink(tmp_path: Path):
    """Route the trace to a temp file and always disable afterwards."""
    prompt_trace.enable(str(tmp_path / "trace.jsonl"))
    yield tmp_path / "trace.jsonl"
    prompt_trace.disable()


def _lines(sink: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]


class _FakeBackend:
    """Emits one trace line per call, exactly as VllmBackend._emit_trace does.

    Deliberately does NOT implement ``generate_batch``: that is the live shape
    on this branch (no backend in the repo implements it), so the sequential
    path is the one a real run takes.
    """

    supports_batch = False

    def generate(self, image: object, prompt: str, params: GenerationParams) -> str:
        prompt_trace.record(prompt=prompt, response="1. NO\n7. GOOD")
        return "1. NO\n7. GOOD"


class _Screener:
    """The two orchestrator methods screen_batch needs, and nothing else.

    Building a real DocumentOrchestrator here would drag in the field schema,
    prompt catalogue and response handler — none of which the screen path
    touches — so the method under test is borrowed onto a minimal object.
    """

    screen_batch = DocumentOrchestrator.screen_batch
    supports_batch = False

    def __init__(self) -> None:
        self._backend = _FakeBackend()

    def load_document_image(self, path: str) -> str:
        return f"<image {path}>"

    def generate(self, image: object, prompt: str, max_tokens: int, extra: dict | None = None) -> str:
        return self._backend.generate(image, prompt, GenerationParams(max_tokens=max_tokens))


def _screen(image_names: list[str]) -> list[str]:
    return _Screener().screen_batch([f"/data/{name}" for name in image_names], "PROMPT", 400)


class TestScreenBatchTraceContext:
    def test_each_line_carries_its_image_name(self, _sink: Path) -> None:
        _screen(["CASE001_receipt.png", "CASE001_invoice_heavy.png"])

        assert [row["image_name"] for row in _lines(_sink)] == [
            "CASE001_receipt.png",
            "CASE001_invoice_heavy.png",
        ]

    def test_lines_carry_pipeline_and_label(self, _sink: Path) -> None:
        _screen(["CASE001_receipt.png"])
        row = _lines(_sink)[0]

        assert row["label"] == "quality_screen"
        assert row["pipeline"] == "quality_screen"

    def test_context_does_not_leak_past_the_loop(self, _sink: Path) -> None:
        # After the batch returns, an unrelated record() must not inherit the
        # last image's name — that would silently misattribute later calls.
        _screen(["CASE001_receipt.png"])
        prompt_trace.record(prompt="unrelated", response="x")

        assert _lines(_sink)[-1]["image_name"] is None

    def test_one_line_per_image_none_dropped(self, _sink: Path) -> None:
        names = [f"CASE{i:03d}_receipt.png" for i in range(1, 6)]
        responses = _screen(names)

        assert len(responses) == len(names)
        assert len(_lines(_sink)) == len(names)


def test_trace_context_has_a_production_caller() -> None:
    """The guard against the way this regressed.

    ``trace_context`` sitting unused is not a test failure anywhere else — the
    trace keeps being written, just anonymously. So assert directly that
    something outside prompt_trace itself calls it.
    """
    root = Path(__file__).resolve().parents[2]
    callers = [
        path.relative_to(root).as_posix()
        for directory in ("common", "models", "stages")
        for path in (root / directory).rglob("*.py")
        if "__pycache__" not in path.parts
        and path.name != "prompt_trace.py"
        and "trace_context(" in path.read_text()
    ]

    assert callers, "trace_context has no production caller: every trace line will be anonymous"
