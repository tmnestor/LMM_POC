"""Tests for the raw-prompt trace sink (common/prompt_trace.py).

tests/ is gitignored — local-only. Covers enable/disable no-op behaviour, JSONL
record shape, contextvar labels, and effective_trace_path resolution.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import prompt_trace


@pytest.fixture(autouse=True)
def _reset_trace():
    prompt_trace.disable()
    yield
    prompt_trace.disable()


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestSink:
    def test_disabled_is_noop(self, tmp_path: Path) -> None:
        prompt_trace.record(prompt="p", response="r")
        assert not list(tmp_path.iterdir())  # nothing written

    def test_enable_records_jsonl(self, tmp_path: Path) -> None:
        sink = tmp_path / "trace.jsonl"
        prompt_trace.enable(sink)
        prompt_trace.record(
            prompt="P", response="R", model="internvl3-vllm", prompt_tokens=10, completion_tokens=2
        )
        rows = _lines(sink)
        assert len(rows) == 1
        row = rows[0]
        assert row["prompt"] == "P" and row["raw_response"] == "R"
        assert row["model"] == "internvl3-vllm"
        assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 2
        assert row["image_name"] is None and row["label"] is None  # no context set

    def test_appends(self, tmp_path: Path) -> None:
        sink = tmp_path / "trace.jsonl"
        prompt_trace.enable(sink)
        prompt_trace.record(prompt="a", response="1")
        prompt_trace.record(prompt="b", response="2")
        assert len(_lines(sink)) == 2

    def test_disable_stops_recording(self, tmp_path: Path) -> None:
        sink = tmp_path / "trace.jsonl"
        prompt_trace.enable(sink)
        prompt_trace.record(prompt="a", response="1")
        prompt_trace.disable()
        prompt_trace.record(prompt="b", response="2")
        assert len(_lines(sink)) == 1


class TestContext:
    def test_context_fields_flow_into_record(self, tmp_path: Path) -> None:
        sink = tmp_path / "trace.jsonl"
        prompt_trace.enable(sink)
        with prompt_trace.trace_context(
            image_name="CASE012_westpac_premium.png", pipeline="transaction_link"
        ):
            with prompt_trace.trace_context(label="bank.turn1_debit_credit"):
                prompt_trace.record(prompt="p", response="r")
        row = _lines(sink)[0]
        assert row["image_name"] == "CASE012_westpac_premium.png"
        assert row["pipeline"] == "transaction_link"
        assert row["label"] == "bank.turn1_debit_credit"

    def test_context_resets_after_block(self, tmp_path: Path) -> None:
        sink = tmp_path / "trace.jsonl"
        prompt_trace.enable(sink)
        with prompt_trace.trace_context(image_name="x.png"):
            pass
        prompt_trace.record(prompt="p", response="r")
        assert _lines(sink)[0]["image_name"] is None


class TestEffectiveTracePath:
    def test_disabled_returns_none(self) -> None:
        cfg = SimpleNamespace(trace_raw_prompts=False, trace_path=None, output_dir="/out")
        assert prompt_trace.effective_trace_path(cfg) is None

    def test_explicit_path_wins(self) -> None:
        cfg = SimpleNamespace(trace_raw_prompts=True, trace_path="/x/t.jsonl", output_dir="/out")
        assert prompt_trace.effective_trace_path(cfg) == "/x/t.jsonl"

    def test_default_under_output_dir(self) -> None:
        cfg = SimpleNamespace(trace_raw_prompts=True, trace_path=None, output_dir="/out")
        assert prompt_trace.effective_trace_path(cfg) == "/out/raw_prompt_trace.jsonl"
