"""The DP workers must attribute every traced VLM call to its image.

tests/ is gitignored — local-only.

Locks in the 2026-08-11 finding: ``prompt_trace.trace_context()`` existed and
worked, but had NO production caller, so every trace line carried
``image_name: null`` / ``label: null`` / ``pipeline: null``. A trace you cannot
attribute to an image is close to useless once there is more than one image, or
when one image makes several VLM calls (bank turns, graph nodes).

These tests patch the worker's four dependencies and assert the trace lines the
processor emits inherit the surrounding context.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from common import prompt_trace


@pytest.fixture(autouse=True)
def _sink(tmp_path: Path):
    """Route the trace to a temp file and always disable afterwards."""
    prompt_trace.enable(str(tmp_path / "trace.jsonl"))
    yield tmp_path / "trace.jsonl"
    prompt_trace.disable()


def _lines(sink: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]


class _FakeCM:
    """Stand-in for the load_model context manager."""

    def __enter__(self) -> tuple[object, object]:
        return object(), object()

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeProcessor:
    """Emits one trace line per call, exactly as VllmBackend._emit_trace does."""

    def detect_and_classify_document(self, image_path: str, verbose: bool = False) -> dict[str, Any]:
        prompt_trace.record(prompt="detect", response="1. NONE\n2. YES\n3. 2\n4. NO")
        return {
            "document_type": "RECEIPT",
            "confidence": 1.0,
            "raw_response": "1. NONE\n2. YES\n3. 2\n4. NO",
            "prompt_used": "detection",
        }


class _FakeConfig:
    model_type = "gemma4-12b-unified-w4a16-vllm"
    verbose = False


class _FakeAppConfig:
    pipeline = _FakeConfig()


def _run_classify_worker(image_names: list[str]) -> None:
    from common import vllm_dp_workers

    with (
        patch("common.app_config.AppConfig.load", return_value=_FakeAppConfig()),
        patch("cli.load_pipeline_configs", return_value=({}, [], {})),
        patch("common.pipeline_ops.load_model", return_value=_FakeCM()),
        patch("common.pipeline_ops.create_processor", return_value=_FakeProcessor()),
    ):
        vllm_dp_workers.classify_worker(
            gpu_id=0,
            image_paths=[f"/data/{n}" for n in image_names],
            config_path=None,
            cli_overrides={},
        )


class TestClassifyWorkerTraceContext:
    def test_each_line_carries_its_image_name(self, _sink: Path) -> None:
        _run_classify_worker(["CASE001_bank_statement.png", "CASE001_receipt.png"])
        rows = _lines(_sink)
        assert [r["image_name"] for r in rows] == [
            "CASE001_bank_statement.png",
            "CASE001_receipt.png",
        ]

    def test_lines_carry_pipeline_and_label(self, _sink: Path) -> None:
        _run_classify_worker(["CASE001_receipt.png"])
        row = _lines(_sink)[0]
        assert row["label"] == "classify"
        assert row["pipeline"] == "information_extraction"

    def test_context_does_not_leak_past_the_loop(self, _sink: Path) -> None:
        # After the worker returns, an unrelated record() must not inherit the
        # last image's name — that would silently misattribute later calls.
        _run_classify_worker(["CASE001_receipt.png"])
        prompt_trace.record(prompt="unrelated", response="x")
        assert _lines(_sink)[-1]["image_name"] is None
