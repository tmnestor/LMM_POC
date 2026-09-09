"""VllmBackend.generate_batch must submit one batched chat() call.

The ModelBackend protocol has declared generate_batch since the backend
was written, but VllmBackend never implemented it — so every stage fell
back to one request at a time and vLLM's continuous batching never
engaged. With max_num_seqs > 1 this is the difference between using one
sequence slot and filling the scheduler.

vllm is not installed locally, so SamplingParams is stubbed.
"""

import sys
import types
from typing import Any

import pytest
from PIL import Image

from models.backend import GenerationParams
from models.backends.vllm_backend import VllmBackend

_IMG = Image.new("RGB", (8, 8))


@pytest.fixture(autouse=True)
def _stub_vllm(monkeypatch):
    """Provide a minimal vllm module so the backend can import it."""
    module = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


class _RecordingEngine:
    """Captures chat() calls and returns one canned output per conversation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        # A batch submits a list of conversations; a single call submits one.
        count = len(messages) if messages and isinstance(messages[0], list) else 1

        def _output(index: int) -> Any:
            completion = types.SimpleNamespace(text=f"RESPONSE {index}", token_ids=[1], logprobs=None)
            return types.SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2], num_cached_tokens=0)

        return [_output(i) for i in range(count)]


def _backend(engine: _RecordingEngine) -> VllmBackend:
    return VllmBackend(engine, model_type_key="internvl3")


def test_a_batch_is_one_engine_call_not_several() -> None:
    """The whole point: N images must reach the scheduler together."""
    engine = _RecordingEngine()

    _backend(engine).generate_batch(
        [_IMG, _IMG, _IMG],
        ["a", "b", "c"],
        GenerationParams(max_tokens=64),
    )

    assert len(engine.calls) == 1


def test_a_batch_submits_one_conversation_per_image() -> None:
    engine = _RecordingEngine()

    _backend(engine).generate_batch(
        [_IMG, _IMG, _IMG],
        ["a", "b", "c"],
        GenerationParams(max_tokens=64),
    )

    submitted = engine.calls[0]["messages"]
    assert len(submitted) == 3
    assert all(isinstance(conversation, list) for conversation in submitted)


def test_responses_come_back_in_submission_order() -> None:
    """Order is how each response is matched to its receipt; if vLLM ever
    reordered, every prediction would attach to the wrong image."""
    engine = _RecordingEngine()

    texts = _backend(engine).generate_batch(
        [_IMG, _IMG, _IMG],
        ["a", "b", "c"],
        GenerationParams(max_tokens=64),
    )

    assert texts == ["RESPONSE 0", "RESPONSE 1", "RESPONSE 2"]


def test_mismatched_images_and_prompts_is_an_error() -> None:
    """Silently truncating would score responses against the wrong images."""
    engine = _RecordingEngine()

    with pytest.raises(ValueError):
        _backend(engine).generate_batch([_IMG, _IMG], ["only one"], GenerationParams(max_tokens=64))


def test_an_empty_batch_does_not_call_the_engine() -> None:
    engine = _RecordingEngine()

    assert _backend(engine).generate_batch([], [], GenerationParams(max_tokens=64)) == []
    assert engine.calls == []
