"""VllmBackend must forward per-model chat_template_kwargs to engine.chat().

tests/ is gitignored — local-only. Gemma 4's template reasons by default and
honours enable_thinking (unlike InternVL3.5, where thinking is driven by the
system prompt). Without these kwargs reaching the engine, <think> blocks land in
extraction output.

See plans/2026-07-27-reintegrate-gemma4-vllm.md (gaps G1 + G4).
"""

from typing import Any

from PIL import Image

from models.backend import GenerationParams
from models.backends.vllm_backend import VllmBackend

_IMG = Image.new("RGB", (8, 8))


class _RecordingEngine:
    """Captures the kwargs of the last chat() call and returns a canned output."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)

        class _Completion:
            text = "RESPONSE"
            token_ids = [1, 2, 3]
            logprobs = None

        class _Output:
            outputs = [_Completion()]
            prompt_token_ids = [1, 2]
            num_cached_tokens = 0

        return [_Output()]

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


def _params(**extra: Any) -> GenerationParams:
    return GenerationParams(max_tokens=16, extra=extra)


class TestChatTemplateKwargs:
    def test_forwarded_when_configured(self) -> None:
        engine = _RecordingEngine()
        backend = VllmBackend(engine, chat_template_kwargs={"enable_thinking": False})
        backend.generate(_IMG, "PROMPT", _params())
        assert engine.last["chat_template_kwargs"] == {"enable_thinking": False}

    def test_absent_when_not_configured(self) -> None:
        # InternVL must keep the exact call shape it had before this change.
        engine = _RecordingEngine()
        VllmBackend(engine).generate(_IMG, "PROMPT", _params())
        assert "chat_template_kwargs" not in engine.last

    def test_forwarded_on_the_graph_path_too(self) -> None:
        from common.extraction_types import NodeGenParams

        engine = _RecordingEngine()
        backend = VllmBackend(engine, chat_template_kwargs={"enable_thinking": False})
        backend.generate_for_graph(_IMG, "PROMPT", NodeGenParams(max_tokens=16))
        assert engine.last["chat_template_kwargs"] == {"enable_thinking": False}

    def test_mutating_the_returned_kwargs_cannot_corrupt_the_backend(self) -> None:
        backend = VllmBackend(_RecordingEngine(), chat_template_kwargs={"enable_thinking": False})
        backend._chat_kwargs()["chat_template_kwargs"]["enable_thinking"] = True
        assert backend._chat_template_kwargs == {"enable_thinking": False}


class TestDefaultImageFirst:
    def test_default_image_first_puts_image_before_text(self) -> None:
        engine = _RecordingEngine()
        backend = VllmBackend(engine, default_image_first=True)
        backend.generate(_IMG, "PROMPT", _params())
        content = engine.last["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[-1]["type"] == "text"

    def test_text_first_remains_the_default(self) -> None:
        engine = _RecordingEngine()
        VllmBackend(engine).generate(_IMG, "PROMPT", _params())
        content = engine.last["messages"][0]["content"]
        assert content[0]["type"] == "text"

    def test_explicit_extra_overrides_the_model_default(self) -> None:
        # Callers that deliberately ask for text-first must still win.
        engine = _RecordingEngine()
        backend = VllmBackend(engine, default_image_first=True)
        backend.generate(_IMG, "PROMPT", _params(image_first=False))
        content = engine.last["messages"][0]["content"]
        assert content[0]["type"] == "text"

    def test_explicit_extra_can_also_force_image_first(self) -> None:
        engine = _RecordingEngine()
        VllmBackend(engine).generate(_IMG, "PROMPT", _params(image_first=True))
        content = engine.last["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
