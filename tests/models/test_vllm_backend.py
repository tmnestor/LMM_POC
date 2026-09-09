"""Tests for VllmBackend message construction (no system prompt + ordering + pre-tiling).

tests/ is gitignored — local-only. Locks in the 2026-06-04 finding that on the
InternVL3.5 family sending ANY system message turns thinking ON; detection must
send a bare user message. Also covers image/text ordering, the chat_template
override knob, and app-side pre-tiling (plans/2026-06-04-adaptive-pre-tiling.md).
No vLLM engine needed — message construction is pure.
"""

from PIL import Image

from models.backends.vllm_backend import VllmBackend

_IMG = Image.new("RGB", (8, 8))
_WIDE = Image.new("RGB", (1344, 448))  # 3:1 -> 3 detail tiles at 448


def _backend(model_type_key: str, **kw: object) -> VllmBackend:
    return VllmBackend(None, model_type_key=model_type_key, **kw)


def _image_parts(content: list[dict]) -> list[dict]:
    return [p for p in content if p["type"] == "image_url"]


class TestNoSystemMessage:
    def test_internvl3_sends_bare_user_message(self) -> None:
        # Critical: a system role here triggers <think> reasoning on InternVL3.5.
        messages = _backend("internvl3")._build_messages(_IMG, "PROMPT")
        assert [m["role"] for m in messages] == ["user"]

    def test_other_models_also_have_no_system_role(self) -> None:
        messages = _backend("qwen35")._build_messages(_IMG, "PROMPT")
        assert [m["role"] for m in messages] == ["user"]


class TestMessageOrdering:
    def test_classification_is_text_first(self) -> None:
        messages = _backend("internvl3")._build_messages(_IMG, "PROMPT")
        content = messages[0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_extraction_is_image_first(self) -> None:
        messages = _backend("internvl3")._build_messages(_IMG, "PROMPT", image_first=True)
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"


class TestChatTemplateOverride:
    def test_defaults_to_none(self) -> None:
        assert _backend("internvl3")._chat_template is None

    def test_stores_provided_template(self) -> None:
        backend = _backend("internvl3", chat_template="/tmp/no_think.jinja")
        assert backend._chat_template == "/tmp/no_think.jinja"


class TestPreTiling:
    def test_disabled_sends_single_image_even_with_max_tiles(self) -> None:
        backend = _backend("internvl3")  # pre_tiling_enabled defaults False
        parts = backend._image_parts(_WIDE, max_tiles=6)
        assert len(parts) == 1

    def test_enabled_but_no_max_tiles_sends_single_image(self) -> None:
        backend = _backend("internvl3", pre_tiling_enabled=True)
        parts = backend._image_parts(_WIDE, max_tiles=None)
        assert len(parts) == 1

    def test_enabled_with_max_tiles_emits_one_part_per_tile(self) -> None:
        backend = _backend("internvl3", pre_tiling_enabled=True)
        # 3:1 image, budget 6, thumbnail on -> 3 detail tiles + 1 thumbnail.
        parts = backend._image_parts(_WIDE, max_tiles=6)
        assert len(parts) == 4
        assert all(p["type"] == "image_url" for p in parts)

    def test_thumbnail_toggle_respected(self) -> None:
        backend = _backend("internvl3", pre_tiling_enabled=True, tile_use_thumbnail=False)
        parts = backend._image_parts(_WIDE, max_tiles=6)
        assert len(parts) == 3

    def test_build_messages_keeps_all_tiles_before_text_when_image_first(self) -> None:
        backend = _backend("internvl3", pre_tiling_enabled=True)
        messages = backend._build_messages(_WIDE, "PROMPT", image_first=True, max_tiles=6)
        content = messages[0]["content"]
        # All image parts precede the single trailing text part.
        assert content[-1]["type"] == "text"
        assert len(_image_parts(content)) == 4

    def test_build_messages_text_first_keeps_text_leading(self) -> None:
        backend = _backend("internvl3", pre_tiling_enabled=True)
        messages = backend._build_messages(_WIDE, "PROMPT", max_tiles=6)
        content = messages[0]["content"]
        assert content[0]["type"] == "text"
        assert len(_image_parts(content)) == 4
