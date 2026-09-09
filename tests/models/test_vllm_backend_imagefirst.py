from PIL import Image

from models.backends.vllm_backend import VllmBackend


def _backend() -> VllmBackend:
    # engine is never called by _build_messages; pre-tiling OFF -> single image part.
    return VllmBackend(engine=object(), pre_tiling_enabled=False)


def test_build_messages_image_first_puts_image_before_text():
    msgs = _backend()._build_messages(Image.new("RGB", (8, 8), "white"), "QUESTION", image_first=True)
    content = msgs[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1]["type"] == "text"
    assert content[-1]["text"] == "QUESTION"


def test_build_messages_text_first_is_default():
    msgs = _backend()._build_messages(Image.new("RGB", (8, 8), "white"), "QUESTION")
    content = msgs[0]["content"]
    assert content[0]["type"] == "text"
    assert content[-1]["type"] == "image_url"
