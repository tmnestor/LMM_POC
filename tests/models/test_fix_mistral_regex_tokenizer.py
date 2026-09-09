"""Tests for ensure_corrected_tokenizer (fix_mistral_regex baked to disk).

tests/ is gitignored — local-only. vLLM loads its tokenizer in both the
front-end AND every spawned EngineCore child, so a load-time patch can't reach
the child. Instead we save a fix_mistral_regex-corrected tokenizer to a cache
dir once and hand vLLM that path. This locks: the flag is passed, the corrected
copy is cached + reused, and concurrent callers don't re-load. See
plans/2026-06-04-adaptive-tiling-dense-bank.md.
"""

from pathlib import Path

import transformers

from models.model_loader import ensure_corrected_tokenizer


class _FakeTokenizer:
    """Stand-in whose save_pretrained writes the marker file vLLM looks for."""

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer_config.json").write_text("{}")


def test_creates_corrected_copy_with_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LMM_TOKENIZER_CACHE", str(tmp_path))
    calls: list[dict] = []

    def _fake(*args, **kwargs):
        calls.append(kwargs)
        return _FakeTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fake)

    out = ensure_corrected_tokenizer("/models/InternVL3_5-8B")

    assert out == str(tmp_path / "InternVL3_5-8B")
    assert (Path(out) / "tokenizer_config.json").exists()
    assert calls[0]["fix_mistral_regex"] is True


def test_reuses_cache_without_reloading(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LMM_TOKENIZER_CACHE", str(tmp_path))
    calls: list[dict] = []
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: calls.append(k) or _FakeTokenizer(),
    )

    first = ensure_corrected_tokenizer("/models/InternVL3_5-8B")
    second = ensure_corrected_tokenizer("/models/InternVL3_5-8B")

    assert first == second
    assert len(calls) == 1  # second call served from cache, no reload


def test_no_temp_dir_left_behind(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LMM_TOKENIZER_CACHE", str(tmp_path))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer())

    ensure_corrected_tokenizer("/models/InternVL3_5-8B")

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftovers == []
