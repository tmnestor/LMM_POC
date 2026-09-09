"""Guards for the fix_mistral_regex tokenizer cache location.

The baked tokenizer must land somewhere writable. ``entrypoint.sh`` exports
``LMM_TOKENIZER_CACHE`` to ``<run output dir>/tokenizer_cache`` so it works in
the KFP prod pod, where ``~/.cache`` is read-only and only the run_config.yml
output directory is writable. These tests pin the env-var contract that fix
relies on. Local-only (``tests/`` gitignored), CPU-only: a fake tokenizer
stands in for the real one so no model weights are needed.
"""

from pathlib import Path

import transformers

from models import model_loader


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def save_pretrained(self, dest):
        Path(dest, "tokenizer_config.json").write_text("{}")


def test_cache_honors_env_var(monkeypatch, tmp_path):
    """LMM_TOKENIZER_CACHE redirects the baked tokenizer off ~/.cache."""
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setenv("LMM_TOKENIZER_CACHE", str(tmp_path))

    out = model_loader.ensure_corrected_tokenizer("/nfs/models/InternVL3_5-8B")

    assert Path(out) == tmp_path / "InternVL3_5-8B"
    assert (Path(out) / "tokenizer_config.json").exists()
    assert str(Path.home() / ".cache") not in out


def test_cache_reused_when_already_present(monkeypatch, tmp_path):
    """An already-baked tokenizer dir is returned without re-baking (idempotent)."""
    cached = tmp_path / "InternVL3_5-8B"
    cached.mkdir(parents=True)
    (cached / "tokenizer_config.json").write_text("{}")

    class _Boom:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("should not re-bake when the cache already exists")

    monkeypatch.setattr(transformers, "AutoTokenizer", _Boom)
    monkeypatch.setenv("LMM_TOKENIZER_CACHE", str(tmp_path))

    out = model_loader.ensure_corrected_tokenizer("/nfs/models/InternVL3_5-8B")

    assert Path(out) == cached
