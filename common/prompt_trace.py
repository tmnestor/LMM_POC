"""Raw VLM prompt/response trace -- one JSONL line per extraction call.

Single-seam observability: ``VllmBackend`` records every ``generate()`` /
``generate_for_graph()`` call here, so the trace covers *every* extraction
prompt across *every* pipeline (classify, batch, graph-robust, trust,
transaction-link) without per-stage plumbing.

Callers may set per-call context (``image_name``, ``pipeline``, ``label``) via
``trace_context()``; because it uses ``contextvars``, setting ``image_name``
once per image at a stage loop propagates to every nested VLM call (detection,
bank turns, graph nodes).

Disabled by default: with no sink configured, ``record()`` is a cheap no-op.
"""

import contextvars
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_CTX: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("prompt_trace_ctx", default={})
_lock = threading.Lock()
_sink_path: Path | None = None


def enable(path: str | Path) -> None:
    """Enable tracing; each ``record()`` appends a JSONL line to *path*.

    Idempotent: calling again with the same path is a no-op (the file is opened
    per-write in append mode). Creates the parent directory if needed.
    """
    global _sink_path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _sink_path = p


def disable() -> None:
    """Disable tracing (subsequent ``record()`` calls are no-ops)."""
    global _sink_path
    _sink_path = None


def is_enabled() -> bool:
    return _sink_path is not None


@contextmanager
def trace_context(**fields: Any):
    """Set per-call context for nested ``record()`` calls (merges with current).

    Example::

        with trace_context(image_name="CASE012_westpac_premium.png", pipeline="transaction_link"):
            ...  # every VLM call inside inherits these fields
    """
    base = _CTX.get()
    token = _CTX.set({**base, **fields})
    try:
        yield
    finally:
        _CTX.reset(token)


def record(
    *,
    prompt: str,
    response: str,
    model: str = "",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    """Append one trace line. No-op when disabled.

    The ``image_name`` / ``pipeline`` / ``label`` fields come from the current
    ``trace_context()`` (or None when unset).
    """
    if _sink_path is None:
        return
    ctx = _CTX.get()
    row = {
        "image_name": ctx.get("image_name"),
        "pipeline": ctx.get("pipeline"),
        "label": ctx.get("label"),
        "model": model,
        "prompt": prompt,
        "raw_response": response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    line = json.dumps(row, ensure_ascii=False)
    with _lock, _sink_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def effective_trace_path(config: Any) -> str | None:
    """Resolve the trace path for a ``PipelineConfig``-like object, or None.

    Returns None when ``trace_raw_prompts`` is false. When enabled, uses
    ``trace_path`` if set, else ``<output_dir>/raw_prompt_trace.jsonl``.
    """
    if not getattr(config, "trace_raw_prompts", False):
        return None
    if getattr(config, "trace_path", None):
        return str(config.trace_path)
    return str(Path(config.output_dir) / "raw_prompt_trace.jsonl")
