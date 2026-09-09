"""The data-parallel wiring must be checkable without a GPU.

``run_dp`` resolves the worker by dotted string inside a subprocess, and
passes it keyword arguments the stage chooses. Both are the kind of
mistake that stays invisible until minutes into a GPU run.
"""

import importlib
import inspect

import stages.sroie
from common.sroie.runner import benchmark_from_worker_responses


def _resolve(dotted: str):
    module_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def test_the_worker_dotted_path_resolves() -> None:
    worker = _resolve(stages.sroie.SROIE_WORKER_FN)

    assert callable(worker)


def test_the_worker_accepts_exactly_the_kwargs_the_stage_sends() -> None:
    """run_dp forwards worker_kwargs verbatim; a name mismatch is a
    TypeError in the subprocess, long after the model has loaded."""
    worker = _resolve(stages.sroie.SROIE_WORKER_FN)
    params = inspect.signature(worker).parameters

    assert list(params)[:2] == ["gpu_id", "image_paths"]
    for sent in ("config_path", "cli_overrides", "max_new_tokens", "tile_budget", "batch_size"):
        assert sent in params, f"worker cannot accept {sent!r}"
        assert params[sent].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_worker_return_shape_is_what_reassembly_expects() -> None:
    """The worker's dict keys and benchmark_from_worker_responses must
    agree, or every image silently becomes a 'no response' error."""
    from pathlib import Path

    from common.sroie.ground_truth import SroieRecord

    record = SroieRecord(
        image_id="X001",
        image_path=Path("X001.jpg"),
        company="ACME",
        date="15/01/2019",
        address="27 JALAN",
        total="9.00",
    )
    worker_shaped = [
        {
            "image_id": "X001",
            "image_path": "X001.jpg",
            "raw_response": "company: ACME\ntotal: 9.00",
            "error": None,
        }
    ]

    run = benchmark_from_worker_responses([record], worker_shaped)

    assert run.errors == {}
    assert run.predictions["X001"] == {"company": "ACME", "total": "9.00"}
