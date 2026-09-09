"""Tests for GraphExecutor tile-budget wiring.

Regression guard for the pre-tiling no-op bug: the transaction-link / trust-link
graph path constructed ``GraphExecutor`` without a tile-budget resolver, so
``NodeGenParams.max_tiles`` stayed ``None`` and the vLLM backend's app-side
pre-tiling never engaged (it silently sent single images at the checkpoint
default of 12 tiles). These tests assert that, given a ``tile_budget_resolver``,
each model-call node receives the ``max_tiles`` budget for its ``image:`` ref —
and that without one the behaviour stays ``None`` (single-image path).
"""

from pathlib import Path

import pytest
from PIL import Image

from common.extraction_types import GenerateResult, NodeGenParams
from common.graph_executor import GraphExecutor


@pytest.fixture
def two_images(tmp_path: Path) -> dict[str, str]:
    """Write two tiny PNGs and return an image-ref -> path mapping."""
    paths: dict[str, str] = {}
    for ref in ("bank_statement", "receipt"):
        p = tmp_path / f"{ref}.png"
        Image.new("RGB", (32, 32), color="white").save(p)
        paths[ref] = str(p)
    return paths


def _capture_executor(captured: dict[str, NodeGenParams]) -> GraphExecutor:
    """Build a GraphExecutor whose generate_fn records params per node.

    Uses ``output_schema`` so the JSON path is taken (no parser needed); the
    captured key is the node prompt, which we make equal to the image ref.
    """

    def generate_fn(image: Image.Image, prompt: str, params: NodeGenParams) -> GenerateResult:
        captured[prompt] = params
        return GenerateResult(text="{}")

    budgets = {
        "bank_statement": {"min_tiles": 12, "max_tiles": 18},
        "receipt": {"min_tiles": 1, "max_tiles": 6},
    }
    return GraphExecutor(
        generate_fn,
        parsers={},
        tile_budget_resolver=lambda ref: budgets.get(ref, {"min_tiles": 1, "max_tiles": 12}),
    )


def _two_node_workflow() -> dict:
    """A cross-image workflow with one model-call node per image ref."""
    return {
        "inputs": ["bank_statement", "receipt"],
        "nodes": {
            "read_bank": {
                "image": "bank_statement",
                "template": "bank_statement",  # prompt == ref, for capture keying
                "output_schema": {"type": "object"},
                "edges": {"ok": "read_receipt"},
            },
            "read_receipt": {
                "image": "receipt",
                "template": "receipt",
                "output_schema": {"type": "object"},
                "edges": {"ok": "done"},
            },
        },
    }


def test_resolver_sets_per_node_tile_budget(two_images: dict[str, str]) -> None:
    """Each node's min/max tiles come from its image ref via the resolver."""
    captured: dict[str, NodeGenParams] = {}
    executor = _capture_executor(captured)

    executor.run(
        document_type="TRANSACTION_LINK",
        definition=_two_node_workflow(),
        images=two_images,
        image_name="pair_000",
    )

    assert captured["bank_statement"].max_tiles == 18
    assert captured["bank_statement"].min_tiles == 12
    assert captured["receipt"].max_tiles == 6
    assert captured["receipt"].min_tiles == 1


def test_token_budget_resolver_sets_max_tokens(two_images: dict[str, str]) -> None:
    """A node's token_budget name resolves to max_tokens via budget_resolver.

    Guards the second half of the same wiring gap: link.py constructed the
    executor without budget_resolver, so token_budget: names silently fell back
    to the 4096 default instead of their YAML values.
    """
    captured: dict[str, NodeGenParams] = {}

    def generate_fn(image: Image.Image, prompt: str, params: NodeGenParams) -> GenerateResult:
        captured[prompt] = params
        return GenerateResult(text="{}")

    executor = GraphExecutor(
        generate_fn,
        parsers={},
        budget_resolver=lambda name: {"receipt_extract": 500, "transaction_match": 2000}[name],
    )

    definition = {
        "inputs": ["bank_statement", "receipt"],
        "nodes": {
            "read_bank": {
                "image": "bank_statement",
                "template": "bank_statement",
                "token_budget": "transaction_match",
                "output_schema": {"type": "object"},
                "edges": {"ok": "read_receipt"},
            },
            "read_receipt": {
                "image": "receipt",
                "template": "receipt",
                "token_budget": "receipt_extract",
                "output_schema": {"type": "object"},
                "edges": {"ok": "done"},
            },
        },
    }

    executor.run(
        document_type="TRANSACTION_LINK",
        definition=definition,
        images=two_images,
        image_name="pair_000",
    )

    assert captured["bank_statement"].max_tokens == 2000
    assert captured["receipt"].max_tokens == 500


def test_no_resolver_leaves_max_tiles_none(two_images: dict[str, str]) -> None:
    """Without a resolver, max_tiles stays None (single-image path)."""
    captured: dict[str, NodeGenParams] = {}

    def generate_fn(image: Image.Image, prompt: str, params: NodeGenParams) -> GenerateResult:
        captured[prompt] = params
        return GenerateResult(text="{}")

    executor = GraphExecutor(generate_fn, parsers={})  # no tile_budget_resolver

    executor.run(
        document_type="TRANSACTION_LINK",
        definition=_two_node_workflow(),
        images=two_images,
        image_name="pair_000",
    )

    assert captured["bank_statement"].max_tiles is None
    assert captured["bank_statement"].min_tiles == 1
    assert captured["receipt"].max_tiles is None
    assert captured["receipt"].min_tiles == 1
