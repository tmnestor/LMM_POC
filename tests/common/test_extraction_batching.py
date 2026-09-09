"""Tests for grouping classified documents into extraction batches.

One engine call carries ONE GenerationParams, so every document in a batch
must share a tile budget — which means batching strictly within a document
type. classify runs first, so the type is already known here.
"""

from common.extraction_batching import plan_extraction_batches

_SIZES = {"receipt": 4, "invoice": 2, "bank_statement": 2, "default": 2}
_UNBATCHABLE = {"BANK_STATEMENT"}


def _docs(*types: str) -> list[dict]:
    return [{"image_name": f"{i}.png", "document_type": t} for i, t in enumerate(types)]


def _names(batches: list[list[dict]]) -> list[list[str]]:
    return [[d["image_name"] for d in b] for b in batches]


def test_same_type_fills_up_to_its_batch_size() -> None:
    batches = plan_extraction_batches(
        _docs("RECEIPT", "RECEIPT", "RECEIPT", "RECEIPT", "RECEIPT"),
        batch_sizes=_SIZES,
        unbatchable=_UNBATCHABLE,
    )

    assert _names(batches) == [["0.png", "1.png", "2.png", "3.png"], ["4.png"]]


def test_a_type_change_starts_a_new_batch() -> None:
    """A mixed batch could not carry one tile budget."""
    batches = plan_extraction_batches(
        _docs("RECEIPT", "RECEIPT", "INVOICE", "INVOICE"),
        batch_sizes=_SIZES,
        unbatchable=_UNBATCHABLE,
    )

    assert _names(batches) == [["0.png", "1.png"], ["2.png", "3.png"]]


def test_every_batch_holds_exactly_one_document_type() -> None:
    batches = plan_extraction_batches(
        _docs("RECEIPT", "INVOICE", "RECEIPT", "INVOICE"),
        batch_sizes=_SIZES,
        unbatchable=_UNBATCHABLE,
    )

    for batch in batches:
        assert len({d["document_type"] for d in batch}) == 1


def test_unbatchable_types_come_back_as_singletons() -> None:
    """Bank statements run multi-turn through UnifiedBankExtractor, where
    turn N depends on turn N-1. They must never share a call."""
    batches = plan_extraction_batches(
        _docs("BANK_STATEMENT", "BANK_STATEMENT", "BANK_STATEMENT"),
        batch_sizes=_SIZES,
        unbatchable=_UNBATCHABLE,
    )

    assert _names(batches) == [["0.png"], ["1.png"], ["2.png"]]


def test_order_is_preserved_exactly() -> None:
    """classifications arrive already sorted by sort_for_extraction, which
    honours extraction_order and secondary_sort. Batching groups CONSECUTIVE
    runs so that ordering survives — and so the bank header cache still sees
    each bank's statements together."""
    docs = _docs("BANK_STATEMENT", "RECEIPT", "RECEIPT", "INVOICE", "RECEIPT")

    batches = plan_extraction_batches(docs, batch_sizes=_SIZES, unbatchable=_UNBATCHABLE)

    flattened = [d for batch in batches for d in batch]
    assert flattened == docs


def test_an_unknown_type_uses_the_default_size() -> None:
    batches = plan_extraction_batches(
        _docs("TRAVEL_EXPENSE", "TRAVEL_EXPENSE", "TRAVEL_EXPENSE"),
        batch_sizes=_SIZES,
        unbatchable=_UNBATCHABLE,
    )

    assert _names(batches) == [["0.png", "1.png"], ["2.png"]]


def test_batch_size_of_one_gives_the_old_per_image_behaviour() -> None:
    """The escape hatch: setting every size to 1 reproduces exactly what
    the stage did before batching existed."""
    batches = plan_extraction_batches(
        _docs("RECEIPT", "RECEIPT", "INVOICE"),
        batch_sizes={"default": 1},
        unbatchable=set(),
    )

    assert _names(batches) == [["0.png"], ["1.png"], ["2.png"]]


def test_no_documents_gives_no_batches() -> None:
    assert plan_extraction_batches([], batch_sizes=_SIZES, unbatchable=_UNBATCHABLE) == []
