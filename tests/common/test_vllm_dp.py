"""Tests for common.vllm_dp round-robin partition + order reconstruction."""

from pathlib import Path

from common.vllm_dp import _partition_images, _reconstruct_order


def _names(chunk):
    return [p.name for p in chunk]


def test_partition_round_robin_balances_types():
    # Type-sorted list (extraction_order): banks first, then invoices, receipts.
    banks = [Path(f"b{i}.png") for i in range(8)]
    invoices = [Path(f"i{i}.png") for i in range(8)]
    receipts = [Path(f"r{i}.png") for i in range(8)]
    images = banks + invoices + receipts  # 24 images, type-sorted

    chunks = _partition_images(images, 4)
    assert len(chunks) == 4

    # Each GPU should get a balanced share of the EXPENSIVE banks, not a clump.
    for chunk in chunks:
        n_banks = sum(1 for p in chunk if p.name.startswith("b"))
        assert n_banks == 2  # 8 banks / 4 GPUs, evenly dealt
        assert len(chunk) == 6


def test_partition_preserves_per_gpu_type_ordering():
    # Within a chunk, the stride keeps the global (type-sorted) order:
    # banks still precede invoices precede receipts -> prefix-cache locality.
    images = (
        [Path(f"b{i}.png") for i in range(4)]
        + [Path(f"i{i}.png") for i in range(4)]
        + [Path(f"r{i}.png") for i in range(4)]
    )
    chunks = _partition_images(images, 2)
    for chunk in chunks:
        prefixes = [p.name[0] for p in chunk]
        # b's come before i's come before r's
        assert prefixes == sorted(prefixes, key="bir".index)


def test_partition_fewer_images_than_gpus():
    images = [Path("a.png"), Path("b.png")]
    chunks = _partition_images(images, 8)
    assert len(chunks) == 2
    assert all(len(c) == 1 for c in chunks)


def test_reconstruct_order_round_trips():
    # Simulate run_dp: deal images round-robin, each worker returns one record
    # per image in chunk order; reconstruction must restore original order.
    images = [Path(f"img{i}.png") for i in range(10)]
    chunks = _partition_images(images, 3)
    actual_gpus = len(chunks)

    gpu_results = {gpu_id: [{"image_name": p.name} for p in chunk] for gpu_id, chunk in enumerate(chunks)}
    merged = _reconstruct_order(gpu_results, actual_gpus)
    assert [r["image_name"] for r in merged] == [p.name for p in images]


def test_reconstruct_order_single_gpu():
    images = [Path(f"img{i}.png") for i in range(5)]
    chunks = _partition_images(images, 1)
    gpu_results = {0: [{"image_name": p.name} for p in chunks[0]]}
    merged = _reconstruct_order(gpu_results, 1)
    assert [r["image_name"] for r in merged] == [p.name for p in images]
