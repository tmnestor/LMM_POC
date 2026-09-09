"""Tests for the classify stage's resume filter (no GPU/model needed)."""

from stages.classify import _filter_already_classified
from stages.io import write_jsonl


def test_filter_no_existing_output_returns_all(tmp_path):
    imgs = [tmp_path / "a.png", tmp_path / "b.png"]
    remaining, done = _filter_already_classified(imgs, tmp_path / "out.jsonl")
    assert remaining == imgs
    assert done == 0


def test_filter_skips_already_classified(tmp_path):
    out = tmp_path / "out.jsonl"
    write_jsonl(out, [{"image_name": "a.png", "document_type": "RECEIPT"}])
    imgs = [tmp_path / "a.png", tmp_path / "b.png"]
    remaining, done = _filter_already_classified(imgs, out)
    assert remaining == [tmp_path / "b.png"]
    assert done == 1


def test_filter_all_done_returns_empty(tmp_path):
    out = tmp_path / "out.jsonl"
    write_jsonl(out, [{"image_name": "a.png"}, {"image_name": "b.png"}])
    imgs = [tmp_path / "a.png", tmp_path / "b.png"]
    remaining, done = _filter_already_classified(imgs, out)
    assert remaining == []
    assert done == 2
