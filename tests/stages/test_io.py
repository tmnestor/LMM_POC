"""Tests for stages.io JSONL helpers — focus on append_jsonl (resume support)."""

from stages.io import append_jsonl, read_jsonl, write_jsonl


def test_append_jsonl_creates_when_absent(tmp_path):
    p = tmp_path / "out.jsonl"
    n = append_jsonl(p, [{"image_name": "a.png", "x": 1}])
    assert n == 1
    assert read_jsonl(p) == [{"image_name": "a.png", "x": 1}]


def test_append_jsonl_preserves_existing(tmp_path):
    p = tmp_path / "out.jsonl"
    write_jsonl(p, [{"image_name": "a.png"}])
    n = append_jsonl(p, [{"image_name": "b.png"}, {"image_name": "c.png"}])
    assert n == 2
    names = [r["image_name"] for r in read_jsonl(p)]
    assert names == ["a.png", "b.png", "c.png"]  # existing 'a' NOT truncated
