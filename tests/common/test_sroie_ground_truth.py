"""Tests for loading the SROIE ground-truth split."""

import json
from pathlib import Path

import pytest

from common.sroie.ground_truth import SroieDatasetError, load_sroie_split

_ENTITY = {
    "company": "OJC MARKETING SDN BHD",
    "date": "15/01/2019",
    "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, B1750 MASAI, JOHOR",
    "total": "193.00",
}


def _make_split(root: Path, entities: dict[str, dict], images: list[str]) -> Path:
    """Build a split directory mirroring the real img/ + entities/ layout."""
    (root / "img").mkdir(parents=True)
    (root / "entities").mkdir(parents=True)
    for stem, payload in entities.items():
        (root / "entities" / f"{stem}.txt").write_text(json.dumps(payload))
    for stem in images:
        (root / "img" / f"{stem}.jpg").write_bytes(b"not-a-real-jpeg")
    return root


def test_loads_one_record_per_entity_file(tmp_path: Path) -> None:
    split = _make_split(tmp_path, {"X001": _ENTITY}, ["X001"])

    records = load_sroie_split(split)

    assert len(records) == 1
    assert records[0].image_id == "X001"
    assert records[0].company == "OJC MARKETING SDN BHD"
    assert records[0].total == "193.00"
    assert records[0].image_path == split / "img" / "X001.jpg"


def test_entity_without_an_image_is_an_error(tmp_path: Path) -> None:
    """Scoring a record whose image never reached the model would count a
    guaranteed miss as a model failure."""
    split = _make_split(tmp_path, {"X001": _ENTITY, "X002": _ENTITY}, ["X001"])

    with pytest.raises(SroieDatasetError) as excinfo:
        load_sroie_split(split)

    message = str(excinfo.value)
    assert "X002" in message
    assert str(split / "img") in message


def test_image_without_ground_truth_is_an_error(tmp_path: Path) -> None:
    """An unscoreable image silently shrinks the denominator."""
    split = _make_split(tmp_path, {"X001": _ENTITY}, ["X001", "X999"])

    with pytest.raises(SroieDatasetError) as excinfo:
        load_sroie_split(split)

    assert "X999" in str(excinfo.value)


def test_missing_field_in_an_entity_file_is_an_error(tmp_path: Path) -> None:
    """A record short a field must stop the run, not score as NOT_FOUND."""
    incomplete = {k: v for k, v in _ENTITY.items() if k != "total"}
    split = _make_split(tmp_path, {"X001": incomplete}, ["X001"])

    with pytest.raises(SroieDatasetError) as excinfo:
        load_sroie_split(split)

    message = str(excinfo.value)
    assert "total" in message
    assert "X001" in message


def test_blank_field_value_is_an_error(tmp_path: Path) -> None:
    """One train-split record ships an empty 'total'. A blank answer key is
    unscoreable, and must be caught when the split loads rather than
    surfacing part-way through a run."""
    blank = {**_ENTITY, "total": "   "}
    split = _make_split(tmp_path, {"X001": blank}, ["X001"])

    with pytest.raises(SroieDatasetError) as excinfo:
        load_sroie_split(split)

    message = str(excinfo.value)
    assert "total" in message
    assert "X001" in message


def test_malformed_json_names_the_offending_file(tmp_path: Path) -> None:
    split = _make_split(tmp_path, {"X001": _ENTITY}, ["X001"])
    (split / "entities" / "X001.txt").write_text("{not json")

    with pytest.raises(SroieDatasetError) as excinfo:
        load_sroie_split(split)

    assert "X001.txt" in str(excinfo.value)


def test_empty_split_directory_is_an_error(tmp_path: Path) -> None:
    """An empty run reporting F1 0.0 looks like a catastrophic model, not
    a mis-set path."""
    split = _make_split(tmp_path, {}, [])

    with pytest.raises(SroieDatasetError) as excinfo:
        load_sroie_split(split)

    assert str(split / "entities") in str(excinfo.value)
