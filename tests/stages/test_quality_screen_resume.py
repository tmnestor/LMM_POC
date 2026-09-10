"""Resume: screen the new arrivals, not the whole directory again.

tests/ is gitignored — local-only.

This runs as a pipeline and images arrive over time, so rescreening everything
on every run is waste that grows with the corpus. Resume makes a re-run cost
only what is new.

What makes it safe is the settings check. A kept record must carry the same
prompt variant AND the same tile budget as the run resuming from it, because
both change the answers -- variants differ in polarity and in what they ask,
and the tile floor is the difference between a heavy receipt reading as damaged
and reading as fine. A file holding some v11 answers and some v12 answers is
not a run, and nothing downstream can tell: evaluate would score the mixture
and report an ordinary-looking number. So a settings change discards
everything rather than resuming partially.
"""

import json

import pytest

from stages.quality_screen import (
    load_existing_records,
    partition_for_resume,
    write_screen_records,
)

TILING = {"min_tiles": 12, "max_tiles": 12}
VARIANT = "quality_screen_v12"


def record(name, *, variant=VARIANT, tiling=TILING):
    return {
        "image_path": f"/data/{name}",
        "image_name": name,
        "variant": variant,
        "tiling": tiling,
        "answers": {},
        "overall": "GOOD",
        "malformed": False,
        "malformed_reason": None,
        "think_drift": False,
        "raw_response": "7. OVERALL: GOOD",
    }


@pytest.fixture
def corpus(tmp_path):
    """Four images on disk."""
    names = ["a.png", "b.png", "c.png", "d.png"]
    for name in names:
        (tmp_path / name).touch()
    return sorted(tmp_path / name for name in names)


class TestPartition:
    def test_an_empty_output_screens_everything(self, corpus):
        to_screen, kept = partition_for_resume(corpus, [], variant=VARIANT, tiling=TILING)

        assert to_screen == corpus
        assert kept == []

    def test_only_the_new_arrivals_are_screened(self, corpus):
        existing = [record("a.png"), record("b.png")]

        to_screen, kept = partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert [path.name for path in to_screen] == ["c.png", "d.png"]
        assert [r["image_name"] for r in kept] == ["a.png", "b.png"]

    def test_a_fully_screened_corpus_screens_nothing(self, corpus):
        existing = [record(path.name) for path in corpus]

        to_screen, kept = partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert to_screen == []
        assert len(kept) == 4

    def test_records_for_removed_images_are_dropped(self, corpus):
        """Otherwise the report scores rows for images nobody can look at."""
        existing = [record(path.name) for path in corpus] + [record("deleted.png")]

        to_screen, kept = partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert to_screen == []
        assert "deleted.png" not in {r["image_name"] for r in kept}
        assert len(kept) == 4


class TestSettingsGuard:
    """A settings change is not a partial resume."""

    def test_a_different_variant_discards_everything(self, corpus):
        existing = [record("a.png", variant="quality_screen_v11"), record("b.png")]

        to_screen, kept = partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert to_screen == corpus, "a mixed-variant file must be rescreened, not topped up"
        assert kept == []

    def test_a_different_tile_budget_discards_everything(self, corpus):
        """min_tiles 6 vs 12 was the measured difference between a heavy
        receipt reading as damaged and reading as being in good condition."""
        existing = [record("a.png", tiling={"min_tiles": 6, "max_tiles": 6})]

        to_screen, kept = partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert to_screen == corpus
        assert kept == []

    def test_one_stale_record_discards_the_whole_file(self, corpus):
        """Not just the stale one. The rest were written by the same run."""
        existing = [
            record("a.png"),
            record("b.png"),
            record("c.png", variant="quality_screen_v11"),
        ]

        to_screen, kept = partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert to_screen == corpus
        assert kept == []

    def test_records_with_no_tiling_stamp_are_treated_as_stale(self, corpus):
        """Files from before the stamp existed cannot be shown to match, so
        they are rescreened rather than assumed compatible."""
        stamped_none = record("a.png")
        del stamped_none["tiling"]

        to_screen, kept = partition_for_resume(corpus, [stamped_none], variant=VARIANT, tiling=TILING)

        assert to_screen == corpus
        assert kept == []

    def test_the_guard_is_reported_not_silent(self, corpus, caplog):
        existing = [record("a.png", variant="quality_screen_v11")]

        with caplog.at_level("WARNING"):
            partition_for_resume(corpus, existing, variant=VARIANT, tiling=TILING)

        assert "Not resuming" in caplog.text


class TestLoadExisting:
    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert load_existing_records(tmp_path / "absent.jsonl") == []

    def test_records_round_trip(self, tmp_path):
        records = [record("a.png"), record("b.png")]
        path = write_screen_records(records, tmp_path / "quality_screen.jsonl")

        assert load_existing_records(path) == records

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "quality_screen.jsonl"
        path.write_text(json.dumps(record("a.png")) + "\n\n")

        assert len(load_existing_records(path)) == 1

    def test_a_corrupt_file_fails_rather_than_rescreening_silently(self, assert_diagnostic_error, tmp_path):
        """Returning [] here would rescreen the corpus quietly -- the exact
        waste resume exists to avoid, done without saying so."""
        path = tmp_path / "quality_screen.jsonl"
        path.write_text(json.dumps(record("a.png")) + "\n{not json\n")

        with pytest.raises(ValueError) as exc_info:
            load_existing_records(path)

        assert_diagnostic_error(str(exc_info.value))


def test_a_resumed_file_is_ordered_by_image_name(tmp_path):
    """Resume appends new records to old, so without an explicit sort the file
    would record the order runs happened in rather than the corpus."""
    from stages.quality_screen import _ordered

    merged = _ordered([record("d.png"), record("a.png"), record("c.png")])

    assert [r["image_name"] for r in merged] == ["a.png", "c.png", "d.png"]
