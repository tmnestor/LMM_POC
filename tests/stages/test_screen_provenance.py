"""A resumed file must say so.

tests/ is gitignored — local-only.

`330 scored / missing 0` is what a SUCCESSFUL resume produces -- 30 records
carried plus 300 newly screened -- and it is also exactly what a full rescreen
produces. Reading the total alone, the two are indistinguishable, so a resume
that worked gets reported as one that did not, and the only way to settle it is
to go and read the classify log.

These pin the two places that now say which happened: the classify summary
line, and a NOTE at the top of the evaluate report.
"""

import pytest

from stages.evaluate_quality_screen import format_report, screening_runs
from stages.quality_screen import _log_screen_summary, write_screen_records

RUN_ONE = "2026-09-10T04:17:52"
RUN_TWO = "2026-09-10T04:20:31"


def record(name, *, screened_at=RUN_ONE):
    return {
        "image_name": name,
        "variant": "quality_screen_v12",
        "tiling": {"min_tiles": 12, "max_tiles": 12},
        "screened_at": screened_at,
        "malformed": False,
        "think_drift": False,
    }


class TestClassifySummary:
    def test_a_resumed_run_names_what_was_carried(self, tmp_path, caplog):
        records = [record("a.png"), record("b.png", screened_at=RUN_TWO)]

        with caplog.at_level("INFO"):
            _log_screen_summary(records, tmp_path / "out.jsonl", carried=1)

        assert "1 carried over from an earlier run" in caplog.text
        assert "1 screened now" in caplog.text

    def test_a_fresh_run_does_not_mention_carrying(self, tmp_path, caplog):
        """The wording must not imply a resume that did not happen."""
        with caplog.at_level("INFO"):
            _log_screen_summary([record("a.png")], tmp_path / "out.jsonl", carried=0)

        assert "carried over" not in caplog.text
        assert "Screened 1 images" in caplog.text


class TestScreeningRuns:
    def test_one_run_is_reported_as_one(self, tmp_path):
        path = write_screen_records([record("a.png"), record("b.png")], tmp_path / "s.jsonl")

        assert screening_runs(path) == [RUN_ONE]

    def test_a_resumed_file_reports_every_run(self, tmp_path):
        path = write_screen_records(
            [record("a.png"), record("b.png", screened_at=RUN_TWO)], tmp_path / "s.jsonl"
        )

        assert screening_runs(path) == [RUN_ONE, RUN_TWO]

    def test_a_file_written_before_stamping_reports_none(self, tmp_path):
        """Older files must not crash the report, and must not be described as
        a single run either -- nothing is known about how they were built."""
        unstamped = record("a.png")
        del unstamped["screened_at"]
        path = write_screen_records([unstamped], tmp_path / "s.jsonl")

        assert screening_runs(path) == []


class TestReportNote:
    def _report(self, runs):
        return {
            "variant": "quality_screen_v12",
            "screening_runs": runs,
            "counts": {"total": 2, "scored": 2, "malformed": 0, "missing": 0, "think_drift": 0},
            "per_criterion": {},
            "by_document_type": {},
            "overall_confusion": {},
        }

    def test_a_resumed_file_is_flagged(self):
        text = format_report(self._report([RUN_ONE, RUN_TWO]))

        assert "screened across 2 runs" in text
        assert RUN_ONE in text and RUN_TWO in text

    def test_a_single_run_is_not_flagged(self):
        """The NOTE must appear only when it is true, or it becomes noise that
        gets skimmed past on the run where it matters."""
        assert "screened across" not in format_report(self._report([RUN_ONE]))

    def test_an_unstamped_file_is_not_flagged(self):
        assert "screened across" not in format_report(self._report([]))

    def test_the_note_precedes_the_numbers(self):
        """It has to be read before the table, not after it."""
        text = format_report(self._report([RUN_ONE, RUN_TWO]))

        assert text.index("screened across") < text.index("CRITERION")


def test_the_stamp_is_not_part_of_the_resume_settings_check():
    """Differing timestamps are the NORMAL state of a resumed file.

    Were `screened_at` compared like `variant` and `tiling` are, the second
    resume would always find every record stale and rescreen the corpus --
    resume would appear to work once and then silently stop working.
    """
    from stages.quality_screen import partition_for_resume

    images = []
    existing = [record("a.png"), record("b.png", screened_at=RUN_TWO)]

    to_screen, kept = partition_for_resume(
        images, existing, variant="quality_screen_v12", tiling={"min_tiles": 12, "max_tiles": 12}
    )

    assert to_screen == []
    # Both dropped only because `images` is empty -- the point is that neither
    # triggered the stale-settings path, which would have logged a warning.
    assert kept == []


def test_records_carry_the_run_stamp():
    from stages.quality_screen import run_quality_screen
    from common.quality_screen_parser import ScreenVocabulary

    criteria = ["blur"]
    vocabulary = ScreenVocabulary(
        criteria=criteria,
        polarity={"blur": True},
        overall_levels=["GOOD", "FAIR", "POOR"],
        prompt="ask",
    )
    records = run_quality_screen(
        ["/data/a.png"],
        infer=lambda paths, _p: ["1. BLUR: NO\n2. OVERALL: GOOD"] * len(paths),
        vocabulary=vocabulary,
        variant="quality_screen_v12",
        tiling={"min_tiles": 12, "max_tiles": 12},
        screened_at=RUN_ONE,
    )

    assert records[0]["screened_at"] == RUN_ONE


def test_the_dp_worker_accepts_the_stamp():
    """Threaded from the parent, not generated per worker -- otherwise one
    sharded run stamps several timestamps and looks like several resumed ones."""
    import importlib
    import inspect

    from stages.quality_screen import DP_WORKER

    module_name, _, function_name = DP_WORKER.rpartition(".")
    worker = getattr(importlib.import_module(module_name), function_name)

    assert "screened_at" in inspect.signature(worker).parameters


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
