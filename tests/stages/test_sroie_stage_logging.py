"""The SROIE stage must show progress while inference runs.

A 347-image run takes ~35 minutes and writes its artefacts only at the
end. Without per-image logging at INFO it is indistinguishable from a
hang, which is exactly how it was first reported.
"""

import logging

from typer.testing import CliRunner

import stages.sroie
from common.sroie.ground_truth import SroieRecord
from common.sroie.runner import run_benchmark
from pathlib import Path


def test_main_configures_info_logging(monkeypatch) -> None:
    """Every other stage calls logging.basicConfig in main(); this one
    must too, or its progress lines never reach a handler."""
    monkeypatch.setattr(stages.sroie, "run", lambda **kwargs: Path("summary.json"))

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers = []
    root.setLevel(logging.WARNING)
    try:
        result = CliRunner().invoke(stages.sroie.app, [])
        assert result.exit_code == 0, result.output
        assert root.handlers, "main() left the root logger with no handler"
        assert logging.getLogger("common.sroie.runner").isEnabledFor(logging.INFO)
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_each_record_emits_a_progress_line(caplog) -> None:
    """One INFO line per receipt, so a long run visibly advances."""
    records = [
        SroieRecord(
            image_id=f"X00{index}",
            image_path=Path(f"X00{index}.jpg"),
            company="ACME",
            date="15/01/2019",
            address="27 JALAN",
            total="9.00",
        )
        for index in range(3)
    ]

    with caplog.at_level(logging.INFO, logger="common.sroie.runner"):
        run_benchmark(records, lambda record: f"company: {record.company}")

    progress_lines = [r for r in caplog.records if "X00" in r.getMessage()]
    assert len(progress_lines) == 3
