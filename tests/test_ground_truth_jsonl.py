"""Tests for JSONL ground truth loading and evaluator JSONL mode."""

import json
from pathlib import Path

from common.evaluation_metrics import load_ground_truth


class TestLoadGroundTruthJsonl:
    """Test load_ground_truth() with .jsonl files."""

    def _write_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "ground_truth.jsonl"
        with path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    def test_basic_load(self, tmp_path: Path) -> None:
        records = [
            {"filename": "img1.png", "DOCUMENT_TYPE": "RECEIPT", "TOTAL_AMOUNT": "$10"},
            {"filename": "img2.png", "DOCUMENT_TYPE": "INVOICE", "TOTAL_AMOUNT": "$20"},
        ]
        path = self._write_jsonl(tmp_path, records)
        gt = load_ground_truth(str(path), verbose=False)

        assert len(gt) == 2
        assert "img1.png" in gt
        assert gt["img1.png"]["DOCUMENT_TYPE"] == "RECEIPT"
        assert gt["img2.png"]["TOTAL_AMOUNT"] == "$20"

    def test_per_type_fields(self, tmp_path: Path) -> None:
        """Each record should only have its type's fields."""
        records = [
            {
                "filename": "receipt.png",
                "DOCUMENT_TYPE": "RECEIPT",
                "SUPPLIER_NAME": "Store",
                "TOTAL_AMOUNT": "$10",
            },
            {
                "filename": "logbook.png",
                "DOCUMENT_TYPE": "LOGBOOK",
                "VEHICLE_MAKE": "Toyota",
                "VEHICLE_MODEL": "Camry",
            },
        ]
        path = self._write_jsonl(tmp_path, records)
        gt = load_ground_truth(str(path), verbose=False)

        # Receipt should NOT have logbook fields
        assert "VEHICLE_MAKE" not in gt["receipt.png"]
        # Logbook should NOT have receipt fields
        assert "SUPPLIER_NAME" not in gt["logbook.png"]

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "ground_truth.jsonl"
        with path.open("w") as f:
            f.write(
                json.dumps({"filename": "a.png", "DOCUMENT_TYPE": "RECEIPT"}) + "\n"
            )
            f.write("\n")
            f.write("  \n")
            f.write(
                json.dumps({"filename": "b.png", "DOCUMENT_TYPE": "INVOICE"}) + "\n"
            )
        gt = load_ground_truth(str(path), verbose=False)
        assert len(gt) == 2

    def test_image_name_key_fallback(self, tmp_path: Path) -> None:
        """Records with 'image_name' instead of 'filename' should work."""
        records = [
            {
                "image_name": "test.png",
                "DOCUMENT_TYPE": "RECEIPT",
                "TOTAL_AMOUNT": "$5",
            },
        ]
        path = self._write_jsonl(tmp_path, records)
        gt = load_ground_truth(str(path), verbose=False)
        assert "test.png" in gt

    def test_not_found_preserved(self, tmp_path: Path) -> None:
        """Within-schema NOT_FOUND values should be preserved for evaluation."""
        records = [
            {
                "filename": "img.png",
                "DOCUMENT_TYPE": "RECEIPT",
                "SUPPLIER_NAME": "Store",
                "BUSINESS_ABN": "NOT_FOUND",
            },
        ]
        path = self._write_jsonl(tmp_path, records)
        gt = load_ground_truth(str(path), verbose=False)
        assert gt["img.png"]["BUSINESS_ABN"] == "NOT_FOUND"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            load_ground_truth(str(tmp_path / "missing.jsonl"), verbose=False)

    def test_csv_still_works(self, tmp_path: Path) -> None:
        """CSV files should still load via the legacy path."""
        csv_path = tmp_path / "ground_truth.csv"
        csv_path.write_text(
            "filename,DOCUMENT_TYPE,TOTAL_AMOUNT\n"
            "img1.png,RECEIPT,$10\n"
            "img2.png,INVOICE,$20\n"
        )
        gt = load_ground_truth(str(csv_path), verbose=False)
        assert len(gt) == 2
        assert gt["img1.png"]["DOCUMENT_TYPE"] == "RECEIPT"


class TestEvaluatorJsonlMode:
    """Test ExtractionEvaluator with JSONL ground truth."""

    def _write_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "ground_truth.jsonl"
        with path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    def test_jsonl_mode_flag(self, tmp_path: Path) -> None:
        from common.extraction_evaluator import ExtractionEvaluator

        path = self._write_jsonl(
            tmp_path,
            [
                {
                    "filename": "img.png",
                    "DOCUMENT_TYPE": "RECEIPT",
                    "TOTAL_AMOUNT": "$10",
                }
            ],
        )
        evaluator = ExtractionEvaluator(
            ground_truth_csv=str(path),
            field_definitions={},
            verbose=False,
        )
        assert evaluator._jsonl_mode is True
        assert evaluator.has_ground_truth

    def test_csv_mode_flag(self, tmp_path: Path) -> None:
        from common.extraction_evaluator import ExtractionEvaluator

        csv_path = tmp_path / "gt.csv"
        csv_path.write_text("filename,DOCUMENT_TYPE\nimg.png,RECEIPT\n")
        evaluator = ExtractionEvaluator(
            ground_truth_csv=str(csv_path),
            field_definitions={},
            verbose=False,
        )
        assert evaluator._jsonl_mode is False

    def test_jsonl_evaluates_per_record_fields(self, tmp_path: Path) -> None:
        """JSONL mode should evaluate only the fields in the GT record."""
        from common.batch_types import ExtractionOutput
        from common.extraction_evaluator import ExtractionEvaluator

        path = self._write_jsonl(
            tmp_path,
            [
                {
                    "filename": "logbook.png",
                    "DOCUMENT_TYPE": "LOGBOOK",
                    "VEHICLE_MAKE": "Toyota",
                    "VEHICLE_MODEL": "Camry",
                },
            ],
        )
        evaluator = ExtractionEvaluator(
            ground_truth_csv=str(path),
            field_definitions={},  # empty -- JSONL mode ignores this
            verbose=False,
        )

        extraction = ExtractionOutput(
            image_path="logbook.png",
            image_name="logbook.png",
            document_type="LOGBOOK",
            extracted_data={
                "DOCUMENT_TYPE": "LOGBOOK",
                "VEHICLE_MAKE": "Toyota",
                "VEHICLE_MODEL": "Camry",
            },
            processing_time=0,
            prompt_used="test",
        )
        result = evaluator.evaluate(extraction)

        assert result["overall_accuracy"] == 1.0
        # Should only evaluate DOCUMENT_TYPE, VEHICLE_MAKE, VEHICLE_MODEL
        assert result["total_fields"] == 3
