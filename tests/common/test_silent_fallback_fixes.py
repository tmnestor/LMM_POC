"""Silent fallbacks in the scoring and config paths must fail loudly.

tests/ is gitignored -- local-only.

Every substantive bug found on 2026-08-11/12 was a degraded code path that
returned a plausible, well-formed answer instead of failing. CLAUDE.md already
mandates the opposite ("NEVER use silent fallbacks ... fail explicitly with
diagnostic errors"); the rule had not been applied to scoring or config reads.

Covers four:
  1. extraction_evaluator: debit filtering that fails must NOT score the
     unfiltered extraction against a filtered ground truth -- that silently
     recreates the row-misalignment bug that gave bank F1 0.316 instead of 0.86.
  3. inference.generation.models keys that no registered model resolves to.
  4. inference.vllm.models keys that are not registered model types.
  5. structure-suffix YAML that cannot be read falling back to Python constants.
"""

from pathlib import Path

import pytest
import yaml

from common.extraction_evaluator import ExtractionEvaluator
from common.pipeline_config import load_structure_suffixes


class TestDebitFilterFailsLoudly:
    """A filter that cannot run must not let scoring proceed on unfiltered data."""

    def _evaluator(self) -> ExtractionEvaluator:
        return ExtractionEvaluator(ground_truth_csv=None, field_definitions={})

    def test_filter_failure_raises(self, assert_diagnostic_error) -> None:
        evaluator = self._evaluator()
        # Register lengths that disagree: pandas construction fails downstream.
        broken = {
            "DOCUMENT_TYPE": "BANK_STATEMENT",
            "LINE_ITEM_DESCRIPTIONS": "A | B | C",
            "TRANSACTION_DATES": "01/01/2024 | 02/01/2024",
            "TRANSACTION_AMOUNTS_PAID": "$1.00 | $2.00",
            "ACCOUNT_BALANCE": "$10.00 | $8.00",
        }
        with pytest.raises(ValueError) as excinfo:
            evaluator._filter_debit_transactions(broken)
        assert_diagnostic_error(str(excinfo.value))

    def test_non_bank_still_passes_through(self) -> None:
        invoice = {"DOCUMENT_TYPE": "INVOICE", "TOTAL_AMOUNT": "$5.00"}
        assert self._evaluator()._filter_debit_transactions(invoice) == invoice


class TestStructureSuffixesFailLoudly:
    """A config file that cannot be read must not become Python defaults."""

    def test_missing_file_raises(self, tmp_path: Path, assert_diagnostic_error) -> None:
        with pytest.raises(ValueError) as excinfo:
            load_structure_suffixes(tmp_path / "nope.yaml")
        assert_diagnostic_error(str(excinfo.value))

    def test_malformed_yaml_raises(self, tmp_path: Path, assert_diagnostic_error) -> None:
        path = tmp_path / "extraction.yaml"
        path.write_text("settings: [this is not: valid mapping\n")
        with pytest.raises(ValueError) as excinfo:
            load_structure_suffixes(path)
        assert_diagnostic_error(str(excinfo.value))

    def test_none_path_still_uses_defaults(self) -> None:
        # No path given is a deliberate "no override", not a failed read.
        assert load_structure_suffixes(None)

    def test_valid_file_is_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "extraction.yaml"
        path.write_text(yaml.safe_dump({"settings": {"structure_suffixes": ["_X", "_Y"]}}))
        assert load_structure_suffixes(path) == ("_X", "_Y")

    def test_absent_key_uses_defaults(self, tmp_path: Path) -> None:
        # The file read fine and simply declares no override -- that is legal.
        path = tmp_path / "extraction.yaml"
        path.write_text(yaml.safe_dump({"settings": {}}))
        assert load_structure_suffixes(path)


class TestModelOverrideKeysValidated:
    """Typo'd per-model override keys must be caught, not silently ignored."""

    def test_unknown_vllm_model_key_raises(self, assert_diagnostic_error) -> None:
        from common.app_config import _validate_model_override_keys

        with pytest.raises(ValueError) as excinfo:
            _validate_model_override_keys(
                vllm_models={"internvl3-vllm": {}, "gemma4-typo-vllm": {}},
                generation_models={},
            )
        message = str(excinfo.value)
        assert_diagnostic_error(message)
        assert "gemma4-typo-vllm" in message

    def test_unknown_generation_key_raises(self, assert_diagnostic_error) -> None:
        from common.app_config import _validate_model_override_keys

        with pytest.raises(ValueError) as excinfo:
            _validate_model_override_keys(vllm_models={}, generation_models={"internvl9": {}})
        message = str(excinfo.value)
        assert_diagnostic_error(message)
        assert "internvl9" in message

    def test_real_config_passes(self) -> None:
        """The shipped run_config.yml must satisfy the new validation."""
        from common.app_config import _validate_model_override_keys

        repo = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((repo / "config" / "run_config.yml").read_text())
        _validate_model_override_keys(
            vllm_models=raw["inference"]["vllm"].get("models", {}),
            generation_models=raw["inference"]["generation"].get("models", {}),
        )

    def test_normalised_generation_keys_accepted(self) -> None:
        from common.app_config import _validate_model_override_keys

        # gemma4-12b-unified is the NORMALISED form of a registered type.
        _validate_model_override_keys(vllm_models={}, generation_models={"gemma4-12b-unified": {}})
