"""Tests for PromptCatalog classification-evidence accessors.

tests/ is gitignored — local-only. Covers the YAML-driven classification work:
``get_column_roles`` and ``get_classification_evidence`` load and fail-fast on
the new blocks in ``document_type_detection.yaml``.
"""

from pathlib import Path

import pytest

from common.prompt_catalog import PromptCatalog


def _write_detection(tmp_path: Path, body: str) -> PromptCatalog:
    (tmp_path / "document_type_detection.yaml").write_text(body, encoding="utf-8")
    return PromptCatalog(prompts_dir=tmp_path)


class TestColumnRolesLoad:
    """get_column_roles reads the real shipped YAML."""

    def test_returns_six_canonical_roles(self) -> None:
        roles = PromptCatalog().get_column_roles()
        for role in ("date", "description", "debit", "credit", "balance", "amount"):
            assert role in roles
            assert isinstance(roles[role], list) and roles[role]

    def test_bank_role_keywords_match_legacy_literal(self) -> None:
        roles = PromptCatalog().get_column_roles()
        assert "withdrawal" in roles["debit"]
        assert "deposit" in roles["credit"]
        assert "running balance" in roles["balance"]


class TestClassificationEvidenceLoad:
    def test_returns_rules_and_default(self) -> None:
        evidence = PromptCatalog().get_classification_evidence()
        assert isinstance(evidence["rules"], list) and evidence["rules"]
        assert "default" in evidence


class TestColumnRolesFailFast:
    def test_missing_block_raises_diagnostic(self, tmp_path: Path) -> None:
        catalog = _write_detection(tmp_path, "prompts: {}\n")
        with pytest.raises(ValueError) as exc:
            catalog.get_column_roles()
        msg = str(exc.value)
        assert "column_roles" in msg
        assert str(tmp_path) in msg  # absolute path / where
        assert "Fix:" in msg  # how to recover

    def test_empty_role_list_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(tmp_path, "column_roles:\n  debit: []\n")
        with pytest.raises(ValueError) as exc:
            catalog.get_column_roles()
        assert "debit" in str(exc.value)


class TestClassificationEvidenceFailFast:
    def test_missing_block_raises_diagnostic(self, tmp_path: Path) -> None:
        catalog = _write_detection(tmp_path, "prompts: {}\n")
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        msg = str(exc.value)
        assert "classification_evidence" in msg
        assert str(tmp_path) in msg
        assert "Fix:" in msg

    def test_empty_rules_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(tmp_path, "classification_evidence:\n  rules: []\n  default: none\n")
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        assert "rules" in str(exc.value)

    def test_missing_default_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(
            tmp_path,
            "classification_evidence:\n  rules:\n    - type: RECEIPT\n      when: { paid: true }\n",
        )
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        assert "default" in str(exc.value)


# A minimal-but-valid column_roles block reused by cross-reference tests.
_ROLES = "column_roles:\n  debit:\n    - debit\n  balance:\n    - balance\n"


class TestEvidenceCrossReferenceValidation:
    """Rules are validated against column_roles and supported_document_types."""

    def test_unknown_role_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(
            tmp_path,
            _ROLES + "classification_evidence:\n"
            "  rules:\n"
            "    - type: BANK_STATEMENT\n"
            "      when: { any_roles: [debit, nonsense] }\n"
            "  default: none\n",
        )
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        msg = str(exc.value)
        assert "nonsense" in msg
        assert "column_roles" in msg

    def test_unsupported_type_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(
            tmp_path,
            _ROLES + "classification_evidence:\n"
            "  rules:\n"
            "    - type: WIDGET\n"
            "      when: { any_roles: [debit] }\n"
            "  default: none\n",
        )
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        assert "WIDGET" in str(exc.value)

    def test_unknown_when_key_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(
            tmp_path,
            _ROLES + "classification_evidence:\n"
            "  rules:\n"
            "    - type: BANK_STATEMENT\n"
            "      when: { any_role: [debit] }\n"
            "  default: none\n",
        )
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        assert "any_role" in str(exc.value)

    def test_unsupported_default_raises(self, tmp_path: Path) -> None:
        catalog = _write_detection(
            tmp_path,
            _ROLES + "classification_evidence:\n"
            "  rules:\n"
            "    - type: BANK_STATEMENT\n"
            "      when: { any_roles: [debit] }\n"
            "  default: WIDGET\n",
        )
        with pytest.raises(ValueError) as exc:
            catalog.get_classification_evidence()
        assert "WIDGET" in str(exc.value)

    def test_valid_config_passes(self, tmp_path: Path) -> None:
        catalog = _write_detection(
            tmp_path,
            _ROLES + "classification_evidence:\n"
            "  rules:\n"
            "    - type: BANK_STATEMENT\n"
            "      when: { any_roles: [debit, balance] }\n"
            "    - type: RECEIPT\n"
            "      when: { paid: true }\n"
            "  default: none\n",
        )
        evidence = catalog.get_classification_evidence()
        assert evidence["default"] == "none"
