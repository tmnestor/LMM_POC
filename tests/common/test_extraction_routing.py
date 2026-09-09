"""Tests for PromptCatalog.build_extraction_routing.

tests/ is gitignored — local-only. Covers the travel/logbook routing fix
(orphaned prompt keys) and the fail-fast orphan guard.
"""

from pathlib import Path

import pytest

from common.prompt_catalog import PromptCatalog


class TestRealRoutingResolves:
    """The shipped internvl3 prompt YAML routes every canonical type."""

    def test_routing_includes_travel_and_logbook(self) -> None:
        routing = PromptCatalog().build_extraction_routing("internvl3-vllm")
        assert routing.get("TRAVEL") == "travel"
        assert routing.get("LOGBOOK") == "logbook"

    def test_routing_keeps_existing_types(self) -> None:
        routing = PromptCatalog().build_extraction_routing("internvl3-vllm")
        assert routing.get("INVOICE") == "invoice"
        assert routing.get("RECEIPT") == "receipt"
        assert routing.get("BANK_STATEMENT") in {
            "bank_statement_flat",
            "bank_statement_date_grouped",
        }
        assert routing.get("UNIVERSAL") == "universal"


def _catalog_with_prompt_keys(tmp_path: Path, keys: list[str]) -> PromptCatalog:
    body = "prompts:\n" + "".join(f'  {k}:\n    prompt: "x"\n' for k in keys)
    (tmp_path / "internvl3_prompts.yaml").write_text(body, encoding="utf-8")
    return PromptCatalog(prompts_dir=tmp_path)


class TestOrphanGuard:
    def test_orphaned_key_raises_diagnostic(self, tmp_path: Path) -> None:
        catalog = _catalog_with_prompt_keys(tmp_path, ["invoice", "travel_expense"])
        with pytest.raises(ValueError) as exc:
            catalog.build_extraction_routing("internvl3-vllm")
        msg = str(exc.value)
        assert "travel_expense" in msg  # what
        assert "internvl3_prompts.yaml" in msg  # where
        assert "supported_document_types" in msg  # what-it-should-be
        assert "Fix:" in msg  # how-to-recover

    def test_all_canonical_keys_do_not_raise(self, tmp_path: Path) -> None:
        catalog = _catalog_with_prompt_keys(
            tmp_path, ["invoice", "travel", "logbook", "bank_statement_flat"]
        )
        routing = catalog.build_extraction_routing("internvl3-vllm")
        assert routing == {
            "INVOICE": "invoice",
            "TRAVEL": "travel",
            "LOGBOOK": "logbook",
            "BANK_STATEMENT": "bank_statement_flat",
        }

    def test_real_internvl3_yaml_has_no_orphans(self) -> None:
        # After Task 1, every shipped prompt key resolves — must not raise.
        PromptCatalog().build_extraction_routing("internvl3-vllm")
