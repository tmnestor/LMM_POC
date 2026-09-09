"""Travel/logbook extraction emits canonical DOCUMENT_TYPE.

tests/ is gitignored — local-only. Covers the legacy-naming bug where the
travel prompt emitted DOCUMENT_TYPE: TRAVEL_EXPENSE, which did not normalise
to canonical 'travel' (logbook's VEHICLE_LOGBOOK already resolved).
"""

from pathlib import Path

import yaml

from common.field_schema import get_field_schema


def _prompt_body(key: str) -> str:
    data = yaml.safe_load(Path("prompts/internvl3_prompts.yaml").read_text(encoding="utf-8"))
    return data["prompts"][key]["prompt"]


class TestDocTypeAliasResolution:
    def test_travel_expense_underscore_resolves_to_travel(self) -> None:
        assert get_field_schema().resolve_doc_type("TRAVEL_EXPENSE") == "travel"
        assert get_field_schema().resolve_doc_type("travel_expense") == "travel"

    def test_vehicle_logbook_underscore_still_resolves(self) -> None:
        assert get_field_schema().resolve_doc_type("VEHICLE_LOGBOOK") == "logbook"


class TestPromptsEmitCanonicalDocType:
    def test_travel_prompt_emits_canonical(self) -> None:
        body = _prompt_body("travel")
        assert "DOCUMENT_TYPE: TRAVEL\n" in body
        assert "TRAVEL_EXPENSE" not in body

    def test_logbook_prompt_emits_canonical(self) -> None:
        body = _prompt_body("logbook")
        assert "DOCUMENT_TYPE: LOGBOOK\n" in body
        assert "VEHICLE_LOGBOOK" not in body


class TestPromptOutputFormatCounts:
    def test_travel_output_format_count(self) -> None:
        assert "OUTPUT FORMAT (9 FIELDS):" in _prompt_body("travel")

    def test_logbook_output_format_count(self) -> None:
        assert "OUTPUT FORMAT (16 FIELDS):" in _prompt_body("logbook")
