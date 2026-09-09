"""The legacy keyword classification path must emit canonical document types.

tests/ is gitignored — local-only. Locks the fix for the TRAVEL_EXPENSE /
VEHICLE_LOGBOOK mismatch: type_mappings values and fallback_keywords keys must
be canonical types (TRAVEL / LOGBOOK), otherwise the classifier emits a string
that extraction routing cannot resolve (silent universal fallback / ValueError).
"""

from common.field_schema import get_field_schema
from common.prompt_catalog import PromptCatalog
from common.turn_parsers import ClassificationParser


def _legacy_type(raw: str) -> str:
    return ClassificationParser()._parse_legacy(raw)["DOCUMENT_TYPE"]


class TestLegacyEmitsCanonicalTypes:
    def test_boarding_pass_is_travel(self) -> None:
        assert _legacy_type("This is a boarding pass for the flight.") == "TRAVEL"

    def test_vehicle_logbook_is_logbook(self) -> None:
        assert _legacy_type("Vehicle logbook for the work car.") == "LOGBOOK"

    def test_odometer_keyword_fallback_is_logbook(self) -> None:
        # No type_mappings hit -> fallback_keywords (odometer) -> LOGBOOK.
        assert _legacy_type("Trip record with odometer readings.") == "LOGBOOK"


class TestDetectionTypesAreSupported:
    """Every type the keyword path can emit must be a supported document type."""

    def test_type_mappings_values_are_canonical(self) -> None:
        config = PromptCatalog().get_detection_config()
        supported = {t.lower() for t in get_field_schema().supported_document_types}
        for variant, canonical in config["type_mappings"].items():
            assert canonical.lower() in supported, (
                f"type_mappings['{variant}'] = '{canonical}' is not a supported type"
            )

    def test_fallback_keyword_keys_are_canonical(self) -> None:
        config = PromptCatalog().get_detection_config()
        supported = {t.lower() for t in get_field_schema().supported_document_types}
        for canonical in config["fallback_keywords"]:
            assert canonical.lower() in supported, (
                f"fallback_keywords key '{canonical}' is not a supported type"
            )
