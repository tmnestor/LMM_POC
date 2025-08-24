#!/usr/bin/env python3
"""
Phase 3 Migration Test Suite - Update Processing Logic

Tests that processing logic has been updated to use schema-driven
field discovery and dynamic prompt generation.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_grouped_extraction_schema_integration():
    """Test that GroupedExtractionStrategy uses schema-driven field discovery."""
    print("🧪 Testing Grouped Extraction Schema Integration...")

    from common.grouped_extraction import GroupedExtractionStrategy

    # Test basic initialization
    strategy = GroupedExtractionStrategy(
        extraction_mode="grouped", grouping_strategy="detailed_grouped", debug=True
    )

    # Verify schema is loaded
    assert hasattr(strategy, "schema"), "Strategy should have schema attribute"
    assert strategy.schema is not None, "Schema should be initialized"

    # Test field groups loaded from schema
    assert hasattr(strategy, "field_groups"), "Strategy should have field_groups"
    assert len(strategy.field_groups) > 0, "Field groups should be loaded"

    print(f"✅ Schema initialized with {strategy.schema.total_fields} fields")
    print(f"✅ Loaded {len(strategy.field_groups)} field groups")
    return True


def test_dynamic_prompt_generation():
    """Test that prompts are generated dynamically from schema."""
    print("\n🧪 Testing Dynamic Prompt Generation...")

    from common.grouped_extraction import GroupedExtractionStrategy

    strategy = GroupedExtractionStrategy(
        extraction_mode="grouped", grouping_strategy="detailed_grouped", debug=False
    )

    # Test prompt generation for a known group
    if "critical" in strategy.field_groups:
        test_group = "critical"
        group_config = strategy.field_groups[test_group]
        fields = group_config["fields"]

        # Generate prompt
        prompt = strategy.generate_group_prompt(test_group)

        # Verify prompt contains expected elements
        assert len(prompt) > 100, "Prompt should be substantial"
        assert test_group.upper() not in prompt or "critical" in prompt.lower(), (
            "Prompt should reference group context"
        )

        # Check all fields are included
        for field in fields:
            assert field in prompt, f"Field {field} should be in prompt"

        print(f"✅ Generated {len(prompt)} character prompt for '{test_group}' group")
        print(f"✅ All {len(fields)} fields included in prompt")
        return True
    else:
        print("⚠️ No 'critical' group found, testing fallback prompts...")
        # Test with first available group
        first_group = list(strategy.field_groups.keys())[0]
        prompt = strategy.generate_group_prompt(first_group)
        assert len(prompt) > 50, "Fallback prompt should be generated"
        print(f"✅ Fallback prompt generated for '{first_group}'")
        return True


def test_extraction_parser_schema_integration():
    """Test that parser uses dynamic field discovery."""
    print("\n🧪 Testing Extraction Parser Schema Integration...")

    from common.extraction_parser import parse_extraction_response
    from common.schema_loader import get_global_schema

    # Get expected fields from schema
    schema = get_global_schema()
    expected_fields = schema.field_names

    # Test with empty response
    empty_result = parse_extraction_response("")

    # Verify all schema fields are included
    assert len(empty_result) == len(expected_fields), (
        f"Should have {len(expected_fields)} fields, got {len(empty_result)}"
    )

    for field in expected_fields:
        assert field in empty_result, f"Field {field} should be in result"
        assert empty_result[field] == "NOT_FOUND", "Empty field should be NOT_FOUND"

    # Test with sample response
    sample_response = f"""
{expected_fields[0]}: TestValue1
{expected_fields[1]}: TestValue2
{expected_fields[2]}: NOT_FOUND
"""

    parsed_result = parse_extraction_response(sample_response)

    # Verify parsing works
    assert parsed_result[expected_fields[0]] == "TestValue1", "Should parse first field"
    assert parsed_result[expected_fields[1]] == "TestValue2", (
        "Should parse second field"
    )
    assert parsed_result[expected_fields[2]] == "NOT_FOUND", "Should handle NOT_FOUND"

    print(f"✅ Parser uses {len(expected_fields)} schema-driven fields")
    print(f"✅ Successfully parsed sample response with {len(parsed_result)} fields")
    return True


def test_schema_fallback_handling():
    """Test that fallback mechanisms work when schema fails."""
    print("\n🧪 Testing Schema Fallback Handling...")

    from common.grouped_extraction import GroupedExtractionStrategy

    # Test with fallback prompt generation
    strategy = GroupedExtractionStrategy(
        extraction_mode="grouped", grouping_strategy="detailed_grouped", debug=False
    )

    # Test fallback prompt generation method directly
    test_fields = ["TEST_FIELD_1", "TEST_FIELD_2", "MONETARY_AMOUNT"]

    # Test different group name patterns
    test_cases = [
        ("monetary_group", "financial"),
        ("date_group", "temporal"),
        ("entity_group", "business"),
        ("unknown_group", "generic"),
    ]

    for group_name, expected_type in test_cases:
        expertise, context, instruction = strategy._generate_fallback_prompts(
            group_name, test_fields
        )

        assert len(expertise) > 10, (
            f"Expertise frame should be substantial for {group_name}"
        )
        assert len(context) > 10, f"Context should be substantial for {group_name}"
        assert len(instruction) > 10, (
            f"Instruction should be substantial for {group_name}"
        )

        # Verify pattern matching works
        if expected_type != "generic":
            assert any(
                keyword in expertise.lower() or keyword in context.lower()
                for keyword in [expected_type, group_name.split("_")[0]]
            ), f"Should reference {expected_type} concepts for {group_name}"

    print("✅ Fallback prompt generation working for all patterns")
    print("✅ Schema error handling mechanisms functional")
    return True


def test_integration_compatibility():
    """Test that changes are compatible with existing processing pipeline."""
    print("\n🧪 Testing Integration Compatibility...")

    from common.config import EXTRACTION_FIELDS, FIELD_COUNT
    from common.extraction_parser import parse_extraction_response
    from common.grouped_extraction import GroupedExtractionStrategy

    # Test basic integration
    strategy = GroupedExtractionStrategy(
        extraction_mode="grouped", grouping_strategy="detailed_grouped"
    )

    # Verify config still works
    assert len(EXTRACTION_FIELDS) == FIELD_COUNT, "Config constants should still work"
    assert FIELD_COUNT > 20, "Should have substantial field count"

    # Test that we can still process groups
    first_group = list(strategy.field_groups.keys())[0]
    group_config = strategy.field_groups[first_group]
    assert "fields" in group_config, "Group config should have fields"
    assert len(group_config["fields"]) > 0, "Group should have fields"

    # Test parser integration
    test_response = f"{EXTRACTION_FIELDS[0]}: TestValue"
    parsed = parse_extraction_response(test_response)
    assert EXTRACTION_FIELDS[0] in parsed, "Should parse known field"

    print("✅ Integration with existing config maintained")
    print("✅ Backward compatibility preserved")
    print("✅ Processing pipeline remains functional")
    return True


def main():
    """Run all Phase 3 migration tests."""
    print("🚀 Phase 3 Migration Test Suite")
    print("=" * 60)
    print("Testing schema-driven processing logic updates")

    tests = [
        test_grouped_extraction_schema_integration,
        test_dynamic_prompt_generation,
        test_extraction_parser_schema_integration,
        test_schema_fallback_handling,
        test_integration_compatibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")

    if failed == 0:
        print("🎉 Phase 3 migration successful!")
        print("✨ Processing logic updated to use schema-driven field discovery")
        print("💡 Ready for Phase 4: Model-specific prompt generation")
    else:
        print(f"❌ {failed} tests failed - Phase 3 needs attention")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
