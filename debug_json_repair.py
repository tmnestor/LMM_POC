#!/usr/bin/env python3
"""
Debug the JSON repair functionality step by step.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_json_repair():
    """Debug JSON repair step by step."""

    print("🔍 DEBUGGING JSON REPAIR STEP BY STEP")
    print("=" * 60)

    from common.extraction_parser import _repair_truncated_json, _fast_json_detection

    # Test with the truncated JSON from image_008
    truncated_json = '''```json
{
  "DOCUMENT_TYPE": "BANK_STATEMENT",
  "STATEMENT_DATE_RANGE": "08/08/2025 to 07/09/2025",
  "LINE_ITEM_DESCRIPTIONS": "EFTPOS Cash Out PRICELINE PHARMACY MACKAY QLD | EFTPOS Purchase OFFICEWORKS",
  "TRANSACTION_DATES": "07/09/2025 | 08/09/2025 | 09/09/2025 |'''

    expected_fields = ["DOCUMENT_TYPE", "STATEMENT_DATE_RANGE", "LINE_ITEM_DESCRIPTIONS", "TRANSACTION_DATES"]

    print("📋 Original text:")
    print(repr(truncated_json))

    print("\n📋 Fast JSON detection result:")
    is_json = _fast_json_detection(truncated_json)
    print(f"Is JSON: {is_json}")

    print("\n📋 Repair attempt:")
    try:
        repaired = _repair_truncated_json(truncated_json, expected_fields)
        print("Repaired text:")
        print(repr(repaired))

        print("\nRepaired text (formatted):")
        print(repaired)

        # Try to parse the repaired JSON
        try:
            import json
            parsed = json.loads(repaired)
            print("\n✅ Successfully parsed repaired JSON:")
            for key, value in parsed.items():
                print(f"  {key}: {value}")
        except Exception as parse_error:
            print(f"\n❌ Failed to parse repaired JSON: {parse_error}")
            print("JSON content that failed:")
            print(repr(repaired))

    except Exception as e:
        print(f"❌ Repair failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_json_repair()