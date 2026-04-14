"""KFP pipeline stages with file-based artifact handoff.

Each stage is an independent executable step:
    classify  -> classifications.jsonl
    extract   -> raw_extractions.jsonl
    clean     -> cleaned_extractions.jsonl
    evaluate  -> evaluation_results.jsonl + reports
"""
