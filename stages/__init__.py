"""KFP pipeline stages with file-based artifact handoff.

Each stage is an independent executable step:
    classify      -> classifications.jsonl
    extract       -> raw_extractions.jsonl
    clean         -> cleaned_extractions.jsonl
    evaluate      -> evaluation_results.jsonl + reports

Trust distribution pipeline:
    trust_classify -> trust_classifications.jsonl + trust_quads.csv (GPU)
    trust_extract  -> raw_extractions.jsonl       (GPU)
    trust_clean    -> trust_compliance_results.jsonl (CPU)
    trust_evaluate -> trust_evaluation_results.jsonl (CPU)
"""
