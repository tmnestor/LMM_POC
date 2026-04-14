  ---
  Candidates

  1. Extraction Parser + Cleaner — Scattered Field-Type Knowledge

  Cluster: common/extraction_parser.py (790 LOC), common/extraction_cleaner.py (817 LOC), common/field_schema.py

  Why they're coupled: Both modules independently look up field types to decide how to handle values. extraction_cleaner.py hardcodes
  ~80 field-name substring patterns (e.g., "AMOUNT", "PRICE") to decide cleaning strategy, duplicating what FieldSchema already
  defines as monetary_fields, date_fields, etc. Callers (orchestrator, bank extractor) must know to call parser then cleaner in order,
   with no composition.

  Dependency category: In-process — pure string→dict transformation, no I/O.

  Test impact: Currently no tests for either module. A unified boundary test could verify "raw model output → cleaned extraction dict"
   in one shot.

  ---
  2. Multi-GPU Orchestrator — Circular Import with cli.py

  Cluster: common/multi_gpu.py (219 LOC), cli.py

  Why they're coupled: multi_gpu.py imports create_processor and load_model from cli.py at runtime. cli.py imports
  MultiGPUOrchestrator from multi_gpu.py. This circular dependency means multi-GPU can't be used from notebooks or other entry points
  without pulling in the entire CLI.

  Dependency category: In-process — the circular import is a code structure problem, not an I/O boundary.

  Test impact: Currently untestable in isolation — requires mocking cli.py functions. Breaking the cycle would let you test multi-GPU
  partitioning logic independently.

  ---
  3. Prompt Loading — 5 Entry Points, No Unified Access

  Cluster: common/simple_prompt_loader.py (169 LOC), common/unified_bank_extractor.py (internal ConfigLoader), models/orchestrator.py
  (prompt methods), cli.py (load_prompt_config())

  Why they're coupled: Five different code paths load prompts from YAML files in prompts/. SimplePromptLoader, the bank extractor's
  ConfigLoader, and CLI's load_prompt_config() all independently resolve prompt file paths and parse YAML. Adding a new document type
  requires touching multiple files to register its prompt routing.

  Dependency category: Local-substitutable — YAML file reads that could be tested with fixture files.

  Test impact: No tests for prompt loading. A single PromptManager boundary test could verify "document type + model → correct prompt
  text" without reading actual YAML files.

  ---
  4. Batch Statistics & Reporting — Stats Computed in 4 Places

  Cluster: common/batch_analytics.py (250 LOC), common/batch_reporting.py (328 LOC), common/batch_visualizations.py (335 LOC),
  common/document_pipeline.py (stats in _run_pipeline)

  Why they're coupled: Statistics are computed in DocumentPipeline, then re-computed in BatchAnalytics, then formatted in
  BatchReporter, then plotted in BatchVisualizer. Each module assumes specific dict/DataFrame shapes from the others. No unified stats
   object — BatchStats (typed) coexists with ad-hoc dicts and DataFrames.

  Dependency category: In-process — pure computation over result dicts.

  Test impact: No tests. A single StatsCollector boundary test could verify "list of batch results → complete stats + report
  artifacts."

  ---
  5. Evaluation Metrics — Field-Type-Aware Logic Scattered Across Functions

  Cluster: common/evaluation_metrics.py (2172 LOC), common/extraction_evaluator.py (441 LOC)

  Why they're coupled: evaluation_metrics.py has 7+ separate functions that each independently check field types (monetary, date,
  list, boolean) against FieldSchema. Ground truth loading hardcodes possible column names. extraction_evaluator.py wraps these calls
  but doesn't own the field-type logic — it just sequences them.

  Dependency category: In-process — pure comparison logic.

  Test impact: Existing tests are minimal. Boundary tests at "ground truth + extraction → per-field F1 scores" would replace needing
  to test each comparison function individually.

  ---