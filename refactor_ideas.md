Here are the deepening opportunities I found, ordered by coupling severity and ROI:                                                                                     
                                                                                                                                                                        
---                                                                                                                                                                     
1. Bank Statement Extraction Cluster (3 files co-owning one domain)                                                                                                     
                                                                                                                                                                        
- Cluster: unified_bank_extractor.py (1,725 LOC), bank_statement_adapter.py (227 LOC), bank_statement_calculator.py (1,339 LOC)                                         
- Why they're coupled: All three co-own "bank statement extraction" — adapter routes to extractor, extractor calls calculator for balance verification, and they share  
types/state (strategy enums, column mappings, extraction results). Understanding any one requires reading all three.
- Dependency category: Intra-module (same domain split across files)                                                                                                    
- Test impact: Currently untestable in isolation because multi-turn state, strategy selection, and balance arithmetic are interleaved. A unified BankStatementService   
with a single extract(image) -> Result boundary would replace integration-style manual testing with proper boundary tests.                                              
                                                                                                                                                                        
---                                                                                                                                                                     
2. Configuration Cascade (mutable globals + immutable dataclass, split authority)

- Cluster: pipeline_config.py (411 LOC), common/config.py / model_config.py (358 LOC), cli.py (945 LOC)
- Why they're coupled: PipelineConfig is an immutable dataclass, but model_config.py has mutable module-level globals (GENERATION_CONFIGS, DEFAULT_BATCH_SIZES) that are
mutated at startup via apply_yaml_overrides(). Two config objects must stay in sync — one dataclass, one set of globals.
- Dependency category: Shared mutable state (module globals act as hidden dependency)
- Test impact: Can't test config merging without importing and mutating module globals. A ConfigurationManager that owns the full cascade (CLI > YAML > ENV > defaults)
would be testable with pure inputs.

---
3. Batch Processing Orchestration (1,302 LOC god method)

- Cluster: batch_processor.py — specifically _process_batch_two_phase() (300+ LOC) and _process_batch_sequential() (200+ LOC)
- Why they're coupled: Detection, extraction, bank-vs-standard partitioning, OOM fallback (recursive batch halving), ground truth override, and evaluation metrics are
all interleaved in nested loops. Progress bar state management is scattered throughout.
- Dependency category: Temporal coupling (phases must run in order, but logic is entangled)
- Test impact: Can't test detection phase without also running extraction. Splitting into DetectionPhase, ExtractionPhase, EvaluationPhase strategies would allow
testing each in isolation.

---
4. GPU Memory Management (fragmented across 4+ files)

- Cluster: gpu_optimization.py (657 LOC), robust_gpu_memory.py (175 LOC), registry.py (_split_internvl_model()), processor files (cleanup calls)
- Why they're coupled: OOM recovery requires coordination across all four — catch in processor, cleanup via gpu_optimization, check fragmentation via robust_gpu_memory,
device map from registry. The critical "never call empty_cache() inside except" pattern is a timing constraint that no interface enforces.
- Dependency category: Cross-cutting concern (GPU memory is global state affecting all processors)
- Test impact: Currently impossible to test OOM recovery without a GPU. A GPUMemoryManager class with injectable allocator/monitor would allow mock-based testing of
recovery logic.

---
5. Extraction Parsing & Cleaning Pipeline (unclear boundary between two stages)

- Cluster: extraction_parser.py (807 LOC), extraction_cleaner.py (817 LOC)
- Why they're coupled: Both load field_definitions.yaml independently, both assume the same output structure, and both use hardcoded magic strings (AMOUNT, PRICE,
LINE_ITEM) for field-type detection. No explicit contract between them — they're called serially in batch_processor.py without error handling between stages.
- Dependency category: Sequential pipeline with implicit contract
- Test impact: Field type detection logic is duplicated. A unified FieldProcessor registry driven by field_definitions.yaml would eliminate the magic strings and make
each field type independently testable.