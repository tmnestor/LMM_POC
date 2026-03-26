# LMM_POC Project: Evidence Against Selection Criteria

**Project**: Vision-Language Model Pipeline for Automated Document Extraction
**Period**: October 2025 -- March 2026 (5 months)
**Scale**: ~1,058 commits across 12 feature branches, 5 models integrated, 5 document types, deployed on multi-GPU production infrastructure

---

## Criterion 1: Technical Proficiency in AI, Machine Learning, Data Visualisation, and Statistical Modelling

### Artificial Intelligence -- Vision-Language Models

Designed and built an end-to-end AI pipeline that uses vision-language models (VLMs) to automatically extract structured data from document images. The system takes raw images of receipts, invoices, and bank statements and returns machine-readable field values -- replacing manual data entry.

- **Oct 2025**: Built initial single-model extraction pipeline with InternVL3.5-8B, including document type classification and field extraction
- **Oct--Dec 2025**: Developed multi-turn bank statement extraction -- a 4-turn conversational protocol (header detection, date format classification, table extraction, schema mapping) with dynamic column detection and mathematical balance correction
- **Feb 2026**: Integrated 5 pre-trained VLMs into a unified pipeline via a model-agnostic registry pattern: InternVL3.5-8B, InternVL3.5-20B-A4B (Mixture of Experts), Llama-3.2-11B-Vision, Qwen3-VL-8B, and GLM-OCR-0.9B
- **Mar 2026**: Solved a critical GPU memory failure by replacing the O(N^2) eager attention mechanism with a custom O(N) Scaled Dot-Product Attention (SDPA) implementation, requiring deep understanding of transformer attention internals, grouped-query attention head expansion, and PyTorch CUDA backends

### Machine Learning -- Document Classification and Extraction

- **Oct 2025**: Built two-phase ML pipeline: Phase 1 (VLM-based document type classification), Phase 2 (type-specific extraction with specialised prompts)
- **Feb 2026**: Implemented batched GPU inference with automatic OOM fallback (halving batch size recursively), achieving 60% throughput improvement over sequential processing
- **Feb--Mar 2026**: Designed multi-GPU data parallelism using ThreadPoolExecutor, exploiting PyTorch's GIL release during CUDA kernel execution for true parallel inference without serialisation overhead
- **Mar 2026**: Built adaptive per-image tile selection based on image quality assessment, balancing extraction accuracy against GPU memory constraints

### Data Visualisation

- **Nov 2025**: Built model comparison dashboards with field-level accuracy heatmaps, radar charts, and executive summary visualisations
- **Nov 2025**: Created confusion matrix analysis for document type classification across 3 model variants
- **Mar 2026**: Generated SVG evaluation visualisations for transaction linking accuracy, including lollipop charts for per-field F1 score differences

### Statistical Modelling

- **Nov 2025**: Implemented per-field F1, precision, and recall metrics with semantic normalisation for dates and monetary amounts
- **Nov 2025**: Applied Cohen's d effect size to quantify statistical significance of accuracy differences between model variants
- **Mar 2026**: Designed ground truth override experiments to isolate the causal impact of document misclassification on extraction accuracy

---

## Criterion 2: Hands-on Python Data Science Ecosystems, Cloud-Based Pipelines, and Pre-Trained Models

### Python Ecosystem

The entire system is built in Python 3.12, leveraging the modern data science and ML stack:

- **PyTorch**: GPU inference, CUDA memory management, mixed-precision (float16/bfloat16), `torch.nn.functional.scaled_dot_product_attention`
- **Transformers (HuggingFace)**: Model loading, tokenisation, `AutoModel`/`AutoTokenizer`, `AutoProcessor`, generation configuration
- **Pandas**: Ground truth CSV management, field-level metric aggregation, result comparison
- **Rich**: Console output, progress bars, GPU status tables
- **Pillow (PIL)**: Image preprocessing, adaptive tile selection based on image quality assessment
- **Conda**: YAML-driven environment management with Flash Attention 2 source builds

### Cloud-Based Data Science Pipelines

- **Feb 2026**: Decomposed the monolithic pipeline into three independently deployable stages for **Kubeflow Pipelines (KFP)**: `classify`, `extract`, `evaluate` -- each callable as a standalone CLI subcommand
- **Feb 2026**: Built `entrypoint.sh` with `KFP_TASK` dispatch routing, CUDA environment diagnostics, and multi-GPU auto-detection
- **Feb 2026**: Configured **AWS EFS** integration for shared model weights and evaluation data across pipeline stages
- **Feb 2026**: Implemented `--run-id` flag for deterministic inter-stage file paths, enabling reliable pipeline orchestration
- **Feb 2026**: Designed fail-fast validation that checks required environment variables and paths before launching GPU-intensive work

### Applying Pre-Trained Models to Real-World Problems

- **Oct 2025 -- Mar 2026**: Applied 5 different pre-trained vision-language models to the real-world problem of automated document data extraction for financial reconciliation
- **Mar 2026**: Extended extraction into transaction linking: a 3-stage pipeline that matches receipts to bank statement debits using chain-of-thought reasoning with amount gating (2% tolerance) and confidence scoring
- **Mar 2026**: Built synthetic data generators for receipts and bank statements to enable controlled testing without exposing production financial documents
- Managed a 3-machine development workflow: local macOS (linting), sandbox GPU (L40, experimentation), production GPU cluster (4x A10G, 24 GiB each)

---

## Criterion 3: Clearly Communicates Complex Technical Information to Diverse Audiences

### Technical Audiences

- **Feb 2026**: Authored detailed design decision documents explaining architectural choices: multi-GPU orchestrator rationale, pipeline vs data parallelism comparison
- **Mar 2026**: Wrote a comprehensive multi-GPU high-tile OOM diagnosis writeup tracing the problem from GPU memory budgets through attention matrix materialisation to the SDPA solution -- with Mermaid diagrams showing standard vs FlashAttention2 memory profiles
- **Feb 2026**: Documented the composable pipeline architecture with stage interfaces, entrypoint usage, and rollback guide for operations teams
- Produced architecture diagrams using Mermaid for pipeline flow, model dispatch, and GPU memory layout

### Non-Technical Audiences

- **Feb--Mar 2026**: Delivered Community of Practice (CoP) presentations on VLM architecture and project findings, translating GPU computing concepts into accessible explanations
- **Nov 2025**: Created executive model comparison dashboards distilling complex F1/precision/recall metrics into actionable recommendations (which model to deploy, for which document types)
- **Feb 2026**: Wrote evaluation methodology defence documents justifying the choice of metrics and ground truth management approach
- **Feb 2026**: Produced presentation materials on threading vs multiprocessing, pipeline vs data parallelism -- with visual diagrams designed for audiences without GPU computing background

### Knowledge Artefacts

- **Jan--Feb 2026**: Technical documentation for Flash Attention 2 troubleshooting (wheel compatibility, ABI issues, source builds) -- enabling other team members to set up GPU environments independently
- **Oct 2025**: Setup automation script (`LMM_POC_setup.sh`) with usage instructions, SSH key configuration, and Jupyter kernel registration
- **Jan 2026**: Amazon Bedrock migration guide assessing cloud deployment options with Australian data residency requirements

---

## Criterion 4: Effective and Ethical Solutions Supporting Responsible Governance and Appropriate Use of Enterprise Data

### Data Privacy and Sovereignty

- Designed the pipeline for **on-premise/private cloud deployment**, ensuring sensitive financial documents (bank statements, invoices, receipts) are never transmitted to external API services
- All VLMs run locally on organisation-controlled GPU infrastructure, maintaining full data custody
- **Jan 2026**: Assessed Amazon Bedrock migration with explicit attention to **Australian data residency requirements**, documenting which models are available in the Sydney region
- Evaluated AWS SageMaker as an alternative deployment path that maintains data within the organisation's VPC

### Responsible AI Practices

- Established **prompt authoring rules** prohibiting the use of real document data in prompts -- all examples use fictitious merchant names and amounts to prevent data leakage into model context
- **Mar 2026**: Built synthetic data generators for testing, eliminating the need to use production financial documents during development and experimentation
- **Nov 2025**: Implemented ground truth validation workflows with manual review and correction steps, ensuring evaluation integrity
- Designed the evaluation framework to surface model failures transparently -- per-field accuracy breakdown, misclassification impact analysis, and worst-performing image identification

### Production Robustness

- Implemented fail-fast validation patterns throughout: configuration errors surface immediately with actionable diagnostics rather than failing silently deep in the pipeline
- OOM fallback mechanisms ensure graceful degradation under memory pressure rather than data loss
- **Feb 2026**: Granular exit codes distinguish model loading failures from partial processing success, supporting operational monitoring

---

## Criterion 5: Delivery Focused, Working Collaboratively Within Multi-Disciplinary Teams

### Delivery Record

- Delivered ~1,058 commits across 12 feature branches over 5 months, progressing from initial prototype to production-grade multi-GPU pipeline
- Maintained a structured branching strategy: each major capability developed on a dedicated feature branch (`batch-inference`, `model-extensibility`, `feature/multi-gpu`, `feature/composable-pipeline`, etc.)
- Achieved measurable production metrics: 95.0% F1 extraction accuracy, 4.25 images/min throughput on 4x A10G, 5 document types supported

### Cross-Functional Collaboration

- Collaborated with **Data Engineering** on KFP pipeline integration: entrypoint scripting, EFS path configuration, environment variable contracts, and inter-stage file formats
- Worked with **Infrastructure/Platform** teams on GPU provisioning across V100, A10G, L4, and L40S architectures, including Flash Attention 2 compatibility testing and Conda environment management on shared NFS storage
- Produced operational documentation (rollback guides, setup scripts, troubleshooting guides) to enable handover to support teams
- Adapted pipeline configuration (YAML-driven, CLI flags) to accommodate different deployment environments without code changes

### Iterative Delivery Milestones

| Month | Milestone |
|-------|-----------|
| **Oct 2025** | Initial prototype: single-model extraction with InternVL3.5-8B |
| **Oct--Dec 2025** | Bank statement multi-turn extraction, structure classification, balance correction |
| **Nov 2025** | Model comparison framework: dashboards, F1 metrics, Cohen's d |
| **Jan 2026** | V100 -> A10G migration, Flash Attention 2 integration, GPU memory optimisation |
| **Feb 2026** | Multi-model CLI, batch inference, model registry, KFP composable pipeline |
| **Feb 2026** | Multi-GPU parallelism (ThreadPoolExecutor), 4x A10G production deployment |
| **Feb--Mar 2026** | CoP presentations, design decision documentation, evaluation methodology defence |
| **Mar 2026** | Transaction linking (receipt-to-bank matching), staged extraction pipeline |
| **Mar 2026** | SDPA attention fix, adaptive tile selection, GPU load balancing experiments |

---

## Criterion 6: Critical Thinking, Continuous Improvement, Problem Solving, and Innovation

### Systematic Problem Solving

- **SDPA attention OOM fix (Mar 2026)**: Traced a production GPU memory failure through 7 iterative commits -- from initial hypothesis (wrong attention implementation) through debugging attention output layouts, grouped-query attention head mismatches, and broadcastable mask handling to a verified solution. Required reading PyTorch and transformers source code to understand the interaction between the model's eager attention, the SDPA backend selection heuristics, and the flash/memory-efficient kernel eligibility criteria
- **GPU load balancing investigation (Mar 2026)**: Systematically evaluated 4 different approaches to distributing work across GPUs (contiguous partitioning, random shuffle, type-aware round-robin, dynamic work-queue dispatch). Each was implemented, measured on production data, and assessed against throughput metrics. Empirically determined that static partitioning with batched inference outperforms dynamic dispatch by 60% -- contradicting the theoretical expectation that dynamic dispatch would be optimal. Accepted the evidence and reverted to the simpler, faster approach

### Continuous Improvement

- **Bank statement extraction evolution (Oct 2025 -- Feb 2026)**: Progressed from single-turn flat extraction (poor accuracy on complex layouts) through multi-turn protocols, adding date format classification, dynamic column detection, and mathematical balance correction at each iteration -- driven by systematic error analysis on diverse real-world bank statement formats
- **Evaluation framework maturation (Nov 2025 -- Mar 2026)**: Started with simple accuracy, added F1/precision/recall, then semantic normalisation for dates and amounts, then Cohen's d for statistical significance, then confusion matrices for classification analysis -- each addition driven by a specific gap identified during model comparison
- **PyTorch OOM cleanup pattern (Feb 2026)**: Discovered that calling `torch.cuda.empty_cache()` inside Python `except` blocks is ineffective because the traceback holds references to activation tensors. Documented the correct pattern (flag-and-exit) for the team

### Innovation

- **Multi-turn bank statement extraction (Oct--Dec 2025)**: Designed a novel 4-turn conversational protocol that decomposes the complex task of extracting structured data from variable-format bank statements into manageable sub-tasks, with each turn's output informing the next turn's prompt. This approach handles layout variations that single-pass extraction cannot
- **Transaction linking with chain-of-thought (Mar 2026)**: Extended document extraction into financial reconciliation by designing a multi-stage pipeline where the VLM performs receipt-to-bank-debit matching with explicit reasoning (AMOUNT_CHECK, NAME_CHECK fields), confidence scoring, and partial match classification
- **Attention mechanism monkey-patching (Mar 2026)**: Developed a technique to replace the transformers library's global attention function registry at runtime, routing eager attention through PyTorch SDPA without modifying library source code -- enabling flash/memory-efficient backends on models that don't natively support them

---

## Quantitative Summary

| Metric | Value |
|--------|-------|
| Development period | 5 months (Oct 2025 -- Mar 2026) |
| Total commits | ~1,058 |
| Feature branches | 12 |
| Models integrated | 5 VLMs |
| Document types | 5 (receipt, invoice, bank statement, travel expense, vehicle logbook) |
| Best extraction accuracy | 95.0% F1 |
| Multi-GPU throughput | 4.25 images/min (4x A10G) |
| GPU architectures tested | 4 (V100, A10G, L4, L40S) |
| Composable pipeline stages | 3 (classify, extract, evaluate) |
| Bank statement extraction turns | 4 conversational turns |
