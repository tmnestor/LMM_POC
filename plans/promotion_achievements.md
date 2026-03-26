# Maurice (Tod) Nestor

**Data Scientist -- ATO**

Ferntree Gully, Victoria
tod.m.nestor@gmail.com | [LinkedIn](https://linkedin.com/in/todnestor) | [GitHub](https://github.com/tmnestor)

---

## Professional Summary

Data scientist with extensive academic preparation and scientific research experience. Began scientific programming in the mid-80's, progressing to scientific applications software development by the late 80's. Expertise in machine learning, natural language processing, and statistical modelling with a keen interest in the application of data science based solutions to complex business problems.

---

## Key Project: Automated Document Extraction using Vision-Language Models

**Document Understanding Group** | October 2025 -- March 2026

Built a production AI pipeline that extracts structured data from financial document images (receipts, invoices, bank statements), replacing manual data entry. Delivered ~1,058 commits across 12 feature branches over 5 months, progressing from proof-of-concept to multi-GPU production deployment.

---

## Criterion 1: Technical Proficiency in AI, Machine Learning, Data Visualisation, and Statistical Modelling

*Demonstrates APS values of being **Committed to service** through innovative, efficient solutions and **Impartial** through evidence-based technical decisions.*

**AI and Machine Learning**: Designed and delivered an end-to-end vision-language model (VLM) pipeline that classifies document types and extracts field-level data from images. Integrated and benchmarked 5 different pre-trained models to identify the best fit for production, rather than committing to a single model early. This evidence-based approach was important because each model architecture has different strengths -- some excel at structured tables, others at free-form text -- and the optimal choice could only be determined through rigorous comparative evaluation on real documents.

Developed a novel multi-turn conversational extraction protocol for bank statements, the most challenging document type. Single-pass extraction failed on bank statements because table layouts vary significantly between institutions (different column orders, date groupings, multi-line descriptions). The multi-turn approach decomposes this complex task into 4 sequential steps -- each step's output informs the next step's prompt -- enabling the system to handle layout variations that defeated simpler approaches.

**Data Visualisation**: Built model comparison dashboards with field-level accuracy heatmaps, radar charts, and executive summary visualisations. These were essential for communicating model selection recommendations to stakeholders who needed to understand performance trade-offs without reading raw metrics tables.

**Statistical Modelling**: Implemented F1, precision, and recall metrics with semantic normalisation for dates and monetary amounts -- because naive string matching would penalise correct extractions that differ only in format (e.g., "01/03/2026" vs "1 March 2026"). Applied Cohen's d effect size to determine whether accuracy differences between models were statistically meaningful rather than noise, ensuring model selection decisions were defensible.

---

## Criterion 2: Python Data Science Ecosystems, Cloud-Based Pipelines, and Pre-Trained Models

*Demonstrates **Committed to service** through efficient delivery and **Stewardship** through building lasting, maintainable infrastructure.*

**Python ecosystem**: Built the entire system in Python 3.12 using PyTorch, HuggingFace Transformers, Pandas, and Conda for environment management. Implemented batched GPU inference that processes multiple images simultaneously, achieving a 60% throughput improvement. This was important because document extraction at scale (hundreds of images per run) would otherwise be impractically slow for operational use.

**Cloud-based pipelines**: Decomposed the monolithic pipeline into three independently deployable stages (classify, extract, evaluate) for Kubeflow Pipelines. This decision was critical for production deployment because it allows each stage to be scaled, monitored, and restarted independently -- a failed extraction stage doesn't require re-running the classification stage, saving GPU compute time and enabling partial recovery. Configured AWS EFS integration for shared model weights across pipeline stages, with fail-fast validation to catch configuration errors before launching expensive GPU work.

**Applying pre-trained models**: Rather than training models from scratch (which would require months and large labelled datasets), applied and adapted 5 existing pre-trained vision-language models to the document extraction problem. This approach delivered production-ready accuracy (95% F1) in months rather than years, while the multi-model evaluation ensured the selected model was the strongest performer for this specific use case.

Planned and executed a GPU platform migration from V100 to A10G hardware when the original V100 GPUs proved too memory-constrained (16 GiB), forcing accuracy-degrading quantization. The migration to A10G (24 GiB, newer architecture) eliminated this compromise, but required resolving 15+ compatibility issues with the Flash Attention 2 library -- a critical dependency for memory-efficient inference at scale.

---

## Criterion 3: Clearly Communicates Complex Technical Information to Diverse Audiences

*Demonstrates **Accountable** through transparent communication and **Stewardship** through building institutional knowledge.*

**Technical audiences**: Authored design decision documents that explained not just what was built, but why alternative approaches were rejected. For example, documented the rationale for choosing thread-based parallelism over process-based parallelism for multi-GPU inference -- a decision that affects performance, debugging complexity, and maintainability. Wrote a detailed OOM diagnosis report tracing a production memory failure from symptoms through root cause to solution, with architecture diagrams -- enabling the team to understand and maintain the fix independently.

**Non-technical audiences**: Delivered Community of Practice presentations translating GPU computing and VLM architecture concepts into accessible explanations for colleagues across the organisation. Created executive model comparison dashboards that distilled complex statistical metrics into clear recommendations (which model to deploy, for which document types, and why). Wrote an evaluation methodology defence explaining why F1 score was chosen over simple accuracy and how ground truth data was validated.

**Knowledge artefacts**: Created troubleshooting documentation for Flash Attention 2 setup, a setup automation script for new team members, and a cloud migration guide assessing deployment options with Australian data residency requirements. These artefacts ensure the team's capability persists beyond individual contributors -- directly supporting the APS Stewardship value.

---

## Criterion 4: Effective and Ethical Solutions Supporting Responsible Governance

*Demonstrates **Ethical** conduct, **Accountable** transparency, and **Impartial** evidence-based advice.*

**Data sovereignty**: Designed the pipeline for on-premise deployment, ensuring sensitive financial documents are never transmitted to external services. This was a deliberate architectural decision -- while cloud AI APIs would have been simpler to integrate, they would have required sending enterprise financial data outside the organisation's control. Assessed cloud deployment alternatives (Amazon Bedrock, SageMaker) with explicit attention to Australian data residency requirements, documenting which options maintain data within approved regions.

**Responsible AI**: Established prompt authoring rules prohibiting real document data in model prompts, and built synthetic data generators for testing. This prevents production financial data from appearing in model context or development logs. Designed the evaluation framework to surface failures transparently -- per-field accuracy breakdowns and misclassification analysis ensure decision-makers see where the system underperforms, not just headline accuracy figures.

**Production robustness**: Implemented fail-fast validation, graceful degradation under memory pressure, and granular exit codes for operational monitoring -- ensuring the system fails visibly and recoverably rather than silently producing incorrect results.

---

## Criterion 5: Delivery Focused, Working Collaboratively Within Multi-Disciplinary Teams

*Demonstrates **Committed to service** through professional, collaborative, results-oriented delivery.*

**Delivery record**: Delivered ~1,058 commits across 12 feature branches over 5 months, with each branch representing a self-contained capability increment: initial prototype (Oct), bank statement extraction (Oct--Dec), model comparison (Nov), GPU migration (Nov--Feb), multi-model support and batch inference (Feb), KFP pipeline (Feb), multi-GPU parallelism (Feb--Mar), and transaction linking (Mar). Achieved measurable production outcomes: 95% F1 extraction accuracy, 4.25 images/min throughput on 4-GPU cluster, 5 document types supported.

**Cross-functional collaboration**: Worked with Data Engineering on pipeline integration (entrypoint configuration, shared storage paths, inter-stage file contracts) and with Infrastructure teams on GPU provisioning across 4 hardware architectures. Produced operational handover documentation (rollback guides, setup scripts, troubleshooting guides) to ensure support teams can maintain the system independently -- reflecting the Stewardship value of building capability that outlasts individual contributors.

---

## Criterion 6: Critical Thinking, Continuous Improvement, Problem Solving, and Innovation

*Demonstrates **Impartial** evidence-based reasoning and **Stewardship** through building lasting capability.*

**Evidence-based decision making**: Systematically evaluated 4 different approaches to distributing work across GPUs. The theoretically optimal approach (dynamic work-queue dispatch, where idle GPUs immediately pick up the next task) was implemented, measured, and found to be 60% slower than the simpler static approach -- because it eliminated the throughput benefits of batched inference. Rather than defending the more sophisticated solution, accepted the empirical evidence and reverted to the simpler, faster approach. This demonstrates a commitment to outcomes over complexity.

**Systematic problem solving**: The V100 to A10G GPU migration required investigating and resolving a cascade of interconnected issues: memory constraints forcing accuracy-degrading quantization, Flash Attention 2 incompatibility with the V100 architecture, and 15+ compatibility issues during the A10G migration (pre-built library failures, compiler ABI mismatches, shared storage workarounds). Each issue was documented, diagnosed, and resolved methodically over 3 months.

Traced a production memory failure through 7 iterative debugging cycles, ultimately discovering that the model's default attention mechanism was materialising a ~1.3 GiB matrix per inference -- exceeding GPU memory on complex documents. Developed a custom replacement that reduced memory usage from O(N^2) to O(N), resolving the failure. This required reading framework source code to understand the interaction between the model architecture and the GPU's memory management.

**Continuous improvement**: The bank statement extraction pipeline evolved through 4 major iterations -- from single-pass extraction (poor accuracy) through progressively more sophisticated multi-turn protocols, with each iteration driven by systematic error analysis on real documents. The evaluation framework similarly matured from simple accuracy to F1 metrics, semantic comparison, statistical significance testing, and confusion matrix analysis -- each addition addressing a specific gap identified during model comparison. Findings were documented and shared with the team at each stage.

**Innovation**: Extended document extraction into financial reconciliation by designing a multi-stage pipeline where the model matches receipts to bank statement transactions using explicit chain-of-thought reasoning, confidence scoring, and partial match classification -- demonstrating the potential for VLMs to automate complex financial workflows beyond simple data extraction.

---

## Education

**Master of Data Science (with Excellence)** | UNSW | 2021 -- 2022

**Graduate Diploma Data Science** | Monash University | 2019 -- 2020

**Doctor of Philosophy (Theoretical Geophysics)** | Australian National University | 1992 -- 1995
*John Conrad Jaeger Scholar*

**Master of Science (Geophysics)** | Monash University | 1989 -- 1991
*Australian Postgraduate Research Award*

**B.App.Sc (Mathematics) with Distinction** | RMIT | 1985 -- 1988
