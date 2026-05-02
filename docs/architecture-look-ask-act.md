# Production Architecture: Vision-Language Structured Extraction and the Look-Ask-Act Pattern

## Background

This document describes the agentic architecture of the LMM_POC Vision-Language
structured information extraction pipeline, developed from June 2025 onward.
It maps the production design against the standard **Look-Ask-Act** agentic loop
and explains how hardware constraints shaped architectural decisions.

---

## The Standard Look-Ask-Act Loop

```
┌──────┐     ┌─────┐     ┌─────┐
│ LOOK │ ──► │ ASK │ ──► │ ACT │ ──► (loop)
└──────┘     └─────┘     └─────┘
 Perceive     Reason       Execute
```

| Step | Role |
|------|------|
| **Look** | The agent perceives its environment (visual input, state) |
| **Ask** | The agent reasons about what it has observed and plans next action |
| **Act** | The agent executes — produces output, updates state, routes forward |

---

## How Our Pipeline Maps to Look-Ask-Act

| Loop Step | Our Implementation | Key Code |
|---|---|---|
| **Look** | VLM observes the document image | `generate_fn(image, prompt, params)` |
| **Ask** | Router evaluates detection output, selects extraction strategy, injects upstream results into next prompt | `_evaluate_router()` + `_resolve_inject()` |
| **Act** | Type-specific structured extraction → validated parsed output stored in typed state | `TurnParser.parse()` + `WorkflowState` |
| **Reflect** | On parse failure: reflection prompt injected → bounded retry | Self-Refine edge (`parse_failed`) |

The graph is defined declaratively in YAML. Each node is a Look-Ask-Act
micro-cycle; the graph as a whole is a macro Look-Ask-Act pipeline.

---

## Two Genuine Agentic Loops

### Loop 1 — Self-Refine (single document, bounded retry)

```
Look (image) → Act (extract) → parse fails?
    └── Ask (reflect: "you produced X, error was Y, try again")
        └── Look (image + error context) → Act (retry)
```

Triggered by parse failure. Bounded by `max_retries` in the YAML node
definition — prevents runaway loops under production load.

### Loop 2 — Bank Statement Multi-Turn (across pages)

```
Look (page 1) → Act (extract transactions)
    └── Ask (inject running state into page 2 prompt)
        └── Look (page 2) → Act → ... → final assembly
```

Multi-page bank statements exceed single-turn context limits. The
`BankStatementAdapter` maintains running state across turns, implementing
a genuine stateful Look-Ask-Act loop at the page level.

---

## Why Hardware Constraints Altered the Design

A naive Look-Ask-Act VLM agent would run an open-ended multi-turn loop,
dynamically deciding what to look at next. Production hardware prevented this.

| Hardware Constraint | Architectural Consequence |
|---|---|
| **A10G: 24 GB VRAM** — open-ended multi-turn exhausts KV cache at scale | Replaced open loop with **deterministic YAML graph** — bounded turns, predictable memory footprint |
| **V100: no BFloat16, no FlashAttn2** — silent accuracy degradation | **SDPA monkey-patch fallback** + explicit dtype gating; hardware declared in config, never auto-assumed |
| **KFP step deadline (~55 min limit)** — CUDA graph compilation alone consumed ~50 min | Defaulted `enforce_eager=True`; made configurable so large jobs can amortise compilation cost |
| **1,000+ image production jobs** — sequential Look-Ask-Act throughput too slow | **vLLM backend** with PagedAttention + data-parallel dispatch; Self-Refine retry only on parse failure, not exploratory reasoning |
| **Multi-page bank statements** — one-shot extraction exceeds context window | Sequential **multi-turn adapter** with running state injection rather than single long-context prompt |
| **M1 Pro (MPS, no CUDA)** — CUDA-specific libraries unavailable | **INT8 quantization via optimum-quanto** (bitsandbytes unsupported on MPS); load to CPU then remap to MPS to avoid Metal per-buffer size limits |

---

## How the Architecture Was Made More Modular

### The Problem: Tight Coupling in the Original Two-Model Design

The pipeline launched with InternVL3 and Llama 3.2 as the only supported
backends. Each required:

- A hand-written loader function (~200 lines) managing `from_pretrained`,
  dtype selection, device placement, and generation config fixups
- A hand-written `processor_creator` function constructing the document
  processor from model + tokenizer
- Model-specific `if model_type == "internvl3"` branches scattered across
  `cli.py`, `batch_processor.py`, and the bank statement adapter

Adding a third model meant touching five or more files, with high risk of
regressions in the existing two paths.

### The Solution: Declarative Model Registry

Modularity was introduced in three layers:

**1. Central registry (`models/registry.py`)**

A `ModelRegistration` dataclass and a `_REGISTRY` dict decouple model
identity from pipeline logic. The pipeline calls `get_model(model_type)` —
it never imports model-specific modules directly. All heavy imports
(`torch`, `transformers`) are deferred to function bodies, so importing the
registry has zero GPU/ML overhead.

**2. Declarative specs (`ModelSpec` / `VllmSpec` in `models/model_loader.py`)**

Each new model is declared as a ~20-line dataclass instance rather than
~200 lines of imperative loader code. Common concerns — dtype selection,
device placement, flash-attention gating, quantization config, generation
warning suppression — are handled once in the loader and driven by spec
fields:

```python
register_hf_model(ModelSpec(
    model_type="qwen3vl",
    model_class="Qwen3VLForConditionalGeneration",
    prompt_file="qwen3vl_prompts.yaml",
    attn_implementation="flash_attention_2",
    suppress_gen_warnings=("temperature", "top_p", "top_k"),
    message_style="two_step",
))
```

**3. Backend abstraction (`models/backends/`)**

Model-specific generation APIs are encapsulated behind a uniform interface:

| Backend | Models | API |
|---|---|---|
| `InternVL3Backend` | InternVL3 family | `.chat()` / `.batch_chat()` |
| `LlamaBackend` | Llama 3.2 | `processor.apply_chat_template()` + `.generate()` |
| `HFChatTemplateBackend` | Llama 4, Qwen3-VL, Qwen3.5, Nemotron, Granite 4, Gemma 4 | `processor.apply_chat_template()` + `.generate()` (two-step or one-step) |
| `VllmBackend` | All vLLM variants | `llm.generate()` via OpenAI-compatible messages |

The document processor (`DocumentAwareProcessor`) calls
`backend.chat(image, prompt)` — it never branches on model type.

### Why Modularity Was Necessary

The pipeline grew from 2 models to 10+ models across four hardware targets
(V100, A10G/L40, M1 MPS) over twelve months. Each new model differed in:

- Loading API (`AutoModel` vs `AutoModelForCausalLM` vs `AutoModelForImageTextToText`)
- Generation API (`.chat()`, `.generate()`, vLLM)
- Chat template format (InternVL system tags, Llama `<|image|>` tokens, Jinja2 templates)
- Quantization requirements (BitsAndBytes NF4, optimum-quanto INT8/INT4)
- Hardware constraints (CUDA-only libraries, MPS buffer limits, KV cache sizing)

Without the registry and backend abstraction, each addition would have
required modifying `cli.py`, `batch_processor.py`, the bank statement
adapter, and the evaluation harness — four files per model, with compounding
regression risk across existing paths.

### The Result

| Action | Code required |
|---|---|
| Add a new HF model | ~20-line `ModelSpec` in `registry.py` — zero pipeline changes |
| Add a new vLLM model | ~10-line `VllmSpec` in `registry.py` — zero pipeline changes |
| Add a new hardware target | New `post_load` hook + spec fields — no model logic changes |
| Swap models at runtime | `--model <type>` CLI flag — no code changes |

Ten models (InternVL3-8B/14B/38B, Llama 3.2, Llama 4 Scout, Qwen3-VL,
Qwen3.5, NVIDIA Nemotron, IBM Granite 4, Gemma 4) were added with zero
modifications to the pipeline execution layer between additions.

---

## Why Not a Standard Agent Framework?

Off-the-shelf frameworks (LangChain, LangGraph) assume a single-GPU
development environment. This pipeline required:

- **Hardware-aware model loading** across V100, A10G, L40, and M1 MPS
- **Automatic OOM recovery** with recursive batch halving
- **Configurable CUDA graph compilation** (`enforce_eager`)
- **Dtype and attention backend gating** per hardware generation
- **Data-parallel dispatch** for 1,000+ image jobs without framework overhead

The `GraphExecutor` provides the Look-Ask-Act structure with full control over
the execution environment — something no off-the-shelf agent framework
offered at the required level of hardware specificity.

---

## The One-Sentence Synthesis

> *"We implemented a hardware-constrained variant of Look-Ask-Act: replacing
> the open agentic loop with a YAML-defined DAG that preserves the
> perceive-reason-act structure but bounds memory usage, constrains KV cache
> growth, and enables data-parallel throughput at scale — while retaining
> genuine agentic behaviour where it adds value: bounded Self-Refine retries
> on parse failure, and sequential multi-turn state injection for multi-page
> documents."*

---

## Timeline

Architecture developed June 2025 – May 2026 across the following model
backends: InternVL3-8B/14B/38B, Llama 3.2 11B, Llama 4 Scout, Qwen3-VL,
Qwen3.5, NVIDIA Nemotron, IBM Granite 4, Gemma 4. Deployed on V100, A10G,
L40, and Apple M1 Pro (MPS).
