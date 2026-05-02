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

### Why Modularity Was Necessary

The pipeline launched with two tightly coupled model backends. Each new model
required changes in multiple places — loading logic, generation logic, and
pipeline orchestration — with growing risk of regressions across existing
paths. As the scope expanded to cover more models and more hardware targets,
it became clear that adding capability through direct modification was
unsustainable.

The core problem was **coupling between model identity and pipeline logic**.
The execution layer should not need to know which model is loaded — it
should only know how to run a Look-Ask-Act cycle.

### How It Was Solved

Modularity was achieved through three principles applied consistently:

**Separation of declaration from execution.**
Each model is declared once — its loading requirements, hardware constraints,
quantization needs, and generation parameters — in a single registration
entry. The pipeline reads this declaration at startup; it never branches on
model identity at runtime.

**Uniform interfaces between layers.**
Every model backend exposes the same interface to the pipeline: given an
image and a prompt, return a text response. The differences in how models
load, quantize, and generate are fully encapsulated behind this boundary.
The orchestration layer calls one method regardless of what is behind it.

**Configuration over code for variation.**
Hardware-specific concerns — dtype selection, attention backend, device
placement, memory limits — are expressed as configuration, not conditional
logic. Adding support for a new hardware target means supplying new
configuration values, not modifying execution paths.

### The Result

The pipeline now supports over ten model variants across four hardware
targets. Adding a new model requires a single declarative registration —
no changes to the pipeline execution layer, no risk to existing paths.
Swapping models at runtime is a one-flag CLI change. Hardware targets can
be added or extended without touching model logic.

This extensibility was a direct consequence of designing for separation
of concerns from the point the architecture became multi-model, rather
than retrofitting it later.

---

## Why a Custom Graph Implementation?

The pipeline includes its own graph execution engine: a directed acyclic
graph of nodes and edges, with typed state, conditional routing, and
declarative YAML definitions. This is the same conceptual territory as
frameworks such as LangGraph. The decision to build a custom implementation
rather than adopt one was deliberate.

### What frameworks do not provide

Agent graph frameworks handle workflow orchestration. They do not handle
model loading, hardware placement, memory management, or inference
throughput. Those concerns are assumed to sit behind an API. For this
pipeline, they were not — the pipeline owns the model, the device, the
memory budget, and the execution path end to end. A framework would have
orchestrated the graph while leaving the harder problems unaddressed.

### Why YAML-declarative graph definition

Defining the workflow graph in YAML — nodes, edges, routing conditions,
retry bounds — separates pipeline behaviour from pipeline code. Analysts
can modify extraction workflows, add document types, or adjust retry
policies without touching Python. A Python-native graph definition (as
frameworks typically provide) would have coupled workflow changes to code
changes, requiring engineering involvement for configuration-level decisions.

### Why typed, domain-specific state

The state object carried through the graph is typed to the extraction
domain: it holds structured fields for each document type, parse results,
confidence scores, and retry counters. This is richer than a generic
key-value state store. The typing enables static analysis, catches
integration errors at development time, and makes the state contract
explicit across every node in the graph.

### The result

A graph execution engine that is purpose-built for the problem: declarative
workflow definition, domain-typed state, hardware-aware execution, and full
ownership of the inference layer — without the abstraction overhead of a
general-purpose framework.

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
