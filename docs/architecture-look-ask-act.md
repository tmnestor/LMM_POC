# Production Architecture: Vision-Language Structured Extraction and the Look-Ask-Act Pattern

## STAR Summary

### Situation

A complex structured information extraction task running entirely
self-hosted on an underpowered GPU cluster — no frontier model APIs, no
managed inference services. The pipeline processed heterogeneous financial
documents across multiple document types with different layouts and field
schemas. It operated in a rapidly evolving model landscape where open-weight
VLMs were improving quickly, but each model required different loading,
quantization, and generation handling. As scope expanded, the cost of
change escalated: each new model or document type required modifications
across multiple layers of the codebase, with compounding regression risk
across existing paths.

### Task

Design an architecture for a self-hosted VLM extraction pipeline that could
absorb this complexity — supporting new models as they become available,
extending to new document types, adapting to hardware constraints — without
requiring changes to the execution layer and without accumulating fragility
over time. The key constraint: extraction quality is ultimately bounded by
model capability, so the architecture had to make model substitution as
frictionless as possible.

### Action

Apply a consistent design philosophy across every layer of the pipeline:
minimise coupling between layers, maximise cohesion within them. The
specific decisions — declarative model registry, YAML-defined workflow
graph, uniform backend interfaces, domain-typed state — all follow from
this principle. The rest of this document explains each decision and its
justification.

### Result

A self-hosted pipeline that supports over ten open-weight model variants
across four hardware targets, with multiple document types — all running
without external API dependencies. Adding a new model requires a single
declarative registration with no changes to the execution layer. As the
open-weight model landscape continues to improve, extraction quality
improves by swapping in a better model — a configuration change, not an
engineering project. The architecture ensures that progress in the VLM
field translates directly into pipeline capability without rework.

---

## Design Philosophy

> *"Minimise coupling between layers. Maximise cohesion within them."*

Every architectural decision in this pipeline traces back to this principle.
Model identity, hardware configuration, workflow topology, and extraction
logic are each declared independently — changes in one do not ripple into
the others. Within each layer, every model-specific, hardware-specific, or
document-type-specific concern has exactly one place it belongs.

This is what allows the pipeline to support over ten model variants, four
hardware targets, and multiple document types — with additions that require
no modification to the execution layer, and changes whose scope is
predictably narrow.

---

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

## Maintainability and Extensibility in Practice

The architectural choices described above have a direct and measurable
impact on how the codebase behaves as requirements change. Two scenarios
illustrate this concretely.

### Adding a new document type

Supporting a new document type — a travel expense claim, a purchase order,
a remittance advice — requires:

1. A field definition entry declaring what to extract and how to evaluate it
2. An extraction prompt in the relevant prompt file
3. A document type alias entry if the model uses variant names

That is the complete change. The detection step, the routing logic, the
extraction engine, the evaluation harness, and the output layer all pick
up the new type automatically. No existing code paths are touched, and
therefore no existing document types are at risk of regression.

This is possible because the pipeline does not contain a list of supported
document types. It reads that list from configuration at startup. The
execution engine is generic over document types; the types themselves are
data.

### Adding a new vision-language model

Supporting a new VLM requires a single declarative registration: the model
class, its prompt file, hardware requirements, and any generation parameter
adjustments. If the model uses a generation API already covered by an
existing backend, no new code is required at all. If it uses a genuinely
new API, a backend can be written in isolation — a self-contained unit with
a single responsibility — without touching the pipeline.

In either case, every existing model continues to work unchanged. The
registry lookup is the only point of contact between the pipeline and the
model; nothing else in the execution layer is aware that a new model was
added.

### Why this matters beyond the initial build

Structured extraction systems are not built once and left. Prompts are
tuned continuously as edge cases surface. Document types are added as scope
expands. Models are replaced as better ones become available. Hardware
environments change.

An architecture that requires code changes for each of these events
accumulates fragility over time: each change is a potential regression,
each regression requires a test cycle, and the test cycle slows the
iteration that drives accuracy improvement.

The design choices here — YAML-declared configuration, uniform interfaces,
declarative registrations — were made specifically to keep the cost of
change low and the scope of each change narrow. The goal was a system
where the most frequent operations (prompt tuning, model swapping, document
type extension) carry the lowest possible risk and require the least
possible engineering involvement.

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

Defining pipeline behaviour in YAML separates concerns that change at
different rates and for different reasons. The extraction logic — what
fields to extract, what prompts to use, how to classify document structure,
how to evaluate output quality — changes frequently as the system is tuned
and extended. The execution engine — how to run a node, recover from
failure, route between states — changes rarely. Mixing these in code couples
fast-moving configuration to slow-moving infrastructure, making every
tuning iteration a code change with its associated review and deployment
cost.

In practice, the following are declared entirely in YAML and require no
Python changes to modify:

**Extraction prompts per document type.** Each document type has its own
prompt, independently versioned. Tuning a prompt for better invoice
extraction has no risk of affecting receipt or bank statement extraction.
Prompt iteration — which accounts for the majority of accuracy improvement
work — is a YAML edit, not a code change.

**Field definitions and document taxonomy.** The fields to extract for each
document type, their data types, validation rules, and evaluation weights
are declared as configuration. Adding a new document type means adding a
field definition entry and a prompt — the extraction engine requires no
modification. The document type taxonomy, including aliases and
normalisation rules, is similarly declared rather than hardcoded.

**Document structure detection rules.** For complex document types such as
bank statements, the rules for detecting structural features — which column
headers indicate which semantic roles, which detected patterns trigger which
extraction strategy — are declared as configuration. Domain knowledge about
document structure is expressed where it belongs: as inspectable,
modifiable configuration rather than embedded logic.

**Model and hardware configuration.** Generation parameters, hardware
placement, quantization settings, and attention backend selection are
declared per deployment target. Moving between hardware environments is a
configuration change, not a code change.

This separation is what allows the pipeline to support multiple document
types, multiple models, and multiple hardware targets without the execution
layer accumulating branches for each combination. The execution engine
reads its instructions from configuration; it does not contain them.

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

> *"The design philosophy was to minimise coupling between layers — model
> identity, hardware configuration, workflow topology, and extraction logic
> are each declared independently — while maximising cohesion within layers,
> so that every model-specific, hardware-specific, or document-type-specific
> concern has exactly one place it belongs. Applied to the Look-Ask-Act
> pattern, this produced a hardware-constrained directed graph that preserves
> the perceive-reason-act structure, bounds memory usage and KV cache growth,
> and enables data-parallel throughput at scale — while retaining genuine
> agentic behaviour where it adds value: bounded Self-Refine retries on parse
> failure, and sequential multi-turn state injection for multi-page documents."*

---

## Timeline

Architecture developed June 2025 – May 2026 across the following model
backends: InternVL3-8B/14B/38B, Llama 3.2 11B, Llama 4 Scout, Qwen3-VL,
Qwen3.5, NVIDIA Nemotron, IBM Granite 4, Gemma 4. Deployed on V100, A10G,
L40, and Apple M1 Pro (MPS).
