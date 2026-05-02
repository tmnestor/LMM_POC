# Production Architecture: Vision-Language Structured Extraction and the Look-Ask-Act Pattern

## STAR Summary

### Situation

The requirement was to build a structured information extraction pipeline
for heterogeneous financial documents — receipts, invoices, bank statements,
and an expanding set of document classes — running entirely self-hosted on
an underpowered GPU cluster with no access to frontier model APIs or managed
inference services. Sensitive financial data could not leave the organisation.

Three dimensions of the requirement made this architecturally challenging.
First, document classes were known to grow: each class has different layouts,
field schemas, and extraction complexity, and the system had to accommodate
new classes without structural rework. Second, the open-weight VLM landscape
was evolving rapidly: better models were appearing frequently, each with
different loading, quantization, and generation requirements. The system had
to be able to adopt new models without re-engineering the pipeline around them.
Third, production volume demanded throughput: sequential inference using the
standard HuggingFace framework was too slow to process large document jobs
within the operational time constraints of the GPU cluster.

The core architectural challenge was therefore not just building the pipeline,
but building it so that growth across all three dimensions — more document
classes, better models, and higher throughput — would be absorbed without
structural rework.

### Task

Design an architecture that could grow across all three dimensions — new
document classes, new models, and higher throughput — without requiring
changes to the execution layer.
The pipeline also had to handle the inherent uncertainty of document
processing: document type is not known in advance, extraction can fail to
parse, and some document types require multi-turn reasoning across variable
structure. The architecture had to accommodate adaptive behaviour while
remaining maintainable and predictable at scale.

### Action

**Why an agentic pattern was needed.**
A static pipeline with fixed steps cannot handle document processing: you
do not know the document type until you have looked at it, and the
extraction strategy depends on what you find. The task is inherently
perceive-then-decide — the natural structure of Look-Ask-Act. The pipeline
first perceives the document (classify), then reasons about what it observed
(select extraction strategy), then acts (extract structured fields). When
extraction fails to parse, it reflects on the error and retries — a bounded
self-correction cycle. For multi-page documents, it maintains running state
across turns, injecting prior results into each subsequent prompt.

**How the agentic pattern was implemented.**
The workflow is defined as a directed graph in YAML — nodes are VLM
inference calls, edges are routing conditions based on detection output or
parse results. Each node is a complete perceive-reason-act micro-cycle; the
graph as a whole is the macro pipeline. Retry cycles are explicit in the
graph and bounded by configuration, preventing runaway loops under
production load.

**How maintainability was improved through declarative specifications.**
The design philosophy was to minimise coupling between concerns and maximise
cohesion within them. In practice, this meant expressing every concern that
changes frequently as a declarative specification rather than as code:

- *Model specifications* — each model's loading requirements, hardware
  constraints, quantization, and generation parameters are declared once.
  The execution layer never branches on model identity; it reads the
  specification and applies it.
- *Document type definitions* — the fields to extract, their types,
  validation rules, and evaluation weights are declared as configuration.
  Adding a new document class means adding a specification entry and a
  prompt — no Python changes, no risk to existing classes.
- *Extraction prompts* — each document class has its own prompt, versioned
  independently. Prompt tuning, which drives the majority of accuracy
  improvement, is a YAML edit isolated to one document class.
- *Workflow graph* — routing logic, retry bounds, and extraction strategies
  are declared in YAML. Changing pipeline behaviour is a configuration
  change, not a code change.

The result is that the execution engine is stable and generic — it runs
the graph, it does not contain the graph.

**How the throughput challenge was solved.**
Standard HuggingFace inference loads one model instance per GPU and
processes documents sequentially. At production volume — hundreds to
thousands of documents per job — this was too slow. The solution was to
introduce a second inference backend: vLLM, a high-throughput inference
engine that uses PagedAttention to manage GPU memory more efficiently, and
that supports running multiple model instances in a data-parallel
configuration across available GPUs. Each worker processes an independent
partition of the document batch, and results are reassembled after all
workers complete.

The modular backend design meant that adding vLLM as an inference backend
required no changes to the pipeline execution layer or the YAML workflow
graph. The same document type definitions, prompts, and routing logic work
identically regardless of whether the model is served via HuggingFace or
vLLM. The backend is a swappable component; the pipeline does not know or
care which one is in use.

### Result

A self-hosted pipeline supporting over ten open-weight model variants across
four hardware targets and multiple document classes, with no external API
dependencies.

**Agility.** Adding a new model requires a single declarative registration.
Adding a new document class requires a field definition and a prompt.
Switching to a higher-throughput inference backend required no pipeline
changes. Each of these operations is bounded, predictable, and safe —
the scope of change is narrow by design, so the team can respond quickly
to new requirements without the risk of disturbing working functionality.

**Reduced cognitive complexity.** Because each concern — model behaviour,
document structure, hardware configuration, workflow topology — has exactly
one place it lives, engineers do not need to hold the whole system in mind
to make a change. A prompt engineer works in prompt files. A model
integration touches only the registry. A hardware configuration is a
YAML edit. The system is navigable because its boundaries are clear.

**Easier maintenance.** Low coupling means that failures are localised and
their causes are findable. High cohesion means that a change in one area
does not produce unexpected effects in another. The declarative
specifications serve as living documentation — the configuration files
describe what the system does, not just how it was built. Onboarding a new
engineer, debugging an extraction failure, or extending the system to a
new context all start from the same readable, structured source of truth.

These outcomes were not accidental. They followed from a single design
principle applied consistently across every architectural decision:

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

**Uniform interfaces between components.**
Every model backend exposes the same interface to the pipeline: given an
image and a prompt, return a text response. The differences in how models
load, quantize, and generate are fully encapsulated behind this boundary.
The orchestration layer calls one method regardless of what is behind it.

**Configuration over code for variation.**
Hardware-specific concerns — dtype selection, attention backend, device
placement, memory limits — are expressed as configuration, not conditional
logic. Adding support for a new hardware target means supplying new
configuration values, not modifying execution paths.

**Hooks for model-specific behaviour.**
Where a model requires behaviour that cannot be expressed as a configuration
value — fixing generation config after loading, tying weights, constructing
a model-specific generation backend — the spec declares a callable hook.
The loader runs the hook if one is present; it does not know what the hook
does. This extends the declarative pattern from static configuration to
behaviour: model-specific setup lives in the model's declaration, not in
branching logic inside the loader. The same principle applies to backend
construction — each model family declares a factory hook that builds its
own generation backend, keeping backend-specific knowledge fully
encapsulated.

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

> *"The design philosophy was to minimise coupling between concerns — model
> identity, hardware configuration, workflow topology, and extraction logic
> are each declared independently — while maximising cohesion within each
> concern, so that every model-specific, hardware-specific, or document-type-specific
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
