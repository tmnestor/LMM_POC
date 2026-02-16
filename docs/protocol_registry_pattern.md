# Protocol + Registry: How We Support Multiple Vision-Language Models

## The Problem

We run document extraction with vision-language models. Today we have InternVL3 and Llama. Tomorrow we might add Qwen, Gemma, or a fine-tuned variant. Each model has a completely different API:

| | InternVL3 | Llama |
|---|---|---|
| **Load** | `AutoModel` + `AutoTokenizer` | `MllamaForConditionalGeneration` + `AutoProcessor` |
| **Infer** | `model.chat(tokenizer, image, prompt)` | `processor.apply_chat_template()` then `model.generate()` |
| **Batch** | `model.batch_chat()` with `num_patches_list` | Not supported |

Without a pattern, every time we add a model we'd need to scatter `if model == "llama": ...` conditionals throughout the pipeline, batch processor, bank statement adapter, CLI, and evaluation code.

---

## The Solution: Two Complementary Ideas

### 1. Protocol — "What must every model do?"

A **Protocol** is Python's way of defining an interface without inheritance. Think of it like scikit-learn's estimator contract: if your class has `fit()` and `predict()`, it works with `cross_val_score()`, `Pipeline`, `GridSearchCV`, etc. Nobody checks your class hierarchy — they just call the methods.

Our Protocol (`models/protocol.py`) defines the contract for document extraction, using Python 3.12's `TypedDict` to make return shapes self-documenting:

```python
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

class ClassificationResult(TypedDict):
    document_type: str
    confidence: float
    raw_response: str
    prompt_used: str
    error: NotRequired[str]           # Only present on fallback

class ExtractionResult(TypedDict):
    image_name: str
    extracted_data: dict[str, str]
    raw_response: str
    processing_time: float
    # ... plus metrics and optional fields

@runtime_checkable
class DocumentProcessor(Protocol):
    model: Any           # The loaded VLM
    tokenizer: Any       # Tokenizer or processor
    batch_size: int      # Current batch size

    def detect_and_classify_document(self, ...) -> ClassificationResult: ...
    def process_document_aware(self, ...) -> ExtractionResult: ...
```

**Key insight**: The pipeline code (`batch_processor.py`, `bank_statement_adapter.py`, `cli.py`) only calls these two methods. It never imports `DocumentAwareInternVL3HybridProcessor` or `DocumentAwareLlamaProcessor` directly. It doesn't know which model it's talking to — and it doesn't need to. And with `TypedDict` return types, IDEs provide autocomplete on the results without reading docstrings.

### 2. Registry — "How do we find and load models?"

The **Registry** (`models/registry.py`) is a lookup table that maps a string like `"llama"` to everything needed to load and configure that model:

```python
# Python 3.12 type aliases — self-documenting callable signatures
type ModelLoader = Callable[..., AbstractContextManager[tuple[Any, Any]]]
type ProcessorCreator = Callable[..., Any]

@dataclass
class ModelRegistration:
    model_type: str              # "internvl3", "llama"
    loader: ModelLoader          # Function that loads weights → (model, tokenizer)
    processor_creator: ProcessorCreator  # Wraps them → DocumentProcessor
    prompt_file: str             # "llama_prompts.yaml"
    description: str = ""

# The global registry
_REGISTRY: dict[str, ModelRegistration] = {}

def register_model(registration):
    _REGISTRY[registration.model_type] = registration

def get_model(model_type: str) -> ModelRegistration:
    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")
    return _REGISTRY[model_type]
```

Each model registers itself at the bottom of the file:

```python
register_model(ModelRegistration(
    model_type="internvl3",
    loader=_internvl3_loader,              # Loads InternVL3 weights
    processor_creator=_internvl3_creator,  # Wraps in DocumentProcessor
    prompt_file="internvl3_prompts.yaml",
    description="InternVL3.5-8B vision-language model",
))

register_model(ModelRegistration(
    model_type="llama",
    loader=_llama_loader,                  # Loads Llama weights
    processor_creator=_llama_creator,      # Wraps in DocumentProcessor
    prompt_file="llama_prompts.yaml",
    description="Llama 3.2 11B Vision Instruct",
))
```

**Critical design choice**: All `torch` and `transformers` imports live *inside* the loader/creator function bodies. Importing the registry module itself has zero GPU overhead. This means we can validate `--model llama` at CLI startup without loading 11 billion parameters.

---

## How It Flows End-to-End

```
User runs:  python cli.py --model llama -d ./images -o ./output

1. CLI parses --model "llama"
2. get_model("llama") → returns ModelRegistration
3. registration.loader(config) → loads Llama weights onto GPU → yields (model, processor)
4. registration.processor_creator(model, processor, ...) → returns DocumentAwareLlamaProcessor
5. Pipeline receives an object that satisfies DocumentProcessor Protocol
6. Pipeline calls .detect_and_classify_document() and .process_document_aware()
7. Pipeline never knows it's Llama — it just calls the interface
```

The pipeline also checks for optional batch capabilities at runtime using a second Protocol:

```python
# In models/protocol.py — separate Protocol for optional batch support
@runtime_checkable
class BatchCapableProcessor(Protocol):
    def batch_detect_documents(self, image_paths: list[str], ...) -> list[ClassificationResult]: ...
    def batch_extract_documents(self, image_paths: list[str], ...) -> list[ExtractionResult]: ...
```

```python
# In batch_processor.py — type-safe capability detection
from models.protocol import BatchCapableProcessor

if isinstance(self.model_handler, BatchCapableProcessor):
    # Use batched detection (InternVL3 supports this)
    results = self.model_handler.batch_detect_documents(image_paths)
else:
    # Fall back to sequential (Llama lands here)
    results = [self.model_handler.detect_and_classify_document(p) for p in image_paths]
```

This is a tiered contract: `DocumentProcessor` defines what's *required*, `BatchCapableProcessor` defines what's *optional*. `isinstance()` with a `@runtime_checkable` Protocol verifies all batch methods exist at once — a typo becomes an error, not a silent fallback. Models opt into batch support by implementing the methods — no flags, no configuration, no base class changes.

---

## The Analogy: scikit-learn Estimators

If you've worked with scikit-learn, you already understand this pattern:

| scikit-learn | Our Pipeline |
|---|---|
| `BaseEstimator` with `fit()`/`predict()` | `DocumentProcessor` Protocol with `detect()`/`extract()` |
| `sklearn.linear_model.LogisticRegression` | `DocumentAwareInternVL3HybridProcessor` |
| `sklearn.ensemble.RandomForestClassifier` | `DocumentAwareLlamaProcessor` |
| `Pipeline([...])` that calls `.fit()` | `BatchDocumentProcessor` that calls `.detect_and_classify_document()` |
| Swap models without changing pipeline code | Swap VLMs without changing pipeline code |

Like scikit-learn, we use both: inheritance for implementation sharing (`BaseDocumentProcessor`, similar to `BaseEstimator`) and structural typing for the consumer contract (`DocumentProcessor` Protocol). A processor *typically* inherits from the base class, but a third-party implementation only needs the right method signatures.

---

## What Happens When We Add a New Model?

Suppose we want to add **Qwen2.5-VL**. Here's exactly what we touch:

### Files we CREATE (3):

| File | Purpose |
|---|---|
| `models/document_aware_qwen_processor.py` | Inherit `BaseDocumentProcessor`, implement `generate()` + `_calculate_max_tokens()` + `process_single_image()` |
| `prompts/qwen_prompts.yaml` | Extraction prompts tuned for Qwen's instruction format |
| (add to) `models/registry.py` | `_qwen_loader()`, `_qwen_processor_creator()`, `register_model(...)` |

`BaseDocumentProcessor` provides `detect_and_classify_document()`, `process_document_aware()`, prompt loading, detection parsing, and bank structure classification out of the box.

### Files we DO NOT touch:

- `cli.py` — `--model qwen` works automatically (registry auto-discovery)
- `common/batch_processor.py` — calls Protocol methods, unchanged
- `common/bank_statement_adapter.py` — uses `processor.generate` callable, unchanged
- `common/unified_bank_extractor.py` — uses `generate_fn` callable, unchanged
- `common/evaluation_metrics.py` — compares dicts, model-agnostic
- All reporting and visualization code — model-agnostic

This is the **Open/Closed Principle** in practice: the pipeline is closed for modification but open for extension.

---

## Protocol + ABC: Both, Not Either/Or

We use **both** Protocol and ABC for complementary purposes:

- **Protocol** (`models/protocol.py`) — the *consumer-facing contract*. Pipeline code (`batch_processor.py`, `cli.py`, `bank_statement_adapter.py`) imports only the Protocol. It never imports a concrete processor class.
- **ABC** (`models/base_processor.py`) — the *implementation-sharing base class*. Both `InternVL3` and `Llama` processors inherit from `BaseDocumentProcessor` to share ~400 lines of detection, classification, prompt resolution, and extraction orchestration logic.

```python
# Consumer side: Protocol (structural typing)
from models.protocol import DocumentProcessor
def run_pipeline(processor: DocumentProcessor): ...  # Works with any conforming class

# Implementation side: ABC (code sharing)
from models.base_processor import BaseDocumentProcessor
class InternVL3Processor(BaseDocumentProcessor): ...  # Inherits shared logic
class LlamaProcessor(BaseDocumentProcessor): ...      # Same shared logic
```

This gives us:

1. **Decoupled consumers**: Pipeline code only knows about the Protocol. A third-party processor can satisfy `DocumentProcessor` without inheriting from `BaseDocumentProcessor`.

2. **DRY implementation**: Detection parsing, prompt loading, bank structure classification, and document-aware extraction are written once in the base class. Each model only implements `generate()`, `_calculate_max_tokens()`, and `process_single_image()`.

3. **Easy extensibility**: Adding a new model means inheriting from `BaseDocumentProcessor` and implementing ~3 methods instead of ~1,400 lines of duplicated logic.

---

## Summary

| Concept | What It Does | Where It Lives |
|---|---|---|
| **Protocol** | Defines the consumer-facing interface: "you must have these methods" | `models/protocol.py` |
| **Base class (ABC)** | Shares ~400 lines of implementation between processors | `models/base_processor.py` |
| **Registry** | Maps `"model_name"` string to loader + creator functions | `models/registry.py` |
| **Lazy loading** | Heavy imports (`torch`, `transformers`) deferred to function bodies | Inside registry loaders |
| **Optional Protocol** | `BatchCapableProcessor` — `isinstance()` check for batch methods | `models/protocol.py`, `common/batch_processor.py` |
| **generate_fn callable** | Model-agnostic generation for bank extraction | `common/unified_bank_extractor.py` |

Together, these give us a pipeline where adding a new vision-language model is a purely additive operation — inherit from `BaseDocumentProcessor`, implement ~3 methods, register in the registry. No existing code changes, no risk of breaking what already works.

---

## Multi-GPU: Zero Model Changes Required

The `MultiGPUOrchestrator` (`common/multi_gpu.py`) leverages the Protocol + Registry pattern to run parallel processing across multiple GPUs without any model-specific code:

```python
# For each GPU, the orchestrator calls the SAME registry functions:
gpu_config = replace(config, device_map=f"cuda:{gpu_id}")
model_ctx = load_model(gpu_config)          # registry.loader(config)
processor = create_processor(model, ...)     # registry.processor_creator(...)
```

Each GPU gets its own independent model + processor + bank adapter stack. The registry loaders already accept `device_map` from config, so multi-GPU support works automatically for any registered model — InternVL3, Llama, or future models.

This is the Protocol + Registry pattern paying dividends: because the pipeline code is model-agnostic, multi-GPU orchestration only needed to be implemented once in `multi_gpu.py`. It works with every model past and future.
