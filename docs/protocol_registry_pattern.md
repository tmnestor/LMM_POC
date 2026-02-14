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

Our Protocol (`models/protocol.py`) defines the contract for document extraction:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DocumentProcessor(Protocol):
    model: Any           # The loaded VLM
    tokenizer: Any       # Tokenizer or processor
    batch_size: int      # Current batch size

    def detect_and_classify_document(self, image_path, verbose=False) -> dict:
        """Given an image, return {'document_type': 'INVOICE', ...}"""
        ...

    def process_document_aware(self, image_path, classification_info, verbose=False) -> dict:
        """Given an image + its type, return {'extracted_data': {...}}"""
        ...
```

**Key insight**: The pipeline code (`batch_processor.py`, `bank_statement_adapter.py`, `cli.py`) only calls these two methods. It never imports `DocumentAwareInternVL3HybridProcessor` or `DocumentAwareLlamaProcessor` directly. It doesn't know which model it's talking to — and it doesn't need to.

### 2. Registry — "How do we find and load models?"

The **Registry** (`models/registry.py`) is a lookup table that maps a string like `"llama"` to everything needed to load and configure that model:

```python
@dataclass
class ModelRegistration:
    model_type: str              # "internvl3", "llama"
    loader: Callable             # Function that loads weights → (model, tokenizer)
    processor_creator: Callable  # Function that wraps them → DocumentProcessor
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

The pipeline also checks for optional batch capabilities at runtime:

```python
# In batch_processor.py — duck typing for optional features
if hasattr(self.model_handler, "batch_detect_documents"):
    # Use batched detection (InternVL3 supports this)
    results = self.model_handler.batch_detect_documents(image_paths)
else:
    # Fall back to sequential (Llama lands here)
    results = [self.model_handler.detect_and_classify_document(p) for p in image_paths]
```

This is a tiered contract: the Protocol defines what's *required*, and `hasattr()` checks discover what's *optional*. Models opt into batch support by implementing the methods — no flags, no configuration, no base class changes.

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

The difference: scikit-learn uses inheritance (`BaseEstimator`), we use structural typing (Python `Protocol`). No base class required — if your class has the right methods, it works.

---

## What Happens When We Add a New Model?

Suppose we want to add **Qwen2.5-VL**. Here's exactly what we touch:

### Files we CREATE (3):

| File | Purpose |
|---|---|
| `models/document_aware_qwen_processor.py` | Implement `detect_and_classify_document()` + `process_document_aware()` using Qwen's API |
| `prompts/qwen_prompts.yaml` | Extraction prompts tuned for Qwen's instruction format |
| (add to) `models/registry.py` | `_qwen_loader()`, `_qwen_processor_creator()`, `register_model(...)` |

### Files we DO NOT touch:

- `cli.py` — `--model qwen` works automatically (registry auto-discovery)
- `common/batch_processor.py` — calls Protocol methods, unchanged
- `common/bank_statement_adapter.py` — dispatches via Protocol, unchanged
- `common/evaluation_metrics.py` — compares dicts, model-agnostic
- All reporting and visualization code — model-agnostic

This is the **Open/Closed Principle** in practice: the pipeline is closed for modification but open for extension.

---

## Why Not Just Use Inheritance?

A reasonable question. We could have done:

```python
class BaseDocumentProcessor(ABC):
    @abstractmethod
    def detect_and_classify_document(self, ...): ...

class InternVL3Processor(BaseDocumentProcessor): ...
class LlamaProcessor(BaseDocumentProcessor): ...
```

We chose Protocol (structural typing) over ABC (nominal typing) for three reasons:

1. **No coupling**: Processors don't import or inherit from anything in the pipeline. They're standalone modules that happen to have the right methods.

2. **Gradual adoption**: We started with duck typing (just calling methods). The Protocol formalised the existing contract without requiring any processor to change.

3. **Third-party friendliness**: If someone wraps a HuggingFace pipeline or an API client, they don't need to inherit from our base class. They just need the right method signatures.

---

## Summary

| Concept | What It Does | Where It Lives |
|---|---|---|
| **Protocol** | Defines the interface: "you must have these methods" | `models/protocol.py` |
| **Registry** | Maps `"model_name"` string to loader + creator functions | `models/registry.py` |
| **Lazy loading** | Heavy imports (`torch`, `transformers`) deferred to function bodies | Inside registry loaders |
| **Duck-typed optionals** | `hasattr()` checks for batch methods at runtime | `common/batch_processor.py` |

Together, these give us a pipeline where adding a new vision-language model is a purely additive operation — no existing code changes, no risk of breaking what already works.
