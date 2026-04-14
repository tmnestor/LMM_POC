# Simple Processor Refactor: Eliminating 96% Duplication

## Problem Statement

Five "simple" processors (Llama4, Qwen3VL, Qwen3.5, Nemotron, vLLM) are 96%+ identical.
Only `_load_model()` and `generate()` truly differ. Everything else -- `__init__()`,
`_configure_generation()`, `_calculate_max_tokens()`, `process_single_image()`,
`_resilient_generate()`, the `tokenizer` property, and `get_model_info()` -- is
copy-pasted with only a model name string swapped.

**Lines duplicated**: ~450-600 lines across 5 files (2,250-3,000 total wasted lines).

---

## 1. Interface Signature -- The Minimal API for a New Model

A new model needs to provide exactly two things:

```python
from models.simple_processor import SimpleDocumentProcessor
from PIL import Image

class DocumentAwareGemma4Processor(SimpleDocumentProcessor):
    """Gemma 4 document extraction processor."""

    model_type_key: str = "gemma4"

    def _load_model(self) -> None:
        """Load model and processor from disk (only called if not pre-loaded)."""
        ...

    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run model-specific inference. Return raw text."""
        ...
```

That is the complete contract. Everything else is inherited.

### What `SimpleDocumentProcessor` provides (inherited for free):

| Method | LOC saved per model | What it does |
|--------|-------------------|--------------|
| `__init__()` | ~25 | Stores model/processor, calls `_load_model()` if needed, calls `_init_shared()`, calls `_configure_generation()` |
| `tokenizer` property | ~5 | Returns `self.processor.tokenizer` (or `None`) |
| `_configure_generation()` | ~15 | Loads gen config from `model_config.py` constant, computes `fallback_max_tokens` |
| `_calculate_max_tokens()` | ~10 | `base + (field_count * per_field)`, with bank statement floor |
| `process_single_image()` | ~65 | Full orchestration: load image, get prompt, generate, parse, clean, build result dict |
| `_resilient_generate()` | ~15 | OOM-safe wrapper (halve tokens and retry), cleanup outside except block |
| `get_model_info()` | ~6 | Returns `{model_type, model_path, batch_size}` dict |
| **Total per model** | **~141** | |

### Class attributes that configure behavior (with defaults):

```python
class SimpleDocumentProcessor(BaseDocumentProcessor):
    """Concrete base for standard HuggingFace VLM processors.

    Subclasses MUST override:
        model_type_key: str          -- e.g. "gemma4", used for gen config + batch size lookup
        _load_model() -> None        -- load self.model and self.processor from disk
        generate(image, prompt, max_tokens) -> str  -- model-specific inference

    Subclasses MAY override:
        generation_config_key: str   -- key into GENERATION_CONFIGS registry (default: model_type_key)
        has_oom_recovery: bool       -- set False to skip _resilient_generate (vLLM case)
        tokenizer_attr: str          -- attribute path on self.processor to get tokenizer
                                        (default: "tokenizer", override if processor IS the tokenizer)
    """

    # -- MUST override --
    model_type_key: str  # No default -- forces subclass to declare

    # -- MAY override --
    has_oom_recovery: bool = True          # False for vLLM (engine handles OOM)
    tokenizer_attr: str = "tokenizer"      # How to get tokenizer from self.processor
```

---

## 2. Usage Example: Adding a Gemma 4 HuggingFace Processor in ~30 Lines

```python
"""Gemma 4 document extraction processor.

Uses Gemma4ForConditionalGeneration + AutoProcessor.
"""

from typing import Any, override

import torch
from PIL import Image

from models.simple_processor import SimpleDocumentProcessor


class DocumentAwareGemma4Processor(SimpleDocumentProcessor):
    """Document extraction processor for Google Gemma 4."""

    model_type_key = "gemma4"

    @override
    def _load_model(self) -> None:
        """Load Gemma 4 model and processor from disk."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        """Run Gemma 4 inference on a single image + prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()
```

**Total: 55 lines including imports, docstrings, and whitespace. ~30 lines of actual logic.**

Compare to the current pattern: 313-335 lines per simple processor.

---

## 3. What `SimpleDocumentProcessor` Hides Internally

### 3.1 The `__init__()` Template

Every simple processor's `__init__` follows the exact same sequence. The base class
captures this as a final (non-overridable) implementation:

```python
class SimpleDocumentProcessor(BaseDocumentProcessor):

    def __init__(
        self,
        field_list: list[str],
        model_path: str,
        device: str = "cuda",
        debug: bool = False,
        batch_size: int | None = None,
        pre_loaded_model=None,
        pre_loaded_processor=None,
        prompt_config: dict[str, Any] | None = None,
        field_definitions: dict[str, list[str]] | None = None,
        # vLLM passes model_type_key at construction time
        model_type_key: str | None = None,
    ):
        self.model_path = model_path
        self.model = pre_loaded_model
        self.processor = pre_loaded_processor

        # Allow runtime override of model_type_key (for vLLM multi-model support)
        if model_type_key is not None:
            self._runtime_model_type_key = model_type_key

        # CUDA memory optimization (no-op on vLLM)
        if self.has_oom_recovery:
            from common.gpu_optimization import configure_cuda_memory_allocation
            configure_cuda_memory_allocation()

        # Load model if not pre-loaded
        if self.model is None:
            self._load_model()

        # Shared init from BaseDocumentProcessor
        self._init_shared(
            field_list=field_list,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=debug,
            device=device,
            batch_size=batch_size,
            model_type_key=self._effective_model_type_key,
        )

        self._configure_generation()

        if self.debug:
            print(f"{self.__class__.__name__} initialized for {self.field_count} fields")

    @property
    def _effective_model_type_key(self) -> str:
        """Return runtime override if set, else class-level model_type_key."""
        return getattr(self, "_runtime_model_type_key", self.model_type_key)
```

### 3.2 Generation Config Resolution

Instead of each processor importing its own constant and copying it, the base class
uses a registry lookup:

```python
# In common/model_config.py -- already exists, just needs a lookup function:
_GENERATION_CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    "internvl3": INTERNVL3_GENERATION_CONFIG,
    "llama": LLAMA_GENERATION_CONFIG,
    "llama4scout": LLAMA4SCOUT_GENERATION_CONFIG,
    "qwen3vl": QWEN3VL_GENERATION_CONFIG,
    "qwen35": QWEN3VL_GENERATION_CONFIG,      # Reuses Qwen3VL config
    "nemotron": QWEN3VL_GENERATION_CONFIG,     # Reuses Qwen3VL config
    "gemma4": GEMMA4_GENERATION_CONFIG,
    "granite4": GRANITE4_GENERATION_CONFIG,
}

def get_generation_config(model_type: str) -> dict[str, Any]:
    """Look up generation config by model type key.

    Falls back to a sensible default (QWEN3VL config) for unknown models,
    so new models work out of the box without touching this file.
    """
    return dict(_GENERATION_CONFIG_REGISTRY.get(model_type, QWEN3VL_GENERATION_CONFIG))
```

Then in `SimpleDocumentProcessor._configure_generation()`:

```python
def _configure_generation(self) -> None:
    """Load generation hyper-parameters from model_config registry."""
    from common.model_config import get_generation_config

    self.gen_config: dict[str, Any] = get_generation_config(self._effective_model_type_key)

    self.fallback_max_tokens = max(
        int(self.gen_config.get("max_new_tokens_base", 512)),
        self.field_count * int(self.gen_config.get("max_new_tokens_per_field", 64)),
    )

    if self.debug:
        print(
            f"Generation config: max_new_tokens={self.fallback_max_tokens}, "
            f"do_sample={self.gen_config.get('do_sample', False)}"
        )
```

### 3.3 `_calculate_max_tokens()` -- Identical Across All Simple Processors

```python
@override
def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
    """Calculate token budget based on field count and document type."""
    base = int(self.gen_config.get("max_new_tokens_base", 512))
    per_field = int(self.gen_config.get("max_new_tokens_per_field", 64))
    tokens = base + (field_count * per_field)

    if document_type == "bank_statement":
        tokens = max(tokens, 1500)
    return tokens
```

### 3.4 `process_single_image()` -- The Big One (~65 LOC)

This method is character-for-character identical across Llama4, Qwen3VL, Qwen3.5,
and Nemotron. The vLLM variant differs only in:
- No `handle_memory_fragmentation()` call at the start
- Calls `self.generate()` directly instead of `self._resilient_generate()`
- No `torch.cuda.empty_cache()` at the end

The base class handles this with the `has_oom_recovery` flag:

```python
@override
def process_single_image(
    self,
    image_path: str,
    custom_prompt: str | None = None,
    custom_max_tokens: int | None = None,
    field_list: list[str] | None = None,
) -> dict:
    active_fields = field_list or self.field_list
    active_count = len(active_fields)
    start_time = time.time()
    image_name = Path(image_path).name

    try:
        if self.has_oom_recovery:
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

        image = self.load_document_image(image_path)

        prompt = custom_prompt or self.get_extraction_prompt()
        max_tokens = custom_max_tokens or self._calculate_max_tokens(
            active_count, "universal"
        )

        if self.debug:
            sys.stdout.write(f"Processing {image_name} ({active_count} fields)\n")
            sys.stdout.write(f"Prompt: {len(prompt)} chars, max_tokens: {max_tokens}\n")
            sys.stdout.flush()

        if self.has_oom_recovery:
            raw_response = self._resilient_generate(image, prompt, max_tokens)
        else:
            raw_response = self.generate(image, prompt, max_tokens)

        processing_time = time.time() - start_time

        # ... debug output, parse, clean, build result dict (identical) ...
```

### 3.5 `_resilient_generate()` -- OOM Recovery

```python
def _resilient_generate(
    self, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """Generate with OOM recovery (halve tokens and retry).

    Cleanup happens OUTSIDE the except block -- see MEMORY.md for why.
    """
    oom = False
    try:
        return self.generate(image, prompt, max_tokens)
    except torch.cuda.OutOfMemoryError:
        oom = True

    # Outside except -- traceback released, tensors can be freed
    assert oom  # noqa: S101
    gc.collect()
    torch.cuda.empty_cache()
    if self.debug:
        print(f"OOM at {max_tokens} tokens, retrying at {max_tokens // 2}")
    return self.generate(image, prompt, max_tokens // 2)
```

### 3.6 `tokenizer` Property and `get_model_info()`

```python
@property
def tokenizer(self):
    """Return tokenizer for Protocol / UnifiedBankExtractor compatibility."""
    if self.processor is not None:
        return getattr(self.processor, self.tokenizer_attr, None)
    return None

def get_model_info(self) -> dict:
    """Return model metadata for reporting."""
    return {
        "model_type": self._effective_model_type_key,
        "model_path": self.model_path,
        "batch_size": self.batch_size,
    }
```

---

## 4. How vLLM Fits In

The vLLM processor is the one "simple" processor that differs slightly:
- No `_resilient_generate()` (engine handles OOM via PagedAttention)
- No `torch.cuda.empty_cache()` calls
- No `handle_memory_fragmentation()`
- The `tokenizer` comes from `self.llm_engine`, not `self.processor`
- `_configure_generation()` dispatches to different config constants based on
  `model_type_key` (since one vLLM processor serves multiple model architectures)

**Solution**: `has_oom_recovery = False` handles the first three differences.
The `_effective_model_type_key` property (with runtime override via constructor arg)
handles the config dispatch. A custom `tokenizer` property override handles the
tokenizer source.

```python
class DocumentAwareVllmProcessor(SimpleDocumentProcessor):
    """vLLM-backed document extraction processor."""

    model_type_key = "llama4scout"  # Default, overridden at construction
    has_oom_recovery = False

    def __init__(self, *, model_type_key: str = "llama4scout", **kwargs):
        # vLLM stores the engine as both self.model and self.llm_engine
        kwargs.setdefault("model_type_key", model_type_key)
        super().__init__(**kwargs)
        self.llm_engine = self.model

        if self.llm_engine is None:
            raise ValueError(
                "DocumentAwareVllmProcessor requires a pre-loaded vLLM engine."
            )

    @property
    def tokenizer(self):
        return getattr(self.llm_engine, "tokenizer", None)

    def _load_model(self) -> None:
        raise ValueError("vLLM processor requires pre-loaded engine.")

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
        # ... existing vLLM generate() logic (base64 encoding, SamplingParams, etc.)
        ...
```

---

## 5. How InternVL3 and Llama Still Work

**They are completely unaffected.** The class hierarchy becomes:

```
BaseDocumentProcessor (ABC)        # abstract: generate, _calculate_max_tokens, process_single_image
  |
  +-- SimpleDocumentProcessor      # NEW: concrete implementations of the 3 abstract methods
  |     |                          #      + _resilient_generate, _configure_generation, etc.
  |     +-- Llama4Processor
  |     +-- Qwen3VLProcessor
  |     +-- Qwen35Processor
  |     +-- NemotronProcessor
  |     +-- VllmProcessor
  |     +-- (future: Gemma4HFProcessor, etc.)
  |
  +-- InternVL3HybridProcessor    # UNCHANGED: its own generate(), process_single_image(), batch methods
  +-- LlamaProcessor              # UNCHANGED: its own generate(), _resilient_generate(), etc.
```

InternVL3 and Llama **do not** inherit from `SimpleDocumentProcessor`. They continue
to inherit directly from `BaseDocumentProcessor` and provide their own implementations
of all abstract methods. Their complexity is warranted:

- **InternVL3**: Custom `_resilient_generate()` with 3-attempt OOM recovery + minimal
  generation fallback, `_detect_recursion_pattern()`, `batch_detect_documents()` /
  `batch_extract_documents()` for batched inference, InternVL3-specific `model.chat()`
  API (not `model.generate()`), custom image preprocessing pipeline.

- **Llama**: Custom `_load_model()` with BitsAndBytes quantization, `model.tie_weights()`,
  Llama-specific `MllamaForConditionalGeneration` + `AutoProcessor` APIs, custom
  generation config handling (null temperature/top_p on `generation_config`).

Neither of these models follows the standard "apply_chat_template -> model.generate()
-> batch_decode" pattern. Forcing them into the template would require so many
overrides that the abstraction would provide no benefit.

---

## 6. Trade-offs

### Benefits

| Benefit | Impact |
|---------|--------|
| **~700 LOC eliminated** | 5 processors x 141 LOC each = 705 lines of pure duplication removed |
| **New model in ~30 lines** | Down from 313-335 lines (91% reduction) |
| **Single place for bug fixes** | OOM recovery, process_single_image, token calculation -- fix once |
| **Consistent behavior** | All simple processors guaranteed to have same debug output, error handling, cleanup |
| **No breaking changes** | Public API (Protocol) is unchanged. Registry creator functions unchanged. |

### Costs and Risks

| Cost | Mitigation |
|------|------------|
| **One more class in the hierarchy** | `SimpleDocumentProcessor` is a natural intermediate -- it is not an artificial abstraction |
| **Indirection for `generate()`** | Template Method pattern is well-understood; grep for `def generate` still finds all implementations |
| **vLLM needs special handling** | `has_oom_recovery` flag + `tokenizer` override are minimal; alternative is keeping vLLM as a separate file (acceptable) |
| **InternVL3/Llama excluded** | By design -- they are genuinely different. Forced unification would be worse than duplication |
| **Testing surface** | Need tests for `SimpleDocumentProcessor` itself, but this _replaces_ 5 sets of identical tests |

### What NOT to Do

- **Do NOT make `generate()` configurable via data** (e.g., "message format templates").
  The differences between models' `generate()` methods are in their APIs (kwargs,
  return shapes, preprocessing), not in string formatting. A data-driven approach
  would become a DSL that is harder to debug than Python code.

- **Do NOT use a factory function instead of inheritance.** The processors need to
  satisfy the `DocumentProcessor` Protocol and be passed around as objects. A factory
  that returns configured closures would lose type information and debuggability.

- **Do NOT merge the generation config into the processor class.** The configs are
  already in `common/model_config.py` and are shared with other consumers (cli.py,
  vLLM processor). A lookup function (`get_generation_config()`) is the right seam.

---

## 7. Migration Path

### Step 1: Create `models/simple_processor.py` (~180 LOC)
- `SimpleDocumentProcessor(BaseDocumentProcessor)` with all shared methods
- `get_generation_config()` function in `common/model_config.py`

### Step 2: Migrate one processor (Qwen3VL recommended -- simplest `generate()`)
- Rewrite `document_aware_qwen3vl_processor.py` to inherit from `SimpleDocumentProcessor`
- Verify identical behavior with existing tests / evaluation run

### Step 3: Migrate remaining simple processors one at a time
- Llama4 -> Nemotron -> Qwen3.5 -> vLLM (vLLM last, since it has the most overrides)

### Step 4: Delete dead code
- Remove duplicated methods from migrated processors
- Clean up any local generation config constants (e.g., `NEMOTRON_GENERATION_CONFIG = dict(...)`)

### Risk mitigation
- Each migration is a single-file change with no API changes
- Registry `processor_creator` functions remain identical
- Can be done incrementally, one model at a time
- Rollback = revert one file

---

## 8. File Impact Summary

| File | Change |
|------|--------|
| `models/simple_processor.py` | **NEW** (~180 LOC) |
| `common/model_config.py` | Add `get_generation_config()` function (~15 LOC) |
| `models/document_aware_qwen3vl_processor.py` | 320 LOC -> ~55 LOC |
| `models/document_aware_qwen35_processor.py` | 317 LOC -> ~55 LOC |
| `models/document_aware_llama4_processor.py` | 335 LOC -> ~55 LOC |
| `models/document_aware_nemotron_processor.py` | 313 LOC -> ~65 LOC |
| `models/document_aware_vllm_processor.py` | 277 LOC -> ~85 LOC |
| `models/document_aware_internvl3_processor.py` | **UNCHANGED** |
| `models/document_aware_llama_processor.py` | **UNCHANGED** |
| `models/base_processor.py` | **UNCHANGED** |
| `models/protocol.py` | **UNCHANGED** |
| `models/registry.py` | **UNCHANGED** |

**Net LOC change**: +195 (new file + model_config addition) - 1,047 (removed duplication) = **-852 LOC**
