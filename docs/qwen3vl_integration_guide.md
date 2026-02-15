# Integrating Qwen3-VL-8B-Instruct

A step-by-step guide for downloading Qwen3-VL-8B-Instruct from Hugging Face and integrating it as a third model in the Document Extraction Pipeline.

## Why Qwen3-VL-8B?

| Benchmark | InternVL3.5-8B | Llama 3.2-11B | Qwen3-VL-8B |
|-----------|---------------|---------------|-------------|
| DocVQA | — | — | 96.1% |
| OCRBench | — | — | 89.6% |
| CC-OCR (key info extraction) | — | — | 79.9% |
| MMMU | — | — | 69.6% |
| Our pipeline (9 images) | 94.5% | 95.0% | *TBD* |

Qwen3-VL-8B is a 9B-parameter vision-language model from Alibaba with a 256K context window, 32-language OCR, and state-of-the-art document understanding. Its API (`AutoProcessor` + `model.generate()`) follows the same pattern as Llama, making integration straightforward.

## 1. Download the Model

### Option A: Hugging Face CLI (recommended)

```bash
# Install the CLI if needed
pip install huggingface-cli

# Login (required for gated models — Qwen3-VL is open, but login avoids rate limits)
huggingface-cli login

# Download to your model storage directory
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /path/to/models/Qwen3-VL-8B-Instruct
```

### Option B: Git LFS

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct /path/to/models/Qwen3-VL-8B-Instruct
```

### Option C: Python download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir="/path/to/models/Qwen3-VL-8B-Instruct",
)
```

### Storage requirements

- **Disk**: ~18 GB (BF16 weights)
- **GPU VRAM**: ~18-20 GB at BF16, ~10 GB at FP8

### Dependency check

Qwen3-VL requires `transformers >= 4.57.0`. If your environment has an older version:

```bash
pip install --upgrade transformers>=4.57.0
```

Verify the model class is available:

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print("Qwen3-VL model class available")
```

## 2. Pipeline Integration Overview

Our architecture makes adding a new model a 3-file task. Here's the full picture:

```
Files to CREATE:
  models/document_aware_qwen3vl_processor.py  ← Inherit BaseDocumentProcessor
  prompts/qwen3vl_prompts.yaml                ← Extraction prompts

Files to MODIFY:
  models/registry.py                          ← Loader + creator + register_model()
  common/model_config.py                      ← Generation config + batch sizes
  config/run_config.yml                       ← Default paths + batch sizes

Files that need NO changes:
  cli.py, batch_processor.py, bank_statement_adapter.py,
  unified_bank_extractor.py, evaluation_metrics.py, all reporting code
```

## 3. Create the Processor

Create `models/document_aware_qwen3vl_processor.py`:

```python
"""Qwen3-VL-8B document extraction processor.

Inherits shared detection, classification, prompt resolution, and extraction
orchestration from BaseDocumentProcessor.  Only model-specific inference
(generate, token calculation, single-image processing) is implemented here.
"""

import gc
import time
from pathlib import Path
from typing import Any, override

import torch
from PIL import Image

from common.extraction_cleaner import ExtractionCleaner
from common.extraction_parser import parse_extraction_response
from common.gpu_optimization import configure_cuda_memory_allocation
from common.model_config import QWEN3VL_GENERATION_CONFIG
from models.base_processor import BaseDocumentProcessor


class DocumentAwareQwen3VLProcessor(BaseDocumentProcessor):
    """Document extraction processor for Qwen3-VL-8B-Instruct.

    Satisfies the DocumentProcessor Protocol.  Inherits from
    BaseDocumentProcessor for shared logic — only model-specific
    inference needs to be implemented.
    """

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
    ):
        self.model_path = model_path
        self.model = pre_loaded_model
        self.processor = pre_loaded_processor

        configure_cuda_memory_allocation()

        if self.model is None:
            self._load_model()

        # Shared init: validates config, loads field defs, sets batch size
        self._init_shared(
            field_list=field_list,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=debug,
            device=device,
            batch_size=batch_size,
            model_type_key="qwen3vl",
        )

        self._configure_generation()

    # -- Protocol compatibility ------------------------------------------------

    @property
    def tokenizer(self):
        """Return tokenizer for Protocol / BankStatementAdapter compatibility."""
        if self.processor is not None:
            return self.processor.tokenizer
        return None

    # -- Model loading ---------------------------------------------------------

    def _load_model(self) -> None:
        """Load Qwen3-VL model and processor from disk."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if self.debug:
            print(f"Loading Qwen3-VL from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        # Suppress spurious warnings from generation_config
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from model_config."""
        self.gen_config = dict(QWEN3VL_GENERATION_CONFIG)

    # -- Abstract method implementations ---------------------------------------

    @override
    def generate(
        self, image: Image.Image, prompt: str, max_tokens: int = 1024
    ) -> str:
        """Run Qwen3-VL inference on a single image + prompt.

        This is the core abstract method from BaseDocumentProcessor.
        """
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
            temperature=None,
            top_p=None,
        )

        # Trim input tokens from output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    @override
    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate token budget based on field count and document type."""
        base = self.gen_config.get("max_new_tokens_base", 512)
        per_field = self.gen_config.get("max_new_tokens_per_field", 64)
        tokens = base + (field_count * per_field)

        if document_type == "bank_statement":
            tokens = max(tokens, 1500)
        return tokens

    # -- Single image processing -----------------------------------------------

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict:
        """Process one document image end-to-end.

        Called by the inherited process_document_aware() method.
        """
        start_time = time.time()
        image_name = Path(image_path).name
        active_fields = field_list or self.field_list

        try:
            image = self.load_document_image(image_path)

            prompt = custom_prompt or self.get_extraction_prompt()
            max_tokens = custom_max_tokens or self._calculate_max_tokens(
                len(active_fields), "universal"
            )

            raw_response = self._resilient_generate(image, prompt, max_tokens)
            processing_time = time.time() - start_time

            # Parse structured fields from response
            extracted_data = parse_extraction_response(raw_response, active_fields)

            # Clean values
            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            return {
                "image_name": image_name,
                "extracted_data": extracted_data,
                "raw_response": raw_response,
                "processing_time": processing_time,
                "response_completeness": found / max(len(active_fields), 1),
                "content_coverage": found / max(len(active_fields), 1),
                "extracted_fields_count": found,
                "field_count": len(active_fields),
            }

        except Exception as e:
            processing_time = time.time() - start_time
            if self.debug:
                print(f"Error processing {image_name}: {e}")
            return {
                "image_name": image_name,
                "extracted_data": {f: "NOT_FOUND" for f in active_fields},
                "raw_response": f"ERROR: {e}",
                "processing_time": processing_time,
                "response_completeness": 0.0,
                "content_coverage": 0.0,
                "extracted_fields_count": 0,
                "field_count": len(active_fields),
            }

    def _resilient_generate(
        self, image: Image.Image, prompt: str, max_tokens: int
    ) -> str:
        """Generate with OOM recovery (halve tokens and retry)."""
        oom = False
        try:
            return self.generate(image, prompt, max_tokens)
        except torch.cuda.OutOfMemoryError:
            oom = True

        # Cleanup OUTSIDE except block (see MEMORY.md)
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            if self.debug:
                print(f"OOM at {max_tokens} tokens, retrying at {max_tokens // 2}")
            return self.generate(image, prompt, max_tokens // 2)

    def get_model_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "model_type": "qwen3vl",
            "model_path": self.model_path,
            "batch_size": self.batch_size,
        }
```

> **Key differences from Llama**:
> - Model class: `Qwen3VLForConditionalGeneration` (not `MllamaForConditionalGeneration`)
> - Chat template: `processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")`
> - Image in messages: `{"type": "image", "image": <PIL.Image>}` (Qwen accepts PIL directly)
> - Decode: `processor.batch_decode()` on trimmed IDs

## 4. Create Extraction Prompts

Create `prompts/qwen3vl_prompts.yaml`. Start by copying `prompts/llama_prompts.yaml` and adjusting instruction style if needed. Qwen3-VL responds well to structured instructions:

```yaml
prompts:
  invoice:
    name: "Invoice Extraction"
    description: "Extract invoice fields"
    prompt: |
      Extract ALL data from this invoice image.
      Respond in the exact format below. Use actual values from the image,
      or NOT_FOUND if a field is not present.

      DOCUMENT_TYPE: INVOICE
      BUSINESS_ABN: NOT_FOUND
      SUPPLIER_NAME: NOT_FOUND
      BUSINESS_ADDRESS: NOT_FOUND
      PAYER_NAME: NOT_FOUND
      PAYER_ADDRESS: NOT_FOUND
      INVOICE_NUMBER: NOT_FOUND
      INVOICE_DATE: NOT_FOUND
      DUE_DATE: NOT_FOUND
      LINE_ITEM_DESCRIPTIONS: NOT_FOUND
      LINE_ITEM_QUANTITIES: NOT_FOUND
      LINE_ITEM_UNIT_PRICES: NOT_FOUND
      LINE_ITEM_AMOUNTS: NOT_FOUND
      SUBTOTAL: NOT_FOUND
      GST_AMOUNT: NOT_FOUND
      TOTAL_AMOUNT: NOT_FOUND

  receipt:
    name: "Receipt Extraction"
    description: "Extract receipt fields"
    prompt: |
      Extract ALL data from this receipt image.
      # ... same pattern ...

  bank_statement_flat:
    name: "Flat Bank Statement Extraction"
    description: "Extract flat-format bank statement fields"
    prompt: |
      Extract ALL data from this bank statement image.
      # ... same pattern ...

  bank_statement_date_grouped:
    name: "Date-Grouped Bank Statement Extraction"
    description: "Extract date-grouped bank statement fields"
    prompt: |
      Extract ALL data from this bank statement image.
      # ... same pattern ...

  universal:
    name: "Universal Extraction"
    description: "Extract all possible fields"
    prompt: |
      Extract ALL data from this document image.
      # ... same pattern ...

settings:
  max_tokens: 2000
  temperature: 0.0
```

> **Tip**: Copy from `prompts/llama_prompts.yaml` initially and iterate after benchmarking.

## 5. Register in the Registry

Add to `models/registry.py`:

```python
# ── Qwen3-VL loader ─────────────────────────────────────────────────────

def _qwen3vl_loader(config):
    """Context manager for loading Qwen3-VL-8B-Instruct."""
    from contextlib import contextmanager

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            processor = AutoProcessor.from_pretrained(str(cfg.model_path))

            load_kwargs = {
                "dtype": cfg.torch_dtype,
                "device_map": cfg.device_map,
            }
            if cfg.flash_attn:
                load_kwargs["attn_implementation"] = "flash_attention_2"

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(cfg.model_path), **load_kwargs
            )

            # Suppress spurious generation_config warnings
            if hasattr(model, "generation_config"):
                model.generation_config.temperature = None
                model.generation_config.top_p = None

            _print_gpu_status(console)

            yield model, processor

        finally:
            del model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _qwen3vl_processor_creator(
    model, tokenizer_or_processor, config, prompt_config,
    universal_fields, field_definitions,
):
    """Create a DocumentAwareQwen3VLProcessor from loaded components."""
    from models.document_aware_qwen3vl_processor import (
        DocumentAwareQwen3VLProcessor,
    )

    return DocumentAwareQwen3VLProcessor(
        field_list=universal_fields,
        model_path=str(config.model_path),
        debug=config.verbose,
        pre_loaded_model=model,
        pre_loaded_processor=tokenizer_or_processor,
        prompt_config=prompt_config,
        field_definitions=field_definitions,
    )


register_model(
    ModelRegistration(
        model_type="qwen3vl",
        loader=_qwen3vl_loader,
        processor_creator=_qwen3vl_processor_creator,
        prompt_file="qwen3vl_prompts.yaml",
        description="Qwen3-VL-8B-Instruct vision-language model",
    )
)
```

## 6. Add Generation Config

Add to `common/model_config.py`:

```python
QWEN3VL_GENERATION_CONFIG = {
    "max_new_tokens_base": 512,
    "max_new_tokens_per_field": 64,
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 0.95,
    "use_cache": True,
}
```

Add `"qwen3vl"` entries to the batch size dicts:

```python
DEFAULT_BATCH_SIZES = {
    "internvl3": 4,
    "internvl3-2b": 4,
    "internvl3-8b": 4,
    "qwen3vl": 4,           # ← add
}

MAX_BATCH_SIZES = {
    "internvl3": 8,
    "internvl3-2b": 8,
    "internvl3-8b": 16,
    "qwen3vl": 8,            # ← add
}

CONSERVATIVE_BATCH_SIZES = {
    "internvl3": 1,
    "internvl3-2b": 2,
    "internvl3-8b": 1,
    "qwen3vl": 2,            # ← add
}
```

## 7. Update YAML Config

Add Qwen3-VL to `config/run_config.yml`:

```yaml
model:
  type: qwen3vl                    # ← select Qwen3-VL
  path: /path/to/Qwen3-VL-8B-Instruct

batch:
  default_sizes:
    internvl3: 4
    qwen3vl: 4                     # ← add
  max_sizes:
    internvl3: 8
    qwen3vl: 8                     # ← add
  conservative_sizes:
    internvl3: 1
    qwen3vl: 2                     # ← add

model_loading:
  default_paths:
    internvl3: /models/InternVL3_5-8B
    llama: /models/Llama-3.2-11B-Vision-Instruct
    qwen3vl: /models/Qwen3-VL-8B-Instruct   # ← add
```

## 8. Run It

```bash
# Basic run
python cli.py --model qwen3vl -d ./images -o ./output

# With evaluation
python cli.py --model qwen3vl -d ./images -o ./output -g ./ground_truth.csv

# Explicit model path
python cli.py --model qwen3vl -m /path/to/Qwen3-VL-8B-Instruct -d ./images -o ./output

# V100 (no flash attention, float32)
python cli.py --model qwen3vl --no-flash-attn --dtype float32 -d ./images -o ./output
```

The `--model qwen3vl` flag is auto-discovered from the registry. `--help` will show it as an available option.

## 9. Verification Checklist

| Step | Command | Expected |
|------|---------|----------|
| Import check | `python -c "from models.document_aware_qwen3vl_processor import DocumentAwareQwen3VLProcessor; print('OK')"` | `OK` |
| Registry check | `python -c "from models.registry import get_model; print(get_model('qwen3vl'))"` | `ModelRegistration(model_type='qwen3vl', ...)` |
| CLI discovery | `python cli.py --help` | `--model` shows `qwen3vl` as option |
| Smoke test | `python cli.py --model qwen3vl -d ./test_images -o ./tmp --max-images 1 -v` | Processes 1 image |
| Accuracy test | `python cli.py --model qwen3vl -d ./eval_images -o ./output -g ./ground_truth.csv` | Compare F1 score |

## GPU Memory Requirements

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| H200/H100 | 80 GB | `--dtype bfloat16 --flash-attn` |
| A100 | 40-80 GB | `--dtype bfloat16 --flash-attn` |
| L40S | 48 GB | `--dtype bfloat16 --flash-attn` |
| L4/A10G | 24 GB | `--dtype bfloat16 --flash-attn` (tight) |
| V100 | 16-32 GB | `--dtype float32 --no-flash-attn` |

## Architecture Notes

Because the processor inherits from `BaseDocumentProcessor`, you get these capabilities for free:

- **Document detection**: `detect_and_classify_document()` — YAML-driven classification
- **Document-aware extraction**: `process_document_aware()` — type-specific prompts and fields
- **Bank statement extraction**: `BankStatementAdapter` uses `processor.generate` as a callable — no model-specific code needed in `unified_bank_extractor.py`
- **Batch orchestration**: `BatchDocumentProcessor` calls Protocol methods — sequential fallback if `BatchCapableProcessor` is not satisfied

To add batch inference support later, implement `batch_detect_documents()` and `batch_extract_documents()` on the processor class. The pipeline will detect them automatically via the `BatchCapableProcessor` Protocol.

## Troubleshooting

### `ImportError: Qwen3VLForConditionalGeneration`

Your `transformers` version is too old. Qwen3-VL requires >= 4.57.0:

```bash
pip install --upgrade transformers>=4.57.0
```

### `CUDA out of memory`

Qwen3-VL-8B needs ~18 GB VRAM at BF16. Options:
1. Use FP8 quantized variant: `Qwen/Qwen3-VL-8B-Instruct-FP8` (~10 GB)
2. Reduce `max_new_tokens` in `run_config.yml`
3. Set `batch_size: 1` in config

### Flash Attention not available

Disable it — the model works without it (just slower):

```bash
python cli.py --model qwen3vl --no-flash-attn ...
```

Or in the loader, remove the `attn_implementation` kwarg.

### Chat template differences

If extraction output is garbled, check the chat template. Qwen3-VL uses a specific format. Debug with:

```python
messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(text)  # Inspect the formatted prompt
```
