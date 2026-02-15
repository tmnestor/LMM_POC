# GLM-OCR Integration Guide

## Model Overview

| Property | Value |
|---|---|
| **Model** | GLM-OCR |
| **Developer** | Z.ai (Zhipu AI) |
| **Parameters** | 0.9B (extremely lightweight) |
| **Architecture** | CogViT encoder + cross-modal connector + GLM-0.5B decoder |
| **License** | MIT (model), Apache 2.0 (code) |
| **Released** | January 2026 |
| **HF Repo** | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) |
| **Benchmark** | OmniDocBench V1.5: 94.62 (#1 overall) |

### Why GLM-OCR for Bank Statements?

GLM-OCR has a dedicated **"Table Recognition"** prompt mode specifically designed
for extracting tabular data from document images. At only 0.9B parameters, it uses
a fraction of the VRAM compared to our 8B models (~2GB vs ~16GB), potentially
allowing it to run alongside another model or process batches much faster.

Key advantages for our bank statement pipeline:
- **Built-in table recognition mode** via `"Table Recognition:"` prompt
- **Extremely fast** — 1.86 pages/second throughput
- **HF transformers native** — uses `AutoProcessor` + `GlmOcrForConditionalGeneration`
- **Same two-step inference pattern** as our Qwen3-VL processor

### Potential Concerns

- **0.9B is very small** — may lack the reasoning ability of 8B models for complex
  multi-turn extraction (bank statement column detection, debit/credit filtering)
- **OCR-specialized** — excels at text/table recognition but may not handle
  document classification (INVOICE vs RECEIPT vs BANK_STATEMENT) as well as
  general-purpose VLMs
- **New model** — less community testing than InternVL3 or Qwen3-VL


## Step 1: Download the Model

### Option A: Using `hf` CLI (Recommended)

```bash
# Install/update Hugging Face CLI
pip install -U huggingface_hub[cli]

# Login (use Read token, skip git credential)
hf auth login

# Create directory and download
mkdir -p /home/jovyan/nfs_share/models/GLM-OCR
hf download zai-org/GLM-OCR \
  --local-dir /home/jovyan/nfs_share/models/GLM-OCR
```

### Option B: Python Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zai-org/GLM-OCR",
    local_dir="/home/jovyan/nfs_share/models/GLM-OCR",
)
```

### Option C: Git Clone

```bash
git lfs install
git clone https://huggingface.co/zai-org/GLM-OCR \
  /home/jovyan/nfs_share/models/GLM-OCR
```


## Step 2: Verify Transformers Compatibility

GLM-OCR requires transformers with `GlmOcrForConditionalGeneration` support.
This was added in transformers v5.1.0 (the version already on our server).

```bash
python -c "from transformers import GlmOcrForConditionalGeneration; print('OK')"
```

If this fails, install from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```


## Step 3: Create the Processor

Create `models/document_aware_glmocr_processor.py`:

```python
"""GLM-OCR document extraction processor.

Inherits shared detection, classification, prompt resolution, and extraction
orchestration from BaseDocumentProcessor.  Only model-specific inference
(generate, token calculation, single-image processing) is implemented here.

Uses GlmOcrForConditionalGeneration + AutoProcessor with
processor.apply_chat_template() + model.generate() API.

NOTE: GLM-OCR is a 0.9B OCR-specialized model. It excels at text/table
recognition but may need stronger prompts for document classification
and structured field extraction compared to 8B general-purpose VLMs.
"""

import gc
import time
from pathlib import Path
from typing import Any, override

import torch
from PIL import Image

from common.extraction_parser import parse_extraction_response
from common.gpu_optimization import (
    configure_cuda_memory_allocation,
    handle_memory_fragmentation,
)
from common.model_config import GLMOCR_GENERATION_CONFIG
from models.base_processor import BaseDocumentProcessor


class DocumentAwareGlmOcrProcessor(BaseDocumentProcessor):
    """Document extraction processor for GLM-OCR.

    Satisfies the DocumentProcessor Protocol.  Inherits from
    BaseDocumentProcessor for shared logic (detection, classification,
    prompt resolution, extraction orchestration).
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
            model_type_key="glmocr",
        )

        self._configure_generation()

        if self.debug:
            print(f"GLM-OCR processor initialized for {self.field_count} fields")

    # -- Protocol compatibility ------------------------------------------------

    @property
    def tokenizer(self):
        """Return tokenizer for Protocol / BankStatementAdapter compatibility."""
        if self.processor is not None:
            return self.processor.tokenizer
        return None

    # -- Model loading ---------------------------------------------------------

    def _load_model(self) -> None:
        """Load GLM-OCR model and processor from disk."""
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration

        if self.debug:
            print(f"Loading GLM-OCR from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = GlmOcrForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if self.debug:
            print(f"Device: {self.model.device}")
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {param_count:,}")

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from model_config."""
        self.gen_config = dict(GLMOCR_GENERATION_CONFIG)

        self.fallback_max_tokens = max(
            self.gen_config["max_new_tokens_base"],
            self.field_count * self.gen_config["max_new_tokens_per_field"],
        )

        if self.debug:
            print(
                f"Generation config: max_new_tokens={self.fallback_max_tokens}, "
                f"do_sample={self.gen_config['do_sample']}"
            )

    # -- Abstract method implementations ---------------------------------------

    @override
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> str:
        """Run GLM-OCR inference on a single image + prompt.

        Uses the two-step approach: apply_chat_template for text formatting,
        then processor() for combined text+image tokenization.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Step 1: Build text template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Step 2: Tokenize text + encode image together
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        # GLM-OCR may include token_type_ids which aren't needed for generation
        inputs.pop("token_type_ids", None)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
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
        base = self.gen_config.get("max_new_tokens_base", 2000)
        per_field = self.gen_config.get("max_new_tokens_per_field", 64)
        tokens = base + (field_count * per_field)

        # Bank statements need more tokens for many transactions
        if document_type == "bank_statement":
            tokens = max(tokens, 4000)
        return tokens

    # -- Single image processing -----------------------------------------------

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: str | None = None,
        custom_max_tokens: int | None = None,
        field_list: list[str] | None = None,
    ) -> dict:
        """Process one document image end-to-end."""
        active_fields = field_list or self.field_list
        active_count = len(active_fields)
        start_time = time.time()
        image_name = Path(image_path).name

        try:
            handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

            image = self.load_document_image(image_path)

            prompt = custom_prompt or self.get_extraction_prompt()
            max_tokens = custom_max_tokens or self._calculate_max_tokens(
                active_count, "universal"
            )

            if self.debug:
                import sys

                sys.stdout.write(f"Processing {image_name} ({active_count} fields)\n")
                sys.stdout.write(
                    f"Prompt: {len(prompt)} chars, max_tokens: {max_tokens}\n"
                )
                sys.stdout.flush()

            raw_response = self._resilient_generate(image, prompt, max_tokens)
            processing_time = time.time() - start_time

            if self.debug:
                import sys

                sys.stdout.write(f"Response ({len(raw_response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(raw_response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse structured fields from response
            extracted_data = parse_extraction_response(
                raw_response, expected_fields=active_fields
            )

            # Clean values
            for field_name, value in extracted_data.items():
                extracted_data[field_name] = self.cleaner.clean_field_value(
                    field_name, value
                )

            found = sum(1 for v in extracted_data.values() if v != "NOT_FOUND")

            if self.debug:
                print(f"Extracted {found}/{active_count} fields")

            # Cleanup
            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "image_name": image_name,
                "extracted_data": extracted_data,
                "raw_response": raw_response,
                "processing_time": processing_time,
                "response_completeness": found / max(active_count, 1),
                "content_coverage": found / max(active_count, 1),
                "extracted_fields_count": found,
                "field_count": active_count,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            if self.debug:
                import traceback

                print(f"Error processing {image_name}: {e}")
                traceback.print_exc()
            return {
                "image_name": image_name,
                "extracted_data": {f: "NOT_FOUND" for f in active_fields},
                "raw_response": f"Error: {e}",
                "processing_time": processing_time,
                "response_completeness": 0.0,
                "content_coverage": 0.0,
                "extracted_fields_count": 0,
                "field_count": active_count,
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
            "model_type": "glmocr",
            "model_path": self.model_path,
            "batch_size": self.batch_size,
        }
```


## Step 4: Add Generation Config

Add to `common/model_config.py`:

```python
# GLM-OCR generation configuration
GLMOCR_GENERATION_CONFIG = {
    "max_new_tokens_base": 2000,
    "max_new_tokens_per_field": 64,
    "do_sample": False,
    "use_cache": True,
}
```

And add to the batch size dicts:

```python
DEFAULT_BATCH_SIZES = {
    ...
    "glmocr": 8,    # Very small model, can handle large batches
}
MAX_BATCH_SIZES = {
    ...
    "glmocr": 16,
}
CONSERVATIVE_BATCH_SIZES = {
    ...
    "glmocr": 4,
}
```


## Step 5: Register in Registry

Add to `models/registry.py`:

```python
# ============================================================================
# GLM-OCR Registration (lazy imports)
# ============================================================================

def _glmocr_loader(config):
    """Context manager for loading GLM-OCR model and processor."""
    from contextlib import contextmanager

    import torch
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    console = Console()

    @contextmanager
    def _loader(cfg):
        model = None
        processor = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            console.print(f"\n[bold]Loading GLM-OCR from: {cfg.model_path}[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading processor...", total=None)

                processor = AutoProcessor.from_pretrained(str(cfg.model_path))

                progress.update(task, description="Loading model weights...")

                model = GlmOcrForConditionalGeneration.from_pretrained(
                    str(cfg.model_path),
                    torch_dtype=cfg.torch_dtype,
                    device_map=cfg.device_map,
                )

                progress.update(task, description="Model loaded!")

            console.print("⚡ Flash Attention 2: ❌ not applicable (0.9B model)")

            _print_gpu_status(console)

            yield model, processor

        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _loader(config)


def _glmocr_processor_creator(
    model,
    tokenizer_or_processor,
    config,
    prompt_config,
    universal_fields,
    field_definitions,
):
    """Create a DocumentAwareGlmOcrProcessor from loaded components."""
    from models.document_aware_glmocr_processor import (
        DocumentAwareGlmOcrProcessor,
    )

    return DocumentAwareGlmOcrProcessor(
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
        model_type="glmocr",
        loader=_glmocr_loader,
        processor_creator=_glmocr_processor_creator,
        prompt_file="glmocr_prompts.yaml",
        description="GLM-OCR 0.9B document/table recognition model",
    )
)
```


## Step 6: Create Extraction Prompts

Create `prompts/glmocr_prompts.yaml`. Since GLM-OCR supports a dedicated
`"Table Recognition:"` mode, the bank statement prompts can leverage this
directly. For invoice/receipt extraction, use the same structured field
format as our other models.

> **NOTE**: GLM-OCR's built-in `"Table Recognition:"` prompt returns markdown
> tables. Our UBE pipeline may need adaptation to parse markdown table output
> instead of the current field-per-line format. Initial testing should determine
> whether to use the built-in table mode or our standard extraction prompts.


## Step 7: Add Model Path to Config

Update `config/run_config.yml`:

```yaml
model_loading:
  default_paths:
    internvl3: /home/jovyan/nfs_share/models/InternVL3_5-8B
    llama: /home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct
    qwen3vl: /home/jovyan/nfs_share/models/Qwen3-VL-8B-Instruct
    glmocr: /home/jovyan/nfs_share/models/GLM-OCR
```


## Step 8: Run

```bash
python cli.py --model glmocr \
  -d ../evaluation_data/images \
  -o ../evaluation_data/output \
  -g ../evaluation_data/ground_truth.csv
```


## Key Differences from Other Models

| Feature | InternVL3 (8B) | Qwen3-VL (8B) | GLM-OCR (0.9B) |
|---|---|---|---|
| Parameters | 8.5B | 8B | 0.9B |
| VRAM Usage | ~16GB | ~16GB | ~2GB |
| Model Class | `AutoModel` | `Qwen3VLForConditionalGeneration` | `GlmOcrForConditionalGeneration` |
| Processor | `AutoTokenizer` | `AutoProcessor` | `AutoProcessor` |
| Inference | `.chat()` | `apply_chat_template` + `generate` | `apply_chat_template` + `generate` |
| Table Mode | N/A | N/A | `"Table Recognition:"` built-in |
| Max Tokens | 2000 default | 8192 | 8192 |
| Throughput | ~1 page/30s | ~1 page/20s | 1.86 pages/s |


## Verification Checklist

- [ ] Model downloaded to `/home/jovyan/nfs_share/models/GLM-OCR`
- [ ] `python -c "from transformers import GlmOcrForConditionalGeneration; print('OK')"`
- [ ] `models/document_aware_glmocr_processor.py` created
- [ ] `prompts/glmocr_prompts.yaml` created
- [ ] `models/registry.py` updated with GLM-OCR registration
- [ ] `common/model_config.py` updated with `GLMOCR_GENERATION_CONFIG`
- [ ] `config/run_config.yml` updated with `glmocr` path
- [ ] `python -c "from models.registry import list_models; print(list_models())"` shows `glmocr`
- [ ] `python cli.py --model glmocr --help` loads without error
- [ ] Test run on evaluation dataset completes


## Experimental: Table Recognition Mode

GLM-OCR's built-in table recognition can be tested directly:

```python
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from PIL import Image

model = GlmOcrForConditionalGeneration.from_pretrained(
    "zai-org/GLM-OCR", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR")

image = Image.open("cba_debit_credit.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Table Recognition:"},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[text], images=[image], return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)

output_ids = model.generate(**inputs, max_new_tokens=8192)
generated = output_ids[:, inputs["input_ids"].shape[1]:]
result = processor.batch_decode(
    generated, skip_special_tokens=True
)[0]
print(result)
```

This returns a **markdown table** which could potentially replace our entire
multi-turn UBE header detection + extraction pipeline for bank statements,
since the table structure is recognized in a single pass.

If the markdown table output is accurate, we could add a GLM-OCR-specific
bank statement strategy that:
1. Uses `"Table Recognition:"` to get the full table as markdown
2. Parses the markdown table directly into transaction rows
3. Filters debit/withdrawal rows programmatically

This would eliminate the need for multi-turn prompting entirely.
