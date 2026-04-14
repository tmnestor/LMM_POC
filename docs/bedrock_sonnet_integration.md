# Integrating Claude Sonnet 3.5 (AWS Bedrock) into the Document Extraction Pipeline

## Summary

This document outlines what would be required to add Claude Sonnet 3.5, hosted on AWS Bedrock, as a model backend in our document extraction pipeline. The current architecture is built around **local GPU inference** (HuggingFace models loaded into VRAM, or vLLM offline engines). An API-hosted model like Sonnet is fundamentally different -- it requires a new backend type, a new registration mechanism, and careful consideration of latency, cost, and concurrency.

The good news: the pipeline's composition-based architecture (backend Protocol + orchestrator) means the changes are **additive**. No existing code needs to change for local models to keep working.

---

## Current Architecture (context)

```
cli.py --model <type>
    -> registry.py (ModelSpec / VllmSpec lookup)
        -> model_loader.py (load weights onto GPU)
            -> backends/<type>.py (ModelBackend Protocol)
                -> orchestrator.py (detection, extraction, parsing, cleaning)
                    -> document_pipeline.py (batch routing, evaluation)
```

Every model backend implements the `ModelBackend` Protocol:

```python
class ModelBackend(Protocol):
    model: Any          # The loaded model object (on GPU)
    processor: Any      # Tokenizer or processor

    def generate(
        self, image: Image.Image, prompt: str, params: GenerationParams,
    ) -> str:
        """Single-image inference. Returns raw text."""
        ...
```

The orchestrator calls `backend.generate()` and handles everything else: prompt selection, response parsing, field cleaning, business validation, OOM recovery.

---

## What Needs to Change

### 1. New backend: `models/backends/bedrock.py`

A new class implementing `ModelBackend` that makes API calls instead of local inference.

```python
# Sketch -- not production code
import base64, io, boto3
from PIL import Image
from models.backend import GenerationParams, ModelBackend

class BedrockSonnetBackend:
    def __init__(self, *, region: str, model_id: str):
        self.model = boto3.client("bedrock-runtime", region_name=region)
        self.processor = None  # No local tokenizer

    def generate(self, image: Image.Image, prompt: str, params: GenerationParams) -> str:
        # Encode image as base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Call Bedrock Messages API
        response = self.model.converse(
            modelId="anthropic.claude-sonnet-3-5-v2",
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}},
                    {"text": prompt},
                ],
            }],
            inferenceConfig={"maxTokens": params.max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]
```

Estimated: ~80-100 lines.

### 2. New registration type: `BedrockSpec`

The existing `ModelSpec` assumes HuggingFace (`from_pretrained`, `torch_dtype`, `device_map`). A Bedrock model has none of these. Options:

**Option A (recommended)**: New `BedrockSpec` dataclass + `register_bedrock_model()` factory in `model_loader.py`:

```python
@dataclass(frozen=True)
class BedrockSpec:
    model_type: str                    # e.g. "sonnet"
    prompt_file: str                   # e.g. "sonnet_prompts.yaml"
    description: str = ""
    region: str = "us-east-1"
    model_id: str = "anthropic.claude-sonnet-3-5-v2"
```

**Option B**: Shoehorn into `ModelSpec` with a custom `backend_factory` and no-op loader. Technically possible but semantically misleading -- `ModelSpec` carries GPU-specific fields (`torch_dtype`, `attn_implementation`, `device_map`) that don't apply.

Estimated: ~60 lines for the spec + factory.

### 3. Registration in `registry.py`

```python
register_bedrock_model(
    BedrockSpec(
        model_type="sonnet",
        prompt_file="sonnet_prompts.yaml",
        description="Claude Sonnet 3.5 via AWS Bedrock",
        region="us-east-1",
    )
)
```

### 4. Prompt file: `prompts/sonnet_prompts.yaml`

Claude's prompting conventions differ significantly from open-source VLMs. See the dedicated section [Converting Prompts for Claude](#converting-prompts-for-claude) below for a full worked example using `prompts/universal.yaml` as the exemplar.

### 5. Minor orchestrator changes

`DocumentOrchestrator` currently includes OOM recovery logic (catch `torch.cuda.OutOfMemoryError`, halve batch, retry). This should be skipped for API backends since there's no GPU involved. A simple `self._has_oom_recovery` flag (already exists in the constructor) handles this.

---

## Concerns and Pain Points

### Latency

| Aspect | Local GPU (InternVL3-8B) | API (Sonnet via Bedrock) |
|--------|--------------------------|--------------------------|
| Single inference | ~2-5s (on-device) | ~3-8s (network round-trip) |
| Bank statement (2 turns) | ~60-90s | ~90-120s+ (compounds per turn) |
| 12 images (standard) | ~30-60s | ~36-96s (sequential) |
| 12 images (bank) | ~12-18 min | ~18-24 min (sequential) |

The pipeline currently processes images **sequentially per GPU**. For local models this is fine (GPU is the bottleneck). For an API model, you'd want **concurrent requests** -- the network, not compute, is the bottleneck. The current `MultiGPUOrchestrator` distributes across GPUs and doesn't apply here; a new concurrency strategy (e.g., `asyncio` or `ThreadPoolExecutor` for API calls) would be needed to get reasonable throughput.

### Cost

Local GPU inference has a fixed compute cost regardless of volume. Bedrock charges per input/output token.

Rough per-image estimates (Sonnet 3.5 pricing as of early 2026):

| Document type | Input tokens (approx) | Output tokens (approx) | Est. cost per image |
|---------------|-----------------------|------------------------|---------------------|
| Standard (invoice/receipt) | ~2,000-4,000 (image + prompt) | ~200-500 | ~$0.01-0.02 |
| Bank statement (2 turns) | ~6,000-10,000 | ~500-1,500 | ~$0.03-0.06 |

For a 100-image evaluation run with mixed document types, expect ~$2-5. For production volumes this needs budgeting.

### Generation Parameters Mismatch

The current `GenerationParams` dataclass carries HuggingFace-style parameters:

```python
@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 1024
    do_sample: bool = False        # Not applicable to Claude
    temperature: float | None      # Supported but different semantics
    top_p: float | None            # Supported
    extra: dict[str, Any]          # Escape hatch
```

`do_sample`, `repetition_penalty`, and `num_beams` don't exist in the Anthropic API. The backend would need to translate/ignore these. The `extra` dict provides an escape hatch, but it's not clean.

**Suggestion**: The backend should map `GenerationParams` to Bedrock's `inferenceConfig` and silently ignore inapplicable fields. This keeps the orchestrator model-agnostic.

### Batch Inference

The pipeline's `BatchInference` Protocol enables the orchestrator to batch detection and extraction calls for higher GPU utilization:

```python
class BatchInference(Protocol):
    def generate_batch(self, images, prompts, params) -> list[str]: ...
```

This doesn't map to API-based models. Bedrock doesn't have a batch endpoint for real-time inference. Instead:

- **Real-time**: Concurrent individual API calls (need new concurrency layer)
- **Batch**: Bedrock has an async batch inference API for large volumes, but it's not real-time

The `DocumentPipeline` would need to handle this gracefully -- probably by falling back to sequential processing (which it already does when `BatchInference` is not supported) but ideally with concurrent API calls underneath.

### Authentication and Configuration

Local models just need a file path. Bedrock needs:

- AWS credentials (IAM role, access keys, or instance profile)
- Region selection
- Model ID (and potentially model access permissions in the Bedrock console)
- Endpoint configuration (VPC endpoint for private access)

This config doesn't fit neatly into `run_config.yml` which is currently GPU-focused. Suggest a new `api` section:

```yaml
api:
  bedrock:
    region: us-east-1
    model_id: anthropic.claude-sonnet-3-5-v2
    # Credentials via standard AWS chain (env vars, instance profile, etc.)
```

### Error Handling

Local inference fails with `torch.cuda.OutOfMemoryError` (handled). API inference fails with:

- `ThrottlingException` (rate limits) -- needs retry with backoff
- `ValidationException` (bad request) -- needs logging
- `ServiceUnavailableException` -- needs retry
- Network timeouts -- needs configurable timeout
- `AccessDeniedException` -- needs clear error message about Bedrock model access

The orchestrator's current error handling is GPU-focused. API error handling would need to be added to the backend itself.

### Image Size Limits

Bedrock has a **maximum payload size** (~20 MB per request) and Claude has image token limits. High-resolution document scans may need to be resized before sending. The current pipeline sends PIL Images directly to the backend -- the Bedrock backend would need to handle resizing/compression.

### Response Format Reliability

Claude Sonnet 3.5 is generally more reliable at following structured output formats than open-source VLMs. This is actually a **positive** -- the `ResponseHandler` (parse -> clean -> validate) pipeline may need less aggressive cleaning for Claude outputs. However, this also means the current prompts (which are heavily engineered to coerce specific formats from less capable models) might actually be suboptimal for Claude.

---

## Converting Prompts for Claude

This is the most impactful part of the integration. Our existing prompts are heavily engineered to compensate for the weaknesses of open-source VLMs (format drift, hallucination, ignoring instructions). Claude Sonnet 3.5 doesn't have these problems to the same degree, so the prompts should be **simplified and restructured** rather than used as-is.

### Why the existing prompts won't work well

Our current prompts use patterns that are **counterproductive** with Claude:

| Pattern in current prompts | Why it exists | Problem with Claude |
|---------------------------|---------------|---------------------|
| Repetitive `NOT_FOUND` drilling (every field definition says "return NOT_FOUND if not visible") | Open VLMs ignore instructions they only see once | Claude finds this patronising and the repetition wastes input tokens (~$) |
| `CRITICAL`, `IMPORTANT`, `STEP 1/2/3` shouting | Open VLMs need emphasis to follow multi-step logic | Claude follows instructions on first read; shouting adds noise |
| `CONVERSATION PROTOCOL: Do NOT include conversational text` | Open VLMs prepend "Sure, I'll help..." | Claude doesn't do this when given structured output format |
| Inline field definitions mixed with output format | Open VLMs need everything in one pass | Claude handles separate sections cleanly; mixing hurts readability |
| Negative instructions ("Do NOT guess", "SKIP deposits") repeated 3-4 times | Open VLMs lose track of negatives | Claude follows negative instructions reliably; repetition wastes tokens |

### Conversion principles

1. **Say it once.** Claude follows instructions the first time. Remove all repetition.
2. **Use XML tags for structure.** Claude is trained to respect `<tags>` as semantic boundaries. Replace our flat text sections with tagged blocks.
3. **Separate schema from rules.** Define the output format once, then state business rules separately. Don't interleave them.
4. **Prefer positive instructions over negative.** "Extract only debits" instead of "Do NOT extract credits. SKIP deposits. IGNORE credits."
5. **Use a system prompt.** The Bedrock Converse API supports a `system` parameter -- put the role definition and global rules there, not in every user message.
6. **Trust the model.** Remove the "Do NOT guess" / "Be conservative" guardrails. Instead, state the accuracy contract once: "Return NOT_FOUND for any field not explicitly visible in the document."
7. **Be concise.** Claude charges per input token. Our `universal.yaml` prompt is ~700 words. The Claude version should be ~250-350 words with equal or better accuracy.

### Worked example: `prompts/universal.yaml`

Below is the current prompt alongside the converted Claude version.

#### Current prompt (open VLM style -- `prompts/universal.yaml`)

```yaml
prompts:
  universal:
    name: Universal Single-Pass Extraction
    description: Extract all 17 fields in one pass for any document type
    prompt: |

      You are an expert document analyzer specializing in business document extraction.
      Extract structured data from this document image using precise field-by-field analysis.

      CRITICAL EXTRACTION RULES:
      - If ANY field is not clearly visible in the document, return "NOT_FOUND" for that field
      - Do NOT guess, estimate, or infer missing information
      - Only extract what is EXPLICITLY shown in the image
      - Use exact text from document (preserve original formatting/capitalization)
      - Be conservative: when in doubt, use "NOT_FOUND"

      CONVERSATION PROTOCOL:
      - Start your response immediately with "DOCUMENT_TYPE:"
      - Do NOT include conversational text like "I'll extract..." or "Based on the document..."
      - Do NOT repeat the user's request or add explanations
      - Output ONLY the structured extraction data below
      - End immediately after "STATEMENT_DATE_RANGE:" with no additional text

      Extract the following 17 fields from this document in this exact order:

      IMPORTANT: Some fields only apply to specific document types:
      • INVOICE/RECEIPT fields: INVOICE_DATE, BUSINESS_ABN, SUPPLIER_NAME, ...
      • BANK STATEMENT fields (TAXPAYER EXPENSE CLAIMS): TRANSACTION_DATES, ...
      • CRITICAL FOR BANK STATEMENTS: Extract ONLY DEBIT/WITHDRAWAL transactions ...
      • If field doesn't apply to document type, return "NOT_FOUND"
      • BANK STATEMENT DEBIT-ONLY EXTRACTION: For TRANSACTION_DATES, ...

      FIELD DEFINITIONS:

      DOCUMENT_TYPE: Type of document (INVOICE/RECEIPT/STATEMENT) - return "NOT_FOUND" if unclear
      BUSINESS_ABN: 11-digit Australian Business Number (format: XX XXX XXX XXX) - return "NOT_FOUND" ...
      SUPPLIER_NAME: Company/business name ... - return "NOT_FOUND" if not visible
      [... 14 more field definitions, each repeating "return NOT_FOUND if not visible" ...]

      OUTPUT EXACTLY IN THIS FORMAT (replace [values] with actual extracted data):

      DOCUMENT_TYPE: [INVOICE/RECEIPT/STATEMENT or NOT_FOUND]
      BUSINESS_ABN: [11-digit ABN or NOT_FOUND]
      [... all 17 fields ...]
```

**Problems**: ~700 words. "NOT_FOUND" appears 20+ times. "CRITICAL" appears 3 times. Field definitions and output format are redundantly stated. Bank statement rules are repeated in 4 different places.

#### Converted Claude prompt (`prompts/sonnet_prompts.yaml`)

```yaml
prompts:
  universal:
    name: Universal Single-Pass Extraction
    description: Extract all 17 fields in one pass for any document type
    system: |
      You extract structured data from Australian business documents.
      Return NOT_FOUND for any field not explicitly visible. Never guess.
      Output only the requested fields, nothing else.
    prompt: |
      <task>
      Extract fields from this document image. Output each field on its own line
      in FIELD_NAME: value format. Start with DOCUMENT_TYPE.
      </task>

      <fields>
      DOCUMENT_TYPE: INVOICE, RECEIPT, or BANK_STATEMENT
      BUSINESS_ABN: 11-digit ABN (XX XXX XXX XXX)
      SUPPLIER_NAME: Business providing goods/services
      BUSINESS_ADDRESS: Supplier address
      PAYER_NAME: Customer/payer name
      PAYER_ADDRESS: Customer/payer address
      INVOICE_DATE: Document date (DD/MM/YYYY)
      LINE_ITEM_DESCRIPTIONS: Item names separated by " | "
      LINE_ITEM_QUANTITIES: Quantities separated by " | "
      LINE_ITEM_PRICES: Unit prices with $ separated by " | "
      LINE_ITEM_TOTAL_PRICES: Line totals with $ separated by " | "
      GST_AMOUNT: GST amount with $
      IS_GST_INCLUDED: True if GST > $0.00, else False
      TOTAL_AMOUNT: Final total with $
      TRANSACTION_DATES: Debit dates only (DD/MM/YYYY) separated by " | "
      TRANSACTION_AMOUNTS_PAID: Debit amounts only with $ separated by " | "
      STATEMENT_DATE_RANGE: DD/MM/YYYY to DD/MM/YYYY
      </fields>

      <rules>
      - Fields that don't apply to this document type: NOT_FOUND
      - Bank statements: extract only debit/withdrawal transactions (money out).
        Skip credits, deposits, salary, refunds.
      - Dates, descriptions, and amounts must have matching element counts.
      - Preserve original text. Use exact values from the document.
      </rules>
```

**Result**: ~200 words. Same information, no repetition. XML tags provide clear semantic boundaries. System prompt handles the role and global rules. Business rules stated once.

### Key structural differences

| Aspect | Open VLM prompt | Claude prompt |
|--------|----------------|---------------|
| **Length** | ~700 words | ~200 words |
| **NOT_FOUND mentions** | 20+ | 2 (once in system, once in rules) |
| **Structure** | Flat text with CAPS headings | XML tags: `<task>`, `<fields>`, `<rules>` |
| **Role definition** | In-prompt ("You are an expert...") | System prompt (separate API parameter) |
| **Conversation suppression** | 5-line CONVERSATION PROTOCOL block | Unnecessary -- Claude follows structured format |
| **Field definitions** | Mixed with output format | Clean list in `<fields>` block |
| **Business rules** | Repeated in 4 places | Single `<rules>` block |
| **Negative instructions** | "Do NOT guess, Do NOT include, Do NOT repeat" | "Never guess." (once in system prompt) |

### Converting other prompt files

Apply the same pattern to each prompt in `internvl3_prompts.yaml`:

#### Invoice/Receipt prompts

These are simple -- the current prompts are already short. The Claude version just needs:
- Move "Extract ALL data" preamble to system prompt
- Wrap field list in `<fields>` tags
- Remove per-field "NOT_FOUND" repetition

```yaml
  invoice:
    name: Invoice Extraction
    system: |
      You extract structured data from Australian business documents.
      Return NOT_FOUND for any field not explicitly visible. Never guess.
      Output only the requested fields, nothing else.
    prompt: |
      <task>Extract all fields from this invoice image.</task>

      <fields>
      DOCUMENT_TYPE: INVOICE
      BUSINESS_ABN: 11-digit ABN (XX XXX XXX XXX)
      SUPPLIER_NAME: Business name at top
      BUSINESS_ADDRESS: Supplier address
      PAYER_NAME: Customer from "Bill To" section
      PAYER_ADDRESS: Customer address
      INVOICE_DATE: Date in DD/MM/YYYY
      LINE_ITEM_DESCRIPTIONS: Item names with " | " separator
      LINE_ITEM_QUANTITIES: Quantities with " | "
      LINE_ITEM_PRICES: Unit prices with $ and " | "
      LINE_ITEM_TOTAL_PRICES: Line totals with $ and " | "
      IS_GST_INCLUDED: True/False
      GST_AMOUNT: GST with $
      TOTAL_AMOUNT: Total with $
      </fields>
```

#### Bank statement prompts

These benefit most from simplification. The current `bank_statement_date_grouped` prompt is ~40 lines with debit-vs-credit rules repeated 3 times. The Claude version:

```yaml
  bank_statement_date_grouped:
    name: Date-Grouped Bank Statement Extraction
    system: |
      You extract structured data from Australian bank statements for expense claims.
      Extract only debit/withdrawal transactions (money out). Skip all credits.
      Return NOT_FOUND for any field not visible. Output only the requested fields.
    prompt: |
      <task>
      Extract debit transactions from this date-grouped bank statement.
      Transactions are grouped under date headers.
      </task>

      <fields>
      DOCUMENT_TYPE: BANK_STATEMENT
      STATEMENT_DATE_RANGE: Overall period (DD/MM/YYYY to DD/MM/YYYY)
      TRANSACTION_DATES: Debit dates in DD/MM/YYYY with " | " separator
      LINE_ITEM_DESCRIPTIONS: Debit descriptions with " | " separator
      TRANSACTION_AMOUNTS_PAID: Debit amounts with $ and " | " separator
      </fields>

      <rules>
      - Debit indicators: withdrawal, payment, purchase, fee, ATM, EFTPOS
      - Skip: salary, deposit, credit, transfer in, interest, refund
      - Dates, descriptions, and amounts must have matching counts
      - Process chronologically top to bottom
      </rules>
```

### System prompt handling

The Bedrock Converse API supports system prompts natively:

```python
response = client.converse(
    modelId="anthropic.claude-sonnet-3-5-v2",
    system=[{"text": system_prompt}],       # <-- separate from messages
    messages=[{
        "role": "user",
        "content": [
            {"image": {"format": "png", "source": {"bytes": image_bytes}}},
            {"text": user_prompt},
        ],
    }],
    inferenceConfig={"maxTokens": 800},
)
```

This means the YAML schema needs a new `system` key alongside `prompt`:

```yaml
prompts:
  invoice:
    name: Invoice Extraction
    system: |                    # <-- NEW: sent as Bedrock system parameter
      You extract structured data from Australian business documents.
      ...
    prompt: |                    # <-- existing: sent as user message
      <task>Extract all fields from this invoice image.</task>
      ...
```

The `BedrockSonnetBackend.generate()` method reads both keys and maps them to the correct API parameters. For HF backends, the `system` key is ignored (or prepended to the prompt if desired).

### Prompt conversion checklist

For each prompt in `internvl3_prompts.yaml` / `llama_prompts.yaml`:

| Step | Action |
|------|--------|
| 1 | Extract role/global rules into `system` key |
| 2 | Wrap the core task description in `<task>` tags |
| 3 | List fields in a clean `<fields>` block (one per line, no "NOT_FOUND if not visible" repetition) |
| 4 | Consolidate all business rules into a single `<rules>` block |
| 5 | Remove `CONVERSATION PROTOCOL` / output suppression blocks |
| 6 | Remove `CRITICAL` / `IMPORTANT` emphasis markers |
| 7 | Verify total word count is <50% of original |
| 8 | Test with a few representative images and compare extraction accuracy |

### Cost impact of prompt optimisation

Shorter prompts directly reduce Bedrock costs:

| Prompt version | Approx input tokens | Cost per image (standard) | Cost per image (bank, 2 turns) |
|---------------|---------------------|---------------------------|--------------------------------|
| Current (unmodified) | ~800-1,200 | ~$0.015-0.025 | ~$0.05-0.08 |
| Claude-optimised | ~300-500 | ~$0.006-0.012 | ~$0.02-0.04 |
| **Savings** | **~50-60%** | **~50%** | **~50%** |

At 1,000 images/month, that's roughly $10-20/month saved just from prompt optimisation.

---

## Effort Estimate

| Component | New/Modified | Estimated Lines | Complexity |
|-----------|-------------|-----------------|------------|
| `models/backends/bedrock.py` | New | ~80-100 | Low -- straightforward API wrapper |
| `models/model_loader.py` (BedrockSpec) | Modified | ~60 | Low -- follows VllmSpec pattern |
| `models/registry.py` (registration) | Modified | ~10 | Trivial |
| `prompts/sonnet_prompts.yaml` | New | ~100-150 | Medium -- prompt engineering |
| API error handling + retries | New (in backend) | ~40-60 | Medium |
| Concurrent API calls (optional) | New | ~80-120 | Medium-High |
| Config support (`api.bedrock` section) | Modified | ~30 | Low |
| **Total** | | **~400-520 lines** | |

The core integration (backend + registration + prompts) is ~2-3 days of work. Concurrent API calls and robust error handling add another ~2-3 days. Prompt tuning for Claude is ongoing.

---

## Recommended Approach

1. **Start simple**: Implement the backend with sequential processing. Get a single image extracting correctly end-to-end.
2. **Validate prompts**: Test whether existing prompts work with Claude or if Claude-specific prompts improve accuracy. Claude may significantly outperform the open VLMs on extraction accuracy.
3. **Add concurrency**: Once the basic flow works, add concurrent API calls for throughput.
4. **Benchmark**: Compare accuracy, latency, and cost against local InternVL3/Llama to inform the build-vs-buy decision.

---

## Prerequisites

- AWS account with Bedrock access enabled for Claude Sonnet 3.5
- `boto3` added to the conda environment
- IAM role/credentials with `bedrock:InvokeModel` permission
- Network access to Bedrock endpoint from the execution environment
