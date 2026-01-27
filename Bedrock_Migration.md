# Amazon Bedrock Migration Guide

This document outlines the code changes required to migrate the LMM_POC information extraction pipeline from self-hosted models (Llama-3.2-Vision, InternVL3) to Amazon Bedrock.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Australian Data Residency Requirements](#australian-data-residency-requirements)
3. [Architecture Comparison](#architecture-comparison)
4. [New Dependencies](#new-dependencies)
5. [Bedrock Client Implementation](#bedrock-client-implementation)
6. [Files Requiring Changes](#files-requiring-changes)
7. [Components That Stay the Same](#components-that-stay-the-same)
8. [Bedrock-Specific Considerations](#bedrock-specific-considerations)
9. [AWS Configuration](#aws-configuration)
10. [Available Bedrock Models](#available-bedrock-models)
11. [Migration Strategy Options](#migration-strategy-options)
12. [SageMaker Alternative for Custom Models](#sagemaker-alternative-for-custom-models)
13. [Estimated Effort](#estimated-effort)

---

## Executive Summary

The current architecture is **well-modularized**, making Bedrock migration straightforward. The main work involves replacing the model loading/inference layer while keeping the extraction parser, evaluation metrics, and reporting components unchanged.

> **IMPORTANT: Australian Hosting Requirement**
>
> This deployment requires Australian data residency. See [Australian Data Residency Requirements](#australian-data-residency-requirements) for critical constraints on model selection.

**Key Benefits of Bedrock Migration:**
- No GPU infrastructure management
- Serverless scaling
- Australian data residency compliance (ap-southeast-2 Sydney region)
- Pay-per-use pricing
- Simplified deployment

**Key Trade-offs:**
- Per-request API costs vs. fixed GPU costs
- Network latency vs. local inference
- AWS dependency vs. self-hosted flexibility
- **Model limitation**: Claude models not available in Sydney - must use Amazon Nova

---

## Australian Data Residency Requirements

> **CRITICAL**: This section contains mandatory requirements for Australian deployment.

### Region Selection

| Region | Code | Status |
|--------|------|--------|
| **Sydney** | `ap-southeast-2` | **Required for data residency** |
| Melbourne | `ap-southeast-4` | Available (CRIS routing with Sydney) |

All API calls must use `region_name="ap-southeast-2"` to ensure data stays in Australia.

### Model Availability in Sydney (ap-southeast-2)

**Claude models are NOT available in the Sydney region.** Only Amazon Nova models support Australian data residency via native Bedrock:

| Model | Available in Sydney | Vision Support | Recommendation |
|-------|:------------------:|:--------------:|----------------|
| Claude 3.5 Sonnet | :x: No | Yes | Not available |
| Claude 3 Haiku | :x: No | Yes | Not available |
| **Amazon Nova Pro** | :white_check_mark: Yes | Yes | **Recommended for production** |
| **Amazon Nova Lite** | :white_check_mark: Yes | Yes | Budget option |
| Amazon Nova Micro | :white_check_mark: Yes | No (text only) | Not suitable for document extraction |

### Bedrock Custom Model Import - NOT Available in Sydney

> **CRITICAL**: Bedrock Custom Model Import is **not available** in ap-southeast-2 (Sydney).

Custom Model Import regions:
- :white_check_mark: US-East (N. Virginia)
- :white_check_mark: US-West (Oregon)
- :white_check_mark: Europe (Frankfurt)
- :x: **Sydney (ap-southeast-2) - NOT AVAILABLE**

This means the following models **cannot be used in Australia** via Bedrock:

| Model | Bedrock Support | Sydney Availability | Alternative |
|-------|-----------------|:------------------:|-------------|
| **Qwen2.5-VL** | Custom Import | :x: No | SageMaker in Sydney |
| **InternVL3** | Not supported | :x: No | SageMaker in Sydney |
| Llama 3.2 Vision | Custom Import | :x: No | SageMaker in Sydney |

### Implications for This Project

1. **Must use Amazon Nova Pro or Nova Lite** for serverless Bedrock in Sydney
2. **Cannot use Claude, Qwen2.5-VL, or InternVL3** via Bedrock with Australian data residency
3. **Alternative: Amazon SageMaker** can host Qwen2.5-VL or InternVL3 in Sydney (see [SageMaker Alternative](#sagemaker-alternative-for-custom-models))
4. **Prompts may need adjustment** - Nova has different response characteristics
5. **Evaluation required** - Nova Pro should be benchmarked against current Llama/InternVL3 results

### Cross-Region Inference (CRIS)

AWS offers Cross-Region Inference that routes between Sydney and Melbourne within Australia:
- Traffic stays within Australian geography
- Automatic load balancing between regions
- Does **not** enable Claude models (still unavailable in both Australian regions)

### Code Configuration for Australia

```python
# REQUIRED: Australian region configuration
BEDROCK_REGION = "ap-southeast-2"  # Sydney

# REQUIRED: Use Amazon Nova (Claude not available in Australia)
DEFAULT_MODEL = "amazon.nova-pro-v1:0"

# Initialize client for Australian deployment
client = boto3.client("bedrock-runtime", region_name="ap-southeast-2")
```

### Compliance Checklist

- [ ] Confirm `ap-southeast-2` region in all Bedrock client configurations
- [ ] Use Amazon Nova Pro (not Claude) for all inference
- [ ] Verify no cross-region calls to non-Australian regions
- [ ] Document data residency compliance for audit purposes
- [ ] Test Nova Pro accuracy against current model benchmarks

---

## Architecture Comparison

| Component | Current (HuggingFace) | Bedrock |
|-----------|----------------------|---------|
| **Model Loading** | `AutoModelForVision2Seq.from_pretrained()` | `boto3.client("bedrock-runtime")` |
| **Image Input** | PIL Image to processor tensor | PIL Image to base64 string |
| **Inference** | `model.generate()` | `client.invoke_model()` |
| **Quantization** | BitsAndBytesConfig 8-bit | Not needed (managed by AWS) |
| **GPU Management** | Manual CUDA/device_map | Not needed (serverless) |
| **Token Limits** | Manual truncation | API parameter `max_tokens` |
| **Memory Management** | ResilientGenerator, OOM fallback | Not needed |
| **Multi-GPU** | device_map, split_model() | Not needed |

---

## New Dependencies

Update `environment.yml` to add AWS SDK:

```yaml
name: vision_notebooks
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    # NEW: AWS Bedrock dependencies
    - boto3>=1.34.0
    - botocore>=1.34.0

    # KEEP: Still needed for image processing and evaluation
    - pillow>=10.0.0
    - pandas>=2.0.0
    - pyyaml>=6.0
    - rich>=13.0.0

    # OPTIONAL: Remove if fully migrating away from local models
    # - transformers==4.45.2
    # - torch>=2.0.0
    # - torchvision
    # - accelerate
    # - bitsandbytes
```

---

## Bedrock Client Implementation

Create a new file `common/bedrock_model_client.py`:

```python
"""
Amazon Bedrock client for vision-language model inference.

Replaces HuggingFace model loaders for cloud-based inference.
"""

import base64
import json
import time
from io import BytesIO
from typing import Any

import boto3
from botocore.exceptions import ClientError, ThrottlingException
from PIL import Image
from rich.console import Console

console = Console()


class BedrockVisionClient:
    """Bedrock client for vision-language models."""

    # Supported model IDs
    # NOTE: Claude models NOT available in ap-southeast-2 (Sydney)
    CLAUDE_35_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"  # Not available in Australia
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"  # Not available in Australia

    # Amazon Nova - Available in Sydney (ap-southeast-2)
    AMAZON_NOVA_PRO = "amazon.nova-pro-v1:0"   # Recommended for Australia
    AMAZON_NOVA_LITE = "amazon.nova-lite-v1:0"  # Budget option for Australia

    def __init__(
        self,
        model_id: str = AMAZON_NOVA_PRO,  # Default to Nova Pro for Australian deployment
        region: str = "ap-southeast-2",    # Default to Sydney for Australian data residency
        max_tokens: int = 4096,
    ):
        """
        Initialize Bedrock client.

        Args:
            model_id: Bedrock model identifier
            region: AWS region for Bedrock service
            max_tokens: Maximum tokens in response
        """
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.region = region

        # Track usage for cost estimation
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        console.print(f"[green]Bedrock client initialized[/green]")
        console.print(f"[cyan]Model: {model_id} | Region: {region}[/cyan]")

    def encode_image_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string for Bedrock API.

        Args:
            image: PIL Image object
            format: Image format (PNG recommended for quality)

        Returns:
            Base64-encoded image string
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _get_media_type(self, format: str) -> str:
        """Map image format to MIME type."""
        format_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "GIF": "image/gif",
            "WEBP": "image/webp",
        }
        return format_map.get(format.upper(), "image/png")

    def extract(
        self,
        image: Image.Image,
        prompt: str,
        max_retries: int = 3,
        image_format: str = "PNG",
    ) -> str:
        """
        Send image + prompt to Bedrock and get extraction response.

        Args:
            image: PIL Image of document
            prompt: Extraction prompt text
            max_retries: Number of retries on throttling
            image_format: Format for image encoding

        Returns:
            Model response text
        """
        image_b64 = self.encode_image_base64(image, format=image_format)
        media_type = self._get_media_type(image_format)

        # Build request body based on model type
        if "nova" in self.model_id.lower():
            # Amazon Nova format (required for Australian deployment)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": image_format.lower(),
                                    "source": {"bytes": image_b64},
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
                "inferenceConfig": {"maxTokens": self.max_tokens},
            }
        else:
            # Anthropic Claude format (not available in Australia)
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }

        # Retry logic for throttling
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id, body=json.dumps(body)
                )

                result = json.loads(response["body"].read())

                # Track token usage (format differs by model)
                if "usage" in result:
                    self.total_input_tokens += result["usage"].get("input_tokens", 0)
                    self.total_output_tokens += result["usage"].get("output_tokens", 0)

                # Parse response based on model type
                if "nova" in self.model_id.lower():
                    # Amazon Nova response format
                    return result["output"]["message"]["content"][0]["text"]
                else:
                    # Anthropic Claude response format
                    return result["content"][0]["text"]

            except ThrottlingException:
                wait_time = 2**attempt
                console.print(
                    f"[yellow]Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})[/yellow]"
                )
                time.sleep(wait_time)

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_msg = e.response["Error"]["Message"]
                console.print(f"[red]Bedrock error: {error_code} - {error_msg}[/red]")
                raise

        raise Exception(f"Max retries ({max_retries}) exceeded due to throttling")

    def extract_batch(
        self,
        images: list[Image.Image],
        prompt: str,
        progress_callback: callable = None,
    ) -> list[dict[str, Any]]:
        """
        Process multiple images with the same prompt.

        Args:
            images: List of PIL Images
            prompt: Extraction prompt (same for all)
            progress_callback: Optional callback(current, total)

        Returns:
            List of results with 'response' and 'success' keys
        """
        results = []

        for i, image in enumerate(images):
            try:
                response = self.extract(image, prompt)
                results.append({"response": response, "success": True, "error": None})
            except Exception as e:
                results.append({"response": None, "success": False, "error": str(e)})

            if progress_callback:
                progress_callback(i + 1, len(images))

        return results

    def get_usage_stats(self) -> dict:
        """Get cumulative token usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimate_cost(),
        }

    def estimate_cost(self) -> float:
        """
        Estimate cost based on token usage.

        Pricing (as of 2024, verify current rates):
        - Amazon Nova Pro: $0.0008/1K input, $0.0032/1K output
        - Amazon Nova Lite: $0.00006/1K input, $0.00024/1K output
        - Claude 3.5 Sonnet: $0.003/1K input, $0.015/1K output (not available in Australia)
        """
        if "nova-pro" in self.model_id.lower():
            input_rate = 0.0008 / 1000
            output_rate = 0.0032 / 1000
        elif "nova-lite" in self.model_id.lower():
            input_rate = 0.00006 / 1000
            output_rate = 0.00024 / 1000
        elif "sonnet" in self.model_id.lower():
            input_rate = 0.003 / 1000
            output_rate = 0.015 / 1000
        elif "haiku" in self.model_id.lower():
            input_rate = 0.00025 / 1000
            output_rate = 0.00125 / 1000
        else:
            # Default to Nova Pro pricing
            input_rate = 0.0008 / 1000
            output_rate = 0.0032 / 1000

        return (self.total_input_tokens * input_rate) + (
            self.total_output_tokens * output_rate
        )

    def reset_usage_stats(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# Convenience function matching existing loader pattern
def load_bedrock_client(
    model_id: str = BedrockVisionClient.AMAZON_NOVA_PRO,  # Nova Pro for Australian deployment
    region: str = "ap-southeast-2",  # Sydney for Australian data residency
) -> BedrockVisionClient:
    """
    Load Bedrock client (drop-in replacement for load_llama_model).

    Args:
        model_id: Bedrock model identifier (default: Nova Pro for Australia)
        region: AWS region (default: ap-southeast-2 Sydney for Australia)

    Returns:
        Configured BedrockVisionClient
    """
    return BedrockVisionClient(model_id=model_id, region=region)
```

---

## Files Requiring Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `common/config.py` | Modify | Add Bedrock model IDs, region config; make GPU config optional |
| `common/llama_model_loader_robust.py` | Replace or Keep | Replace with Bedrock wrapper OR keep for hybrid setup |
| `common/internvl3_model_loader.py` | Replace or Keep | Replace with Bedrock wrapper OR keep for hybrid setup |
| `common/gpu_optimization.py` | Make Optional | Not needed for Bedrock; keep for local model fallback |
| `llama_keyvalue.py` | Modify | Swap model loader import, remove CUDA-specific code |
| `internvl3_keyvalue.py` | Modify | Swap model loader import, remove CUDA-specific code |
| `bank_statement/*_batch_table.py` | Modify | Update model loading, remove GPU memory management |
| `prompts/*.yaml` | Review | May need minor adjustments for Claude's response style |
| `common/batch_processor.py` | Modify | Remove GPU memory management, add rate limiting |
| `environment.yml` | Modify | Add boto3, optionally remove torch/transformers |

### Example: Updated Extraction Script

```python
# bedrock_keyvalue.py - New extraction script for Bedrock

from pathlib import Path
from PIL import Image
from rich.console import Console

from common.bedrock_model_client import load_bedrock_client, BedrockVisionClient
from common.simple_prompt_loader import load_prompt
from common.extraction_parser import hybrid_parse_response
from common.image_preprocessing import enhance_statement_quality

console = Console()

def extract_document(image_path: str, document_type: str = "universal") -> dict:
    """Extract fields from document using Bedrock."""

    # Load Bedrock client (no GPU needed!)
    # Using Amazon Nova Pro in Sydney for Australian data residency
    client = load_bedrock_client(
        model_id=BedrockVisionClient.AMAZON_NOVA_PRO,
        region="ap-southeast-2"  # Sydney - Australian data residency
    )

    # Load and preprocess image
    image = Image.open(image_path)
    image = enhance_statement_quality(image)

    # Load prompt (same YAML prompts work)
    prompt = load_prompt("prompts/bedrock_prompts.yaml", document_type)

    # Extract via Bedrock API
    response = client.extract(image, prompt)

    # Parse response (existing parser works unchanged)
    extracted = hybrid_parse_response(response, expected_fields=[])

    # Print cost estimate
    stats = client.get_usage_stats()
    console.print(f"[dim]Cost: ${stats['estimated_cost_usd']:.4f}[/dim]")

    return extracted


if __name__ == "__main__":
    result = extract_document("test_documents/invoice_001.png", "invoice")
    console.print(result)
```

---

## Components That Stay the Same

The modular architecture means these components require **no changes**:

| Component | File | Reason |
|-----------|------|--------|
| Extraction Parser | `common/extraction_parser.py` | Already handles JSON and plain text responses |
| Evaluation Metrics | `common/evaluation_metrics.py` | Model-agnostic comparison logic |
| Field Definitions | `config/field_definitions.yaml` | Schema unchanged |
| Image Preprocessing | `common/image_preprocessing.py` | Still useful for quality enhancement |
| Batch Reporting | `common/batch_reporting.py` | Output format unchanged |
| Batch Visualizations | `common/batch_visualizations.py` | Metrics visualization unchanged |
| Ground Truth | `data/ground_truth.csv` | Evaluation data unchanged |

---

## Bedrock-Specific Considerations

### Rate Limiting

Bedrock has request rate limits. Implement exponential backoff:

```python
import time
from botocore.exceptions import ThrottlingException

def extract_with_retry(client, image, prompt, max_retries=3):
    """Extract with exponential backoff on throttling."""
    for attempt in range(max_retries):
        try:
            return client.extract(image, prompt)
        except ThrottlingException:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            console.print(f"[yellow]Rate limited, waiting {wait_time}s[/yellow]")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

### Cost Tracking

Track token usage for cost management:

```python
def log_batch_cost(client: BedrockVisionClient, batch_name: str):
    """Log cost after batch processing."""
    stats = client.get_usage_stats()
    console.print(f"\n[bold]Batch: {batch_name}[/bold]")
    console.print(f"  Input tokens:  {stats['total_input_tokens']:,}")
    console.print(f"  Output tokens: {stats['total_output_tokens']:,}")
    console.print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
```

### Image Size Optimization

Bedrock charges by token, and large images increase costs:

```python
def optimize_image_for_bedrock(image: Image.Image, max_dimension: int = 2048) -> Image.Image:
    """Resize image to reduce token costs while maintaining quality."""
    width, height = image.size

    if max(width, height) > max_dimension:
        ratio = max_dimension / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    return image
```

### Error Handling

Handle Bedrock-specific errors:

```python
from botocore.exceptions import ClientError

try:
    response = client.extract(image, prompt)
except ClientError as e:
    error_code = e.response["Error"]["Code"]

    if error_code == "ValidationException":
        console.print("[red]Invalid request - check image format/size[/red]")
    elif error_code == "AccessDeniedException":
        console.print("[red]Access denied - check IAM permissions[/red]")
    elif error_code == "ModelNotReadyException":
        console.print("[yellow]Model warming up - retry in a moment[/yellow]")
    else:
        raise
```

---

## AWS Configuration

### Prerequisites

1. **AWS Account** with Bedrock access enabled
2. **IAM User/Role** with Bedrock permissions
3. **Model Access** enabled in Bedrock console for desired models

### IAM Policy

Minimum required permissions for Australian deployment (Sydney region):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:ap-southeast-2::foundation-model/amazon.nova-pro-v1:0",
                "arn:aws:bedrock:ap-southeast-2::foundation-model/amazon.nova-lite-v1:0"
            ]
        }
    ]
}
```

> **Note**: This policy restricts access to the Sydney region (`ap-southeast-2`) and Amazon Nova models only, enforcing Australian data residency at the IAM level.

### Credential Setup

```bash
# Option 1: AWS CLI configuration
aws configure
# Enter: AWS Access Key ID, Secret Access Key
# Region: ap-southeast-2 (Sydney - REQUIRED for Australian data residency)

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-southeast-2"  # Sydney - Australian data residency

# Option 3: IAM Role (recommended for EC2/Lambda in ap-southeast-2)
# Attach IAM role to compute resource - no credentials needed in code
# Ensure EC2/Lambda is deployed in ap-southeast-2 region
```

### Enable Model Access

1. Go to AWS Console > Amazon Bedrock
2. **Ensure you are in the Sydney region (ap-southeast-2)** - check top-right region selector
3. Navigate to "Model access" in left sidebar
4. Click "Manage model access"
5. Enable access for:
   - **Amazon Nova Pro** (required for document extraction)
   - **Amazon Nova Lite** (optional, budget alternative)
   - ~~Anthropic Claude~~ (not available in Sydney region)
6. Submit and wait for access approval

> **Important**: Model access must be enabled in the ap-southeast-2 region specifically. Access enabled in other regions does not carry over.

---

## Available Bedrock Models

### Models Available in Sydney (ap-southeast-2)

> **For Australian data residency, only Amazon Nova models are available via native Bedrock.**

| Model | Model ID | Vision | Sydney (ap-southeast-2) | Recommendation |
|-------|----------|:------:|:-----------------------:|----------------|
| **Amazon Nova Pro** | `amazon.nova-pro-v1:0` | Yes | :white_check_mark: Available | **Production recommended** |
| **Amazon Nova Lite** | `amazon.nova-lite-v1:0` | Yes | :white_check_mark: Available | Budget/high-volume |
| Amazon Nova Micro | `amazon.nova-micro-v1:0` | No | :white_check_mark: Available | Text-only (not suitable) |
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-*` | Yes | :x: Not available | Cannot use in Australia |
| Claude 3 Haiku | `anthropic.claude-3-haiku-*` | Yes | :x: Not available | Cannot use in Australia |

### Models Available via Custom Model Import (NOT in Sydney)

These models can be imported to Bedrock, but **only in US/EU regions** - not Sydney:

| Model | Architecture | Vision | Sydney | US/EU Regions |
|-------|--------------|:------:|:------:|:-------------:|
| **Qwen2.5-VL** | Qwen2_5_VL | Yes | :x: | :white_check_mark: |
| **Llama 3.2 Vision** | Mllama | Yes | :x: | :white_check_mark: |
| InternVL3 | Not supported | Yes | :x: | :x: |

> **For Qwen2.5-VL or InternVL3 in Australia**: Use [SageMaker](#sagemaker-alternative-for-custom-models) instead.

### Model Selection Guide (Australian Deployment)

```python
# RECOMMENDED: Production document extraction in Australia
client = load_bedrock_client(
    model_id=BedrockVisionClient.AMAZON_NOVA_PRO,
    region="ap-southeast-2"
)

# Budget option for high-volume processing
client = load_bedrock_client(
    model_id=BedrockVisionClient.AMAZON_NOVA_LITE,
    region="ap-southeast-2"
)

# NOT AVAILABLE IN AUSTRALIA - will fail:
# client = load_bedrock_client(model_id=BedrockVisionClient.CLAUDE_35_SONNET)
```

### Pricing Comparison (as of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) | Available in Sydney |
|-------|----------------------|------------------------|:-------------------:|
| **Amazon Nova Pro** | $0.0008 | $0.0032 | :white_check_mark: |
| **Amazon Nova Lite** | $0.00006 | $0.00024 | :white_check_mark: |
| Claude 3.5 Sonnet | $0.003 | $0.015 | :x: |
| Claude 3 Haiku | $0.00025 | $0.00125 | :x: |

> **Cost advantage**: Amazon Nova Pro is ~4x cheaper than Claude 3.5 Sonnet for input tokens and ~5x cheaper for output tokens.

*Verify current pricing at [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)*

---

## Migration Strategy Options

### Option A: Full Replacement

Remove all HuggingFace/transformers code and use only Bedrock.

**Pros:**
- Simplest codebase
- No GPU infrastructure needed
- Consistent API across all extractions

**Cons:**
- AWS dependency
- Per-request costs
- Network latency

**Implementation:**
1. Remove `torch`, `transformers`, `bitsandbytes` from environment.yml
2. Delete `llama_model_loader_robust.py`, `internvl3_model_loader.py`
3. Delete `gpu_optimization.py`
4. Update all extraction scripts to use `bedrock_model_client.py`

### Option B: Abstraction Layer (Recommended)

Create a common interface supporting both local and Bedrock models.

**Pros:**
- Flexibility to switch between providers
- Keep local models for development/testing
- Gradual migration path

**Cons:**
- More code to maintain
- Interface overhead

**Implementation:**

```python
# common/model_interface.py
from typing import Protocol
from PIL import Image

class VisionModelClient(Protocol):
    """Protocol for vision-language model clients."""

    def extract(self, image: Image.Image, prompt: str) -> str:
        """Extract text from image using prompt."""
        ...

# Implementations
class LlamaClient(VisionModelClient):
    """Local Llama-3.2-Vision client."""
    ...

class InternVL3Client(VisionModelClient):
    """Local InternVL3 client."""
    ...

class BedrockClient(VisionModelClient):
    """AWS Bedrock client."""
    ...

# Factory function
def get_model_client(provider: str = "bedrock") -> VisionModelClient:
    """Get model client by provider name."""
    clients = {
        "llama": LlamaClient,
        "internvl3": InternVL3Client,
        "bedrock": BedrockClient,
        "bedrock-haiku": lambda: BedrockClient(model_id="anthropic.claude-3-haiku-*"),
    }
    return clients[provider]()
```

### Option C: Hybrid Approach

Use Bedrock for production, local models for development.

**Pros:**
- No API costs during development
- Production reliability of managed service
- Best of both worlds

**Implementation:**
```python
import os

def get_client():
    """Get appropriate client based on environment."""
    if os.environ.get("USE_BEDROCK", "false").lower() == "true":
        return load_bedrock_client()
    else:
        return load_llama_model()  # Local development
```

---

## SageMaker Alternative for Custom Models

> **Use this option if you need Qwen2.5-VL or InternVL3 with Australian data residency.**

Since Bedrock Custom Model Import is not available in Sydney, Amazon SageMaker provides an alternative path to host custom vision-language models in ap-southeast-2.

### When to Use SageMaker

| Requirement | Bedrock (Nova) | SageMaker |
|-------------|:--------------:|:---------:|
| Australian data residency | :white_check_mark: | :white_check_mark: |
| Serverless (no infrastructure) | :white_check_mark: | :x: |
| Use InternVL3 | :x: | :white_check_mark: |
| Use Qwen2.5-VL | :x: | :white_check_mark: |
| Managed scaling | :white_check_mark: | Partial |
| Lowest operational burden | :white_check_mark: | :x: |

### SageMaker Deployment Options

#### Option 1: SageMaker JumpStart (Easiest)

Pre-built model deployments with one-click setup:

```python
from sagemaker.jumpstart.model import JumpStartModel

# Deploy Qwen2.5-VL via JumpStart (check availability)
model = JumpStartModel(
    model_id="huggingface-vlm-qwen2-5-vl-7b-instruct",
    region="ap-southeast-2"
)
predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1
)
```

#### Option 2: SageMaker Endpoint (Full Control)

Custom container deployment for InternVL3 or Qwen2.5-VL:

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Configure for Sydney region
session = sagemaker.Session(boto_session=boto3.Session(region_name="ap-southeast-2"))

# Deploy InternVL3 or Qwen2.5-VL
huggingface_model = HuggingFaceModel(
    model_data="s3://your-bucket-ap-southeast-2/model.tar.gz",
    role="arn:aws:iam::ACCOUNT:role/SageMakerRole",
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
    sagemaker_session=session
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",  # Or ml.p3.2xlarge for V100
    endpoint_name="internvl3-sydney-endpoint"
)
```

#### Option 3: SageMaker Inference Component (Cost-Optimized)

Share GPU instances across multiple models:

```python
# Create inference component for efficient GPU sharing
# Useful if running multiple model variants
```

### SageMaker Client Implementation

Create `common/sagemaker_model_client.py`:

```python
"""
Amazon SageMaker client for custom vision-language models.

Use this for InternVL3 or Qwen2.5-VL in Australian regions where
Bedrock Custom Model Import is not available.
"""

import base64
import json
from io import BytesIO

import boto3
from PIL import Image
from rich.console import Console

console = Console()


class SageMakerVisionClient:
    """SageMaker endpoint client for vision-language models."""

    def __init__(
        self,
        endpoint_name: str,
        region: str = "ap-southeast-2",
    ):
        """
        Initialize SageMaker client.

        Args:
            endpoint_name: Name of deployed SageMaker endpoint
            region: AWS region (default: Sydney for Australian data residency)
        """
        self.client = boto3.client("sagemaker-runtime", region_name=region)
        self.endpoint_name = endpoint_name
        self.region = region

        console.print(f"[green]SageMaker client initialized[/green]")
        console.print(f"[cyan]Endpoint: {endpoint_name} | Region: {region}[/cyan]")

    def encode_image_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def extract(self, image: Image.Image, prompt: str) -> str:
        """
        Send image + prompt to SageMaker endpoint.

        Args:
            image: PIL Image of document
            prompt: Extraction prompt text

        Returns:
            Model response text
        """
        image_b64 = self.encode_image_base64(image)

        # Payload format depends on your endpoint implementation
        payload = {
            "image": image_b64,
            "prompt": prompt,
            "max_new_tokens": 4096,
        }

        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())
        return result["generated_text"]


def load_sagemaker_client(
    endpoint_name: str,
    region: str = "ap-southeast-2",
) -> SageMakerVisionClient:
    """Load SageMaker client for Australian deployment."""
    return SageMakerVisionClient(endpoint_name=endpoint_name, region=region)
```

### Cost Comparison: Bedrock vs SageMaker

| Aspect | Bedrock (Nova Pro) | SageMaker (g5.2xlarge) |
|--------|-------------------|------------------------|
| **Pricing model** | Per-token | Per-hour instance |
| **Idle cost** | $0 | ~$1.50/hour |
| **1000 documents/day** | ~$5-15/day | ~$36/day (24h) |
| **10 documents/day** | ~$0.05-0.15/day | ~$36/day (24h) |
| **Burst capacity** | Automatic | Manual scaling |
| **Cold start** | None | 5-10 minutes |

**Recommendation:**
- **Low/variable volume**: Use Bedrock (Nova Pro) - pay only for usage
- **High sustained volume**: SageMaker may be cost-effective
- **Must use InternVL3/Qwen2.5-VL**: SageMaker is the only option for Australia

### SageMaker in Migration Checklist

If using SageMaker for custom models, add these tasks:

- [ ] Package model artifacts to S3 in ap-southeast-2
- [ ] Create SageMaker execution role with appropriate permissions
- [ ] Deploy endpoint in ap-southeast-2
- [ ] Test endpoint with sample documents
- [ ] Configure auto-scaling policies (if needed)
- [ ] Set up CloudWatch monitoring
- [ ] Create `common/sagemaker_model_client.py`

---

## Estimated Effort

### Bedrock Migration (Nova Pro)

| Task | Complexity | Time Estimate |
|------|------------|---------------|
| Create Bedrock client wrapper | Medium | 1-2 days |
| Update configuration management | Low | 0.5 day |
| Update batch processing scripts | Medium | 1 day |
| Test and tune prompts for Nova | Medium | 1-2 days |
| Add rate limiting and error handling | Low | 0.5 day |
| Update environment.yml | Low | 0.5 hour |
| Documentation and testing | Low | 1 day |
| **Total** | | **5-7 days** |

### SageMaker Migration (InternVL3/Qwen2.5-VL) - Additional Effort

| Task | Complexity | Time Estimate |
|------|------------|---------------|
| Package model for SageMaker | Medium | 1-2 days |
| Create deployment infrastructure | Medium | 1 day |
| Create SageMaker client wrapper | Medium | 1 day |
| Test endpoint in Sydney | Low | 0.5 day |
| Configure scaling/monitoring | Low | 0.5 day |
| **Additional Total** | | **4-5 days** |

### Migration Checklist

**Australian Data Residency Setup:**
- [ ] Confirm AWS account has access to ap-southeast-2 (Sydney) region
- [ ] Enable Amazon Nova Pro model access in Sydney region
- [ ] Configure IAM policy restricted to ap-southeast-2
- [ ] Set AWS_DEFAULT_REGION to ap-southeast-2

**Development:**
- [ ] Create `common/bedrock_model_client.py` with Nova format
- [ ] Update `environment.yml` with boto3
- [ ] Create/update prompts for Amazon Nova (different from Claude)
- [ ] Update extraction scripts to use Bedrock client

**Testing:**
- [ ] Test with sample documents using Nova Pro
- [ ] Benchmark Nova Pro accuracy against current Llama/InternVL3 results
- [ ] Implement rate limiting and error handling
- [ ] Add cost tracking and monitoring

**Validation:**
- [ ] Run full evaluation against ground truth
- [ ] Document any prompt adjustments needed for Nova
- [ ] Verify all API calls use ap-southeast-2 region
- [ ] Audit for any cross-region data leakage

**Deployment:**
- [ ] Deploy to production in ap-southeast-2
- [ ] Document data residency compliance for audit

---

## Appendix: Quick Start

### Minimal Working Example (Australian Deployment)

```python
"""Minimal Bedrock extraction example for Australian data residency."""

import boto3
import base64
import json
from PIL import Image
from io import BytesIO

# Initialize client - Sydney region for Australian data residency
client = boto3.client("bedrock-runtime", region_name="ap-southeast-2")

# Load and encode image
image = Image.open("document.png")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Create request (Amazon Nova format)
body = {
    "messages": [{
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": image_b64}
                }
            },
            {
                "text": "Extract all text fields from this invoice. Return as JSON."
            }
        ]
    }],
    "inferenceConfig": {
        "maxTokens": 4096
    }
}

# Call Bedrock with Amazon Nova Pro (available in Sydney)
response = client.invoke_model(
    modelId="amazon.nova-pro-v1:0",
    body=json.dumps(body)
)

# Parse response
result = json.loads(response["body"].read())
print(result["output"]["message"]["content"][0]["text"])
```

> **Note**: Amazon Nova uses a different request/response format than Claude. The example above shows the Nova-specific format required for Australian deployment.

---

## References

### Amazon Bedrock
- [Amazon Bedrock in Sydney Region Announcement](https://aws.amazon.com/about-aws/whats-new/2024/04/amazon-bedrock-sydney-region/)
- [Bedrock Model Support by Region](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Amazon Nova Documentation](https://docs.aws.amazon.com/nova/)

### Custom Model Import & Qwen
- [Bedrock Custom Model Import Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
- [Qwen Models on Amazon Bedrock](https://aws.amazon.com/blogs/aws/qwen-models-are-now-available-in-amazon-bedrock/)
- [Deploy Qwen Models with Bedrock Custom Model Import](https://aws.amazon.com/blogs/machine-learning/deploy-qwen-models-with-amazon-bedrock-custom-model-import/)
- [Bedrock Custom Model Import - Qwen Support Announcement](https://aws.amazon.com/about-aws/whats-new/2025/06/amazon-bedrock-custom-model-import-qwen-models/)

### Amazon SageMaker
- [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)
- [Deploy HuggingFace Models on SageMaker](https://huggingface.co/docs/sagemaker/inference)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)

---

*Document created: January 2026*
*Last updated: January 2026*
*Australian data residency requirements added: January 2026*
*Qwen2.5-VL and SageMaker alternatives added: January 2026*
