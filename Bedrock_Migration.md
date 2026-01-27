# Amazon Bedrock Migration Guide

This document outlines the code changes required to migrate the LMM_POC information extraction pipeline from self-hosted models (Llama-3.2-Vision, InternVL3) to Amazon Bedrock.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Comparison](#architecture-comparison)
3. [New Dependencies](#new-dependencies)
4. [Bedrock Client Implementation](#bedrock-client-implementation)
5. [Files Requiring Changes](#files-requiring-changes)
6. [Components That Stay the Same](#components-that-stay-the-same)
7. [Bedrock-Specific Considerations](#bedrock-specific-considerations)
8. [AWS Configuration](#aws-configuration)
9. [Available Bedrock Models](#available-bedrock-models)
10. [Migration Strategy Options](#migration-strategy-options)
11. [Estimated Effort](#estimated-effort)

---

## Executive Summary

The current architecture is **well-modularized**, making Bedrock migration straightforward. The main work involves replacing the model loading/inference layer while keeping the extraction parser, evaluation metrics, and reporting components unchanged.

**Key Benefits of Bedrock Migration:**
- No GPU infrastructure management
- Serverless scaling
- Access to Claude 3.5 Sonnet (strong document understanding)
- Pay-per-use pricing
- Simplified deployment

**Key Trade-offs:**
- Per-request API costs vs. fixed GPU costs
- Network latency vs. local inference
- AWS dependency vs. self-hosted flexibility

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
    CLAUDE_35_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    AMAZON_NOVA_PRO = "amazon.nova-pro-v1:0"

    def __init__(
        self,
        model_id: str = CLAUDE_35_SONNET,
        region: str = "us-east-1",
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

        # Bedrock Messages API format (Anthropic models)
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

                # Track token usage
                if "usage" in result:
                    self.total_input_tokens += result["usage"].get("input_tokens", 0)
                    self.total_output_tokens += result["usage"].get("output_tokens", 0)

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
        - Claude 3.5 Sonnet: $0.003/1K input, $0.015/1K output
        - Claude 3 Haiku: $0.00025/1K input, $0.00125/1K output
        """
        if "sonnet" in self.model_id.lower():
            input_rate = 0.003 / 1000
            output_rate = 0.015 / 1000
        elif "haiku" in self.model_id.lower():
            input_rate = 0.00025 / 1000
            output_rate = 0.00125 / 1000
        else:
            # Default/unknown model
            input_rate = 0.003 / 1000
            output_rate = 0.015 / 1000

        return (self.total_input_tokens * input_rate) + (
            self.total_output_tokens * output_rate
        )

    def reset_usage_stats(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# Convenience function matching existing loader pattern
def load_bedrock_client(
    model_id: str = BedrockVisionClient.CLAUDE_35_SONNET,
    region: str = "us-east-1",
) -> BedrockVisionClient:
    """
    Load Bedrock client (drop-in replacement for load_llama_model).

    Args:
        model_id: Bedrock model identifier
        region: AWS region

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
    client = load_bedrock_client(
        model_id=BedrockVisionClient.CLAUDE_35_SONNET,
        region="us-east-1"
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

Minimum required permissions:

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
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-*",
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-*",
                "arn:aws:bedrock:*::foundation-model/amazon.nova-*"
            ]
        }
    ]
}
```

### Credential Setup

```bash
# Option 1: AWS CLI configuration
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Option 3: IAM Role (recommended for EC2/Lambda)
# Attach IAM role to compute resource - no credentials needed in code
```

### Enable Model Access

1. Go to AWS Console > Amazon Bedrock
2. Navigate to "Model access" in left sidebar
3. Click "Manage model access"
4. Enable access for:
   - Anthropic Claude 3.5 Sonnet
   - Anthropic Claude 3 Haiku
   - Amazon Nova (if desired)
5. Submit and wait for access approval

---

## Available Bedrock Models

### Recommended for Document Extraction

| Model | Model ID | Vision | Strengths | Cost |
|-------|----------|--------|-----------|------|
| **Claude 3.5 Sonnet** | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Yes | Best accuracy, complex documents | $$$ |
| **Claude 3 Haiku** | `anthropic.claude-3-haiku-20240307-v1:0` | Yes | Fast, cost-effective, good accuracy | $ |
| **Amazon Nova Pro** | `amazon.nova-pro-v1:0` | Yes | AWS-native, competitive pricing | $$ |
| **Amazon Nova Lite** | `amazon.nova-lite-v1:0` | Yes | Budget option, simpler documents | $ |

### Model Selection Guide

```python
# High accuracy requirement (invoices, complex statements)
client = load_bedrock_client(model_id=BedrockVisionClient.CLAUDE_35_SONNET)

# Cost-sensitive, high volume
client = load_bedrock_client(model_id=BedrockVisionClient.CLAUDE_3_HAIKU)

# AWS-native preference
client = load_bedrock_client(model_id="amazon.nova-pro-v1:0")
```

### Pricing Comparison (as of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| Claude 3.5 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |
| Amazon Nova Pro | $0.0008 | $0.0032 |
| Amazon Nova Lite | $0.00006 | $0.00024 |

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

## Estimated Effort

| Task | Complexity | Time Estimate |
|------|------------|---------------|
| Create Bedrock client wrapper | Medium | 1-2 days |
| Update configuration management | Low | 0.5 day |
| Update batch processing scripts | Medium | 1 day |
| Test and tune prompts for Claude | Medium | 1-2 days |
| Add rate limiting and error handling | Low | 0.5 day |
| Update environment.yml | Low | 0.5 hour |
| Documentation and testing | Low | 1 day |
| **Total** | | **5-7 days** |

### Migration Checklist

- [ ] Enable Bedrock model access in AWS Console
- [ ] Configure IAM permissions
- [ ] Set up AWS credentials
- [ ] Create `common/bedrock_model_client.py`
- [ ] Update `environment.yml` with boto3
- [ ] Create/update prompts for Claude (if needed)
- [ ] Update extraction scripts to use Bedrock client
- [ ] Test with sample documents
- [ ] Implement rate limiting and error handling
- [ ] Add cost tracking and monitoring
- [ ] Run full evaluation against ground truth
- [ ] Document any prompt adjustments needed
- [ ] Deploy to production environment

---

## Appendix: Quick Start

### Minimal Working Example

```python
"""Minimal Bedrock extraction example."""

import boto3
import base64
import json
from PIL import Image
from io import BytesIO

# Initialize client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Load and encode image
image = Image.open("document.png")
buffer = BytesIO()
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Create request
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64
                }
            },
            {
                "type": "text",
                "text": "Extract all text fields from this invoice. Return as JSON."
            }
        ]
    }]
}

# Call Bedrock
response = client.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps(body)
)

# Parse response
result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

---

*Document created: January 2026*
*Last updated: January 2026*
