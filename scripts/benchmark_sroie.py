#!/usr/bin/env python3
# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""SROIE Benchmark Runner — InternVL3.5-8B vs Llama 4 Scout.

Standalone benchmark script for ICDAR 2019 SROIE Task 3 (receipt key
information extraction).  Reuses model loading from the registry but has
its own extraction loop and evaluation.

Usage:
    python benchmark_sroie.py --model internvl3 --data-dir ../data/sroie
    python benchmark_sroie.py --model llama4scout --data-dir ../data/sroie
    python benchmark_sroie.py --model internvl3 llama4scout --data-dir ../data/sroie
    python benchmark_sroie.py --model internvl3 --data-dir ../data/sroie --max-images 10
"""

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from PIL import Image
from rich.console import Console
from rich.table import Table

from common.pipeline_config import (
    PipelineConfig,
    auto_detect_model_path,
    load_yaml_config,
)
from common.sroie_evaluation import (
    SROIE_FIELDS,
    SROIEBenchmarkResult,
    SROIEImageResult,
    compute_sroie_metrics,
    load_sroie_ground_truth,
)
from models.registry import get_model

# Suppress "Setting pad_token_id to eos_token_id" spam from transformers
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

console = Console()
app = typer.Typer(
    name="benchmark-sroie",
    help="SROIE benchmark for vision-language model comparison.",
    add_completion=False,
)


# ============================================================================
# Response parsing
# ============================================================================

_FIELD_RE = re.compile(
    r"^(company|date|address|total)\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_sroie_response(raw_response: str) -> dict[str, str]:
    """Parse model response into SROIE field dict.

    Expects lines like:
        company: SOME STORE NAME
        date: 25/12/2018
        address: 123 Main St
        total: 4.95
    """
    result: dict[str, str] = {}
    for match in _FIELD_RE.finditer(raw_response):
        field = match.group(1).lower()
        value = match.group(2).strip()
        if value.upper() != "NOT_FOUND":
            result[field] = value
    return result


# ============================================================================
# Model inference dispatch
# ============================================================================


def _load_sroie_prompt() -> str:
    """Load the SROIE extraction prompt from YAML."""
    prompt_path = Path(__file__).parent / "prompts" / "sroie_prompts.yaml"
    with prompt_path.open() as f:
        data = yaml.safe_load(f)
    return data["extraction"]["prompt"]


def _run_inference_internvl3(
    model, tokenizer, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """InternVL3: preprocess PIL image to pixel_values tensor, then .chat()."""
    import torch

    from models.internvl3_image_preprocessor import InternVL3ImagePreprocessor

    preprocessor = InternVL3ImagePreprocessor(max_tiles=6)
    pixel_values = preprocessor.load_image_from_pil(image, model)
    model_device = InternVL3ImagePreprocessor.get_model_device(model)
    if pixel_values.device != model_device:
        pixel_values = pixel_values.to(model_device)

    # Set pad_token_id to suppress open-end generation warning
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config={
            "max_new_tokens": max_tokens,
            "do_sample": False,
        },
        history=None,
        return_history=False,
    )

    del pixel_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


def _run_inference_llama4scout(
    model, processor, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """Llama 4 Scout: apply_chat_template all-in-one API."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, output_ids, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


def _run_inference_granite4(
    model, processor, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """Granite 4.0 3B Vision: AutoProcessor + AutoModelForImageTextToText.

    Uses processor.apply_chat_template for message formatting, then
    processor() for image tokenization.  LoRA adapters are pre-merged
    at load time.
    """
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
    )

    # Trim input tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, output_ids, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


def _run_inference_nemotron(
    model, processor, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """Nemotron Nano 2 VL: tokenizer.apply_chat_template + processor().

    Uses two-step API: tokenizer formats chat template, processor handles
    image tokenization.  /no_think system message disables chain-of-thought.
    """
    import torch

    tokenizer = processor.tokenizer

    messages = [
        {"role": "system", "content": "/no_think"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )

    output_ids = model.generate(
        pixel_values=inputs.pixel_values,
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Trim input tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, output_ids, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


def _run_inference_generic(
    model, processor, image: Image.Image, prompt: str, max_tokens: int
) -> str:
    """Generic fallback: two-step apply_chat_template + processor()."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, output_ids, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


def _run_inference_vllm(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    model_type: str = "",
) -> str:
    """vLLM engine: use llm.chat() with base64 data URI for images."""
    import base64
    import io

    from vllm import SamplingParams

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0)

    # Qwen3.5 and Gemma4 enable thinking mode by default — disable it
    # to avoid <think>...</think> blocks in extraction output.
    chat_kwargs: dict = {}
    if model_type.startswith(("qwen35", "gemma4")):
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    outputs = model.chat(
        messages=messages, sampling_params=sampling, use_tqdm=False, **chat_kwargs
    )
    return outputs[0].outputs[0].text.strip()


def run_inference(
    model_type: str,
    model,
    tokenizer_or_processor,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    """Dispatch inference to the correct model-specific function."""
    if model_type in ("internvl3", "internvl3-14b", "internvl3-38b"):
        return _run_inference_internvl3(
            model, tokenizer_or_processor, image, prompt, max_tokens
        )
    if model_type == "llama4scout":
        return _run_inference_llama4scout(
            model, tokenizer_or_processor, image, prompt, max_tokens
        )
    if model_type in (
        "llama4scout-w4a16",
        "internvl3-vllm",
        "internvl3-14b-vllm",
        "internvl3-38b-vllm",
        "qwen3vl-vllm",
        "qwen35-vllm",
        "gemma4",
    ):
        return _run_inference_vllm(
            model, tokenizer_or_processor, image, prompt, max_tokens, model_type
        )
    if model_type == "granite4":
        return _run_inference_granite4(
            model, tokenizer_or_processor, image, prompt, max_tokens
        )
    if model_type == "nemotron":
        return _run_inference_nemotron(
            model, tokenizer_or_processor, image, prompt, max_tokens
        )
    # Fallback for qwen3vl and other models using two-step API
    return _run_inference_generic(
        model, tokenizer_or_processor, image, prompt, max_tokens
    )


# ============================================================================
# Benchmark runner
# ============================================================================


def run_benchmark(
    model_type: str,
    data_dir: Path,
    model_path: str | None = None,
    max_images: int | None = None,
    max_tokens: int = 256,
    quantization: str | None = None,
) -> SROIEBenchmarkResult:
    """Run SROIE benchmark for a single model.

    Args:
        model_type: Registry key (e.g. "internvl3", "llama4scout").
        data_dir: Path to data/sroie/ with img/ and key/ subdirs.
        model_path: Override model path (uses auto-detect if None).
        max_images: Cap on number of images to evaluate.
        max_tokens: Max generation tokens per image.
        quantization: Quantization strategy override.

    Returns:
        SROIEBenchmarkResult with per-image and aggregate metrics.
    """
    img_dir = data_dir / "img"
    key_dir = data_dir / "key"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not key_dir.exists():
        raise FileNotFoundError(f"Key directory not found: {key_dir}")

    # Load ground truth
    ground_truth = load_sroie_ground_truth(key_dir)
    console.print(f"Loaded {len(ground_truth)} ground truth entries")

    # Discover images with matching ground truth
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = sorted(
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in image_extensions and p.stem in ground_truth
    )

    if max_images:
        all_images = all_images[:max_images]

    console.print(f"Evaluating {len(all_images)} images with model: {model_type}")

    # Load model via registry
    registration = get_model(model_type)

    # Resolve model path: CLI > run_config.yml > auto-detect
    resolved_model_path: Path | None = Path(model_path) if model_path else None

    if resolved_model_path is None:
        # Try run_config.yml for default_paths
        run_config_path = Path(__file__).parent / "config" / "run_config.yml"
        if run_config_path.exists():
            _, raw_config = load_yaml_config(run_config_path)
            default_paths = raw_config.get("model_loading", {}).get("default_paths", {})
            if isinstance(default_paths, dict) and model_type in default_paths:
                candidate = Path(default_paths[model_type])
                if candidate.exists():
                    resolved_model_path = candidate
                    console.print(
                        f"Model path from run_config.yml: {resolved_model_path}"
                    )

    if resolved_model_path is None:
        resolved_model_path = auto_detect_model_path()
        if resolved_model_path:
            console.print(f"Auto-detected model path: {resolved_model_path}")

    if resolved_model_path is None:
        raise FileNotFoundError(
            f"No model path found for '{model_type}'. "
            "Use --model-path or add to config/run_config.yml model_loading.default_paths."
        )

    # Build PipelineConfig
    config_kwargs: dict[str, Any] = {
        "data_dir": data_dir,
        "output_dir": data_dir / "output",
        "model_type": model_type,
        "model_path": resolved_model_path,
    }

    # Set model-specific defaults
    if model_type == "internvl3":
        config_kwargs.setdefault("flash_attn", True)
        config_kwargs.setdefault("device_map", "auto")
    elif model_type in ("internvl3-14b", "internvl3-38b"):
        config_kwargs.setdefault("flash_attn", True)
        config_kwargs.setdefault("device_map", "auto")
    elif model_type == "llama4scout":
        config_kwargs.setdefault("flash_attn", False)
        config_kwargs.setdefault("device_map", "auto")
    elif model_type in ("internvl3-vllm", "internvl3-14b-vllm", "internvl3-38b-vllm"):
        config_kwargs.setdefault("flash_attn", False)
        config_kwargs.setdefault("device_map", "auto")
    elif model_type == "nemotron":
        config_kwargs.setdefault("flash_attn", False)
        config_kwargs.setdefault("device_map", "auto")
    elif model_type == "qwen35":
        config_kwargs.setdefault("flash_attn", False)
        config_kwargs.setdefault("device_map", "auto")

    cfg = PipelineConfig(**config_kwargs)

    # Load SROIE prompt
    prompt = _load_sroie_prompt()
    console.print(f"Prompt: {len(prompt)} chars")

    # Run inference
    image_results: list[SROIEImageResult] = []
    start_time = time.time()

    with registration.loader(cfg) as (model, tokenizer_or_processor):
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]{model_type}", total=len(all_images))

            for img_path in all_images:
                image_id = img_path.stem
                gt = ground_truth[image_id]

                try:
                    image = Image.open(img_path).convert("RGB")
                    raw_response = run_inference(
                        model_type,
                        model,
                        tokenizer_or_processor,
                        image,
                        prompt,
                        max_tokens,
                    )
                    predicted = parse_sroie_response(raw_response)
                    del image
                except Exception as e:
                    console.print(f"[red]Error on {image_id}: {e}[/red]")
                    predicted = {}
                    raw_response = f"Error: {e}"

                image_results.append(
                    SROIEImageResult(
                        image_id=image_id,
                        ground_truth=gt,
                        predicted=predicted,
                    )
                )
                progress.advance(task)

    elapsed = time.time() - start_time
    return compute_sroie_metrics(image_results, model_type, elapsed)


# ============================================================================
# Display and export
# ============================================================================


def print_results_table(results: list[SROIEBenchmarkResult]) -> None:
    """Print a Rich comparison table of benchmark results."""
    # Per-field table
    table = Table(
        title="SROIE Benchmark Results (Per-Field F1)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Field", style="white")
    for r in results:
        table.add_column(r.model_name, justify="right")

    for field in SROIE_FIELDS:
        row = [field]
        for r in results:
            fr = r.field_results[field]
            f1_pct = fr.f1 * 100
            color = "green" if f1_pct >= 80 else ("yellow" if f1_pct >= 50 else "red")
            row.append(f"[{color}]{f1_pct:.1f}%[/{color}]")
        table.add_row(*row)

    # Overall row
    row = ["[bold]Overall F1[/bold]"]
    for r in results:
        f1_pct = r.overall_f1 * 100
        color = "green" if f1_pct >= 80 else ("yellow" if f1_pct >= 50 else "red")
        row.append(f"[bold {color}]{f1_pct:.1f}%[/bold {color}]")
    table.add_row(*row)

    console.print(table)

    # Summary table
    summary = Table(
        title="Summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary.add_column("Metric", style="white")
    for r in results:
        summary.add_column(r.model_name, justify="right")

    summary.add_row("Images", *[str(r.total_images) for r in results])
    summary.add_row("Time (s)", *[f"{r.elapsed_seconds:.1f}" for r in results])
    summary.add_row(
        "Img/min",
        *[
            f"{r.total_images / r.elapsed_seconds * 60:.1f}"
            if r.elapsed_seconds > 0
            else "N/A"
            for r in results
        ],
    )

    console.print(summary)


def save_results_json(results: list[SROIEBenchmarkResult], output_dir: Path) -> Path:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {}
    for r in results:
        per_field: dict[str, Any] = {}
        per_image: list[dict[str, Any]] = []
        model_data: dict[str, Any] = {
            "overall_f1": round(r.overall_f1, 4),
            "total_images": r.total_images,
            "elapsed_seconds": round(r.elapsed_seconds, 2),
            "per_field": per_field,
            "per_image": per_image,
        }
        for field in SROIE_FIELDS:
            fr = r.field_results[field]
            per_field[field] = {
                "precision": round(fr.precision, 4),
                "recall": round(fr.recall, 4),
                "f1": round(fr.f1, 4),
                "tp": fr.true_positives,
                "fp": fr.false_positives,
                "fn": fr.false_negatives,
            }
        for img in r.image_results:
            per_image.append(
                {
                    "image_id": img.image_id,
                    "matches": img.matches,
                    "predicted": img.predicted,
                    "ground_truth": img.ground_truth,
                }
            )
        output[r.model_name] = model_data

    out_path = output_dir / "sroie_benchmark_results.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    console.print(f"Results saved to {out_path}")
    return out_path


def save_results_csv(results: list[SROIEBenchmarkResult], output_dir: Path) -> Path:
    """Save benchmark summary and per-image results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Summary CSV (one row per model) ---
    summary_path = output_dir / "sroie_benchmark_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "model",
            "overall_f1",
            "total_images",
            "elapsed_seconds",
            "images_per_min",
        ]
        for field in SROIE_FIELDS:
            header.extend([f"{field}_precision", f"{field}_recall", f"{field}_f1"])
        writer.writerow(header)

        for r in results:
            img_per_min = (
                r.total_images / r.elapsed_seconds * 60
                if r.elapsed_seconds > 0
                else 0.0
            )
            row: list[Any] = [
                r.model_name,
                round(r.overall_f1, 4),
                r.total_images,
                round(r.elapsed_seconds, 2),
                round(img_per_min, 2),
            ]
            for field in SROIE_FIELDS:
                fr = r.field_results[field]
                row.extend(
                    [round(fr.precision, 4), round(fr.recall, 4), round(fr.f1, 4)]
                )
            writer.writerow(row)

    console.print(f"Summary CSV saved to {summary_path}")

    # --- Per-image CSV (one row per model+image) ---
    detail_path = output_dir / "sroie_benchmark_per_image.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "image_id"]
        for field in SROIE_FIELDS:
            header.extend([f"{field}_gt", f"{field}_pred", f"{field}_match"])
        writer.writerow(header)

        for r in results:
            for img in r.image_results:
                row = [r.model_name, img.image_id]
                for field in SROIE_FIELDS:
                    row.extend(
                        [
                            img.ground_truth.get(field, ""),
                            img.predicted.get(field, ""),
                            img.matches.get(field, False),
                        ]
                    )
                writer.writerow(row)

    console.print(f"Per-image CSV saved to {detail_path}")
    return summary_path


# ============================================================================
# CLI
# ============================================================================


@app.command()
def main(
    model: list[str] = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model type(s) to benchmark (e.g. internvl3, llama4scout).",
    ),
    data_dir: Path = typer.Option(
        Path("../data/sroie"),
        "--data-dir",
        "-d",
        help="Path to SROIE data directory (contains img/ and key/).",
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        "-p",
        help="Override model path (auto-detected if omitted).",
    ),
    max_images: int | None = typer.Option(
        None,
        "--max-images",
        "-n",
        help="Maximum images to evaluate (all if omitted).",
    ),
    output_dir: Path = typer.Option(
        Path("output/sroie"),
        "--output-dir",
        "-o",
        help="Directory for JSON results output.",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        help="Maximum generation tokens per image.",
    ),
    quantization: str | None = typer.Option(
        None,
        "--quantization",
        "-q",
        help="Quantization strategy override (fp8, nf4, none).",
    ),
) -> None:
    """Run SROIE benchmark on one or more vision-language models."""
    console.print(
        f"[bold]SROIE Benchmark[/bold] | Models: {', '.join(model)} | Data: {data_dir}"
    )

    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1) from None

    all_results: list[SROIEBenchmarkResult] = []

    for model_type in model:
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold cyan]Benchmarking: {model_type}[/bold cyan]")
        console.print(f"{'=' * 60}")

        result = run_benchmark(
            model_type=model_type,
            data_dir=data_dir,
            model_path=model_path,
            max_images=max_images,
            max_tokens=max_tokens,
            quantization=quantization,
        )
        all_results.append(result)

    # Print comparison table
    console.print()
    print_results_table(all_results)

    # Save results
    save_results_json(all_results, output_dir)
    save_results_csv(all_results, output_dir)


if __name__ == "__main__":
    app()
