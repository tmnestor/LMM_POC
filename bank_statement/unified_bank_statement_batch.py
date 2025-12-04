#!/usr/bin/env python
# ruff: noqa: E402
"""
Unified Bank Statement Batch Extraction and Evaluation

This script runs batch extraction on bank statement images using the
UnifiedBankExtractor with configurable model selection via YAML config.

Features:
- Batch processing of all bank statement images in a directory
- Model selection via config/model_config.yaml
- Full report generation (CSV, JSON, Markdown)
- Ground truth evaluation with F1 scores

Usage:
    python unified_bank_statement_batch.py [OPTIONS]

Options:
    --model MODEL_KEY   Model from config/model_config.yaml (default: internvl3_5_8b)
    --data-dir PATH     Directory containing images (default: evaluation_data/bank/minimal)
    --max-images N      Limit to N images (default: all)
    --verbose           Enable verbose output
    --dry-run           Show what would be processed without running
    --log-file PATH     Save output to log file
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for common module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change working directory to project root so configs are found
os.chdir(project_root)

import argparse
import json as json_module
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from rich.console import Console
from rich.table import Table

from common.evaluation_metrics import (
    calculate_field_accuracy_f1,
    load_ground_truth,
)
from common.reproducibility import set_seed
from common.unified_bank_extractor import UnifiedBankExtractor

# Rich console for styled output
console = Console()
file_console = None


def log_print(msg: str = "", style: str | None = None):
    """Print to both terminal and log file if configured."""
    console.print(msg, style=style)
    if file_console:
        file_console.print(msg, style=style)


def log_rule(title: str):
    """Print rule to both terminal and log file if configured."""
    console.rule(title)
    if file_console:
        file_console.rule(title)


def log_table(table):
    """Print table to both terminal and log file if configured."""
    console.print(table)
    if file_console:
        file_console.print(table)


# Bank statement fields to evaluate
BANK_STATEMENT_FIELDS = [
    "DOCUMENT_TYPE",
    "STATEMENT_DATE_RANGE",
    "TRANSACTION_DATES",
    "LINE_ITEM_DESCRIPTIONS",
    "TRANSACTION_AMOUNTS_PAID",
]


# ============================================================================
# SEMANTIC NORMALIZATION (for evaluation comparison)
# ============================================================================
def normalize_date(date_str: str) -> str:
    """Normalize date string to canonical format YYYY-MM-DD for semantic comparison."""
    from dateutil import parser as date_parser

    if not date_str or pd.isna(date_str):
        return ""

    date_str = str(date_str).strip()
    if not date_str:
        return ""

    try:
        parsed = date_parser.parse(date_str, dayfirst=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return date_str


def normalize_amount(amount_str: str) -> str:
    """Normalize amount string for semantic comparison.

    Handles: "$48.50" → "48.5", "2,000.00" → "2000", "-78.90" → "78.9"
    """
    import re

    if not amount_str or pd.isna(amount_str):
        return ""

    amount_str = str(amount_str).strip()
    if not amount_str:
        return ""

    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[$£€¥₹\s]", "", amount_str)
    # Remove thousand separators
    cleaned = cleaned.replace(",", "")

    try:
        value = abs(float(cleaned))
        # Format consistently, removing trailing zeros
        return f"{value:.2f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def normalize_pipe_delimited(value: str, normalizer_fn) -> str:
    """Apply normalizer function to each item in a pipe-delimited string."""
    if not value or pd.isna(value):
        return ""

    value = str(value).strip()
    if not value:
        return ""

    items = [item.strip() for item in value.split("|")]
    normalized = [normalizer_fn(item) for item in items]
    return " | ".join(normalized)


def normalize_field_for_comparison(field_name: str, value: str) -> str:
    """Normalize a field value based on its type for semantic comparison."""
    if not value or pd.isna(value):
        return ""

    value = str(value).strip()

    if field_name == "TRANSACTION_DATES":
        return normalize_pipe_delimited(value, normalize_date)
    elif field_name == "TRANSACTION_AMOUNTS_PAID":
        return normalize_pipe_delimited(value, normalize_amount)
    elif field_name == "STATEMENT_DATE_RANGE":
        # Handle "18 Mar 2024 - 14 Apr 2024" format
        if " - " in value:
            parts = value.split(" - ")
            if len(parts) == 2:
                start = normalize_date(parts[0].strip())
                end = normalize_date(parts[1].strip())
                return f"{start} - {end}"
        return value
    else:
        # For other fields (DOCUMENT_TYPE, LINE_ITEM_DESCRIPTIONS), return as-is
        return value




def load_model_from_config(model_key: str, verbose: bool = False):
    """Load model based on config/model_config.yaml settings.

    Returns:
        Tuple of (model, tokenizer, processor, model_type, model_dtype, image_processing_config)
    """
    config_path = project_root / "config" / "model_config.yaml"
    with config_path.open() as f:
        models_config = yaml.safe_load(f)

    if model_key not in models_config["models"]:
        available = list(models_config["models"].keys())
        raise ValueError(f"Model '{model_key}' not found. Available: {available}")

    model_config = models_config["models"][model_key]
    loading_config = model_config.get("loading", {})
    image_processing_config = model_config.get("image_processing", {})

    model_path = model_config["default_path"]
    model_type = model_config["type"]
    quantization = loading_config.get("quantization", "none")
    torch_dtype_str = loading_config.get("torch_dtype", "bfloat16")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    if verbose:
        log_print(f"[cyan]Loading model:[/cyan] {model_config['name']}")
        log_print(f"  Path: {model_path}")
        log_print(f"  Type: {model_type}")
        log_print(f"  Quantization: {quantization}")
        log_print(f"  Dtype: {torch_dtype_str}")

    if model_type == "llama":
        from common.llama_model_loader_robust import load_llama_model_robust

        gen_config = model_config.get("generation", {})

        model, processor = load_llama_model_robust(
            model_path=model_path,
            use_quantization=(quantization == "8bit"),
            device_map=loading_config.get("device_map", "auto"),
            max_new_tokens=gen_config.get("max_new_tokens", 4096),
            torch_dtype=torch_dtype_str,
            low_cpu_mem_usage=loading_config.get("low_cpu_mem_usage", True),
            verbose=verbose,
        )

        if loading_config.get("tie_weights", True):
            try:
                model.tie_weights()
            except Exception:
                pass

        tokenizer = None
        model_dtype = torch_dtype

    else:
        # InternVL3 models
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        def split_model(model_path):
            """Official InternVL3 multi-GPU device mapping."""
            device_map = {}
            world_size = torch.cuda.device_count()
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            num_layers = config.llm_config.num_hidden_layers

            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for _ in range(num_layer):
                    device_map[f"language_model.model.layers.{layer_cnt}"] = i
                    layer_cnt += 1

            device_map["vision_model"] = 0
            device_map["mlp1"] = 0
            device_map["language_model.model.tok_embeddings"] = 0
            device_map["language_model.model.embed_tokens"] = 0
            device_map["language_model.output"] = 0
            device_map["language_model.model.norm"] = 0
            device_map["language_model.model.rotary_emb"] = 0
            device_map["language_model.lm_head"] = 0
            device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

            return device_map

        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=loading_config.get("low_cpu_mem_usage", True),
                use_flash_attn=loading_config.get("use_flash_attn", False),
                trust_remote_code=loading_config.get("trust_remote_code", True),
                quantization_config=quantization_config,
                device_map={"": 0},
            ).eval()
        else:
            world_size = torch.cuda.device_count()
            if world_size > 1:
                device_map = split_model(model_path)
            else:
                device_map = {"": 0}

            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=loading_config.get("low_cpu_mem_usage", True),
                use_flash_attn=loading_config.get("use_flash_attn", False),
                trust_remote_code=loading_config.get("trust_remote_code", True),
                device_map=device_map,
            ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        processor = None
        model_dtype = torch_dtype

    if verbose:
        log_print("[green]Model loaded successfully![/green]")

    return model, tokenizer, processor, model_type, model_dtype, image_processing_config, model_config["name"]


def evaluate_extraction(schema_fields: dict, image_name: str, ground_truth_map: dict, method: str = "order_aware_f1") -> dict:
    """Evaluate extracted schema fields against ground truth with semantic normalization."""
    gt_data = ground_truth_map.get(image_name, {})

    if not gt_data:
        return {"error": "No ground truth found", "image_name": image_name}

    field_scores = {}
    total_f1 = 0.0

    for field in BANK_STATEMENT_FIELDS:
        extracted_value = schema_fields.get(field, "NOT_FOUND")
        gt_value = gt_data.get(field, "NOT_FOUND")

        if pd.isna(gt_value):
            gt_value = "NOT_FOUND"

        # Normalize values for semantic comparison (dates, amounts)
        normalized_extracted = normalize_field_for_comparison(field, extracted_value)
        normalized_gt = normalize_field_for_comparison(field, gt_value)

        result = calculate_field_accuracy_f1(normalized_extracted, normalized_gt, field)

        if result:
            field_scores[field] = {
                "f1_score": result.get("f1_score", 0.0),
                "precision": result.get("precision", 0.0),
                "recall": result.get("recall", 0.0),
                "extracted": str(extracted_value)[:100],
                "ground_truth": str(gt_value)[:100],
            }
            total_f1 += result.get("f1_score", 0.0)

    overall_accuracy = total_f1 / len(BANK_STATEMENT_FIELDS) if BANK_STATEMENT_FIELDS else 0.0

    return {
        "image_name": image_name,
        "method": method,
        "overall_accuracy": overall_accuracy,
        "field_scores": field_scores,
    }


def display_field_comparison(schema_fields: dict, ground_truth_map: dict, image_name: str, eval_result: dict):
    """Display comparison of extracted vs ground truth fields."""
    gt_data = ground_truth_map.get(image_name, {})
    field_scores = eval_result.get("field_scores", {})

    table = Table(title="Field Comparison", show_header=True)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Field", style="cyan")
    table.add_column("F1", justify="right", width=8)
    table.add_column("Extracted", overflow="fold", max_width=40)
    table.add_column("Ground Truth", overflow="fold", max_width=40)

    for field in BANK_STATEMENT_FIELDS:
        extracted_val = schema_fields.get(field, "NOT_FOUND")
        ground_val = gt_data.get(field, "NOT_FOUND")

        if pd.isna(ground_val):
            ground_val = "NOT_FOUND"

        if isinstance(field_scores.get(field), dict):
            f1_score = field_scores[field].get("f1_score", 0.0)
        else:
            f1_score = field_scores.get(field, 0.0)

        if f1_score == 1.0:
            status = "[green]OK[/green]"
        elif f1_score >= 0.5:
            status = "[yellow]PART[/yellow]"
        else:
            status = "[red]FAIL[/red]"

        # Truncate long values
        extracted_str = str(extracted_val)[:80] + "..." if len(str(extracted_val)) > 80 else str(extracted_val)
        ground_str = str(ground_val)[:80] + "..." if len(str(ground_val)) > 80 else str(ground_val)

        table.add_row(status, field, f"{f1_score:.1%}", extracted_str, ground_str)

    log_table(table)


def generate_reports(batch_results: list, output_dir: Path, model_key: str, model_name: str, batch_timestamp: str):
    """Generate CSV, JSON, and Markdown reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in batch_results if "error" not in r]

    # CSV Report - match existing format for compatibility
    if successful:
        csv_data = []
        for result in successful:
            row = {
                "image_file": result["image_name"],
                "document_type": result["extracted_fields"].get("DOCUMENT_TYPE", "BANK_STATEMENT"),
                "overall_accuracy": result["evaluation"]["overall_accuracy"],
                "processing_time": result["processing_time"],
                "date_format": result["metadata"].get("date_format", ""),
                "total_rows": result["metadata"].get("total_rows", 0),
                "debit_rows": result["metadata"].get("debit_rows", 0),
            }

            # Add extracted fields
            for field, value in result["extracted_fields"].items():
                row[field] = value

            # Add field-level scores
            field_scores = result["evaluation"].get("field_scores", {})
            for field, scores in field_scores.items():
                if isinstance(scores, dict):
                    row[f"{field}_f1"] = scores.get("f1_score", 0.0)
                else:
                    row[f"{field}_f1"] = scores

            csv_data.append(row)

        results_df = pd.DataFrame(csv_data)
        csv_path = output_dir / f"unified_bank_batch_{model_key}_{batch_timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        log_print(f"[green]CSV saved:[/green] {csv_path}")

    # JSON Report - match existing format for compatibility
    processing_times = [r["processing_time"] for r in successful]

    json_report = {
        "metadata": {
            "batch_id": batch_timestamp,
            "model": model_name,  # Full model name for compatibility
            "evaluation_method": "order_aware_f1",
            "total_images": len(batch_results),
            "successful": len(successful),
            "failed": len(batch_results) - len(successful),
        },
        "summary": {
            "avg_accuracy": float(np.mean([r["evaluation"]["overall_accuracy"] for r in successful]))
            if successful
            else 0.0,
            "min_accuracy": float(min([r["evaluation"]["overall_accuracy"] for r in successful]))
            if successful
            else 0.0,
            "max_accuracy": float(max([r["evaluation"]["overall_accuracy"] for r in successful]))
            if successful
            else 0.0,
            "avg_processing_time": float(np.mean(processing_times)) if processing_times else 0.0,
        },
        "results": batch_results,
    }

    json_path = output_dir / f"unified_bank_evaluation_{model_key}_{batch_timestamp}.json"
    with json_path.open("w") as f:
        json_module.dump(json_report, f, indent=2, default=str)
    log_print(f"[green]JSON saved:[/green] {json_path}")

    # Markdown Report
    if successful:
        md_lines = [
            "# Unified Bank Statement Batch Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Batch ID:** {batch_timestamp}",
            f"**Model:** {model_name}",
            "**Evaluation Method:** order_aware_f1",
            "",
            "## Executive Summary",
            "",
            f"- **Total Images:** {len(batch_results)}",
            f"- **Successful:** {len(successful)} ({len(successful) / len(batch_results) * 100:.1f}%)",
            f"- **Failed:** {len(batch_results) - len(successful)}",
            "",
        ]

        avg_acc = np.mean([r["evaluation"]["overall_accuracy"] for r in successful])
        min_acc = min([r["evaluation"]["overall_accuracy"] for r in successful])
        max_acc = max([r["evaluation"]["overall_accuracy"] for r in successful])

        md_lines.extend([
            f"- **Average Accuracy:** {avg_acc:.1%}",
            f"- **Min Accuracy:** {min_acc:.1%}",
            f"- **Max Accuracy:** {max_acc:.1%}",
            f"- **Avg Processing Time:** {np.mean(processing_times):.2f}s",
            "",
            "## Per-Image Results",
            "",
            "| Image | Accuracy | Date Format | Rows | Time |",
            "|-------|----------|-------------|------|------|",
        ])

        for r in successful:
            acc = r["evaluation"]["overall_accuracy"]
            date_fmt = r["metadata"].get("date_format", "N/A")
            rows = r["metadata"].get("debit_rows", 0)
            time_s = r["processing_time"]
            md_lines.append(f"| {r['image_name']} | {acc:.1%} | {date_fmt} | {rows} | {time_s:.1f}s |")

        # Field-level summary
        md_lines.extend([
            "",
            "## Field-Level Accuracy",
            "",
            "| Field | Avg F1 |",
            "|-------|--------|",
        ])

        for field in BANK_STATEMENT_FIELDS:
            field_f1s = []
            for r in successful:
                scores = r["evaluation"].get("field_scores", {}).get(field, {})
                if isinstance(scores, dict):
                    field_f1s.append(scores.get("f1_score", 0.0))
            avg_field_f1 = np.mean(field_f1s) if field_f1s else 0.0
            md_lines.append(f"| {field} | {avg_field_f1:.1%} |")

        md_path = output_dir / f"unified_bank_summary_{model_key}_{batch_timestamp}.md"
        with md_path.open("w") as f:
            f.write("\n".join(md_lines))
        log_print(f"[green]Markdown saved:[/green] {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch extraction and evaluation of bank statements with UnifiedBankExtractor"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="internvl3_5_8b",
        help="Model key from config/model_config.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="evaluation_data/bank/minimal",
        help="Directory containing bank statement images",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="evaluation_data/bank/ground_truth_bank.csv",
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output reports",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit to N images",
    )
    parser.add_argument(
        "--balance-correction",
        action="store_true",
        help="Enable balance correction (requires chronological order)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )

    args = parser.parse_args()

    # Setup file logging if requested
    global file_console
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_console = Console(file=log_path.open("w"), force_terminal=True, width=200)
        log_print(f"[dim]Logging to: {log_path}[/dim]")

    set_seed(42)
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_rule("[bold blue]UNIFIED BANK STATEMENT BATCH EXTRACTION")
    log_print(f"[cyan]Model:[/cyan] {args.model}")
    log_print(f"[cyan]Data directory:[/cyan] {args.data_dir}")
    log_print(f"[cyan]Balance correction:[/cyan] {'enabled' if args.balance_correction else 'disabled'}")

    # Load ground truth
    log_print("\n[yellow]Loading ground truth...[/yellow]")
    ground_truth_map = load_ground_truth(args.ground_truth, verbose=args.verbose)

    # Discover bank statement images
    data_dir = Path(args.data_dir)
    bank_images = []

    for img_name, gt_data in ground_truth_map.items():
        doc_type = str(gt_data.get("DOCUMENT_TYPE", "")).upper()
        if doc_type == "BANK_STATEMENT":
            img_path = data_dir / img_name
            if img_path.exists():
                bank_images.append(str(img_path))
            else:
                log_print(f"  [yellow]Warning: Image not found: {img_path}[/yellow]")

    if args.max_images:
        bank_images = bank_images[: args.max_images]

    log_print(f"\n[green]Found {len(bank_images)} bank statement images[/green]")

    if args.dry_run:
        log_print("\n[yellow]DRY RUN - Images that would be processed:[/yellow]")
        for img in bank_images:
            log_print(f"  - {Path(img).name}")
        return

    # Load model
    log_print("\n[yellow]Loading model...[/yellow]")
    model, tokenizer, processor, model_type, model_dtype, image_processing_config, model_name = load_model_from_config(
        args.model, verbose=args.verbose
    )

    # Create extractor
    extractor = UnifiedBankExtractor(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        model_dtype=model_dtype,
        image_processing_config=image_processing_config,
        use_balance_correction=args.balance_correction,
    )

    log_print(f"[green]Extractor created:[/green] max_tiles={extractor.max_tiles}, input_size={extractor.input_size}")

    # Process images
    log_rule("[bold blue]BATCH PROCESSING")

    batch_results = []

    for idx, image_path in enumerate(bank_images, 1):
        image_name = Path(image_path).name
        log_print(f"\n[bold cyan][{idx}/{len(bank_images)}][/bold cyan] Processing: [white]{image_name}[/white]")

        # Aggressive GPU memory cleanup before each image
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start_time = time.time()

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Extract using UnifiedBankExtractor
            result = extractor.extract(image)
            schema_fields = result.to_schema_dict()

            # Cleanup after extraction
            del image
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            processing_time = time.time() - start_time

            # Evaluate
            eval_result = evaluate_extraction(schema_fields, image_name, ground_truth_map)

            # Build result - metadata format matches existing batch scripts for compatibility
            batch_result = {
                "image_name": image_name,
                "image_path": image_path,
                "extracted_fields": schema_fields,
                "metadata": {
                    "headers_detected": result.headers_detected,
                    "date_format": result.strategy_used,  # e.g., "balance_description_2turn"
                    "corrections_made": result.correction_stats.corrections_made if result.correction_stats else 0,
                    "total_rows": len(result.transaction_dates) + (2 if result.correction_stats else 0),  # Approximate
                    "debit_rows": len(result.transaction_dates),
                    "turn0_raw_response": result.raw_responses.get("turn0", ""),
                    "turn1_raw_response": result.raw_responses.get("turn1", ""),
                },
                "evaluation": eval_result,
                "processing_time": processing_time,
            }

            batch_results.append(batch_result)

            accuracy = eval_result.get("overall_accuracy", 0.0)
            acc_color = "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.5 else "red"
            log_print(f"  [{acc_color}]Accuracy: {accuracy:.1%}[/{acc_color}]  Time: {processing_time:.2f}s")

            if args.verbose:
                display_field_comparison(schema_fields, ground_truth_map, image_name, eval_result)

        except Exception as e:
            log_print(f"  [red]ERROR: {e}[/red]")
            batch_results.append({
                "image_name": image_name,
                "image_path": image_path,
                "error": str(e),
                "processing_time": time.time() - start_time,
            })

    # Summary
    log_rule("[bold blue]BATCH SUMMARY")

    successful = [r for r in batch_results if "error" not in r]
    failed_count = len(batch_results) - len(successful)

    log_print(f"[cyan]Total:[/cyan] {len(batch_results)} images")
    log_print(f"[green]Successful:[/green] {len(successful)}")
    if failed_count > 0:
        log_print(f"[red]Failed:[/red] {failed_count}")

    if successful:
        accuracies = [r["evaluation"]["overall_accuracy"] for r in successful]
        avg_acc = np.mean(accuracies)
        log_print("\n[bold]Accuracy Statistics:[/bold]")
        log_print(f"  Average: [{'green' if avg_acc >= 0.8 else 'yellow'}]{avg_acc:.1%}[/]")
        log_print(f"  Min: {min(accuracies):.1%}")
        log_print(f"  Max: {max(accuracies):.1%}")

        # Field-level summary
        log_print("\n[bold]Field-Level Accuracy:[/bold]")
        for field in BANK_STATEMENT_FIELDS:
            field_f1s = []
            for r in successful:
                scores = r["evaluation"].get("field_scores", {}).get(field, {})
                if isinstance(scores, dict):
                    field_f1s.append(scores.get("f1_score", 0.0))
            avg_field_f1 = np.mean(field_f1s) if field_f1s else 0.0
            status = "green" if avg_field_f1 >= 0.8 else "yellow" if avg_field_f1 >= 0.5 else "red"
            log_print(f"  [{status}]{field}: {avg_field_f1:.1%}[/{status}]")

    # Generate reports
    log_rule("[bold blue]GENERATING REPORTS")
    generate_reports(batch_results, Path(args.output_dir), args.model, model_name, batch_timestamp)

    log_print("\n[bold green]Batch processing complete![/bold green]")


if __name__ == "__main__":
    main()
