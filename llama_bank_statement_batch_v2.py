#!/usr/bin/env python
"""
Llama Bank Statement Batch Extraction and Evaluation - V2

This script uses a balance-description extraction approach that works
for both date-per-row and date-grouped bank statements.

Key changes from v1:
- Removed Turn 0.5 (date format classification)
- Turn 0: Header detection (determines extraction method)
- If balance column detected: Use balance-description prompt
- If no balance column: Use table extraction prompt (4-column tables)

Usage:
    python llama_bank_statement_batch_v2.py [OPTIONS]

Options:
    --max-images N      Limit to N images (default: all)
    --method METHOD     Evaluation method (default: order_aware_f1)
    --verbose           Enable verbose output
    --dry-run           Show what would be processed without running
"""

import argparse
import json as json_module
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dateutil import parser as date_parser
from PIL import Image
from rich.console import Console
from rich.table import Table

from common.evaluation_metrics import (
    calculate_correlation_aware_f1,
    calculate_field_accuracy_f1,
    calculate_field_accuracy_f1_position_agnostic,
    calculate_field_accuracy_kieval,
    load_ground_truth,
)
from common.llama_model_loader_robust import load_llama_model_robust
from common.reproducibility import set_seed

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


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "DATA_DIR": Path(
        "/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data/bank/date_grouped"
    ),
    "GROUND_TRUTH": Path(
        "/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data/bank/ground_truth_bank.csv"
    ),
    "OUTPUT_BASE": Path("/home/jovyan/nfs_share/tod/LMM_POC/output"),
    "MODEL_PATH": "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct",
    "DOCUMENT_TYPE_FILTER": "BANK_STATEMENT",
    "MAX_IMAGES": None,
    "EVALUATION_METHOD": "order_aware_f1",
    "VERBOSE": True,
    "GENERATE_CSV": True,
    "GENERATE_JSON": True,
    "GENERATE_MARKDOWN": True,
}

BATCH_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

BANK_STATEMENT_FIELDS = [
    "DOCUMENT_TYPE",
    "STATEMENT_DATE_RANGE",
    "TRANSACTION_DATES",
    "LINE_ITEM_DESCRIPTIONS",
    "TRANSACTION_AMOUNTS_PAID",
]

# ============================================================================
# SEMANTIC NORMALIZATION
# ============================================================================


def normalize_date(date_str):
    """Normalize date string to canonical format YYYY-MM-DD."""
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


def normalize_amount(amount_str):
    """Normalize amount string for semantic comparison."""
    if not amount_str or pd.isna(amount_str):
        return ""
    amount_str = str(amount_str).strip()
    if not amount_str:
        return ""
    cleaned = re.sub(r"[$£€¥₹\s]", "", amount_str)
    cleaned = cleaned.replace(",", "")
    try:
        value = abs(float(cleaned))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def normalize_pipe_delimited(value, normalizer_fn):
    """Apply normalizer function to each item in a pipe-delimited string."""
    if not value or pd.isna(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""
    items = [item.strip() for item in value.split("|")]
    normalized = [normalizer_fn(item) for item in items]
    return " | ".join(normalized)


def normalize_field_for_comparison(field_name, value):
    """Normalize a field value based on its type for semantic comparison."""
    if not value or pd.isna(value):
        return ""
    value = str(value).strip()
    if field_name == "TRANSACTION_DATES":
        return normalize_pipe_delimited(value, normalize_date)
    elif field_name == "TRANSACTION_AMOUNTS_PAID":
        return normalize_pipe_delimited(value, normalize_amount)
    elif field_name == "STATEMENT_DATE_RANGE":
        if " - " in value:
            parts = value.split(" - ")
            if len(parts) == 2:
                start = normalize_date(parts[0].strip())
                end = normalize_date(parts[1].strip())
                return f"{start} - {end}"
        return value
    else:
        return value


# ============================================================================
# PATTERN MATCHING
# ============================================================================
DATE_PATTERNS = ["date", "day", "transaction date", "trans date"]
DESCRIPTION_PATTERNS = [
    "description",
    "details",
    "transaction details",
    "trans details",
    "particulars",
    "narrative",
    "transaction",
    "trans",
]
DEBIT_PATTERNS = [
    "debit",
    "debits",
    "withdrawal",
    "withdrawals",
    "paid",
    "paid out",
    "spent",
    "dr",
]
CREDIT_PATTERNS = ["credit", "credits", "deposit", "deposits", "received", "cr"]
BALANCE_PATTERNS = ["balance", "bal", "running balance"]
AMOUNT_PATTERNS = ["amount", "amt", "value", "total"]


def match_header(headers, patterns, fallback=None):
    """Match a header using pattern keywords."""
    headers_lower = [h.lower() for h in headers]
    for pattern in patterns:
        for i, header_lower in enumerate(headers_lower):
            if pattern == header_lower:
                return headers[i]
    for pattern in patterns:
        if len(pattern) > 2:
            for i, header_lower in enumerate(headers_lower):
                if pattern in header_lower:
                    return headers[i]
    return fallback


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================
def parse_headers_from_response(response_text):
    """Parse column headers from Turn 0 response."""
    header_lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    identified_headers = []
    for line in header_lines:
        cleaned = line.lstrip("0123456789.-•* ").strip()
        cleaned = cleaned.replace("**", "").replace("__", "")
        if cleaned.endswith(":"):
            continue
        if len(cleaned) > 40:
            continue
        if cleaned and len(cleaned) > 2:
            identified_headers.append(cleaned)
    return identified_headers


def build_date_per_row_example(headers):
    """Build example for date-per-row format (4-column tables)."""
    rows = []
    for date, desc, deb, cred in [
        ("15 Jan", "ATM Withdrawal", "200.00", ""),
        ("16 Jan", "Salary Payment", "", "3,500.00"),
        ("17 Jan", "Online Purchase", "150.00", ""),
    ]:
        row = []
        for h in headers:
            hl = h.lower()
            if hl in ["date", "day"]:
                row.append(date)
            elif any(p in hl for p in ["desc", "particular", "detail", "transaction"]):
                row.append(desc)
            elif any(p in hl for p in ["debit", "withdrawal"]):
                row.append(deb)
            elif any(p in hl for p in ["credit", "deposit"]):
                row.append(cred)
            elif "amount" in hl:
                row.append(deb if deb else cred)
            else:
                row.append("")
        rows.append("| " + " | ".join(row) + " |")
    return rows


def parse_markdown_table(markdown_text):
    """Parse markdown table into list of dictionaries."""
    lines = [line.strip() for line in markdown_text.strip().split("\n") if line.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if "|" in line:
            cleaned = line.replace("|", "").replace("-", "").replace(" ", "")
            if cleaned:
                header_idx = i
                break
    if header_idx is None:
        return []
    header_line = lines[header_idx]
    header_parts = [h.strip() for h in header_line.split("|")]
    if header_parts and header_parts[0] == "":
        header_parts = header_parts[1:]
    if header_parts and header_parts[-1] == "":
        header_parts = header_parts[:-1]
    headers = [h for h in header_parts if h]
    rows = []
    for line in lines[header_idx + 1 :]:
        if "|" not in line:
            continue
        cleaned = (
            line.replace("|", "").replace("-", "").replace(" ", "").replace(":", "")
        )
        if not cleaned:
            continue
        value_parts = [v.strip() for v in line.split("|")]
        if value_parts and value_parts[0] == "":
            value_parts = value_parts[1:]
        if value_parts and value_parts[-1] == "":
            value_parts = value_parts[:-1]
        if len(value_parts) == len(headers):
            rows.append(dict(zip(headers, value_parts, strict=False)))
    return rows


def parse_balance_description_response(response_text, date_col, desc_col, debit_col, credit_col, balance_col):
    """
    Parse the hierarchical balance-description response into transaction rows.

    Handles format like:
    1. **Thu 04 Sep 2025**
       - Description: Direct Debit DOMINO'S PTY LTD
       - Debit: $117.57
       - Balance: $8586.28 CR

    Returns list of dicts with standardized column names.
    """
    rows = []
    current_date = None
    current_transaction = {}

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for date header (numbered item with bold date)
        # Pattern: "1. **Thu 04 Sep 2025**" or "1. Thu 04 Sep 2025"
        date_match = re.match(r"^\d+\.\s*\*?\*?([A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?", line)
        if not date_match:
            # Also try without day name: "1. **04 Sep 2025**"
            date_match = re.match(r"^\d+\.\s*\*?\*?(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\*?\*?", line)

        if date_match:
            # Save previous transaction if exists
            if current_transaction and current_date:
                current_transaction[date_col] = current_date
                rows.append(current_transaction)
                current_transaction = {}

            current_date = date_match.group(1).strip()
            continue

        # Check for field lines
        # Pattern: "- Description: ..." or "   - Debit: ..."
        field_match = re.match(r"^\s*-\s*(\w+):\s*(.+)$", line)
        if field_match:
            field_name = field_match.group(1).strip().lower()
            field_value = field_match.group(2).strip()

            # Map field names to column names
            if field_name == "description":
                # If we already have a description, this is a new transaction under same date
                if desc_col in current_transaction and current_transaction[desc_col]:
                    # Save current transaction
                    if current_date:
                        current_transaction[date_col] = current_date
                    rows.append(current_transaction)
                    current_transaction = {}
                current_transaction[desc_col] = field_value

            elif field_name == "debit":
                current_transaction[debit_col] = field_value

            elif field_name == "credit":
                current_transaction[credit_col] = field_value

            elif field_name == "balance":
                current_transaction[balance_col] = field_value

            elif field_name == "amount":
                # Generic amount - put in debit by default
                if debit_col not in current_transaction:
                    current_transaction[debit_col] = field_value

    # Don't forget the last transaction
    if current_transaction and current_date:
        current_transaction[date_col] = current_date
        rows.append(current_transaction)

    return rows


def parse_amount(value):
    """Extract numeric value from formatted currency string."""
    if not value or value.strip() == "":
        return 0.0
    cleaned = (
        value.replace("$", "")
        .replace(",", "")
        .replace("CR", "")
        .replace("DR", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def is_non_transaction_row(row, desc_col):
    """Check if this row is NOT an actual transaction."""
    desc = row.get(desc_col, "").strip().upper()
    if "OPENING BALANCE" in desc:
        return True
    if "CLOSING BALANCE" in desc:
        return True
    if "BROUGHT FORWARD" in desc:
        return True
    if "CARRIED FORWARD" in desc:
        return True
    return False


def filter_debit_transactions(rows, debit_col, desc_col=None):
    """Filter rows to only those with actual debit transactions."""
    debit_rows = []
    for row in rows:
        debit_value = row.get(debit_col, "").strip()
        if not debit_value:
            continue
        if desc_col and is_non_transaction_row(row, desc_col):
            continue
        debit_rows.append(row)
    return debit_rows


def extract_schema_fields(debit_rows, date_col, desc_col, debit_col, all_rows=None):
    """Extract fields in universal.yaml schema format."""
    if not debit_rows:
        return {
            "DOCUMENT_TYPE": "BANK_STATEMENT",
            "STATEMENT_DATE_RANGE": "NOT_FOUND",
            "TRANSACTION_DATES": "NOT_FOUND",
            "LINE_ITEM_DESCRIPTIONS": "NOT_FOUND",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND",
        }

    debit_dates = []
    descriptions = []
    amounts = []

    for row in debit_rows:
        date = row.get(date_col, "").strip()
        desc = row.get(desc_col, "").strip()
        amount = row.get(debit_col, "").strip()
        if date:
            debit_dates.append(date)
        if desc:
            descriptions.append(desc)
        if amount:
            amounts.append(amount)

    date_range = "NOT_FOUND"
    rows_for_range = all_rows if all_rows is not None else debit_rows
    all_dates = [row.get(date_col, "").strip() for row in rows_for_range]
    all_dates = [d for d in all_dates if d]
    if all_dates:
        date_range = f"{all_dates[0]} - {all_dates[-1]}"

    return {
        "DOCUMENT_TYPE": "BANK_STATEMENT",
        "STATEMENT_DATE_RANGE": date_range,
        "TRANSACTION_DATES": " | ".join(debit_dates) if debit_dates else "NOT_FOUND",
        "LINE_ITEM_DESCRIPTIONS": " | ".join(descriptions) if descriptions else "NOT_FOUND",
        "TRANSACTION_AMOUNTS_PAID": " | ".join(amounts) if amounts else "NOT_FOUND",
    }


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_field_comparison(schema_fields, ground_truth_map, image_name, eval_result):
    """Display stacked comparison of extracted vs ground truth fields."""
    gt_data = ground_truth_map.get(image_name, {})
    field_scores = eval_result.get("field_scores", {})

    table = Table(title="Field Comparison (semantic matching)", show_header=True)
    table.add_column("Status", style="bold", width=8)
    table.add_column("Field", style="cyan")
    table.add_column("F1", justify="right", width=8)
    table.add_column("Extracted", overflow="fold")
    table.add_column("Ground Truth", overflow="fold")

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
            status = "[yellow]~ PART[/yellow]"
        else:
            status = "[red]FAIL[/red]"

        table.add_row(status, field, f"{f1_score:.1%}", str(extracted_val), str(ground_val))

    log_table(table)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def evaluate_field(extracted_value, gt_value, field_name, method):
    """Route to appropriate evaluation function."""
    if method == "order_aware_f1":
        return calculate_field_accuracy_f1(extracted_value, gt_value, field_name)
    elif method == "position_agnostic_f1":
        return calculate_field_accuracy_f1_position_agnostic(extracted_value, gt_value, field_name)
    elif method == "kieval":
        return calculate_field_accuracy_kieval(extracted_value, gt_value, field_name)
    elif method == "correlation":
        return None
    else:
        raise ValueError(f"Unknown evaluation method: {method}")


def evaluate_extraction(schema_fields, image_name, ground_truth_map, method):
    """Evaluate extracted schema fields against ground truth."""
    gt_data = ground_truth_map.get(image_name, {})
    if not gt_data:
        return {"error": "No ground truth found", "image_name": image_name}

    if method == "correlation":
        normalized_extracted = {
            field: normalize_field_for_comparison(field, schema_fields.get(field, ""))
            for field in BANK_STATEMENT_FIELDS
        }
        normalized_gt = {
            field: normalize_field_for_comparison(field, gt_data.get(field, ""))
            for field in BANK_STATEMENT_FIELDS
        }
        result = calculate_correlation_aware_f1(
            extracted_data=normalized_extracted,
            ground_truth_data=normalized_gt,
            document_type="bank_statement",
            debug=CONFIG["VERBOSE"],
        )
        return {
            "image_name": image_name,
            "method": method,
            "overall_accuracy": result.get("combined_f1", 0.0),
            "standard_f1": result.get("standard_f1", 0.0),
            "alignment_score": result.get("alignment_score", 0.0),
            "field_scores": result.get("field_f1_scores", {}),
        }

    field_scores = {}
    total_f1 = 0.0

    for field in BANK_STATEMENT_FIELDS:
        extracted_value = schema_fields.get(field, "NOT_FOUND")
        gt_value = gt_data.get(field, "NOT_FOUND")
        if pd.isna(gt_value):
            gt_value = "NOT_FOUND"

        normalized_extracted = normalize_field_for_comparison(field, extracted_value)
        normalized_gt = normalize_field_for_comparison(field, gt_value)
        result = evaluate_field(normalized_extracted, normalized_gt, field, method)

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


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================
def extract_bank_statement(image_path, model, processor, verbose=False):
    """
    Extract fields from a single bank statement image.

    V2 Pipeline:
    - Turn 0: Header detection
    - Check if balance column exists:
      - YES: Use balance-description prompt (works for all formats)
      - NO: Use table extraction prompt (4-column tables)
    - Parse response and extract schema fields

    Returns:
        tuple: (schema_fields dict, metadata dict)
    """
    metadata = {
        "headers_detected": [],
        "extraction_method": None,
        "total_rows": 0,
        "debit_rows": 0,
    }

    image = Image.open(image_path)
    images = [image]

    # ========== TURN 0: Header Detection ==========
    turn0_prompt = """Look at the transaction table in this bank statement image.

What are the exact column header names used in the transaction table?

List each column header exactly as it appears, in order from left to right.
Do not interpret or rename them - use the EXACT text from the image.
"""

    message_turn0 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": turn0_prompt}],
        }
    ]

    text_input = processor.apply_chat_template(message_turn0, add_generation_prompt=True)
    inputs = processor(images=images, text=text_input, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    generate_ids = output[:, inputs["input_ids"].shape[1] : -1]
    turn0_response = processor.decode(generate_ids[0], clean_up_tokenization_spaces=False)

    del inputs, output, generate_ids
    torch.cuda.empty_cache()

    table_headers = parse_headers_from_response(turn0_response)
    metadata["headers_detected"] = table_headers
    metadata["turn0_raw_response"] = turn0_response

    if verbose:
        print(f"  Turn 0 Headers: {table_headers}")

    # Pattern matching for column names
    amount_col = match_header(table_headers, AMOUNT_PATTERNS, fallback=None)
    date_col = match_header(
        table_headers,
        DATE_PATTERNS,
        fallback=table_headers[0] if table_headers else "Date",
    )
    desc_col = match_header(
        table_headers,
        DESCRIPTION_PATTERNS,
        fallback=table_headers[1] if len(table_headers) > 1 else "Description",
    )
    debit_col = match_header(
        table_headers, DEBIT_PATTERNS, fallback=amount_col if amount_col else "Debit"
    )
    credit_col = match_header(
        table_headers, CREDIT_PATTERNS, fallback=amount_col if amount_col else "Credit"
    )
    balance_col = match_header(table_headers, BALANCE_PATTERNS, fallback=None)

    # ========== Decide Extraction Method Based on Balance Column ==========
    has_balance = balance_col is not None and balance_col in table_headers

    if has_balance:
        # Use balance-description prompt (works for both date formats)
        metadata["extraction_method"] = "balance-description"

        extraction_prompt = f"""Describe all the balances in the {balance_col} column, including:
- date from the Date Header
- {desc_col}
- amount
- whether it was a {debit_col} or a {credit_col}"""

        if verbose:
            print(f"  Extraction Method: balance-description (balance column: {balance_col})")
            print(f"  Extraction Prompt:\n{extraction_prompt}")

    else:
        # Use table extraction prompt (4-column tables without balance)
        metadata["extraction_method"] = "table-extraction"

        example_rows = build_date_per_row_example(table_headers)
        header_row = "| " + " | ".join(table_headers) + " |"
        example_table = header_row + "\n" + "\n".join(example_rows)

        extraction_prompt = f"""Extract the transaction table as markdown.

Example format:
{example_table}

Extract ALL transactions.

Output: Markdown table only."""

        if verbose:
            print("  Extraction Method: table-extraction (no balance column)")

    # ========== TURN 1: Extraction ==========
    message_turn1 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": extraction_prompt}],
        }
    ]

    text_input = processor.apply_chat_template(message_turn1, add_generation_prompt=True)
    inputs = processor(images=images, text=text_input, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=4096,  # Increased for balance-description output
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    generate_ids = output[:, inputs["input_ids"].shape[1] : -1]
    extraction_response = processor.decode(generate_ids[0], clean_up_tokenization_spaces=False)

    del inputs, output, generate_ids
    torch.cuda.empty_cache()

    if verbose:
        print(f"  Turn 1: Extraction complete ({len(extraction_response)} chars)")
        print(f"  Turn 1 Response:\n{extraction_response[:2000]}")

    # ========== Parse Response ==========
    if has_balance:
        # Parse balance-description format
        all_rows = parse_balance_description_response(
            extraction_response, date_col, desc_col, debit_col, credit_col, balance_col
        )
    else:
        # Parse markdown table format
        all_rows = parse_markdown_table(extraction_response)

    metadata["total_rows"] = len(all_rows)

    if verbose:
        print(f"  Parsed: {len(all_rows)} rows")

    # Filter to debit transactions only
    debit_rows = filter_debit_transactions(all_rows, debit_col, desc_col)
    metadata["debit_rows"] = len(debit_rows)

    if verbose:
        print(f"  Filtered: {len(debit_rows)} debit transactions")

    # Extract schema fields
    schema_fields = extract_schema_fields(debit_rows, date_col, desc_col, debit_col, all_rows=all_rows)

    return schema_fields, metadata


# ============================================================================
# REPORT GENERATION
# ============================================================================
def generate_reports(batch_results, output_dir):
    """Generate CSV, JSON, and Markdown reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in batch_results if "error" not in r]

    # CSV Report
    if CONFIG["GENERATE_CSV"] and successful:
        csv_data = []
        for result in successful:
            row = {
                "image_file": result["image_name"],
                "overall_accuracy": result["evaluation"]["overall_accuracy"],
                "processing_time": result["processing_time"],
                "extraction_method": result["metadata"].get("extraction_method", ""),
                "total_rows": result["metadata"].get("total_rows", 0),
                "debit_rows": result["metadata"].get("debit_rows", 0),
            }
            for field, value in result["extracted_fields"].items():
                row[field] = value
            field_scores = result["evaluation"].get("field_scores", {})
            for field, scores in field_scores.items():
                if isinstance(scores, dict):
                    row[f"{field}_f1"] = scores.get("f1_score", 0.0)
                else:
                    row[f"{field}_f1"] = scores
            csv_data.append(row)

        results_df = pd.DataFrame(csv_data)
        csv_path = output_dir / f"llama_bank_v2_{BATCH_TIMESTAMP}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")

    # JSON Report
    if CONFIG["GENERATE_JSON"]:
        processing_times = [r["processing_time"] for r in successful]
        json_report = {
            "metadata": {
                "batch_id": BATCH_TIMESTAMP,
                "model": "Llama-3.2-11B-Vision-Instruct",
                "version": "v2-balance-description",
                "evaluation_method": CONFIG["EVALUATION_METHOD"],
                "total_images": len(batch_results),
                "successful": len(successful),
                "failed": len(batch_results) - len(successful),
            },
            "summary": {
                "avg_accuracy": float(np.mean([r["evaluation"]["overall_accuracy"] for r in successful])) if successful else 0.0,
                "min_accuracy": float(min([r["evaluation"]["overall_accuracy"] for r in successful])) if successful else 0.0,
                "max_accuracy": float(max([r["evaluation"]["overall_accuracy"] for r in successful])) if successful else 0.0,
                "avg_processing_time": float(np.mean(processing_times)) if processing_times else 0.0,
            },
            "results": batch_results,
        }
        json_path = output_dir / f"llama_bank_v2_{BATCH_TIMESTAMP}.json"
        with json_path.open("w") as f:
            json_module.dump(json_report, f, indent=2, default=str)
        print(f"JSON saved: {json_path}")

    # Markdown Report
    if CONFIG["GENERATE_MARKDOWN"] and successful:
        processing_times = [r["processing_time"] for r in successful]
        md_lines = [
            "# Llama Bank Statement V2 Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Batch ID:** {BATCH_TIMESTAMP}",
            "**Model:** Llama-3.2-11B-Vision-Instruct",
            "**Version:** v2-balance-description",
            f"**Evaluation Method:** {CONFIG['EVALUATION_METHOD']}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Images:** {len(batch_results)}",
            f"- **Successful:** {len(successful)} ({len(successful) / len(batch_results) * 100:.1f}%)",
            f"- **Failed:** {len(batch_results) - len(successful)}",
            "",
        ]

        if successful:
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
                "| Image | Accuracy | Method | Rows | Time |",
                "|-------|----------|--------|------|------|",
            ])

            for r in successful:
                acc = r["evaluation"]["overall_accuracy"]
                method = r["metadata"].get("extraction_method", "N/A")
                rows = r["metadata"].get("debit_rows", 0)
                time_s = r["processing_time"]
                md_lines.append(f"| {r['image_name']} | {acc:.1%} | {method} | {rows} | {time_s:.1f}s |")

        md_path = output_dir / f"llama_bank_v2_{BATCH_TIMESTAMP}.md"
        with md_path.open("w") as f:
            f.write("\n".join(md_lines))
        print(f"Markdown saved: {md_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch extraction and evaluation of bank statements with Llama-3.2-Vision (V2)"
    )
    parser.add_argument("--max-images", type=int, default=None, help="Limit to N images")
    parser.add_argument(
        "--method",
        type=str,
        default="order_aware_f1",
        choices=["order_aware_f1", "position_agnostic_f1", "kieval", "correlation"],
        help="Evaluation method",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")

    args = parser.parse_args()

    CONFIG["MAX_IMAGES"] = args.max_images
    CONFIG["EVALUATION_METHOD"] = args.method
    CONFIG["VERBOSE"] = args.verbose

    global file_console
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_console = Console(file=log_path.open("w"), force_terminal=True, width=200)
        log_print(f"[dim]Logging to: {log_path}[/dim]")

    set_seed(42)

    log_rule("[bold blue]LLAMA BANK STATEMENT V2 - BALANCE DESCRIPTION")
    log_print(f"[cyan]Evaluation Method:[/cyan] {CONFIG['EVALUATION_METHOD']}")

    log_print("\n[yellow]Loading ground truth...[/yellow]")
    ground_truth_map = load_ground_truth(str(CONFIG["GROUND_TRUTH"]), verbose=True)

    bank_images = []
    for img_name, gt_data in ground_truth_map.items():
        doc_type = str(gt_data.get("DOCUMENT_TYPE", "")).upper()
        if doc_type == CONFIG["DOCUMENT_TYPE_FILTER"]:
            img_path = CONFIG["DATA_DIR"] / img_name
            if img_path.exists():
                bank_images.append(str(img_path))
            else:
                log_print(f"  [yellow]Warning: Image not found: {img_path}[/yellow]")

    if CONFIG["MAX_IMAGES"]:
        bank_images = bank_images[: CONFIG["MAX_IMAGES"]]

    log_print(f"\n[green]Found {len(bank_images)} bank statement images[/green]")

    if args.dry_run:
        log_print("\n[yellow]DRY RUN - Images that would be processed:[/yellow]")
        for img in bank_images:
            log_print(f"  - {Path(img).name}")
        return

    log_print("\n[yellow]Loading Llama-3.2-Vision model...[/yellow]")
    model, processor = load_llama_model_robust(
        model_path=CONFIG["MODEL_PATH"],
        use_quantization=False,
        device_map="auto",
        max_new_tokens=4096,
        torch_dtype="bfloat16",
        low_cpu_mem_usage=True,
        verbose=True,
    )

    try:
        model.tie_weights()
    except Exception:
        pass

    log_rule("[bold blue]BATCH PROCESSING")

    batch_results = []

    for idx, image_path in enumerate(bank_images, 1):
        image_name = Path(image_path).name
        log_print(f"\n[bold cyan][{idx}/{len(bank_images)}][/bold cyan] Processing: [white]{image_name}[/white]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()

        try:
            schema_fields, metadata = extract_bank_statement(
                image_path, model, processor, verbose=CONFIG["VERBOSE"]
            )

            processing_time = time.time() - start_time

            eval_result = evaluate_extraction(
                schema_fields, image_name, ground_truth_map, CONFIG["EVALUATION_METHOD"]
            )

            result = {
                "image_name": image_name,
                "image_path": image_path,
                "extracted_fields": schema_fields,
                "metadata": metadata,
                "evaluation": eval_result,
                "processing_time": processing_time,
            }

            batch_results.append(result)

            accuracy = eval_result.get("overall_accuracy", 0.0)
            acc_color = "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.5 else "red"
            log_print(f"  [{acc_color}]Accuracy: {accuracy:.1%}[/{acc_color}]  {processing_time:.2f}s")

            display_field_comparison(schema_fields, ground_truth_map, image_name, eval_result)

        except Exception as e:
            log_print(f"  [red]ERROR: {e}[/red]")
            batch_results.append({
                "image_name": image_name,
                "image_path": image_path,
                "error": str(e),
                "processing_time": time.time() - start_time,
            })

    log_rule("[bold blue]BATCH SUMMARY")

    successful = [r for r in batch_results if "error" not in r]
    failed_count = len(batch_results) - len(successful)

    log_print(f"[cyan]Total:[/cyan] {len(batch_results)} images")
    log_print(f"[green]Successful:[/green] {len(successful)}")
    if failed_count > 0:
        log_print(f"[red]Failed:[/red] {failed_count}")
    else:
        log_print("[dim]Failed:[/dim] 0")

    if successful:
        accuracies = [r["evaluation"]["overall_accuracy"] for r in successful]
        avg_acc = np.mean(accuracies)
        log_print("\n[bold]Accuracy Statistics:[/bold]")
        log_print(f"  Average: [{'green' if avg_acc >= 0.8 else 'yellow'}]{avg_acc:.1%}[/]")
        log_print(f"  Min: {min(accuracies):.1%}")
        log_print(f"  Max: {max(accuracies):.1%}")

    log_rule("[bold blue]GENERATING REPORTS")
    generate_reports(batch_results, CONFIG["OUTPUT_BASE"])

    log_print("\n[bold green]Batch processing complete![/bold green]")


if __name__ == "__main__":
    main()
