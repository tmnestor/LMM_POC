#!/usr/bin/env python
"""
InternVL3.5-8B Bank Statement Batch Extraction and Evaluation

This script runs batch extraction on bank statement images using the
InternVL3.5-8B model with the independent single-turn approach.

Features:
- Batch processing of all bank statement images
- Configurable evaluation methods (order_aware_f1, position_agnostic_f1, kieval, correlation)
- Full report generation (CSV, JSON, Markdown)
- H200 GPU optimized (bfloat16, flash attention, MAX_TILES=36)

Usage:
    python ivl3_5_8b_bank_statement_batch.py [OPTIONS]

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
import torchvision.transforms as T
from dateutil import parser as date_parser
from PIL import Image
from rich.console import Console
from rich.table import Table
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from common.evaluation_metrics import (
    calculate_correlation_aware_f1,
    calculate_field_accuracy_f1,
    calculate_field_accuracy_f1_position_agnostic,
    calculate_field_accuracy_kieval,
    load_ground_truth,
)
from common.reproducibility import set_seed

# Rich console for styled output
console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Data paths
    "DATA_DIR": Path("/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data/bank"),
    "GROUND_TRUTH": Path(
        "/home/jovyan/nfs_share/tod/LMM_POC/evaluation_data/bank/ground_truth_bank.csv"
    ),
    "OUTPUT_BASE": Path("/home/jovyan/nfs_share/tod/LMM_POC/output"),
    # Model path
    "MODEL_PATH": "/home/jovyan/nfs_share/models/InternVL3_5-8B",
    # H200 TILE CONFIGURATION
    "MAX_TILES": 36,  # H200 optimized - InternVL3.5 training max for dense OCR
    # Generation settings
    "MAX_NEW_TOKENS": 2000,
    # H200 precision settings
    "TORCH_DTYPE": "bfloat16",
    "USE_FLASH_ATTN": True,
    # Filtering
    "DOCUMENT_TYPE_FILTER": "BANK_STATEMENT",
    "MAX_IMAGES": None,
    # Evaluation
    "EVALUATION_METHOD": "order_aware_f1",
    "VERBOSE": True,
    # Reports
    "GENERATE_CSV": True,
    "GENERATE_JSON": True,
    "GENERATE_MARKDOWN": True,
}

BATCH_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Bank statement fields to evaluate
BANK_STATEMENT_FIELDS = [
    "DOCUMENT_TYPE",
    "STATEMENT_DATE_RANGE",
    "TRANSACTION_DATES",
    "LINE_ITEM_DESCRIPTIONS",
    "TRANSACTION_AMOUNTS_PAID",
]

# ============================================================================
# INTERNVL3.5 IMAGE PREPROCESSING (from notebook)
# ============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.229)


def build_transform(input_size):
    """Build image transformation pipeline with ImageNet normalization."""
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios based on image dimensions."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=None, image_size=448, use_thumbnail=False
):
    """
    Dynamically preprocess image by splitting into tiles based on aspect ratio.
    """
    if max_num is None:
        max_num = CONFIG["MAX_TILES"]

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate target aspect ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file, input_size=448, max_num=None):
    """Load and preprocess image for InternVL3.5."""
    if max_num is None:
        max_num = CONFIG["MAX_TILES"]

    # Handle both path string and PIL Image
    if isinstance(image_file, str):
        image = Image.open(image_file).convert("RGB")
    else:
        image = image_file

    # Build transform and preprocess
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values


# ============================================================================
# INTERNVL3.5 MODEL LOADING (H200 optimized - simpler than V100)
# ============================================================================
def load_internvl3_5_model(verbose=True):
    """Load InternVL3.5-8B model optimized for H200 GPU.

    H200 has 80GB HBM3 memory, so we can use:
    - Full bfloat16 precision (no quantization needed)
    - Flash Attention for efficiency
    - Simple device_map="auto"
    """
    if verbose:
        print("Loading InternVL3.5-8B for H200 GPU...")
        print(f"  Model path: {CONFIG['MODEL_PATH']}")
        print("  Precision: bfloat16")
        print(f"  Flash Attention: {CONFIG['USE_FLASH_ATTN']}")
        print(f"  Max Tiles: {CONFIG['MAX_TILES']}")

    # Load model with bfloat16 and flash attention for H200
    model = AutoModel.from_pretrained(
        CONFIG["MODEL_PATH"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_flash_attn=CONFIG["USE_FLASH_ATTN"],
    ).eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["MODEL_PATH"], trust_remote_code=True, use_fast=False
    )

    # Set generation config
    model.config.max_new_tokens = CONFIG["MAX_NEW_TOKENS"]

    # Fix pad_token_id to suppress warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")
        print(f"  Device map: {model.hf_device_map}")
        print("  Model loaded successfully!")

    return model, tokenizer


def clean_internvl3_response(response):
    """Remove InternVL3 markdown artifacts."""
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!["):  # Skip image markdown
            continue
        if stripped in ["```markdown", "```", "```md"]:  # Skip code fences
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ============================================================================
# SEMANTIC NORMALIZATION (for evaluation comparison only)
# ============================================================================
def normalize_date(date_str):
    """Normalize date string to canonical format YYYY-MM-DD for semantic comparison."""
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

    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[$£€¥₹\s]", "", amount_str)
    cleaned = cleaned.replace(",", "")

    try:
        value = float(cleaned)
        value = abs(value)  # Ignore sign for matching
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
# PATTERN MATCHING (from notebook Cell 14)
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
    "withdrawal",
    "withdrawals",
    "paid",
    "paid out",
    "spent",
    "dr",
]
CREDIT_PATTERNS = ["credit", "deposit", "deposits", "received", "cr"]
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


def is_balance_row(row, desc_col):
    """Check if this row is an opening/closing balance row."""
    desc = row.get(desc_col, "").upper()
    return "OPENING BALANCE" in desc or "CLOSING BALANCE" in desc


def validate_and_correct_alignment(rows, balance_col, debit_col, credit_col, desc_col):
    """Use balance changes to validate and correct debit/credit alignment."""
    if not rows or balance_col not in rows[0]:
        return rows, 0

    corrected_rows = []
    corrections_made = 0
    start_idx = 0

    if rows and is_balance_row(rows[0], desc_col):
        start_idx = 1
    elif rows:
        corrected_rows.append(rows[0].copy())
        start_idx = 1

    for i in range(start_idx, len(rows)):
        current_row = rows[i].copy()

        if is_balance_row(current_row, desc_col):
            continue

        prev_idx = i - 1
        while prev_idx >= 0 and is_balance_row(rows[prev_idx], desc_col):
            prev_idx -= 1

        if prev_idx < 0:
            corrected_rows.append(current_row)
            continue

        prev_balance = parse_amount(rows[prev_idx].get(balance_col, "0"))
        curr_balance = parse_amount(current_row.get(balance_col, "0"))
        balance_change = curr_balance - prev_balance

        debit_value = parse_amount(current_row.get(debit_col, ""))
        credit_value = parse_amount(current_row.get(credit_col, ""))

        if balance_change > 0.01:
            if debit_value > 0 and credit_value == 0:
                current_row[credit_col] = current_row[debit_col]
                current_row[debit_col] = ""
                corrections_made += 1
        elif balance_change < -0.01:
            if credit_value > 0 and debit_value == 0:
                current_row[debit_col] = current_row[credit_col]
                current_row[credit_col] = ""
                corrections_made += 1

        corrected_rows.append(current_row)

    return corrected_rows, corrections_made


def is_non_transaction_row(row, desc_col):
    """Check if this row is NOT an actual transaction."""
    desc = row.get(desc_col, "").strip().upper()

    if "OPENING BALANCE" in desc:
        return True
    if "CLOSING BALANCE" in desc:
        return True

    return False


def filter_debit_transactions(rows, debit_col, desc_col=None):
    """Filter rows to only those with actual debit (purchase) transactions."""
    debit_rows = []
    for row in rows:
        debit_value = row.get(debit_col, "").strip()

        if not debit_value:
            continue

        if desc_col and is_non_transaction_row(row, desc_col):
            continue

        debit_rows.append(row)

    return debit_rows


def extract_schema_fields(rows, date_col, desc_col, debit_col):
    """Extract fields in universal.yaml schema format."""
    if not rows:
        return {
            "DOCUMENT_TYPE": "BANK_STATEMENT",
            "STATEMENT_DATE_RANGE": "NOT_FOUND",
            "TRANSACTION_DATES": "NOT_FOUND",
            "LINE_ITEM_DESCRIPTIONS": "NOT_FOUND",
            "TRANSACTION_AMOUNTS_PAID": "NOT_FOUND",
        }

    dates = []
    descriptions = []
    amounts = []

    for row in rows:
        date = row.get(date_col, "").strip()
        desc = row.get(desc_col, "").strip()
        amount = row.get(debit_col, "").strip()

        if date:
            dates.append(date)
        if desc:
            descriptions.append(desc)
        if amount:
            amounts.append(amount)

    date_range = "NOT_FOUND"
    if dates:
        date_range = f"{dates[0]} - {dates[-1]}"

    return {
        "DOCUMENT_TYPE": "BANK_STATEMENT",
        "STATEMENT_DATE_RANGE": date_range,
        "TRANSACTION_DATES": " | ".join(dates) if dates else "NOT_FOUND",
        "LINE_ITEM_DESCRIPTIONS": " | ".join(descriptions)
        if descriptions
        else "NOT_FOUND",
        "TRANSACTION_AMOUNTS_PAID": " | ".join(amounts) if amounts else "NOT_FOUND",
    }


# ============================================================================
# PROMPT BUILDING
# ============================================================================
def build_dynamic_example(
    headers, date_col, desc_col, debit_col, credit_col, balance_col
):
    """Generate example rows matching detected column structure."""
    has_separate_debit_credit = (
        debit_col in headers and credit_col in headers and debit_col != credit_col
    )

    rows = []

    if has_separate_debit_credit:
        # 5-column format
        for date, desc, deb, cred, bal in [
            ("15 Jan", "ATM Withdrawal City Branch", "200.00", "", "$1,500.00 CR"),
            (
                "16 Jan",
                "Salary Employer Name Ref 12345",
                "",
                "3,500.00",
                "$5,000.00 CR",
            ),
            ("17 Jan", "Online Purchase Store Name", "150.00", "", "$4,850.00 CR"),
        ]:
            row = []
            for h in headers:
                if h == date_col:
                    row.append(date)
                elif h == desc_col:
                    row.append(desc)
                elif h == debit_col:
                    row.append(deb)
                elif h == credit_col:
                    row.append(cred)
                elif h == balance_col:
                    row.append(bal)
                else:
                    row.append("")
            rows.append(" | ".join(row))
    else:
        # 4-column format
        for date, desc, amt, bal in [
            ("15 Jan", "ATM Withdrawal City Branch", "200.00", "$1,500.00 CR"),
            ("16 Jan", "Salary Employer Name Ref 12345", "3,500.00", "$5,000.00 CR"),
        ]:
            row = []
            for h in headers:
                if h == date_col:
                    row.append(date)
                elif h == desc_col:
                    row.append(desc)
                elif h == debit_col:
                    row.append(amt)
                elif h == balance_col:
                    row.append(bal)
                else:
                    row.append("")
            rows.append(" | ".join(row))

    return rows


def build_multiline_rule(headers):
    """Generate multi-line extraction rule using ACTUAL column structure from Turn 0."""
    num_cols = len(headers)

    # Find actual Debit, Credit, and Balance columns by name
    debit_idx = None
    credit_idx = None
    balance_idx = None

    for i, header in enumerate(headers):
        h_lower = header.lower()
        if any(p in h_lower for p in ["debit", "withdrawal", "paid", "spent", "dr"]):
            debit_idx = i
        if any(p in h_lower for p in ["credit", "deposit", "received", "cr"]):
            credit_idx = i
        if any(p in h_lower for p in ["balance", "bal"]):
            balance_idx = i

    if debit_idx is None or credit_idx is None:
        return "Multi-line: combine description lines into single row."

    def format_aligned_table(rows):
        """Format rows with properly aligned vertical pipes."""
        if not rows:
            return []

        num_cols_local = len(rows[0])

        # Calculate max width for each column
        col_widths = [0] * num_cols_local
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        # Find last non-empty column index
        last_col = 0
        for row in rows:
            for i, val in enumerate(row):
                if val:
                    last_col = max(last_col, i)

        # Ensure empty MIDDLE columns have minimum width
        for i in range(1, last_col):  # Skip first column, only middle columns
            if col_widths[i] == 0:
                col_widths[i] = 7

        # Format each row with proper alignment
        formatted = []
        for row in rows:
            # Determine how many columns to include
            end_col = last_col + 2 if last_col < len(row) - 1 else last_col + 1
            end_col = min(end_col, len(row))

            # Pad each column value to its width
            parts = []
            for i in range(end_col):
                val = row[i] if i < len(row) else ""
                parts.append(val.ljust(col_widths[i]))

            line = " | ".join(parts)

            # CRITICAL: If first column is empty, add leading spaces to align pipes
            if not row[0]:
                line = " " * col_widths[0] + " | " + " | ".join(parts[1:])

            formatted.append(line)

        return formatted

    # Create example rows using ACTUAL column positions
    # Credit example (amount in credit_idx position)
    credit_rows = [[""] * num_cols for _ in range(3)]
    credit_rows[0][0] = "a date"
    credit_rows[0][1] = "line 1"
    credit_rows[0][credit_idx] = "85.50"
    # Add Balance column value if it exists
    if balance_idx is not None:
        credit_rows[0][balance_idx] = "$1,085.50 CR"
    credit_rows[1][0] = ""  # Empty date for continuation
    credit_rows[1][1] = "line 2"
    credit_rows[2][0] = "a date"
    credit_rows[2][1] = "line 1 line 2"
    credit_rows[2][credit_idx] = "85.50"
    # Add Balance column value if it exists
    if balance_idx is not None:
        credit_rows[2][balance_idx] = "$1,085.50 CR"

    # Debit example (amount in debit_idx position)
    debit_rows = [[""] * num_cols for _ in range(3)]
    debit_rows[0][0] = "a date"
    debit_rows[0][1] = "line 1"
    debit_rows[0][debit_idx] = "150.00"
    # Add Balance column value if it exists
    if balance_idx is not None:
        debit_rows[0][balance_idx] = "$850.00 CR"
    debit_rows[1][0] = ""  # Empty date for continuation
    debit_rows[1][1] = "line 2"
    debit_rows[2][0] = "a date"
    debit_rows[2][1] = "line 1 line 2"
    debit_rows[2][debit_idx] = "150.00"
    # Add Balance column value if it exists
    if balance_idx is not None:
        debit_rows[2][balance_idx] = "$850.00 CR"

    # Format both examples
    credit_fmt = format_aligned_table(credit_rows)
    debit_fmt = format_aligned_table(debit_rows)

    # Build rule with LABELED examples using actual header names
    rule = f"""  {headers[credit_idx]} example:
       {credit_fmt[0]}
       {credit_fmt[1]}
    you must extract it as:
       {credit_fmt[2]}

  {headers[debit_idx]} example:
       {debit_fmt[0]}
       {debit_fmt[1]}
    you must extract it as:
       {debit_fmt[2]}"""

    return rule


def build_extraction_prompt(
    table_headers, date_col, desc_col, debit_col, credit_col, balance_col
):
    """Build the Turn 1 extraction prompt with dynamic examples."""
    header_string = " | ".join(table_headers)

    # Build separator row
    separator_parts = []
    for h in table_headers:
        h_lower = h.lower()
        if any(
            keyword in h_lower
            for keyword in ["debit", "credit", "balance", "amount", "total"]
        ):
            separator_parts.append("---:")
        else:
            separator_parts.append(":---")
    separator_row = " | ".join(separator_parts)

    # Build example rows
    example_rows = build_dynamic_example(
        table_headers, date_col, desc_col, debit_col, credit_col, balance_col
    )

    example_table = f"""| {header_string} |
| {separator_row} |
""" + "\n".join([f"| {row} |" for row in example_rows])

    # Build multi-line rule
    multiline_rule = build_multiline_rule(table_headers)

    prompt = f"""
Extract the transaction table from this bank statement image in markdown format.

Example showing the format I want:

{example_table}

## CRITICAL: COLUMN ALIGNMENT

Before extracting ANY row, locate the header row with these column names:
{" | ".join(table_headers)}

For EACH transaction, you must check which column the amount appears under:

**Step-by-step process:**
1. Find the header row
2. Look at the transaction row
3. Draw an imaginary vertical line from the amount UP to the header
4. Read which header the amount aligns with
5. Put the amount in that SAME column in your markdown table

**Column placement rules:**
- Amount aligns with "{debit_col}" header → put amount in {debit_col} column, leave {credit_col} EMPTY
- Amount aligns with "{credit_col}" header → put amount in {credit_col} column, leave {debit_col} EMPTY

**Do NOT guess based on description text. Use visual alignment ONLY.**

## OTHER RULES

**Multi-line transactions:** Combine description lines into single row:
{multiline_rule}

**Empty columns:** Leave empty (|  |)

**Output:** Markdown table only, no explanations
"""
    return prompt


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_field_comparison(schema_fields, ground_truth_map, image_name, eval_result):
    """Display stacked comparison of extracted vs ground truth fields using Rich."""
    gt_data = ground_truth_map.get(image_name, {})
    field_scores = eval_result.get("field_scores", {})

    normalized_fields = {
        "TRANSACTION_DATES",
        "TRANSACTION_AMOUNTS_PAID",
        "STATEMENT_DATE_RANGE",
    }

    # Create Rich table for field comparison
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
            status = "[green]✓ OK[/green]"
        elif f1_score >= 0.5:
            status = "[yellow]~ PART[/yellow]"
        else:
            status = "[red]✗ FAIL[/red]"

        # Truncate long values for display
        ext_display = (
            str(extracted_val)[:80] + "..."
            if len(str(extracted_val)) > 80
            else str(extracted_val)
        )
        gt_display = (
            str(ground_val)[:80] + "..."
            if len(str(ground_val)) > 80
            else str(ground_val)
        )

        table.add_row(status, field, f"{f1_score:.1%}", ext_display, gt_display)

    console.print(table)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def evaluate_field(extracted_value, gt_value, field_name, method):
    """Route to appropriate evaluation function."""
    if method == "order_aware_f1":
        return calculate_field_accuracy_f1(extracted_value, gt_value, field_name)
    elif method == "position_agnostic_f1":
        return calculate_field_accuracy_f1_position_agnostic(
            extracted_value, gt_value, field_name
        )
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

    overall_accuracy = (
        total_f1 / len(BANK_STATEMENT_FIELDS) if BANK_STATEMENT_FIELDS else 0.0
    )

    return {
        "image_name": image_name,
        "method": method,
        "overall_accuracy": overall_accuracy,
        "field_scores": field_scores,
    }


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================
def extract_bank_statement(image_path, model, tokenizer, verbose=False):
    """
    Extract fields from a single bank statement image using InternVL3.5-8B.

    Two-turn independent extraction with Python post-processing.
    """
    metadata = {
        "headers_detected": [],
        "date_format": "Date-per-row",
        "corrections_made": 0,
        "total_rows": 0,
        "debit_rows": 0,
    }

    # Determine model dtype (bfloat16 for H200)
    model_dtype = torch.bfloat16

    # ========== TURN 0: Header Detection ==========
    turn0_prompt = """
Look at the transaction table in this bank statement image.

IMPORTANT STRUCTURAL NOTE:
Some bank statements show dates as section headings with multiple transactions underneath.
If you see this structure, remember that each transaction needs its explicit date in the final output.

What are the exact column header names used in the transaction table?

List each column header exactly as it appears, in order from left to right.
Do not interpret or rename them - use the EXACT text from the image.
"""

    pixel_values = load_image(str(image_path), input_size=448)
    pixel_values = pixel_values.to(dtype=model_dtype, device="cuda:0")

    turn0_response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=turn0_prompt,
        generation_config={"max_new_tokens": 500, "do_sample": False},
    )

    # Free Turn 0 pixel values immediately
    del pixel_values
    torch.cuda.empty_cache()

    turn0_response = clean_internvl3_response(turn0_response)
    table_headers = parse_headers_from_response(turn0_response)
    metadata["headers_detected"] = table_headers

    if verbose:
        print(f"  Turn 0: {len(table_headers)} headers detected")

    # Pattern matching
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
    balance_col = match_header(table_headers, BALANCE_PATTERNS, fallback="Balance")

    # ========== TURN 1: Table Extraction (fresh context) ==========
    extraction_prompt = build_extraction_prompt(
        table_headers, date_col, desc_col, debit_col, credit_col, balance_col
    )

    # Reload image for fresh context
    pixel_values = load_image(str(image_path), input_size=448)
    pixel_values = pixel_values.to(dtype=model_dtype, device="cuda:0")

    turn1_response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=extraction_prompt,
        generation_config={
            "max_new_tokens": CONFIG["MAX_NEW_TOKENS"],
            "do_sample": False,
        },
    )

    # Free Turn 1 pixel values immediately
    del pixel_values
    torch.cuda.empty_cache()

    turn1_response = clean_internvl3_response(turn1_response)

    if verbose:
        print("  Turn 1: Table extraction complete")

    # ========== Parse and Validate ==========
    all_rows = parse_markdown_table(turn1_response)
    metadata["total_rows"] = len(all_rows)

    corrections = 0
    if all_rows and balance_col in all_rows[0]:
        all_rows, corrections = validate_and_correct_alignment(
            all_rows, balance_col, debit_col, credit_col, desc_col
        )
    metadata["corrections_made"] = corrections

    debit_rows = filter_debit_transactions(all_rows, debit_col, desc_col)
    metadata["debit_rows"] = len(debit_rows)

    if verbose:
        print(f"  Parsed: {len(all_rows)} rows, {len(debit_rows)} debits")

    schema_fields = extract_schema_fields(debit_rows, date_col, desc_col, debit_col)

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
                "date_format": result["metadata"].get("date_format", ""),
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
        csv_path = output_dir / f"ivl3_5_8b_bank_statement_batch_{BATCH_TIMESTAMP}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")

    # JSON Report
    if CONFIG["GENERATE_JSON"]:
        processing_times = [r["processing_time"] for r in successful]

        json_report = {
            "metadata": {
                "batch_id": BATCH_TIMESTAMP,
                "model": "InternVL3.5-8B",
                "evaluation_method": CONFIG["EVALUATION_METHOD"],
                "total_images": len(batch_results),
                "successful": len(successful),
                "failed": len(batch_results) - len(successful),
            },
            "summary": {
                "avg_accuracy": float(
                    np.mean([r["evaluation"]["overall_accuracy"] for r in successful])
                )
                if successful
                else 0.0,
                "min_accuracy": float(
                    min([r["evaluation"]["overall_accuracy"] for r in successful])
                )
                if successful
                else 0.0,
                "max_accuracy": float(
                    max([r["evaluation"]["overall_accuracy"] for r in successful])
                )
                if successful
                else 0.0,
                "avg_processing_time": float(np.mean(processing_times))
                if processing_times
                else 0.0,
            },
            "results": batch_results,
        }

        json_path = (
            output_dir / f"ivl3_5_8b_bank_statement_evaluation_{BATCH_TIMESTAMP}.json"
        )
        with json_path.open("w") as f:
            json_module.dump(json_report, f, indent=2, default=str)
        print(f"JSON saved: {json_path}")

    # Markdown Report
    if CONFIG["GENERATE_MARKDOWN"] and successful:
        processing_times = [r["processing_time"] for r in successful]

        md_lines = [
            "# InternVL3.5-8B Bank Statement Batch Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Batch ID:** {BATCH_TIMESTAMP}",
            "**Model:** InternVL3.5-8B",
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

            md_lines.extend(
                [
                    f"- **Average Accuracy:** {avg_acc:.1%}",
                    f"- **Min Accuracy:** {min_acc:.1%}",
                    f"- **Max Accuracy:** {max_acc:.1%}",
                    f"- **Avg Processing Time:** {np.mean(processing_times):.2f}s",
                    "",
                    "## Per-Image Results",
                    "",
                    "| Image | Accuracy | Date Format | Rows | Time |",
                    "|-------|----------|-------------|------|------|",
                ]
            )

            for r in successful:
                acc = r["evaluation"]["overall_accuracy"]
                fmt = r["metadata"].get("date_format", "N/A")
                rows = r["metadata"].get("debit_rows", 0)
                time_s = r["processing_time"]
                md_lines.append(
                    f"| {r['image_name']} | {acc:.1%} | {fmt} | {rows} | {time_s:.1f}s |"
                )

        md_path = output_dir / f"ivl3_5_8b_bank_statement_summary_{BATCH_TIMESTAMP}.md"
        with md_path.open("w") as f:
            f.write("\n".join(md_lines))
        print(f"Markdown saved: {md_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch extraction and evaluation of bank statements with InternVL3.5-8B"
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit to N images"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="order_aware_f1",
        choices=["order_aware_f1", "position_agnostic_f1", "kieval", "correlation"],
        help="Evaluation method",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed"
    )

    args = parser.parse_args()

    CONFIG["MAX_IMAGES"] = args.max_images
    CONFIG["EVALUATION_METHOD"] = args.method
    CONFIG["VERBOSE"] = args.verbose

    set_seed(42)

    console.rule("[bold blue]INTERNVL3.5-8B BANK STATEMENT BATCH EXTRACTION")
    console.print(f"[cyan]Evaluation Method:[/cyan] {CONFIG['EVALUATION_METHOD']}")
    console.print(f"[cyan]Max Tiles:[/cyan] {CONFIG['MAX_TILES']} (H200 optimized)")
    console.print("[cyan]Precision:[/cyan] bfloat16 with Flash Attention")

    # Load ground truth
    console.print("\n[yellow]Loading ground truth...[/yellow]")
    ground_truth_map = load_ground_truth(str(CONFIG["GROUND_TRUTH"]), verbose=True)

    # Discover bank statement images
    bank_images = []
    for img_name, gt_data in ground_truth_map.items():
        doc_type = str(gt_data.get("DOCUMENT_TYPE", "")).upper()
        if doc_type == CONFIG["DOCUMENT_TYPE_FILTER"]:
            img_path = CONFIG["DATA_DIR"] / img_name
            if img_path.exists():
                bank_images.append(str(img_path))
            else:
                console.print(
                    f"  [yellow]Warning: Image not found: {img_path}[/yellow]"
                )

    if CONFIG["MAX_IMAGES"]:
        bank_images = bank_images[: CONFIG["MAX_IMAGES"]]

    console.print(f"\n[green]✓ Found {len(bank_images)} bank statement images[/green]")

    if args.dry_run:
        console.print("\n[yellow]DRY RUN - Images that would be processed:[/yellow]")
        for img in bank_images:
            console.print(f"  - {Path(img).name}")
        return

    # Load model
    console.print("\n[yellow]Loading InternVL3.5-8B model...[/yellow]")
    model, tokenizer = load_internvl3_5_model(verbose=True)

    # Process images
    console.rule("[bold blue]BATCH PROCESSING")

    batch_results = []

    for idx, image_path in enumerate(bank_images, 1):
        image_name = Path(image_path).name
        console.print(
            f"\n[bold cyan][{idx}/{len(bank_images)}][/bold cyan] Processing: [white]{image_name}[/white]"
        )

        # Clear GPU memory before each image to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()

        try:
            schema_fields, metadata = extract_bank_statement(
                image_path, model, tokenizer, verbose=CONFIG["VERBOSE"]
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
            acc_color = (
                "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.5 else "red"
            )
            console.print(
                f"  [{acc_color}]✓ Accuracy: {accuracy:.1%}[/{acc_color}]  ⏱ {processing_time:.2f}s"
            )

            display_field_comparison(
                schema_fields, ground_truth_map, image_name, eval_result
            )

        except Exception as e:
            console.print(f"  [red]✗ ERROR: {e}[/red]")
            batch_results.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                }
            )

    # Summary
    console.rule("[bold blue]BATCH SUMMARY")

    successful = [r for r in batch_results if "error" not in r]
    failed_count = len(batch_results) - len(successful)

    console.print(f"[cyan]Total:[/cyan] {len(batch_results)} images")
    console.print(f"[green]Successful:[/green] {len(successful)}")
    if failed_count > 0:
        console.print(f"[red]Failed:[/red] {failed_count}")
    else:
        console.print("[dim]Failed:[/dim] 0")

    if successful:
        accuracies = [r["evaluation"]["overall_accuracy"] for r in successful]
        avg_acc = np.mean(accuracies)
        console.print("\n[bold]Accuracy Statistics:[/bold]")
        console.print(
            f"  Average: [{'green' if avg_acc >= 0.8 else 'yellow'}]{avg_acc:.1%}[/]"
        )
        console.print(f"  Min: {min(accuracies):.1%}")
        console.print(f"  Max: {max(accuracies):.1%}")

    # Generate reports
    console.rule("[bold blue]GENERATING REPORTS")
    generate_reports(batch_results, CONFIG["OUTPUT_BASE"])

    console.print("\n[bold green]✓ Batch processing complete![/bold green]")


if __name__ == "__main__":
    main()
