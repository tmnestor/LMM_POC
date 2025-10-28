"""Text cleaning utilities for VLM response post-processing."""

import re


def clean_llama_response(response: str) -> str:
    """
    Remove chat template artifacts and extract only the assistant's response.

    Llama models wrap responses in chat template markers like:
    <|start_header_id|>assistant<|end_header_id|>
    ... actual response ...
    <|eot_id|>

    Args:
        response: Raw response string from Llama model

    Returns:
        Cleaned response with template artifacts removed
    """
    start_marker = "<|start_header_id|>assistant<|end_header_id|>"
    end_marker = "<|eot_id|>"

    start_idx = response.find(start_marker)
    if start_idx != -1:
        start_idx += len(start_marker)
        end_idx = response.find(end_marker, start_idx)
        if end_idx != -1:
            return response[start_idx:end_idx].strip()

    return response.replace("***", "").strip()


def clean_markdown_table(markdown_text: str) -> str:
    """
    Replace empty cells in markdown table with NOT_FOUND.

    Handles patterns like:
    - "|  |" → "| NOT_FOUND |"
    - "| |"  → "| NOT_FOUND |"
    - "|   |" → "| NOT_FOUND |"

    Only processes table rows, skips header separator lines (| --- | --- |).

    Args:
        markdown_text: Markdown text containing table

    Returns:
        Markdown text with empty cells replaced by NOT_FOUND
    """
    lines = markdown_text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip header separator lines (like "| --- | --- | --- |")
        if re.match(r'^\|\s*-+\s*\|', line):
            cleaned_lines.append(line)
            continue

        # Replace empty cells in data rows
        if '|' in line:
            # Pattern: pipe followed by only whitespace followed by pipe
            cleaned_line = re.sub(r'\|\s+\|', '| NOT_FOUND |', line)
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
