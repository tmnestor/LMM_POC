# Prompt Templating Improvement Proposal

## 1. Current Prompt Templating Issues

### What I Observed

The notebook uses **f-strings with embedded Python logic**:

```python
# Repeated pattern matching in multiple places
for h in table_headers:
    if h.lower() in ['date', 'day']:
        row.append(date)
    elif 'desc' in h.lower() or 'particular' in h.lower() or 'detail' in h.lower():
        row.append(desc)
    elif 'debit' in h.lower() or 'withdrawal' in h.lower():
        row.append(deb)
    # ... repeated 5+ times across different functions
```

```python
follow_up_prompt = f"""Extract the transaction table as markdown.

If you see this structure (dates as section headers with empty cells):
{source_table}

Transform to this format (date on every row):
{target_table}
...
"""
```

### Current Problems

| Issue | Impact |
|-------|--------|
| **Repeated column-matching logic** | Same `if h.lower() in [...]` patterns in 5+ places |
| **Prompts embedded in code** | Can't review/edit prompts without reading Python |
| **No prompt versioning** | Hard to A/B test or rollback prompt changes |
| **Complex example generation** | `build_date_grouped_source()`, `build_date_grouped_target()` etc. are hard to maintain |
| **Tight coupling** | Prompt text, structure, and examples all intertwined |
| **Notebook diffs are noisy** | JSON format makes git diffs unreadable |

---

## 2. YAML Advantages

### A. Separation of Concerns

```yaml
# prompts/bank_statement/extraction.yaml
version: "2.1"
description: "Balance-description extraction prompt"
author: "tod"
last_updated: "2025-12-01"

prompts:
  turn0_header_detection:
    template: |
      Look at the transaction table in this bank statement image.

      What are the exact column header names used in the transaction table?
      List each column header exactly as it appears, in order from left to right.
      Do not interpret or rename them - use the EXACT text from the image.

  turn1_extraction:
    template: |
      Extract ALL transactions from this bank statement.
      ...
```

### B. Column Mapping Configuration

```yaml
# config/column_mappings.yaml
column_patterns:
  date:
    patterns: ["date", "day", "transaction date", "trans date"]
    priority: 1

  description:
    patterns: ["description", "details", "particulars", "narrative", "transaction"]
    priority: 2

  debit:
    patterns: ["debit", "debits", "withdrawal", "withdrawals", "paid", "dr"]
    priority: 3

  credit:
    patterns: ["credit", "credits", "deposit", "deposits", "received", "cr"]
    priority: 4

  balance:
    patterns: ["balance", "bal", "running balance"]
    priority: 5
```

### C. Example Data Separate from Logic

```yaml
# prompts/examples/date_per_row.yaml
format: date_per_row
description: "Each transaction has its own date"

examples:
  - date: "15 Jan"
    description: "ATM Withdrawal"
    debit: "200.00"
    credit: ""
    balance: "$1,500.00 CR"

  - date: "16 Jan"
    description: "Salary Payment"
    debit: ""
    credit: "3,500.00"
    balance: "$5,000.00 CR"
```

### D. Variant Management

```yaml
# prompts/variants/extraction_v2.yaml
base: extraction_v1
changes:
  - Added explicit column alignment instructions
  - Removed redundant balance verification step

metrics:
  f1_score: 0.87
  tested_on: ["image_003", "image_008", "image_009"]
```

---

## 3. Jinja2 Advantages

### A. Template Inheritance

```jinja2
{# templates/base_extraction.j2 #}
{% block instructions %}
Extract the transaction table from this bank statement.
{% endblock %}

{% block format_rules %}
- Use markdown table format
- Include ALL transactions
{% endblock %}

{% block column_rules %}
{% for col in columns %}
- {{ col.name }}: {{ col.instruction }}
{% endfor %}
{% endblock %}
```

```jinja2
{# templates/date_grouped_extraction.j2 #}
{% extends "base_extraction.j2" %}

{% block format_rules %}
{{ super() }}
- Dates appear as section headers - distribute to each transaction row
{% endblock %}
```

### B. Conditional Logic in Templates (Not Python)

```jinja2
{# Current approach: Python code decides which prompt to use #}
{# Jinja2 approach: Template handles conditions #}

{% if date_format == "date_grouped" %}
If you see dates as section headers with multiple transactions underneath:
{{ source_example | indent(2) }}

Transform to this format (date on every row):
{{ target_example | indent(2) }}
{% else %}
Extract directly - each row already has its date.
{% endif %}
```

### C. Filters for Formatting

```jinja2
{# Clean formatting without Python string manipulation #}
| {{ headers | join(' | ') }} |
{% for row in examples %}
| {{ row.values() | join(' | ') }} |
{% endfor %}

{# Dynamic column width alignment #}
{% for row in examples %}
| {% for col, val in row.items() %}{{ val | center(widths[col]) }}{% if not loop.last %} | {% endif %}{% endfor %} |
{% endfor %}
```

### D. Macros for Reusable Components

```jinja2
{# macros/table_formatting.j2 #}
{% macro render_example_table(headers, rows) %}
| {{ headers | join(' | ') }} |
|{% for h in headers %}{{ '-' * (h|length + 2) }}|{% endfor %}
{% for row in rows %}
| {{ row | join(' | ') }} |
{% endfor %}
{% endmacro %}

{# Usage in prompts #}
Example output format:
{{ render_example_table(table_headers, example_rows) }}
```

---

## 4. Combined Architecture Proposal

```
prompts/
├── config/
│   ├── column_mappings.yaml      # Pattern definitions
│   └── generation_params.yaml    # Model settings
├── templates/
│   ├── base.j2                   # Base template
│   ├── turn0_headers.j2          # Header detection
│   ├── turn1_extraction.j2       # Main extraction
│   └── macros/
│       └── tables.j2             # Reusable table formatting
├── examples/
│   ├── date_per_row.yaml         # Example data
│   └── date_grouped.yaml
└── variants/
    ├── v1_baseline.yaml          # Prompt versions for A/B testing
    └── v2_improved.yaml
```

### Loader Code (One-Time Setup)

```python
# common/prompt_loader.py
from jinja2 import Environment, FileSystemLoader
import yaml

class PromptLoader:
    def __init__(self, prompts_dir: str = "prompts"):
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}/templates"))
        self.config = self._load_yaml(f"{prompts_dir}/config/column_mappings.yaml")

    def render(self, template_name: str, **context) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)

    def get_extraction_prompt(self, table_headers: list, date_format: str) -> str:
        mapped_cols = self._map_columns(table_headers)
        examples = self._load_examples(date_format)

        return self.render(
            "turn1_extraction.j2",
            headers=table_headers,
            columns=mapped_cols,
            date_format=date_format,
            examples=examples
        )
```

### Notebook Usage (Clean)

```python
# In notebook - simple and clean
from common.prompt_loader import PromptLoader

loader = PromptLoader()

# Turn 0
turn0_prompt = loader.render("turn0_headers.j2")

# Turn 1 (after header detection)
extraction_prompt = loader.get_extraction_prompt(
    table_headers=table_headers,
    date_format=date_format
)
```

---

## 5. Summary: Why This Matters

| Aspect | Current (f-strings) | YAML + Jinja2 |
|--------|---------------------|---------------|
| **Readability** | Prompts buried in Python | Prompts are first-class artifacts |
| **Maintainability** | Change requires understanding code | Edit YAML/template directly |
| **Version control** | Noisy notebook diffs | Clean text file diffs |
| **A/B testing** | Copy-paste code | Load different variant file |
| **Collaboration** | Only devs can edit | Non-devs can review/edit prompts |
| **Reusability** | Functions with repeated logic | Macros and inheritance |
| **DRY principle** | Column patterns repeated 5+ times | Single source of truth |

The investment is **one-time setup** of the loader infrastructure, with ongoing benefits for every prompt iteration.
