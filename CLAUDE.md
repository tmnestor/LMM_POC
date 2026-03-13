# CLAUDE.md - InternVL3.5-8B Focused Branch

This branch focuses solely on the InternVL3.5-8B vision-language model for document extraction.

## Project Structure

```
.
├── ivl3_5_8b.ipynb          # Main notebook - InternVL3.5-8B batch extraction
├── common/                   # Shared utilities and processing modules
├── models/                   # Model processor implementations
├── prompts/                  # YAML prompt configurations
├── config/                   # Field definitions and model config
├── environment.yml           # Conda environment specification
└── environment_ivl35.yml     # IVL3.5-specific environment
```

## Environment Setup

```bash
# Create and activate environment
conda env create -f environment_ivl35.yml
conda activate vision_notebooks

# Key requirements:
# - transformers>=4.52.1 (critical for InternVL3.5 support)
# - flash-attn (for optimized inference)
# - torch with CUDA support
```

## Local Script Execution

For **all** local Python commands (imports, scripts, verification), use the `du` conda environment:

```bash
# Activate before any local python command
conda activate /opt/homebrew/Caskroom/miniforge/base/envs/du

# Or use the full path directly
/opt/homebrew/Caskroom/miniforge/base/envs/du/bin/python your_script.py
```

## Model: InternVL3.5-8B

- **Parameters**: 8.5B (0.3B vision + 8.2B language)
- **Features**: Cascade RL, Visual Resolution Router (ViR)
- **Precision**: bfloat16 (H200/H100/A100) or float32 (V100)
- **Model path**: Configure in notebook CONFIG section

## Running the Notebook

1. Update model paths in the CONFIG section
2. Configure data directory and ground truth paths
3. Select "Kernel → Restart Kernel" then "Cell → Run All"

## YAML-First Configuration Policy (MANDATORY)

**NEVER hardcode configuration values in Python code.** All tuneable parameters MUST come from YAML config files, with the config cascade: CLI > YAML > defaults.

### Config sources (single source of truth)
- **`config/run_config.yml`** — model paths, batch sizes, generation params, GPU settings
- **`config/field_definitions.yaml`** — document types, field lists, min_tokens per type
- **`prompts/document_type_detection.yaml`** — detection prompts, type mappings, fallback_type
- **`prompts/internvl3_prompts.yaml`** / **`prompts/llama_prompts.yaml`** — extraction prompts
- **`config/bank_prompts.yaml`** — bank statement extraction prompts and patterns
- **`models/registry.py`** — model registration (prompt_file, loader, processor_creator)

### Rules
- **No hardcoded model paths** — use `run_config.yml:model.path` or `model_loading.default_paths`
- **No hardcoded YAML filenames** — derive from registry `prompt_file` or `prompt_config`
- **No hardcoded fallback types** — read from detection YAML `settings.fallback_type`
- **No hardcoded token limits or thresholds** — put in `run_config.yml` or `field_definitions.yaml`
- **No hardcoded field lists** — always load from `field_definitions.yaml`
- **Magic numbers need a config home** — if a number controls behaviour, it belongs in YAML
- **Defaults in code are OK only as last-resort fallbacks** — and must match the YAML value with a comment referencing the YAML source

### Pattern
```python
# GOOD: Read from config with documented fallback
max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)  # see detection YAML

# BAD: Hardcoded magic number
max_tokens = 50
```

## Python 3.12.12 Standards (MANDATORY)

This project targets **Python 3.12.12**. All code MUST use modern 3.12 idioms.

### Banned patterns — NEVER use these
| Banned | Use instead |
|--------|-------------|
| `from __future__ import annotations` | Not needed — 3.12 evaluates `X \| Y` natively |
| `from typing import Optional` | `X \| None` |
| `from typing import Union` | `X \| Y` |
| `from typing import List, Dict, Tuple, Set` | `list`, `dict`, `tuple`, `set` (builtin generics) |
| `from typing import TYPE_CHECKING` + guarded imports | Import directly — no lazy type-only blocks needed |
| `typing.Optional[X]` | `X \| None` |
| `typing.Union[X, Y]` | `X \| Y` |
| bare `except Exception: pass` | Catch specific exceptions; log in debug mode |

### Required patterns
- **Builtin generics everywhere**: `list[str]`, `dict[str, Any]`, `tuple[int, ...]`, `set[Path]`
- **Union syntax**: `str | None`, `int | float`, `Path | str`
- **`typing.override`**: On all methods that override an abstract or parent method
- **`type` keyword for aliases**: `type Vector = list[float]` (where appropriate)
- **`match` statements**: Prefer over `if/elif` chains for type/value dispatch (where cleaner)
- **Exception groups** (`ExceptionGroup`): Use when handling multiple concurrent errors
- **f-strings**: Always — no `%` formatting or `.format()`

### Example
```python
# GOOD — 3.12 style
def process(items: list[str], config: dict[str, Any] | None = None) -> tuple[int, str]:
    ...

type DocFields = dict[str, list[str]]

# BAD — pre-3.10 style
from typing import List, Dict, Optional, Tuple
def process(items: List[str], config: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    ...
```

## Code Quality

```bash
# MANDATORY before commits
ruff check *.py common/*.py models/*.py
ruff check . --fix
ruff format .
```

## Git Commit Standards

```
<gitmoji> <type>: <description>
```

Examples:
- `✨ feat:` - New features
- `🐛 fix:` - Bug fixes
- `📝 docs:` - Documentation
- `♻️ refactor:` - Code restructuring

**NO Claude attributions in commits.**
