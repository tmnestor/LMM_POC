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

## Python 3.12 Standards (MANDATORY)

This project targets **Python 3.12.12**. All code MUST use 3.12 best practices:

- **No `from __future__ import annotations`** — 3.12 supports `X | Y` union syntax natively
- **Use `typing.override`** on all methods that override an abstract or parent method
- **Use builtin generics** — `list[str]`, `dict[str, Any]`, `tuple[int, ...]` (not `List`, `Dict`, `Tuple`)
- **Use `X | Y`** for unions (not `Union[X, Y]`) and `X | None` for optionals (not `Optional[X]`)
- **Use `type` keyword** for type aliases where appropriate: `type Vector = list[float]`
- **Specific exception types** — never bare `except Exception: pass`; catch specific exceptions and log in debug mode

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
