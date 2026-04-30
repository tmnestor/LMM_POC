# Plan: Make `enforce_eager` Configurable

**Branch:** `feature/vllm-data-parallel`
**Goal:** Expose `enforce_eager` (vLLM CUDA graph toggle) through the standard
config cascade — YAML → ENV → CLI — so it can be disabled without touching
source code, enabling CUDA graph compilation for DP throughput experiments.

**Background:** `enforce_eager=True` was hardcoded to avoid a KFP step
deadline timeout caused by CUDA graph compilation (~55 min → ~4 min startup).
In DP mode each worker processes ~49 images, so compilation amortises quickly
and disabling `enforce_eager` may improve per-image throughput.

---

## Changes

### 1. `common/pipeline_config.py`

**a) Add field to `PipelineConfig`** — mirror the `flash_attn` pattern:

```python
# Model options
flash_attn: bool = True
enforce_eager: bool = True   # vLLM only: True = skip CUDA graph compilation
```

**b) Read from YAML in `load_yaml_config`** — under the `model:` block:

```python
flat_config["flash_attn"]     = raw_config["model"].get("flash_attn")
flat_config["enforce_eager"]  = raw_config["model"].get("enforce_eager")  # ADD
```

**c) Add ENV var in `load_env_config`**:

```python
f"{ENV_PREFIX}ENFORCE_EAGER": ("enforce_eager", lambda x: x.lower() == "true"),
```

(`ENV_PREFIX` = `"IVL_"`, so the var is `IVL_ENFORCE_EAGER=true/false`)

---

### 2. `models/model_loader.py`

**a) Use `cfg.enforce_eager` instead of hardcoded `True`:**

```python
# Before
"enforce_eager": True,

# After
"enforce_eager": cfg.enforce_eager,
```

**b) Log the setting** so it's visible at startup (after the existing
`console.print(f"Model path: ...")` line):

```python
console.print(
    f"[dim]enforce_eager: {cfg.enforce_eager} "
    f"({'skip CUDA graphs' if cfg.enforce_eager else 'CUDA graphs enabled'})[/dim]"
)
```

---

### 3. `config/run_config.yml`

Add under the `model:` section, next to `flash_attn`:

```yaml
model:
  ...
  flash_attn: true       # Use Flash Attention 2 (disable for V100)
  enforce_eager: true    # vLLM: true = skip CUDA graph compilation (faster startup,
                         #        lower throughput); false = compile graphs (slower
                         #        startup, higher throughput for long runs)
```

---

### 4. `cli.py`

**a) Add CLI option** — mirror `--flash-attn/--no-flash-attn`:

```python
enforce_eager: bool | None = typer.Option(
    None,
    "--enforce-eager/--no-enforce-eager",
    help="vLLM: skip CUDA graph compilation. Default from config or True.",
),
```

**b) Add to `cli_args` dict** (in the `run` function, alongside `flash_attn`):

```python
"enforce_eager": enforce_eager,
```

**c) Add to the startup config table** (alongside the Flash attention row):

```python
config_table.add_row("Enforce eager", str(config.enforce_eager))
```

---

## File Summary

| File | Change |
|------|--------|
| `common/pipeline_config.py` | Add `enforce_eager` field, YAML read, ENV var |
| `models/model_loader.py` | Use `cfg.enforce_eager`, log the value |
| `config/run_config.yml` | Add `enforce_eager: true` with comment |
| `cli.py` | Add `--enforce-eager/--no-enforce-eager` option, cli_args, table row |

---

## Verification

After implementing, test the cascade on the sandbox:

```bash
# Default (enforce_eager=True from YAML) — should behave as today
python cli.py --model internvl3-vllm ...

# Disable via CLI — CUDA graphs should compile
python cli.py --model internvl3-vllm --no-enforce-eager ...

# Disable via ENV
IVL_ENFORCE_EAGER=false python cli.py --model internvl3-vllm ...
```

Log should show:
```
enforce_eager: False (CUDA graphs enabled)
```

On the prod 4-GPU run, compare throughput (images/min) with and without
`enforce_eager` to quantify the compilation trade-off.
