# Configuration Unification: Interface Design

## 1. Interface Signature

### Core: `AppConfig` -- The Single Resolved Configuration Object

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self


@dataclass(frozen=True)
class GenerationConfig:
    """Per-model generation hyper-parameters (immutable after construction)."""

    max_new_tokens_base: int = 512
    max_new_tokens_per_field: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 0.95
    use_cache: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class BatchConfig:
    """Batch processing parameters (immutable after construction)."""

    default_sizes: dict[str, int] = field(default_factory=lambda: {"internvl3": 4, "qwen3vl": 4})
    max_sizes: dict[str, int] = field(default_factory=lambda: {"internvl3": 8, "qwen3vl": 8})
    conservative_sizes: dict[str, int] = field(default_factory=lambda: {"internvl3": 1, "qwen3vl": 2})
    min_size: int = 1
    strategy: str = "balanced"            # conservative | balanced | aggressive
    auto_detect: bool = True
    memory_safety_margin: float = 0.8
    clear_cache_after_batch: bool = True
    timeout_seconds: int = 300
    fallback_enabled: bool = True
    fallback_steps: list[int] = field(default_factory=lambda: [8, 4, 2, 1])


@dataclass(frozen=True)
class GpuConfig:
    """GPU memory thresholds for automatic batch sizing."""

    memory_thresholds: dict[str, int] = field(
        default_factory=lambda: {"low": 8, "medium": 16, "high": 24, "very_high": 64}
    )


@dataclass(frozen=True)
class FieldSchema:
    """Resolved field definitions and type classifications.

    Loaded once from field_definitions.yaml at construction time.
    """

    extraction_fields: list[str]
    field_count: int
    field_types: dict[str, str]           # field_name -> "text" | "monetary" | ...
    monetary_fields: list[str]
    date_fields: list[str]
    list_fields: list[str]
    boolean_fields: list[str]
    calculated_fields: list[str]
    transaction_list_fields: list[str]
    validation_only_fields: list[str]
    document_fields: dict[str, list[str]]  # doc_type -> [field_names]
    supported_document_types: list[str]

    def fields_for_type(self, document_type: str) -> list[str]:
        """Get evaluation-eligible fields for a document type."""
        ...

    def is_evaluation_field(self, field_name: str) -> bool:
        """True if not in validation_only_fields."""
        ...


@dataclass
class PipelineConfig:
    """Pipeline runtime config -- the existing dataclass, preserved."""
    # (unchanged from current pipeline_config.py)
    data_dir: Path
    output_dir: Path
    model_path: Path | None = None
    ground_truth: Path | None = None
    max_images: int | None = None
    document_types: list[str] | None = None
    batch_size: int | None = None
    bank_v2: bool = True
    balance_correction: bool = True
    model_type: str = "internvl3"
    max_tiles: int = 11
    min_tiles: int | None = None
    flash_attn: bool = True
    dtype: str = "bfloat16"
    max_new_tokens: int = 2000
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = False
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"
    num_gpus: int = 0
    skip_visualizations: bool = False
    skip_reports: bool = False
    verbose: bool = True
    timestamp: str = field(default_factory=...)


@dataclass(frozen=True)
class AppConfig:
    """The single fully-resolved configuration for the application.

    Immutable after construction. Every consumer receives this or
    a sub-object from it -- no module globals, no mutation.
    """

    pipeline: PipelineConfig
    generation: dict[str, GenerationConfig]   # model_type -> GenerationConfig
    batch: BatchConfig
    gpu: GpuConfig
    fields: FieldSchema

    # --- Convenience accessors (preserve existing call-site signatures) ---

    def get_generation_config(self, model_type: str) -> dict[str, Any]:
        """Drop-in replacement for model_config.get_generation_config().

        Returns a mutable dict copy so callers can modify without affecting AppConfig.
        """
        ...

    def get_auto_batch_size(self, model_name: str, available_memory_gb: float | None = None) -> int:
        """Drop-in replacement for model_config.get_auto_batch_size()."""
        ...

    def get_max_new_tokens(self, field_count: int | None = None, document_type: str | None = None) -> int:
        """Drop-in replacement for model_config.get_max_new_tokens()."""
        ...

    # Field accessors (replace field_config module-level functions)
    def get_monetary_fields(self) -> list[str]: ...
    def get_date_fields(self) -> list[str]: ...
    def get_list_fields(self) -> list[str]: ...
    def get_boolean_fields(self) -> list[str]: ...
    def get_calculated_fields(self) -> list[str]: ...
    def get_transaction_list_fields(self) -> list[str]: ...
    def get_document_type_fields(self, document_type: str) -> list[str]: ...
```

### Builder: `ConfigBuilder` -- The Single Entry Point

```python
class ConfigSource(Protocol):
    """Protocol for pluggable config sources (future: API, database, etc.)."""

    def load(self) -> dict[str, Any]:
        """Return a flat or nested dict of config values."""
        ...

    @property
    def priority(self) -> int:
        """Higher number = higher priority. CLI=400, YAML=300, ENV=200, defaults=100."""
        ...


class ConfigBuilder:
    """Builds an AppConfig from layered sources.

    Enforces: CLI > YAML > ENV > defaults.
    Validates eagerly. Fails fast with actionable messages.
    """

    def __init__(self) -> None:
        self._sources: list[ConfigSource] = []
        self._overrides: dict[str, Any] = {}

    # --- Fluent source registration ---

    def with_yaml(self, path: Path) -> Self:
        """Add a YAML file as a config source (priority 300)."""
        ...

    def with_env(self, prefix: str = "IVL_") -> Self:
        """Add environment variables as a config source (priority 200)."""
        ...

    def with_cli(self, args: dict[str, Any]) -> Self:
        """Add CLI arguments as a config source (priority 400)."""
        ...

    def with_source(self, source: ConfigSource) -> Self:
        """Add a custom config source (API, database, etc.)."""
        ...

    def with_overrides(self, **kwargs: Any) -> Self:
        """Programmatic overrides (priority 500 -- highest)."""
        ...

    def with_field_definitions(self, *paths: Path) -> Self:
        """Load field definitions from one or more YAML files.

        Multiple files are merged (later files override earlier).
        Default: config/field_definitions.yaml
        """
        ...

    def with_generation_config(self, model_type: str, **kwargs: Any) -> Self:
        """Override generation config for a specific model type."""
        ...

    # --- Build ---

    def build(self) -> AppConfig:
        """Merge all sources, validate, and return an immutable AppConfig.

        Raises:
            ConfigurationError: With actionable message listing what's wrong
                and how to fix it.
        """
        ...

    # --- Convenience class methods ---

    @classmethod
    def from_cli(
        cls,
        cli_args: dict[str, Any],
        config_file: Path | None = None,
    ) -> AppConfig:
        """One-call factory for CLI usage (the common case).

        Equivalent to:
            ConfigBuilder()
                .with_env()
                .with_yaml(config_file)
                .with_cli(cli_args)
                .build()
        """
        ...

    @classmethod
    def for_testing(
        cls,
        data_dir: Path = Path("/tmp/test_data"),
        output_dir: Path = Path("/tmp/test_output"),
        **overrides: Any,
    ) -> AppConfig:
        """Minimal config for unit tests. No file I/O, no env, no YAML."""
        ...

    @classmethod
    def for_notebook(
        cls,
        data_dir: Path,
        output_dir: Path,
        model_type: str = "internvl3",
        config_file: Path | None = None,
        **overrides: Any,
    ) -> AppConfig:
        """Notebook-friendly factory with sensible defaults."""
        ...
```

---

## 2. Usage Examples

### CLI (cli.py -- unchanged call sites, different wiring)

```python
# Before (current cli.py lines 878-894):
yaml_config, raw_config = load_yaml_config(resolved_config)
if raw_config:
    from common.model_config import apply_yaml_overrides
    apply_yaml_overrides(raw_config)
env_config = load_env_config()
config = merge_configs(cli_args, yaml_config, env_config, raw_config)

# After:
from common.config import ConfigBuilder

app_config = ConfigBuilder.from_cli(cli_args, config_file=resolved_config)
config = app_config.pipeline  # PipelineConfig -- same as before

# Downstream consumers receive app_config or its sub-objects:
processor = create_processor(model, tokenizer, app_config, ...)
```

### Notebook

```python
from common.config import ConfigBuilder
from pathlib import Path

app_config = ConfigBuilder.for_notebook(
    data_dir=Path("../evaluation_data/synthetic"),
    output_dir=Path("./notebook_output"),
    model_type="qwen3vl",
    batch_size=2,
    verbose=True,
)

# Override generation params for experimentation:
app_config = (
    ConfigBuilder()
    .with_yaml(Path("config/run_config.yml"))
    .with_generation_config("qwen3vl", temperature=0.3, do_sample=True)
    .with_overrides(
        data_dir=Path("../evaluation_data/synthetic"),
        output_dir=Path("./notebook_output"),
    )
    .build()
)

# Access anything:
gen = app_config.get_generation_config("qwen3vl")
fields = app_config.fields.fields_for_type("invoice")
```

### Tests

```python
from common.config import ConfigBuilder

def test_invoice_extraction():
    app_config = ConfigBuilder.for_testing(
        model_type="internvl3",
        max_new_tokens=500,
    )
    # No YAML read, no env pollution, deterministic
    assert app_config.fields.field_count > 0
    assert app_config.get_generation_config("internvl3")["do_sample"] is False

def test_custom_fields():
    """Test with field definitions from a fixture file."""
    app_config = (
        ConfigBuilder()
        .with_field_definitions(Path("tests/fixtures/custom_fields.yaml"))
        .with_overrides(
            data_dir=Path("/tmp/test"),
            output_dir=Path("/tmp/out"),
        )
        .build()
    )
    assert "CUSTOM_FIELD" in app_config.fields.extraction_fields
```

### Evaluation code (backwards-compatible shim)

```python
# common/field_config.py becomes a thin shim:

from common.config import get_app_config  # module-level singleton (see section 4)

def get_monetary_fields():
    return get_app_config().get_monetary_fields()

def get_document_type_fields(document_type: str) -> list:
    return get_app_config().get_document_type_fields(document_type)

# ... etc. All existing call sites keep working.
```

---

## 3. What Complexity It Hides Internally

### 3a. Source merging and precedence

The builder collects sources into a priority-sorted list, then merges dicts
right-to-left (lowest priority first). The nested YAML structure
(`batch.default_sizes`, `generation.max_new_tokens_base`, `gpu.memory_thresholds`)
is handled by a single internal `_flatten()` / `_unflatten()` pass -- callers never
see the nested-vs-flat impedance mismatch that currently forces `load_yaml_config`
to return both `flat_config` and `raw_config`.

### 3b. Field definitions loading

`FieldSchema` is built from `field_definitions.yaml` exactly once during
`build()`. The current codebase reads this file in four separate places:
- `field_config._get_config()` (via `SimpleFieldLoader`)
- `field_config.get_document_type_fields()` (creates a NEW `SimpleFieldLoader`)
- `model_config._get_min_tokens_for_type()` (raw `yaml.safe_load`)
- `cli.py:load_prompt_config()` (raw `yaml.safe_load` for `supported_document_types`)

All four collapse into the single `FieldSchema` construction inside `build()`.

### 3c. Generation config registry

Currently split across:
- `INTERNVL3_GENERATION_CONFIG`, `LLAMA_GENERATION_CONFIG`, etc. (module dicts)
- `_GENERATION_CONFIG_REGISTRY` (lookup dict mapping model_type -> one of the above)
- `GENERATION_CONFIGS` (separate dict for YAML overrides, only covers internvl3)
- `INTERNVL3_TOKEN_LIMITS` (another separate dict)

All of this collapses into `AppConfig.generation: dict[str, GenerationConfig]`.
The `GenerationConfig` dataclass replaces untyped dicts, so typos like
`gen_config["masx_new_tokens_base"]` become `AttributeError` at attribute access
time instead of silently returning `None`.

### 3d. Global mutation

`apply_yaml_overrides()` currently mutates 13 module globals. After this
refactor, there are zero mutable globals. `AppConfig` is `frozen=True`.
If you need a variant, use `dataclasses.replace(app_config.batch, strategy="aggressive")`.

### 3e. Validation

Currently scattered: `validate_config()` checks paths and dtypes in
`pipeline_config.py`, but batch/generation/field errors silently pass.
`build()` validates everything in one pass and raises `ConfigurationError`
with all problems at once (not just the first).

### 3f. YAML-to-dataclass mapping

The manual `if "trust_remote_code" in ml: flat_config["trust_remote_code"] = ml[...]`
pattern (30+ lines in `load_yaml_config`) is replaced by a declarative mapping
inside the builder that uses the dataclass field names directly.

---

## 4. Dependency Strategy

### Module-level singleton (for backwards compatibility)

```python
# common/config.py (bottom of module)

_app_config: AppConfig | None = None

def init_app_config(config: AppConfig) -> None:
    """Called once from CLI startup or notebook setup."""
    global _app_config
    _app_config = config

def get_app_config() -> AppConfig:
    """Get the initialized AppConfig.

    Raises ConfigurationError if init_app_config() was never called.
    """
    if _app_config is None:
        raise ConfigurationError(
            "Configuration not initialized. Call init_app_config() first.\n"
            "  CLI: this happens automatically in main()\n"
            "  Notebook: use ConfigBuilder.for_notebook(...)\n"
            "  Tests: use ConfigBuilder.for_testing(...)"
        )
    return _app_config
```

### Dependency flow

```
cli.py main()
    |
    v
ConfigBuilder.from_cli(cli_args, config_file)
    |
    +-- reads run_config.yml (once)
    +-- reads field_definitions.yaml (once)
    +-- reads env vars (once)
    +-- merges, validates
    |
    v
AppConfig (frozen)
    |
    +-- init_app_config(app_config)      # set module singleton
    |
    +-- app_config.pipeline              # -> PipelineConfig (passed to run_pipeline)
    +-- app_config.generation            # -> dict[str, GenerationConfig]
    +-- app_config.batch                 # -> BatchConfig
    +-- app_config.fields                # -> FieldSchema
    |
    v
consumers read from AppConfig (never from globals)
```

### What changes per consumer

| Consumer | Current dependency | New dependency |
|----------|-------------------|----------------|
| `cli.py` | `load_yaml_config`, `load_env_config`, `merge_configs`, `apply_yaml_overrides`, `validate_config` | `ConfigBuilder.from_cli()` |
| `SimpleDocumentProcessor.__init__` | `get_generation_config(model_type_key)` | `app_config.get_generation_config(model_type_key)` (or receive `GenerationConfig` in constructor) |
| `base_processor._configure_batch_processing` | `get_auto_batch_size(model_type_key, mem)` | `app_config.get_auto_batch_size(model_type_key, mem)` |
| `evaluation_metrics.py` | `get_monetary_fields()`, `get_document_type_fields()`, etc. | Same functions, backed by shim that delegates to `get_app_config().fields` |

### Migration path (incremental, not big-bang)

1. **Phase 1**: Create `AppConfig`, `ConfigBuilder`, `FieldSchema` in new
   `common/config.py`. Wire `cli.py` to use `ConfigBuilder.from_cli()`.
   Keep all shim functions in `field_config.py` and `model_config.py`.
   Zero changes to evaluation_metrics.py or processors.

2. **Phase 2**: Thread `AppConfig` (or sub-objects) through processor constructors.
   Remove `get_generation_config()` global call from `SimpleDocumentProcessor`.

3. **Phase 3**: Remove shims from `field_config.py` and `model_config.py`.
   Delete the 13 mutable globals. These modules become re-exports only
   (or are deleted entirely once all imports are updated).

---

## 5. Trade-offs

### What you gain

| Benefit | Detail |
|---------|--------|
| **Single source of truth** | One `AppConfig` object, constructed once, used everywhere. No "did `apply_yaml_overrides` run yet?" questions. |
| **Immutability** | `frozen=True` dataclasses. No spooky mutation at a distance. |
| **Testability** | `ConfigBuilder.for_testing()` -- zero file I/O, zero env leakage, deterministic. |
| **Extensibility** | `ConfigSource` protocol -- add API/database/remote config sources without touching core. `with_field_definitions(*paths)` -- merge multiple YAML files. `with_generation_config()` -- per-run generation overrides. |
| **Typed generation config** | `GenerationConfig` dataclass instead of `dict[str, Any]`. IDE autocomplete, typo detection. |
| **Single YAML read** | `field_definitions.yaml` read once instead of 3-4 times. `run_config.yml` read once instead of split across `load_yaml_config` + `apply_yaml_overrides`. |
| **Eager validation** | All errors surfaced at `build()` time with actionable messages, not silently swallowed in `except: pass` blocks. |

### What you pay

| Cost | Detail |
|------|--------|
| **New module** | `common/config.py` (~300-400 LOC). Net reduction after shim deletion in phase 3, but initially additive. |
| **Module singleton** | `get_app_config()` is still a global, just a better-managed one. Pure DI (pass `AppConfig` everywhere) is cleaner but requires touching every constructor signature. The singleton is the pragmatic bridge. |
| **Frozen dataclass friction** | Can't do `app_config.batch.strategy = "aggressive"` -- must use `dataclasses.replace()`. This is intentional (prevents mutation bugs) but requires callers to adjust. |
| **Migration effort** | Three phases. Phase 1 is safe and self-contained. Phases 2-3 touch more files but are independently deployable. |
| **`GenerationConfig` may need `extra` dict** | If a model needs a parameter not in the dataclass (e.g., `pad_token_id` set dynamically from tokenizer), we need either `extra_params: dict[str, Any]` or a subclass. The `extra_params` dict keeps the dataclass open for extension without subclassing. |

### What stays the same

- `PipelineConfig` dataclass -- unchanged, just nested inside `AppConfig.pipeline`
- CLI > YAML > ENV > defaults precedence
- `get_generation_config(model_type)` return type (mutable dict copy)
- `get_auto_batch_size(model_name, memory)` return type (int)
- All field accessor function signatures in evaluation code (via shims in phase 1-2)
- `run_config.yml` format -- no YAML schema changes

### Design decisions left open

1. **Should `AppConfig` own prompt config too?** Currently `load_prompt_config()`
   is separate. It could become `AppConfig.prompts` but this couples document-type
   routing to the config layer. Recommend keeping separate for now.

2. **Should `FieldSchema` cache `SimpleFieldLoader`?** Yes -- the loader instance
   should be created once during `build()` and its data frozen into `FieldSchema`.
   The loader itself is not exposed.

3. **Thread-safety of `get_app_config()`?** The singleton is set once at startup
   and never mutated. Read-only access is safe. If multi-process (multi-GPU
   orchestrator) workers need different configs, they should construct their own
   `AppConfig` in the worker process.
