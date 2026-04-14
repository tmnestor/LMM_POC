# Config Unification Refactor: Eliminate Global Mutation

## Problem Statement

Configuration is scattered across three modules with different loading patterns:

| Module | Pattern | Globals | Problem |
|--------|---------|---------|---------|
| `model_config.py` | Mutable module globals | 13 | `apply_yaml_overrides()` mutates them as a side effect; ordering matters |
| `field_config.py` | Lazy-init module globals | 13 | Initialized at import time; `field_definitions.yaml` read again separately by model_config |
| `pipeline_config.py` | Immutable dataclass | 0 | Clean design, but returns `raw_config` that must be fed to `apply_yaml_overrides` separately |

**Symptoms:**
- `field_definitions.yaml` read 3-4 times across modules with separate caches
- CLI startup requires a 7-step orchestration dance across 3 imports
- `apply_yaml_overrides()` must be called before processor construction, but nothing enforces this
- Silent exception swallowing in `model_config._get_min_tokens_for_type()`
- No way to construct a fully-resolved config for tests without triggering module-level side effects

---

## Design: AppConfig (Hybrid C + selective B)

### Core Idea

A single `AppConfig.load(cli_args, config_path)` call replaces the 7-step dance.
Two new frozen dataclasses (`BatchSettings`, `FieldSchema`) replace 26 mutable module globals.
All existing function signatures (`get_generation_config`, `get_auto_batch_size`, field accessors) survive as methods on `AppConfig`.

### Class Hierarchy

```
AppConfig                         # Constructed once, threaded everywhere
  .pipeline: PipelineConfig       # UNCHANGED dataclass (paths, flags, model_type)
  .batch: BatchSettings           # FROZEN dataclass (replaces 13 model_config globals)
  .fields: FieldSchema            # FROZEN dataclass (replaces 13 field_config globals)
  ._generation_registry: dict     # Built from _GENERATION_CONFIG_REGISTRY + YAML overrides

  .load(cli_args, config_path)    # Single entry point — no side effects
  .get_generation_config(model_type) -> dict
  .get_auto_batch_size(model_name, available_memory_gb) -> int
  .get_max_new_tokens(field_count, document_type) -> int
```

---

## 1. Interface Signature

```python
# common/app_config.py

@dataclass(frozen=True)
class BatchSettings:
    """Typed replacement for the 13 mutable batch/GPU globals in model_config.py."""

    default_sizes: dict[str, int]
    max_sizes: dict[str, int]
    conservative_sizes: dict[str, int]
    min_size: int = 1
    strategy: str = "balanced"
    auto_detect: bool = True
    memory_safety_margin: float = 0.8
    clear_cache_after_batch: bool = True
    timeout_seconds: int = 300
    fallback_enabled: bool = True
    fallback_steps: tuple[int, ...] = (8, 4, 2, 1)
    gpu_memory_thresholds: dict[str, int] = field(
        default_factory=lambda: {"low": 8, "medium": 16, "high": 24, "very_high": 64}
    )

    @classmethod
    def from_raw(cls, raw_config: dict) -> "BatchSettings":
        """Build from raw YAML config (replaces apply_yaml_overrides batch/gpu sections)."""


@dataclass(frozen=True)
class FieldSchema:
    """Typed replacement for the 13 lazy globals in field_config.py.

    Loaded once from field_definitions.yaml via SimpleFieldLoader.
    """

    extraction_fields: tuple[str, ...]
    field_count: int
    field_types: dict[str, str]
    monetary_fields: tuple[str, ...]
    date_fields: tuple[str, ...]
    list_fields: tuple[str, ...]
    boolean_fields: tuple[str, ...]
    calculated_fields: tuple[str, ...]
    transaction_list_fields: tuple[str, ...]
    text_fields: tuple[str, ...]
    phone_fields: tuple[str, ...]
    numeric_id_fields: tuple[str, ...]
    validation_only_fields: tuple[str, ...] = ("TRANSACTION_AMOUNTS_RECEIVED", "ACCOUNT_BALANCE")

    @classmethod
    def from_yaml(cls) -> "FieldSchema":
        """Load from field_definitions.yaml (single read, single cache)."""

    def is_evaluation_field(self, field_name: str) -> bool:
        """Check if a field should be included in evaluation metrics."""

    def filter_evaluation_fields(self, fields: list[str]) -> list[str]:
        """Filter a list to exclude validation-only fields."""

    def get_document_type_fields(self, document_type: str) -> list[str]:
        """Get fields specific to a document type, filtered for evaluation."""


class ConfigError(Exception):
    """Raised by AppConfig.load() when validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class AppConfig:
    """Unified, immutable configuration surface.

    Constructed once at startup, threaded to all consumers.
    Replaces the 7-step config dance in cli.py and eliminates
    all mutable module globals in model_config and field_config.
    """

    def __init__(
        self,
        pipeline: PipelineConfig,
        batch: BatchSettings,
        fields: FieldSchema,
        generation_registry: dict[str, dict],
    ) -> None:
        self.pipeline = pipeline
        self.batch = batch
        self.fields = fields
        self._generation_registry = generation_registry

    @classmethod
    def load(
        cls,
        cli_args: dict[str, Any],
        *,
        config_path: Path | None = None,
    ) -> "AppConfig":
        """Single entry point. No side effects. No globals mutated.

        Handles: YAML loading, ENV loading, merge (CLI > YAML > ENV > defaults),
        validation, batch settings, generation registry, field schema.

        Raises:
            ConfigError: On validation failure.
        """

    # -- Drop-in replacements for model_config functions -----------------------

    def get_generation_config(self, model_type: str) -> dict[str, Any]:
        """Same signature as model_config.get_generation_config().

        Returns a copy so callers can mutate freely.
        """

    def get_auto_batch_size(
        self, model_name: str, available_memory_gb: float | None = None
    ) -> int:
        """Same signature as model_config.get_auto_batch_size()."""

    def get_max_new_tokens(
        self, field_count: int | None = None, document_type: str | None = None
    ) -> int:
        """Same signature as model_config.get_max_new_tokens()."""
```

---

## 2. Usage: How cli.py Simplifies

### Before (7 steps, 2 side effects, 3 imports)

```python
from common.pipeline_config import load_yaml_config, load_env_config, merge_configs, validate_config
from common.model_config import apply_yaml_overrides

yaml_config, raw_config = load_yaml_config(config_path)       # step 1
apply_yaml_overrides(raw_config)                               # step 2 (SIDE EFFECT!)
env_config = load_env_config()                                 # step 3
config = merge_configs(cli_args, yaml_config, env_config, raw_config)  # step 4
errors = validate_config(config)                               # step 5
if errors: ...                                                 # step 6
# ... later in run_pipeline():
prompt_config, fields, field_defs = load_pipeline_configs(config.model_type)  # step 7
```

### After (1 call, 0 side effects)

```python
from common.app_config import AppConfig, ConfigError

try:
    app = AppConfig.load(cli_args, config_path=resolved_config)
except ConfigError as e:
    for error in e.errors:
        console.print(f"[red]FATAL: {error}[/red]")
    raise typer.Exit(EXIT_CONFIG_ERROR) from None

config = app.pipeline  # PipelineConfig, unchanged
run_pipeline(app)      # pass AppConfig, not just PipelineConfig
```

### Processor construction (before vs after)

```python
# BEFORE: base_processor.py reads from module globals
from common.model_config import get_auto_batch_size
self.batch_size = get_auto_batch_size(model_type_key, available_memory)

# AFTER: reads from AppConfig instance (same signature)
self.batch_size = app_config.get_auto_batch_size(model_type_key, available_memory)
```

```python
# BEFORE: simple_processor.py reads from module globals
from common.model_config import get_generation_config
self.gen_config = get_generation_config(self._effective_model_type_key)

# AFTER: reads from AppConfig instance (same signature)
self.gen_config = app_config.get_generation_config(self._effective_model_type_key)
```

---

## 3. What AppConfig.load() Hides Internally

```python
@classmethod
def load(cls, cli_args, *, config_path=None):
    # 1. Resolve config path
    default = Path(__file__).parent.parent / "config" / "run_config.yml"
    resolved = config_path or (default if default.exists() else None)

    # 2. Load YAML
    yaml_config, raw_config = {}, {}
    if resolved:
        yaml_config, raw_config = load_yaml_config(resolved)

    # 3. Load ENV
    env_config = load_env_config()

    # 4. Merge with precedence: CLI > YAML > ENV > defaults
    pipeline = merge_configs(cli_args, yaml_config, env_config, raw_config)

    # 5. Validate
    errors = validate_config(pipeline)
    if errors:
        raise ConfigError(errors)

    # 6. Build BatchSettings (replaces apply_yaml_overrides batch/gpu sections)
    batch = BatchSettings.from_raw(raw_config)

    # 7. Build generation registry (replaces YAML mutation of INTERNVL3_GENERATION_CONFIG)
    generation_registry = _build_generation_registry(raw_config)

    # 8. Load field schema (single YAML read, replaces field_config globals)
    fields = FieldSchema.from_yaml()

    return cls(pipeline=pipeline, batch=batch, fields=fields,
               generation_registry=generation_registry)
```

Key internal details:

- **`BatchSettings.from_raw()`**: Reads `raw_config["batch"]` and `raw_config["gpu"]`, applies same
  merge logic as `apply_yaml_overrides`, returns frozen dataclass. The 13 mutable globals cease to exist.

- **`_build_generation_registry()`**: Takes `_GENERATION_CONFIG_REGISTRY` as base, applies YAML overrides
  from `raw_config["generation"]`, returns `dict[str, dict]`. Stored immutably on AppConfig.

- **`FieldSchema.from_yaml()`**: Calls `SimpleFieldLoader()` once, builds frozen dataclass.
  Replaces `field_config._ensure_fields_loaded()` and the 13 globals it populates.

---

## 4. Threading AppConfig Through Processors

The main change is threading `AppConfig` (or just the parts it needs) into processor constructors.

### Option A: Thread whole AppConfig (simpler, fewer changes)

```python
# SimpleDocumentProcessor.__init__ gains app_config parameter
class SimpleDocumentProcessor(BaseDocumentProcessor):
    def __init__(self, ..., app_config: AppConfig | None = None):
        ...
        # _init_shared uses app_config for batch size
        # _configure_generation uses app_config for gen config

# BaseDocumentProcessor._init_shared gains app_config
def _init_shared(self, ..., app_config: AppConfig | None = None):
    if app_config and batch_size is None:
        self.batch_size = app_config.get_auto_batch_size(model_type_key, available_memory)
    ...
```

### Option B: Pass resolved values directly (no new dependency on AppConfig)

```python
# Registry creator functions resolve config BEFORE constructing the processor
# cli.py / registry.py:
gen_config = app.get_generation_config(model_type)
batch_size = app.get_auto_batch_size(model_type, available_memory)

processor = DocumentAwareQwen3VLProcessor(
    ...,
    batch_size=batch_size,       # already resolved
    generation_config=gen_config, # already resolved
)
```

**Recommendation: Option A for Phase 1** (fewer file changes, backward compatible via `app_config=None`
defaulting to current global reads). Phase 2 can tighten to Option B if desired.

---

## 5. Backward Compatibility: field_config Shims

Evaluation code (`evaluation_metrics.py`, `simple_model_evaluator.py`) calls `field_config.get_monetary_fields()` etc.
These functions continue to work unchanged in Phase 1 -- they still lazy-init from YAML.

In Phase 2, they become one-line shims delegating to a module-level `_app_config` singleton:

```python
# field_config.py (Phase 2 -- thin shims)
_app_config: AppConfig | None = None

def _init(app_config: AppConfig) -> None:
    global _app_config
    _app_config = app_config

def get_monetary_fields():
    if _app_config:
        return list(_app_config.fields.monetary_fields)
    _ensure_initialized()  # legacy fallback
    return MONETARY_FIELDS
```

---

## 6. Migration Path

### Phase 1: Add AppConfig, wire CLI (this PR)

| File | Change | LOC |
|------|--------|-----|
| `common/app_config.py` | **NEW**: `AppConfig`, `BatchSettings`, `FieldSchema`, `ConfigError` | +250 |
| `cli.py` | Replace 7-step dance with `AppConfig.load()`, change `run_pipeline(config)` to `run_pipeline(app)` | ~-25, +10 |
| `models/base_processor.py` | Add `app_config: AppConfig | None = None` to `_init_shared`, use for batch size if provided | +8 |
| `models/simple_processor.py` | Add `app_config` param, pass to `_init_shared` and `_configure_generation` | +6 |
| `models/registry.py` | Thread `app_config` through processor creator functions (already accept `**kwargs`-ish patterns) | +15 |
| `common/model_config.py` | **NO CHANGES** -- globals remain as fallback defaults, `apply_yaml_overrides` still exists but no longer called | 0 |
| `common/field_config.py` | **NO CHANGES** -- lazy init still works for evaluation code | 0 |

**Validation**: Import smoke test + existing evaluation runs produce identical results.

### Phase 2: Delete globals (follow-up PR)

| File | Change |
|------|--------|
| `common/model_config.py` | Remove `apply_yaml_overrides()`, remove 13 mutable globals, keep `_GENERATION_CONFIG_REGISTRY` and `get_generation_config()` as thin re-exports from AppConfig |
| `common/field_config.py` | Remove 13 mutable globals + `_ensure_fields_loaded()`, replace accessor functions with shims delegating to `AppConfig.fields` |
| `common/model_config.py` | Remove `_MIN_TOKENS_CACHE` and `_get_min_tokens_for_type()` -- absorbed into `AppConfig.get_max_new_tokens()` |
| All processors | Remove `app_config=None` default, make it required |

---

## 7. What NOT To Do

- **Do NOT make AppConfig a singleton at module level.** Pass it explicitly. Singletons recreate
  the same "when was it initialized?" problem we're fixing.

- **Do NOT delete field_config.py accessor functions in Phase 1.** Evaluation code depends on them
  and doesn't touch batch/generation config. Fix the dangerous global mutation first.

- **Do NOT merge PipelineConfig into AppConfig.** PipelineConfig is a clean dataclass -- AppConfig
  *composes* it, not replaces it. `app.pipeline.data_dir` reads well.

- **Do NOT add a ConfigSource protocol.** There's zero evidence of non-YAML config sources.
  Add extensibility when a real use case appears.

- **Do NOT turn generation configs into typed dataclasses.** The `dict[str, Any]` pattern was just
  established with `get_generation_config()` and works fine. Typing the dicts adds ceremony for
  configs that differ per model (some have `temperature`, some don't).

---

## 8. File Impact Summary

| File | Phase 1 | Phase 2 |
|------|---------|---------|
| `common/app_config.py` | **NEW** (~250 LOC) | Unchanged |
| `cli.py` | Simplify main() (~-15 net) | Unchanged |
| `models/base_processor.py` | Add `app_config` param (+8) | Make required |
| `models/simple_processor.py` | Add `app_config` param (+6) | Make required |
| `models/registry.py` | Thread `app_config` (+15) | Unchanged |
| `common/model_config.py` | Unchanged | Remove globals + `apply_yaml_overrides` (-80) |
| `common/field_config.py` | Unchanged | Replace with shims (-100) |
| `common/pipeline_config.py` | Unchanged | Unchanged |

**Phase 1 net**: +250 new, -15 removed = +235 LOC (new capability)
**Phase 2 net**: -180 LOC (cleanup)
**Total**: +55 LOC net, but 26 mutable globals eliminated and `field_definitions.yaml` read once instead of 3-4 times.

---

## 9. Trade-offs

### Benefits

- **CLI startup: 7 steps -> 1.** Most common caller drops ~25 lines of boilerplate.
- **Zero mutable module globals.** `apply_yaml_overrides` mutating 13 globals was the most dangerous pattern -- any import before the call got stale defaults.
- **Single YAML read.** `field_definitions.yaml` loaded once instead of 3-4 times.
- **Testable.** `AppConfig.load({"data_dir": ..., "output_dir": ...})` works in tests without module-level side effects.
- **Discoverable.** One type to find all config. IDE autocomplete on `app.` shows everything.
- **Immutable.** Frozen dataclasses prevent accidental mutation of batch sizes or field lists.

### Costs

- **Signature change for `run_pipeline`.** `run_pipeline(config)` -> `run_pipeline(app)`. ~4 files affected.
- **Processor constructors gain `app_config` param.** Optional in Phase 1, so backward compatible.
- **Two-phase migration.** `field_config` globals survive in Phase 1 for evaluation code compatibility.
- **+250 LOC new file.** But it replaces 260+ LOC of scattered globals and side-effect functions.
