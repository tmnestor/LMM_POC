"""Guardrails for the InternVL-vLLM-only collapse refactor.

Plan: ``plans/2026-06-04-collapse-to-internvl-vllm-only.md`` (amended 2026-06-07).

Phases 1-5 of the collapse have landed, so every guard below is now a permanent
invariant (the goal-state tests that were ``xfail(strict=True)`` during the
refactor have been promoted):

* **Section A** — the live InternVL-vLLM path is intact: kept modules import on
  CPU, the registry resolves the 3 vLLM sizes, the vLLM seam builds messages and
  delegates pre-tiling to the shared tiling module, and unknown model types fail
  fast with a 4-element diagnostic.

* **Section B** — the HF inference path and all non-InternVL registrations are
  gone: no HF/.chat() or non-InternVL vLLM model types remain, the HF backend
  modules are deleted, the registry is exactly the 3 InternVL vLLM sizes, the
  dead prompt YAMLs are removed, and the generation-config schema is InternVL-only.

Local-only (``tests/`` is gitignored) and CPU-only: every kept module imports
torch/transformers/vllm lazily, so nothing here needs a GPU. The one thing
these guards CANNOT verify is that vLLM actually loads and infers on the
cluster — that remains the plan's Phase 6 GPU smoke.
"""

import importlib
import inspect
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS = REPO_ROOT / "prompts"

# The InternVL sizes the collapse kept.
KEPT_VLLM_TYPES = {"internvl3-vllm", "internvl3-14b-vllm", "internvl3-38b-vllm"}

# The three Gemma 4 registrations added after the collapse are NOT on this
# branch. The standalone screen is measured on InternVL3.5-8B only, and the
# Gemma modules went with the phase 2 deletion of everything the screen cannot
# reach. Re-adding one is a deliberate act, so this set stays closed.
POST_COLLAPSE_VLLM_TYPES: set[str] = set()

EXPECTED_VLLM_TYPES = KEPT_VLLM_TYPES | POST_COLLAPSE_VLLM_TYPES

# Non-InternVL HF (.chat()/AutoModel) registrations deleted in Phase 2.
PHASE2_DELETED_HF_TYPES = {
    "llama",
    "llama4scout",
    "granite4",
    "qwen3vl",
    "nemotron",
    "qwen35",
}
# InternVL HF (.chat()) registrations deleted in Phase 3 — distinct from the -vllm ones.
PHASE3_DELETED_HF_TYPES = {"internvl3", "internvl3-14b", "internvl3-38b"}
# Non-InternVL vLLM registrations the plan deletes.
DELETED_VLLM_TYPES = {"llama4scout-w4a16", "qwen3vl-vllm", "qwen35-vllm", "gemma4"}

# Live modules that must keep importing on CPU.
#
# This is the standalone screen's whole import surface, and the reason it is
# checked on CPU is that everything below models.registry is otherwise only
# exercised on the GPU box -- a dangling import from the phase 2 deletions
# would surface after the model has loaded, not before.
LIVE_MODULES = [
    "models.registry",
    "models.model_loader",
    "models.backend",
    "models.backends.vllm_backend",
    "models.orchestrator",
    "common.image_tiling",  # kept shared tiling primitive (Amendment 1)
    "common.pipeline_prompts",  # what cli.py's surviving half became
    "common.vllm_dp",
    "common.vllm_dp_workers",
    "stages.quality_screen",
    "stages.evaluate_quality_screen",
]

# HF backend modules the plan deletes.
DELETED_HF_BACKEND_MODULES = [
    "models.backends.hf_chat_template",
    "models.backends.internvl3",
    "models.backends.llama",
]

# Prompt YAMLs the plan deletes.
DELETED_PROMPT_FILES = [
    "llama_prompts.yaml",
    "llama4scout_prompts.yaml",
    "qwen3vl_prompts.yaml",
]


# ===========================================================================
# Section A — INVARIANTS (green now, must stay green through the refactor)
# ===========================================================================


@pytest.mark.parametrize("module", LIVE_MODULES)
def test_live_module_imports(module):
    """Every kept live module imports on CPU — catches dangling HF imports."""
    importlib.import_module(module)


def test_internvl_vllm_resolves():
    """The production default model type resolves to a vLLM registration."""
    from models import registry as registry_mod

    reg = registry_mod.get_model("internvl3-vllm")
    assert reg.is_vllm is True
    assert reg.prompt_file == "internvl3_prompts.yaml"
    assert registry_mod.is_vllm_model("internvl3-vllm") is True


@pytest.mark.parametrize("model_type", sorted(KEPT_VLLM_TYPES))
def test_three_internvl_sizes_registered(model_type):
    """All three InternVL vLLM sizes survive and point at the InternVL prompts."""
    from models import registry as registry_mod

    reg = registry_mod.get_model(model_type)
    assert reg.is_vllm is True
    assert reg.prompt_file == "internvl3_prompts.yaml"


def test_internvl_prompt_file_present_and_parses():
    """The kept prompt YAML exists and is valid YAML."""
    prompt_file = PROMPTS / "internvl3_prompts.yaml"
    assert prompt_file.is_file(), f"missing {prompt_file}"
    assert yaml.safe_load(prompt_file.read_text()), "internvl3_prompts.yaml parsed empty"


def test_vllm_seam_builds_single_image_message():
    """The kept seam builds a text-first, single-image message with no vLLM import.

    ``_build_messages`` does not trigger the lazy ``from vllm import ...`` that
    lives inside ``generate()``, so this runs on a Mac with no vllm installed.
    """
    from models.backends.vllm_backend import VllmBackend

    backend = VllmBackend(engine=object(), model_type_key="internvl3-vllm")
    messages = backend._build_messages(Image.new("RGB", (64, 64), "white"), "hello")
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert content[0]["type"] == "text"  # text-first by default
    assert sum(part["type"] == "image_url" for part in content) == 1


def test_vllm_seam_pretiling_delegates_to_image_tiling():
    """Pre-tiling uses the kept shared common.image_tiling module (Amendment 1)."""
    from models.backends.vllm_backend import VllmBackend

    backend = VllmBackend(engine=object(), model_type_key="internvl3-vllm", pre_tiling_enabled=True)
    # Tall, dense aspect ratio so the tiler returns more than one crop.
    messages = backend._build_messages(
        Image.new("RGB", (448, 1600), "white"), "hello", max_tiles=6, min_tiles=2
    )
    n_images = sum(p["type"] == "image_url" for p in messages[0]["content"])
    assert n_images >= 2, "pre-tiling did not crop into multiple sub-images"


def test_unknown_model_raises_four_element_diagnostic(assert_diagnostic_error):
    """get_model on an unknown type raises a 4-element fail-fast diagnostic.

    After the collapse only three model types exist, so a bad ``model.type`` in
    run_config.yml hits this chokepoint far more often — the error must tell the
    operator What/Where/Expected/How-to-fix, per CLAUDE.md. (Done ahead of the
    collapse, so it is a live invariant, not a goal-state xfail.)
    """
    from models import registry as registry_mod

    with pytest.raises(ValueError) as exc_info:
        registry_mod.get_model("definitely-not-a-real-model")
    assert_diagnostic_error(str(exc_info.value))


# ===========================================================================
# Section B — END STATE (was xfail goal-state during the collapse; Phases 1-5
# have landed, so these are now permanent invariants asserting the HF path and
# non-InternVL registrations are gone).
# ===========================================================================


@pytest.mark.parametrize("model_type", sorted(PHASE2_DELETED_HF_TYPES))
def test_non_internvl_hf_types_unregistered(model_type):
    """No non-InternVL HF (.chat()/AutoModel) model type remains registered (Phase 2 — DONE)."""
    from models import registry as registry_mod

    assert model_type not in registry_mod.list_models()


@pytest.mark.parametrize("model_type", sorted(PHASE3_DELETED_HF_TYPES))
def test_internvl_hf_types_unregistered(model_type):
    """The InternVL HF (.chat()) registrations are gone (Phase 3 — DONE)."""
    from models import registry as registry_mod

    assert model_type not in registry_mod.list_models()


@pytest.mark.parametrize("model_type", sorted(DELETED_VLLM_TYPES))
def test_non_internvl_vllm_types_unregistered(model_type):
    """No non-InternVL vLLM model type remains registered (Phase 2 — DONE)."""
    from models import registry as registry_mod

    assert model_type not in registry_mod.list_models()


def test_registry_holds_only_the_expected_vllm_types():
    """No surprise registrations: the 3 InternVL sizes plus deliberate additions.

    Kept as an exact-set tripwire (not a subset check) so resurrecting an HF
    model, or adding a vLLM model without recording it, still fails here.
    """
    from models import registry as registry_mod

    assert set(registry_mod.list_models()) == EXPECTED_VLLM_TYPES


def test_every_registered_model_is_vllm():
    """The real collapse invariant: nothing in the registry uses the HF path."""
    from models import registry as registry_mod

    non_vllm = [m for m in registry_mod.list_models() if not registry_mod.is_vllm_model(m)]
    assert non_vllm == []


@pytest.mark.parametrize("module", DELETED_HF_BACKEND_MODULES)
def test_hf_backend_modules_deleted(module):
    """The three HF backend modules are gone (Phase 3 — DONE)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("func_name", ["load_prompt_config", "load_pipeline_configs"])
def test_prompt_loader_defaults_flipped_to_vllm(func_name):
    """The prompt-config loaders default to internvl3-vllm, not the old HF internvl3.

    These lived in cli.py until the standalone strip deleted it; the screen is
    the only remaining caller, and it passes the model type explicitly. The
    default still matters because it is what any new caller inherits.
    """
    from common import pipeline_prompts

    sig = inspect.signature(getattr(pipeline_prompts, func_name))
    assert sig.parameters["model_type"].default == "internvl3-vllm"


def test_cli_is_gone():
    """cli.py was deleted by the standalone strip.

    Its only surviving half is common.pipeline_prompts. A cli module coming
    back would drag the extraction command tree with it.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("cli")


def test_pipeline_config_default_model_type_is_vllm():
    """PipelineConfig's runtime default no longer points at a soon-to-be-deleted HF type."""
    from common.pipeline_config import PipelineConfig

    assert PipelineConfig.model_type == "internvl3-vllm"


@pytest.mark.parametrize("fname", DELETED_PROMPT_FILES)
def test_deleted_prompt_files_absent(fname):
    """The non-InternVL prompt YAMLs are removed (Phase 5 — DONE)."""
    assert not (PROMPTS / fname).exists()


def test_generation_schema_collapsed_to_internvl():
    """The generation-config schema drops non-InternVL entries (Phase 4 — DONE)."""
    from common.model_config import _GENERATION_CONFIG_SCHEMA as schema

    assert "internvl3" in schema
    assert "qwen3vl" not in schema
    assert "llama" not in schema
