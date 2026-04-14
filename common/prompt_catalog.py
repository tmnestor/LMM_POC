"""Unified prompt loading -- single entry point for all YAML prompt access.

Consolidates SimplePromptLoader, ConfigLoader, and inline YAML parsing
behind a single class with cached file reads.

Usage:
    from common.prompt_catalog import PromptCatalog

    catalog = PromptCatalog()
    prompt = catalog.get_prompt("internvl3", "invoice")
    bank_prompt = catalog.get_prompt("bank", "turn0_header_detection")
"""

from pathlib import Path
from typing import Any

import yaml

# Fixed namespace -> filename mapping (everything else resolves via registry)
_FIXED_NAMESPACES: dict[str, str] = {
    "detection": "document_type_detection.yaml",
    "bank": "bank_prompts.yaml",
}


class PromptCatalog:
    """Unified prompt loading -- single entry point for all YAML prompt access.

    All YAML files live in one directory (``prompts/``).  Namespaces map to
    filenames: ``"detection"`` -> ``document_type_detection.yaml``,
    ``"bank"`` -> ``bank_prompts.yaml``, or any registered model_type ->
    its ``ModelRegistration.prompt_file``.

    Files are parsed once and cached per instance.

    Args:
        prompts_dir: Directory containing all prompt YAML files.
            Default: auto-detected from this file's location.
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "prompts"
        self._dir = prompts_dir
        self._cache: dict[Path, dict] = {}

    # -- Internal helpers ------------------------------------------------------

    def _load_yaml(self, path: Path) -> dict:
        """Load and cache a YAML file."""
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}\nExpected location: {path.absolute()}"
            )
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._cache[path] = data
        return data

    def _resolve_filename(self, namespace: str) -> str:
        """Resolve a namespace string to a YAML filename."""
        if namespace in _FIXED_NAMESPACES:
            return _FIXED_NAMESPACES[namespace]
        # Dynamic model namespace -- look up via registry
        from models.registry import get_model

        registration = get_model(namespace)
        return registration.prompt_file

    # -- Entry point 1: Load a single prompt -----------------------------------

    def get_prompt(
        self,
        namespace: str,
        key: str,
        *,
        format_args: dict[str, str] | None = None,
    ) -> str:
        """Load a prompt string by namespace and key.

        Args:
            namespace: Logical grouping that maps to a YAML file.
                ``"detection"`` -> ``document_type_detection.yaml``
                ``"bank"``      -> ``bank_prompts.yaml``
                Or any registered model_type -> its prompt_file.
            key: Prompt key within the YAML file's ``prompts`` section.
            format_args: Optional dict for ``str.format()`` on the template.
                Used by bank prompts with placeholders like ``{balance_col}``.

        Returns:
            The prompt text string, ready to send to the model.

        Raises:
            FileNotFoundError: YAML file for namespace not found.
            KeyError: Prompt key not found (with available keys).
        """
        filename = self._resolve_filename(namespace)
        path = self._dir / filename
        data = self._load_yaml(path)

        prompts = data.get("prompts", {})
        if key not in prompts:
            available = sorted(prompts.keys())
            raise KeyError(
                f"Prompt '{key}' not found in {filename}. "
                f"Available: {', '.join(available)}"
            )

        entry = prompts[key]
        # Normalize: extraction YAMLs use "prompt", bank uses "template"
        text = entry.get("prompt") or entry.get("template", "")

        if format_args:
            text = text.format(**format_args)
        return text

    # -- Entry point 2: Discovery / introspection -----------------------------

    def list_keys(self, namespace: str) -> list[str]:
        """List available prompt keys for a namespace.

        Returns:
            Sorted list of available keys, or ``[]`` if file not found.
        """
        try:
            filename = self._resolve_filename(namespace)
            path = self._dir / filename
            data = self._load_yaml(path)
            return sorted(data.get("prompts", {}).keys())
        except (FileNotFoundError, ValueError):
            return []

    # -- Entry point 3: Detection config (structured data) --------------------

    def get_detection_config(self) -> dict[str, Any]:
        """Load the full detection YAML as a dict.

        Needed by DocumentOrchestrator for type_mappings, fallback_keywords,
        settings.fallback_type, and settings.max_new_tokens.

        Returns:
            Parsed dict from ``document_type_detection.yaml``.
        """
        path = self._dir / "document_type_detection.yaml"
        return self._load_yaml(path)

    # -- Auxiliary: column patterns (bank-specific structured data) ------------

    def get_column_patterns(self) -> dict[str, Any]:
        """Load bank column patterns from ``bank_column_patterns.yaml``.

        Returns:
            The ``patterns`` dict mapping column types to keyword lists.
        """
        path = self._dir / "bank_column_patterns.yaml"
        data = self._load_yaml(path)
        return data.get("patterns", {})

    # -- Auxiliary: extraction routing (replaces cli.py:load_prompt_config) ----

    def build_extraction_routing(self, model_type: str) -> dict[str, str]:
        """Derive the ``{DOC_TYPE: prompt_key}`` mapping for a given model.

        Cross-references the model's extraction YAML keys against
        ``field_definitions.yaml`` to determine which keys are document types,
        strips structure suffixes, and returns the canonical mapping.

        Args:
            model_type: Registered model type string (e.g. ``"internvl3"``).

        Returns:
            Dict mapping uppercase canonical types to prompt keys.
            e.g. ``{"INVOICE": "invoice", "RECEIPT": "receipt",
                     "BANK_STATEMENT": "bank_statement_flat"}``

        Raises:
            FileNotFoundError: If extraction YAML not found.
            ValueError: If no document type keys found.
        """
        from common.field_schema import get_field_schema
        from common.pipeline_config import (
            load_structure_suffixes,
            strip_structure_suffixes,
        )

        filename = self._resolve_filename(model_type)
        path = self._dir / filename
        data = self._load_yaml(path)

        prompt_keys = set(data.get("prompts", {}).keys())
        supported_types = set(get_field_schema().supported_document_types)
        suffixes = load_structure_suffixes(path)

        routing: dict[str, str] = {}
        for key in prompt_keys:
            base_type = strip_structure_suffixes(key, suffixes)
            if base_type in supported_types:
                canonical = base_type.upper()
                routing[canonical] = key

        if not routing:
            raise ValueError(
                f"No document type prompts found in {filename}. "
                f"Expected keys like 'invoice', 'receipt', etc. in prompts section. "
                f"Found keys: {sorted(prompt_keys)}"
            )

        return routing
