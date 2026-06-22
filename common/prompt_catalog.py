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
            raise FileNotFoundError(f"Prompt file not found: {path}\nExpected location: {path.absolute()}")
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
            raise KeyError(f"Prompt '{key}' not found in {filename}. Available: {', '.join(available)}")

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

        Provides ``type_mappings`` and ``fallback_keywords`` for document type
        normalization.  Runtime settings (``fallback_type``, token budgets)
        live in ``run_config.yml`` and are accessed via ``AppConfig``.

        Returns:
            Parsed dict from ``document_type_detection.yaml``.
        """
        path = self._dir / "document_type_detection.yaml"
        return self._load_yaml(path)

    def get_column_roles(self) -> dict[str, list[str]]:
        """Load semantic column-role -> header-keyword lists from detection YAML.

        Source of truth for ``HeaderListParser._match_columns``.

        Returns:
            Mapping of role name to its list of header keywords.

        Raises:
            ValueError: If the ``column_roles`` block is missing or malformed.
        """
        path = self._dir / "document_type_detection.yaml"
        data = self._load_yaml(path)
        roles = data.get("column_roles")
        if not isinstance(roles, dict) or not roles:
            raise ValueError(
                "Missing or empty 'column_roles' block in "
                f"{path.absolute()} (key: column_roles).\n"
                "Expected a non-empty mapping of role -> keyword list, e.g.:\n"
                "  column_roles:\n"
                "    debit:\n      - debit\n      - withdrawal\n"
                "Fix: add the column_roles: section to document_type_detection.yaml."
            )
        for role, keywords in roles.items():
            if not isinstance(keywords, list) or not keywords:
                raise ValueError(
                    f"Role '{role}' in 'column_roles' must be a non-empty list of "
                    f"keyword strings.\n"
                    f"Where: {path.absolute()} (key: column_roles.{role}).\n"
                    f"Found: {keywords!r}.\n"
                    f"Fix: give '{role}' at least one keyword, e.g. "
                    f"`{role}:\n  - <header keyword>`."
                )
        return roles

    def get_classification_evidence(self) -> dict[str, Any]:
        """Load the evidence-based classification rule set from detection YAML.

        Drives ``ClassificationParser._parse_enriched``: an ordered list of
        ``rules`` (first match wins) plus a ``default`` for when none match.

        Returns:
            Dict with keys ``rules`` (list) and ``default`` (str).

        Raises:
            ValueError: If the ``classification_evidence`` block is missing or
                structurally invalid.
        """
        path = self._dir / "document_type_detection.yaml"
        data = self._load_yaml(path)
        evidence = data.get("classification_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(
                "Missing 'classification_evidence' block in "
                f"{path.absolute()} (key: classification_evidence).\n"
                "Expected a mapping with 'rules' and 'default', e.g.:\n"
                "  classification_evidence:\n"
                "    rules:\n"
                "      - type: BANK_STATEMENT\n"
                "        when: { any_roles: [debit, credit, balance] }\n"
                "    default: none\n"
                "Fix: add the classification_evidence: section to "
                "document_type_detection.yaml."
            )
        rules = evidence.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError(
                "'classification_evidence.rules' must be a non-empty list.\n"
                f"Where: {path.absolute()} (key: classification_evidence.rules).\n"
                f"Found: {rules!r}.\n"
                "Fix: add at least one rule with 'type' and a 'when' clause."
            )
        if "default" not in evidence:
            raise ValueError(
                "'classification_evidence.default' is required.\n"
                f"Where: {path.absolute()} (key: classification_evidence.default).\n"
                "Expected 'none' (defer to keyword fallback) or a document-type "
                "string (e.g. INVOICE).\n"
                "Fix: add `default: none` under classification_evidence."
            )
        self._validate_evidence_references(evidence, rules, path)
        return evidence

    def _validate_evidence_references(self, evidence: dict[str, Any], rules: list, path: Path) -> None:
        """Cross-reference rules against column_roles and supported types.

        Ensures every referenced role exists, every `when:` uses known keys,
        and every emitted type (and the default) is a supported document type.
        """
        from common.field_schema import get_field_schema

        known_roles = set(self.get_column_roles())
        supported = {t.lower() for t in get_field_schema().supported_document_types}
        allowed_when_keys = {"any_roles", "all_roles", "paid", "travel"}

        for i, rule in enumerate(rules):
            loc = f"classification_evidence.rules[{i}]"
            if not isinstance(rule, dict) or "type" not in rule or "when" not in rule:
                raise ValueError(
                    f"Each classification rule needs 'type' and 'when'.\n"
                    f"Where: {path.absolute()} ({loc}).\n"
                    f"Found: {rule!r}.\n"
                    "Fix: e.g. `- type: RECEIPT\n    when: { paid: true }`."
                )
            rtype = rule["type"]
            if not isinstance(rtype, str) or rtype.lower() not in supported:
                raise ValueError(
                    f"Rule type '{rtype}' is not a supported document type.\n"
                    f"Where: {path.absolute()} ({loc}.type).\n"
                    f"Allowed: {sorted(supported)}.\n"
                    "Fix: use one of the supported types, or register the new "
                    "type in config/field_definitions.yaml (supported_document_types)."
                )
            when = rule["when"]
            if not isinstance(when, dict) or not when:
                raise ValueError(
                    f"Rule 'when:' must be a non-empty mapping.\n"
                    f"Where: {path.absolute()} ({loc}.when).\n"
                    f"Found: {when!r}.\n"
                    "Fix: add a condition, e.g. `when: { any_roles: [debit] }`."
                )
            unknown_keys = set(when) - allowed_when_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown 'when:' key(s) {sorted(unknown_keys)}.\n"
                    f"Where: {path.absolute()} ({loc}.when).\n"
                    f"Allowed keys: {sorted(allowed_when_keys)}.\n"
                    "Fix: remove the unknown key or correct the spelling."
                )
            for role_key in ("any_roles", "all_roles"):
                for role in when.get(role_key, []):
                    if role not in known_roles:
                        raise ValueError(
                            f"Rule references unknown column role '{role}'.\n"
                            f"Where: {path.absolute()} ({loc}.when.{role_key}).\n"
                            f"Known roles: {sorted(known_roles)}.\n"
                            f"Fix: add '{role}' under column_roles, or use a "
                            "known role name."
                        )

        default = evidence["default"]
        if isinstance(default, str) and default.lower() != "none" and default.lower() not in supported:
            raise ValueError(
                f"classification_evidence.default '{default}' is not a supported type.\n"
                f"Where: {path.absolute()} (key: classification_evidence.default).\n"
                f"Allowed: 'none' or one of {sorted(supported)}.\n"
                "Fix: set `default: none` or a supported document type."
            )

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
        orphans: list[str] = []
        for key in prompt_keys:
            base_type = strip_structure_suffixes(key, suffixes)
            if base_type in supported_types:
                routing[base_type.upper()] = key
            else:
                orphans.append(key)

        if orphans:
            raise ValueError(
                "Orphaned extraction prompt key(s) do not map to any supported "
                f"document type: {sorted(orphans)}.\n"
                f"Where: {path.absolute()} (under the 'prompts:' section).\n"
                "Each prompt key (after stripping structure suffixes "
                f"{list(suffixes)}) must equal a supported_document_types entry in "
                "config/field_definitions.yaml.\n"
                f"Allowed types: {sorted(supported_types)}.\n"
                "Fix: rename the key to its canonical type (e.g. 'travel_expense' "
                "-> 'travel'), or add the new type to supported_document_types."
            )

        if not routing:
            raise ValueError(
                f"No document type prompts found in {filename}. "
                f"Expected keys like 'invoice', 'receipt', etc. in prompts section. "
                f"Found keys: {sorted(prompt_keys)}"
            )

        return routing
