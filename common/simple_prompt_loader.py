#!/usr/bin/env python3
"""
Simple Prompt Loader - Dead simple, no templates, no rendering.

This replaces the complex template system with direct prompt loading.
What you see in YAML is exactly what gets sent to the model.
"""

from pathlib import Path

import yaml


class SimplePromptLoader:
    """Dead simple prompt loader - no templates, no rendering, no complexity."""

    @staticmethod
    def load_prompt(filename: str, prompt_key: str = "universal") -> str:
        """
        Load a prompt from a YAML file. That's it.

        Args:
            filename: Name of the YAML file in prompts/ directory
            prompt_key: Key of the prompt to load (e.g., "invoice", "universal")

        Returns:
            The complete prompt text as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            KeyError: If the prompt key doesn't exist in the file
        """
        # Strip leading 'prompts/' if present for user convenience
        if filename.startswith("prompts/"):
            filename = filename[8:]  # Remove 'prompts/'
        prompt_path = Path(__file__).parent.parent / "prompts" / filename

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"❌ Prompt file not found: {prompt_path}\n"
                f"💡 Expected location: {prompt_path.absolute()}"
            )

        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"❌ Invalid YAML in {filename}: {e}") from e

        if "prompts" not in data:
            raise ValueError(f"❌ Missing 'prompts' section in {filename}")

        if prompt_key not in data["prompts"]:
            available = list(data["prompts"].keys())
            raise KeyError(
                f"❌ Prompt '{prompt_key}' not found in {filename}\n"
                f"💡 Available prompts: {', '.join(available)}"
            )

        return data["prompts"][prompt_key]["prompt"]

    @staticmethod
    def get_available_prompts(filename: str) -> list[str]:
        """
        Get list of available prompt keys in a file.

        Args:
            filename: Name of the YAML file in prompts/ directory

        Returns:
            List of available prompt keys
        """
        # Strip leading 'prompts/' if present for user convenience
        if filename.startswith("prompts/"):
            filename = filename[8:]  # Remove 'prompts/'
        prompt_path = Path(__file__).parent.parent / "prompts" / filename

        if not prompt_path.exists():
            return []

        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return list(data.get("prompts", {}).keys())
        except (FileNotFoundError, yaml.YAMLError):
            return []

    @staticmethod
    def load_prompt_info(
        filename: str, prompt_key: str = "universal"
    ) -> dict[str, str]:
        """
        Load prompt with metadata (name, description).

        Args:
            filename: Name of the YAML file in prompts/ directory
            prompt_key: Key of the prompt to load

        Returns:
            Dictionary with 'prompt', 'name', and 'description'
        """
        # Strip leading 'prompts/' if present for user convenience
        if filename.startswith("prompts/"):
            filename = filename[8:]  # Remove 'prompts/'
        prompt_path = Path(__file__).parent.parent / "prompts" / filename

        with prompt_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        prompt_data = data["prompts"][prompt_key]
        return {
            "prompt": prompt_data["prompt"],
            "name": prompt_data.get("name", prompt_key),
            "description": prompt_data.get("description", ""),
        }

    @staticmethod
    def get_settings(filename: str) -> dict:
        """
        Get settings from prompt file if available.

        Args:
            filename: Name of the YAML file in prompts/ directory

        Returns:
            Settings dictionary or empty dict if no settings
        """
        # Strip leading 'prompts/' if present for user convenience
        if filename.startswith("prompts/"):
            filename = filename[8:]  # Remove 'prompts/'
        prompt_path = Path(__file__).parent.parent / "prompts" / filename

        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("settings", {})
        except (FileNotFoundError, yaml.YAMLError):
            return {}


# Convenience functions for common use cases
def load_internvl3_prompt(document_type: str = "universal") -> str:
    """Load an InternVL3 prompt for the specified document type."""
    return SimplePromptLoader.load_prompt("internvl3_prompts.yaml", document_type)


def load_llama_prompt(document_type: str = "universal") -> str:
    """Load a Llama prompt for the specified document type."""
    return SimplePromptLoader.load_prompt("llama_prompts.yaml", document_type)


# Testing
if __name__ == "__main__":
    print("🧪 Testing SimplePromptLoader\n")

    try:
        prompt = load_internvl3_prompt("bank_statement")
        print(f"✅ Loaded InternVL3 bank statement prompt ({len(prompt)} chars)")

        available = SimplePromptLoader.get_available_prompts("internvl3_prompts.yaml")
        print(f"✅ Available InternVL3 prompts: {', '.join(available)}")

        settings = SimplePromptLoader.get_settings("internvl3_prompts.yaml")
        print(f"✅ Settings: {settings}")

        print("\n✅ All tests passed! Simple is better.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
