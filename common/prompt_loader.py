#!/usr/bin/env python3
"""
YAML-First Prompt Loader

Configurable prompt loading system that eliminates hardcoded paths and enables
easy maintenance and experimentation with different prompt strategies.

Part of Integrated V4 + YAML-First Implementation Plan.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import yaml


class PromptLoader:
    """YAML-based configurable prompt loader for maintainable prompt management."""
    
    def __init__(self, config_file: Union[str, Path] = "prompts/prompt_config.yaml"):
        """
        Initialize prompt loader with configuration file.
        
        Args:
            config_file: Path to prompt configuration YAML file
        """
        # Resolve path relative to project root, not current directory
        if not Path(config_file).is_absolute():
            # Find project root by looking for common project files
            current_dir = Path.cwd()
            project_root = self._find_project_root(current_dir)
            self.config_file = project_root / config_file
        else:
            self.config_file = Path(config_file)
        
        self.config = self._load_config()
        self._validate_config()
    
    def _find_project_root(self, start_path: Path) -> Path:
        """Find project root by looking for indicator files."""
        current = start_path
        
        # Look for common project root indicators
        indicators = [
            "llama_document_aware.py",
            "common",
            "models", 
            "prompts",
            ".git"
        ]
        
        while current != current.parent:  # Stop at filesystem root
            if any((current / indicator).exists() for indicator in indicators):
                return current
            current = current.parent
        
        # Fallback to current directory if no project root found
        return start_path
        
    def _load_config(self) -> Dict:
        """Load prompt configuration from YAML file."""
        try:
            with self.config_file.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ FATAL: Prompt configuration file not found: {self.config_file.absolute()}\n"
                f"💡 Expected location: {self.config_file.absolute()}\n"
                f"💡 Create this file with prompt path configurations"
            ) from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"❌ FATAL: Invalid YAML in prompt configuration: {e}\n"
                f"💡 Check YAML syntax in {self.config_file}"
            ) from e
    
    def _validate_config(self) -> None:
        """Validate prompt configuration structure."""
        required_keys = ["prompts", "base_path", "schema_version"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"❌ FATAL: Missing required key '{key}' in prompt configuration\n"
                    f"💡 Add '{key}' to {self.config_file}"
                )
                
        # Validate prompt structure
        if "llama" not in self.config["prompts"] or "internvl3" not in self.config["prompts"]:
            raise ValueError(
                f"❌ FATAL: Prompt configuration must contain 'llama' and 'internvl3' sections\n"
                f"💡 Check prompts section in {self.config_file}"
            )
    
    def get_prompt_path(self, model_name: str, strategy: str) -> Path:
        """
        Get full path to prompt file based on configuration.
        
        Args:
            model_name: Model identifier ('llama' or 'internvl3')
            strategy: Extraction strategy ('single_pass' or 'grouped')
            
        Returns:
            Path: Full path to prompt YAML file
            
        Raises:
            ValueError: If model or strategy not found in configuration
        """
        try:
            prompt_file = self.config["prompts"][model_name][strategy]
        except KeyError as e:
            available_models = list(self.config["prompts"].keys())
            if model_name not in self.config["prompts"]:
                raise ValueError(
                    f"❌ Model '{model_name}' not found in configuration\n"
                    f"💡 Available models: {available_models}\n"
                    f"💡 Add '{model_name}' section to {self.config_file}"
                ) from e
            else:
                available_strategies = list(self.config["prompts"][model_name].keys())
                raise ValueError(
                    f"❌ Strategy '{strategy}' not found for model '{model_name}'\n"
                    f"💡 Available strategies: {available_strategies}\n"
                    f"💡 Add '{strategy}' to {model_name} section in {self.config_file}"
                ) from e
        
        # Build full path
        base_path = Path(self.config["base_path"])
        full_path = base_path / prompt_file
        
        return full_path
    
    def load_prompt_config(self, model_name: str, strategy: str) -> Dict:
        """
        Load and parse prompt configuration from YAML file.
        
        Args:
            model_name: Model identifier ('llama' or 'internvl3')  
            strategy: Extraction strategy ('single_pass' or 'grouped')
            
        Returns:
            Dict: Parsed prompt configuration
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
            yaml.YAMLError: If prompt file has invalid YAML
        """
        prompt_path = self.get_prompt_path(model_name, strategy)
        
        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ FATAL: Prompt file not found: {prompt_path.absolute()}\n"
                f"💡 Expected location: {prompt_path.absolute()}\n"
                f"💡 Create this file with {model_name} {strategy} prompts"
            ) from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"❌ FATAL: Invalid YAML in prompt file: {e}\n"
                f"💡 Check YAML syntax in {prompt_path}"
            ) from e
    
    def get_schema_version(self) -> str:
        """Get schema version from configuration."""
        return self.config.get("schema_version", "v4")
    
    def get_field_count(self) -> int:
        """Get expected field count from configuration."""
        return self.config.get("field_count", 49)
    
    def load_detection_prompts(self) -> Dict:
        """
        Load document type detection prompts from YAML configuration.
        
        Returns:
            Dict: Parsed detection prompt configuration
            
        Raises:
            FileNotFoundError: If detection prompts file doesn't exist
            yaml.YAMLError: If detection prompts file has invalid YAML
        """
        detection_file = self.config.get("detection_prompts")
        if not detection_file:
            raise ValueError(
                f"❌ FATAL: No detection_prompts configured in {self.config_file}\n"
                f"💡 Add 'detection_prompts: \"document_type_detection.yaml\"' to config"
            )
        
        # Build full path to detection prompts file
        base_path = Path(self.config["base_path"])
        detection_path = base_path / detection_file
        
        try:
            with detection_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ FATAL: Detection prompts file not found: {detection_path.absolute()}\n"
                f"💡 Expected location: {detection_path.absolute()}\n"
                f"💡 Create this file with document type detection prompts"
            ) from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"❌ FATAL: Invalid YAML in detection prompts file: {e}\n"
                f"💡 Check YAML syntax in {detection_path}"
            ) from e
    
    def load_debug_ocr_prompts(self) -> Dict:
        """
        Load debug OCR prompts from YAML configuration.
        
        Returns:
            Dict: Parsed debug OCR prompt configuration
            
        Raises:
            FileNotFoundError: If debug OCR prompts file doesn't exist
            yaml.YAMLError: If debug OCR prompts file has invalid YAML
        """
        debug_file = self.config.get("debug_ocr_prompts")
        if not debug_file:
            raise ValueError(
                f"❌ FATAL: No debug_ocr_prompts configured in {self.config_file}\n"
                f"💡 Add 'debug_ocr_prompts: \"debug_ocr_prompts.yaml\"' to config"
            )
        
        # Build full path to debug OCR prompts file
        base_path = Path(self.config["base_path"])
        debug_path = base_path / debug_file
        
        try:
            with debug_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ FATAL: Debug OCR prompts file not found: {debug_path.absolute()}\n"
                f"💡 Expected location: {debug_path.absolute()}\n"
                f"💡 Create this file with debug OCR prompts for raw markdown output"
            ) from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"❌ FATAL: Invalid YAML in debug OCR prompts file: {e}\n"
                f"💡 Check YAML syntax in {debug_path}"
            ) from e
    
    def get_experimental_prompt_path(self, experiment_name: str) -> Optional[Path]:
        """
        Get path to experimental prompt file.
        
        Args:
            experiment_name: Name of experimental prompt variant
            
        Returns:
            Path: Full path to experimental prompt file, or None if not found
        """
        experimental = self.config.get("experimental", {})
        if experiment_name not in experimental:
            return None
            
        base_path = Path(self.config["base_path"])
        return base_path / experimental[experiment_name]
    
    def list_available_models(self) -> list:
        """List all available model configurations."""
        return list(self.config["prompts"].keys())
    
    def list_available_strategies(self, model_name: str) -> list:
        """List all available strategies for a model."""
        if model_name not in self.config["prompts"]:
            return []
        return list(self.config["prompts"][model_name].keys())
    
    def validate_prompt_files_exist(self) -> bool:
        """
        Validate that all configured prompt files exist.
        
        Returns:
            bool: True if all files exist, False otherwise
        """
        missing_files = []
        
        for model_name, strategies in self.config["prompts"].items():
            for strategy, _filename in strategies.items():
                try:
                    prompt_path = self.get_prompt_path(model_name, strategy)
                    if not prompt_path.exists():
                        missing_files.append(str(prompt_path))
                except Exception as e:
                    missing_files.append(f"{model_name}/{strategy}: {e}")
        
        if missing_files:
            print("⚠️ Missing prompt files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        return True


def get_default_prompt_loader() -> PromptLoader:
    """Get default prompt loader instance."""
    return PromptLoader()


# Convenience functions for backward compatibility
def load_prompt_config(model_name: str, strategy: str) -> Dict:
    """Load prompt configuration using default loader."""
    loader = get_default_prompt_loader()
    return loader.load_prompt_config(model_name, strategy)


def get_prompt_path(model_name: str, strategy: str) -> Path:
    """Get prompt file path using default loader."""
    loader = get_default_prompt_loader()
    return loader.get_prompt_path(model_name, strategy)