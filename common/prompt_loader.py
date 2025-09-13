"""YAML prompt loading utilities.

This module provides functionality to load and manage document-specific
extraction prompts from YAML configuration files.
"""

from pathlib import Path

import yaml
from rich import print as rprint
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table


class PromptLoader:
    """Handles loading and managing YAML-based extraction prompts."""

    def __init__(self):
        """Initialize the prompt loader."""
        self.console = Console()

    def load_prompt(
        self,
        prompt_file: str,
        prompt_key: str,
        document_type: str,
        verbose: bool = True,
    ) -> tuple[str, str, str]:
        """Load a specific prompt from a YAML file.

        Args:
            prompt_file: Path to the YAML prompt file
            prompt_key: Key of the prompt to load
            document_type: Type of document (for display purposes)

        Returns:
            Tuple of (prompt_text, prompt_name, prompt_description)

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            KeyError: If the prompt key is not found
            yaml.YAMLError: If the YAML is invalid
        """
        if verbose:
            rprint(f"[cyan]📁 Document type: {document_type}[/cyan]")
            rprint(f"[cyan]📁 Prompt file: {prompt_file}[/cyan]")
            rprint(f"[cyan]🔑 Selected prompt: {prompt_key}[/cyan]")

        try:
            prompt_path = Path(prompt_file)

            if not prompt_path.exists():
                rprint(f"[red]❌ Prompt file not found at {prompt_file}[/red]")
                rprint(
                    f"[yellow]💡 Please ensure the YAML file exists at: {prompt_path.absolute()}[/yellow]"
                )
                raise FileNotFoundError(
                    f"Required prompt file not found: {prompt_file}"
                )

            # Load the YAML file
            with prompt_path.open("r") as f:
                prompt_config = yaml.safe_load(f)

            # Extract the selected prompt
            if prompt_key in prompt_config.get("prompts", {}):
                prompt_data = prompt_config["prompts"][prompt_key]
                prompt_text = prompt_data["prompt"]
                prompt_name = prompt_data.get("name", prompt_key)
                prompt_description = prompt_data.get("description", "")

                if verbose:
                    rprint(f"[green]✅ Loaded prompt: {prompt_name}[/green]")
                    if prompt_description:
                        rprint(f"[dim]{prompt_description}[/dim]")
            else:
                available_keys = list(prompt_config.get("prompts", {}).keys())
                rprint(
                    f"[red]❌ Prompt key '{prompt_key}' not found in {prompt_file}[/red]"
                )
                rprint(
                    f"[yellow]Available prompts: {', '.join(available_keys)}[/yellow]"
                )
                raise KeyError(
                    f"Prompt '{prompt_key}' not found. Available: {available_keys}"
                )

            # Load settings if available
            settings = prompt_config.get("settings", {})
            if settings and verbose:
                rprint("[dim]📊 Loaded settings from YAML:[/dim]")
                for key, value in settings.items():
                    rprint(f"[dim]  • {key}: {value}[/dim]")

            return prompt_text, prompt_name, prompt_description

        except FileNotFoundError:
            rprint(f"[red]❌ CRITICAL: Prompt file not found: {prompt_file}[/red]")
            rprint(
                "[yellow]Please create the prompt YAML file before running.[/yellow]"
            )
            raise

        except yaml.YAMLError as e:
            rprint(f"[red]❌ Error parsing YAML file: {e}[/red]")
            rprint(f"[yellow]Please check the YAML syntax in {prompt_file}[/yellow]")
            raise

        except Exception as e:
            rprint(f"[red]❌ Error loading prompt from YAML: {e}[/red]")
            raise

    def display_prompt_info(
        self,
        prompt_text: str,
        prompt_name: str,
        prompt_description: str,
        document_type: str,
        prompt_file: str,
        prompt_key: str,
    ):
        """Display detailed information about the loaded prompt.

        Args:
            prompt_text: The prompt text content
            prompt_name: Name of the prompt
            prompt_description: Description of the prompt
            document_type: Type of document
            prompt_file: Path to the prompt file
            prompt_key: Key of the prompt
        """
        # Display loaded prompt header
        self.console.rule("[bold blue]Loaded Extraction Prompt Display[/bold blue]")

        # Display as syntax-highlighted markdown
        rprint(f"[bold cyan]📋 {prompt_name} for {document_type}:[/bold cyan]")
        if prompt_description:
            rprint(f"[dim]{prompt_description}[/dim]")

        # Show full prompt with syntax highlighting
        syntax = Syntax(prompt_text, "markdown", theme="monokai", line_numbers=True)
        self.console.print(syntax)

        # Display prompt statistics
        self._display_prompt_statistics(
            prompt_text, document_type, prompt_file, prompt_key
        )

        self.console.rule("[bold green]Prompt Loading Complete[/bold green]")

    def _display_prompt_statistics(
        self, prompt_text: str, document_type: str, prompt_file: str, prompt_key: str
    ):
        """Display statistics about the prompt.

        Args:
            prompt_text: The prompt text content
            document_type: Type of document
            prompt_file: Path to the prompt file
            prompt_key: Key of the prompt
        """
        prompt_lines = prompt_text.strip().split("\n")
        prompt_words = len(prompt_text.split())
        prompt_chars = len(prompt_text)

        stats_table = Table(title="📊 Prompt Statistics", border_style="green")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")

        stats_table.add_row("Document Type", document_type)
        stats_table.add_row("Lines", str(len(prompt_lines)))
        stats_table.add_row("Words", str(prompt_words))
        stats_table.add_row("Characters", str(prompt_chars))
        stats_table.add_row("Source", str(Path(prompt_file).name))
        stats_table.add_row("Prompt Key", prompt_key)

        self.console.print(stats_table)


def load_document_prompt(
    prompt_files: dict, prompt_keys: dict, document_type: str, verbose: bool = True
) -> tuple[str, str, str]:
    """Convenience function to load a document-specific prompt.

    Args:
        prompt_files: Dictionary mapping document types to prompt files
        prompt_keys: Dictionary mapping document types to prompt keys
        document_type: Type of document to load prompt for
        verbose: Whether to display loading information

    Returns:
        Tuple of (prompt_text, prompt_name, prompt_description)
    """
    prompt_file = prompt_files.get(document_type, "prompts/invoice_extraction.yaml")
    prompt_key = prompt_keys.get(document_type, "standard")

    loader = PromptLoader()
    prompt_text, prompt_name, prompt_description = loader.load_prompt(
        prompt_file, prompt_key, document_type, verbose=verbose
    )

    # Display the prompt information only if verbose
    if verbose:
        loader.display_prompt_info(
            prompt_text,
            prompt_name,
            prompt_description,
            document_type,
            prompt_file,
            prompt_key,
        )

    return prompt_text, prompt_name, prompt_description
