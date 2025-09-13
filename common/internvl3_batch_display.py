"""
InternVL3 Enhanced Batch Display Module

Enhanced display functions for InternVL3 batch processing notebooks.
Provides comprehensive debugging/analysis features with full detail display capabilities.
Features full prompts, raw responses, and detailed field comparisons like Llama version.
"""

from typing import Any, Dict, Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Default display configuration
DEFAULT_DISPLAY_CONFIG = {
    'show_images': True,
    'truncate_responses': False,  # Changed default to show full responses
    'show_full_prompts': True,   # Show full prompts by default
    'max_prompt_length': None,   # None = no truncation
    'max_response_length': None, # None = no truncation
    'max_field_length': None,    # None = no truncation
    'table_width': 120,          # Match Llama's detailed tables
    'field_column_width': 40,    # Wider field display
    'use_syntax_highlighting': True,
    'show_prompt_statistics': True
}


class InternVL3BatchDisplay:
    """Enhanced display functions for InternVL3 batch processing results."""

    def __init__(self, console: Optional[Console] = None, display_config: Optional[Dict] = None):
        """Initialize with optional console and display configuration."""
        self.console = console or Console()
        self.config = {**DEFAULT_DISPLAY_CONFIG, **(display_config or {})}

    def display_prompt_info(
        self,
        result: Dict[str, Any],
        document_type: str,
        display_config: Optional[Dict] = None,
        show_full: Optional[bool] = None
    ) -> None:
        """
        Enhanced Feature 1: Display the prompt used for extraction with full detail.

        Args:
            result: Processing result dictionary
            document_type: Detected document type
            display_config: Display configuration options
            show_full: Override to show full prompt (overrides config)
        """
        config = {**DEFAULT_DISPLAY_CONFIG, **(display_config or {})}

        # Override with explicit parameter
        if show_full is not None:
            config['show_full_prompts'] = show_full
            config['max_prompt_length'] = None if show_full else 200

        # Get prompt info from processor result
        prompt_info = result.get("prompt_info", {})

        if not prompt_info:
            # Fallback - show basic prompt info
            rprint(f"[cyan]🔧 Prompt: Document-aware {document_type} extraction[/cyan]")
            return

        # Extract prompt details
        prompt_text = prompt_info.get("prompt", "No prompt captured")
        prompt_source = prompt_info.get("source", "Unknown")
        field_count = prompt_info.get("field_count", 0)
        prompt_file = prompt_info.get("prompt_file", "Unknown")
        prompt_key = prompt_info.get("prompt_key", "Unknown")

        # Create enhanced prompt display
        header_content = f"""[bold cyan]Document Type:[/bold cyan] {document_type}
[bold cyan]Field Count:[/bold cyan] {field_count} fields
[bold cyan]Prompt Source:[/bold cyan] {prompt_source}
[bold cyan]Prompt File:[/bold cyan] {prompt_file}
[bold cyan]Prompt Key:[/bold cyan] {prompt_key}"""

        # Show prompt statistics if enabled
        if config['show_prompt_statistics']:
            word_count = len(prompt_text.split())
            char_count = len(prompt_text)
            line_count = len(prompt_text.splitlines())
            header_content += f"""
[bold cyan]Statistics:[/bold cyan] {word_count} words, {char_count} chars, {line_count} lines"""

        self.console.print(
            Panel(header_content, title="🔧 Prompt Information", border_style="cyan")
        )

        # Determine if we should truncate
        max_length = config.get('max_prompt_length')
        show_full_prompt = config.get('show_full_prompts', True)

        if not show_full_prompt and max_length and len(prompt_text) > max_length:
            displayed_prompt = prompt_text[:max_length] + "\n\n[dim]... (truncated for display - use show_full=True for complete prompt)[/dim]"
        else:
            displayed_prompt = prompt_text

        # Use syntax highlighting if enabled
        if config.get('use_syntax_highlighting', True):
            syntax = Syntax(
                displayed_prompt,
                "text",
                theme="monokai",
                line_numbers=False,
                word_wrap=True
            )
            self.console.print(
                Panel(syntax, title="📝 Full Prompt Text", border_style="yellow")
            )
        else:
            prompt_panel = f"""[bold yellow]Complete Prompt Text:[/bold yellow]

{displayed_prompt}"""
            self.console.print(
                Panel(prompt_panel, title="📝 Prompt Text", border_style="yellow")
            )

    def display_raw_and_cleaned(
        self,
        result: Dict[str, Any],
        display_config: Optional[Dict] = None,
        show_full: Optional[bool] = None
    ) -> None:
        """
        Enhanced Feature 2 & 3: Display raw extracted text and cleaned text with full detail.

        Args:
            result: Processing result dictionary
            display_config: Display configuration options
            show_full: Override to show full raw response (overrides config)
        """
        config = {**DEFAULT_DISPLAY_CONFIG, **(display_config or {})}

        # Override with explicit parameter
        if show_full is not None:
            config['truncate_responses'] = not show_full
            config['max_response_length'] = None if show_full else 300

        raw_response = result.get("raw_response", "No raw response captured")
        extracted_data = result.get("extracted_data", {})

        # Create two-column comparison with configurable width
        table_width = config.get('table_width', 120)
        col_width = table_width // 2 - 5  # Account for padding

        table = Table(show_header=True, header_style="bold magenta", width=table_width)
        table.add_column("Raw Model Response", style="yellow", width=col_width)
        table.add_column("Cleaned/Parsed Data", style="green", width=col_width)

        # Format raw response with configurable truncation
        max_length = config.get('max_response_length')
        should_truncate = config.get('truncate_responses', False)

        if should_truncate and max_length and len(raw_response) > max_length:
            raw_display = (
                raw_response[:max_length] + "\n\n[dim]... (truncated for display - use show_full=True for complete response)[/dim]"
            )
        else:
            raw_display = raw_response

        # Format cleaned data with enhanced field display
        cleaned_lines = []
        found_count = 0
        for field, value in extracted_data.items():
            if value != "NOT_FOUND":
                cleaned_lines.append(f"✅ {field}: {value}")
                found_count += 1
            else:
                cleaned_lines.append(f"❌ {field}: NOT_FOUND")

        cleaned_display = "\n".join(cleaned_lines)
        cleaned_display += f"\n\n[bold]Summary: {found_count}/{len(extracted_data)} fields found[/bold]"

        # Add response statistics
        response_stats = f"""[dim]Response length: {len(raw_response)} characters
Lines: {len(raw_response.splitlines())}
Words: {len(raw_response.split())}[/dim]"""

        if should_truncate and max_length and len(raw_response) > max_length:
            raw_display = response_stats + "\n\n" + raw_display
        else:
            raw_display = response_stats + "\n\n" + raw_display

        table.add_row(raw_display, cleaned_display)

        rprint("\n")
        self.console.print(
            Panel(table, title="📝 Raw Response vs Cleaned Data (Enhanced)", border_style="blue")
        )

    def display_field_comparison(
        self,
        result: Dict[str, Any],
        ground_truth: Dict[str, Any],
        document_type: str,
        display_config: Optional[Dict] = None,
        show_full: Optional[bool] = None
    ) -> None:
        """
        Enhanced Feature 4: Display comprehensive fieldwise comparison like Llama version.

        Args:
            result: Processing result dictionary
            ground_truth: Ground truth data for this image
            document_type: Document type for context
            display_config: Display configuration options
            show_full: Override to show full field values (overrides config)
        """
        config = {**DEFAULT_DISPLAY_CONFIG, **(display_config or {})}

        # Override with explicit parameter
        if show_full is not None:
            config['max_field_length'] = None if show_full else 40

        if not ground_truth:
            rprint("[yellow]⚠️ No ground truth available for comparison[/yellow]")
            return

        extracted_data = result.get("extracted_data", {})
        image_name = result.get("image_name", "Unknown")

        # Create enhanced comparison display similar to Llama's _display_detailed_field_comparison
        rprint(f"\n{'=' * config['table_width']}")
        rprint("📋 COMPREHENSIVE FIELD COMPARISON (Enhanced InternVL3 Display)")
        rprint("=" * config['table_width'])

        # Display extracted data summary first
        rprint("\n🔍 EXTRACTED DATA SUMMARY:")
        found_fields = []
        for field, value in extracted_data.items():
            if value != "NOT_FOUND":
                rprint(f"✅ {field}: {value}")
                found_fields.append(field)
            else:
                rprint(f"❌ {field}: {value}")

        # Enhanced comparison table with full width like Llama version
        rprint(f"\n📊 Ground truth loaded for {image_name}")
        rprint("-" * config['table_width'])

        # Table header with enhanced spacing (match Llama's format)
        field_width = config.get('field_column_width', 40)
        extracted_width = field_width
        gt_width = field_width
        status_width = 8

        rprint(f"{'STATUS':<{status_width}} {'FIELD':<25} {'EXTRACTED':<{extracted_width}} {'GROUND TRUTH':<{gt_width}}")
        rprint("=" * config['table_width'])

        # Enhanced field-by-field comparison
        matches = 0
        exact_matches = 0
        total_fields = 0

        # Process all fields that exist in either extracted data or ground truth
        all_fields = set(extracted_data.keys()) | set(ground_truth.keys())

        for field in sorted(all_fields):
            extracted_value = extracted_data.get(field, "NOT_FOUND")
            gt_value = ground_truth.get(field, "NOT_AVAILABLE")

            # Enhanced match checking with partial matching
            if extracted_value != "NOT_FOUND" and gt_value != "NOT_AVAILABLE":
                total_fields += 1

                # Exact match check
                is_exact_match = (
                    str(extracted_value).strip().lower()
                    == str(gt_value).strip().lower()
                )

                # Partial match check (for fuzzy matching)
                is_partial_match = (
                    str(extracted_value).strip().lower() in str(gt_value).strip().lower() or
                    str(gt_value).strip().lower() in str(extracted_value).strip().lower()
                )

                if is_exact_match:
                    status = "✅"
                    exact_matches += 1
                    matches += 1
                elif is_partial_match:
                    status = "≈"
                    matches += 1
                else:
                    status = "❌"

            elif extracted_value == "NOT_FOUND":
                total_fields += 1
                status = "❌"
            else:
                # GT not available, don't count in totals
                status = "➖"

            # Format values for display with configurable truncation
            max_field_length = config.get('max_field_length')

            if max_field_length:
                extracted_display = str(extracted_value)[:max_field_length] + (
                    "..." if len(str(extracted_value)) > max_field_length else ""
                )
                gt_display = str(gt_value)[:max_field_length] + (
                    "..." if len(str(gt_value)) > max_field_length else ""
                )
            else:
                # Show full values
                extracted_display = str(extracted_value)
                gt_display = str(gt_value)

            rprint(
                f"{status:<{status_width}} {field:<25} {extracted_display:<{extracted_width}} {gt_display:<{gt_width}}"
            )

        # Enhanced summary section (match Llama's format)
        overall_accuracy = (matches / total_fields) if total_fields > 0 else 0
        exact_accuracy = (exact_matches / total_fields) if total_fields > 0 else 0

        rprint("\n📊 EXTRACTION SUMMARY (Enhanced):")
        rprint(
            f"✅ Fields Found: {len(found_fields)}/{total_fields} ({len(found_fields) / total_fields * 100:.1f}%)"
        )
        rprint(
            f"🎯 Exact Matches: {exact_matches}/{total_fields} ({exact_accuracy * 100:.1f}%)"
        )
        rprint(
            f"≈ Partial Matches: {matches - exact_matches}/{total_fields} ({(matches - exact_matches) / total_fields * 100:.1f}%)"
        )
        rprint(f"📈 Overall Match Rate: {overall_accuracy * 100:.1f}%")
        rprint(f"🤖 Document Type: {document_type}")
        rprint("🔧 Model: InternVL3 (Enhanced Display)")

        # Additional enhanced metrics
        threshold = 0.8
        meets_threshold = overall_accuracy >= threshold

        rprint("\n📋 LEGEND:")
        rprint("✅ = Exact match")
        rprint("≈ = Partial match (substring found)")
        rprint("❌ = No match")
        rprint("➖ = Ground truth not available")
        rprint(
            f"\n📊 Meets accuracy threshold ({threshold * 100:.0f}%): {'✅ Yes' if meets_threshold else '❌ No'}"
        )
        rprint("=" * config['table_width'])

    def display_image(
        self,
        image_path: str,
        display_config: Optional[Dict] = None,
        width: int = 800
    ) -> None:
        """
        New Feature: Display the image being processed.

        Args:
            image_path: Path to the image file
            display_config: Display configuration options
            width: Display width in pixels
        """
        config = {**DEFAULT_DISPLAY_CONFIG, **(display_config or {})}

        if not config.get('show_images', True):
            return

        try:
            from IPython.display import Image, display
            from pathlib import Path

            image_file = Path(image_path)
            if image_file.exists():
                rprint(f"\n[bold blue]📸 Processing Image: {image_file.name}[/bold blue]")
                display(Image(str(image_path), width=width))
                rprint(f"[dim]📄 File: {image_file.name} | Size: {image_file.stat().st_size:,} bytes[/dim]")
            else:
                rprint(f"[red]❌ Image not found: {image_path}[/red]")

        except ImportError:
            rprint(f"[yellow]⚠️ IPython.display not available - showing path only: {image_path}[/yellow]")
        except Exception as e:
            rprint(f"[red]❌ Error displaying image {image_path}: {e}[/red]")


# Convenience functions for easy notebook use with enhanced capabilities
_display = InternVL3BatchDisplay()

# Configuration management
def set_display_config(**kwargs) -> Dict:
    """Set global display configuration options."""
    global DEFAULT_DISPLAY_CONFIG
    DEFAULT_DISPLAY_CONFIG.update(kwargs)
    return DEFAULT_DISPLAY_CONFIG

def get_display_config() -> Dict:
    """Get current display configuration."""
    return DEFAULT_DISPLAY_CONFIG.copy()

# Enhanced convenience functions with configuration support
def display_prompt_info(
    result: Dict[str, Any],
    document_type: str,
    show_full: Optional[bool] = None,
    display_config: Optional[Dict] = None
) -> None:
    """Display the prompt used for extraction with enhanced options."""
    _display.display_prompt_info(result, document_type, display_config, show_full)


def display_raw_and_cleaned(
    result: Dict[str, Any],
    show_full: Optional[bool] = None,
    display_config: Optional[Dict] = None
) -> None:
    """Display raw extracted text and cleaned text with enhanced options."""
    _display.display_raw_and_cleaned(result, display_config, show_full)


def display_field_comparison(
    result: Dict[str, Any],
    ground_truth: Dict[str, Any],
    document_type: str,
    show_full: Optional[bool] = None,
    display_config: Optional[Dict] = None
) -> None:
    """Display comprehensive fieldwise comparison with enhanced options."""
    _display.display_field_comparison(result, ground_truth, document_type, display_config, show_full)


def display_image(
    image_path: str,
    width: int = 800,
    display_config: Optional[Dict] = None
) -> None:
    """Display the image being processed."""
    _display.display_image(image_path, display_config, width)


# Batch display function for complete processing display
def display_complete_processing(
    image_path: str,
    result: Dict[str, Any],
    ground_truth: Dict[str, Any],
    document_type: str,
    show_full: bool = True,
    display_config: Optional[Dict] = None
) -> None:
    """Display complete processing information including image, prompts, and comparison."""
    config = {**DEFAULT_DISPLAY_CONFIG, **(display_config or {})}

    # Display image first
    display_image(image_path, display_config=config)

    # Display prompt information
    display_prompt_info(result, document_type, show_full=show_full, display_config=config)

    # Display raw and cleaned data
    display_raw_and_cleaned(result, show_full=show_full, display_config=config)

    # Display field comparison
    display_field_comparison(result, ground_truth, document_type, show_full=show_full, display_config=config)
