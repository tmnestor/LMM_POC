"""
InternVL3 Batch Display Module

Lightweight display functions for InternVL3 batch processing notebooks.
Provides the 5 key debugging/analysis features without causing notebook bloat.
"""

from typing import Any, Dict, Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class InternVL3BatchDisplay:
    """Display functions for InternVL3 batch processing results."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize with optional console."""
        self.console = console or Console()
    
    def display_prompt_info(self, result: Dict[str, Any], document_type: str) -> None:
        """
        Feature 1: Display the prompt used for extraction.
        
        Args:
            result: Processing result dictionary
            document_type: Detected document type
        """
        # Get prompt info from processor result
        prompt_info = result.get("prompt_info", {})
        
        if not prompt_info:
            # Fallback - show basic prompt info
            rprint(f"[cyan]🔧 Prompt: Document-aware {document_type} extraction[/cyan]")
            return
        
        # Create prompt display panel
        prompt_text = prompt_info.get("prompt", "No prompt captured")
        prompt_source = prompt_info.get("source", "Unknown")
        field_count = prompt_info.get("field_count", 0)
        
        panel_content = f"""[bold cyan]Document Type:[/bold cyan] {document_type}
[bold cyan]Field Count:[/bold cyan] {field_count} fields
[bold cyan]Prompt Source:[/bold cyan] {prompt_source}

[bold yellow]Prompt Text:[/bold yellow]
{prompt_text[:200]}{"..." if len(prompt_text) > 200 else ""}"""
        
        self.console.print(Panel(
            panel_content,
            title="🔧 Extraction Prompt",
            border_style="cyan"
        ))
    
    def display_raw_and_cleaned(self, result: Dict[str, Any]) -> None:
        """
        Feature 2 & 3: Display raw extracted text and cleaned text.
        
        Args:
            result: Processing result dictionary
        """
        raw_response = result.get("raw_response", "No raw response captured")
        extracted_data = result.get("extracted_data", {})
        
        # Create two-column comparison
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Raw Model Response", style="yellow", width=60)
        table.add_column("Cleaned/Parsed Data", style="green", width=60)
        
        # Format raw response (truncate if too long)
        if len(raw_response) > 300:
            raw_display = raw_response[:300] + "\n\n[dim]... (truncated for display)[/dim]"
        else:
            raw_display = raw_response
        
        # Format cleaned data
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
        
        table.add_row(raw_display, cleaned_display)
        
        rprint("\n")
        self.console.print(Panel(
            table,
            title="📝 Raw Response vs Cleaned Data",
            border_style="blue"
        ))
    
    def display_field_comparison(
        self, 
        result: Dict[str, Any], 
        ground_truth: Dict[str, Any],
        document_type: str
    ) -> None:
        """
        Feature 4: Display fieldwise comparison between extracted and ground truth.
        
        Args:
            result: Processing result dictionary
            ground_truth: Ground truth data for this image
            document_type: Document type for context
        """
        if not ground_truth:
            rprint("[yellow]⚠️ No ground truth available for comparison[/yellow]")
            return
        
        extracted_data = result.get("extracted_data", {})
        
        # Create comparison table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Field", style="cyan", width=25)
        table.add_column("Extracted", style="yellow", width=35)
        table.add_column("Ground Truth", style="green", width=35)
        table.add_column("Match", style="bold", width=10)
        
        matches = 0
        total_fields = 0
        
        for field in extracted_data.keys():
            extracted_value = extracted_data.get(field, "NOT_FOUND")
            gt_value = ground_truth.get(field, "NOT_AVAILABLE")
            
            # Simple exact match check
            if extracted_value != "NOT_FOUND" and gt_value != "NOT_AVAILABLE":
                total_fields += 1
                is_match = str(extracted_value).strip().lower() == str(gt_value).strip().lower()
                if is_match:
                    matches += 1
                    match_icon = "✅"
                    match_style = "green"
                else:
                    match_icon = "❌"
                    match_style = "red"
            elif extracted_value == "NOT_FOUND":
                total_fields += 1
                match_icon = "❌"
                match_style = "red"
            else:
                # GT not available, skip comparison
                match_icon = "➖"
                match_style = "dim"
            
            # Truncate long values for display
            extracted_display = str(extracted_value)[:30] + "..." if len(str(extracted_value)) > 30 else str(extracted_value)
            gt_display = str(gt_value)[:30] + "..." if len(str(gt_value)) > 30 else str(gt_value)
            
            table.add_row(
                field,
                extracted_display,
                gt_display,
                f"[{match_style}]{match_icon}[/{match_style}]"
            )
        
        # Calculate accuracy
        accuracy = (matches / total_fields * 100) if total_fields > 0 else 0
        
        rprint("\n")
        self.console.print(Panel(
            table,
            title=f"📊 Field Comparison - {document_type} ({accuracy:.1f}% accuracy)",
            border_style="blue"
        ))
        
        # Summary stats
        rprint(f"[bold cyan]📈 Summary: {matches}/{total_fields} fields correct ({accuracy:.1f}% accuracy)[/bold cyan]")


# Convenience functions for easy notebook use
_display = InternVL3BatchDisplay()

def display_prompt_info(result: Dict[str, Any], document_type: str) -> None:
    """Display the prompt used for extraction."""
    _display.display_prompt_info(result, document_type)

def display_raw_and_cleaned(result: Dict[str, Any]) -> None:
    """Display raw extracted text and cleaned text."""
    _display.display_raw_and_cleaned(result)

def display_field_comparison(
    result: Dict[str, Any], 
    ground_truth: Dict[str, Any],
    document_type: str
) -> None:
    """Display fieldwise comparison between extracted and ground truth."""
    _display.display_field_comparison(result, ground_truth, document_type)