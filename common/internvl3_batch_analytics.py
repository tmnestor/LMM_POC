"""
InternVL3 Batch Analytics Module

Simple analytics and dashboard generation for InternVL3 batch processing.
Provides summary dashboard (Feature 5) without complex dependencies.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich import print as rprint
from rich.console import Console
from rich.table import Table


class InternVL3BatchAnalytics:
    """Simple analytics for InternVL3 batch processing results."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize with optional console."""
        self.console = console or Console()
    
    def generate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from batch results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful = [r for r in results if "error" not in r and "extracted_data" in r]
        
        # Calculate basic stats
        total_images = len(results)
        successful_count = len(successful)
        error_count = total_images - successful_count
        
        # Processing time stats
        processing_times = [r.get("processing_time", 0) for r in successful]
        avg_time = np.mean(processing_times) if processing_times else 0
        total_time = sum(processing_times)
        
        # Field extraction stats
        total_fields_processed = 0
        total_fields_found = 0
        
        for result in successful:
            extracted_data = result.get("extracted_data", {})
            total_fields_processed += len(extracted_data)
            found_count = len([v for v in extracted_data.values() if v != "NOT_FOUND"])
            total_fields_found += found_count
        
        # Overall accuracy
        overall_accuracy = (total_fields_found / total_fields_processed * 100) if total_fields_processed > 0 else 0
        
        # Document type breakdown
        doc_types = {}
        for result in successful:
            doc_type = result.get("document_type", "unknown")
            if doc_type not in doc_types:
                doc_types[doc_type] = {"count": 0, "fields_found": 0, "fields_total": 0}
            
            doc_types[doc_type]["count"] += 1
            
            extracted_data = result.get("extracted_data", {})
            doc_types[doc_type]["fields_total"] += len(extracted_data)
            found_count = len([v for v in extracted_data.values() if v != "NOT_FOUND"])
            doc_types[doc_type]["fields_found"] += found_count
        
        return {
            "total_images": total_images,
            "successful_extractions": successful_count,
            "failed_extractions": error_count,
            "success_rate": (successful_count / total_images * 100) if total_images > 0 else 0,
            "average_processing_time": avg_time,
            "total_processing_time": total_time,
            "throughput_images_per_minute": (total_images / total_time * 60) if total_time > 0 else 0,
            "total_fields_processed": total_fields_processed,
            "total_fields_found": total_fields_found,
            "overall_accuracy": overall_accuracy,
            "document_types": doc_types
        }
    
    def display_summary_table(self, summary_stats: Dict[str, Any]) -> None:
        """
        Display summary statistics in a nice table.
        
        Args:
            summary_stats: Summary statistics dictionary
        """
        if "error" in summary_stats:
            rprint(f"[red]❌ {summary_stats['error']}[/red]")
            return
        
        # Create summary table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", width=20)
        
        # Add rows
        table.add_row("Total Images", str(summary_stats["total_images"]))
        table.add_row("Successful Extractions", str(summary_stats["successful_extractions"]))
        table.add_row("Failed Extractions", str(summary_stats["failed_extractions"]))
        table.add_row("Success Rate", f"{summary_stats['success_rate']:.1f}%")
        table.add_row("", "")  # Separator
        table.add_row("Avg Processing Time", f"{summary_stats['average_processing_time']:.2f}s")
        table.add_row("Total Processing Time", f"{summary_stats['total_processing_time']:.2f}s")
        table.add_row("Throughput", f"{summary_stats['throughput_images_per_minute']:.1f} img/min")
        table.add_row("", "")  # Separator
        table.add_row("Total Fields Processed", str(summary_stats["total_fields_processed"]))
        table.add_row("Total Fields Found", str(summary_stats["total_fields_found"]))
        table.add_row("Overall Field Accuracy", f"{summary_stats['overall_accuracy']:.1f}%")
        
        self.console.print(table)
        
        # Document type breakdown
        if summary_stats["document_types"]:
            rprint("\n[bold blue]📋 Document Type Breakdown:[/bold blue]")
            doc_table = Table(show_header=True, header_style="bold blue")
            doc_table.add_column("Document Type", style="cyan")
            doc_table.add_column("Count", style="green")
            doc_table.add_column("Field Accuracy", style="yellow")
            
            for doc_type, stats in summary_stats["document_types"].items():
                accuracy = (stats["fields_found"] / stats["fields_total"] * 100) if stats["fields_total"] > 0 else 0
                doc_table.add_row(
                    doc_type,
                    str(stats["count"]),
                    f"{accuracy:.1f}%"
                )
            
            self.console.print(doc_table)
    
    def create_simple_dashboard(
        self, 
        results: List[Dict[str, Any]], 
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Optional[Path]:
        """
        Feature 5: Create a simple dashboard visualization.
        
        Args:
            results: List of processing results
            save_path: Path to save the dashboard
            show: Whether to display the dashboard
            
        Returns:
            Path to saved dashboard or None
        """
        if not results:
            rprint("[red]❌ No results to visualize[/red]")
            return None
        
        # Generate summary stats
        summary_stats = self.generate_summary_stats(results)
        if "error" in summary_stats:
            rprint(f"[red]❌ Cannot create dashboard: {summary_stats['error']}[/red]")
            return None
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('InternVL3 Batch Processing Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Pie Chart
        success_data = [summary_stats["successful_extractions"], summary_stats["failed_extractions"]]
        success_labels = ['Successful', 'Failed']
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(success_data, labels=success_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Extraction Success Rate')
        
        # 2. Processing Time Distribution
        successful_results = [r for r in results if "error" not in r and "processing_time" in r]
        if successful_results:
            times = [r["processing_time"] for r in successful_results]
            ax2.hist(times, bins=min(10, len(times)), color='#3498db', alpha=0.7, edgecolor='black')
            ax2.set_title('Processing Time Distribution')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.2f}s')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Processing Time Distribution')
        
        # 3. Document Type Distribution
        doc_types = summary_stats["document_types"]
        if doc_types:
            doc_names = list(doc_types.keys())
            doc_counts = [doc_types[dt]["count"] for dt in doc_names]
            bars = ax3.bar(doc_names, doc_counts, color=['#9b59b6', '#f39c12', '#1abc9c'])
            ax3.set_title('Document Type Distribution')
            ax3.set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No document type data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Document Type Distribution')
        
        # 4. Field Accuracy by Document Type
        if doc_types:
            doc_names = list(doc_types.keys())
            accuracies = []
            for dt in doc_names:
                stats = doc_types[dt]
                accuracy = (stats["fields_found"] / stats["fields_total"] * 100) if stats["fields_total"] > 0 else 0
                accuracies.append(accuracy)
            
            bars = ax4.bar(doc_names, accuracies, color=['#e67e22', '#27ae60', '#8e44ad'])
            ax4.set_title('Field Accuracy by Document Type')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies, strict=False):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.1f}%', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Field Accuracy by Document Type')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            rprint(f"[green]✅ Dashboard saved to: {save_path}[/green]")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path


# Convenience functions for easy notebook use
_analytics = InternVL3BatchAnalytics()

def generate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from batch results."""
    return _analytics.generate_summary_stats(results)

def display_summary_table(summary_stats: Dict[str, Any]) -> None:
    """Display summary statistics in a nice table."""
    _analytics.display_summary_table(summary_stats)

def create_simple_dashboard(
    results: List[Dict[str, Any]], 
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[Path]:
    """Create a simple dashboard visualization."""
    return _analytics.create_simple_dashboard(results, save_path, show)