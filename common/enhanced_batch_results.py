"""
Enhanced Batch Results Analysis Module

Provides comprehensive analysis and CSV export functionality for batch processing results.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rich import print as rprint


class EnhancedBatchResults:
    """Enhanced batch results analyzer with comprehensive metrics."""

    def __init__(self):
        """Initialize the results analyzer."""
        pass

    def analyze_batch_results(self, batch_results: List[Dict]) -> Dict:
        """
        Analyze batch processing results and generate comprehensive statistics.

        Args:
            batch_results: List of processing result dictionaries

        Returns:
            Dictionary containing analysis results
        """
        rprint("[bold blue]📊 ENHANCED BATCH RESULTS ANALYSIS[/bold blue]")
        rprint("=" * 60)

        # Calculate basic statistics
        successful_results = [r for r in batch_results if "error" not in r]
        error_count = len(batch_results) - len(successful_results)

        analysis = {
            "total_images": len(batch_results),
            "successful_results": successful_results,
            "successful_count": len(successful_results),
            "error_count": error_count,
            "success_rate": len(successful_results) / len(batch_results) * 100 if batch_results else 0
        }

        if successful_results:
            # Field statistics
            total_fields = sum(r.get("field_count", r.get("found_fields", 0)) for r in successful_results)
            total_found = sum(r.get("found_fields", 0) for r in successful_results)
            avg_processing_time = sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results)
            total_processing_time = sum(r.get("processing_time", 0) for r in successful_results)

            analysis.update({
                "total_fields": total_fields,
                "total_found": total_found,
                "field_coverage": total_found / total_fields * 100 if total_fields > 0 else 0,
                "avg_processing_time": avg_processing_time,
                "total_processing_time": total_processing_time,
                "processing_speed": total_found / total_processing_time if total_processing_time > 0 else 0
            })

            # Document type breakdown
            doc_types = {}
            for result in successful_results:
                dt = result.get("document_type", "unknown")
                if dt not in doc_types:
                    doc_types[dt] = {"count": 0, "total_fields": 0, "found_fields": 0}
                doc_types[dt]["count"] += 1
                doc_types[dt]["total_fields"] += result.get("field_count", result.get("found_fields", 0))
                doc_types[dt]["found_fields"] += result.get("found_fields", 0)

            analysis["doc_types"] = doc_types

            # Accuracy statistics if available
            results_with_accuracy = [r for r in successful_results if r.get("evaluation", {}).get("overall_accuracy", 0) > 0]
            if results_with_accuracy:
                avg_accuracy = sum(r["evaluation"]["overall_accuracy"] for r in results_with_accuracy) / len(results_with_accuracy) * 100
                analysis["avg_accuracy"] = avg_accuracy
                analysis["accuracy_count"] = len(results_with_accuracy)

            # Display analysis
            self._display_analysis(analysis)

        else:
            rprint("[yellow]⚠️ No successful extractions to analyze[/yellow]")

        rprint("\n[cyan]💾 Results ready for advanced analytics processing...[/cyan]")
        rprint("[green]✅ Enhanced batch analysis complete - proceeding to comprehensive analytics[/green]")

        return analysis

    def _display_analysis(self, analysis: Dict) -> None:
        """Display the analysis results in a formatted way."""
        rprint(f"[green]✅ Successful extractions: {analysis['successful_count']}/{analysis['total_images']} ({analysis['success_rate']:.1f}%)[/green]")
        rprint(f"[red]❌ Errors: {analysis['error_count']}[/red]")
        rprint(f"[cyan]📊 Total fields processed: {analysis['total_fields']}[/cyan]")
        rprint(f"[cyan]🎯 Total fields found: {analysis['total_found']}[/cyan]")
        rprint(f"[cyan]📈 Overall field coverage: {analysis['field_coverage']:.1f}%[/cyan]")
        rprint(f"[cyan]⏱️ Average processing time: {analysis['avg_processing_time']:.2f}s per image[/cyan]")
        rprint(f"[cyan]⚡ Processing speed: {analysis['processing_speed']:.1f} fields/second[/cyan]")

        # Document type breakdown
        rprint("\n[bold blue]📋 Document type breakdown:[/bold blue]")
        for doc_type, stats in analysis["doc_types"].items():
            coverage = stats["found_fields"] / stats["total_fields"] * 100 if stats["total_fields"] > 0 else 0
            rprint(f"   📄 {doc_type}: {stats['count']} docs, {stats['found_fields']}/{stats['total_fields']} fields ({coverage:.1f}% coverage)")

        # Accuracy information if available
        if "avg_accuracy" in analysis:
            rprint("\n[bold blue]🎯 Accuracy Performance:[/bold blue]")
            rprint(f"   📈 Average accuracy: {analysis['avg_accuracy']:.2f}% (from {analysis['accuracy_count']} evaluated images)")

    def export_to_csv(
        self,
        batch_results: List[Dict],
        output_path: Path,
        timestamp: str,
        config: Dict = None
    ) -> Path:
        """
        Export batch results to comprehensive CSV format.

        Args:
            batch_results: List of processing result dictionaries
            output_path: Directory to save CSV file
            timestamp: Timestamp for filename
            config: Configuration dictionary with model info

        Returns:
            Path to saved CSV file
        """
        rprint("[bold blue]📊 ENHANCED CSV EXPORT[/bold blue]")

        # Create comprehensive CSV export path (dynamic based on model)
        if config is None:
            config = {}
        model_suffix = config.get('model_suffix', 'internvl3')
        csv_path = output_path / f"{model_suffix}_batch_results_{timestamp}.csv"

        # Prepare enhanced data for CSV export
        csv_data = []
        for result in batch_results:
            if "error" not in result:
                # Create a row with comprehensive metadata + all extracted fields
                row = {
                    "model": config.get('model_name', 'InternVL3'),
                    "image_file": result.get("image_file", result.get("image_name", "")),
                    "image_name": result.get("image_name", ""),
                    "document_type": result.get("document_type", ""),
                    "processing_time": result.get("processing_time", 0.0),
                    "field_count": result.get("field_count", 0),
                    "found_fields": result.get("found_fields", 0),
                    "field_coverage": result.get("found_fields", 0) / max(result.get("field_count", 1), 1) * 100,
                    "prompt_used": result.get("prompt_used", ""),
                    "timestamp": result.get("timestamp", "")
                }

                # Add evaluation metrics if available
                evaluation = result.get("evaluation", {})
                if evaluation and "overall_accuracy" in evaluation:
                    row["overall_accuracy"] = evaluation["overall_accuracy"] * 100
                    row["fields_extracted"] = evaluation.get("fields_extracted", 0)
                    row["fields_matched"] = evaluation.get("fields_matched", 0)
                    row["total_fields"] = evaluation.get("total_fields", 0)

                # Add all extracted fields as columns
                extracted_data = result.get("extracted_data", {})
                for field_name, field_value in extracted_data.items():
                    row[field_name] = field_value

                csv_data.append(row)

        # Save enhanced CSV
        if csv_data:
            results_df = pd.DataFrame(csv_data)
            results_df.to_csv(csv_path, index=False)
            rprint(f"[green]✅ Enhanced results saved to: {csv_path}[/green]")
            rprint(f"[cyan]📊 CSV contains {len(results_df)} rows and {len(results_df.columns)} columns[/cyan]")

            # Show enhanced sample results
            rprint("\n[bold blue]📋 Enhanced sample results (first 3 rows):[/bold blue]")
            display_cols = ["image_file", "document_type", "found_fields", "field_count", "field_coverage", "processing_time"]
            if "overall_accuracy" in results_df.columns:
                display_cols.append("overall_accuracy")
            available_cols = [col for col in display_cols if col in results_df.columns]
            rprint(results_df[available_cols].head(3).to_string(index=False))
        else:
            rprint("[yellow]⚠️ No data to save to CSV - check batch processing results[/yellow]")

        rprint("\n[cyan]💾 CSV export complete - file ready for external analysis[/cyan]")
        return csv_path

    def generate_analytics_dataframes(
        self,
        batch_results: List[Dict],
        processing_times: List[float],
        output_dirs: Dict[str, Path],
        timestamp: str,
        config: Dict
    ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate comprehensive analytics using BatchAnalytics module.

        Args:
            batch_results: List of processing result dictionaries
            processing_times: List of processing times
            output_dirs: Dictionary of output directories
            timestamp: Timestamp for filenames
            config: Configuration dictionary

        Returns:
            Tuple of (saved_files, df_results, df_summary, df_doctype_stats, df_field_stats)
        """
        from .batch_analytics import BatchAnalytics

        rprint("[bold blue]📊 COMPREHENSIVE ANALYTICS GENERATION[/bold blue]")
        rprint("[cyan]📊 Generating comprehensive analytics with BatchAnalytics...[/cyan]")

        # Create comprehensive analytics using BatchAnalytics
        analytics = BatchAnalytics(batch_results, processing_times)

        # Generate and save all DataFrames
        rprint("[cyan]💾 Creating and saving comprehensive DataFrames...[/cyan]")
        saved_files, df_results, df_summary, df_doctype_stats, df_field_stats = analytics.save_all_dataframes(
            output_dirs['csv'], timestamp, verbose=config['VERBOSE']
        )

        # Display enhanced summary statistics
        rprint("\n[bold blue]📊 ENHANCED RESULTS SUMMARY[/bold blue]")
        if len(df_results) > 0:
            # Display summary (would use display() in notebook)
            rprint("Summary statistics generated successfully")
        else:
            rprint("[yellow]⚠️ No results available for summary[/yellow]")

        # Display document type performance if available
        if not df_doctype_stats.empty:
            rprint("\n[bold blue]📋 DOCUMENT TYPE PERFORMANCE[/bold blue]")
            rprint("Document type statistics generated successfully")
        else:
            rprint("[yellow]⚠️ No document type statistics available[/yellow]")

        # Display field-level statistics if available
        if df_field_stats is not None and not df_field_stats.empty:
            rprint("\n[bold blue]🎯 FIELD-LEVEL ACCURACY STATISTICS[/bold blue]")
            rprint("Field-level accuracy statistics generated successfully")
        else:
            rprint("[yellow]⚠️ No field-level accuracy data available[/yellow]")

        # Analytics summary
        rprint(f"\n[green]✅ Analytics generated and saved to {output_dirs['csv']}[/green]")
        rprint("[cyan]📊 Available DataFrames:[/cyan]")
        for name, path in saved_files.items():
            rprint(f"   📄 {name}: {path.name}")

        rprint("\n[bold green]🚀 Ready for professional visualizations and reporting![/bold green]")

        return saved_files, df_results, df_summary, df_doctype_stats, df_field_stats


def analyze_and_export_results(
    batch_results: List[Dict],
    processing_times: List[float],
    output_dirs: Dict[str, Path],
    timestamp: str,
    config: Dict
) -> Tuple[Dict, Path, Tuple]:
    """
    Convenience function to analyze results and export to CSV.

    Args:
        batch_results: List of processing result dictionaries
        processing_times: List of processing times
        output_dirs: Dictionary of output directories
        timestamp: Timestamp for filenames
        config: Configuration dictionary

    Returns:
        Tuple of (analysis_results, csv_path, analytics_tuple)
    """
    analyzer = EnhancedBatchResults()

    # Analyze results
    analysis = analyzer.analyze_batch_results(batch_results)

    # Export to CSV
    csv_path = analyzer.export_to_csv(batch_results, output_dirs['csv'], timestamp, config)

    # Generate comprehensive analytics
    analytics_tuple = analyzer.generate_analytics_dataframes(
        batch_results, processing_times, output_dirs, timestamp, config
    )

    return analysis, csv_path, analytics_tuple