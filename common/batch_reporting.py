"""
Batch Reporting Module for Document Extraction Results

Generates executive summaries and comprehensive reports in various formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rich import print as rprint


class BatchReporter:
    """Generate comprehensive reports from batch processing results."""

    def __init__(
        self,
        batch_results: List[Dict],
        processing_times: List[float],
        document_types_found: Dict[str, int],
        timestamp: str,
    ):
        """
        Initialize reporter with batch results.

        Args:
            batch_results: List of extraction result dictionaries
            processing_times: List of processing times
            document_types_found: Dictionary of document type counts
            timestamp: Batch timestamp
        """
        self.batch_results = batch_results
        self.processing_times = processing_times
        self.document_types_found = document_types_found
        self.timestamp = timestamp
        self.successful_results = [r for r in batch_results if "error" not in r]

    def generate_executive_summary(
        self,
        df_results: pd.DataFrame,
        df_doctype_stats: pd.DataFrame,
        output_base: Path,
    ) -> str:
        """
        Generate executive summary in Markdown format.

        Args:
            df_results: Results DataFrame
            df_doctype_stats: Document type statistics
            output_base: Base output directory path

        Returns:
            Markdown formatted executive summary
        """
        # Calculate key metrics
        total_images = len(self.batch_results)
        successful_extractions = len(self.successful_results)

        # Check if we're in inference-only mode
        inference_only = (
            df_results.get("inference_only", pd.Series([False])).any()
            if len(df_results) > 0
            else False
        )

        if not inference_only and len(df_results) > 0:
            # Evaluation mode - calculate accuracy metrics
            accuracy_series = df_results["overall_accuracy"].dropna()
            avg_accuracy = accuracy_series.mean() if len(accuracy_series) > 0 else 0

            # Determine deployment readiness
            if avg_accuracy >= 95:
                readiness = "✅ **Production Ready**"
            elif avg_accuracy >= 80:
                readiness = "🟡 **Pilot Ready**"
            else:
                readiness = "🔴 **Needs Improvement**"
        else:
            # Inference-only mode - show extraction metrics instead
            avg_accuracy = None
            avg_fields_found = (
                df_results["fields_extracted"].mean() if len(df_results) > 0 else 0
            )
            readiness = (
                f"📋 **Inference-Only Mode** (Avg: {avg_fields_found:.1f} fields found)"
            )

        total_time = sum(self.processing_times) if self.processing_times else 0
        throughput = 60 / np.mean(self.processing_times) if self.processing_times else 0

        # Extract model info from timestamp for dynamic title and model name
        model_name = "Vision Model"
        model_version = "Unknown Version"

        if self.timestamp.startswith("internvl3_"):
            model_name = "InternVL3"
            model_version = (
                "InternVL3-8B"  # Default to 8B, could be enhanced to detect 2B
            )
        elif self.timestamp.startswith("llama_") or "llama" in self.timestamp.lower():
            model_name = "Llama Vision"
            model_version = "Llama-3.2-11B-Vision-Instruct"

        # Build report
        report = f"""# {model_name} Batch Processing Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Batch ID:** {self.timestamp}
**Model:** {model_version}

## Executive Summary

### Overall Performance
- **Total Images Processed:** {total_images}
- **Successful Extractions:** {successful_extractions} ({successful_extractions / total_images * 100:.1f}%)"""

        if not inference_only and avg_accuracy is not None:
            report += f"""
- **Average Accuracy:** {avg_accuracy:.2f}%"""

        report += f"""
- **Status:** {readiness}

### Processing Efficiency
- **Total Processing Time:** {total_time:.2f} seconds ({total_time / 60:.1f} minutes)
- **Average Time per Image:** {np.mean(self.processing_times):.2f} seconds
- **Throughput:** {throughput:.1f} images/minute

### Document Type Distribution
"""

        # Add document type distribution
        for doc_type, count in self.document_types_found.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            report += f"- **{doc_type}:** {count} ({percentage:.1f}%)\n"

        # Add accuracy by document type (only in evaluation mode)
        if not inference_only and not df_doctype_stats.empty:
            report += "\n### Accuracy by Document Type\n"
            for doc_type in df_doctype_stats.index:
                mean_acc = df_doctype_stats.loc[doc_type, "overall_accuracy_mean"]
                report += f"- **{doc_type}:** {mean_acc:.2f}%\n"

        # Add top performing images (only in evaluation mode)
        if not inference_only and len(df_results) > 0:
            accuracy_results = df_results.dropna(subset=["overall_accuracy"])
            if len(accuracy_results) > 0:
                top_performers = accuracy_results.nlargest(
                    min(5, len(accuracy_results)), "overall_accuracy"
                )[["image_name", "overall_accuracy", "document_type"]]

                report += "\n### Top Performing Images\n"
                for _, row in top_performers.iterrows():
                    report += f"- {row['image_name']}: {row['overall_accuracy']:.1f}% ({row['document_type']})\n"
        elif inference_only and len(df_results) > 0:
            # In inference-only mode, show top field extraction results
            top_extractors = df_results.nlargest(
                min(5, len(df_results)), "fields_extracted"
            )[["image_name", "fields_extracted", "document_type"]]

            report += "\n### Best Field Extraction Results\n"
            for _, row in top_extractors.iterrows():
                report += f"- {row['image_name']}: {row['fields_extracted']} fields extracted ({row['document_type']})\n"

        # Add areas for improvement (only in evaluation mode)
        if not inference_only and len(df_results) > 0:
            accuracy_results = df_results.dropna(subset=["overall_accuracy"])
            if len(accuracy_results) > 0:
                poor_performers = accuracy_results.nsmallest(
                    min(5, len(accuracy_results)), "overall_accuracy"
                )[["image_name", "overall_accuracy", "document_type"]]

                if poor_performers["overall_accuracy"].min() < 80:
                    report += "\n### Areas for Improvement\n"
                    for _, row in poor_performers.iterrows():
                        report += f"- {row['image_name']}: {row['overall_accuracy']:.1f}% ({row['document_type']})\n"

        # Add output files section
        report += f"""\n## Output Files Generated

All results have been saved to: `{output_base}`

- **CSV Files:** `csv/batch_{self.timestamp}_*.csv`
- **Visualizations:** `visualizations/*_{self.timestamp}.png`
- **Full Report:** `reports/batch_report_{self.timestamp}.md`

## Technical Details

- **V100 Optimizations:** Enabled (ResilientGenerator, Memory Cleanup)
- **Quantization:** 8-bit with BitsAndBytesConfig
- **Max Tokens:** 4000
- **Device:** CUDA (auto-mapped)
"""

        return report

    def generate_json_report(
        self,
        df_summary: pd.DataFrame,
        model_path: str,
        batch_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive JSON report.

        Args:
            df_summary: Summary statistics DataFrame
            model_path: Path to model
            batch_config: Batch configuration

        Returns:
            Dictionary with complete batch results
        """
        # Extract summary values
        summary_dict = df_summary["Value"].to_dict()

        # Calculate metrics
        total_images = len(self.batch_results)
        successful_extractions = len(self.successful_results)
        avg_accuracy = summary_dict.get("Average Accuracy (%)", 0)
        total_time = summary_dict.get("Total Processing Time (s)", 0)

        # Determine readiness
        if avg_accuracy >= 95:
            readiness = "Production Ready"
        elif avg_accuracy >= 80:
            readiness = "Pilot Ready"
        else:
            readiness = "Needs Improvement"

        # Build JSON structure
        export_data = {
            "metadata": {
                "batch_id": self.timestamp,
                "timestamp": datetime.now().isoformat(),
                "model": model_path,
                "total_images": total_images,
                "successful_extractions": successful_extractions,
                "average_accuracy": avg_accuracy,
                "total_processing_time": total_time,
                "deployment_status": readiness,
            },
            "configuration": {
                "data_directory": batch_config.get("data_dir"),
                "ground_truth": batch_config.get("ground_truth"),
                "max_images": batch_config.get("max_images"),
                "document_types": batch_config.get("document_types"),
                "flash_attn": batch_config.get("flash_attn"),
                "dtype": batch_config.get("dtype"),
            },
            "summary_statistics": summary_dict,
            "document_type_distribution": self.document_types_found,
            "results": [],
        }

        # Add individual results (excluding large extraction data)
        for result in self.successful_results:
            # Handle both inference-only and evaluation modes
            evaluation = result.get("evaluation", {})
            inference_only = evaluation.get("inference_only", False)

            # In inference-only mode, we may not have evaluation data at all
            if not evaluation:
                # Fallback for inference-only results without evaluation key
                export_result = {
                    "image_name": result["image_name"],
                    "document_type": result["document_type"],
                    "prompt_used": result["prompt_used"],
                    "processing_time": result["processing_time"],
                    "evaluation_summary": {
                        "overall_accuracy": None,
                        "fields_extracted": 0,  # Would need to count from extraction_result
                        "fields_matched": None,
                        "total_fields": None,
                        "inference_only": True,
                    },
                }
            else:
                export_result = {
                    "image_name": result["image_name"],
                    "document_type": result["document_type"],
                    "prompt_used": result["prompt_used"],
                    "processing_time": result["processing_time"],
                    "evaluation_summary": {
                        "overall_accuracy": evaluation.get("overall_accuracy", 0)
                        if not inference_only
                        else None,
                        "fields_extracted": evaluation.get("fields_extracted", 0),
                        "fields_matched": evaluation.get("fields_matched", 0),
                        "total_fields": evaluation.get("total_fields", 0),
                        "inference_only": inference_only,
                    },
                }
            export_data["results"].append(export_result)

        return export_data

    def save_all_reports(
        self,
        output_dirs: Dict[str, Path],
        df_results: pd.DataFrame,
        df_summary: pd.DataFrame,
        df_doctype_stats: pd.DataFrame,
        model_path: str,
        batch_config: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[str, Path]:
        """
        Save all reports to files.

        Args:
            output_dirs: Dictionary of output directories
            df_results: Results DataFrame
            df_summary: Summary statistics DataFrame
            df_doctype_stats: Document type statistics
            model_path: Path to model
            batch_config: Batch configuration
            verbose: Whether to print save confirmations

        Returns:
            Dictionary mapping report types to saved paths
        """
        saved_files = {}

        # Generate and save Markdown report
        markdown_report = self.generate_executive_summary(
            df_results, df_doctype_stats, output_dirs["base"]
        )

        report_path = output_dirs["reports"] / f"batch_report_{self.timestamp}.md"
        with report_path.open("w") as f:
            f.write(markdown_report)
        saved_files["markdown_report"] = report_path

        if verbose:
            rprint(f"[green]✅ Executive summary saved to {report_path}[/green]")

        # Generate and save JSON report
        json_report = self.generate_json_report(df_summary, model_path, batch_config)

        json_path = output_dirs["batch"] / f"batch_results_{self.timestamp}.json"
        with json_path.open("w") as f:
            json.dump(json_report, f, indent=2, default=str)
        saved_files["json_report"] = json_path

        if verbose:
            rprint(f"[green]✅ Complete results exported to {json_path}[/green]")

        return saved_files
