"""
Batch Analytics Module for Document Extraction Results

Creates comprehensive DataFrames and statistics from batch processing results.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rich import print as rprint


class BatchAnalytics:
    """Generate analytics DataFrames and statistics from batch results."""

    def __init__(self, batch_results: List[Dict], processing_times: List[float]):
        """
        Initialize analytics with batch results.

        Args:
            batch_results: List of extraction result dictionaries
            processing_times: List of processing times
        """
        self.batch_results = batch_results
        self.processing_times = processing_times
        self.successful_results = [r for r in batch_results if "error" not in r]

    def create_results_dataframe(self) -> pd.DataFrame:
        """
        Create main results DataFrame with extraction details.

        Returns:
            DataFrame with extraction results and field-level accuracies
        """
        results_data = []

        for result in self.successful_results:
            # Handle both inference-only and evaluation modes
            evaluation = result.get("evaluation", {})
            inference_only = evaluation.get("inference_only", False)

            row = {
                "image_name": result["image_name"],
                "document_type": result["document_type"],
                "overall_accuracy": evaluation.get("overall_accuracy", 0)
                * 100 if not inference_only else None,
                "fields_extracted": evaluation.get("fields_extracted", 0),
                "fields_matched": evaluation.get("fields_matched", 0),
                "total_fields": evaluation.get("total_fields", 0),
                "processing_time": result["processing_time"],
                "prompt_used": result["prompt_used"],
                "inference_only": inference_only,
            }

            # Add individual field accuracies (only in evaluation mode)
            if not inference_only and "field_accuracies" in evaluation:
                for field, data in evaluation["field_accuracies"].items():
                    row[f"accuracy_{field}"] = data.get("accuracy", 0) * 100

            results_data.append(row)

        return pd.DataFrame(results_data)

    def create_summary_statistics(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics DataFrame.

        Args:
            df_results: Results DataFrame

        Returns:
            DataFrame with summary statistics
        """
        # Check if we're in inference-only mode
        inference_only = df_results.get("inference_only", pd.Series([False])).any() if len(df_results) > 0 else False

        summary_stats = {
            "Total Images": len(self.batch_results),
            "Successful Extractions": len(self.successful_results),
            "Failed Extractions": len(self.batch_results)
            - len(self.successful_results),
        }

        # Only include accuracy metrics in evaluation mode
        if not inference_only and len(df_results) > 0:
            accuracy_series = df_results["overall_accuracy"].dropna()
            if len(accuracy_series) > 0:
                summary_stats.update({
                    "Average Accuracy (%)": accuracy_series.mean(),
                    "Median Accuracy (%)": accuracy_series.median(),
                    "Min Accuracy (%)": accuracy_series.min(),
                    "Max Accuracy (%)": accuracy_series.max(),
                })
        elif inference_only and len(df_results) > 0:
            # In inference-only mode, show field extraction statistics instead
            summary_stats.update({
                "Average Fields Found": df_results["fields_extracted"].mean(),
                "Min Fields Found": df_results["fields_extracted"].min(),
                "Max Fields Found": df_results["fields_extracted"].max(),
            })

        # Add processing time statistics for both modes
        summary_stats.update({
            "Average Processing Time (s)": np.mean(self.processing_times)
            if self.processing_times
            else 0,
            "Total Processing Time (s)": sum(self.processing_times)
            if self.processing_times
            else 0,
            "Throughput (images/min)": 60 / np.mean(self.processing_times)
            if self.processing_times
            else 0,
        })

        df_summary = pd.DataFrame([summary_stats]).T
        df_summary.columns = ["Value"]
        return df_summary

    def create_doctype_statistics(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create document type statistics DataFrame.

        Args:
            df_results: Results DataFrame

        Returns:
            DataFrame with statistics grouped by document type
        """
        if len(df_results) == 0:
            return pd.DataFrame()

        df_stats = (
            df_results.groupby("document_type")
            .agg(
                {
                    "overall_accuracy": ["mean", "median", "std", "min", "max"],
                    "processing_time": ["mean", "median"],
                    "image_name": "count",
                }
            )
            .round(2)
        )

        df_stats.columns = [
            "_".join(col).strip() for col in df_stats.columns.to_numpy()
        ]
        df_stats = df_stats.rename(columns={"image_name_count": "count"})

        return df_stats

    def create_field_statistics(
        self, df_results: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Create field-level accuracy statistics.

        Args:
            df_results: Results DataFrame

        Returns:
            DataFrame with field-level statistics or None if no field data
        """
        if len(df_results) == 0:
            return None

        # Extract field accuracy columns
        field_cols = [col for col in df_results.columns if col.startswith("accuracy_")]

        if not field_cols:
            return None

        # Create field accuracy matrix
        field_accuracy_data = df_results[field_cols]
        field_accuracy_data.columns = [
            col.replace("accuracy_", "") for col in field_accuracy_data.columns
        ]

        # Calculate statistics
        field_stats = pd.DataFrame(
            {
                "Average Accuracy (%)": field_accuracy_data.mean(),
                "Min Accuracy (%)": field_accuracy_data.min(),
                "Max Accuracy (%)": field_accuracy_data.max(),
                "Std Dev (%)": field_accuracy_data.std(),
            }
        ).round(2)

        return field_stats.sort_values("Average Accuracy (%)", ascending=False)

    def save_all_dataframes(
        self, output_dir: Path, timestamp: str, verbose: bool = True
    ) -> Dict[str, Path]:
        """
        Save all DataFrames to CSV files.

        Args:
            output_dir: Directory to save CSV files
            timestamp: Timestamp for filenames
            verbose: Whether to print save confirmations

        Returns:
            Dictionary mapping DataFrame names to saved file paths
        """
        saved_files = {}

        # Create DataFrames
        df_results = self.create_results_dataframe()
        df_summary = self.create_summary_statistics(df_results)
        df_doctype_stats = self.create_doctype_statistics(df_results)
        df_field_stats = self.create_field_statistics(df_results)

        # Save DataFrames
        csv_prefix = output_dir / f"batch_{timestamp}"

        # Save results
        results_path = Path(f"{csv_prefix}_results.csv")
        df_results.to_csv(results_path, index=False)
        saved_files["results"] = results_path

        # Save summary
        summary_path = Path(f"{csv_prefix}_summary.csv")
        df_summary.to_csv(summary_path)
        saved_files["summary"] = summary_path

        # Save doctype stats if available
        if not df_doctype_stats.empty:
            doctype_path = Path(f"{csv_prefix}_doctype_stats.csv")
            df_doctype_stats.to_csv(doctype_path)
            saved_files["doctype_stats"] = doctype_path

        # Save field stats if available
        if df_field_stats is not None:
            field_path = Path(f"{csv_prefix}_field_stats.csv")
            df_field_stats.to_csv(field_path)
            saved_files["field_stats"] = field_path

        if verbose:
            rprint(f"[green]✅ DataFrames saved to {output_dir}[/green]")

        return saved_files, df_results, df_summary, df_doctype_stats, df_field_stats
