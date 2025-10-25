"""
Extract field-level accuracy from batch CSV results without re-running models.

This module provides functionality to analyze already-generated batch extraction
CSV files and compute per-field accuracy metrics by comparing against ground truth.
"""

from pathlib import Path

import pandas as pd

from common.evaluation_metrics import calculate_field_accuracy, load_ground_truth


def extract_field_level_accuracy_from_csv(
    output_dir: str, pattern: str, model_name: str, ground_truth_path: str
) -> pd.DataFrame:
    """
    Extract field-level accuracy by comparing CSV batch results against ground truth.

    This function:
    1. Loads batch results CSV (which has extracted field values)
    2. Loads ground truth CSV
    3. Compares field-by-field using the SAME evaluation logic as llama_batch.ipynb
    4. Returns per-field accuracy WITHOUT re-running document-type filtering

    Args:
        output_dir: Directory containing batch results CSV files
        pattern: Glob pattern to match files
        model_name: Name of the model
        ground_truth_path: Path to ground truth CSV

    Returns:
        DataFrame with columns: model, field_name, accuracy, correct_count, total_count
    """
    import sys

    sys.path.insert(0, str(Path.cwd()))

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    files = list(output_path.glob(pattern))

    if not files:
        print(f"[yellow] ⚠ No batch results found: {pattern}[/yellow]")
        print(f"[dim]  Searched in: {output_path}[/dim]")
        return pd.DataFrame()

    # Get the most recent file
    latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
    print(f"[cyan]□ Evaluating {model_name} from: {Path(latest_file).name}[/cyan]")

    try:
        # Load batch results CSV
        batch_df = pd.read_csv(latest_file)
        total_images = len(batch_df)
        print(f"[dim]  DEBUG: Loaded {total_images} rows from batch CSV[/dim]")

        # Load ground truth
        gt_path = Path(ground_truth_path)
        if not gt_path.exists():
            print(f"[red] ❌ Ground truth not found: {ground_truth_path}[/red]")
            return pd.DataFrame()

        ground_truth_map_raw = load_ground_truth(
            str(gt_path), show_sample=False, verbose=False
        )

        # Normalize ground truth keys to stems (remove extensions)
        # The ground_truth_map keys might be "image_001", "image_001.png", or "image_001.jpeg"
        # We normalize ALL keys to stem format for consistent matching
        ground_truth_map = {}
        for key, value in ground_truth_map_raw.items():
            stem_key = Path(key).stem
            ground_truth_map[stem_key] = value

        print(f"[dim]  DEBUG: Ground truth has {len(ground_truth_map)} entries[/dim]")

        # Determine which column to use for matching
        # The identifier column might be "image_file" or "image_name" in batch CSV
        image_col = "image_file" if "image_file" in batch_df.columns else "image_name"

        # DEBUG: Show sample image names (both normalized to stem format)
        sample_batch_names = [Path(x).stem for x in batch_df[image_col].head(3)]
        sample_gt_names = list(ground_truth_map.keys())[:3]
        print(
            f"[dim]  DEBUG: Sample batch stems (from {image_col}): {sample_batch_names}[/dim]"
        )
        print(f"[dim]  DEBUG: Sample ground truth stems: {sample_gt_names}[/dim]")

        # FILTER: Only evaluate images that have ground truth
        # Both batch CSV and ground truth are normalized to stem (no extension) for matching
        batch_df["image_stem"] = batch_df[image_col].apply(lambda x: Path(x).stem)
        batch_df_filtered = batch_df[
            batch_df["image_stem"].isin(ground_truth_map.keys())
        ]

        filtered_count = len(batch_df_filtered)
        skipped_count = total_images - filtered_count

        print(
            f"[dim]  DEBUG: Filtered to {filtered_count} matching images (skipped {skipped_count})[/dim]"
        )

        if filtered_count == 0:
            print("[red] ❌ No images in batch match ground truth entries[/red]")
            return pd.DataFrame()

        print(
            f"[cyan]📊 Evaluating {filtered_count}/{total_images} images with ground truth[/cyan]"
        )

        # Track field accuracies - accumulate across all images
        field_accuracies = {}

        # Get all possible field columns (exclude metadata columns)
        metadata_cols = [
            "image_file",
            "image_name",
            "document_type",
            "processing_time",
            "field_count",
            "found_fields",
            "field_coverage",
            "prompt_used",
            "timestamp",
            "overall_accuracy",
            "fields_extracted",
            "fields_matched",
            "total_fields",
            "inference_only",
            "model",
            "image_stem",
        ]
        field_columns = [
            col for col in batch_df_filtered.columns if col not in metadata_cols
        ]

        print(
            f"[dim]  DEBUG: Found {len(field_columns)} field columns to evaluate[/dim]"
        )

        # Evaluate each image
        for _, row in batch_df_filtered.iterrows():
            image_identifier = row["image_stem"]

            # Get ground truth for this image (use image_identifier as key)
            gt_data = ground_truth_map.get(image_identifier)
            if not gt_data:
                continue

            # Compare each field
            for field_name in field_columns:
                extracted_value = row.get(field_name, "NOT_FOUND")
                ground_truth_value = gt_data.get(field_name, "NOT_FOUND")

                # Skip if both are NOT_FOUND (field not applicable)
                if (
                    str(extracted_value).upper() == "NOT_FOUND"
                    and str(ground_truth_value).upper() == "NOT_FOUND"
                ):
                    continue

                # Calculate accuracy score using the same logic as batch notebooks
                accuracy_score = calculate_field_accuracy(
                    extracted_value, ground_truth_value, field_name, debug=False
                )

                # Initialize field tracking if needed
                if field_name not in field_accuracies:
                    field_accuracies[field_name] = {"correct": 0.0, "total": 0}

                field_accuracies[field_name]["total"] += 1
                field_accuracies[field_name]["correct"] += accuracy_score

        print(
            f"[dim]  DEBUG: Computed accuracies for {len(field_accuracies)} fields[/dim]"
        )

        if not field_accuracies:
            print("[yellow]⚠ No field-level accuracy data available[/yellow]")
            print(
                "[yellow]💡 This requires batch results CSVs and ground truth for evaluation[/yellow]"
            )
            field_level_df = pd.DataFrame()
        else:
            # Create DataFrame from field_data_frames
            field_data_frames = []
            for field_name, data in field_accuracies.items():
                accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0.0
                field_data_frames.append(
                    {
                        "model": model_name,
                        "field_name": field_name,
                        "accuracy": accuracy,
                        "correct_count": data["correct"],
                        "total_count": data["total"],
                    }
                )

            field_level_df = pd.DataFrame(field_data_frames)
            print(
                f"[green]✅ Field measurements recorded: {len(field_level_df)} field measurements[/green]"
            )
            print(
                f"[cyan]📊 Unique fields: {field_level_df['field_name'].nunique()}[/cyan]"
            )
            print(f"[cyan]📊 Models analyzed: {field_level_df['model'].nunique()}[/cyan]")

    except Exception as e:
        print(f"[red]❌ Error processing {latest_file}: {e}[/red]")
        return pd.DataFrame()

    return field_level_df
