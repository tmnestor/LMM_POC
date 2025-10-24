#!/usr/bin/env python3
"""
Scikit-learn Multiclass Classification Evaluation

Evaluates model predictions against ground truth annotations using comprehensive metrics.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# Global plot configuration
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 10

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def evaluate_multiclass(df, pred_col="pred", truth_col="annotator", verbose=True):
    """
    Comprehensive multiclass evaluation of predictions vs ground truth.

    Args:
        df: DataFrame with prediction and ground truth columns
        pred_col: Name of prediction column
        truth_col: Name of ground truth column
        verbose: Print detailed results

    Returns:
        Dictionary containing all evaluation metrics
    """

    # Ensure columns exist
    if pred_col not in df.columns or truth_col not in df.columns:
        raise ValueError(
            f"Required columns {pred_col} and {truth_col} not found in DataFrame"
        )

    # Get predictions and ground truth
    y_true = df[truth_col].to_numpy()
    y_pred = df[pred_col].to_numpy()

    # Handle missing values
    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if verbose:
        print("=" * 70)
        print("MULTICLASS CLASSIFICATION EVALUATION")
        print("=" * 70)
        print(f"\nSamples evaluated: {len(y_true):,}")
        print(f"Unique classes in ground truth: {len(np.unique(y_true))}")
        print(f"Unique classes in predictions: {len(np.unique(y_pred))}")

    # Calculate metrics
    metrics = {}

    # Basic accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro, micro, weighted averages
    metrics["precision_macro"] = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[0]
    metrics["recall_macro"] = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[1]
    metrics["f1_macro"] = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]

    metrics["precision_micro"] = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )[0]
    metrics["recall_micro"] = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )[1]
    metrics["f1_micro"] = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )[2]

    metrics["precision_weighted"] = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[0]
    metrics["recall_weighted"] = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[1]
    metrics["f1_weighted"] = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]

    # Agreement metrics
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0
    )

    # Per-class metrics DataFrame
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_metrics = []

    for i, cls in enumerate(unique_classes):
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            cls_pred = y_pred[cls_mask]
            class_metrics.append(
                {
                    "class": cls,
                    "support": cls_mask.sum(),
                    "accuracy": (cls_pred == cls).mean(),
                    "precision": precision[i] if i < len(precision) else 0,
                    "recall": recall[i] if i < len(recall) else 0,
                    "f1": f1[i] if i < len(f1) else 0,
                }
            )

    metrics["per_class_metrics"] = pd.DataFrame(class_metrics)

    if verbose:
        print("\n" + "=" * 70)
        print("OVERALL METRICS")
        print("=" * 70)
        print(f"Accuracy:                {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}")
        print(f"Cohen's Kappa:           {metrics['cohen_kappa']:.4f}")
        print(f"Matthews Corr Coef:      {metrics['matthews_corrcoef']:.4f}")

        print("\n" + "=" * 70)
        print("AVERAGED METRICS")
        print("=" * 70)
        print("\nMacro Averages (unweighted mean):")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

        print("\nMicro Averages (globally counted):")
        print(f"  Precision: {metrics['precision_micro']:.4f}")
        print(f"  Recall:    {metrics['recall_micro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_micro']:.4f}")

        print("\nWeighted Averages (by support):")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")

        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(metrics["classification_report"])

        # Show top/bottom performing classes
        if not metrics["per_class_metrics"].empty:
            print("\n" + "=" * 70)
            print("TOP 5 BEST PERFORMING CLASSES")
            print("=" * 70)
            top_classes = metrics["per_class_metrics"].nlargest(5, "f1")
            print(
                top_classes[
                    ["class", "f1", "precision", "recall", "support"]
                ].to_string(index=False)
            )

            print("\n" + "=" * 70)
            print("TOP 5 WORST PERFORMING CLASSES")
            print("=" * 70)
            bottom_classes = metrics["per_class_metrics"].nsmallest(5, "f1")
            print(
                bottom_classes[
                    ["class", "f1", "precision", "recall", "support"]
                ].to_string(index=False)
            )

    return metrics


def plot_confusion_matrix(metrics, save_path=None, figsize=(12, 10), top_n=20):
    """
    Plot confusion matrix heatmap.

    Args:
        metrics: Dictionary containing confusion_matrix
        save_path: Path to save figure
        figsize: Figure size
        top_n: Show only top N most frequent classes
    """
    cm = metrics["confusion_matrix"]

    # If too many classes, show only top N
    if cm.shape[0] > top_n:
        # Get indices of top N classes by support
        support = cm.sum(axis=1)
        top_indices = np.argsort(support)[-top_n:]
        cm = cm[top_indices][:, top_indices]
        title_suffix = f" (Top {top_n} Classes)"
    else:
        title_suffix = ""

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
    plt.title(f"Confusion Matrix{title_suffix}", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\nConfusion matrix saved to: {save_path}")

    plt.show()


def plot_class_performance(metrics, save_path=None, figsize=(14, 8), top_n=30):
    """
    Plot per-class performance metrics.

    Args:
        metrics: Dictionary containing per_class_metrics DataFrame
        save_path: Path to save figure
        figsize: Figure size
        top_n: Show only top N classes
    """
    class_metrics_df = metrics["per_class_metrics"].copy()

    # Sort by F1 score and take top N
    if len(class_metrics_df) > top_n:
        class_metrics_df = class_metrics_df.nlargest(top_n, "support")
        title_suffix = f" (Top {top_n} Classes by Support)"
    else:
        title_suffix = ""

    class_metrics_df = class_metrics_df.sort_values("f1", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Precision
    sns.barplot(data=class_metrics_df, y="class", x="precision", ax=axes[0], orient="h")
    axes[0].set_xlabel("Precision")
    axes[0].set_title("Precision by Class", fontweight="bold")
    axes[0].set_ylabel("")

    # Recall
    sns.barplot(data=class_metrics_df, y="class", x="recall", ax=axes[1], orient="h")
    axes[1].set_xlabel("Recall")
    axes[1].set_title("Recall by Class", fontweight="bold")
    axes[1].set_ylabel("")

    # F1-Score
    sns.barplot(data=class_metrics_df, y="class", x="f1", ax=axes[2], orient="h")
    axes[2].set_xlabel("F1-Score")
    axes[2].set_title("F1-Score by Class", fontweight="bold")
    axes[2].set_ylabel("")

    plt.suptitle(f"Per-Class Performance Metrics{title_suffix}", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Class performance plot saved to: {save_path}")

    plt.show()


def analyze_errors(df, pred_col="pred", truth_col="annotator", top_n=10):
    """
    Analyze common error patterns.

    Args:
        df: DataFrame with predictions and ground truth
        pred_col: Name of prediction column
        truth_col: Name of ground truth column
        top_n: Number of top error patterns to show

    Returns:
        DataFrame with error analysis
    """
    # Find misclassifications
    errors = df[df[pred_col] != df[truth_col]].copy()

    if len(errors) == 0:
        print("No errors found!")
        return pd.DataFrame()

    # Create error pattern column
    errors["error_pattern"] = (
        errors[truth_col].astype(str) + " → " + errors[pred_col].astype(str)
    )

    # Count error patterns
    error_counts = errors["error_pattern"].value_counts().head(top_n)

    print("\n" + "=" * 70)
    print(f"TOP {top_n} ERROR PATTERNS")
    print("=" * 70)
    print(f"\nTotal errors: {len(errors):,} ({len(errors) / len(df) * 100:.2f}%)")
    print("\nMost common misclassifications:")
    print("-" * 50)

    for pattern, count in error_counts.items():
        percentage = count / len(errors) * 100
        print(f"{pattern:<40} : {count:>5} ({percentage:>5.1f}%)")

    # Create error analysis DataFrame
    error_analysis = pd.DataFrame(
        {
            "error_pattern": error_counts.index,
            "count": error_counts.to_numpy(),
            "percentage": error_counts.to_numpy() / len(errors) * 100,
        }
    )

    return error_analysis


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    classes = ["class_A", "class_B", "class_C", "class_D", "class_E"]

    # Generate ground truth
    annotator = np.random.choice(classes, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])

    # Generate predictions with some errors
    pred = annotator.copy()
    error_mask = np.random.random(n_samples) < 0.2  # 20% error rate
    pred[error_mask] = np.random.choice(classes, error_mask.sum())

    # Create DataFrame
    sample_df = pd.DataFrame({"pred": pred, "annotator": annotator})

    print("Sample DataFrame:")
    print(sample_df.head(10))
    print(f"\nDataFrame shape: {sample_df.shape}")

    # Run evaluation
    metrics = evaluate_multiclass(sample_df, pred_col="pred", truth_col="annotator")

    # Plot confusion matrix
    plot_confusion_matrix(metrics)

    # Plot class performance
    plot_class_performance(metrics)

    # Analyze errors
    error_analysis = analyze_errors(sample_df, pred_col="pred", truth_col="annotator")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
