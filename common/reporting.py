"""
Comprehensive reporting utilities for vision model evaluation.

This module provides functions for generating executive summaries,
deployment checklists, and various evaluation reports.
"""

import json
from datetime import datetime
from pathlib import Path

from .config import (
    DEPLOYMENT_READY_THRESHOLD,
    EXCELLENT_FIELD_THRESHOLD,
    FIELD_COUNT,
    FIELD_GROUPS,
    PILOT_READY_THRESHOLD,
    VISUALIZATION_ENABLED,
)
from .evaluation_utils import generate_overall_classification_summary
from .visualizations import LMMVisualizer


def generate_executive_summary(evaluation_summary, model_name, model_full_name):
    """
    Generate executive summary report for model evaluation.

    Args:
        evaluation_summary (dict): Evaluation results and metrics
        model_name (str): Short model name (e.g., "llama", "internvl3")
        model_full_name (str): Full model name for display

    Returns:
        str: Formatted executive summary markdown content
    """
    summary_stats = evaluation_summary
    sorted_fields = sorted(
        summary_stats["field_accuracies"].items(), key=lambda x: x[1], reverse=True
    )

    # Calculate document quality distribution
    evaluation_data = summary_stats.get("evaluation_data", [])
    perfect_docs = sum(1 for doc in evaluation_data if doc["overall_accuracy"] >= 0.99)
    good_docs = sum(
        1 for doc in evaluation_data if 0.8 <= doc["overall_accuracy"] < 0.99
    )
    fair_docs = sum(
        1 for doc in evaluation_data if 0.6 <= doc["overall_accuracy"] < 0.8
    )
    poor_docs = sum(1 for doc in evaluation_data if doc["overall_accuracy"] < 0.6)

    executive_summary = f"""# {model_full_name} - Executive Summary

## Model Performance Overview
**Model:** {model_full_name}  
**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Documents Processed:** {summary_stats["total_images"]}  
**Average Accuracy:** {summary_stats["overall_accuracy"]:.1%}

## Key Findings

1. **Document Analysis:** Processed {summary_stats["total_images"]} business documents with comprehensive field extraction
2. **Field Extraction:** Successfully extracts {len([f for f, acc in summary_stats["field_accuracies"].items() if acc >= EXCELLENT_FIELD_THRESHOLD])} out of {FIELD_COUNT} fields with ≥90% accuracy
3. **Best Performance:** {summary_stats["best_performing_image"]} ({summary_stats["best_performance_accuracy"]:.1%} accuracy)
4. **Challenging Cases:** {summary_stats["worst_performing_image"]} ({summary_stats["worst_performance_accuracy"]:.1%} accuracy)

## Field Performance Analysis

### Top Performing Fields (≥90% accuracy)
"""

    excellent_fields = [
        field
        for field, accuracy in sorted_fields
        if accuracy >= EXCELLENT_FIELD_THRESHOLD
    ]
    if excellent_fields:
        for i, (field, accuracy) in enumerate(
            [item for item in sorted_fields if item[1] >= EXCELLENT_FIELD_THRESHOLD][
                :10
            ],
            1,
        ):
            executive_summary += f"{i:2d}. {field:<25} {accuracy:.1%}\n"
    else:
        executive_summary += "No fields achieved ≥90% accuracy\n"

    executive_summary += """
### Challenging Fields (Requires Attention)
"""

    challenging_fields = [
        (field, accuracy)
        for field, accuracy in sorted_fields[-5:]
        if accuracy < EXCELLENT_FIELD_THRESHOLD
    ]
    for i, (field, accuracy) in enumerate(challenging_fields, 1):
        executive_summary += f"{i}. {field:<25} {accuracy:.1%}\n"

    # Production readiness assessment
    if summary_stats["overall_accuracy"] >= DEPLOYMENT_READY_THRESHOLD:
        grade = "A+ (Excellent)"
        status = "✅ **READY FOR PRODUCTION:** Model demonstrates excellent accuracy and consistency"
    elif summary_stats["overall_accuracy"] >= PILOT_READY_THRESHOLD:
        grade = "A (Good)"
        status = "✅ **READY FOR PILOT:** Model shows good performance with minor limitations"
    elif summary_stats["overall_accuracy"] >= 0.7:
        grade = "B (Fair)"
        status = (
            "⚠️ **REQUIRES OPTIMIZATION:** Consider fine-tuning or prompt engineering"
        )
    else:
        grade = "C (Needs Improvement)"
        status = (
            "❌ **NOT READY FOR PRODUCTION:** Significant accuracy improvements needed"
        )

    executive_summary += f"""
**Overall Grade:** {grade}

## Production Readiness Assessment

{status}

## Document Quality Distribution
- Perfect Documents (≥99%): {perfect_docs} ({perfect_docs / summary_stats["total_images"] * 100:.1f}%)
- Good Documents (80-98%): {good_docs} ({good_docs / summary_stats["total_images"] * 100:.1f}%)  
- Fair Documents (60-79%): {fair_docs} ({fair_docs / summary_stats["total_images"] * 100:.1f}%)
- Poor Documents (<60%): {poor_docs} ({poor_docs / summary_stats["total_images"] * 100:.1f}%)

## Recommendations

### Immediate Actions
{"1. ✅ DEPLOY TO PRODUCTION - Model ready for automated processing" if summary_stats["overall_accuracy"] >= DEPLOYMENT_READY_THRESHOLD else "1. ⚠️ PILOT DEPLOYMENT - Test with subset of documents" if summary_stats["overall_accuracy"] >= PILOT_READY_THRESHOLD else "1. 🔧 OPTIMIZATION REQUIRED - Improve model before deployment"}
2. 📋 Establish monitoring dashboards for accuracy tracking
3. 🎯 Focus improvement efforts on challenging fields: {", ".join([f[0] for f in challenging_fields[:3]])}

### Strategic Initiatives  
- 🔄 Implement continuous evaluation pipeline
- 📊 Expand ground truth dataset for challenging document types
- ⚡ Optimize inference pipeline for production scale

---
📊 {model_full_name} achieved {summary_stats["overall_accuracy"]:.1%} average accuracy
"""

    return executive_summary


def generate_deployment_checklist(evaluation_summary, model_name, model_full_name):
    """
    Generate deployment readiness checklist for model evaluation.

    Args:
        evaluation_summary (dict): Evaluation results and metrics
        model_name (str): Short model name (e.g., "llama", "internvl3")
        model_full_name (str): Full model name for display

    Returns:
        str: Formatted deployment checklist markdown content
    """
    summary_stats = evaluation_summary
    sorted_fields = sorted(
        summary_stats["field_accuracies"].items(), key=lambda x: x[1], reverse=True
    )
    excellent_fields = [
        field
        for field, accuracy in sorted_fields
        if accuracy >= EXCELLENT_FIELD_THRESHOLD
    ]
    challenging_fields = [
        (field, accuracy)
        for field, accuracy in sorted_fields[-5:]
        if accuracy < EXCELLENT_FIELD_THRESHOLD
    ]

    deployment_checklist = f"""# {model_full_name} Deployment Readiness Checklist

## Model Information
- **Model:** {model_full_name}
- **Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Overall Accuracy:** {summary_stats["overall_accuracy"]:.1%}

## Production Readiness Checklist

### Performance Metrics
- [{"x" if summary_stats["overall_accuracy"] >= PILOT_READY_THRESHOLD else " "}] Overall accuracy ≥80% ({summary_stats["overall_accuracy"]:.1%})
- [{"x" if len(excellent_fields) >= max(15, FIELD_COUNT * 0.6) else " "}] At least {max(15, int(FIELD_COUNT * 0.6))} fields with ≥90% accuracy ({len(excellent_fields)}/{FIELD_COUNT})
- [{"x" if summary_stats["perfect_documents"] >= summary_stats["total_images"] * 0.3 else " "}] At least 30% perfect documents ({summary_stats["perfect_documents"]}/{summary_stats["total_images"]})

### Quality Assessment
- Best Case: {summary_stats["best_performance_accuracy"]:.1%} accuracy
- Worst Case: {summary_stats["worst_performance_accuracy"]:.1%} accuracy

### Field Performance
- Track accuracy for critical fields: {", ".join(excellent_fields[:5])}
- Monitor challenging fields: {", ".join([f[0] for f in challenging_fields[:3]])}

## Deployment Strategy

{"✅ **APPROVED FOR PRODUCTION DEPLOYMENT**" if summary_stats["overall_accuracy"] >= PILOT_READY_THRESHOLD else "⚠️ **PILOT DEPLOYMENT RECOMMENDED**" if summary_stats["overall_accuracy"] >= 0.7 else "🔧 **OPTIMIZATION REQUIRED BEFORE DEPLOYMENT**"}

### Next Steps
1. {"✅ Deploy to production environment" if summary_stats["overall_accuracy"] >= PILOT_READY_THRESHOLD else "🧪 Run pilot with subset of documents" if summary_stats["overall_accuracy"] >= 0.7 else "🔧 Optimize model performance"}
2. 📊 Implement real-time accuracy monitoring
3. 🔄 Establish continuous evaluation pipeline
4. 📋 Create operational runbooks and troubleshooting guides

## Operational Requirements

### Infrastructure
- GPU memory requirements (estimated based on model size)
- Batch processing capabilities for production scale
- Monitoring and alerting systems

### Data Management
- Ground truth data maintenance process
- Regular evaluation against new document types

## Risk Assessment

### Known Limitations
- Challenging fields: {", ".join([f[0] for f in challenging_fields[:3]])}
- Document types requiring attention: Review worst-performing documents

### Mitigation Strategies
- Manual review process for low-confidence extractions
- Continuous model improvement pipeline
- Fallback mechanisms for critical fields

---
*Generated by Vision Model Evaluation Pipeline*
"""

    return deployment_checklist


def generate_classification_report(
    extraction_results, ground_truth_data, model_name, model_full_name
):
    """
    Generate comprehensive sklearn classification report for field extraction.

    Args:
        extraction_results: List of extraction result dictionaries
        ground_truth_data: Ground truth mapping
        model_name: Short model name
        model_full_name: Full model name for display

    Returns:
        str: Formatted classification report in markdown
    """
    print("📊 Generating sklearn classification report...")

    try:
        classification_summary = generate_overall_classification_summary(
            extraction_results, ground_truth_data
        )
    except Exception as e:
        return f"❌ Error generating classification report: {e}"

    # Generate markdown report
    report = f"""# {model_full_name} - Classification Report

## Overview
**Model:** {model_full_name}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Analysis Type:** Binary Classification (Field Extracted vs Not Extracted)

## Overall Performance Metrics

"""

    overall_metrics = classification_summary.get("overall_metrics", {})
    if overall_metrics and "error" not in overall_metrics:
        report += """### Macro Averages (Unweighted Mean)
"""
        macro = overall_metrics.get("macro_avg", {})
        if macro:
            report += f"""- **Precision:** {macro.get("precision", 0):.3f}
- **Recall:** {macro.get("recall", 0):.3f}  
- **F1-Score:** {macro.get("f1_score", 0):.3f}

"""

        report += """### Micro Averages (Global)
"""
        micro = overall_metrics.get("micro_avg", {})
        if micro:
            report += f"""- **Precision:** {micro.get("precision", 0):.3f}
- **Recall:** {micro.get("recall", 0):.3f}
- **F1-Score:** {micro.get("f1_score", 0):.3f}

"""

        report += """### Weighted Averages (By Support)
"""
        weighted = overall_metrics.get("weighted_avg", {})
        if weighted:
            report += f"""- **Precision:** {weighted.get("precision", 0):.3f}
- **Recall:** {weighted.get("recall", 0):.3f}
- **F1-Score:** {weighted.get("f1_score", 0):.3f}

"""

        total_preds = overall_metrics.get("total_predictions", 0)
        report += f"**Total Predictions:** {total_preds}\n\n"
    else:
        report += f"❌ Error in overall metrics: {overall_metrics.get('error', 'Unknown error')}\n\n"

    # Field-level metrics
    field_metrics = classification_summary.get("field_metrics", {})
    if field_metrics:
        report += """## Field-Level Performance Summary

| Field | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
        # Sort by F1-score descending
        sorted_fields = sorted(
            field_metrics.items(), key=lambda x: x[1].get("f1_score", 0), reverse=True
        )

        for field, metrics in sorted_fields:
            if "error" not in metrics:
                # Handle support more robustly - convert None to 0
                support_val = metrics.get("support", 0)
                support_int = int(support_val) if support_val is not None else 0
                report += f"| {field} | {metrics.get('precision', 0):.3f} | {metrics.get('recall', 0):.3f} | {metrics.get('f1_score', 0):.3f} | {support_int} |\n"
            else:
                report += f"| {field} | Error | Error | Error | 0 |\n"

    # Top and bottom performers
    if field_metrics:
        top_f1_fields = sorted(
            [
                (f, m.get("f1_score", 0))
                for f, m in field_metrics.items()
                if "error" not in m
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        bottom_f1_fields = sorted(
            [
                (f, m.get("f1_score", 0))
                for f, m in field_metrics.items()
                if "error" not in m
            ],
            key=lambda x: x[1],
        )[:5]

        if top_f1_fields:
            report += """
## Best Performing Fields (by F1-Score)

"""
            for i, (field, f1) in enumerate(top_f1_fields, 1):
                report += f"{i}. **{field}**: {f1:.3f}\n"

        if bottom_f1_fields:
            report += """
## Lowest Performing Fields (by F1-Score)  

"""
            for i, (field, f1) in enumerate(bottom_f1_fields, 1):
                report += f"{i}. **{field}**: {f1:.3f}\n"

    # Detailed per-field reports
    classification_reports = classification_summary.get("classification_reports", {})
    if classification_reports:
        report += """
## Detailed Field Classification Reports

"""
        for field, field_report in classification_reports.items():
            if (
                "Error" not in field_report
                and "Insufficient" not in field_report
                and "No data" not in field_report
            ):
                report += f"""### {field}

```
{field_report}
```

"""

    report += """## Interpretation Guide

### Metrics Explained:
- **Precision**: Of all fields predicted as "Extracted", how many were correctly identified?
- **Recall**: Of all fields that should be "Extracted", how many were correctly identified?  
- **F1-Score**: Harmonic mean of Precision and Recall (balanced measure)
- **Support**: Number of actual instances in each class

### Averaging Methods:
- **Macro Avg**: Unweighted mean (treats all fields equally)
- **Micro Avg**: Globally computed metrics (accounts for class imbalance)
- **Weighted Avg**: Mean weighted by support (accounts for field frequency)

### Classification Task:
For each field, we classify whether the model should extract a value:
- **Class 0 (Not Extracted)**: Field should be N/A or empty
- **Class 1 (Extracted)**: Field should contain a value

---
*Generated by Vision Model Evaluation Pipeline*
"""

    return report


def generate_comprehensive_reports(
    evaluation_summary,
    output_dir_path,
    model_name,
    model_full_name,
    batch_statistics=None,
    extraction_results=None,
    ground_truth_data=None,
):
    """
    Generate comprehensive evaluation reports including executive summary, JSON results, visualizations, and classification reports.

    Args:
        evaluation_summary (dict): Evaluation results and metrics
        output_dir_path (Path): Output directory path
        model_name (str): Short model name (e.g., "llama", "internvl3")
        model_full_name (str): Full model name for display
        batch_statistics (dict, optional): Processing statistics for visualizations
        extraction_results (list, optional): Raw extraction results for classification analysis
        ground_truth_data (dict, optional): Ground truth mapping for classification analysis

    Returns:
        dict: Paths to generated reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = Path(output_dir_path)

    # Generate executive summary
    executive_summary = generate_executive_summary(
        evaluation_summary, model_name, model_full_name
    )

    # Save executive summary
    report_filename = f"{model_name}_comprehensive_evaluation_report_{timestamp}.md"
    report_path = output_dir_path / report_filename
    with report_path.open("w", encoding="utf-8") as f:
        f.write(executive_summary)

    # Deployment checklist generation removed per user request

    # Save JSON evaluation results
    json_filename = f"{model_name}_evaluation_results_{timestamp}.json"
    json_path = output_dir_path / json_filename

    # Prepare JSON-serializable data
    json_data = {
        "model_name": model_full_name,
        "evaluation_date": datetime.now().isoformat(),
        "total_images": evaluation_summary["total_images"],
        "overall_accuracy": evaluation_summary["overall_accuracy"],
        "best_performing_image": evaluation_summary["best_performing_image"],
        "best_performance_accuracy": evaluation_summary["best_performance_accuracy"],
        "worst_performing_image": evaluation_summary["worst_performing_image"],
        "worst_performance_accuracy": evaluation_summary["worst_performance_accuracy"],
        "perfect_documents": evaluation_summary["perfect_documents"],
        "field_accuracies": evaluation_summary["field_accuracies"],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Generate classification report if data available
    classification_report_path = None
    if extraction_results is not None and ground_truth_data is not None:
        print("📊 Generating sklearn classification report...")
        try:
            classification_report_content = generate_classification_report(
                extraction_results, ground_truth_data, model_name, model_full_name
            )

            if (
                classification_report_content
                and "Error" not in classification_report_content
            ):
                classification_filename = (
                    f"{model_name}_classification_report_{timestamp}.md"
                )
                classification_report_path = output_dir_path / classification_filename

                with classification_report_path.open("w", encoding="utf-8") as f:
                    f.write(classification_report_content)

                print(
                    f"✅ Classification report saved: {classification_report_path.name}"
                )
            else:
                print(
                    "⚠️ Classification report generation had issues - check data quality"
                )

        except Exception as e:
            print(f"❌ Classification report generation failed: {e}")
    elif extraction_results is None or ground_truth_data is None:
        print(
            "📊 Skipping classification report - extraction results or ground truth not provided"
        )

    # Generate visualizations if enabled and batch_statistics available
    visualization_paths = []
    html_report_path = None

    if VISUALIZATION_ENABLED and batch_statistics is not None:
        print("\n🎨 Generating visualizations...")

        try:
            # Initialize visualizer
            visualizer = LMMVisualizer(output_dir=str(output_dir_path))

            # Generate complete visualization suite
            visualization_paths = visualizer.generate_model_visualizations(
                evaluation_summary,
                batch_statistics,
                model_name,
                extraction_results,
                ground_truth_data,
            )

            # Generate HTML summary with embedded visualizations
            if visualization_paths:
                html_report_path = visualizer.create_html_summary(
                    evaluation_summary, model_full_name, visualization_paths
                )

            print(f"✅ Generated {len(visualization_paths)} visualizations")

        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
            print("📊 Continuing with text-based reports...")
    elif VISUALIZATION_ENABLED:
        print("📊 Visualizations enabled but batch statistics not provided")
    else:
        print("📊 Visualizations disabled in configuration")

    print("\n📋 EVALUATION REPORTS GENERATED")
    print("=" * 50)
    print(f"✅ Executive Summary: {report_path.name}")
    print(f"✅ JSON Results: {json_path.name}")

    if classification_report_path:
        print(f"✅ Classification Report: {classification_report_path.name}")

    if visualization_paths:
        print("✅ Visualizations:")
        for viz_path in visualization_paths:
            print(f"   • {Path(viz_path).name}")

    if html_report_path:
        print(f"✅ HTML Summary: {Path(html_report_path).name}")

    # Prepare return dictionary
    result = {"executive_summary": report_path, "json_results": json_path}

    if classification_report_path:
        result["classification_report"] = classification_report_path

    if visualization_paths:
        result["visualizations"] = [Path(p) for p in visualization_paths]

    if html_report_path:
        result["html_summary"] = Path(html_report_path)

    return result


def print_evaluation_summary(evaluation_summary, model_full_name):
    """
    Print evaluation summary to console.

    Args:
        evaluation_summary (dict): Evaluation results
        model_full_name (str): Full model name for display
    """
    print("\n" + "=" * 80)
    print(f"🎉 {model_full_name.upper()} EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Overall Accuracy: {evaluation_summary['overall_accuracy']:.1%}")
    print(f"📷 Documents Processed: {evaluation_summary['total_images']}")
    print(f"⭐ Perfect Documents: {evaluation_summary['perfect_documents']}")
    print(
        f"🎯 Best Performance: {evaluation_summary['best_performing_image']} ({evaluation_summary['best_performance_accuracy']:.1%})"
    )
    print(
        f"⚠️ Worst Performance: {evaluation_summary['worst_performing_image']} ({evaluation_summary['worst_performance_accuracy']:.1%})"
    )

    # Show top fields
    sorted_fields = sorted(
        evaluation_summary["field_accuracies"].items(), key=lambda x: x[1], reverse=True
    )
    print("\n📈 Top 5 Performing Fields:")
    for i, (field, accuracy) in enumerate(sorted_fields[:5], 1):
        print(f"   {i}. {field:<25} {accuracy:.1%}")

    # Production readiness
    if evaluation_summary["overall_accuracy"] >= DEPLOYMENT_READY_THRESHOLD:
        print("\n✅ MODEL IS READY FOR PRODUCTION DEPLOYMENT")
    elif evaluation_summary["overall_accuracy"] >= PILOT_READY_THRESHOLD:
        print("\n⚠️ MODEL IS READY FOR PILOT TESTING")
    else:
        print("\n❌ MODEL REQUIRES FURTHER OPTIMIZATION")


def analyze_group_performance(evaluation_summary, extraction_results=None):
    """
    Analyze performance by field groups for grouped extraction.

    Args:
        evaluation_summary (dict): Evaluation results with field accuracies
        extraction_results (list): Individual extraction results (optional)

    Returns:
        dict: Group performance analysis
    """
    if not evaluation_summary or "field_accuracies" not in evaluation_summary:
        return {}

    field_accuracies = evaluation_summary["field_accuracies"]

    # Calculate group-level accuracies
    group_accuracies = {}
    group_field_counts = {}

    for group_name, group_config in FIELD_GROUPS.items():
        group_fields = group_config["fields"]
        group_scores = []

        for field in group_fields:
            if field in field_accuracies:
                group_scores.append(field_accuracies[field])

        if group_scores:
            group_accuracies[group_name] = {
                "accuracy": sum(group_scores) / len(group_scores),
                "field_count": len(group_scores),
                "total_fields": len(group_fields),
                "priority": group_config["priority"],
                "description": group_config["description"],
                "best_field": max(
                    group_fields, key=lambda f: field_accuracies.get(f, 0)
                ),
                "worst_field": min(
                    group_fields, key=lambda f: field_accuracies.get(f, 0)
                ),
                "coverage": len(group_scores) / len(group_fields)
                if group_fields
                else 0,
            }

    # Sort groups by priority for reporting
    sorted_groups = sorted(group_accuracies.items(), key=lambda x: x[1]["priority"])

    # Analyze group metadata if available from grouped extraction
    group_timing = {}
    if extraction_results:
        for result in extraction_results:
            if "group_metadata" in result and result["group_metadata"]:
                metadata = result["group_metadata"]
                for group_name, group_data in metadata.items():
                    if group_name not in group_timing:
                        group_timing[group_name] = []
                    group_timing[group_name].append(
                        group_data.get("processing_time", 0)
                    )

    # Calculate average timing per group
    avg_group_timing = {}
    for group_name, times in group_timing.items():
        if times:
            avg_group_timing[group_name] = sum(times) / len(times)

    return {
        "group_accuracies": dict(sorted_groups),
        "group_timing": avg_group_timing,
        "total_groups": len(group_accuracies),
        "excellent_groups": len(
            [
                g
                for g in group_accuracies.values()
                if g["accuracy"] >= EXCELLENT_FIELD_THRESHOLD
            ]
        ),
        "good_groups": len(
            [
                g
                for g in group_accuracies.values()
                if 0.8 <= g["accuracy"] < EXCELLENT_FIELD_THRESHOLD
            ]
        ),
        "poor_groups": len(
            [g for g in group_accuracies.values() if g["accuracy"] < 0.5]
        ),
        "average_group_accuracy": sum(g["accuracy"] for g in group_accuracies.values())
        / len(group_accuracies)
        if group_accuracies
        else 0,
    }


def generate_group_performance_report(group_analysis, model_name):
    """
    Generate a detailed report of group performance.

    Args:
        group_analysis (dict): Group performance analysis
        model_name (str): Model name for report title

    Returns:
        str: Formatted group performance report
    """
    if not group_analysis or not group_analysis.get("group_accuracies"):
        return "Group analysis not available (single-pass extraction used)."

    report = f"""# {model_name.upper()} - Group Performance Analysis

## Group Extraction Summary
- **Total Groups**: {group_analysis["total_groups"]}
- **Excellent Groups** (≥90%): {group_analysis["excellent_groups"]}
- **Good Groups** (80-89%): {group_analysis["good_groups"]}
- **Poor Groups** (<50%): {group_analysis["poor_groups"]}
- **Average Group Accuracy**: {group_analysis["average_group_accuracy"]:.1%}

## Group-by-Group Performance

"""

    for group_name, stats in group_analysis["group_accuracies"].items():
        status = (
            "🟢"
            if stats["accuracy"] >= EXCELLENT_FIELD_THRESHOLD
            else "🟡"
            if stats["accuracy"] >= 0.8
            else "🔴"
        )

        report += f"""### {status} {FIELD_GROUPS[group_name]["name"]} ({stats["accuracy"]:.1%})
- **Priority**: {stats["priority"]} | **Coverage**: {stats["coverage"]:.1%}
- **Description**: {stats["description"]}
- **Fields**: {stats["field_count"]}/{stats["total_fields"]} processed
- **Best Field**: {stats["best_field"]} | **Challenging**: {stats["worst_field"]}
"""

        # Add timing info if available
        if group_name in group_analysis.get("group_timing", {}):
            timing = group_analysis["group_timing"][group_name]
            report += f"- **Average Processing Time**: {timing:.2f}s\n"

        report += "\n"

    report += """## Optimization Recommendations

"""

    # Add recommendations based on group performance
    poor_groups = [
        name
        for name, stats in group_analysis["group_accuracies"].items()
        if stats["accuracy"] < 0.5
    ]
    if poor_groups:
        report += (
            f"1. **Priority Focus**: Improve extraction for {', '.join(poor_groups)}\n"
        )

    critical_group = group_analysis["group_accuracies"].get("critical", {})
    if critical_group and critical_group.get("accuracy", 1.0) < 0.9:
        report += "2. **Critical Fields**: Focus on ABN and TOTAL accuracy for business validation\n"

    report += "3. **Group Optimization**: Consider adjusting prompt templates for underperforming groups\n"
    report += "4. **Processing Efficiency**: Monitor group timing for bottleneck identification\n"

    return report


def print_group_performance_summary(group_analysis, model_name):
    """
    Print a concise group performance summary to console.

    Args:
        group_analysis (dict): Group performance analysis
        model_name (str): Model name for display
    """
    if not group_analysis or not group_analysis.get("group_accuracies"):
        print("🔄 Single-pass extraction used (no group analysis)")
        return

    print(f"\n📊 {model_name.upper()} GROUP PERFORMANCE ANALYSIS")
    print("=" * 60)

    print(f"📈 Groups Processed: {group_analysis['total_groups']}")
    print(
        f"⭐ Excellent Groups: {group_analysis['excellent_groups']} | Good: {group_analysis['good_groups']} | Poor: {group_analysis['poor_groups']}"
    )
    print(f"🎯 Average Group Accuracy: {group_analysis['average_group_accuracy']:.1%}")

    print("\n🏆 Top 3 Performing Groups:")
    sorted_groups = sorted(
        group_analysis["group_accuracies"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    for i, (group_name, stats) in enumerate(sorted_groups[:3]):
        status = "🟢" if stats["accuracy"] >= EXCELLENT_FIELD_THRESHOLD else "🟡"
        print(
            f"   {i + 1}. {status} {FIELD_GROUPS[group_name]['name']}: {stats['accuracy']:.1%}"
        )

    # Show timing summary if available
    if group_analysis.get("group_timing"):
        total_time = sum(group_analysis["group_timing"].values())
        print(f"\n⏱️ Total Group Processing Time: {total_time:.2f}s")
        slowest_group = max(group_analysis["group_timing"].items(), key=lambda x: x[1])
        print(
            f"🐌 Slowest Group: {FIELD_GROUPS[slowest_group[0]]['name']} ({slowest_group[1]:.2f}s)"
        )
