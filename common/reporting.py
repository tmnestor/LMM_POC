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
    PILOT_READY_THRESHOLD,
)


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
    sorted_fields = sorted(summary_stats['field_accuracies'].items(), key=lambda x: x[1], reverse=True)
    
    # Calculate document quality distribution
    evaluation_data = summary_stats.get('evaluation_data', [])
    perfect_docs = sum(1 for doc in evaluation_data if doc['overall_accuracy'] >= 0.99)
    good_docs = sum(1 for doc in evaluation_data if 0.8 <= doc['overall_accuracy'] < 0.99)
    fair_docs = sum(1 for doc in evaluation_data if 0.6 <= doc['overall_accuracy'] < 0.8)
    poor_docs = sum(1 for doc in evaluation_data if doc['overall_accuracy'] < 0.6)
    
    executive_summary = f"""# {model_full_name} - Executive Summary

## Model Performance Overview
**Model:** {model_full_name}  
**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Documents Processed:** {summary_stats['total_images']}  
**Average Accuracy:** {summary_stats['overall_accuracy']:.1%}

## Key Findings

1. **Document Analysis:** Processed {summary_stats['total_images']} business documents with comprehensive field extraction
2. **Field Extraction:** Successfully extracts {len([f for f, acc in summary_stats['field_accuracies'].items() if acc >= EXCELLENT_FIELD_THRESHOLD])} out of {FIELD_COUNT} fields with ≥90% accuracy
3. **Best Performance:** {summary_stats['best_performing_image']} ({summary_stats['best_performance_accuracy']:.1%} accuracy)
4. **Challenging Cases:** {summary_stats['worst_performing_image']} ({summary_stats['worst_performance_accuracy']:.1%} accuracy)

## Field Performance Analysis

### Top Performing Fields (≥90% accuracy)
"""
    
    excellent_fields = [field for field, accuracy in sorted_fields if accuracy >= EXCELLENT_FIELD_THRESHOLD]
    if excellent_fields:
        for i, (field, accuracy) in enumerate([item for item in sorted_fields if item[1] >= EXCELLENT_FIELD_THRESHOLD][:10], 1):
            executive_summary += f"{i:2d}. {field:<25} {accuracy:.1%}\n"
    else:
        executive_summary += "No fields achieved ≥90% accuracy\n"
    
    executive_summary += """
### Challenging Fields (Requires Attention)
"""
    
    challenging_fields = [(field, accuracy) for field, accuracy in sorted_fields[-5:] if accuracy < EXCELLENT_FIELD_THRESHOLD]
    for i, (field, accuracy) in enumerate(challenging_fields, 1):
        executive_summary += f"{i}. {field:<25} {accuracy:.1%}\n"
    
    # Production readiness assessment
    if summary_stats['overall_accuracy'] >= DEPLOYMENT_READY_THRESHOLD:
        grade = "A+ (Excellent)"
        status = "✅ **READY FOR PRODUCTION:** Model demonstrates excellent accuracy and consistency"
    elif summary_stats['overall_accuracy'] >= PILOT_READY_THRESHOLD:
        grade = "A (Good)" 
        status = "✅ **READY FOR PILOT:** Model shows good performance with minor limitations"
    elif summary_stats['overall_accuracy'] >= 0.7:
        grade = "B (Fair)"
        status = "⚠️ **REQUIRES OPTIMIZATION:** Consider fine-tuning or prompt engineering"
    else:
        grade = "C (Needs Improvement)"
        status = "❌ **NOT READY FOR PRODUCTION:** Significant accuracy improvements needed"
    
    executive_summary += f"""
**Overall Grade:** {grade}

## Production Readiness Assessment

{status}

## Document Quality Distribution
- Perfect Documents (≥99%): {perfect_docs} ({perfect_docs/summary_stats['total_images']*100:.1f}%)
- Good Documents (80-98%): {good_docs} ({good_docs/summary_stats['total_images']*100:.1f}%)  
- Fair Documents (60-79%): {fair_docs} ({fair_docs/summary_stats['total_images']*100:.1f}%)
- Poor Documents (<60%): {poor_docs} ({poor_docs/summary_stats['total_images']*100:.1f}%)

## Recommendations

### Immediate Actions
{"1. ✅ DEPLOY TO PRODUCTION - Model ready for automated processing" if summary_stats['overall_accuracy'] >= DEPLOYMENT_READY_THRESHOLD else "1. ⚠️ PILOT DEPLOYMENT - Test with subset of documents" if summary_stats['overall_accuracy'] >= PILOT_READY_THRESHOLD else "1. 🔧 OPTIMIZATION REQUIRED - Improve model before deployment"}
2. 📋 Establish monitoring dashboards for accuracy tracking
3. 🎯 Focus improvement efforts on challenging fields: {', '.join([f[0] for f in challenging_fields[:3]])}

### Strategic Initiatives  
- 🔄 Implement continuous evaluation pipeline
- 📊 Expand ground truth dataset for challenging document types
- ⚡ Optimize inference pipeline for production scale

---
📊 {model_full_name} achieved {summary_stats['overall_accuracy']:.1%} average accuracy
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
    sorted_fields = sorted(summary_stats['field_accuracies'].items(), key=lambda x: x[1], reverse=True)
    excellent_fields = [field for field, accuracy in sorted_fields if accuracy >= EXCELLENT_FIELD_THRESHOLD]
    challenging_fields = [(field, accuracy) for field, accuracy in sorted_fields[-5:] if accuracy < EXCELLENT_FIELD_THRESHOLD]
    
    deployment_checklist = f"""# {model_full_name} Deployment Readiness Checklist

## Model Information
- **Model:** {model_full_name}
- **Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Overall Accuracy:** {summary_stats['overall_accuracy']:.1%}

## Production Readiness Checklist

### Performance Metrics
- [{'x' if summary_stats['overall_accuracy'] >= PILOT_READY_THRESHOLD else ' '}] Overall accuracy ≥80% ({summary_stats['overall_accuracy']:.1%})
- [{'x' if len(excellent_fields) >= max(15, FIELD_COUNT * 0.6) else ' '}] At least {max(15, int(FIELD_COUNT * 0.6))} fields with ≥90% accuracy ({len(excellent_fields)}/{FIELD_COUNT})
- [{'x' if summary_stats['perfect_documents'] >= summary_stats['total_images'] * 0.3 else ' '}] At least 30% perfect documents ({summary_stats['perfect_documents']}/{summary_stats['total_images']})

### Quality Assessment
- Best Case: {summary_stats['best_performance_accuracy']:.1%} accuracy
- Worst Case: {summary_stats['worst_performance_accuracy']:.1%} accuracy

### Field Performance
- Track accuracy for critical fields: {', '.join(excellent_fields[:5])}
- Monitor challenging fields: {', '.join([f[0] for f in challenging_fields[:3]])}

## Deployment Strategy

{"✅ **APPROVED FOR PRODUCTION DEPLOYMENT**" if summary_stats['overall_accuracy'] >= PILOT_READY_THRESHOLD else "⚠️ **PILOT DEPLOYMENT RECOMMENDED**" if summary_stats['overall_accuracy'] >= 0.7 else "🔧 **OPTIMIZATION REQUIRED BEFORE DEPLOYMENT**"}

### Next Steps
1. {'✅ Deploy to production environment' if summary_stats['overall_accuracy'] >= PILOT_READY_THRESHOLD else '🧪 Run pilot with subset of documents' if summary_stats['overall_accuracy'] >= 0.7 else '🔧 Optimize model performance'}
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
- Challenging fields: {', '.join([f[0] for f in challenging_fields[:3]])}
- Document types requiring attention: Review worst-performing documents

### Mitigation Strategies
- Manual review process for low-confidence extractions
- Continuous model improvement pipeline
- Fallback mechanisms for critical fields

---
*Generated by Vision Model Evaluation Pipeline*
"""
    
    return deployment_checklist


def generate_comprehensive_reports(evaluation_summary, output_dir_path, model_name, model_full_name):
    """
    Generate comprehensive evaluation reports including executive summary and deployment checklist.
    
    Args:
        evaluation_summary (dict): Evaluation results and metrics
        output_dir_path (Path): Output directory path
        model_name (str): Short model name (e.g., "llama", "internvl3")
        model_full_name (str): Full model name for display
        
    Returns:
        dict: Paths to generated reports
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_path = Path(output_dir_path)
    
    # Generate executive summary
    executive_summary = generate_executive_summary(evaluation_summary, model_name, model_full_name)
    
    # Save executive summary
    report_filename = f"{model_name}_comprehensive_evaluation_report_{timestamp}.md"
    report_path = output_dir_path / report_filename
    with report_path.open('w', encoding='utf-8') as f:
        f.write(executive_summary)
    
    # Generate deployment checklist
    deployment_checklist = generate_deployment_checklist(evaluation_summary, model_name, model_full_name)
    
    # Save deployment checklist  
    checklist_filename = f"{model_name}_deployment_checklist_{timestamp}.md"
    checklist_path = output_dir_path / checklist_filename
    with checklist_path.open('w', encoding='utf-8') as f:
        f.write(deployment_checklist)
    
    # Save JSON evaluation results
    json_filename = f"{model_name}_evaluation_results_{timestamp}.json"
    json_path = output_dir_path / json_filename
    
    # Prepare JSON-serializable data
    json_data = {
        'model_name': model_full_name,
        'evaluation_date': datetime.now().isoformat(),
        'total_images': evaluation_summary['total_images'],
        'overall_accuracy': evaluation_summary['overall_accuracy'],
        'best_performing_image': evaluation_summary['best_performing_image'],
        'best_performance_accuracy': evaluation_summary['best_performance_accuracy'],
        'worst_performing_image': evaluation_summary['worst_performing_image'],
        'worst_performance_accuracy': evaluation_summary['worst_performance_accuracy'],
        'perfect_documents': evaluation_summary['perfect_documents'],
        'field_accuracies': evaluation_summary['field_accuracies']
    }
    
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print("\n📋 EVALUATION REPORTS GENERATED")
    print("=" * 50)
    print(f"✅ Executive Summary: {report_path.name}")
    print(f"✅ Deployment Checklist: {checklist_path.name}")
    print(f"✅ JSON Results: {json_path.name}")
    
    return {
        'executive_summary': report_path,
        'deployment_checklist': checklist_path,
        'json_results': json_path
    }


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
    print(f"🎯 Best Performance: {evaluation_summary['best_performing_image']} ({evaluation_summary['best_performance_accuracy']:.1%})")
    print(f"⚠️ Worst Performance: {evaluation_summary['worst_performing_image']} ({evaluation_summary['worst_performance_accuracy']:.1%})")
    
    # Show top fields
    sorted_fields = sorted(evaluation_summary['field_accuracies'].items(), key=lambda x: x[1], reverse=True)
    print("\n📈 Top 5 Performing Fields:")
    for i, (field, accuracy) in enumerate(sorted_fields[:5], 1):
        print(f"   {i}. {field:<25} {accuracy:.1%}")
    
    # Production readiness
    if evaluation_summary['overall_accuracy'] >= DEPLOYMENT_READY_THRESHOLD:
        print("\n✅ MODEL IS READY FOR PRODUCTION DEPLOYMENT")
    elif evaluation_summary['overall_accuracy'] >= PILOT_READY_THRESHOLD:
        print("\n⚠️ MODEL IS READY FOR PILOT TESTING")
    else:
        print("\n❌ MODEL REQUIRES FURTHER OPTIMIZATION")