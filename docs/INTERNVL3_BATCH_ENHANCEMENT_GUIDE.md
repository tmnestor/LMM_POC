# InternVL3 Document-Aware Batch Enhancement Guide

## Overview

This guide explains how to enhance `internvl3_document_aware_batch.ipynb` with the advanced features from `llama_document_aware_batch.ipynb` while **completely avoiding the BatchProcessor class** that causes infinite recursion issues.

## Critical Design Principle

**🚫 NO BatchProcessor Integration** - The BatchProcessor class has proven problematic with infinite recursion in InternVL3. This guide maintains the clean, direct processing approach that already works while adding sophisticated analytics and reporting.

## Architecture Comparison

### Current InternVL3 Approach (Working ✅)
```python
# Simple, direct processing loop
for image_path in image_files:
    result = processor.process_single_image(str(image_path))
    results.append(result)
```

### Enhanced InternVL3 Approach (Target ✅)
```python
# Same direct loop + advanced analytics
for image_path in image_files:
    result = processor.process_single_image(str(image_path))
    results.append(result)

# NEW: Add comprehensive analytics
analytics = BatchAnalytics(results, processing_times)
visualizer = BatchVisualizer()
reporter = BatchReporter(results, processing_times, doc_types, timestamp)
```

## Missing Features Analysis

### ❌ Current InternVL3 Limitations
1. **Basic results storage** - Simple list and CSV export only
2. **Minimal analytics** - Basic success rate and timing
3. **No visualizations** - Text-based dashboard only
4. **Limited reporting** - Simple text summary file
5. **Ad-hoc output management** - Basic directory structure

### ✅ Target Enhanced Features
1. **Comprehensive DataFrames** - Multiple structured analysis tables
2. **Advanced analytics** - Statistical summaries, accuracy distributions
3. **Professional visualizations** - 2x2 dashboards, accuracy heatmaps
4. **Executive reporting** - Markdown summaries, JSON exports
5. **Structured output system** - Organized directory hierarchy

---

## Implementation Guide

## 1. Enhanced Configuration System

### Replace Basic Configuration
Replace the simple configuration in InternVL3 with the comprehensive CONFIG system:

```python
# BEFORE (InternVL3 current approach)
MODEL_PATH = "/home/jovyan/nfs_share/models/InternVL3-8B"
IMAGE_DIR = "evaluation_data"
OUTPUT_DIR = "batch_results"

# AFTER (Enhanced approach)
CONFIG = {
    # Model settings
    'MODEL_PATH': "/home/jovyan/nfs_share/models/InternVL3-8B",

    # Batch settings
    'DATA_DIR': 'evaluation_data',
    'GROUND_TRUTH': 'evaluation_data/ground_truth.csv',
    'MAX_IMAGES': None,  # None for all, or set limit
    'DOCUMENT_TYPES': None,  # None for all, or ['invoice', 'receipt']

    # Output settings
    'OUTPUT_BASE': os.getenv('OUTPUT_DIR', 'output'),
    'VERBOSE': True,

    # V100 optimization
    'USE_QUANTIZATION': True,
    'DEVICE_MAP': 'auto',
    'MAX_NEW_TOKENS': 4000,
    'TORCH_DTYPE': 'bfloat16',
    'LOW_CPU_MEM_USAGE': True
}
```

### Add Structured Output Directory Setup
```python
# Enhanced output directory structure
from datetime import datetime
from pathlib import Path

OUTPUT_BASE = Path(CONFIG['OUTPUT_BASE'])
if not OUTPUT_BASE.is_absolute():
    OUTPUT_BASE = Path.cwd() / OUTPUT_BASE

BATCH_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIRS = {
    'base': OUTPUT_BASE,
    'batch': OUTPUT_BASE / 'batch_results',
    'csv': OUTPUT_BASE / 'csv',
    'visualizations': OUTPUT_BASE / 'visualizations',
    'reports': OUTPUT_BASE / 'reports'
}

for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)
```

## 2. Imports Enhancement

### Add Required Import Section
Add these imports to the existing InternVL3 notebook:

```python
# NEW: Add these imports for advanced analytics
from common.batch_analytics import BatchAnalytics
from common.batch_visualizations import BatchVisualizer
from common.batch_reporting import BatchReporter
from common.evaluation_metrics import load_ground_truth

# Keep existing InternVL3 imports
from models.document_aware_internvl3_processor import DocumentAwareInternVL3Processor
from common.unified_schema import DocumentTypeFieldSchema
```

## 3. Processing Loop Enhancement

### Current InternVL3 Processing Loop
```python
# Current approach (keep this structure!)
for i, image_path in enumerate(image_files, 1):
    print(f"📄 Processing {i}/{len(image_files)}: {image_path.name}")

    try:
        image_start = time.perf_counter()

        # Document type detection and processing
        doc_type = detect_document_type(image_path)
        field_list = schema_loader.get_document_fields(doc_type)
        processor.field_list = field_list
        result = processor.process_single_image(str(image_path))

        # Basic metadata
        result["image_file"] = image_path.name
        result["document_type"] = doc_type
        result["processing_time"] = time.perf_counter() - image_start

        results.append(result)
```

### Enhanced Processing Loop
```python
# Enhanced approach - ADD evaluation and structured data
import time
from pathlib import Path

# Load ground truth data ONCE before loop
try:
    ground_truth_data = load_ground_truth(CONFIG['GROUND_TRUTH'])
    print(f"✅ Ground truth loaded for {len(ground_truth_data)} images")
except Exception as e:
    print(f"⚠️ Could not load ground truth: {e}")
    ground_truth_data = {}

# Initialize enhanced storage
batch_results = []
processing_times = []
document_types_found = {}

# Enhanced processing loop (KEEP same direct structure)
for i, image_path in enumerate(image_files, 1):
    print(f"📄 Processing {i}/{len(image_files)}: {image_path.name}")

    try:
        image_start = time.perf_counter()

        # SAME: Document type detection and processing
        doc_type = detect_document_type(image_path)
        field_list = schema_loader.get_document_fields(doc_type)
        processor.field_list = field_list
        result = processor.process_single_image(str(image_path))

        # SAME: Basic metadata
        image_name = image_path.name
        processing_time = time.perf_counter() - image_start

        # NEW: Enhanced metadata for analytics
        result.update({
            "image_file": image_name,
            "image_name": image_name,  # BatchAnalytics expects this
            "image_path": str(image_path),
            "document_type": doc_type,
            "processing_time": processing_time,
            "prompt_used": f"internvl3_{doc_type.lower()}",
            "timestamp": datetime.now().isoformat(),
        })

        # NEW: Ground truth evaluation
        ground_truth = ground_truth_data.get(image_name, {})
        if ground_truth:
            from common.document_type_metrics import DocumentTypeEvaluator
            evaluator = DocumentTypeEvaluator()

            extracted_data = result.get("extracted_data", {})
            evaluation = evaluator.evaluate_extraction(
                extracted_data, ground_truth, doc_type
            )

            # Format for BatchAnalytics compatibility
            if evaluation and "overall_metrics" in evaluation:
                evaluation["overall_accuracy"] = evaluation["overall_metrics"].get("overall_accuracy", 0)
                evaluation["fields_extracted"] = evaluation["overall_metrics"].get("total_fields_evaluated", 0)
                evaluation["fields_matched"] = evaluation["overall_metrics"].get("fields_correct", 0)
                evaluation["total_fields"] = evaluation["overall_metrics"].get("total_fields_evaluated", 0)

            result["evaluation"] = evaluation
        else:
            result["evaluation"] = {
                "error": f"No ground truth for {image_name}",
                "overall_accuracy": 0,
            }

        # Store results
        batch_results.append(result)
        processing_times.append(processing_time)
        document_types_found[doc_type] = document_types_found.get(doc_type, 0) + 1

        print(f"   ✅ Completed in {processing_time:.2f}s")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Store error result
        batch_results.append({
            "image_name": image_path.name,
            "image_path": str(image_path),
            "error": str(e),
            "processing_time": time.perf_counter() - image_start if 'image_start' in locals() else 0,
        })
```

## 4. Advanced Analytics Integration

### Add BatchAnalytics After Processing
```python
# NEW: Advanced analytics section (add after processing loop)
print("\n" + "="*60)
print("📊 ADVANCED ANALYTICS GENERATION")
print("="*60)

# Create analytics with enhanced data
analytics = BatchAnalytics(batch_results, processing_times)

# Generate comprehensive DataFrames
saved_files, df_results, df_summary, df_doctype_stats, df_field_stats = analytics.save_all_dataframes(
    OUTPUT_DIRS['csv'], BATCH_TIMESTAMP, verbose=CONFIG['VERBOSE']
)

# Display enhanced summary
print("\n[bold blue]📊 Enhanced Results Summary[/bold blue]")
display(df_summary)

# Display document type statistics if available
if not df_doctype_stats.empty:
    print("\n[bold blue]📋 Document Type Performance[/bold blue]")
    display(df_doctype_stats)
```

## 5. Professional Visualizations

### Add BatchVisualizer Integration
```python
# NEW: Professional visualization section
print("\n" + "="*60)
print("📈 VISUALIZATION GENERATION")
print("="*60)

# Create visualizations
visualizer = BatchVisualizer()

viz_files = visualizer.create_all_visualizations(
    df_results,
    df_doctype_stats,
    OUTPUT_DIRS['visualizations'],
    BATCH_TIMESTAMP,
    show=False  # Set to True to display in notebook
)

print("✅ Visualizations created:")
for viz_type, path in viz_files.items():
    print(f"   📊 {viz_type}: {path.name}")
```

## 6. Executive Reporting System

### Add BatchReporter Integration
```python
# NEW: Executive reporting section
print("\n" + "="*60)
print("📝 EXECUTIVE REPORT GENERATION")
print("="*60)

# Generate comprehensive reports
reporter = BatchReporter(
    batch_results,
    processing_times,
    document_types_found,
    BATCH_TIMESTAMP
)

# Save all reports
report_files = reporter.save_all_reports(
    OUTPUT_DIRS,
    df_results,
    df_summary,
    df_doctype_stats,
    CONFIG['MODEL_PATH'],
    {
        'data_dir': CONFIG['DATA_DIR'],
        'ground_truth': CONFIG['GROUND_TRUTH'],
        'max_images': CONFIG['MAX_IMAGES'],
        'document_types': CONFIG['DOCUMENT_TYPES']
    },
    {
        'use_quantization': CONFIG['USE_QUANTIZATION'],
        'device_map': CONFIG['DEVICE_MAP'],
        'max_new_tokens': CONFIG['MAX_NEW_TOKENS'],
        'torch_dtype': CONFIG['TORCH_DTYPE'],
        'low_cpu_mem_usage': CONFIG['LOW_CPU_MEM_USAGE']
    },
    verbose=CONFIG['VERBOSE']
)

print("✅ Reports generated:")
for report_type, path in report_files.items():
    print(f"   📄 {report_type}: {path.name}")
```

## 7. Enhanced Final Summary

### Replace Simple Summary with Professional Dashboard
```python
# NEW: Professional final summary section
from rich.console import Console
from IPython.display import Image, display

console = Console()
console.rule("[bold green]InternVL3 Batch Processing Complete[/bold green]")

# Calculate enhanced metrics
total_images = len(batch_results)
successful = len([r for r in batch_results if 'error' not in r])
avg_accuracy = df_results['overall_accuracy'].mean() if len(df_results) > 0 else 0

print(f"[bold green]✅ Processed: {total_images} images[/bold green]")
print(f"[cyan]Success Rate: {(successful/total_images*100):.1f}%[/cyan]")
print(f"[cyan]Average Accuracy: {avg_accuracy:.2f}%[/cyan]")
print(f"[cyan]Output Directory: {OUTPUT_BASE}[/cyan]")

# Display visual dashboard if available
dashboard_files = list(OUTPUT_DIRS['visualizations'].glob(f"dashboard_{BATCH_TIMESTAMP}.png"))
if dashboard_files:
    dashboard_path = dashboard_files[0]
    print(f"\n[bold blue]📊 Visual Dashboard:[/bold blue]")
    display(Image(str(dashboard_path)))
else:
    print(f"\n[yellow]⚠️ Dashboard not found in {OUTPUT_DIRS['visualizations']}[/yellow]")

# Output file summary
print(f"\n[bold blue]📁 Generated Files:[/bold blue]")
print(f"   📊 CSV Files: {len(saved_files)} files in csv/")
print(f"   📈 Visualizations: {len(viz_files)} files in visualizations/")
print(f"   📝 Reports: {len(report_files)} files in reports/")
```

---

## Implementation Steps

### Step 1: Backup Current Notebook
```bash
cp internvl3_document_aware_batch.ipynb internvl3_document_aware_batch_backup.ipynb
```

### Step 2: Add Enhanced Configuration
- Replace simple variables with CONFIG dictionary
- Add OUTPUT_DIRS structure setup
- Import datetime and Path utilities

### Step 3: Add Required Imports
- Import BatchAnalytics, BatchVisualizer, BatchReporter
- Import evaluation utilities
- Keep all existing InternVL3 imports

### Step 4: Enhance Processing Loop
- Keep exact same direct processing approach
- Add enhanced metadata to results
- Add ground truth evaluation
- Maintain structured result storage

### Step 5: Add Analytics Section
- Create BatchAnalytics instance
- Generate comprehensive DataFrames
- Display enhanced summaries

### Step 6: Add Visualization Section
- Create BatchVisualizer instance
- Generate professional dashboards and heatmaps
- Save visualization files

### Step 7: Add Reporting Section
- Create BatchReporter instance
- Generate Markdown and JSON reports
- Save comprehensive documentation

### Step 8: Enhance Final Summary
- Replace basic output with professional dashboard
- Display visual results
- Provide comprehensive file listing

## Key Benefits of This Approach

### ✅ Maintains Working Architecture
- Keeps proven direct processing loop
- No BatchProcessor complexity
- No recursion risks

### ✅ Adds Professional Features
- Comprehensive analytics and reporting
- Publication-quality visualizations
- Executive-level summaries

### ✅ Production-Ready Output
- Structured file organization
- Multiple export formats
- Deployment readiness assessment

### ✅ Enhanced Debugging
- Detailed error tracking
- Performance metrics
- Ground truth comparison

## Troubleshooting

### If You Encounter Issues
1. **Import errors**: Ensure all common modules are available
2. **Path errors**: Verify OUTPUT_DIRS are created correctly
3. **Data format errors**: Check that result structure matches BatchAnalytics expectations
4. **Visualization errors**: Ensure matplotlib/seaborn are installed

### Performance Optimization
- Keep existing V100 optimizations
- Use show=False for visualizations to reduce memory usage
- Consider MAX_IMAGES limit for testing

### Validation Checklist
- [ ] Direct processing loop still works
- [ ] No BatchProcessor imported or used
- [ ] Enhanced analytics generate successfully
- [ ] Visualizations save correctly
- [ ] Reports are comprehensive and readable
- [ ] Output directory structure is clean

## Conclusion

This enhancement approach maintains the reliability of the current InternVL3 batch processing while adding the sophisticated analytics, visualization, and reporting capabilities that make the Llama version so powerful. The key is avoiding BatchProcessor entirely and building analytics around the existing working structure.

The result is a production-ready batch processing system that provides comprehensive insights into model performance while maintaining the simplicity and reliability that makes the InternVL3 approach successful.