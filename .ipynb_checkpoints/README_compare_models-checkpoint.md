# Model Comparison Dashboard

A standalone script to generate comparative visualizations between Llama and InternVL3 model evaluation results.

## Usage

### Automatic Discovery (Recommended)
```bash
# Auto-discover the latest JSON evaluation files in the output directory
python compare_models.py --auto
```

### Manual File Specification
```bash
# Specify exact JSON files
python compare_models.py \
  --llama /path/to/llama_evaluation_results_20250811_020529.json \
  --internvl3 /path/to/internvl3_evaluation_results_20250811_020530.json
```

### Custom Output Directory
```bash
# Use a different output directory
python compare_models.py --auto --output-dir /custom/output/path
```

## Generated Files

The script creates three types of outputs:

1. **Side-by-side Comparison Dashboard** (`model_comparison_dashboard_YYYYMMDD_HHMMSS.png`)
   - Overall accuracy comparison
   - Document processing statistics
   - Field quality distribution
   - Performance range analysis

2. **Field Accuracy Heatmap** (`field_accuracy_heatmap_YYYYMMDD_HHMMSS.png`)
   - Field-by-field accuracy comparison
   - Performance delta visualization (Llama - InternVL3)
   - Color-coded performance levels

3. **Executive Comparison Summary** (`model_comparison_summary_YYYYMMDD_HHMMSS.md`)
   - Key performance metrics
   - Deployment recommendations
   - Technical specifications comparison

## Requirements

- Both `llama_evaluation_results_*.json` and `internvl3_evaluation_results_*.json` files
- Dependencies: matplotlib, pandas, seaborn, numpy (included in environment.yml)

## Example Output

```
📊 MODEL COMPARISON DASHBOARD GENERATOR
===============================================
🔍 Auto-discovering latest evaluation results...
✅ Loaded results for Llama-3.2-11B-Vision-Instruct
✅ Loaded results for InternVL3-2B
📁 Llama results: /output/llama_evaluation_results_20250811_020529.json
📁 InternVL3 results: /output/internvl3_evaluation_results_20250811_020530.json

🎨 Generating comparison visualizations...
🎨 Creating side-by-side model comparison...
✅ Comparison dashboard saved: model_comparison_dashboard_20250811_030245.png
🎨 Creating field accuracy heatmap...
✅ Field accuracy heatmap saved: field_accuracy_heatmap_20250811_030245.png
📝 Generating comparison summary...
✅ Comparison summary saved: model_comparison_summary_20250811_030245.md

✅ MODEL COMPARISON COMPLETED SUCCESSFULLY!
=================================================
📁 Generated files:
   • model_comparison_dashboard_20250811_030245.png
   • field_accuracy_heatmap_20250811_030245.png
   • model_comparison_summary_20250811_030245.md

📈 Key Insights:
   • Llama accuracy: 85.2%
   • InternVL3 accuracy: 82.1%
   • Performance delta: +3.1%
   🦙 Llama leads in overall accuracy
```

## Integration with Main Pipeline

The comparison script works seamlessly with the main evaluation pipeline:

1. Run Llama evaluation: `python llama_keyvalue.py`
2. Run InternVL3 evaluation: `python internvl3_keyvalue.py`
3. Generate comparison: `python compare_models.py --auto`

All outputs will be saved to the configured OUTPUT_DIR with timestamps for easy tracking.