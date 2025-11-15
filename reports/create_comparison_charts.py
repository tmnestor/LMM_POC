#!/usr/bin/env python3
"""
V100 vs H200 Hardware Comparison Visualizations
Generates colored charts for the comparison report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'V100 Quantized': '#e74c3c',      # Red - V100
    'H200 Quantized': '#3498db',      # Blue - H200 Quantized
    'H200 bfloat16': '#2ecc71',       # Green - H200 bfloat16
    'H200 InternVL3.5': '#9b59b6',    # Purple - H200 InternVL3.5
}

DOCTYPE_COLORS = {
    'RECEIPT': '#2ecc71',             # Green - easiest
    'INVOICE': '#f39c12',             # Orange - moderate
    'BANK_STATEMENT': '#e74c3c',      # Red - hardest
}

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Read data
processing_df = pd.read_csv(OUTPUT_DIR / 'graph_data_processing_time.csv')
accuracy_df = pd.read_csv(OUTPUT_DIR / 'graph_data_accuracy.csv')
doctype_df = pd.read_csv(OUTPUT_DIR / 'graph_data_document_type_accuracy.csv')
detailed_df = pd.read_csv(OUTPUT_DIR / 'graph_data_detailed_results.csv')

def create_processing_time_chart():
    """Processing time horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = processing_df['Configuration']
    times = processing_df['Processing_Time_s']
    colors = [COLORS[c] for c in configs]

    bars = ax.barh(configs, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, time, throughput in zip(bars, times, processing_df['Throughput_img_per_min'], strict=False):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{time:.2f}s ({throughput:.2f} img/min)',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Processing Time per Image (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Processing Time Comparison: V100 vs H200\n(Lower is Better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add speedup annotations
    v100_time = processing_df.loc[0, 'Processing_Time_s']
    for i, time in enumerate(times[1:], 1):
        speedup = v100_time / time
        ax.text(0.5, i - 0.3, f'{speedup:.1f}x faster',
                ha='left', va='center', fontsize=9, style='italic', color='darkgreen')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_processing_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: chart_processing_time.png")

def create_accuracy_comparison_chart():
    """Accuracy comparison grouped bar chart"""
    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(accuracy_df))
    width = 0.2

    metrics = ['Avg_Accuracy', 'Median_Accuracy', 'Min_Accuracy', 'Max_Accuracy']
    metric_labels = ['Average', 'Median', 'Minimum', 'Maximum']
    metric_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors, strict=False)):
        offset = width * (i - 1.5)
        bars = ax.bar([xi + offset for xi in x], accuracy_df[metric], width,
                      label=label, color=color, alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison: V100 vs H200\n(All Metrics)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(accuracy_df['Configuration'], rotation=0, ha='center')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: chart_accuracy_comparison.png")

def create_document_type_chart():
    """Document type accuracy grouped by configuration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Grouped by configuration
    configs = doctype_df['Configuration'].unique()
    doc_types = ['RECEIPT', 'INVOICE', 'BANK_STATEMENT']
    x = range(len(configs))
    width = 0.25

    for i, doc_type in enumerate(doc_types):
        offset = width * (i - 1)
        data = doctype_df[doctype_df['Document_Type'] == doc_type]
        accuracies = [data[data['Configuration'] == c]['Accuracy'].to_numpy()[0] for c in configs]

        bars = ax1.bar([xi + offset for xi in x], accuracies, width,
                      label=doc_type, color=DOCTYPE_COLORS[doc_type],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, acc in zip(bars, accuracies, strict=False):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Document Type Accuracy by Configuration\n(Grouped by Configuration)',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=9)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    # Chart 2: Grouped by document type
    x = range(len(doc_types))
    width = 0.2

    for i, config in enumerate(configs):
        offset = width * (i - 1.5)
        data = doctype_df[doctype_df['Configuration'] == config]
        accuracies = [data[data['Document_Type'] == dt]['Accuracy'].to_numpy()[0] for dt in doc_types]

        bars = ax2.bar([xi + offset for xi in x], accuracies, width,
                      label=config, color=COLORS[config],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, acc in zip(bars, accuracies, strict=False):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=7)

    ax2.set_xlabel('Document Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Document Type Accuracy by Configuration\n(Grouped by Document Type)',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(doc_types, fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_document_type_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: chart_document_type_accuracy.png")

def create_speed_vs_accuracy_scatter():
    """Speed vs Accuracy scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 8))

    configs = detailed_df['Configuration']
    accuracies = detailed_df['Avg_Accuracy']
    throughputs = detailed_df['Throughput_img_per_min']
    colors = [COLORS[c] for c in configs]

    scatter = ax.scatter(throughputs, accuracies, s=500, c=colors,
                        alpha=0.7, edgecolors='black', linewidth=2)

    # Add labels
    for i, (config, x, y) in enumerate(zip(configs, throughputs, accuracies, strict=False)):
        ax.annotate(config, (x, y), xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Add quadrant lines
    ax.axhline(y=accuracies.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=throughputs.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Quadrant labels
    ax.text(throughputs.max() * 0.95, accuracies.max() * 0.98,
           'IDEAL\n(Fast & Accurate)', ha='right', va='top',
           fontsize=11, style='italic', color='green', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.set_xlabel('Throughput (images/min)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Speed vs Accuracy Trade-off\n(Top-right quadrant is ideal)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_speed_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: chart_speed_vs_accuracy.png")

def create_throughput_improvement_chart():
    """Throughput improvement relative to V100"""
    fig, ax = plt.subplots(figsize=(12, 6))

    v100_throughput = processing_df.loc[0, 'Throughput_img_per_min']
    improvements = [(t / v100_throughput - 1) * 100 for t in processing_df['Throughput_img_per_min']]

    configs = processing_df['Configuration']
    colors = [COLORS[c] for c in configs]

    bars = ax.barh(configs, improvements, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, imp in zip(bars, improvements, strict=False):
        width = bar.get_width()
        label = f'+{imp:.0f}%' if imp > 0 else f'{imp:.0f}%'
        ax.text(width + 5, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Throughput Improvement vs V100 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Improvement Over V100 Baseline\n(Percentage Increase)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='-', linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_throughput_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: chart_throughput_improvement.png")

def create_executive_dashboard():
    """Executive summary dashboard with all key metrics"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Processing Time (top-left, wide)
    ax1 = fig.add_subplot(gs[0, :2])
    configs = processing_df['Configuration']
    times = processing_df['Processing_Time_s']
    colors = [COLORS[c] for c in configs]
    bars = ax1.barh(configs, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, time in zip(bars, times, strict=False):
        ax1.text(time + 2, bar.get_y() + bar.get_height()/2, f'{time:.1f}s',
                ha='left', va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Processing Time (s)', fontsize=10, fontweight='bold')
    ax1.set_title('Processing Time per Image', fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 2. Average Accuracy (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    accuracies = detailed_df['Avg_Accuracy']
    bars = ax2.bar(range(len(configs)), accuracies, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, acc in zip(bars, accuracies, strict=False):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=8)
    ax2.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Average Accuracy', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)

    # 3. Document Type Accuracy (middle row, full width)
    ax3 = fig.add_subplot(gs[1, :])
    doc_types = ['RECEIPT', 'INVOICE', 'BANK_STATEMENT']
    x = range(len(configs))
    width = 0.25
    for i, doc_type in enumerate(doc_types):
        offset = width * (i - 1)
        data = doctype_df[doctype_df['Document_Type'] == doc_type]
        accuracies = [data[data['Configuration'] == c]['Accuracy'].to_numpy()[0] for c in configs]
        bars = ax3.bar([xi + offset for xi in x], accuracies, width,
                      label=doc_type, color=DOCTYPE_COLORS[doc_type],
                      alpha=0.8, edgecolor='black', linewidth=1)
        for bar, acc in zip(bars, accuracies, strict=False):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=9)
    ax3.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Accuracy by Document Type', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)

    # 4. Throughput (bottom-left)
    ax4 = fig.add_subplot(gs[2, 0])
    throughputs = processing_df['Throughput_img_per_min']
    bars = ax4.bar(range(len(configs)), throughputs, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, tp in zip(bars, throughputs, strict=False):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{tp:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=8)
    ax4.set_ylabel('Throughput (img/min)', fontsize=10, fontweight='bold')
    ax4.set_title('Processing Throughput', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 5. Speedup vs V100 (bottom-middle)
    ax5 = fig.add_subplot(gs[2, 1])
    speedups = processing_df['Speedup_vs_V100']
    bars = ax5.bar(range(len(configs)), speedups, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, sp in zip(bars, speedups, strict=False):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{sp:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5.set_xticks(range(len(configs)))
    ax5.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=8)
    ax5.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold')
    ax5.set_title('Speedup vs V100', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 6. Summary Text (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    # Convert to pandas Series for min/max operations
    times_series = pd.Series(times.to_numpy() if hasattr(times, 'to_numpy') else times)
    accuracies_series = pd.Series(accuracies.to_numpy() if hasattr(accuracies, 'to_numpy') else accuracies)
    throughputs_series = pd.Series(throughputs.to_numpy() if hasattr(throughputs, 'to_numpy') else throughputs)

    summary_text = f"""
KEY FINDINGS

Speed Impact:
• V100: {times_series.iloc[0]:.1f}s/image
• H200 best: {times_series.min():.1f}s/image
• Improvement: {times_series.iloc[0]/times_series.min():.1f}x faster

Accuracy Impact:
• V100: {accuracies_series.iloc[0]:.1f}%
• H200 best: {accuracies_series.max():.1f}%
• Difference: {accuracies_series.max()-accuracies_series.iloc[0]:.1f}%

Throughput:
• V100: {throughputs_series.iloc[0]:.2f} img/min
• H200 best: {throughputs_series.max():.2f} img/min
• Improvement: {(throughputs_series.max()/throughputs_series.iloc[0]-1)*100:.0f}%

Model Impact:
• InternVL3-8B: {accuracies_series.iloc[2]:.1f}%
• InternVL3.5-8B: {accuracies_series.iloc[3]:.1f}%
• Improvement: {accuracies_series.iloc[3]-accuracies_series.iloc[2]:.1f}%
"""
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('V100 vs H200 Hardware Comparison - Executive Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_DIR / 'chart_executive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: chart_executive_dashboard.png")

def main():
    """Generate all charts"""
    print("\n🎨 Generating V100 vs H200 comparison charts...\n")

    create_processing_time_chart()
    create_accuracy_comparison_chart()
    create_document_type_chart()
    create_speed_vs_accuracy_scatter()
    create_throughput_improvement_chart()
    create_executive_dashboard()

    print(f"\n✅ All charts created successfully in {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • chart_processing_time.png - Processing time comparison")
    print("  • chart_accuracy_comparison.png - Accuracy metrics comparison")
    print("  • chart_document_type_accuracy.png - Document type breakdown")
    print("  • chart_speed_vs_accuracy.png - Speed vs accuracy scatter plot")
    print("  • chart_throughput_improvement.png - Throughput improvement over V100")
    print("  • chart_executive_dashboard.png - Executive summary dashboard")
    print("\nYou can now embed these images in your markdown report!")

if __name__ == '__main__':
    main()
