#!/usr/bin/env python
"""
Compare Baseline Hallucinations - Analyze and visualize hallucination rates across baselines.

This script compares multiple baseline runs to analyze hallucination patterns across
different datasets and models. It's particularly useful for:
- Validating baseline ensemble stability
- Comparing hallucination rates across datasets (NQSwap, MMLU, HellaSwag)
- Identifying dataset difficulty for hallucination detection
- Verifying baseline quality before steering experiments

Output Visualizations:
    1. Bar chart: Hallucination rates across baselines
    2. Statistics table: API failures, valid samples, hallucination counts
    3. Per-sample overlap analysis (if baselines share samples)

Usage Examples:

1. Compare Llama-3 baselines across 3 datasets:
   python -m steer.compare_baseline
   # Edit BASELINE_DIRS to include Llama baselines
   # Uncomment the Llama section in BASELINE_DIRS list

2. Compare Qwen vs Llama on same dataset:
   # Set BASELINE_DIRS = [
   #     "./data/baseline_results_qwen2.5_7b/ensembled_nqswap/BASELINE_...",
   #     "./data/baseline_results_llama/ensembled_nqswap/BASELINE_..."
   # ]

3. Validate ensemble stability (compare 5 individual runs):
   # Set BASELINE_DIRS to 5 individual baseline runs before ensemble
   # Check if hallucination rates are similar (within 2-3%)

Configuration:
    1. Edit BASELINE_DIRS list (lines 23-42) with paths to baseline directories
    2. Each directory must contain: baseline_evaluation.pkl
    3. Set OUTPUT_DIR for plot output location
    4. Run: python -m steer.compare_baseline

Hallucination Scoring:
    - Score 0: No hallucination (faithful answer)
    - Score 1: Hallucination (incorrect/unfaithful answer)
    - Score 2: API failure (excluded from rate calculations)
    - Rate = count(score==1) / count(score in {0,1})

Requirements:
    - Baseline directories from baseline_run.py or baseline_ensemble_voter.py
    - Each must have baseline_evaluation.pkl file
"""

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these paths to your baseline directories
# ============================================================================
BASELINE_DIRS = [
    # Gemma Ensembled Baselines
    # "./data/baseline_results_gemma/ensembled_nqswap/BASELINE_ENSEMBLE_VOTED_20260119_061318",
    # "./data/baseline_results_gemma/ensembled_mmlu/BASELINE_ENSEMBLE_VOTED_20260119_064314",
    # "./data/baseline_results_gemma/ensembled_hellaswag/BASELINE_ENSEMBLE_VOTED_20260119_061211",

    # Qwen Ensembled Baselines
    # "./data/baseline_results_qwen2.5_7b/ensembled_hellaswag/BASELINE_ENSEMBLE_VOTED_20260128_022207",
    # "./data/baseline_results_qwen2.5_7b/ensembled_mmlu/BASELINE_ENSEMBLE_VOTED_20260128_022220",
    # "./data/baseline_results_qwen2.5_7b/ensembled_nqswap/BASELINE_ENSEMBLE_VOTED_20260119_062044",

    # Llama Ensembled Baselines
    "./data/baseline_results_llama/ensembled_hellaswag/BASELINE_ENSEMBLE_VOTED_20260105_053200",
    "./data/baseline_results_llama/ensembled_mmlu/BASELINE_ENSEMBLE_VOTED_20260105_015656",
    "./data/baseline_results_llama/ensembled_nqswap/BASELINE_ENSEMBLE_VOTED_20251215_013922"
]

OUTPUT_DIR = "./data/baseline_results_llama/baseline_comparison_plots"  # Where to save plots
# ============================================================================


def get_dataset_name(baseline_dir):
    """Extract dataset name from parent directory path."""
    path = Path(baseline_dir)
    # Get parent directory name (e.g., 'ensembled_coqa')
    parent_name = path.parent.name
    # Remove 'ensembled_' prefix if present
    if parent_name.startswith('ensembled_'):
        return parent_name.replace('ensembled_', '')
    return parent_name


def load_baseline_evaluation(baseline_dir):
    """Load baseline_evaluation.pkl from a directory."""
    eval_file = Path(baseline_dir) / "baseline_evaluation.pkl"
    if not eval_file.exists():
        raise FileNotFoundError(f"Not found: {eval_file}")
    
    with open(eval_file, 'rb') as f:
        return pickle.load(f)


def load_baseline_config(baseline_dir):
    """Load baseline_config.json to extract model name."""
    import json
    config_file = Path(baseline_dir) / "baseline_config.json"
    if not config_file.exists():
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
        return config.get('model', 'Unknown')


def analyze_baselines(baseline_dirs):
    """Analyze hallucination rates for each baseline.
    
    Excludes data points with hallucination label 2 (API failures).
    Only counts valid samples (labels 0 or 1) in statistics.
    Groups results by dataset and tracks model information.
    """
    from collections import defaultdict
    results_by_dataset = defaultdict(list)  # dataset -> list of {model, stats}
    
    for baseline_dir in baseline_dirs:
        dataset_name = get_dataset_name(baseline_dir)
        model_name = load_baseline_config(baseline_dir)
        
        try:
            evaluation = load_baseline_evaluation(baseline_dir)
            
            # Filter out API failures (label 2) - only keep valid samples (0 or 1)
            valid_samples = {k: v for k, v in evaluation.items() if v != 2}
            api_failures = len(evaluation) - len(valid_samples)
            
            total = len(valid_samples)
            hallucinations = sum(1 for v in valid_samples.values() if v == 1)
            rate = (hallucinations / total * 100) if total > 0 else 0
            
            result_entry = {
                'model': model_name or 'Unknown',
                'total': total,
                'hallucinations': hallucinations,
                'no_hallucinations': total - hallucinations,
                'rate': rate,
                'api_failures': api_failures,
                'baseline_dir': baseline_dir
            }
            
            results_by_dataset[dataset_name].append(result_entry)
            
            display_name = f"{dataset_name} ({model_name})" if model_name else dataset_name
            if api_failures > 0:
                print(f"✓ {display_name}: {hallucinations}/{total} ({rate:.1f}%) [excluded {api_failures} API failures]")
            else:
                print(f"✓ {display_name}: {hallucinations}/{total} ({rate:.1f}%)")
        except Exception as e:
            print(f"✗ {dataset_name}: Error - {e}")
    
    return results_by_dataset


def plot_bar_chart(results, output_dir):
    """Create bar chart comparing hallucination counts."""
    # Sort by hallucination count (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['hallucinations'], reverse=True)
    names = [item[0] for item in sorted_results]
    hallucinations = [item[1]['hallucinations'] for item in sorted_results]
    no_hallucinations = [item[1]['no_hallucinations'] for item in sorted_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(names))
    width = 0.35
    
    ax.bar(x, no_hallucinations, width, label='No Hallucination', color='#2ecc71')
    ax.bar(x, hallucinations, width, bottom=no_hallucinations, 
           label='Hallucination', color='#e74c3c')
    
    ax.set_xlabel('Baseline', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Hallucination Comparison Across Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / "hallucination_comparison_bar.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved bar chart: {output_file}")
    plt.close()


def plot_rate_comparison(results, output_dir):
    """Create bar chart of hallucination rates."""
    # Sort by hallucination rate (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rate'], reverse=True)
    names = [item[0] for item in sorted_results]
    rates = [item[1]['rate'] for item in sorted_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, rates, color='#3498db', edgecolor='black', linewidth=1.2)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Baseline', fontsize=12)
    ax.set_ylabel('Hallucination Rate (%)', fontsize=12)
    ax.set_title('Hallucination Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, max(rates) * 1.15)  # Extra space for labels
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / "hallucination_rate_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved rate comparison: {output_file}")
    plt.close()


def get_model_color_mapping(results_by_dataset):
    """Create a consistent color mapping for all unique models across datasets.
    
    Returns a dictionary mapping model names to colors.
    """
    # Collect all unique model names
    all_models = set()
    for models_list in results_by_dataset.values():
        for model_data in models_list:
            all_models.add(model_data['model'])
    
    # Sort for consistency
    sorted_models = sorted(all_models)
    
    # Define a rich color palette (handles up to 20 models)
    color_palette = [
        '#e74c3c',  # Red
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#e67e22',  # Carrot
        '#34495e',  # Wet Asphalt
        '#16a085',  # Green Sea
        '#27ae60',  # Nephritis
        '#2980b9',  # Belize Hole
        '#8e44ad',  # Wisteria
        '#c0392b',  # Pomegranate
        '#d35400',  # Pumpkin
        '#7f8c8d',  # Asbestos
        '#f1c40f',  # Sun Flower
        '#e91e63',  # Pink
        '#00bcd4',  # Cyan
        '#4caf50',  # Light Green
        '#ff5722',  # Deep Orange
    ]
    
    # Create mapping
    model_colors = {}
    for idx, model in enumerate(sorted_models):
        model_colors[model] = color_palette[idx % len(color_palette)]
    
    return model_colors


def plot_pie_charts(results, output_dir):
    """Create individual pie charts for each baseline."""
    # Sort by hallucination rate (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rate'], reverse=True)
    num_baselines = len(sorted_results)
    cols = min(3, num_baselines)
    rows = (num_baselines + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_baselines == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_baselines > 1 else [axes]
    
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, (name, data) in enumerate(sorted_results):
        ax = axes[idx]
        sizes = [data['no_hallucinations'], data['hallucinations']]
        labels = [f"No Hall.\n({data['no_hallucinations']})", 
                  f"Hall.\n({data['hallucinations']})"]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 10})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(name, fontsize=11, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(num_baselines, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Hallucination Distribution per Baseline', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_file = Path(output_dir) / "hallucination_pie_charts.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved pie charts: {output_file}")
    plt.close()


def plot_per_dataset_model_comparison(results_by_dataset, output_dir, model_colors=None):
    """Create per-dataset bar charts comparing hallucination counts across models.
    
    For each dataset with multiple models, creates a graph with:
    - X-axis: model names
    - Y-axis: hallucination counts
    - Bars showing hallucinations vs no hallucinations
    - Consistent colors per model across all datasets
    """
    datasets_with_multiple_models = {k: v for k, v in results_by_dataset.items() if len(v) > 1}
    
    if not datasets_with_multiple_models:
        print("\n[INFO] No datasets with multiple models found. Skipping per-dataset model comparison.")
        return []
    
    # Generate color mapping if not provided
    if model_colors is None:
        model_colors = get_model_color_mapping(results_by_dataset)
    
    print(f"\n[INFO] Found {len(datasets_with_multiple_models)} dataset(s) with multiple models")
    plot_files = []
    
    for dataset_name, models_data in datasets_with_multiple_models.items():
        # Sort by model name for consistent ordering
        models_data_sorted = sorted(models_data, key=lambda x: x['model'])
        
        models = [d['model'] for d in models_data_sorted]
        hallucinations = [d['hallucinations'] for d in models_data_sorted]
        no_hallucinations = [d['no_hallucinations'] for d in models_data_sorted]
        rates = [d['rate'] for d in models_data_sorted]
        colors = [model_colors.get(model, '#95a5a6') for model in models]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 7))
        x = range(len(models))
        width = 0.6
        
        # Stacked bar chart with consistent colors per model
        # Use lighter shade for no hallucination, darker shade for hallucination
        bars1_colors = [mcolors.to_rgba(c, alpha=0.3) for c in colors]
        bars2_colors = colors
        
        bars1 = ax.bar(x, no_hallucinations, width, label='No Hallucination', color=bars1_colors, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, hallucinations, width, bottom=no_hallucinations, label='Hallucination', color=bars2_colors, edgecolor='black', linewidth=1)
        
        # Add percentage labels on top
        for i, (rate, halluc, total) in enumerate(zip(rates, hallucinations, [h + nh for h, nh in zip(hallucinations, no_hallucinations)])):
            ax.text(i, total, f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize axes
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=13, fontweight='bold')
        ax.set_title(f'Hallucination Comparison for {dataset_name.upper()} Dataset\n(Across Different Models)', 
                     fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = Path(output_dir) / f"model_comparison_{dataset_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison for {dataset_name}: {output_file}")
        plt.close()
        
        plot_files.append(str(output_file))
    
    return plot_files


def plot_per_dataset_rate_comparison(results_by_dataset, output_dir, model_colors=None):
    """Create per-dataset bar charts comparing hallucination rates across models.
    
    For each dataset with multiple models, creates a graph with:
    - X-axis: model names
    - Y-axis: hallucination rate (%)
    - Consistent colors per model across all datasets
    - Sorted by hallucination rate (descending)
    """
    datasets_with_multiple_models = {k: v for k, v in results_by_dataset.items() if len(v) > 1}
    
    if not datasets_with_multiple_models:
        return []
    
    # Generate color mapping if not provided
    if model_colors is None:
        model_colors = get_model_color_mapping(results_by_dataset)
    
    plot_files = []
    
    for dataset_name, models_data in datasets_with_multiple_models.items():
        # Sort by hallucination rate (descending) for better visualization
        models_data_sorted = sorted(models_data, key=lambda x: x['rate'], reverse=True)
        
        models = [d['model'] for d in models_data_sorted]
        rates = [d['rate'] for d in models_data_sorted]
        colors = [model_colors.get(model, '#95a5a6') for model in models]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 7))
        
        bars = ax.bar(models, rates, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize axes
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Hallucination Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Hallucination Rate Comparison for {dataset_name.upper()} Dataset\n(Across Different Models)', 
                     fontsize=15, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, max(rates) * 1.15 if rates else 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = Path(output_dir) / f"rate_comparison_{dataset_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved rate comparison for {dataset_name}: {output_file}")
        plt.close()
        
        plot_files.append(str(output_file))
    
    return plot_files


def plot_combined_dataset_comparison(results_by_dataset, output_dir, model_colors=None):
    """Create a combined comparison showing all datasets with multiple models.
    
    Creates two visualizations:
    1. Grouped bar chart: All datasets side-by-side for direct comparison
    2. Grid of subplots: Each dataset in its own subplot for detailed view
    """
    datasets_with_multiple_models = {k: v for k, v in results_by_dataset.items() if len(v) > 1}
    
    if not datasets_with_multiple_models:
        print("\n[INFO] No datasets with multiple models for combined comparison.")
        return []
    
    # Generate color mapping if not provided
    if model_colors is None:
        model_colors = get_model_color_mapping(results_by_dataset)
    
    plot_files = []
    
    # --- Visualization 1: Grouped Bar Chart ---
    # Collect all unique models across all datasets
    all_models = set()
    for models_data in datasets_with_multiple_models.values():
        for model_data in models_data:
            all_models.add(model_data['model'])
    sorted_models = sorted(all_models)
    
    # Prepare data structure: model -> {dataset: rate}
    model_rates = {model: {} for model in sorted_models}
    for dataset_name, models_data in datasets_with_multiple_models.items():
        for model_data in models_data:
            model_rates[model_data['model']][dataset_name] = model_data['rate']
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(max(14, len(sorted_models) * 3), 8))
    
    dataset_names = sorted(datasets_with_multiple_models.keys())
    x = range(len(sorted_models))
    width = 0.8 / len(dataset_names)
    
    # Dataset-specific colors for grouping
    dataset_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:len(dataset_names)]
    
    for idx, dataset in enumerate(dataset_names):
        rates = [model_rates[model].get(dataset, 0) for model in sorted_models]
        offset = (idx - len(dataset_names)/2) * width + width/2
        bars = ax.bar([xi + offset for xi in x], rates, width, 
                      label=dataset.upper(), color=dataset_colors[idx], 
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, rates):
            if rate > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hallucination Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Combined Hallucination Rate Comparison Across All Datasets\n(Grouped by Model)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11, loc='upper right', title='Dataset', title_fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / "combined_rate_comparison_grouped.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined grouped comparison: {output_file}")
    plt.close()
    plot_files.append(str(output_file))
    
    # --- Visualization 2: Grid of Subplots ---
    num_datasets = len(datasets_with_multiple_models)
    cols = min(3, num_datasets)
    rows = (num_datasets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    
    # Handle single subplot case
    if num_datasets == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = axes  # Already 1D array
    elif rows > 1:
        axes = axes.flatten()
    
    for idx, (dataset_name, models_data) in enumerate(sorted(datasets_with_multiple_models.items())):
        ax = axes[idx] if num_datasets > 1 else axes[0]
        
        # Sort by rate (descending)
        models_data_sorted = sorted(models_data, key=lambda x: x['rate'], reverse=True)
        models = [d['model'] for d in models_data_sorted]
        rates = [d['rate'] for d in models_data_sorted]
        colors = [model_colors.get(model, '#95a5a6') for model in models]
        
        bars = ax.bar(models, rates, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hallucination Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name.upper()} Dataset', fontsize=13, fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, max(rates) * 1.15 if rates else 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_datasets, len(axes) if isinstance(axes, list) or hasattr(axes, '__len__') else 1):
        if num_datasets > 1:
            axes[idx].axis('off')
    
    plt.suptitle('Hallucination Rate Comparison: All Datasets', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_file = Path(output_dir) / "combined_rate_comparison_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined grid comparison: {output_file}")
    plt.close()
    plot_files.append(str(output_file))
    
    return plot_files


def main():
    print("\n" + "="*80)
    print("BASELINE HALLUCINATION COMPARISON")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze baselines
    print(f"\nAnalyzing {len(BASELINE_DIRS)} baseline(s)...\n")
    results_by_dataset = analyze_baselines(BASELINE_DIRS)
    
    if not results_by_dataset:
        print("\n✗ No valid baselines found. Check paths in BASELINE_DIRS.")
        return
    
    # Flatten results for traditional plots (if needed)
    results_flat = {}
    for dataset, models_list in results_by_dataset.items():
        if len(models_list) == 1:
            # Single model per dataset - use dataset name only
            results_flat[dataset] = models_list[0]
        else:
            # Multiple models per dataset - use dataset_model format
            for model_data in models_list:
                key = f"{dataset} ({model_data['model']})"
                results_flat[key] = model_data
    
    # Generate traditional plots (all baselines together)
    if results_flat:
        print(f"\nGenerating traditional visualizations...\n")
        plot_bar_chart(results_flat, OUTPUT_DIR)
        plot_rate_comparison(results_flat, OUTPUT_DIR)
        plot_pie_charts(results_flat, OUTPUT_DIR)
    
    # Generate consistent color mapping for models
    model_colors = get_model_color_mapping(results_by_dataset)
    
    # Generate per-dataset model comparison plots with consistent colors
    print(f"\nGenerating per-dataset model comparison plots...\n")
    plot_per_dataset_model_comparison(results_by_dataset, OUTPUT_DIR, model_colors)
    plot_per_dataset_rate_comparison(results_by_dataset, OUTPUT_DIR, model_colors)
    
    # Generate combined comparison (all datasets together)
    print(f"\nGenerating combined multi-dataset comparison...\n")
    plot_combined_dataset_comparison(results_by_dataset, OUTPUT_DIR, model_colors)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for dataset, models_list in results_by_dataset.items():
        print(f"\n{dataset.upper()}:")
        for model_data in models_list:
            api_failures_str = f" | API Failures: {model_data['api_failures']}" if model_data.get('api_failures', 0) > 0 else ""
            print(f"  Model: {model_data['model']}")
            print(f"    Valid Samples: {model_data['total']} | Hallucinations: {model_data['hallucinations']} "
                  f"| Rate: {model_data['rate']:.2f}%{api_failures_str}")
    
    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    print(f"Note: API failures (label 2) are excluded from all statistics and visualizations")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()