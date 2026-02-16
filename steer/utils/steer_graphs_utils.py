#!/usr/bin/env python3
"""
Steering Graphs Utilities: Comprehensive visualization functions for hyperparameter sweep analysis
=====================================================================================================

Provides multiple visualization types for analyzing ITI steering hyperparameter (k, alpha) combinations:
- Line plots: K-value trends with alpha overlays
- 2D scatter plots: Color-gradient representation
- Top-N combination ranking: Bar charts
- Box plots: Distribution analysis by k and alpha
- Heatmaps: Overall grid visualization

All functions accept Stage 1/2 results DataFrames and generate high-quality PNG outputs.
"""

import os
import logging
import pickle
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def setup_logger(name: str = 'steer_graphs') -> logging.Logger:
    """Setup logger for this module"""
    try:
        from logger import consolidated_logger
        return consolidated_logger
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)


logger = setup_logger()


# ================================================================
# Utility: Load Baseline Rate from Baseline Directory
# ================================================================

def load_baseline_rate(baseline_dir: str) -> float:
    """
    Load baseline hallucination rate from baseline directory.
    
    Reads baseline_evaluation.pkl from baseline_dir and calculates the proper
    hallucination rate (counting only score=1, excluding API failures score=2).
    
    Args:
        baseline_dir: Path to baseline results directory
        
    Returns:
        Baseline hallucination rate as a decimal (0-1)
        
    Raises:
        FileNotFoundError: If baseline_evaluation.pkl not found
        ValueError: If baseline evaluation data is empty or invalid
    """
    baseline_eval_path = os.path.join(baseline_dir, 'baseline_evaluation.pkl')
    
    if not os.path.exists(baseline_eval_path):
        logger.error(f"baseline_evaluation.pkl not found in {baseline_dir}")
        raise FileNotFoundError(f"Baseline evaluation file not found at {baseline_eval_path}")
    
    try:
        with open(baseline_eval_path, 'rb') as f:
            baseline_evaluation = pickle.load(f)
        
        # Calculate hallucination rate by counting samples where score == 1
        # baseline_evaluation is Dict[idx: hallucination_score] where:
        #   0 = CORRECT (no hallucination)
        #   1 = HALLUCINATED (incorrect)
        #   2 = API FAILURE (excluded from calculation)
        halluc_values = list(baseline_evaluation.values())
        
        if not halluc_values:
            logger.error("Baseline evaluation dict is empty - cannot calculate baseline rate")
            raise ValueError("Baseline evaluation is empty")
        
        # Count hallucinations (score == 1) and exclude API failures (score == 2)
        halluc_count = sum(1 for score in halluc_values if score == 1)
        total_valid = sum(1 for score in halluc_values if score != 2)
        api_failures = sum(1 for score in halluc_values if score == 2)
        
        # Calculate rate
        if total_valid <= 0:
            logger.error(f"No valid samples in baseline evaluation (only {api_failures} API failures)")
            raise ValueError("No valid samples in baseline evaluation")
        
        baseline_rate = halluc_count / total_valid
        
        logger.info(f"✓ Loaded baseline rate: {baseline_rate*100:.2f}% from {baseline_eval_path}")
        logger.info(f"  - Total samples: {len(halluc_values)}")
        logger.info(f"  - Hallucinated: {halluc_count}")
        logger.info(f"  - Correct: {total_valid - halluc_count}")
        logger.info(f"  - API failures: {api_failures}")
        
        return baseline_rate
        
    except Exception as e:
        logger.error(f"Error loading baseline rate from {baseline_eval_path}: {e}")
        raise


# ================================================================
# 1. LINE PLOT: K-Value Trends with Alpha Overlays
# ================================================================

def generate_line_plot_k_trends(results_df: pd.DataFrame, output_dir: str, 
                                baseline_rate: float = 35.0, stage_name: str = "Stage 1"):
    """
    Generate line plot showing K-value trends with separate lines for each alpha value.
    
    Visualizes how performance changes across k values for each alpha, making it easy to
    identify which alpha values are most robust and which k values perform best overall.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        baseline_rate: Baseline hallucination rate (%)
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for line plot")
        return None
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get sorted alpha values and color map
    alpha_values = sorted(results_df['alpha'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(alpha_values)))
    
    # Plot a line for each alpha value
    for idx, alpha in enumerate(alpha_values):
        alpha_data = results_df[results_df['alpha'] == alpha].sort_values('k')
        
        # Calculate absolute improvement from baseline
        alpha_data_sorted = alpha_data.copy()
        alpha_data_sorted['absolute_improvement'] = (alpha_data_sorted['relative_reduction'] / 100.0) * baseline_rate
        
        ax.plot(alpha_data_sorted['k'], alpha_data_sorted['absolute_improvement'], 
               marker='o', label=f'α={alpha:.1f}', linewidth=2, markersize=6, 
               color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Reduction (percentage points)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Hallucination Reduction Trends Across K Values', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_LINE_PLOT_K_TRENDS.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved line plot (K trends): {output_path}")
    logger.info(f"  - Alpha values: {len(alpha_values)}")
    logger.info(f"  - K range: {results_df['k'].min()}-{results_df['k'].max()}")
    
    return output_path


# ================================================================
# 2. 2D SCATTER PLOT with Color Gradient
# ================================================================

def generate_2d_scatter_plot(results_df: pd.DataFrame, output_dir: str, 
                             baseline_rate: float = 35.0, stage_name: str = "Stage 1"):
    """
    Generate 2D scatter plot with k on x-axis, alpha on y-axis, and color/size showing improvement.
    
    Each point represents one (k, alpha) combination. Point color and size indicate the 
    absolute improvement in hallucination reduction. Makes it easy to spot optimal regions
    and see trade-offs between parameters.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        baseline_rate: Baseline hallucination rate (%)
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for scatter plot")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate absolute improvement
    scatter_df = results_df.copy()
    scatter_df['absolute_improvement'] = (scatter_df['relative_reduction'] / 100.0) * baseline_rate
    
    # Create scatter plot
    scatter = ax.scatter(scatter_df['k'], scatter_df['alpha'], 
                        c=scatter_df['absolute_improvement'], 
                        s=scatter_df['absolute_improvement'] * 50 + 50,  # Size proportional to improvement
                        cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Steering Strength (α)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Hyperparameter Space - Color/Size = Improvement', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Reduction (percentage points)', fontsize=10, fontweight='bold')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_SCATTER_2D.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved 2D scatter plot: {output_path}")
    logger.info(f"  - Total points: {len(scatter_df)}")
    logger.info(f"  - K range: {scatter_df['k'].min()}-{scatter_df['k'].max()}")
    logger.info(f"  - Alpha range: {scatter_df['alpha'].min():.2f}-{scatter_df['alpha'].max():.2f}")
    
    return output_path


# ================================================================
# 3. TOP-N COMBINATIONS Bar Chart
# ================================================================

def generate_top_n_combinations(results_df: pd.DataFrame, output_dir: str, 
                               baseline_rate: float = 35.0, top_n: int = 15, 
                               stage_name: str = "Stage 1"):
    """
    Generate bar chart showing top-N best (k, alpha) combinations ranked by improvement.
    
    Each bar represents one combination, labeled with (k, α) pair. Clear ranking shows
    which combinations are truly optimal and performance gaps between them.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        baseline_rate: Baseline hallucination rate (%)
        top_n: Number of top combinations to show (default 15)
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for top-N plot")
        return None
    
    # Get top N
    top_results = results_df.nlargest(top_n, 'relative_reduction').copy()
    top_results['absolute_improvement'] = (top_results['relative_reduction'] / 100.0) * baseline_rate
    top_results = top_results.sort_values('absolute_improvement', ascending=True)
    
    # Create labels
    top_results['label'] = top_results.apply(
        lambda row: f"k={int(row['k'])}, α={row['alpha']:.1f}", axis=1
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['green' if imp > 0 else 'red' for imp in top_results['absolute_improvement']]
    bars = ax.barh(range(len(top_results)), top_results['absolute_improvement'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels(top_results['label'], fontsize=9)
    ax.set_xlabel('Absolute Reduction (percentage points)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Top {top_n} Best (K, Alpha) Combinations', 
                fontsize=12, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Calculate x-axis limits to add space for labels
    x_min = min(top_results['absolute_improvement']) * 1.2
    x_max = max(top_results['absolute_improvement']) * 1.15
    ax.set_xlim(x_min, x_max)
    
    # Add value labels on bars with better positioning
    for i, (idx, row) in enumerate(top_results.iterrows()):
        value = row['absolute_improvement']
        relative = row['relative_reduction']
        # Position text to the right of the bar with good spacing
        # Note: value is in percentage points (e.g., 2.5 means 2.5 pp reduction)
        ax.text(value, i, f'   {value:.2f}pp, Rel: {relative:.1f}%', 
               va='center', ha='left', fontsize=7, fontweight='bold')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_TOP{top_n}_COMBINATIONS.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved top-N combinations plot: {output_path}")
    logger.info(f"  - Showing top {top_n} of {len(results_df)} combinations")
    logger.info(f"  - Best: k={int(top_results.iloc[-1]['k'])}, α={top_results.iloc[-1]['alpha']:.1f} ({top_results.iloc[-1]['absolute_improvement']:.2f}% improvement)")
    
    return output_path


# ================================================================
# 4. BOX PLOT: Distribution by K Value
# ================================================================

def generate_boxplot_by_k(results_df: pd.DataFrame, output_dir: str, 
                         baseline_rate: float = 35.0, stage_name: str = "Stage 1"):
    """
    Generate box plot showing distribution of performance across alpha values for each k.
    
    Each box shows min, Q1, median, Q3, and max performance for a given k value across
    all alpha values. Helps identify which k values are most robust and consistent.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        baseline_rate: Baseline hallucination rate (%)
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for boxplot by k")
        return None
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data: group by k, calculate absolute improvements for each alpha
    k_values = sorted(results_df['k'].unique())
    boxplot_data = []
    k_labels = []
    
    for k in k_values:
        k_subset = results_df[results_df['k'] == k].copy()
        improvements = (k_subset['relative_reduction'] / 100.0) * baseline_rate
        boxplot_data.append(improvements.values)
        k_labels.append(str(int(k)))
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, labels=k_labels, patch_artist=True, 
                    widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('green')
        mean.set_markeredgecolor('darkgreen')
        mean.set_markersize(6)
    
    ax.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Reduction (percentage points)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Performance Distribution Across Alpha Values (by K)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.7, label='25%-75% range'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', 
                  markeredgecolor='darkgreen', markersize=6, label='Mean')
    ]    
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_BOXPLOT_BY_K.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved boxplot by K: {output_path}")
    logger.info(f"  - K values: {len(k_values)}")
    logger.info(f"  - Alpha values per K: {len(results_df) // len(k_values)}")
    
    return output_path


# ================================================================
# 5. BOX PLOT: Distribution by Alpha Value
# ================================================================

def generate_boxplot_by_alpha(results_df: pd.DataFrame, output_dir: str, 
                             baseline_rate: float = 35.0, stage_name: str = "Stage 1"):
    """
    Generate box plot showing distribution of performance across k values for each alpha.
    
    Each box shows min, Q1, median, Q3, and max performance for a given alpha value across
    all k values. Helps identify which alpha values are most robust and consistent.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        baseline_rate: Baseline hallucination rate (%)
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for boxplot by alpha")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data: group by alpha, calculate absolute improvements for each k
    alpha_values = sorted(results_df['alpha'].unique())
    boxplot_data = []
    alpha_labels = []
    
    for alpha in alpha_values:
        alpha_subset = results_df[results_df['alpha'] == alpha].copy()
        improvements = (alpha_subset['relative_reduction'] / 100.0) * baseline_rate
        boxplot_data.append(improvements.values)
        alpha_labels.append(f'{alpha:.1f}')
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, labels=alpha_labels, patch_artist=True, 
                    widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('blue')
        median.set_linewidth(2)
    
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('green')
        mean.set_markeredgecolor('darkgreen')
        mean.set_markersize(6)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Reduction (percentage points)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Performance Distribution Across K Values (by Alpha)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', alpha=0.7, label='25%-75% range'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', 
                  markeredgecolor='darkgreen', markersize=6, label='Mean')
    ]    
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_BOXPLOT_BY_ALPHA.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved boxplot by Alpha: {output_path}")
    logger.info(f"  - Alpha values: {len(alpha_values)}")
    logger.info(f"  - K values per Alpha: {len(results_df) // len(alpha_values)}")
    
    return output_path


# ================================================================
# 6. HEATMAP: Overall Grid Visualization - ABSOLUTE VERSION
# ================================================================

def generate_heatmap(results_df: pd.DataFrame, output_path: str, 
                     baseline_rate: float = 35.0, stage_name: str = "Combined"):
    """
    Generate heatmap visualization showing absolute reduction in hallucinations.
    
    Maps (k, alpha) combinations to absolute improvement in percentage points
    from baseline hallucination rate.
    
    Args:
        results_df: DataFrame with k, alpha, absolute_reduction
        output_path: Output PNG path
        baseline_rate: Baseline hallucination rate (%) for context
        stage_name: Name for title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for heatmap")
        return None
    
    # Calculate absolute improvements
    heatmap_df = results_df.copy()
    heatmap_df['absolute_improvement'] = (heatmap_df['absolute_reduction']) * 100
    
    # Pivot to create matrix
    pivot_df = heatmap_df.pivot_table(index='k', columns='alpha', 
                                      values='absolute_improvement', aggfunc='max')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f'{a:.1f}' for a in pivot_df.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=8)
    
    ax.set_xlabel('Alpha (Steering Strength)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name} Grid Search: Absolute Hallucination Reduction (percentage points)', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Reduction (pp)', fontsize=10, fontweight='bold')
    
    # Add value annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color='white' if val > 5 else 'black', fontsize=7)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved heatmap (absolute): {output_path}")
    
    return output_path


# ================================================================
# 6b. HEATMAP: Overall Grid Visualization - RELATIVE VERSION
# ================================================================

def generate_heatmap_relative(results_df: pd.DataFrame, output_path: str, 
                              stage_name: str = "Combined"):
    """
    Generate heatmap visualization showing relative reduction in hallucinations.
    
    Maps (k, alpha) combinations to relative improvement (%) from baseline,
    independent of actual baseline rate.
    
    Args:
        results_df: DataFrame with k, alpha, relative_reduction
        output_path: Output PNG path
        stage_name: Name for title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative heatmap")
        return None
    
    # Pivot to create matrix using relative_reduction directly
    pivot_df = results_df.pivot_table(index='k', columns='alpha', 
                                      values='relative_reduction', aggfunc='max')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f'{a:.1f}' for a in pivot_df.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=8)
    
    ax.set_xlabel('Alpha (Steering Strength)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name} Grid Search: Relative Hallucination Reduction (%)', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Reduction (%)', fontsize=10, fontweight='bold')
    
    # Add value annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color='white' if val > 15 else 'black', fontsize=7)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved heatmap (relative): {output_path}")
    
    return output_path


def generate_heatmap_hallucination_count(results_df: pd.DataFrame, output_path: str, 
                                         baseline_rate: float, stage_name: str = "Combined"):
    """
    Generate heatmap visualization showing resulting hallucination rates after steering.
    
    Maps (k, alpha) combinations to final hallucination rate (%) after applying steering.
    Lower values are better (green). Useful for understanding absolute performance.
    
    Args:
        results_df: DataFrame with k, alpha, absolute_reduction
        output_path: Output PNG path
        baseline_rate: Baseline hallucination rate (decimal, e.g., 0.35 for 35%)
        stage_name: Name for title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for hallucination count heatmap")
        return None
    
    # Calculate final hallucination counts: baseline_rate - reduced_amount
    results_with_counts = results_df.copy()
    results_with_counts['hallucination_rate'] = (
        baseline_rate - results_df['absolute_reduction']
    ) * 100
    
    # Pivot to create matrix
    pivot_df = results_with_counts.pivot_table(index='k', columns='alpha', 
                                               values='hallucination_rate', aggfunc='min')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use inverted colormap: lower values (fewer hallucinations) = green
    im = ax.imshow(pivot_df.values, cmap='RdYlGn_r', aspect='auto', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f'{a:.1f}' for a in pivot_df.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=8)
    
    ax.set_xlabel('Alpha (Steering Strength)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name} Grid Search: Final Hallucination Rate (%)', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Final Hallucination Rate (%)', fontsize=10, fontweight='bold')
    
    # Add value annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color='white' if val > (baseline_rate * 100 / 2) else 'black', fontsize=7)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved hallucination count heatmap: {output_path}")
    
    return output_path


# ================================================================
# 7. COMPARATIVE BARS: Average by K and Alpha
# ================================================================

def generate_comparative_bars(results_df: pd.DataFrame, output_dir: str, 
                             baseline_rate: float = 35.0, stage_name: str = "Stage 1"):
    """
    Generate comparative bar graphs averaging results by k and alpha values.
    
    Shows absolute improvement in hallucination reduction (% points) vs baseline,
    averaged across all other parameters.
    
    Args:
        results_df: DataFrame with k, alpha, relative_reduction
        output_dir: Output directory for PNG files
        baseline_rate: Baseline hallucination rate (%)
        stage_name: Name for title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for comparative bars")
        return None
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============ Graph 1: Average improvement by K value ============
    k_groups = results_df.groupby('k')['relative_reduction'].mean().reset_index()
    k_groups.columns = ['k', 'avg_relative_reduction']
    
    # Calculate absolute improvement from baseline
    k_groups['absolute_improvement'] = (k_groups['avg_relative_reduction'] / 100.0) * baseline_rate
    
    colors_k = ['green' if imp > 0 else 'red' for imp in k_groups['absolute_improvement']]
    
    bars1 = ax1.bar(range(len(k_groups)), k_groups['absolute_improvement'], 
                     color=colors_k, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Absolute Reduction (percentage points)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{stage_name}: Average Reduction vs Baseline (by K)', 
                  fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(range(len(k_groups)))
    ax1.set_xticklabels(k_groups['k'].astype(int), rotation=45, fontsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='y', labelsize=8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(k_groups.iterrows()):
        height = row['absolute_improvement']
        ax1.text(i, height, f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=7, fontweight='bold')
    
    # ============ Graph 2: Average improvement by Alpha value ============
    alpha_groups = results_df.groupby('alpha')['relative_reduction'].mean().reset_index()
    alpha_groups.columns = ['alpha', 'avg_relative_reduction']
    
    # Calculate absolute improvement from baseline
    alpha_groups['absolute_improvement'] = (alpha_groups['avg_relative_reduction'] / 100.0) * baseline_rate
    
    colors_alpha = ['green' if imp > 0 else 'red' for imp in alpha_groups['absolute_improvement']]
    
    bars2 = ax2.bar(range(len(alpha_groups)), alpha_groups['absolute_improvement'], 
                     color=colors_alpha, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Steering Strength (α)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Absolute Reduction (percentage points)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{stage_name}: Average Reduction vs Baseline (by Alpha)', 
                  fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(range(len(alpha_groups)))
    ax2.set_xticklabels([f'{a:.1f}' for a in alpha_groups['alpha']], rotation=45, fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='y', labelsize=8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(alpha_groups.iterrows()):
        height = row['absolute_improvement']
        ax2.text(i, height, f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=7, fontweight='bold')
    
    # Save figure
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_COMPARATIVE_BARS.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved comparative bar graphs: {output_path}")
    logger.info(f"  - K values averaged: {len(k_groups)} groups")
    logger.info(f"  - Alpha values averaged: {len(alpha_groups)} groups")
    
    return output_path


# ================================================================
# Master Function: Generate All Visualizations
# ================================================================

def generate_all_stage_visualizations(results_df: pd.DataFrame, output_dir: str,
                                     baseline_dir: Optional[str] = None,
                                     baseline_rate: Optional[float] = None, 
                                     stage_name: str = "Stage 1",
                                     top_n: int = 15) -> dict:
    """
    Generate all visualization types for a stage's results.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG files
        baseline_dir: Path to baseline directory (used to load baseline_evaluation.pkl)
                      If provided, baseline_rate will be loaded from here
        baseline_rate: Baseline hallucination rate (%). 
                      If None and baseline_dir provided, loaded from baseline_dir.
                      If both None, defaults to 35.0%
        stage_name: Name for plots (e.g., "Stage 1", "Stage 2", "Combined")
        top_n: Number of top combinations to display
        
    Returns:
        Dictionary mapping graph name -> output path
    """
    if len(results_df) == 0:
        logger.warning(f"No results to visualize for {stage_name}")
        return {}
    
    # Load baseline rate from directory if not provided
    if baseline_rate is None:
        if baseline_dir is not None and os.path.exists(baseline_dir):
            baseline_rate = load_baseline_rate(baseline_dir)
        else:
            logger.warning("No baseline_dir provided and baseline_rate is None, using default 35.0%")
            baseline_rate = 35.0
    
    logger.info(f"\n[VISUALIZATIONS] Generating all plots for {stage_name}...")
    logger.info(f"  - Baseline rate: {baseline_rate:.2f}%")
    
    graphs = {}
    
    # Generate all visualizations
    graphs['heatmap'] = generate_heatmap(results_df, 
                                         os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_HEATMAP.png"),
                                         stage_name)
    
    graphs['comparative_bars'] = generate_comparative_bars(results_df, output_dir, 
                                                           baseline_rate, stage_name)
    
    graphs['line_plot_k_trends'] = generate_line_plot_k_trends(results_df, output_dir, 
                                                               baseline_rate, stage_name)
    
    graphs['scatter_2d'] = generate_2d_scatter_plot(results_df, output_dir, 
                                                    baseline_rate, stage_name)
    
    graphs['top_n_combinations'] = generate_top_n_combinations(results_df, output_dir, 
                                                               baseline_rate, top_n, stage_name)
    
    graphs['boxplot_by_k'] = generate_boxplot_by_k(results_df, output_dir, 
                                                   baseline_rate, stage_name)
    
    graphs['boxplot_by_alpha'] = generate_boxplot_by_alpha(results_df, output_dir, 
                                                           baseline_rate, stage_name)
    
    logger.info(f"✓ All visualizations complete for {stage_name}: {len(graphs)} graphs generated")
    
    return graphs


# ================================================================
# RELATIVE REDUCTION VERSIONS: Same graphs but showing relative % reduction
# ================================================================

def generate_line_plot_relative(results_df: pd.DataFrame, output_dir: str, 
                                stage_name: str = "Stage 1"):
    """
    Generate line plot showing K-value trends with separate lines for each alpha value.
    Uses RELATIVE REDUCTION (%) instead of absolute reduction.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative line plot")
        return None
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get sorted alpha values and color map
    alpha_values = sorted(results_df['alpha'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(alpha_values)))
    
    # Plot a line for each alpha value
    for idx, alpha in enumerate(alpha_values):
        alpha_data = results_df[results_df['alpha'] == alpha].sort_values('k')
        
        ax.plot(alpha_data['k'], alpha_data['relative_reduction'], 
               marker='o', label=f'α={alpha:.1f}', linewidth=2, markersize=6, 
               color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Reduction (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Relative Hallucination Reduction Trends Across K Values', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_LINE_PLOT_RELATIVE_K_TRENDS.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved relative line plot (K trends): {output_path}")
    logger.info(f"  - Alpha values: {len(alpha_values)}")
    logger.info(f"  - K range: {results_df['k'].min()}-{results_df['k'].max()}")
    
    return output_path


def generate_2d_scatter_plot_relative(results_df: pd.DataFrame, output_dir: str, 
                                      stage_name: str = "Stage 1"):
    """
    Generate 2D scatter plot with k on x-axis, alpha on y-axis, and color/size showing relative reduction.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative scatter plot")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create scatter plot using relative_reduction directly
    scatter = ax.scatter(results_df['k'], results_df['alpha'], 
                        c=results_df['relative_reduction'], 
                        s=results_df['relative_reduction'] * 3 + 50,  # Size proportional to reduction
                        cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Steering Strength (α)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Hyperparameter Space - Color/Size = Relative Reduction', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Relative Reduction (%)', fontsize=10, fontweight='bold')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_SCATTER_2D_RELATIVE.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved relative 2D scatter plot: {output_path}")
    logger.info(f"  - Total points: {len(results_df)}")
    logger.info(f"  - K range: {results_df['k'].min()}-{results_df['k'].max()}")
    logger.info(f"  - Alpha range: {results_df['alpha'].min():.2f}-{results_df['alpha'].max():.2f}")
    
    return output_path


def generate_top_n_combinations_relative(results_df: pd.DataFrame, output_dir: str, 
                                         top_n: int = 15, stage_name: str = "Stage 1"):
    """
    Generate bar chart showing top-N best (k, alpha) combinations ranked by relative reduction.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        top_n: Number of top combinations to show (default 15)
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative top-N plot")
        return None
    
    # Get top N
    top_results = results_df.nlargest(top_n, 'relative_reduction').copy()
    top_results = top_results.sort_values('relative_reduction', ascending=True)
    
    # Create labels
    top_results['label'] = top_results.apply(
        lambda row: f"k={int(row['k'])}, α={row['alpha']:.1f}", axis=1
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['green' if imp > 0 else 'red' for imp in top_results['relative_reduction']]
    bars = ax.barh(range(len(top_results)), top_results['relative_reduction'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels(top_results['label'], fontsize=9)
    ax.set_xlabel('Relative Reduction (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Top {top_n} Best (K, Alpha) Combinations (Relative)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Calculate x-axis limits to add space for labels
    x_min = min(top_results['relative_reduction']) * 1.2
    x_max = max(top_results['relative_reduction']) * 1.15
    ax.set_xlim(x_min, x_max)
    
    # Add value labels on bars with better positioning
    for i, (idx, row) in enumerate(top_results.iterrows()):
        value = row['relative_reduction']
        # Position text to the right of the bar with good spacing
        ax.text(value, i, f'   {value:.1f}%', 
               va='center', ha='left', fontsize=7, fontweight='bold')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_TOP{top_n}_COMBINATIONS_RELATIVE.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved relative top-N combinations plot: {output_path}")
    logger.info(f"  - Showing top {top_n} of {len(results_df)} combinations")
    logger.info(f"  - Best: k={int(top_results.iloc[-1]['k'])}, α={top_results.iloc[-1]['alpha']:.1f} ({top_results.iloc[-1]['relative_reduction']:.1f}% reduction)")
    
    return output_path


def generate_boxplot_by_k_relative(results_df: pd.DataFrame, output_dir: str, 
                                   stage_name: str = "Stage 1"):
    """
    Generate box plot showing distribution of relative reduction across alpha values for each k.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative boxplot by k")
        return None
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data: group by k
    k_values = sorted(results_df['k'].unique())
    boxplot_data = []
    k_labels = []
    
    for k in k_values:
        k_subset = results_df[results_df['k'] == k].copy()
        boxplot_data.append(k_subset['relative_reduction'].values)
        k_labels.append(str(int(k)))
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, labels=k_labels, patch_artist=True, 
                    widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('green')
        mean.set_markeredgecolor('darkgreen')
        mean.set_markersize(6)
    
    ax.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Reduction (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Relative Performance Distribution Across Alpha Values (by K)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.7, label='25%-75% range'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', 
                  markeredgecolor='darkgreen', markersize=6, label='Mean')
    ]    
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_BOXPLOT_BY_K_RELATIVE.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved relative boxplot by K: {output_path}")
    logger.info(f"  - K values: {len(k_values)}")
    logger.info(f"  - Alpha values per K: {len(results_df) // len(k_values)}")
    
    return output_path


def generate_boxplot_by_alpha_relative(results_df: pd.DataFrame, output_dir: str, 
                                       stage_name: str = "Stage 1"):
    """
    Generate box plot showing distribution of relative reduction across k values for each alpha.
    
    Args:
        results_df: DataFrame with columns ['k', 'alpha', 'relative_reduction']
        output_dir: Output directory for PNG
        stage_name: Name for plot title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative boxplot by alpha")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data: group by alpha
    alpha_values = sorted(results_df['alpha'].unique())
    boxplot_data = []
    alpha_labels = []
    
    for alpha in alpha_values:
        alpha_subset = results_df[results_df['alpha'] == alpha].copy()
        boxplot_data.append(alpha_subset['relative_reduction'].values)
        alpha_labels.append(f'{alpha:.1f}')
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, labels=alpha_labels, patch_artist=True, 
                    widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('blue')
        median.set_linewidth(2)
    
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('green')
        mean.set_markeredgecolor('darkgreen')
        mean.set_markersize(6)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Reduction (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{stage_name}: Relative Performance Distribution Across K Values (by Alpha)', 
                fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', alpha=0.7, label='25%-75% range'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', 
                  markeredgecolor='darkgreen', markersize=6, label='Mean')
    ]    
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_BOXPLOT_BY_ALPHA_RELATIVE.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved relative boxplot by Alpha: {output_path}")
    logger.info(f"  - Alpha values: {len(alpha_values)}")
    logger.info(f"  - K values per Alpha: {len(results_df) // len(alpha_values)}")
    
    return output_path


def generate_comparative_bars_relative(results_df: pd.DataFrame, output_dir: str, 
                                       stage_name: str = "Stage 1"):
    """
    Generate comparative bar graphs averaging results by k and alpha values (relative reduction).
    
    Args:
        results_df: DataFrame with k, alpha, relative_reduction
        output_dir: Output directory for PNG files
        stage_name: Name for title
        
    Returns:
        Path to saved PNG file
    """
    if len(results_df) == 0:
        logger.warning("No results for relative comparative bars")
        return None
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============ Graph 1: Average relative reduction by K value ============
    k_groups = results_df.groupby('k')['relative_reduction'].mean().reset_index()
    k_groups.columns = ['k', 'avg_relative_reduction']
    
    colors_k = ['green' if imp > 0 else 'red' for imp in k_groups['avg_relative_reduction']]
    
    bars1 = ax1.bar(range(len(k_groups)), k_groups['avg_relative_reduction'], 
                     color=colors_k, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Top-K Heads', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Relative Reduction (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{stage_name}: Average Relative Reduction (by K)', 
                  fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(range(len(k_groups)))
    ax1.set_xticklabels(k_groups['k'].astype(int), rotation=45, fontsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='y', labelsize=8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(k_groups.iterrows()):
        height = row['avg_relative_reduction']
        ax1.text(i, height, f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=7, fontweight='bold')
    
    # ============ Graph 2: Average relative reduction by Alpha value ============
    alpha_groups = results_df.groupby('alpha')['relative_reduction'].mean().reset_index()
    alpha_groups.columns = ['alpha', 'avg_relative_reduction']
    
    colors_alpha = ['green' if imp > 0 else 'red' for imp in alpha_groups['avg_relative_reduction']]
    
    bars2 = ax2.bar(range(len(alpha_groups)), alpha_groups['avg_relative_reduction'], 
                     color=colors_alpha, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Steering Strength (α)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Reduction (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{stage_name}: Average Relative Reduction (by Alpha)', 
                  fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(range(len(alpha_groups)))
    ax2.set_xticklabels([f'{a:.1f}' for a in alpha_groups['alpha']], rotation=45, fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='y', labelsize=8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(alpha_groups.iterrows()):
        height = row['avg_relative_reduction']
        ax2.text(i, height, f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=7, fontweight='bold')
    
    # Save figure
    output_path = os.path.join(output_dir, f"{stage_name.replace(' ', '_').upper()}_COMPARATIVE_BARS_RELATIVE.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved relative comparative bar graphs: {output_path}")
    logger.info(f"  - K values averaged: {len(k_groups)} groups")
    logger.info(f"  - Alpha values averaged: {len(alpha_groups)} groups")
    
    return output_path
