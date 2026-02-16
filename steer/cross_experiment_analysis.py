#!/usr/bin/env python
"""
Cross-Experiment Analysis Script

Analyzes steering experiments across multiple output directories to find optimal
configurations that reduce hallucinations while maintaining general capabilities.

Key features:
- Identifies valid samples (no API failures across all experiments in an output directory)
- Calculates hallucination rates based only on valid samples
- Compares primary (hallucination reduction) vs secondary (general abilities) directories
- Generates comprehensive reports and visualizations

Usage:
    python -m pipeline.steer.ITI.cross_experiment_analysis --primary-dir ./data/ITI/steering_experiment_gemma_3_percent/round1/evalonnqswap --secondary-dir-1 ./data/ITI/steering_experiment_gemma_3_percent/round1/evalonmmlu --secondary-dir-2 ./data/ITI/steering_experiment_gemma_3_percent/round1/evalonhellaswag --output-dir ./data/ITI/steering_experiment_gemma_3_percent/round1/cross_analysis --secondary-threshold 3.0 --secondary-threshold-2 3.0
    
Scoring methodology:
    - Hallucination scores: 0 (no hallucination), 1 (hallucination), 2 (API failure)
    - Valid sample: No API failures (score != 2) across ALL experiments in the output directory
    - Absolute reduction: baseline_rate - steered_rate (in percentage points)
    - Relative reduction: (baseline_rate - steered_rate) / baseline_rate * 100
    
Primary vs Secondary comparison:
    - Primary directory: Maximize absolute hallucination reduction (higher is better, measured in pp)
    - Secondary directory: Check that hallucination rate doesn't increase by more than
      threshold percentage points. A config PASSES if:
      secondary_steered_rate <= secondary_baseline_rate + threshold/100
    - Final recommendation: Best absolute reduction among configs that pass secondary check
"""

import os
import sys
import argparse
import json
import pickle
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class ExperimentConfig:
    """Configuration extracted from an experiment directory."""
    experiment_dir: str
    k: int  # top_k from ITI config
    alpha_values: List[float]
    iti_config_path: str
    baseline_dir: str


@dataclass
class ExperimentResult:
    """Results from a single (k, alpha) experiment."""
    k: int
    alpha: float
    experiment_dir: str
    results_by_idx: Dict[int, Dict]  # {sample_idx: {steered_hallucination, baseline_hallucination, ...}}
    

@dataclass
class DirectoryAnalysis:
    """Complete analysis results for one output directory."""
    output_dir: str
    baseline_dir: str
    all_experiments: List[ExperimentResult]
    valid_sample_indices: Set[int]
    baseline_hallucination_rate: float
    baseline_hallucination_count: int
    total_valid_samples: int
    results_by_config: Dict[Tuple[int, float], Dict]  # {(k, alpha): {halluc_rate, reduction, etc.}}


# ================================================================
# DISCOVERY FUNCTIONS
# ================================================================

def discover_experiments_in_directory(output_dir: str) -> List[str]:
    """
    Discover all experiment run directories within an output directory.
    
    Looks for directories with pattern: STEERING_*
    Each should contain a 'results' subdirectory with ALL_RESULTS_CONSOLIDATED.json
    
    Returns:
        List of experiment directory paths
    """
    experiment_dirs = []
    
    if not os.path.exists(output_dir):
        print(f"[ERROR] Output directory does not exist: {output_dir}")
        return experiment_dirs
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        
        # Look for STEERING_* directories
        if os.path.isdir(item_path) and item.startswith("STEERING_"):
            # Check for consolidated results file
            consolidated_path = os.path.join(item_path, "results", "ALL_RESULTS_CONSOLIDATED.json")
            if os.path.exists(consolidated_path):
                experiment_dirs.append(item_path)
            else:
                print(f"[WARNING] No consolidated results in: {item_path}")
    
    print(f"[INFO] Found {len(experiment_dirs)} experiment(s) in {output_dir}")
    return sorted(experiment_dirs)


def extract_k_from_config_path(config_path: str) -> int:
    """
    Extract top_k value from ITI config path.
    
    Patterns supported:
    - iti_intervention_config_top10.pkl -> 10
    - iti_intervention_config_top100.pkl -> 100
    - iti_intervention_config_top1000.pkl -> 1000
    """
    # Pattern: top followed by digits
    match = re.search(r'top(\d+)', config_path)
    if match:
        return int(match.group(1))
    
    # Fallback: try to load the pickle and extract
    print(f"[WARNING] Could not extract k from path: {config_path}")
    return -1


def load_experiment_results(experiment_dir: str) -> Tuple[ExperimentConfig, List[ExperimentResult]]:
    """
    Load all results from a single experiment directory.
    
    Reads steering_config.json from config/ subdirectory to get:
    - baseline_dir
    - iti_config_path
    - steering_strengths (alpha values)
    
    Prefers lightweight ALL_RESULTS_ANALYSIS.json for efficiency.
    Falls back to ALL_RESULTS_CONSOLIDATED.json for backwards compatibility.
    
    Returns:
        Tuple of (config, list of ExperimentResult for each alpha)
    """
    # Try lightweight analysis file first (no text data - faster to load)
    analysis_path = os.path.join(experiment_dir, "results", "ALL_RESULTS_ANALYSIS.json")
    consolidated_path = os.path.join(experiment_dir, "results", "ALL_RESULTS_CONSOLIDATED.json")
    
    # Prefer analysis file if available
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            consolidated = json.load(f)
    # Fallback to consolidated for backwards compatibility
    elif os.path.exists(consolidated_path):
        with open(consolidated_path, 'r') as f:
            consolidated = json.load(f)
    else:
        raise FileNotFoundError(f"No results file found in {experiment_dir}")
    
    # Load steering_config.json from config/ subdirectory
    steering_config_path = os.path.join(experiment_dir, "config", "steering_config.json")
    
    baseline_dir = ""
    iti_config_path = ""
    
    if os.path.exists(steering_config_path):
        with open(steering_config_path, 'r') as f:
            steering_config = json.load(f)
            baseline_dir = steering_config.get('baseline_dir', '')
            iti_config_path = steering_config.get('iti_config_path', '')
    else:
        # Fallback to metadata in consolidated results
        print(f"[WARNING] steering_config.json not found in {experiment_dir}, using metadata fallback")
        metadata = consolidated.get('metadata', {})
        iti_config_path = metadata.get('iti_config_path', '')
    
    # Extract k from ITI config path
    k = extract_k_from_config_path(iti_config_path)
    
    # Get alpha values from consolidated metadata
    metadata = consolidated.get('metadata', {})
    alpha_values = metadata.get('steering_strengths', [])
    
    config = ExperimentConfig(
        experiment_dir=experiment_dir,
        k=k,
        alpha_values=alpha_values,
        iti_config_path=iti_config_path,
        baseline_dir=baseline_dir
    )
    
    # Extract results for each alpha
    experiment_results = []
    results_by_strength = consolidated.get('results_by_strength', {})
    
    for alpha_str, results_list in results_by_strength.items():
        alpha = float(alpha_str)
        
        # Convert to dict by sample index
        results_by_idx = {}
        for r in results_list:
            idx = r['sample_idx']
            results_by_idx[idx] = {
                'sample_idx': idx,
                'baseline_hallucination': r['baseline_hallucination'],
                'steered_hallucination': r['steered_hallucination'],
            }
        
        experiment_results.append(ExperimentResult(
            k=k,
            alpha=alpha,
            experiment_dir=experiment_dir,
            results_by_idx=results_by_idx
        ))
    
    return config, experiment_results


# ================================================================
# VALIDATION AND FILTERING
# ================================================================

def find_valid_samples_across_experiments(
    all_experiments: List[ExperimentResult]
) -> Set[int]:
    """
    Find sample indices that have NO API failures across ALL experiments.
    
    This function implements strict validity criteria: a sample is only valid if
    it has successful hallucination evaluations (no API failures) in BOTH the baseline
    AND all steering configurations tested. This ensures fair comparison by excluding
    samples where API issues prevented proper evaluation.
    
    Validity Criteria:
    A sample index is valid if and only if:
    1. Baseline hallucination score != 2 (baseline evaluation succeeded)
    2. Steered hallucination score != 2 for EVERY (k, alpha) combination tested
    3. Sample appears in all experiment directories (consistent coverage)
    
    Args:
        all_experiments: List of ExperimentResult objects, each containing results
                        for one (k, alpha) configuration with per-sample scores
    
    Returns:
        Set[int]: Sample indices that passed validity check across all experiments
    
    Example:
        If testing 4 configs [(k=5, α=1.0), (k=5, α=2.0), (k=10, α=1.0), (k=10, α=2.0)],
        sample 42 is valid only if:
        - baseline_hallucination != 2 in all 4 directories
        - steered_hallucination != 2 in all 4 directories
        Total: 8 non-API-failure checks per sample
    
    Note:
        - Score 0 = No hallucination (correct)
        - Score 1 = Hallucination (incorrect)  
        - Score 2 = API failure (excluded from metrics)
    """
    if not all_experiments:
        return set()
    
    # Start with all indices from first experiment
    all_indices = set(all_experiments[0].results_by_idx.keys())
    
    # Find samples with API failures in any experiment
    api_failure_indices = set()
    
    for exp in all_experiments:
        for idx, result in exp.results_by_idx.items():
            # Check baseline API failure
            if result['baseline_hallucination'] == 2:
                api_failure_indices.add(idx)
            
            # Check steered API failure
            if result['steered_hallucination'] == 2:
                api_failure_indices.add(idx)
    
    valid_indices = all_indices - api_failure_indices
    
    print(f"[INFO] Total samples: {len(all_indices)}")
    print(f"[INFO] Samples with API failures (excluded): {len(api_failure_indices)}")
    print(f"[INFO] Valid samples: {len(valid_indices)}")
    
    return valid_indices


def calculate_metrics_on_valid_samples(
    all_experiments: List[ExperimentResult],
    valid_indices: Set[int]
) -> Tuple[float, int, int, Dict[Tuple[int, float], Dict]]:
    """
    Calculate hallucination metrics using only valid samples.
    
    Returns:
        Tuple of:
        - baseline_hallucination_rate (float)
        - baseline_hallucination_count (int)
        - total_valid_samples (int)
        - results_by_config: Dict mapping (k, alpha) to metrics dict
    """
    if not valid_indices or not all_experiments:
        return 0.0, 0, 0, {}
    
    total_valid = len(valid_indices)
    
    # Calculate baseline rate from first experiment (all share same baseline)
    first_exp = all_experiments[0]
    baseline_halluc_count = sum(
        1 for idx in valid_indices
        if first_exp.results_by_idx.get(idx, {}).get('baseline_hallucination') == 1
    )
    baseline_rate = baseline_halluc_count / total_valid if total_valid > 0 else 0
    
    # Calculate metrics for each (k, alpha) configuration
    results_by_config = {}
    
    for exp in all_experiments:
        # Count hallucinations on valid samples only
        steered_halluc_count = sum(
            1 for idx in valid_indices
            if exp.results_by_idx.get(idx, {}).get('steered_hallucination') == 1
        )
        
        steered_rate = steered_halluc_count / total_valid if total_valid > 0 else 0
        
        # Calculate reductions (positive = improvement/reduction)
        # absolute_reduction in percentage points (multiply by 100 to convert from decimal)
        absolute_reduction = (baseline_rate - steered_rate) * 100
        relative_reduction = (absolute_reduction / (baseline_rate * 100) * 100) if baseline_rate > 0 else 0
        
        # Calculate change from baseline (negative = improvement, positive = degradation)
        # This is the OPPOSITE of reduction - used for secondary threshold check
        change_from_baseline = steered_rate - baseline_rate
        change_from_baseline_pct = (change_from_baseline / baseline_rate * 100) if baseline_rate > 0 else 0
        
        results_by_config[(exp.k, exp.alpha)] = {
            'k': exp.k,
            'alpha': exp.alpha,
            'steered_halluc_count': steered_halluc_count,
            'steered_halluc_rate': steered_rate,
            'absolute_reduction': absolute_reduction,
            'relative_reduction': relative_reduction,
            'change_from_baseline': change_from_baseline,
            'change_from_baseline_pct': change_from_baseline_pct,
            'total_valid_samples': total_valid,
            'experiment_dir': exp.experiment_dir,
        }
    
    return baseline_rate, baseline_halluc_count, total_valid, results_by_config


# ================================================================
# DIRECTORY ANALYSIS
# ================================================================

def analyze_output_directory(output_dir: str) -> Optional[DirectoryAnalysis]:
    """
    Perform complete analysis of a single output directory containing multiple steering experiments.
    
    This is the main workhorse function that processes all experiments in a directory and
    calculates hallucination metrics based on valid samples only.
    
    Workflow:
    1. Discovery: Find all STEERING_* subdirectories in output_dir
    2. Loading: Parse config and load ALL_RESULTS_CONSOLIDATED.json from each experiment  
    3. Validation: Identify samples with no API failures across ALL experiments
    4. Baseline calculation: Compute baseline hallucination rate on valid samples
    5. Per-config metrics: Calculate steered rates, reductions, and improvements for each (k, α)
    6. Aggregation: Package results into DirectoryAnalysis object
    
    Args:
        output_dir: Path to directory containing multiple steering experiment runs
                   Expected structure:
                   output_dir/
                   ├── STEERING_k5_alpha1.0_timestamp/results/ALL_RESULTS_CONSOLIDATED.json
                   ├── STEERING_k5_alpha2.0_timestamp/results/ALL_RESULTS_CONSOLIDATED.json
                   └── ...
    
    Returns:
        DirectoryAnalysis: Complete analysis with:
            - valid_sample_indices: Samples with no API failures
            - baseline_hallucination_rate: Unsteered rate on valid samples
            - results_by_config: Per (k, α) metrics (rate, reduction, improvement)
        Returns None if no experiments found or all experiments have errors
    
    Metrics Calculated:
        - hallucination_rate: Fraction of valid samples with score=1
        - absolute_reduction: baseline_rate - steered_rate (percentage points)
        - relative_reduction: absolute_reduction / baseline_rate * 100 (percent)
    
    Example:
        >>> analysis = analyze_output_directory("./data/ITI/steering_qwen/nqswap")
        >>> print(f"Baseline: {analysis.baseline_hallucination_rate:.1%}")
        >>> best_cfg = max(analysis.results_by_config.items(), 
        ...                key=lambda x: x[1]['absolute_reduction'])
        >>> print(f"Best config: k={best_cfg[0][0]}, α={best_cfg[0][1]}")
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING OUTPUT DIRECTORY")
    print(f"{'='*80}")
    print(f"Directory: {output_dir}")
    
    # Discover experiments
    experiment_dirs = discover_experiments_in_directory(output_dir)
    
    if not experiment_dirs:
        print(f"[ERROR] No experiments found in {output_dir}")
        return None
    
    # Load all experiment results
    all_experiments = []
    baseline_dir = None
    
    for exp_dir in experiment_dirs:
        print(f"\n[INFO] Loading experiment: {os.path.basename(exp_dir)}")
        
        try:
            config, results = load_experiment_results(exp_dir)
            all_experiments.extend(results)
            
            if baseline_dir is None and config.baseline_dir:
                baseline_dir = config.baseline_dir
            
            print(f"       k={config.k}, alphas={config.alpha_values}")
            print(f"       baseline_dir={config.baseline_dir}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load {exp_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_experiments:
        print(f"[ERROR] No valid experiments loaded from {output_dir}")
        return None
    
    print(f"\n[INFO] Total (k, alpha) configurations loaded: {len(all_experiments)}")
    print(f"[INFO] Baseline directory: {baseline_dir}")
    
    # Find valid samples
    print(f"\n[INFO] Finding valid samples (no API failures across all experiments)...")
    valid_indices = find_valid_samples_across_experiments(all_experiments)
    
    # Calculate metrics
    print(f"\n[INFO] Calculating metrics on valid samples only...")
    baseline_rate, baseline_count, total_valid, results_by_config = calculate_metrics_on_valid_samples(
        all_experiments, valid_indices
    )
    
    print(f"\n[RESULT] Baseline hallucination rate: {baseline_rate*100:.2f}% ({baseline_count}/{total_valid})")
    
    return DirectoryAnalysis(
        output_dir=output_dir,
        baseline_dir=baseline_dir or "Unknown",
        all_experiments=all_experiments,
        valid_sample_indices=valid_indices,
        baseline_hallucination_rate=baseline_rate,
        baseline_hallucination_count=baseline_count,
        total_valid_samples=total_valid,
        results_by_config=results_by_config
    )


# ================================================================
# CROSS-DIRECTORY COMPARISON
# ================================================================

def compare_primary_secondary(
    primary: DirectoryAnalysis,
    secondary: DirectoryAnalysis,
    secondary_threshold_pct: float = 1.0
) -> pd.DataFrame:
    """
    Compare primary (hallucination reduction) and secondary (general abilities) analyses.
    
    Scoring methodology:
    - Primary: Higher absolute hallucination reduction is better (measured in percentage points)
    - Secondary: Steered hallucination rate should NOT increase by more than threshold percentage points (pp) in absolute terms
      A config FAILS if: secondary_steered_rate > secondary_baseline_rate + threshold/100
      Equivalently: secondary absolute increase > threshold percentage points
    
    Args:
        primary: Analysis of primary directory (hallucination reduction focus)
        secondary: Analysis of secondary directory (general abilities focus)
        secondary_threshold_pct: Maximum allowed absolute increase in secondary hallucination rate (percentage points)
    
    Returns:
        DataFrame with combined analysis and recommendations
    """
    print(f"\n{'='*80}")
    print(f"CROSS-DIRECTORY COMPARISON")
    print(f"{'='*80}")
    print(f"Primary (hallucination reduction): {primary.output_dir}")
    print(f"Secondary (general abilities): {secondary.output_dir}")
    print(f"Secondary threshold: {secondary_threshold_pct} percentage points (pp) absolute increase allowed")
    
    results = []
    
    # Get all unique (k, alpha) configurations
    primary_configs = set(primary.results_by_config.keys())
    secondary_configs = set(secondary.results_by_config.keys())
    common_configs = primary_configs & secondary_configs
    
    print(f"\n[INFO] Primary configurations: {len(primary_configs)}")
    print(f"[INFO] Secondary configurations: {len(secondary_configs)}")
    print(f"[INFO] Common configurations: {len(common_configs)}")
    
    # Only primary configs (not in secondary)
    primary_only = primary_configs - secondary_configs
    if primary_only:
        print(f"[WARNING] {len(primary_only)} config(s) only in primary (no secondary check possible)")
    
    for k, alpha in sorted(common_configs):
        primary_result = primary.results_by_config[(k, alpha)]
        secondary_result = secondary.results_by_config[(k, alpha)]
        
        # Primary metrics
        primary_rel_reduction = primary_result['relative_reduction']
        primary_abs_reduction = primary_result['absolute_reduction']
        primary_steered_rate = primary_result['steered_halluc_rate']
        
        # Secondary metrics
        secondary_steered_rate = secondary_result['steered_halluc_rate']
        secondary_baseline_rate = secondary.baseline_hallucination_rate
        secondary_rel_reduction = secondary_result['relative_reduction']  # positive = good, negative = bad
        
        # Check if secondary constraint is satisfied
        # Constraint: steered_rate <= baseline_rate + threshold/100 (absolute threshold in percentage points)
        # Equivalently: absolute increase in hallucination rate <= threshold pp
        max_allowed_secondary_rate = secondary_baseline_rate + secondary_threshold_pct / 100
        secondary_passes = secondary_steered_rate <= max_allowed_secondary_rate
        
        results.append({
            'k': k,
            'alpha': alpha,
            # Primary metrics
            'primary_baseline_rate': primary.baseline_hallucination_rate,
            'primary_steered_rate': primary_steered_rate,
            'primary_absolute_reduction': primary_abs_reduction,
            'primary_relative_reduction': primary_rel_reduction,
            # Secondary metrics
            'secondary_baseline_rate': secondary_baseline_rate,
            'secondary_steered_rate': secondary_steered_rate,
            'secondary_absolute_reduction': secondary_result['absolute_reduction'],
            'secondary_relative_reduction': secondary_rel_reduction,
            'secondary_max_allowed_rate': max_allowed_secondary_rate,
            # Evaluation
            'secondary_passes': secondary_passes,
            'valid_config': secondary_passes,
            # Sample counts
            'primary_valid_samples': primary.total_valid_samples,
            'secondary_valid_samples': secondary.total_valid_samples,
        })
    
    # Also add primary-only configs with warning
    for k, alpha in sorted(primary_only):
        primary_result = primary.results_by_config[(k, alpha)]
        
        results.append({
            'k': k,
            'alpha': alpha,
            'primary_baseline_rate': primary.baseline_hallucination_rate,
            'primary_steered_rate': primary_result['steered_halluc_rate'],
            'primary_relative_reduction': primary_result['relative_reduction'],
            'primary_absolute_reduction': primary_result['absolute_reduction'],
            'secondary_baseline_rate': np.nan,
            'secondary_steered_rate': np.nan,
            'secondary_relative_reduction': np.nan,
            'secondary_max_allowed_rate': np.nan,
            'secondary_passes': None,  # Unknown
            'valid_config': None,  # Unknown - needs secondary validation
            'primary_valid_samples': primary.total_valid_samples,
            'secondary_valid_samples': np.nan,
        })
    
    df = pd.DataFrame(results)
    
    # Sort by primary absolute reduction (descending)
    df = df.sort_values('primary_absolute_reduction', ascending=False)
    
    return df


def compare_primary_with_secondaries(
    primary: DirectoryAnalysis,
    secondaries: List[DirectoryAnalysis],
    secondary_thresholds: List[float],
    secondary_labels: List[str]
) -> pd.DataFrame:
    """
    Compare primary (hallucination reduction) with multiple secondary directories.
    
    A config is valid only if it passes ALL secondary checks.
    
    Args:
        primary: Analysis of primary directory (hallucination reduction focus)
        secondaries: List of secondary analyses (1-2 directories)
        secondary_thresholds: List of thresholds for each secondary (max allowed absolute increase in pp)
        secondary_labels: List of labels for each secondary (e.g., "General Abilities", "Knowledge Retention")
    
    Returns:
        DataFrame with combined analysis and per-secondary pass/fail tracking
    """
    print(f"\n{'='*80}")
    print(f"CROSS-DIRECTORY COMPARISON (DUAL SECONDARIES)")
    print(f"{'='*80}")
    print(f"Primary (hallucination reduction): {primary.output_dir}")
    
    for i, secondary in enumerate(secondaries):
        print(f"Secondary {i+1} ({secondary_labels[i]}): {secondary.output_dir}")
        print(f"  Threshold: {secondary_thresholds[i]} pp absolute increase allowed")
    
    results = []
    
    # Get all unique (k, alpha) configurations from primary
    primary_configs = set(primary.results_by_config.keys())
    
    # Find common configs across all secondaries
    common_configs = primary_configs.copy()
    for secondary in secondaries:
        secondary_configs = set(secondary.results_by_config.keys())
        common_configs &= secondary_configs
    
    # Configs only in primary (no secondary validation possible)
    primary_only = primary_configs - common_configs
    
    print(f"\n[INFO] Primary configurations: {len(primary_configs)}")
    for i, secondary in enumerate(secondaries):
        secondary_configs = set(secondary.results_by_config.keys())
        print(f"[INFO] Secondary {i+1} configurations: {len(secondary_configs)}")
    print(f"[INFO] Common configurations: {len(common_configs)}")
    
    if primary_only:
        print(f"[WARNING] {len(primary_only)} config(s) only in primary (no secondary validation possible)")
    
    # Process common configs
    for k, alpha in sorted(common_configs):
        primary_result = primary.results_by_config[(k, alpha)]
        
        # Primary metrics
        primary_rel_reduction = primary_result['relative_reduction']
        primary_abs_reduction = primary_result['absolute_reduction']
        primary_steered_rate = primary_result['steered_halluc_rate']
        
        row = {
            'k': k,
            'alpha': alpha,
            'primary_baseline_rate': primary.baseline_hallucination_rate,
            'primary_steered_rate': primary_steered_rate,
            'primary_absolute_reduction': primary_abs_reduction,
            'primary_relative_reduction': primary_rel_reduction,
            'primary_valid_samples': primary.total_valid_samples,
        }
        
        # Process each secondary
        all_secondaries_pass = True
        
        for sec_idx, (secondary, threshold, label) in enumerate(zip(secondaries, secondary_thresholds, secondary_labels)):
            sec_num = sec_idx + 1
            secondary_result = secondary.results_by_config[(k, alpha)]
            
            # Secondary metrics
            secondary_steered_rate = secondary_result['steered_halluc_rate']
            secondary_baseline_rate = secondary.baseline_hallucination_rate
            secondary_rel_reduction = secondary_result['relative_reduction']
            
            # Check if secondary constraint is satisfied
            # Constraint: steered_rate <= baseline_rate + threshold/100 (absolute threshold in percentage points)
            max_allowed_secondary_rate = secondary_baseline_rate + threshold / 100
            secondary_passes = secondary_steered_rate <= max_allowed_secondary_rate
            
            # Generate failure reason if applicable
            fail_reason = ""
            if not secondary_passes:
                degradation_pp = (secondary_steered_rate - secondary_baseline_rate) * 100
                fail_reason = f"{label}: {degradation_pp:.2f} pp > {threshold:.1f} pp threshold"
                all_secondaries_pass = False
            
            # Add columns for this secondary
            row[f'secondary_{sec_num}_baseline_rate'] = secondary_baseline_rate
            row[f'secondary_{sec_num}_steered_rate'] = secondary_steered_rate
            row[f'secondary_{sec_num}_absolute_reduction'] = secondary_result['absolute_reduction']
            row[f'secondary_{sec_num}_relative_reduction'] = secondary_rel_reduction
            row[f'secondary_{sec_num}_max_allowed_rate'] = max_allowed_secondary_rate
            row[f'secondary_{sec_num}_passes'] = secondary_passes
            row[f'secondary_{sec_num}_fail_reason'] = fail_reason
            row[f'secondary_{sec_num}_label'] = label
            row[f'secondary_{sec_num}_valid_samples'] = secondary.total_valid_samples
        
        # Overall validity: must pass ALL secondaries
        row['valid_config'] = all_secondaries_pass
        
        results.append(row)
    
    # Also add primary-only configs
    for k, alpha in sorted(primary_only):
        primary_result = primary.results_by_config[(k, alpha)]
        
        row = {
            'k': k,
            'alpha': alpha,
            'primary_baseline_rate': primary.baseline_hallucination_rate,
            'primary_steered_rate': primary_result['steered_halluc_rate'],
            'primary_relative_reduction': primary_result['relative_reduction'],
            'primary_absolute_reduction': primary_result['absolute_reduction'],
            'primary_valid_samples': primary.total_valid_samples,
        }
        
        # Set secondary columns to NaN for configs not in secondaries
        for sec_idx in range(len(secondaries)):
            sec_num = sec_idx + 1
            row[f'secondary_{sec_num}_baseline_rate'] = np.nan
            row[f'secondary_{sec_num}_steered_rate'] = np.nan
            row[f'secondary_{sec_num}_absolute_reduction'] = np.nan
            row[f'secondary_{sec_num}_relative_reduction'] = np.nan
            row[f'secondary_{sec_num}_max_allowed_rate'] = np.nan
            row[f'secondary_{sec_num}_passes'] = None
            row[f'secondary_{sec_num}_fail_reason'] = "Not evaluated"
            row[f'secondary_{sec_num}_label'] = secondary_labels[sec_idx]
            row[f'secondary_{sec_num}_valid_samples'] = np.nan
        
        row['valid_config'] = None  # Unknown
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Sort by primary absolute reduction (descending)
    df = df.sort_values('primary_absolute_reduction', ascending=False)
    
    return df


# ================================================================
# VISUALIZATION
# ================================================================

def generate_per_directory_plots(
    analysis: DirectoryAnalysis,
    output_dir: str,
    dir_label: str,
    timestamp: str,
    secondary_analysis: Optional[DirectoryAnalysis] = None,
    secondary_threshold_pct: float = 1.0
) -> List[str]:
    """Generate plots for a single directory analysis.
    
    Args:
        analysis: Primary analysis results
        output_dir: Output directory for plots
        dir_label: Label for plots (e.g., "Primary")
        timestamp: Timestamp for filenames
        secondary_analysis: Optional secondary analysis for cross-comparison plots
        secondary_threshold_pct: Threshold in percentage points (pp) for secondary comparison (default: 1.0)
    """
    
    plot_files = []
    
    # Prepare data
    data = []
    for (k, alpha), metrics in analysis.results_by_config.items():
        data.append({
            'k': k,
            'alpha': alpha,
            'absolute_reduction': metrics['absolute_reduction'],
            'steered_rate': metrics['steered_halluc_rate'],
            'relative_reduction': metrics['relative_reduction']
        })
    
    if not data:
        return plot_files
    
    df = pd.DataFrame(data)
    unique_k = sorted(df['k'].unique())
    unique_alpha = sorted(df['alpha'].unique())
    
    # --- Plot 1: Heatmap of absolute reduction ---
    if len(unique_k) > 1 or len(unique_alpha) > 1:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        pivot_df = df.pivot_table(index='k', columns='alpha', values='absolute_reduction', aggfunc='max')
        
        # Create diverging colormap: red (negative) -> white (0) -> green (positive)
        vmin = pivot_df.values.min()
        vmax = pivot_df.values.max()
        # Ensure vmin < 0 < vmax for TwoSlopeNorm
        # If all values are on one side of zero, extend the range
        if vmin >= 0:  # All non-negative
            vmin = -vmax if vmax > 0 else -1
        if vmax <= 0:  # All non-positive
            vmax = -vmin if vmin < 0 else 1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        im = ax.imshow(pivot_df.values, cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
        
        ax.set_xticks(range(len(pivot_df.columns)))
        ax.set_xticklabels([f'{a:.1f}' for a in pivot_df.columns], rotation=45, fontsize=9)
        ax.set_yticks(range(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index, fontsize=9)
        
        ax.set_xlabel('Alpha (Steering Strength)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Top-K Heads', fontsize=12, fontweight='bold')
        ax.set_title(f'{dir_label}: Absolute Hallucination Reduction (%)\nBaseline: {analysis.baseline_hallucination_rate*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Absolute Reduction (%)', fontsize=11, fontweight='bold')
        
        # Add value annotations
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                val = pivot_df.iloc[i, j]
                if not pd.isna(val):
                    color = 'white' if abs(val) > 15 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f"{dir_label.lower()}_heatmap_{timestamp}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(heatmap_path)
        print(f"[INFO] {dir_label} heatmap: {heatmap_path}")
    
    # --- Plot 2: Line plot per k value ---
    if len(unique_k) > 0 and len(unique_alpha) > 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_k)))
        
        for idx, k in enumerate(unique_k):
            k_data = df[df['k'] == k].sort_values('alpha')
            ax.plot(k_data['alpha'], k_data['absolute_reduction'], 
                   marker='o', linewidth=2, markersize=6, label=f'k={k}', color=colors[idx])
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Alpha (Steering Strength)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Hallucination Reduction (percentage points)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dir_label}: Absolute Reduction by K and Alpha\nBaseline: {analysis.baseline_hallucination_rate*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        ax.legend(title='Top-K', fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        line_path = os.path.join(output_dir, f"{dir_label.lower()}_lineplot_{timestamp}.png")
        plt.savefig(line_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(line_path)
        print(f"[INFO] {dir_label} line plot: {line_path}")
    
    # --- Plot 3: Bar chart of top 15 configs ---
    # If secondary analysis available, create cross-comparison chart
    if secondary_analysis is not None:
        cross_compare_path = generate_top15_cross_comparison_chart(
            analysis, secondary_analysis, output_dir, timestamp, secondary_threshold_pct
        )
        if cross_compare_path:
            plot_files.append(cross_compare_path)
    else:
        # Standard single-directory top 15 chart
        top_configs = df.nlargest(15, 'absolute_reduction')
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        labels = [f'k={int(row["k"])}, α={row["alpha"]:.1f}' for _, row in top_configs.iterrows()]
        values = top_configs['absolute_reduction'].values
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Absolute Hallucination Reduction (percentage points)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dir_label}: Top 15 Configurations by Absolute Reduction\nBaseline: {analysis.baseline_hallucination_rate*100:.2f}%', 
                     fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"{dir_label.lower()}_top15_{timestamp}.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(bar_path)
        print(f"[INFO] {dir_label} top 15 bar chart: {bar_path}")
    
    # --- Plot 4: Dual-value heatmap (both primary and secondary) ---
    if secondary_analysis is not None:
        dual_heatmap_path = generate_dual_value_heatmap(
            analysis, secondary_analysis, output_dir, timestamp
        )
        if dual_heatmap_path:
            plot_files.append(dual_heatmap_path)
    
    # --- Plot 5: Individual Confusion Matrices for each (k, alpha) ---
    # Only for primary analysis - create in dedicated folder
    if dir_label == "Primary":
        cm_folder = os.path.join(output_dir, "confusion_matrix")
        os.makedirs(cm_folder, exist_ok=True)
        print(f"[INFO] Creating confusion matrices in: {cm_folder}")
        
        # Generate confusion matrix for each (k, alpha) configuration
        for (k, alpha) in sorted(analysis.results_by_config.keys()):
            cm_path = generate_confusion_matrix_plot(
                analysis, cm_folder, dir_label, timestamp, k=k, alpha=alpha
            )
            if cm_path:
                plot_files.append(cm_path)
    
    return plot_files


def generate_top15_cross_comparison_chart(
    primary: DirectoryAnalysis,
    secondary: DirectoryAnalysis,
    output_dir: str,
    timestamp: str,
    secondary_threshold_pct: float = 1.0
) -> Optional[str]:
    """
    Generate side-by-side bar chart comparing top 15 primary configs with secondary values.
    
    Shows:
    - Secondary on the LEFT (negative x-axis)
    - Primary on the RIGHT (positive x-axis)
    - Threshold line for secondary degradation (absolute percentage points)
    - Each (k, alpha) pair labeled
    """
    
    # Build comparison data for common configs
    comparison_data = []
    
    for (k, alpha), primary_metrics in primary.results_by_config.items():
        if (k, alpha) in secondary.results_by_config:
            secondary_metrics = secondary.results_by_config[(k, alpha)]
            
            comparison_data.append({
                'k': k,
                'alpha': alpha,
                'label': f'k={k}, α={alpha:.1f}',
                'primary_reduction': primary_metrics['absolute_reduction'],
                'secondary_reduction': secondary_metrics['absolute_reduction'],
            })
    
    if not comparison_data:
        print("[WARNING] No common configurations for cross-comparison chart")
        return None
    
    # Sort by primary absolute reduction and take top 15
    comparison_data.sort(key=lambda x: x['primary_reduction'], reverse=True)
    top_15 = comparison_data[:15]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    y_pos = np.arange(len(top_15))
    bar_height = 0.35
    
    # Extract values
    labels = [item['label'] for item in top_15]
    secondary_reductions = [-item['secondary_reduction'] for item in top_15]  # Negative for left side
    primary_reductions = [item['primary_reduction'] for item in top_15]
    
    # Create bars
    bars_secondary = ax.barh(y_pos - bar_height/2, secondary_reductions, bar_height, 
                             label='General Abilities', color='steelblue', 
                             alpha=0.8, edgecolor='black', linewidth=0.7)
    bars_primary = ax.barh(y_pos + bar_height/2, primary_reductions, bar_height, 
                           label='Primary (Hallucination Reduction)', color='forestgreen', 
                           alpha=0.8, edgecolor='black', linewidth=0.7)
    
    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Absolute Hallucination Reduction (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Configurations by Absolute Reduction: Primary vs General Abilities\n(Secondary on left, Primary on right)', 
                 fontsize=14, fontweight='bold')
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add threshold line for secondary using the command-line specified threshold (absolute percentage points)
    # Since secondary bars are negated for left-side display, threshold line is at positive value
    threshold_line = secondary_threshold_pct
    ax.axvline(x=threshold_line, color='orange', linestyle='--', linewidth=2.5, 
               label=f'General Abilities threshold (±{secondary_threshold_pct} pp)')
    
    # Add value labels on bars
    for i, (sec_val, prim_val) in enumerate(zip(secondary_reductions, primary_reductions)):
        # Secondary (left side)
        ax.text(sec_val - 0.5, i - bar_height/2, f'{-sec_val:.1f} pp', 
               va='center', ha='right', fontsize=8, fontweight='bold', color='darkblue')
        
        # Primary (right side)
        ax.text(prim_val + 0.5, i + bar_height/2, f'{prim_val:.1f} pp', 
               va='center', ha='left', fontsize=8, fontweight='bold', color='darkgreen')
    
    # Legend and info
    ax.legend(fontsize=11, loc='lower right')
    
    info_text = f'Primary baseline: {primary.baseline_hallucination_rate*100:.2f}%\nSecondary baseline: {secondary.baseline_hallucination_rate*100:.2f}%'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"top15_cross_comparison_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Top 15 cross-comparison chart: {chart_path}")
    return chart_path


def generate_dual_value_heatmap(
    primary: DirectoryAnalysis,
    secondary: DirectoryAnalysis,
    output_dir: str,
    timestamp: str
) -> Optional[str]:
    """
    Generate heatmap with dual values (Primary / General Abilities) in each cell.
    
    Cell format: "P: primary_reduction\nG: general_abilities_reduction" 
    Colors based on primary reduction (red -> white -> green)
    """
    
    # Get common (k, alpha) configurations
    common_configs = set(primary.results_by_config.keys()) & set(secondary.results_by_config.keys())
    
    if not common_configs:
        print("[WARNING] No common configurations for dual heatmap")
        return None
    
    # Build data structure for heatmap
    # Organize by (k, alpha)
    unique_k = sorted(set(k for k, a in common_configs))
    unique_alpha = sorted(set(a for k, a in common_configs))
    
    # Create matrices for values
    primary_matrix = np.full((len(unique_k), len(unique_alpha)), np.nan)
    secondary_matrix = np.full((len(unique_k), len(unique_alpha)), np.nan)
    
    for k_idx, k in enumerate(unique_k):
        for alpha_idx, alpha in enumerate(unique_alpha):
            if (k, alpha) in common_configs:
                primary_val = primary.results_by_config[(k, alpha)]['absolute_reduction']
                secondary_val = secondary.results_by_config[(k, alpha)]['absolute_reduction']
                primary_matrix[k_idx, alpha_idx] = primary_val
                secondary_matrix[k_idx, alpha_idx] = secondary_val
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use primary values for color mapping
    vmin = np.nanmin(primary_matrix)
    vmax = np.nanmax(primary_matrix)
    
    # Ensure vmin < 0 < vmax for TwoSlopeNorm
    if vmin >= 0:
        vmin = -vmax if vmax > 0 else -1
    if vmax <= 0:
        vmax = -vmin if vmin < 0 else 1
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Create heatmap with primary values for coloring
    im = ax.imshow(primary_matrix, cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
    
    # Set ticks and labels
    ax.set_xticks(range(len(unique_alpha)))
    ax.set_xticklabels([f'{a:.1f}' for a in unique_alpha], fontsize=10, rotation=45)
    ax.set_yticks(range(len(unique_k)))
    ax.set_yticklabels(unique_k, fontsize=10)
    
    ax.set_xlabel('Alpha (Steering Strength)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-K Heads', fontsize=12, fontweight='bold')
    ax.set_title('Dual-Value Heatmap: Primary / General Abilities Absolute Reduction (%)\nColor based on Primary Reduction', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Primary Reduction (%)', fontsize=11, fontweight='bold')
    
    # Add cell annotations with both values
    for i in range(len(unique_k)):
        for j in range(len(unique_alpha)):
            if not np.isnan(primary_matrix[i, j]):
                primary_val = primary_matrix[i, j]
                secondary_val = secondary_matrix[i, j]
                
                # Choose text color based on primary value
                text_color = 'white' if abs(primary_val) > 15 else 'black'
                
                # Format: P: Primary, G: General Abilities
                text = f'P: {primary_val:.1f}\nG: {secondary_val:.1f}'
                ax.text(j, i, text, ha='center', va='center', color=text_color, 
                       fontsize=9, fontweight='bold', linespacing=1.5)
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"dual_value_heatmap_{timestamp}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Dual-value heatmap: {heatmap_path}")
    return heatmap_path


def generate_triple_heatmap(
    primary: DirectoryAnalysis,
    secondaries: List[DirectoryAnalysis],
    secondary_labels: List[str],
    output_dir: str,
    timestamp: str
) -> Optional[str]:
    """
    Generate 3×1 heatmap matrix showing Primary | Secondary1 | Secondary2 side-by-side.
    
    Option C visualization: Each cell shows relative reduction value.
    Colors are based on primary reduction (red -> white -> green).
    Pass/fail status overlaid as border color for secondary heatmaps.
    """
    
    # Get common (k, alpha) configurations
    common_configs = set(primary.results_by_config.keys())
    for secondary in secondaries:
        secondary_configs = set(secondary.results_by_config.keys())
        common_configs &= secondary_configs
    
    if not common_configs:
        print("[WARNING] No common configurations for triple heatmap")
        return None
    
    # Build data structure for heatmaps
    unique_k = sorted(set(k for k, a in common_configs))
    unique_alpha = sorted(set(a for k, a in common_configs))
    
    # Create matrices for values
    primary_matrix = np.full((len(unique_k), len(unique_alpha)), np.nan)
    secondary_matrices = [np.full((len(unique_k), len(unique_alpha)), np.nan) for _ in secondaries]
    pass_matrices = [np.full((len(unique_k), len(unique_alpha)), True, dtype=bool) for _ in secondaries]
    
    for k_idx, k in enumerate(unique_k):
        for alpha_idx, alpha in enumerate(unique_alpha):
            if (k, alpha) in common_configs:
                primary_val = primary.results_by_config[(k, alpha)]['absolute_reduction']
                primary_matrix[k_idx, alpha_idx] = primary_val
                
                for sec_idx, secondary in enumerate(secondaries):
                    secondary_val = secondary.results_by_config[(k, alpha)]['absolute_reduction']
                    secondary_matrices[sec_idx][k_idx, alpha_idx] = secondary_val
                    # For now, assume pass (will be determined by thresholds in comparison df)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Use primary values for shared color mapping
    vmin = np.nanmin(primary_matrix)
    vmax = np.nanmax(primary_matrix)
    
    # Ensure vmin < 0 < vmax for TwoSlopeNorm
    if vmin >= 0:
        vmin = -vmax if vmax > 0 else -1
    if vmax <= 0:
        vmax = -vmin if vmin < 0 else 1
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Primary heatmap
    im0 = axes[0].imshow(primary_matrix, cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
    axes[0].set_xticks(range(len(unique_alpha)))
    axes[0].set_xticklabels([f'{a:.1f}' for a in unique_alpha], fontsize=9, rotation=45)
    axes[0].set_yticks(range(len(unique_k)))
    axes[0].set_yticklabels(unique_k, fontsize=9)
    axes[0].set_xlabel('Alpha', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Top-K', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Primary\nAbsolute Reduction (%)', fontsize=12, fontweight='bold')
    
    # Add cell annotations for primary
    for i in range(len(unique_k)):
        for j in range(len(unique_alpha)):
            if not np.isnan(primary_matrix[i, j]):
                val = primary_matrix[i, j]
                text_color = 'white' if abs(val) > 15 else 'black'
                axes[0].text(j, i, f'{val:.1f}', ha='center', va='center', 
                           color=text_color, fontsize=9, fontweight='bold')
    
    # Secondary heatmaps
    for sec_idx, (secondary_matrix, secondary_label, ax) in enumerate(zip(secondary_matrices, secondary_labels, axes[1:])):
        im_sec = ax.imshow(secondary_matrix, cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
        ax.set_xticks(range(len(unique_alpha)))
        ax.set_xticklabels([f'{a:.1f}' for a in unique_alpha], fontsize=9, rotation=45)
        ax.set_yticks(range(len(unique_k)))
        ax.set_yticklabels(unique_k, fontsize=9)
        ax.set_xlabel('Alpha', fontsize=11, fontweight='bold')
        ax.set_ylabel('Top-K', fontsize=11, fontweight='bold')
        ax.set_title(f'{secondary_label}\nAbsolute Reduction (%)', fontsize=12, fontweight='bold')
        
        # Add cell annotations
        for i in range(len(unique_k)):
            for j in range(len(unique_alpha)):
                if not np.isnan(secondary_matrix[i, j]):
                    val = secondary_matrix[i, j]
                    text_color = 'white' if abs(val) > 15 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                           color=text_color, fontsize=9, fontweight='bold')
    
    # Add shared colorbar on the right side without overlapping
    fig.subplots_adjust(right=0.88)  # Make room for colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im0, cax=cbar_ax, label='Absolute Reduction (%)')
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Adjust layout to not overlap colorbar
    heatmap_path = os.path.join(output_dir, f"triple_heatmap_{timestamp}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Triple heatmap: {heatmap_path}")
    return heatmap_path


def generate_dual_secondary_scatter_plots(
    primary: DirectoryAnalysis,
    secondaries: List[DirectoryAnalysis],
    secondary_labels: List[str],
    comparison_df: pd.DataFrame,
    output_dir: str,
    secondary_thresholds: List[float],
    timestamp: str
) -> Optional[str]:
    """
    Generate side-by-side scatter plots: Primary vs Secondary1 (left) and Primary vs Secondary2 (right).
    
    Option A visualization: Each plot shows:
    - Green circles: config passes all checks (including this secondary)
    - Red X: config fails this specific secondary
    - Blue star: best passing config overall
    - Threshold line showing the limit for each secondary
    """
    
    if len(secondaries) < 1 or len(secondaries) > 2:
        print(f"[WARNING] Dual scatter plots require 1-2 secondaries, got {len(secondaries)}")
        return None
    
    # Filter out NaN values from comparison_df
    valid_df = comparison_df.dropna(subset=['secondary_1_relative_reduction'])
    
    fig, axes = plt.subplots(1, len(secondaries), figsize=(16, 7))
    
    # Ensure axes is always a list for consistent indexing
    if len(secondaries) == 1:
        axes = [axes]
    
    for sec_idx, (ax, secondary, threshold, label) in enumerate(zip(axes, secondaries, secondary_thresholds, secondary_labels)):
        sec_num = sec_idx + 1
        
        # Separate configs by pass/fail for this specific secondary
        passing_this_sec = valid_df[valid_df[f'secondary_{sec_num}_passes'] == True]
        failing_this_sec = valid_df[valid_df[f'secondary_{sec_num}_passes'] == False]
        
        # Also identify configs that pass ALL secondaries
        all_passing = valid_df[valid_df['valid_config'] == True]
        
        # Plot failing configs for this secondary
        if not failing_this_sec.empty:
            ax.scatter(
                failing_this_sec[f'secondary_{sec_num}_absolute_reduction'],
                failing_this_sec['primary_absolute_reduction'],
                c='red', label=f'Fails {label}', s=120, alpha=0.6, marker='X', 
                edgecolors='black', linewidths=1.5
            )
        
        # Plot passing configs (but only this secondary, may fail other secondaries)
        passing_only_this = passing_this_sec[passing_this_sec['valid_config'] != True]
        if not passing_only_this.empty:
            ax.scatter(
                passing_only_this[f'secondary_{sec_num}_absolute_reduction'],
                passing_only_this['primary_absolute_reduction'],
                c='lightgreen', label=f'Passes {label} only', s=100, alpha=0.5, marker='o',
                edgecolors='black', linewidths=1
            )
        
        # Plot configs passing ALL secondaries (including this one)
        if not all_passing.empty:
            ax.scatter(
                all_passing[f'secondary_{sec_num}_absolute_reduction'],
                all_passing['primary_absolute_reduction'],
                c='darkgreen', label='Passes all checks', s=130, alpha=0.8, marker='o',
                edgecolors='black', linewidths=1.5
            )
        
        # Add threshold line (vertical line at -threshold)
        ax.axvline(x=-threshold, color='orange', linestyle='--', linewidth=2.5, 
                   label=f'Threshold (-{threshold}%)')
        
        # Add zero lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Highlight best config passing all checks (by absolute reduction)
        if not all_passing.empty:
            best_config = all_passing.loc[all_passing['primary_absolute_reduction'].idxmax()]
            ax.scatter(
                best_config[f'secondary_{sec_num}_absolute_reduction'],
                best_config['primary_absolute_reduction'],
                marker='*', s=600, color='gold', edgecolor='black', linewidth=2,
                label=f"★ Best: k={int(best_config['k'])}, α={best_config['alpha']:.1f} (max absolute reduction)", zorder=5
            )
        
        ax.set_xlabel(f'{label}: Absolute Reduction (percentage points)\n(negative = degradation, positive = improvement)', 
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('Primary: Absolute Reduction (percentage points)', fontsize=10, fontweight='bold')
        ax.set_title(f'Primary vs {label}\nAbsolute Hallucination Reduction', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Primary: Absolute Reduction (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'Primary vs {label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"dual_scatter_plots_{timestamp}.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Dual scatter plots: {scatter_path}")
    return scatter_path


def generate_confusion_matrix_plot(
    analysis: DirectoryAnalysis,
    output_dir: str,
    dir_label: str,
    timestamp: str,
    k: Optional[int] = None,
    alpha: Optional[float] = None
) -> Optional[str]:
    """
    Generate confusion matrix comparing baseline vs steered hallucination labels.
    
    If k and alpha are provided, generates confusion matrix for that specific (k, alpha) combination.
    Otherwise, generates aggregate confusion matrix across all configurations.
    
    Uses only valid samples (no API failures in either baseline or steered).
    Shows counts of: (0,0), (0,1), (1,0), (1,1) combinations.
    """
    
    # Collect baseline and steered labels from valid samples
    baseline_labels = []
    steered_labels = []
    
    if k is not None and alpha is not None:
        # Generate for specific (k, alpha) configuration
        for exp in analysis.all_experiments:
            if exp.k == k and exp.alpha == alpha:
                for sample_idx in sorted(analysis.valid_sample_indices):
                    result = exp.results_by_idx.get(sample_idx)
                    if result:
                        baseline_labels.append(result['baseline_hallucination'])
                        steered_labels.append(result['steered_hallucination'])
                break
    else:
        # Generate aggregate confusion matrix across all experiments
        for exp in analysis.all_experiments:
            for sample_idx in sorted(analysis.valid_sample_indices):
                result = exp.results_by_idx.get(sample_idx)
                if result:
                    baseline_labels.append(result['baseline_hallucination'])
                    steered_labels.append(result['steered_hallucination'])
    
    # Use only the intersection of valid samples from both lists
    min_len = min(len(baseline_labels), len(steered_labels))
    baseline_labels = baseline_labels[:min_len]
    steered_labels = steered_labels[:min_len]
    
    if not baseline_labels:
        print(f"[WARNING] No valid samples for confusion matrix in {dir_label}")
        return None
    
    # Compute confusion matrix
    cm = confusion_matrix(baseline_labels, steered_labels, labels=[0, 1])
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='RdPu', 
        cbar_kws={'label': 'Count'},
        xticklabels=['No Hallucination (0)', 'Hallucination (1)'],
        yticklabels=['No Hallucination (0)', 'Hallucination (1)'],
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'},
        cbar=True
    )
    
    ax.set_xlabel('Steered Hallucination Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Baseline Hallucination Label', fontsize=12, fontweight='bold')
    
    if k is not None and alpha is not None:
        title = f'{dir_label}: k={k}, α={alpha:.2f}\nConfusion Matrix: Baseline vs Steered Hallucination (n={len(baseline_labels)})'
    else:
        title = f'{dir_label}: Confusion Matrix (Aggregate)\nBaseline vs Steered Hallucination (n={len(baseline_labels)})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add statistics below the heatmap
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]
    
    total = cm.sum()
    accuracy = (true_negatives + true_positives) / total if total > 0 else 0
    
    stats_text = f'TN={true_negatives}, FP={false_positives}, FN={false_negatives}, TP={true_positives}\nAccuracy: {accuracy*100:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, -0.15, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    if k is not None and alpha is not None:
        cm_path = os.path.join(output_dir, f"{dir_label.lower()}_k{k}_a{alpha:.2f}_confusion_matrix.png")
    else:
        cm_path = os.path.join(output_dir, f"{dir_label.lower()}_confusion_matrix_{timestamp}.png")
    
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] {dir_label} confusion matrix: {cm_path}")
    return cm_path


def generate_cross_directory_plots(
    primary: DirectoryAnalysis,
    secondary: DirectoryAnalysis,
    comparison_df: pd.DataFrame,
    output_dir: str,
    secondary_threshold_pct: float,
    timestamp: str
) -> List[str]:
    """Generate plots comparing primary and secondary directories (threshold in absolute percentage points)."""
    
    plot_files = []
    
    if comparison_df.empty:
        return plot_files
    
    # --- Plot 1: Scatter plot - Primary reduction vs Secondary reduction ---
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Filter out NaN values
    valid_df = comparison_df.dropna(subset=['secondary_absolute_reduction'])
    
    # Separate valid and invalid configs
    passing = valid_df[valid_df['valid_config'] == True]
    failing = valid_df[valid_df['valid_config'] == False]
    
    if not passing.empty:
        scatter1 = ax.scatter(
            passing['secondary_absolute_reduction'],
            passing['primary_absolute_reduction'],
            c='green', label=f'Passes general abilities check (≥ -{secondary_threshold_pct}%)', 
            s=100, alpha=0.7, marker='o', edgecolors='black', linewidths=1
        )
    
    if not failing.empty:
        scatter2 = ax.scatter(
            failing['secondary_absolute_reduction'],
            failing['primary_absolute_reduction'],
            c='red', label=f'Fails general abilities check (< -{secondary_threshold_pct}%)', 
            s=100, alpha=0.7, marker='X', edgecolors='black', linewidths=1
        )
    
    # Add threshold line (vertical line at -threshold in percentage)
    ax.axvline(x=-secondary_threshold_pct, color='orange', linestyle='--', 
               linewidth=2, label=f'Threshold (-{secondary_threshold_pct}%)')
    
    # Add zero lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Highlight best valid point (by absolute reduction)
    if not passing.empty:
        best = passing.loc[passing['primary_absolute_reduction'].idxmax()]
        ax.scatter(
            [best['secondary_absolute_reduction']],
            [best['primary_absolute_reduction']],
            c='gold', s=400, marker='*', edgecolors='black', linewidths=2,
            label=f"★ Best: k={int(best['k'])}, α={best['alpha']:.1f} (max absolute reduction)", zorder=5
        )
    
    ax.set_xlabel('General Abilities: Absolute Hallucination Reduction (percentage points)\n(negative = degradation, positive = improvement)', 
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Primary: Absolute Hallucination Reduction (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Directory Comparison\nAbsolute Hallucination Reduction: Primary vs General Abilities', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add annotation box
    textstr = f'Primary baseline: {primary.baseline_hallucination_rate*100:.2f}%\nGeneral Abilities baseline: {secondary.baseline_hallucination_rate*100:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"cross_comparison_scatter_{timestamp}.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(scatter_path)
    print(f"[INFO] Cross-comparison scatter: {scatter_path}")
    
    # --- Plot 2: Combined heatmap (only for passing configs) ---
    if not passing.empty and len(passing) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Primary heatmap
        try:
            pivot_primary = passing.pivot_table(index='k', columns='alpha', 
                                                values='primary_absolute_reduction', aggfunc='max')
            if not pivot_primary.empty:
                # Create diverging colormap: red (negative) -> white (0) -> green (positive)
                vmin = pivot_primary.values.min()
                vmax = pivot_primary.values.max()
                # Ensure vmin < 0 < vmax for TwoSlopeNorm
                # If all values are on one side of zero, extend the range
                if vmin >= 0:  # All non-negative
                    vmin = -vmax if vmax > 0 else -1
                if vmax <= 0:  # All non-positive
                    vmax = -vmin if vmin < 0 else 1
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
                im1 = ax1.imshow(pivot_primary.values, cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
                ax1.set_xticks(range(len(pivot_primary.columns)))
                ax1.set_xticklabels([f'{a:.1f}' for a in pivot_primary.columns], rotation=45)
                ax1.set_yticks(range(len(pivot_primary.index)))
                ax1.set_yticklabels(pivot_primary.index)
                ax1.set_xlabel('Alpha', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Top-K', fontsize=11, fontweight='bold')
                ax1.set_title('Primary: Absolute Reduction (%)\n(Only configs passing general abilities check)', fontsize=12, fontweight='bold')
                
                # Add cell value annotations
                for i in range(len(pivot_primary.index)):
                    for j in range(len(pivot_primary.columns)):
                        val = pivot_primary.iloc[i, j]
                        if not pd.isna(val):
                            color = 'white' if abs(val) > 20 else 'black'
                            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=9, fontweight='bold')
                
                plt.colorbar(im1, ax=ax1, label='Reduction (%)')
        except Exception as e:
            ax1.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
            ax1.set_title('Primary: Relative Reduction (%)')
        
        # Secondary heatmap
        try:
            pivot_secondary = passing.pivot_table(index='k', columns='alpha', 
                                                  values='secondary_absolute_reduction', aggfunc='max')
            if not pivot_secondary.empty:
                # Create diverging colormap: red (negative) -> white (0) -> green (positive)
                vmin = pivot_secondary.values.min()
                vmax = pivot_secondary.values.max()
                # Ensure vmin < 0 < vmax for TwoSlopeNorm
                # If all values are on one side of zero, extend the range
                if vmin >= 0:  # All non-negative
                    vmin = -vmax if vmax > 0 else -1
                if vmax <= 0:  # All non-positive
                    vmax = -vmin if vmin < 0 else 1
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
                im2 = ax2.imshow(pivot_secondary.values, cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
                ax2.set_xticks(range(len(pivot_secondary.columns)))
                ax2.set_xticklabels([f'{a:.1f}' for a in pivot_secondary.columns], rotation=45)
                ax2.set_yticks(range(len(pivot_secondary.index)))
                ax2.set_yticklabels(pivot_secondary.index)
                ax2.set_xlabel('Alpha', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Top-K', fontsize=11, fontweight='bold')
                ax2.set_title('General Abilities: Absolute Reduction (%)\n(Only configs passing general abilities check)', fontsize=12, fontweight='bold')
                
                # Add cell value annotations
                for i in range(len(pivot_secondary.index)):
                    for j in range(len(pivot_secondary.columns)):
                        val = pivot_secondary.iloc[i, j]
                        if not pd.isna(val):
                            color = 'white' if abs(val) > 15 else 'black'
                            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=9, fontweight='bold')
                
                plt.colorbar(im2, ax=ax2, label='Reduction (%)')
        except Exception as e:
            ax2.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
            ax2.set_title('General Abilities: Relative Reduction (%)')
        
        plt.tight_layout()
        combined_heatmap_path = os.path.join(output_dir, f"cross_comparison_heatmaps_{timestamp}.png")
        plt.savefig(combined_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(combined_heatmap_path)
        print(f"[INFO] Combined heatmaps: {combined_heatmap_path}")
    
    # --- Plot 3: Pareto frontier visualization ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    valid_df = comparison_df.dropna(subset=['secondary_absolute_reduction'])
    
    # Plot all points
    passing = valid_df[valid_df['valid_config'] == True]
    failing = valid_df[valid_df['valid_config'] == False]
    
    if not failing.empty:
        ax.scatter(
            -failing['secondary_absolute_reduction'],  # Negate so higher = worse degradation
            failing['primary_absolute_reduction'],
            c='lightcoral', label='Fails general abilities check', s=60, alpha=0.5, marker='x'
        )
    
    if not passing.empty:
        ax.scatter(
            -passing['secondary_absolute_reduction'],
            passing['primary_absolute_reduction'],
            c='green', label='Passes general abilities check', s=80, alpha=0.7, marker='o', edgecolors='black'
        )
        
        # Find and highlight Pareto optimal points
        pareto_mask = []
        for i, row in passing.iterrows():
            is_pareto = True
            for j, other in passing.iterrows():
                if i != j:
                    # Check if 'other' dominates 'row'
                    if (other['primary_absolute_reduction'] >= row['primary_absolute_reduction'] and
                        other['secondary_absolute_reduction'] >= row['secondary_absolute_reduction'] and
                        (other['primary_absolute_reduction'] > row['primary_absolute_reduction'] or
                         other['secondary_absolute_reduction'] > row['secondary_absolute_reduction'])):
                        is_pareto = False
                        break
            pareto_mask.append(is_pareto)
        
        pareto_points = passing[pareto_mask]
        if not pareto_points.empty:
            ax.scatter(
                -pareto_points['secondary_absolute_reduction'],
                pareto_points['primary_absolute_reduction'],
                c='gold', s=200, marker='*', edgecolors='black', linewidths=2,
                label='Pareto optimal', zorder=5
            )
    
    ax.axvline(x=secondary_threshold_pct, color='orange', linestyle='--', linewidth=2, 
               label=f'Threshold ({secondary_threshold_pct}%)')
    
    ax.set_xlabel('General Abilities: Degradation (percentage points increase in hallucination rate)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Primary: Absolute Reduction (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Absolute Primary Reduction vs General Abilities Degradation\n(Higher absolute primary reduction + Lower general abilities degradation = Better)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pareto_path = os.path.join(output_dir, f"cross_comparison_pareto_{timestamp}.png")
    plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(pareto_path)
    print(f"[INFO] Pareto frontier: {pareto_path}")
    
    return plot_files


# ================================================================
# REPORTING
# ================================================================

def generate_summary_report(
    primary: DirectoryAnalysis,
    secondary: Optional[DirectoryAnalysis],
    comparison_df: Optional[pd.DataFrame],
    output_dir: str,
    secondary_threshold_pct: float
) -> str:
    """Generate comprehensive text report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"cross_analysis_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("╔" + "═" * 78 + "╗\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("║" + "CROSS-EXPERIMENT ANALYSIS REPORT".center(78) + "║\n")
        f.write("║" + f"Generated: {timestamp}".center(78) + "║\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("╚" + "═" * 78 + "╝\n\n")
        
        # Primary directory summary
        f.write("=" * 80 + "\n")
        f.write("PRIMARY DIRECTORY (Hallucination Reduction Focus)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Directory: {primary.output_dir}\n")
        f.write(f"Baseline directory: {primary.baseline_dir}\n")
        f.write(f"Valid samples (no API failures): {primary.total_valid_samples}\n")
        f.write(f"Baseline hallucination rate: {primary.baseline_hallucination_rate*100:.2f}%\n")
        f.write(f"Baseline hallucination count: {primary.baseline_hallucination_count}\n")
        f.write(f"Total configurations tested: {len(primary.results_by_config)}\n\n")
        
        # Best configs in primary
        f.write("-" * 80 + "\n")
        f.write("TOP 10 CONFIGURATIONS (by absolute reduction)\n")
        f.write("-" * 80 + "\n")
        
        sorted_configs = sorted(
            primary.results_by_config.items(),
            key=lambda x: x[1]['absolute_reduction'],
            reverse=True
        )[:10]
        
        for rank, ((k, alpha), metrics) in enumerate(sorted_configs, 1):
            f.write(f"\n{rank}. k={k}, alpha={alpha:.2f}\n")
            f.write(f"   Steered rate: {metrics['steered_halluc_rate']*100:.2f}%\n")
            f.write(f"   Absolute reduction: {metrics['absolute_reduction']:.2f} pp\n")
            f.write(f"   Relative reduction: {metrics['relative_reduction']:.2f}%\n")
        
        # General Abilities directory summary (if provided)
        if secondary:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("GENERAL ABILITIES DIRECTORY (General Abilities Check)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Directory: {secondary.output_dir}\n")
            f.write(f"Baseline directory: {secondary.baseline_dir}\n")
            f.write(f"Valid samples (no API failures): {secondary.total_valid_samples}\n")
            f.write(f"Baseline hallucination rate: {secondary.baseline_hallucination_rate*100:.2f}%\n")
            f.write(f"Baseline hallucination count: {secondary.baseline_hallucination_count}\n")
            f.write(f"Total configurations tested: {len(secondary.results_by_config)}\n")
            f.write(f"\nDegradation threshold: {secondary_threshold_pct} percentage points (pp) absolute increase allowed\n")
            f.write(f"  (Config passes if: secondary_steered_rate <= {secondary.baseline_hallucination_rate*100:.2f}% + {secondary_threshold_pct:.2f} pp)\n")
            f.write(f"  (Equivalently: secondary hallucination rate increase <= {secondary_threshold_pct} pp absolute)\n")
        
        # Cross-comparison results
        if comparison_df is not None and not comparison_df.empty:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("CROSS-DIRECTORY COMPARISON RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Count valid/invalid configs
            valid_configs = comparison_df[comparison_df['valid_config'] == True]
            invalid_configs = comparison_df[comparison_df['valid_config'] == False]
            unknown_configs = comparison_df[comparison_df['valid_config'].isna()]
            
            f.write(f"Configurations passing general abilities check: {len(valid_configs)}\n")
            f.write(f"Configurations failing general abilities check: {len(invalid_configs)}\n")
            f.write(f"Configurations without general abilities data: {len(unknown_configs)}\n\n")
            
            # Best valid configurations
            if not valid_configs.empty:
                f.write("-" * 80 + "\n")
                f.write("RECOMMENDED CONFIGURATIONS\n")
                f.write("(Pass general abilities check AND maximize primary hallucination reduction)\n")
                f.write("-" * 80 + "\n\n")
                
                top_valid = valid_configs.nlargest(10, 'primary_absolute_reduction')
                for rank, (_, row) in enumerate(top_valid.iterrows(), 1):
                    f.write(f"{rank}. k={int(row['k'])}, alpha={row['alpha']:.2f}\n")
                    f.write(f"   Primary absolute reduction: {row['primary_absolute_reduction']:.2f} pp\n")
                    f.write(f"   Primary relative reduction: {row['primary_relative_reduction']:.2f}%\n")
                    f.write(f"   Primary steered rate: {row['primary_steered_rate']*100:.2f}%\n")
                    f.write(f"   General abilities absolute reduction: {row['secondary_absolute_reduction']:.2f} pp\n")
                    f.write(f"   General abilities relative reduction: {row['secondary_relative_reduction']:.2f}%\n")
                    f.write(f"   General abilities steered rate: {row['secondary_steered_rate']*100:.2f}%\n")
                    f.write(f"   General abilities check: ✓ PASSES\n\n")
            
            # Top failing configurations (for reference)
            if not invalid_configs.empty:
                f.write("-" * 80 + "\n")
                f.write("CONFIGURATIONS THAT FAIL GENERAL ABILITIES CHECK\n")
                f.write(f"(Secondary absolute increase > {secondary_threshold_pct} pp, i.e., too much degradation)\n")
                f.write("-" * 80 + "\n\n")
                
                # Sort failing by primary absolute reduction to show best-performing failures
                top_invalid = invalid_configs.nlargest(5, 'primary_absolute_reduction')
                for rank, (_, row) in enumerate(top_invalid.iterrows(), 1):
                    f.write(f"{rank}. k={int(row['k'])}, alpha={row['alpha']:.2f}\n")
                    f.write(f"   Primary absolute reduction: {row['primary_absolute_reduction']:.2f} pp\n")
                    f.write(f"   Primary relative reduction: {row['primary_relative_reduction']:.2f}%\n")
                    f.write(f"   General abilities absolute reduction: {row['secondary_absolute_reduction']:.2f} pp\n")
                    f.write(f"   General abilities relative reduction: {row['secondary_relative_reduction']:.2f}%\n")
                    f.write(f"   General abilities check: ✗ FAILS (degradation exceeds {secondary_threshold_pct} pp)\n\n")
            
            # Overall best recommendation
            if not valid_configs.empty:
                best = valid_configs.loc[valid_configs['primary_absolute_reduction'].idxmax()]
                f.write("\n" + "=" * 80 + "\n")
                f.write("★ OVERALL RECOMMENDATION ★\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Best configuration: k={int(best['k'])}, alpha={best['alpha']:.2f}\n")
                f.write(f"\nPrimary (hallucination reduction):\n")
                f.write(f"  - Baseline rate: {best['primary_baseline_rate']*100:.2f}%\n")
                f.write(f"  - Steered rate: {best['primary_steered_rate']*100:.2f}%\n")
                f.write(f"  - Absolute reduction: {best['primary_absolute_reduction']:.2f} pp\n")
                f.write(f"  - Relative reduction: {best['primary_relative_reduction']:.2f}%\n")
                f.write(f"\nGeneral Abilities (general abilities preservation):\n")
                f.write(f"  - Baseline rate: {best['secondary_baseline_rate']*100:.2f}%\n")
                f.write(f"  - Steered rate: {best['secondary_steered_rate']*100:.2f}%\n")
                f.write(f"  - Absolute reduction: {best['secondary_absolute_reduction']:.2f} pp\n")
                f.write(f"  - Relative reduction: {best['secondary_relative_reduction']:.2f}%\n")
                f.write(f"  - Relative reduction: {best['secondary_relative_reduction']:.2f}%\n")
                f.write(f"  - Max allowed rate ({secondary_threshold_pct}% threshold): {best['secondary_max_allowed_rate']*100:.2f}%\n")
                f.write(f"  - Status: ✓ PASSES (no significant degradation)\n")
    
    print(f"\n[INFO] Report saved: {report_path}")
    return report_path


def generate_csv_reports(
    primary: DirectoryAnalysis,
    secondary: Optional[DirectoryAnalysis],
    comparison_df: Optional[pd.DataFrame],
    output_dir: str
) -> List[str]:
    """Generate CSV reports for further analysis."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_files = []
    
    # Primary results CSV
    primary_data = []
    for (k, alpha), metrics in primary.results_by_config.items():
        primary_data.append({
            'k': k,
            'alpha': alpha,
            'baseline_rate': primary.baseline_hallucination_rate,
            'steered_rate': metrics['steered_halluc_rate'],
            'halluc_count': metrics['steered_halluc_count'],
            'relative_reduction': metrics['relative_reduction'],
            'absolute_reduction': metrics['absolute_reduction'],
            'valid_samples': metrics['total_valid_samples'],
        })
    
    primary_df = pd.DataFrame(primary_data)
    primary_df = primary_df.sort_values('absolute_reduction', ascending=False)
    primary_csv = os.path.join(output_dir, f"primary_results_{timestamp}.csv")
    primary_df.to_csv(primary_csv, index=False)
    csv_files.append(primary_csv)
    print(f"[INFO] Primary results CSV: {primary_csv}")
    
    # Secondary results CSV (if available)
    if secondary:
        secondary_data = []
        for (k, alpha), metrics in secondary.results_by_config.items():
            secondary_data.append({
                'k': k,
                'alpha': alpha,
                'baseline_rate': secondary.baseline_hallucination_rate,
                'steered_rate': metrics['steered_halluc_rate'],
                'halluc_count': metrics['steered_halluc_count'],
                'relative_reduction': metrics['relative_reduction'],
                'absolute_reduction': metrics['absolute_reduction'],
                'valid_samples': metrics['total_valid_samples'],
            })
        
        secondary_df = pd.DataFrame(secondary_data)
        secondary_csv = os.path.join(output_dir, f"general_abilities_results_{timestamp}.csv")
        secondary_df.to_csv(secondary_csv, index=False)
        csv_files.append(secondary_csv)
        print(f"[INFO] General abilities results CSV: {secondary_csv}")
    
    # Comparison CSV (if available)
    if comparison_df is not None and not comparison_df.empty:
        comparison_csv = os.path.join(output_dir, f"comparison_results_{timestamp}.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        csv_files.append(comparison_csv)
        print(f"[INFO] Comparison results CSV: {comparison_csv}")
    
    return csv_files


def generate_single_directory_report(
    analysis: DirectoryAnalysis,
    output_dir: str,
    is_primary: bool = True
) -> str:
    """Generate report for single directory analysis."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_type = "primary" if is_primary else "secondary"
    report_path = os.path.join(output_dir, f"{dir_type}_analysis_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("╔" + "═" * 78 + "╗\n")
        f.write("║" + " " * 78 + "║\n")
        title = "PRIMARY DIRECTORY ANALYSIS" if is_primary else "SECONDARY DIRECTORY ANALYSIS"
        f.write("║" + title.center(78) + "║\n")
        f.write("║" + f"Generated: {timestamp}".center(78) + "║\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("╚" + "═" * 78 + "╝\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DIRECTORY SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Directory: {analysis.output_dir}\n")
        f.write(f"Baseline directory: {analysis.baseline_dir}\n")
        f.write(f"Valid samples (no API failures across all experiments): {analysis.total_valid_samples}\n")
        f.write(f"Baseline hallucination rate: {analysis.baseline_hallucination_rate*100:.2f}%\n")
        f.write(f"Baseline hallucination count: {analysis.baseline_hallucination_count}\n")
        f.write(f"Total (k, alpha) configurations tested: {len(analysis.results_by_config)}\n\n")
        
        # All configurations sorted by relative reduction
        f.write("=" * 80 + "\n")
        f.write("ALL CONFIGURATIONS (sorted by absolute reduction)\n")
        f.write("=" * 80 + "\n\n")
        
        sorted_configs = sorted(
            analysis.results_by_config.items(),
            key=lambda x: x[1]['absolute_reduction'],
            reverse=True
        )
        
        f.write(f"{'Rank':<6} {'K':<8} {'Alpha':<10} {'Steered Rate':<15} {'Abs. Reduction':<18} {'Rel. Reduction':<15}\n")
        f.write("-" * 80 + "\n")
        
        for rank, ((k, alpha), metrics) in enumerate(sorted_configs, 1):
            f.write(f"{rank:<6} {k:<8} {alpha:<10.2f} {metrics['steered_halluc_rate']*100:>12.2f}% {metrics['absolute_reduction']:>15.2f} pp {metrics['relative_reduction']:>12.2f}%\n")
        
        # Best configuration
        if sorted_configs:
            best_config, best_metrics = sorted_configs[0]
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("BEST CONFIGURATION (by absolute reduction)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"k = {best_config[0]}\n")
            f.write(f"alpha = {best_config[1]}\n")
            f.write(f"Baseline hallucination rate: {analysis.baseline_hallucination_rate*100:.2f}%\n")
            f.write(f"Steered hallucination rate: {best_metrics['steered_halluc_rate']*100:.2f}%\n")
            f.write(f"Absolute reduction: {best_metrics['absolute_reduction']:.2f} pp\n")
            f.write(f"Relative reduction: {best_metrics['relative_reduction']:.2f}%\n")
    
    print(f"[INFO] Report saved: {report_path}")
    return report_path


def generate_dual_secondary_report(
    primary: DirectoryAnalysis,
    secondaries: List[DirectoryAnalysis],
    secondary_labels: List[str],
    secondary_thresholds: List[float],
    comparison_df: Optional[pd.DataFrame],
    output_dir: str
) -> str:
    """
    Generate comprehensive report for dual secondary analysis with per-secondary failure details.
    
    Reports which specific secondaries caused a config to fail, enabling diagnostic analysis.
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"cross_analysis_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("╔" + "═" * 78 + "╗\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("║" + "CROSS-EXPERIMENT ANALYSIS REPORT (DUAL SECONDARIES)".center(78) + "║\n")
        f.write("║" + f"Generated: {timestamp}".center(78) + "║\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("╚" + "═" * 78 + "╝\n\n")
        
        # Primary directory summary
        f.write("=" * 80 + "\n")
        f.write("PRIMARY DIRECTORY (Hallucination Reduction Focus)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Directory: {primary.output_dir}\n")
        f.write(f"Baseline directory: {primary.baseline_dir}\n")
        f.write(f"Valid samples (no API failures): {primary.total_valid_samples}\n")
        f.write(f"Baseline hallucination rate: {primary.baseline_hallucination_rate*100:.2f}%\n")
        f.write(f"Baseline hallucination count: {primary.baseline_hallucination_count}\n")
        f.write(f"Total configurations tested: {len(primary.results_by_config)}\n\n")
        
        # Best configs in primary
        f.write("-" * 80 + "\n")
        f.write("TOP 10 PRIMARY CONFIGURATIONS (by absolute reduction)\n")
        f.write("-" * 80 + "\n")
        
        sorted_configs = sorted(
            primary.results_by_config.items(),
            key=lambda x: x[1]['absolute_reduction'],
            reverse=True
        )[:10]
        
        for rank, ((k, alpha), metrics) in enumerate(sorted_configs, 1):
            f.write(f"\n{rank}. k={k}, alpha={alpha:.2f}\n")
            f.write(f"   Steered rate: {metrics['steered_halluc_rate']*100:.2f}%\n")
            f.write(f"   Absolute reduction: {metrics['absolute_reduction']:.2f} pp\n")
            f.write(f"   Relative reduction: {metrics['relative_reduction']:.2f}%\n")
        
        # Secondary directory summaries
        for sec_idx, (secondary, label, threshold) in enumerate(zip(secondaries, secondary_labels, secondary_thresholds), 1):
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"SECONDARY DIRECTORY {sec_idx} ({label})\n")
            f.write("=" * 80 + "\n")
            f.write(f"Directory: {secondary.output_dir}\n")
            f.write(f"Baseline directory: {secondary.baseline_dir}\n")
            f.write(f"Valid samples (no API failures): {secondary.total_valid_samples}\n")
            f.write(f"Baseline hallucination rate: {secondary.baseline_hallucination_rate*100:.2f}%\n")
            f.write(f"Baseline hallucination count: {secondary.baseline_hallucination_count}\n")
            f.write(f"Total configurations tested: {len(secondary.results_by_config)}\n")
            f.write(f"\nDegradation threshold: {threshold} percentage points (pp) absolute increase allowed\n")
            f.write(f"  (Config passes if: steered_rate <= {secondary.baseline_hallucination_rate*100:.2f}% + {threshold:.2f} pp)\n")
            f.write(f"  (Equivalently: hallucination rate increase <= {threshold} pp absolute)\n")
        
        # Cross-comparison results
        if comparison_df is not None and not comparison_df.empty:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("CROSS-DIRECTORY COMPARISON RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Count valid/invalid configs
            valid_configs = comparison_df[comparison_df['valid_config'] == True]
            invalid_configs = comparison_df[comparison_df['valid_config'] == False]
            unknown_configs = comparison_df[comparison_df['valid_config'].isna()]
            
            f.write(f"Configurations passing ALL secondary checks: {len(valid_configs)}\n")
            f.write(f"Configurations failing at least one secondary check: {len(invalid_configs)}\n")
            f.write(f"Configurations without secondary data: {len(unknown_configs)}\n\n")
            
            # Best valid configurations (passing both secondaries)
            if not valid_configs.empty:
                f.write("-" * 80 + "\n")
                f.write("★ RECOMMENDED CONFIGURATIONS ★\n")
                f.write("(Pass ALL secondary checks AND maximize primary hallucination reduction)\n")
                f.write("-" * 80 + "\n\n")
                
                top_valid = valid_configs.nlargest(10, 'primary_absolute_reduction')
                for rank, (_, row) in enumerate(top_valid.iterrows(), 1):
                    f.write(f"{rank}. k={int(row['k'])}, alpha={row['alpha']:.2f}\n")
                    f.write(f"   Primary: {row['primary_absolute_reduction']:.2f} pp reduction ({row['primary_relative_reduction']:.2f}%, steered: {row['primary_steered_rate']*100:.2f}%)\n")
                    
                    # Show status for each secondary
                    for sec_idx in range(len(secondaries)):
                        sec_num = sec_idx + 1
                        abs_red = row[f'secondary_{sec_num}_absolute_reduction']
                        rel_red = row[f'secondary_{sec_num}_relative_reduction']
                        label = row[f'secondary_{sec_num}_label']
                        f.write(f"   {label}: {abs_red:.2f} pp reduction ({rel_red:.2f}%, steered: {row[f'secondary_{sec_num}_steered_rate']*100:.2f}%) ✓ PASSES\n")
                    f.write("\n")
            
            # Configurations failing secondary checks with detailed reasons
            if not invalid_configs.empty:
                f.write("-" * 80 + "\n")
                f.write("CONFIGURATIONS FAILING SECONDARY CHECKS\n")
                f.write("(Sorted by primary absolute reduction - shows best performers that fail secondaries)\n")
                f.write("-" * 80 + "\n\n")
                
                # Sort by primary absolute reduction to show best failures
                top_invalid = invalid_configs.nlargest(15, 'primary_absolute_reduction')
                for rank, (_, row) in enumerate(top_invalid.iterrows(), 1):
                    f.write(f"{rank}. k={int(row['k'])}, alpha={row['alpha']:.2f}\n")
                    f.write(f"   Primary: {row['primary_absolute_reduction']:.2f} pp reduction ({row['primary_relative_reduction']:.2f}%, steered: {row['primary_steered_rate']*100:.2f}%)\n")
                    
                    # Show which secondaries failed
                    failures = []
                    for sec_idx in range(len(secondaries)):
                        sec_num = sec_idx + 1
                        passes = row[f'secondary_{sec_num}_passes']
                        fail_reason = row[f'secondary_{sec_num}_fail_reason']
                        
                        if passes == False:
                            failures.append(fail_reason)
                        else:
                            label = row[f'secondary_{sec_num}_label']
                            abs_red = row[f'secondary_{sec_num}_absolute_reduction']
                            rel_red = row[f'secondary_{sec_num}_relative_reduction']
                            f.write(f"   {label}: {abs_red:.2f} pp reduction ({rel_red:.2f}%) ✓ PASSES\n")
                    
                    for failure_reason in failures:
                        f.write(f"   ✗ FAILS: {failure_reason}\n")
                    
                    f.write("\n")
            
            # Overall best recommendation
            if not valid_configs.empty:
                best = valid_configs.loc[valid_configs['primary_absolute_reduction'].idxmax()]
                f.write("\n" + "=" * 80 + "\n")
                f.write("★★★ OVERALL BEST RECOMMENDATION ★★★\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Configuration: k={int(best['k'])}, alpha={best['alpha']:.2f}\n\n")
                
                f.write(f"Primary (Hallucination Reduction):\n")
                f.write(f"  - Baseline rate: {best['primary_baseline_rate']*100:.2f}%\n")
                f.write(f"  - Steered rate: {best['primary_steered_rate']*100:.2f}%\n")
                f.write(f"  - Absolute reduction: {best['primary_absolute_reduction']:.2f} pp\n")
                f.write(f"  - Relative reduction: {best['primary_relative_reduction']:.2f}%\n\n")
                
                for sec_idx in range(len(secondaries)):
                    sec_num = sec_idx + 1
                    label = best[f'secondary_{sec_num}_label']
                    f.write(f"{label}:\n")
                    f.write(f"  - Baseline rate: {best[f'secondary_{sec_num}_baseline_rate']*100:.2f}%\n")
                    f.write(f"  - Steered rate: {best[f'secondary_{sec_num}_steered_rate']*100:.2f}%\n")
                    f.write(f"  - Absolute reduction: {best[f'secondary_{sec_num}_absolute_reduction']:.2f} pp\n")
                    f.write(f"  - Relative reduction: {best[f'secondary_{sec_num}_relative_reduction']:.2f}%\n")
                    f.write(f"  - Status: ✓ PASSES (within {secondary_thresholds[sec_idx]} pp threshold)\n\n")
            else:
                f.write("\n" + "=" * 80 + "\n")
                f.write("⚠️  NO CONFIGURATIONS PASS ALL SECONDARY CHECKS\n")
                f.write("=" * 80 + "\n\n")
                f.write("Consider relaxing secondary thresholds or exploring configurations that pass\n")
                f.write("individual secondaries but not both simultaneously.\n")
    
    print(f"\n[INFO] Dual secondary report saved: {report_path}")
    return report_path


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-experiment analysis for ITI steering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single directory (primary only)
  python -m pipeline.steer.ITI.cross_experiment_analysis \\
      --primary-dir ./data/ITI/steering_experiment_more_eval/round1/evalonnqswap \\
      --output-dir ./data/ITI/steering_experiment_more_eval/round1/cross_analysis

  # Compare primary and secondary directory (backward compatible)
  python -m pipeline.steer.ITI.cross_experiment_analysis \\
      --primary-dir ./data/ITI/steering_experiment_more_eval/round1/evalonnqswap \\
      --secondary-dir ./data/ITI/steering_experiment_more_eval/round1/evalonmmlu \\
      --output-dir ./data/ITI/steering_experiment_more_eval/round1/cross_analysis \\
      --secondary-threshold 1.0

  # Compare with TWO secondary directories (dual validation)
  python -m pipeline.steer.ITI.cross_experiment_analysis \\
      --primary-dir ./data/ITI/steering_experiment_more_eval/round1/evalonnqswap \\
      --secondary-dir-1 ./data/ITI/steering_experiment_more_eval/round1/evalonmmlu \\
      --secondary-dir-2 ./data/ITI/steering_experiment_more_eval/round1/evalonsquad \\
      --output-dir ./data/ITI/steering_experiment_more_eval/round1/cross_analysis \\
      --secondary-threshold-1 1.0 \\
      --secondary-threshold-2 1.0
        """
    )
    
    parser.add_argument(
        '--primary-dir', type=str, required=True,
        help='Primary output directory (focus: hallucination reduction)'
    )
    parser.add_argument(
        '--secondary-dir', '--secondary-dir-1', type=str, default=None, dest='secondary_dir_1',
        help='First secondary output directory (e.g., general abilities check). Alias: --secondary-dir'
    )
    parser.add_argument(
        '--secondary-dir-2', type=str, default=None, dest='secondary_dir_2',
        help='Second secondary output directory (e.g., knowledge retention check)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./cross_analysis_results',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--secondary-threshold', '--secondary-threshold-1', type=float, default=1.0, dest='secondary_threshold_1',
        help='Threshold for first secondary (max allowed absolute hallucination increase in percentage points, default: 1.0 pp). Alias: --secondary-threshold'
    )
    parser.add_argument(
        '--secondary-threshold-2', type=float, default=1.0, dest='secondary_threshold_2',
        help='Threshold for second secondary (max allowed absolute hallucination increase in percentage points, default: 1.0 pp)'
    )
    parser.add_argument(
        '--secondary-label-1', type=str, default='General Abilities (MMLU)',
        help='Label for first secondary directory (default: "General Abilities")'
    )
    parser.add_argument(
        '--secondary-label-2', type=str, default='General Abilities (HellaSwag)',
        help='Label for second secondary directory (default: "General Abilities")'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating visualization plots'
    )
    
    args = parser.parse_args()
    
    # Create timestamp for run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory and run subdirectory
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, f"RUN_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CROSS-EXPERIMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Primary directory: {args.primary_dir}")
    print(f"Secondary dir 1: {args.secondary_dir_1 or 'None'} ({args.secondary_label_1})")
    print(f"Secondary dir 2: {args.secondary_dir_2 or 'None'} ({args.secondary_label_2})")
    print(f"Output directory: {args.output_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Secondary 1 threshold: {args.secondary_threshold_1}%")
    print(f"Secondary 2 threshold: {args.secondary_threshold_2}%")
    
    # Analyze primary directory
    primary_analysis = analyze_output_directory(args.primary_dir)
    
    if primary_analysis is None:
        print("\n[ERROR] Failed to analyze primary directory. Exiting.")
        sys.exit(1)
    
    # Determine whether to use single or dual secondary mode
    secondaries_list = []
    secondary_labels = []
    secondary_thresholds = []
    comparison_df = None
    
    # Collect secondary directories if provided
    if args.secondary_dir_1:
        secondaries_list.append(args.secondary_dir_1)
        secondary_labels.append(args.secondary_label_1)
        secondary_thresholds.append(args.secondary_threshold_1)
    
    if args.secondary_dir_2:
        secondaries_list.append(args.secondary_dir_2)
        secondary_labels.append(args.secondary_label_2)
        secondary_thresholds.append(args.secondary_threshold_2)
    
    # Analyze secondary directories
    secondary_analyses = []
    
    for sec_dir, sec_label in zip(secondaries_list, secondary_labels):
        print(f"\n[INFO] Analyzing secondary directory: {sec_label}")
        sec_analysis = analyze_output_directory(sec_dir)
        
        if sec_analysis is None:
            print(f"\n[WARNING] Failed to analyze secondary directory ({sec_label}). Proceeding with remaining secondaries.")
            secondaries_list.remove(sec_dir)
            secondary_labels.remove(sec_label)
            secondary_thresholds.pop(secondary_analyses.__len__())
        else:
            secondary_analyses.append(sec_analysis)
    
    # Perform comparison if at least one secondary was analyzed
    if secondary_analyses:
        if len(secondary_analyses) == 2:
            # Dual secondary mode
            comparison_df = compare_primary_with_secondaries(
                primary_analysis,
                secondary_analyses,
                secondary_thresholds,
                secondary_labels
            )
        else:
            # Single secondary mode (backward compatible)
            comparison_df = compare_primary_secondary(
                primary_analysis,
                secondary_analyses[0],
                secondary_thresholds[0]
            )
    
    # Generate reports
    print(f"\n{'='*80}")
    print("GENERATING REPORTS")
    print(f"{'='*80}")
    
    if secondary_analyses and comparison_df is not None:
        if len(secondary_analyses) == 2:
            # Dual secondary report
            generate_dual_secondary_report(
                primary_analysis,
                secondary_analyses,
                secondary_labels,
                secondary_thresholds,
                comparison_df,
                run_dir
            )
        else:
            # Single secondary report (backward compatible)
            generate_summary_report(
                primary_analysis,
                secondary_analyses[0],
                comparison_df,
                run_dir,
                secondary_thresholds[0]
            )
    else:
        # Single directory report
        generate_single_directory_report(primary_analysis, run_dir, is_primary=True)
    
    # Generate CSV files
    if len(secondary_analyses) == 1:
        generate_csv_reports(
            primary_analysis,
            secondary_analyses[0],
            comparison_df,
            run_dir
        )
    elif len(secondary_analyses) == 2:
        # Multi-secondary CSV generation
        timestamp_csv = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Primary CSV
        primary_data = []
        for (k, alpha), metrics in primary_analysis.results_by_config.items():
            primary_data.append({
                'k': k,
                'alpha': alpha,
                'baseline_rate': primary_analysis.baseline_hallucination_rate,
                'steered_rate': metrics['steered_halluc_rate'],
                'halluc_count': metrics['steered_halluc_count'],
                'relative_reduction': metrics['relative_reduction'],
                'absolute_reduction': metrics['absolute_reduction'],
                'valid_samples': metrics['total_valid_samples'],
            })
        
        primary_df = pd.DataFrame(primary_data)
        primary_df = primary_df.sort_values('relative_reduction', ascending=False)
        primary_csv = os.path.join(run_dir, f"primary_results_{timestamp_csv}.csv")
        primary_df.to_csv(primary_csv, index=False)
        print(f"[INFO] Primary results CSV: {primary_csv}")
        
        # Comparison CSV
        if comparison_df is not None:
            comparison_csv = os.path.join(run_dir, f"comparison_results_{timestamp_csv}.csv")
            comparison_df.to_csv(comparison_csv, index=False)
            print(f"[INFO] Comparison results CSV: {comparison_csv}")
    
    # Generate visualizations
    if not args.no_plots:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Per-directory plots for primary
        if len(secondary_analyses) == 2:
            # For dual secondaries, use the new visualizations
            generate_per_directory_plots(
                primary_analysis, run_dir, "Primary", timestamp,
                secondary_analysis=None
            )
            
            # Generate plots for each secondary directory
            for i, (sec_analysis, sec_label) in enumerate(zip(secondary_analyses, secondary_labels), 1):
                print(f"\n[INFO] Generating {sec_label} directory plots...")
                generate_per_directory_plots(
                    sec_analysis, run_dir, f"Secondary_{i}_{sec_label.replace(' ', '_').replace('(', '').replace(')', '')}", timestamp,
                    secondary_analysis=None
                )
            
            # Generate triple heatmap
            generate_triple_heatmap(
                primary_analysis,
                secondary_analyses,
                secondary_labels,
                run_dir,
                timestamp
            )
            
            # Generate dual scatter plots
            if comparison_df is not None:
                generate_dual_secondary_scatter_plots(
                    primary_analysis,
                    secondary_analyses,
                    secondary_labels,
                    comparison_df,
                    run_dir,
                    secondary_thresholds,
                    timestamp
                )
        else:
            # Single secondary mode (original visualizations)
            if secondary_analyses:
                generate_per_directory_plots(
                    primary_analysis, run_dir, "Primary", timestamp, 
                    secondary_analysis=secondary_analyses[0],
                    secondary_threshold_pct=secondary_thresholds[0]
                )
                
                generate_per_directory_plots(
                    secondary_analyses[0], run_dir, "Secondary", timestamp,
                    secondary_threshold_pct=secondary_thresholds[0]
                )
                
                # Cross-directory plots
                if comparison_df is not None:
                    generate_cross_directory_plots(
                        primary_analysis,
                        secondary_analyses[0],
                        comparison_df,
                        run_dir,
                        secondary_thresholds[0],
                        timestamp
                    )
            else:
                # No secondaries, just primary plots
                generate_per_directory_plots(
                    primary_analysis, run_dir, "Primary", timestamp
                )
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {run_dir}")
    
    # Quick summary
    print(f"\nPrimary Directory Summary:")
    print(f"  Valid samples: {primary_analysis.total_valid_samples}")
    print(f"  Baseline rate: {primary_analysis.baseline_hallucination_rate*100:.2f}%")
    print(f"  Configurations tested: {len(primary_analysis.results_by_config)}")
    
    # Find best primary config
    best_primary = max(
        primary_analysis.results_by_config.items(),
        key=lambda x: x[1]['absolute_reduction']
    )
    print(f"  Best config: k={best_primary[0][0]}, alpha={best_primary[0][1]}")
    print(f"  Best reduction: {best_primary[1]['absolute_reduction']:.2f} pp (absolute)")
    
    # Secondary summaries
    for i, sec_analysis in enumerate(secondary_analyses, 1):
        print(f"\nSecondary {i} ({secondary_labels[i-1]}) Summary:")
        print(f"  Valid samples: {sec_analysis.total_valid_samples}")
        print(f"  Baseline rate: {sec_analysis.baseline_hallucination_rate*100:.2f}%")
        print(f"  Configurations tested: {len(sec_analysis.results_by_config)}")
    
    # Comparison results
    if comparison_df is not None and not comparison_df.empty:
        valid_configs = comparison_df[comparison_df['valid_config'] == True]
        invalid_configs = comparison_df[comparison_df['valid_config'] == False]
        
        if len(secondary_analyses) == 2:
            print(f"\nDual-Secondary Comparison Summary:")
            print(f"  Configs passing BOTH checks: {len(valid_configs)}")
            print(f"  Configs failing at least one: {len(invalid_configs)}")
        else:
            print(f"\nCross-Comparison Summary:")
            print(f"  Configs passing secondary check: {len(valid_configs)}")
            print(f"  Configs failing secondary check: {len(invalid_configs)}")
        
        if not valid_configs.empty:
            best_valid = valid_configs.iloc[0]
            print(f"\n  ★ RECOMMENDED: k={int(best_valid['k'])}, alpha={best_valid['alpha']:.2f}")
            print(f"    Primary relative reduction: {best_valid['primary_relative_reduction']:.2f}%")
            
            if len(secondary_analyses) == 2:
                for i in range(len(secondary_analyses)):
                    sec_num = i + 1
                    print(f"    {secondary_labels[i]} relative reduction: {best_valid[f'secondary_{sec_num}_relative_reduction']:.2f}%")
            else:
                print(f"    Secondary relative reduction: {best_valid['secondary_relative_reduction']:.2f}%")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()