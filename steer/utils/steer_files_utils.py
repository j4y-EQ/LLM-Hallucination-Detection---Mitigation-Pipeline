"""
Analysis and output generation module for ITI steering evaluation.

Handles:
- Improvement analysis computation
- Visualization generation
- Results saving (JSON, text files)
- Best strength analysis
- Baseline data persistence (save/load)
"""

import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from typing import Dict, List, Tuple, Any, Optional
from .dataset_formats import VALID_DATASET_FORMATS, MCQ_FORMATS

from .steer_common_utils import parse_choices


# ================================================================
# HELPER FUNCTIONS FOR DETAILED TEXT RESULTS
# ================================================================

def _format_choices_for_display(choices: Any) -> str:
    """
    Format choices dict for display in text output.
    Uses EXACT same formatting logic as make_qa_prompt() in steer_common_utils.
    
    Args:
        choices: Dict with 'label' and 'text' keys, string repr, or None
        
    Returns:
        Formatted string of choices exactly as they appear in the prompt:
        "A. Option text\nB. Option text\n..." or empty string if None
    """
    if choices is None:
        return ""
    
    # Parse choices if it's a string representation
    if isinstance(choices, str):
        choices = parse_choices(choices)
        if choices is None:
            return ""
    
    # Format as dict - match exactly with make_qa_prompt()
    if isinstance(choices, dict):
        labels = choices.get('label', [])
        texts = choices.get('text', [])
        
        # Handle numpy arrays (convert to list if needed)
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Only format if we have valid labels and texts
        if not labels or not texts or len(labels) != len(texts):
            return ""
        
        # Format exactly as in make_qa_prompt: "{label}. {text}\n"
        choices_str = ""
        for label, text in zip(labels, texts):
            choices_str += f"{label}. {text}\n"
        
        # Remove trailing newline to avoid extra blank lines in output
        return choices_str.rstrip('\n')
    
    return ""


def _reconstruct_and_validate_prompt_components(result: Dict, baseline_dataset_format: str, sample_idx: int, logger) -> Dict:
    """
    Reconstruct and validate prompt components from result dict.
    
    Args:
        result: Single result dict from all_results list
        baseline_dataset_format: Format of dataset ('mcq', 'non_mcq', 'mmlu')
        sample_idx: Sample index for error reporting
        logger: Logger instance
        
    Returns:
        Dict with validated components:
        {
            'context': str,
            'question': str,
            'answerKey': str or None,
            'answer_text': str,
            'choices_str': str (formatted choices or empty)
        }
        
    Raises:
        ValueError: If MCQ/MMLU dataset missing choices for a sample
    """
    components = {
        'context': result.get('context', ''),
        'question': result.get('question', ''),
        'answer_text': result.get('answer_text', ''),
        'answerKey': None,
        'choices_str': ''
    }
    
    # Validate based on dataset format
    if baseline_dataset_format in ['mcq', 'mmlu']:
        choices = result.get('choices')
        answerKey = result.get('answerKey', '')
        
        # Strict validation: MCQ/MMLU must have choices
        if choices is None or (isinstance(choices, dict) and not choices):
            logger.error(f"Sample {sample_idx}: MCQ/MMLU dataset must have 'choices' field. Found: {choices}")
            raise ValueError(f"Sample {sample_idx}: MCQ/MMLU dataset must have non-empty 'choices' field. Data integrity error. Found: {choices}")
        
        # Format choices for display
        components['choices_str'] = _format_choices_for_display(choices)
        components['answerKey'] = answerKey if answerKey else None
    
    # For non-MCQ, skip answerKey and choices
    
    return components

def create_improvement_analysis(all_results_by_strength, run_dir, logger):
    """
    Create detailed improvement analysis comparing baseline vs each steering strength.
    
    Returns analysis data for visualization and reporting.
    """
    analysis_data = {
        'baseline_rate': None,
        'steering_rates': {},
        'improvements': {},
        'details': []
    }
    
    # Get baseline hallucination rate
    # All strengths share the same baseline results, so we pick the first one
    baseline_results = all_results_by_strength[min(all_results_by_strength.keys())]
    baseline_halluc_count = sum(1 for r in baseline_results if r['baseline_hallucination'] == 1)
    baseline_api_failures = sum(1 for r in baseline_results if r['baseline_hallucination'] == 2)
    baseline_valid_count = len(baseline_results) - baseline_api_failures
    baseline_rate = baseline_halluc_count / baseline_valid_count if baseline_valid_count > 0 else 0
    analysis_data['baseline_rate'] = baseline_rate
    
    logger.info(f"\nBaseline hallucination rate: {baseline_rate*100:.2f}% ({baseline_halluc_count}/{baseline_valid_count} valid samples)")
    if baseline_api_failures > 0:
        logger.warning(f"  - Baseline API failures: {baseline_api_failures}")
    
    # Calculate rates and improvements for each strength
    for steering_strength in sorted(all_results_by_strength.keys()):
        results = all_results_by_strength[steering_strength]
        steered_halluc_count = sum(1 for r in results if r['steered_hallucination'] == 1)
        steered_api_failures = sum(1 for r in results if r['steered_hallucination'] == 2)
        steered_valid_count = len(results) - steered_api_failures
        steered_rate = steered_halluc_count / steered_valid_count if steered_valid_count > 0 else 0
        
        analysis_data['steering_rates'][steering_strength] = steered_rate
        
        # Calculate improvements
        absolute_reduction = baseline_rate - steered_rate
        relative_reduction = (baseline_rate - steered_rate) / baseline_rate * 100 if baseline_rate > 0 else 0
        
        analysis_data['improvements'][steering_strength] = {
            'absolute_reduction': absolute_reduction,
            'relative_reduction': relative_reduction,
            'halluc_count': steered_halluc_count,
            'halluc_rate': steered_rate,
            'baseline_api_failures': baseline_api_failures,
            'steered_api_failures': steered_api_failures,
        }
        
        logger.info(f"\nSteering strength α={steering_strength}:")
        logger.info(f"  Hallucination rate: {steered_rate*100:.2f}% ({steered_halluc_count}/{steered_valid_count} valid samples)")
        logger.info(f"  Absolute reduction: {absolute_reduction*100:.2f} percentage points")
        logger.info(f"  Relative reduction: {relative_reduction:.2f}%")
        if steered_api_failures > 0:
            logger.warning(f"  - Steered API failures: {steered_api_failures}")
    
    return analysis_data

def create_improvement_visualization(analysis_data, run_dir, logger):
    """Create and save visualization of hallucination rates across steering strengths."""
    
    # Prepare data for plotting
    strengths = sorted(analysis_data['improvements'].keys())
    baseline_rate = analysis_data['baseline_rate']
    steered_rates = [analysis_data['steering_rates'][s] for s in strengths]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Hallucination rates
    ax1.axhline(y=baseline_rate*100, color='red', linestyle='--', linewidth=2.5, label=f'Baseline ({baseline_rate*100:.1f}%)', marker='o', markersize=8)
    ax1.plot(strengths, [r*100 for r in steered_rates], color='green', linestyle='-', linewidth=2.5, marker='s', markersize=8, label='Steered (ITI)')
    
    ax1.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hallucination Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Hallucination Rate vs Steering Strength', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Absolute improvement
    improvements = [baseline_rate - r for r in steered_rates]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(range(len(strengths)), [imp*100 for imp in improvements], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (% points)', fontsize=12, fontweight='bold')
    ax2.set_title('Hallucination Reduction vs Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(strengths)))
    ax2.set_xticklabels(strengths)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    graph_file = os.path.join(run_dir, "improvement_analysis.png")
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {graph_file}")
    
    plt.close()
    return graph_file


def create_independent_hallucination_analysis(all_results_by_strength, run_dir, logger):
    """
    Create independent hallucination analysis for each steering strength.
    
    This function analyzes hallucination scores independently for each steering strength
    WITHOUT comparing to baseline. It identifies the tentative best strength by
    maximizing the count of non-hallucinated responses.
    
    Args:
        all_results_by_strength: Dict[float, List[Dict]] - Results keyed by steering strength
                                Each result dict contains hallucination_score field
        run_dir: Directory to save analysis outputs (CSV files)
        logger: Logger instance for reporting
    
    Returns:
        Dict with structure:
            {
                'steering_stats': {strength: {halluc_count, no_halluc_count, api_fail_count, halluc_rate}},
                'tentative_best_strength': float or None,
                'tentative_best_count': int,
                'details': [list of per-strength summaries]
            }
    
    Output Files:
        - independent_hallucination_analysis.csv: Per-strength statistics table
    
    Note:
        - Hallucination score values: 0 (no halluc), 1 (halluc), 2 (API failure)
        - Hallucination rate excludes API failures from denominator
        - Best strength selection prioritizes non-hallucination count, not rate
    """
    analysis_data = {
        'steering_stats': {},
        'tentative_best_strength': None,
        'tentative_best_count': 0,
        'details': []
    }
    
    logger.info(f"\nAnalyzing independent hallucination statistics per steering strength...")
    
    # Calculate statistics for each strength
    for steering_strength in sorted(all_results_by_strength.keys()):
        results = all_results_by_strength[steering_strength]
        
        # Count by score: 0 (no hallucination), 1 (hallucination), 2 (API failure)
        count_no_halluc = sum(1 for r in results if r['steered_hallucination'] == 0)
        count_halluc = sum(1 for r in results if r['steered_hallucination'] == 1)
        count_api_failure = sum(1 for r in results if r['steered_hallucination'] == 2)
        
        total_valid = count_no_halluc + count_halluc
        halluc_rate = count_halluc / total_valid if total_valid > 0 else 0
        
        analysis_data['steering_stats'][steering_strength] = {
            'no_hallucination_count': count_no_halluc,
            'hallucination_count': count_halluc,
            'api_failure_count': count_api_failure,
            'total_valid_samples': total_valid,
            'hallucination_rate': halluc_rate,
        }
        
        logger.info(f"\nSteering strength α={steering_strength}:")
        logger.info(f"  No Hallucinations: {count_no_halluc}")
        logger.info(f"  Hallucinations: {count_halluc}")
        logger.info(f"  API Failures: {count_api_failure}")
        logger.info(f"  Hallucination Rate: {halluc_rate*100:.2f}%")
        
        # Track tentative best by highest non-hallucination count
        if count_no_halluc > analysis_data['tentative_best_count']:
            analysis_data['tentative_best_count'] = count_no_halluc
            analysis_data['tentative_best_strength'] = steering_strength
    
    # Log tentative best
    if analysis_data['tentative_best_strength'] is not None:
        best_stats = analysis_data['steering_stats'][analysis_data['tentative_best_strength']]
        logger.info(f"\n{'='*80}")
        logger.info(f"TENTATIVE BEST STEERING STRENGTH: α={analysis_data['tentative_best_strength']}")
        logger.info(f"{'='*80}")
        logger.info(f"Non-hallucinations: {best_stats['no_hallucination_count']} (highest)")
        logger.info(f"Hallucinations: {best_stats['hallucination_count']}")
        logger.info(f"API Failures: {best_stats['api_failure_count']}")
        logger.info(f"Hallucination Rate: {best_stats['hallucination_rate']*100:.2f}%")
    
    return analysis_data


def create_independent_hallucination_visualization(analysis_data, run_dir, logger):
    """
    Create bar chart visualization showing non-hallucination counts per steering strength.
    Highlights TENTATIVE best steering strength.
    """
    
    # Prepare data for plotting
    strengths = sorted(analysis_data['steering_stats'].keys())
    no_halluc_counts = [analysis_data['steering_stats'][s]['no_hallucination_count'] for s in strengths]
    halluc_counts = [analysis_data['steering_stats'][s]['hallucination_count'] for s in strengths]
    api_fail_counts = [analysis_data['steering_stats'][s]['api_failure_count'] for s in strengths]
    
    tentative_best = analysis_data['tentative_best_strength']
    
    # Create figure with bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar positions
    x_pos = np.arange(len(strengths))
    bar_width = 0.6
    
    # Colors for each bar
    colors = []
    for strength in strengths:
        if strength == tentative_best:
            colors.append('#2ecc71')  # Green for tentative best
        else:
            colors.append('#3498db')  # Blue for others
    
    # Create bars for non-hallucination counts
    bars = ax.bar(x_pos, no_halluc_counts, bar_width, label='Non-Hallucinations', 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    # Add star marker for tentative best
    if tentative_best is not None:
        best_idx = list(strengths).index(tentative_best)
        best_count = no_halluc_counts[best_idx]
        ax.plot(best_idx, best_count + 1, marker='*', markersize=25, color='gold', 
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, no_halluc_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, count + 1, str(count), 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Non-Hallucination Count', fontsize=12, fontweight='bold')
    ax.set_title('Non-Hallucination Counts per Steering Strength\n(★ marks TENTATIVE best)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strengths)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    if tentative_best is not None:
        ax.text(0.98, 0.97, f'★ TENTATIVE BEST: α={tentative_best}', 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    
    # Save figure
    graph_file = os.path.join(run_dir, "independent_hallucination_analysis.png")
    plt.savefig(graph_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved independent hallucination visualization: {graph_file}")
    
    plt.close()
    return graph_file


def save_tentative_best_strength(analysis_data: Dict, run_dir: str, logger) -> str:
    """Save tentative best steering strength (highest non-hallucination count) to JSON file."""
    
    tentative_best_strength = analysis_data['tentative_best_strength']
    
    if tentative_best_strength is None:
        logger.warning("No tentative best strength identified")
        return None
    
    best_stats = analysis_data['steering_stats'][tentative_best_strength]
    
    best_strength_data = {
        'tentative_best_strength': float(tentative_best_strength) if isinstance(tentative_best_strength, (int, float)) else tentative_best_strength,
        'criteria': 'Highest non-hallucination count',
        'non_hallucination_count': best_stats['no_hallucination_count'],
        'hallucination_count': best_stats['hallucination_count'],
        'api_failure_count': best_stats['api_failure_count'],
        'total_valid_samples': best_stats['total_valid_samples'],
        'hallucination_rate': float(best_stats['hallucination_rate']),
    }
    
    # Save to JSON file
    best_strength_file = os.path.join(run_dir, "tentative_best_strength.json")
    with open(best_strength_file, 'w') as f:
        json.dump(best_strength_data, f, indent=2)
    
    logger.info(f"✓ Saved tentative best strength: {best_strength_file}")
    return best_strength_file



def save_best_strength_sorted_results(all_results_by_strength, analysis_data, run_dir, logger):
    """
    Save all rows from the best steering strength, sorted by:
    1. Cases where baseline hallucinated and steered did not (top priority)
    2. All other cases
    """
    # Find best steering strength (maximum relative reduction - primary ranking metric)
    # Note: Both absolute_reduction (percentage points) and relative_reduction (%) are calculated
    # but relative_reduction is used for determining best parameter combinations
    best_strength = max(
        analysis_data['improvements'], 
        key=lambda s: analysis_data['improvements'][s]['relative_reduction']
    )
    best_result_summary = analysis_data['improvements'][best_strength]
    baseline_rate = analysis_data['baseline_rate']

    logger.info(f"\nGenerating best strength sorted results for α={best_strength}...")
    
    all_results = all_results_by_strength[best_strength]
    
    # Separate results into two categories
    halluc_fixed = []  # baseline_hallucination=1, steered_hallucination=0
    other_results = []  # all other cases
    
    for r in all_results:
        if r['baseline_hallucination'] == 1 and r['steered_hallucination'] == 0:
            halluc_fixed.append(r)
        else:
            other_results.append(r)
    
    # Combine: fixed cases first, then others
    sorted_results = halluc_fixed + other_results
    
    logger.info(f"  Hallucination fixed (baseline→no halluc): {len(halluc_fixed)}")
    logger.info(f"  Other cases: {len(other_results)}")
    logger.info(f"  Total: {len(sorted_results)}")
    
    # Save to text file
    best_strength_file = os.path.join(run_dir, f"BEST_STRENGTH_SORTED_RESULTS_alpha_{best_strength}.txt")
    with open(best_strength_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write("BEST STEERING STRENGTH - SORTED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        
        # Add hallucination rate comparison at the top
        f.write(f"HALLUCINATION RATE COMPARISON\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Baseline Hallucination Rate: {baseline_rate*100:.2f}% ({int(baseline_rate * len(all_results))}/{len(all_results)})\n")
        f.write(f"Steered Hallucination Rate (α={best_strength}): {best_result_summary['halluc_rate']*100:.2f}% ({best_result_summary['halluc_count']}/{len(all_results)})\n")
        f.write(f"\n")
        f.write(f"Absolute Reduction: {best_result_summary['absolute_reduction']*100:.2f} percentage points\n")
        f.write(f"Relative Reduction: {best_result_summary['relative_reduction']:.2f}%\n")
        f.write(f"\n{'='*80}\n\n")
        
        f.write(f"Steering Strength (α): {best_strength}\n")
        f.write(f"Total Rows: {len(sorted_results)}\n")
        f.write(f"Hallucination Fixed (Baseline→No Halluc): {len(halluc_fixed)}\n")
        f.write(f"Other Cases: {len(other_results)}\n")
        f.write(f"\n")
        f.write(f"SORTING: Hallucination fixed cases appear first, followed by all other cases\n")
        f.write(f"\n\n")
        
        for i, r in enumerate(sorted_results, 1):
            f.write(f"{'-'*80}\n")
            f.write(f"SAMPLE {i}/{len(sorted_results)}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Sample Index: {r['sample_idx']}\n")
            f.write(f"Baseline Hallucination: {r['baseline_hallucination']}\n")
            f.write(f"Steered Hallucination: {r['steered_hallucination']}\n")
            f.write(f"Hallucination Reduced: {r['hallucination_reduced']}\n")
            f.write(f"Steering Effect: {r['steering_effect']:+d}\n\n")
            
            f.write(f"CONTEXT:\n{r['context']}\n\n")
            
            f.write(f"QUESTION:\n{r['question']}\n\n")
            
            f.write(f"RIGHT ANSWER:\n{r['right_answer']}\n\n")
            
            f.write(f"BASELINE ANSWER (extracted):\n{r['baseline_answer']}\n\n")
            
            f.write(f"STEERED ANSWER (extracted):\n{r['steered_answer']}\n\n")
    
    logger.info(f"✓ Saved best strength sorted results: {best_strength_file}")
    return best_strength_file


def save_consolidated_results(all_results_by_strength: Dict, 
                             results_dir: str, args: Any, logger, baseline_dataset_path: str = None) -> str:
    """
    Save consolidated JSON results across all steering strengths (independent statistics only).
    
    This function creates a comprehensive JSON file containing:
    - Experiment metadata (dataset, config, hyperparameters)
    - Per-strength hallucination statistics (counts and rates)
    - Summary of best-performing steering strength
    
    Args:
        all_results_by_strength: Dict mapping steering_strength -> list of result dicts
                                Each result dict contains: sample_idx, generated_text, 
                                hallucination_score, right_answer, prompt_info
        results_dir: Directory path to save the consolidated JSON file
        args: Argparse namespace with experiment parameters (num_samples, batch_size,
              iti_config_path, steering_strength list, max_tokens)
        logger: Logger instance for progress reporting
        baseline_dataset_path: Optional path to original dataset file
    
    Returns:
        str: Path to the saved consolidated results file
    
    Output File Structure (consolidated_results.json):
        {
            'metadata': {...},
            'results_by_strength': {
                '0.5': [list of sample results],
                '1.0': [list of sample results],
                ...
            },
            'hallucination_summary': {
                '0.5': {halluc_count, no_halluc_count, api_fail_count, halluc_rate},
                ...
            }
        }
    """
    
    consolidated_results = {
        'metadata': {
            'dataset_path': baseline_dataset_path,
            'num_samples': args.num_samples,
            'iti_config_path': args.iti_config_path,
            'steering_strengths': sorted(args.steering_strength),
            'max_tokens': args.max_tokens,
            'batch_size': args.batch_size,
        },
        'results_by_strength': {},
        'hallucination_summary': {}
    }
    
    for steering_strength in sorted(args.steering_strength):
        all_results = all_results_by_strength[steering_strength]
        
        json_results = []
        steered_api_failures = 0
        steered_halluc_count = 0
        steered_no_halluc_count = 0
        
        for r in all_results:
            steered_api_failures += 1 if r['steered_hallucination'] == 2 else 0
            steered_halluc_count += 1 if r['steered_hallucination'] == 1 else 0
            steered_no_halluc_count += 1 if r['steered_hallucination'] == 0 else 0
            
            json_results.append({
                'sample_idx': int(r['sample_idx']),
                'qa_prompt': r['qa_prompt'],
                'question': r['question'],
                'context': r['context'],
                'right_answer': r['right_answer'],
                'baseline_answer': r['baseline_answer'],         # Extracted answer
                'steered_answer': r['steered_answer'],           # Extracted answer
                'baseline_output': r['baseline_output'],
                'steered_output': r['steered_output'],
                'baseline_hallucination': int(r['baseline_hallucination']),
                'steered_hallucination': int(r['steered_hallucination']),
            })
        
        consolidated_results['results_by_strength'][steering_strength] = json_results
        
        # Store independent statistics for this strength
        total_valid = steered_halluc_count + steered_no_halluc_count
        steered_halluc_rate = steered_halluc_count / total_valid if total_valid > 0 else 0
        
        consolidated_results['hallucination_summary'][steering_strength] = {
            'no_hallucination_count': steered_no_halluc_count,
            'hallucination_count': steered_halluc_count,
            'api_failure_count': steered_api_failures,
            'total_valid_samples': total_valid,
            'hallucination_rate': float(steered_halluc_rate),
        }
    
    # Save consolidated file
    consolidated_file = os.path.join(results_dir, "ALL_RESULTS_CONSOLIDATED.json")
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated_results, f, indent=2)
    logger.info(f"✓ Saved consolidated results: {consolidated_file}")
    
    # Save lightweight analysis file (no text data - for cross-experiment analysis only)
    analysis_results = {
        'metadata': {
            'iti_config_path': consolidated_results['metadata']['iti_config_path'],
            'steering_strengths': consolidated_results['metadata']['steering_strengths'],
            'num_samples': consolidated_results['metadata']['num_samples'],
        },
        'results_by_strength': {},
        'hallucination_summary': consolidated_results['hallucination_summary']
    }
    
    # Save only the hallucination scores (no text data)
    for steering_strength in sorted(args.steering_strength):
        all_results = all_results_by_strength[steering_strength]
        analysis_results_list = []
        
        for r in all_results:
            analysis_results_list.append({
                'sample_idx': int(r['sample_idx']),
                'baseline_hallucination': int(r['baseline_hallucination']),
                'steered_hallucination': int(r['steered_hallucination']),
            })
        
        analysis_results['results_by_strength'][steering_strength] = analysis_results_list
    
    analysis_file = os.path.join(results_dir, "ALL_RESULTS_ANALYSIS.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    logger.info(f"✓ Saved lightweight analysis results: {analysis_file}")
    
    return consolidated_file


def save_detailed_text_results(all_results_by_strength: Dict, results_dir: str, 
                               args: Any, logger, baseline_dataset_path: str = None,
                               baseline_dataset_format: str = None) -> List[str]:
    """
    Save separate detailed text files for each steering strength (alpha value) with independent statistics.
    
    Args:
        all_results_by_strength: Dict of {steering_strength: [results]}
        results_dir: Directory to save text files
        args: Command line args
        logger: Logger instance
        baseline_dataset_path: Path to the dataset file
        baseline_dataset_format: Dataset format ('mcq', 'non_mcq', 'mmlu')
        
    Returns:
        List of saved file paths
        
    Raises:
        ValueError: If baseline_dataset_format is invalid or missing required fields for MCQ datasets
    """
    
    # Strict validation of dataset format
    if baseline_dataset_format is None:
        logger.error("baseline_dataset_format parameter is required")
        raise ValueError("baseline_dataset_format must be provided to save_detailed_text_results()")
    
    if baseline_dataset_format not in VALID_DATASET_FORMATS:
        logger.error(f"Invalid baseline_dataset_format: {baseline_dataset_format}")
        raise ValueError(f"Invalid baseline_dataset_format: '{baseline_dataset_format}'. Must be one of: {VALID_DATASET_FORMATS}")
    
    saved_files = []
    
    for steering_strength in sorted(all_results_by_strength.keys()):
        all_results = all_results_by_strength[steering_strength]
        
        # Count statistics
        baseline_api_failures = sum(1 for r in all_results if r['baseline_hallucination'] == 2)
        steered_api_failures = sum(1 for r in all_results if r['steered_hallucination'] == 2)
        steered_no_halluc = sum(1 for r in all_results if r['steered_hallucination'] == 0)
        steered_halluc = sum(1 for r in all_results if r['steered_hallucination'] == 1)
        steered_valid = steered_no_halluc + steered_halluc
        steered_halluc_rate = steered_halluc / steered_valid if steered_valid > 0 else 0
        
        # Create filename with alpha value
        text_file = os.path.join(results_dir, f"DETAILED_RESULTS_alpha_{steering_strength}.txt")
        
        with open(text_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write("ITI STEERING EVALUATION - DETAILED RESULTS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Dataset: {baseline_dataset_path}\n")
            f.write(f"Dataset Format: {baseline_dataset_format}\n")
            f.write(f"Num Samples: {args.num_samples}\n")
            f.write(f"ITI Config: {args.iti_config_path}\n")
            f.write(f"Steering Strength (alpha): {steering_strength}\n")
            f.write(f"Max Tokens: {args.max_tokens}\n")
            f.write(f"\n")
            f.write(f"INDEPENDENT HALLUCINATION STATISTICS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Non-Hallucinations: {steered_no_halluc}\n")
            f.write(f"Hallucinations: {steered_halluc}\n")
            f.write(f"API Failures: {steered_api_failures}\n")
            f.write(f"Hallucination Rate: {steered_halluc_rate*100:.2f}% ({steered_halluc}/{steered_valid} valid samples)\n")
            f.write(f"\n\n")
            
            for i, r in enumerate(all_results, 1):
                f.write(f"{'-'*80}\n")
                f.write(f"SAMPLE {i}/{len(all_results)}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Sample Index: {r['sample_idx']}\n")
                f.write(f"Baseline Hallucination: {r['baseline_hallucination']} ")
                if r['baseline_hallucination'] == 0:
                    f.write("(CORRECT)\n")
                elif r['baseline_hallucination'] == 1:
                    f.write("(HALLUCINATED)\n")
                else:
                    f.write("(API FAILURE)\n")
                f.write(f"Steered Hallucination: {r['steered_hallucination']} ")
                if r['steered_hallucination'] == 0:
                    f.write("(CORRECT)\n")
                elif r['steered_hallucination'] == 1:
                    f.write("(HALLUCINATED)\n")
                else:
                    f.write("(API FAILURE)\n\n")
                
                # Reconstruct and validate prompt components
                try:
                    components = _reconstruct_and_validate_prompt_components(
                        r, baseline_dataset_format, r['sample_idx'], logger
                    )
                except ValueError as e:
                    # Re-raise with context
                    logger.error(str(e))
                    raise
                
                # PROMPT COMPONENTS section
                f.write(f"PROMPT COMPONENTS (As Built)\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Context:\n{components['context']}\n\n")
                f.write(f"Question:\n{components['question']}\n\n")
                
                # Conditionally include choices and answerKey based on dataset format
                if baseline_dataset_format in ['mcq', 'mmlu']:
                    if components['answerKey']:
                        f.write(f"Answer Key (Letter): {components['answerKey']}\n\n")
                    
                    if components['choices_str']:
                        f.write(f"Choices:\n{components['choices_str']}\n\n")
                
                f.write(f"Ground Truth Answer Text: {components['answer_text']}\n\n")
                
                # FULL QA PROMPT section (exact string sent to model)
                f.write(f"FULL QA PROMPT (Exact String Sent to Model)\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{r['qa_prompt']}\n\n")
                
                # MODEL ANSWERS section
                f.write(f"MODEL ANSWERS\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Baseline Answer (extracted):\n{r['baseline_answer']}\n\n")
                f.write(f"Steered Answer (extracted):\n{r['steered_answer']}\n\n")
        
        logger.info(f"✓ Saved detailed results for α={steering_strength}: {text_file}")
        saved_files.append(text_file)
    
    return saved_files



def save_analysis_summary(all_results_by_strength: Dict, analysis_data: Dict, 
                         analysis_dir: str, args: Any, logger, baseline_dataset_path: str = None) -> str:
    """Save hallucination analysis summary comparing all steering strengths."""
    
    best_strength = max(
        analysis_data['improvements'], 
        key=lambda s: analysis_data['improvements'][s]['absolute_reduction']
    )
    best_result_summary = analysis_data['improvements'][best_strength]
    baseline_rate = analysis_data['baseline_rate']
    
    analysis_file = os.path.join(analysis_dir, "HALLUCINATION_ANALYSIS_SUMMARY.txt")
    with open(analysis_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write("HALLUCINATION ANALYSIS - BASELINE VS STEERING STRENGTH\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dataset: {baseline_dataset_path}\n")
        f.write(f"Num Samples: {args.num_samples}\n")
        f.write(f"ITI Config: {args.iti_config_path}\n")
        f.write(f"Steering Strengths (alpha): {sorted(args.steering_strength)}\n")
        f.write(f"\n\n")
        
        f.write(f"{'='*80}\n")
        f.write("BEST STEERING STRENGTH SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Optimal Steering Strength (α): {best_strength}\n")
        f.write(f"Baseline Hallucination Rate: {baseline_rate:.4f} ({baseline_rate*100:.2f}%)\n")
        f.write(f"Steered Hallucination Rate: {best_result_summary['halluc_rate']:.4f} ({best_result_summary['halluc_rate']*100:.2f}%)\n")
        f.write(f"Absolute Reduction: {best_result_summary['absolute_reduction']:.4f} ({best_result_summary['absolute_reduction']*100:.2f} percentage points)\n")
        f.write(f"Relative Reduction: {best_result_summary['relative_reduction']:.2f}%\n")
        f.write(f"\n\n")
        
        f.write(f"{'='*80}\n")
        f.write("DETAILED RESULTS BY STEERING STRENGTH\n")
        f.write(f"{'='*80}\n\n")
        
        for strength, summary in sorted(analysis_data['improvements'].items()):
            f.write(f"Steering Strength (α): {strength}\n")
            f.write(f"  Baseline Hallucination Rate: {baseline_rate:.4f}\n")
            f.write(f"  Steered Hallucination Rate: {summary['halluc_rate']:.4f}\n")
            f.write(f"  Absolute Reduction (pts): {summary['absolute_reduction']:.4f}\n")
            f.write(f"  Relative Reduction (%): {summary['relative_reduction']:.2f}\n")
            f.write(f"  Baseline API Failures: {summary.get('baseline_api_failures', 'N/A')}\n")
            f.write(f"  Steered API Failures: {summary.get('steered_api_failures', 'N/A')}\n\n")
    
    logger.info(f"✓ Saved analysis summary: {analysis_file}")
    return analysis_file


def log_final_summary(analysis_data: Dict, logger) -> None:
    """Log final analysis summary to console."""
    # Find best steering strength using relative_reduction (primary ranking metric)
    # Note: Both absolute_reduction (percentage points) and relative_reduction (%) are calculated
    # but relative_reduction is used for determining best parameter combinations
    best_strength = max(
        analysis_data['improvements'], 
        key=lambda s: analysis_data['improvements'][s]['relative_reduction']
    )
    best_result_summary = analysis_data['improvements'][best_strength]
    baseline_rate = analysis_data['baseline_rate']

    logger.info(f"\n{'='*80}")
    logger.info("FINAL ANALYSIS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Optimal Steering Strength (α): {best_strength}")
    logger.info(f"Baseline Hallucination Rate: {baseline_rate*100:.2f}%")
    logger.info(f"Steered Hallucination Rate: {best_result_summary['halluc_rate']*100:.2f}%")
    logger.info(f"Absolute Reduction: {best_result_summary['absolute_reduction']*100:.2f} percentage points")
    logger.info(f"Relative Reduction: {best_result_summary['relative_reduction']:.2f}%")


# ================================================================
# BASELINE PERSISTENCE FUNCTIONS
# ================================================================

def save_baseline_data(baseline_texts: Dict, baseline_outputs: Dict, prompts: Dict,
                       baseline_evaluation: Dict, run_dir: str, logger) -> None:
    """
    Save baseline data to pickle files for reuse across steering experiments.
    
    Args:
        baseline_texts: Dict of {idx: answer_text}
        baseline_outputs: Dict of {idx: full_generation}
        prompts: Dict of {idx: prompt_info}
        baseline_evaluation: Dict of {idx: hallucination_score}
        run_dir: Directory to save files
        logger: Logger instance
    """
    baseline_texts_file = os.path.join(run_dir, "baseline_texts.pkl")
    baseline_outputs_file = os.path.join(run_dir, "baseline_full_outputs.pkl")
    baseline_prompts_file = os.path.join(run_dir, "baseline_prompts.pkl")
    baseline_evaluation_file = os.path.join(run_dir, "baseline_evaluation.pkl")

    with open(baseline_texts_file, 'wb') as f:
        pickle.dump(baseline_texts, f)
    logger.info(f"✓ Saved baseline texts: {baseline_texts_file}")

    with open(baseline_outputs_file, 'wb') as f:
        pickle.dump(baseline_outputs, f)
    logger.info(f"✓ Saved baseline full outputs: {baseline_outputs_file}")

    with open(baseline_prompts_file, 'wb') as f:
        pickle.dump(prompts, f)
    logger.info(f"✓ Saved baseline prompts: {baseline_prompts_file}")

    with open(baseline_evaluation_file, 'wb') as f:
        pickle.dump(baseline_evaluation, f)
    logger.info(f"✓ Saved baseline evaluation: {baseline_evaluation_file}")


def load_baseline_data(baseline_dir: str, logger, num_samples: Optional[int] = None) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load saved baseline data for comparison in steering experiments.
    
    This function loads all baseline artifacts generated by baseline_run.py.
    The baseline provides ground truth for hallucination comparisons in steering experiments.
    
    Args:
        baseline_dir: Directory containing saved baseline files. Must include:
                     - baseline_texts.pkl: {idx: extracted_answer_text}
                     - baseline_full_outputs.pkl: {idx: full_generation_string}
                     - baseline_prompts.pkl: {idx: prompt_info_dict}
                     - baseline_evaluation.pkl: {idx: hallucination_score}
        logger: Logger instance for progress reporting
        num_samples: Optional maximum number of samples to load. If None, loads all samples.
                    Useful for quick testing or memory-constrained environments.
        
    Returns:
        Tuple[Dict, Dict, Dict, Dict]:
            - baseline_texts: {sample_idx: str} - Extracted answer text per sample
            - baseline_outputs: {sample_idx: str} - Full generation including reasoning
            - prompts: {sample_idx: dict} - Prompt components (context, question, choices)
            - baseline_evaluation: {sample_idx: int} - Hallucination scores (0=correct, 1=halluc, 2=API fail)
    
    Raises:
        FileNotFoundError: If any required baseline file is missing
    
    Example:
        >>> texts, outputs, prompts, evals = load_baseline_data("./data/baseline_results/run_123", logger)
        >>> print(f"Loaded {len(texts)} baseline samples")
        >>> halluc_rate = sum(1 for s in evals.values() if s == 1) / len(evals)
    """
    baseline_texts_file = os.path.join(baseline_dir, "baseline_texts.pkl")
    baseline_outputs_file = os.path.join(
        baseline_dir, "baseline_full_outputs.pkl")
    baseline_prompts_file = os.path.join(baseline_dir, "baseline_prompts.pkl")
    baseline_evaluation_file = os.path.join(
        baseline_dir, "baseline_evaluation.pkl")

    # Check all files exist
    required_files = [baseline_texts_file, baseline_outputs_file,
                      baseline_prompts_file, baseline_evaluation_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Baseline file not found: {file_path}")

    with open(baseline_texts_file, 'rb') as f:
        baseline_texts = pickle.load(f)
    logger.info(f"✓ Loaded baseline texts from {baseline_texts_file}")

    with open(baseline_outputs_file, 'rb') as f:
        baseline_outputs = pickle.load(f)
    logger.info(f"✓ Loaded baseline full outputs from {baseline_outputs_file}")

    with open(baseline_prompts_file, 'rb') as f:
        prompts = pickle.load(f)
    logger.info(f"✓ Loaded baseline prompts from {baseline_prompts_file}")

    with open(baseline_evaluation_file, 'rb') as f:
        baseline_evaluation = pickle.load(f)
    logger.info(
        f"✓ Loaded baseline evaluation from {baseline_evaluation_file}")

    # Filter to num_samples if specified
    if num_samples is not None and num_samples < len(baseline_texts):
        # Get sorted indices to ensure consistent sampling
        all_indices = sorted(baseline_texts.keys())
        selected_indices = all_indices[:num_samples]
        
        baseline_texts = {idx: baseline_texts[idx] for idx in selected_indices}
        baseline_outputs = {idx: baseline_outputs[idx] for idx in selected_indices}
        prompts = {idx: prompts[idx] for idx in selected_indices}
        baseline_evaluation = {idx: baseline_evaluation[idx] for idx in selected_indices}
        
        logger.info(f"Filtered to first {num_samples} samples (was {len(all_indices)} total)")
    
    logger.info(
        f"Successfully loaded baseline data for {len(baseline_texts)} samples")
    return baseline_texts, baseline_outputs, prompts, baseline_evaluation


def create_voting_baseline(baseline_dirs: List[str], logger, num_samples: Optional[int] = None) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create majority-voted baseline from multiple baseline run directories.
    
    Applies per-sample majority voting on hallucination scores. The label with the highest
    ratio of votes wins (e.g., if 2-of-3 runs voted 1, output is 1). In case of exact tie,
    label 1 (hallucination) is preferred over label 0 (no hallucination).
    
    IMPORTANT: For each sample, this function selects the baseline answer and output from
    a run that has the SAME hallucination label as the majority-voted label. This ensures
    consistency between the hallucination score and the actual answer/output text.
    
    Example:
        Sample 0: Run1(halluc=1, answer="A"), Run2(halluc=0, answer="B"), Run3(halluc=1, answer="C")
        Majority vote: 1 (two runs voted hallucination)
        Selected: answer="A" or "C" (from a run where halluc=1)
        Final: baseline_evaluation[0]=1, baseline_answer[0]="A" or "C" (consistent)
    
    Args:
        baseline_dirs: List of baseline run directory paths (each contains baseline_*.pkl files)
        logger: Logger instance
        num_samples: Optional maximum number of samples to load. If None, loads all samples.
        
    Returns:
        Tuple of (baseline_texts, baseline_outputs, prompts, baseline_evaluation_voted)
        - baseline_texts: Dict with answers selected from runs matching voted label
        - baseline_outputs: Dict with outputs selected from runs matching voted label
        - prompts: Dict of prompt info (identical across all runs)
        - baseline_evaluation_voted: Dict with per-sample majority-voted hallucination scores (0 or 1)
        
    Raises:
        FileNotFoundError: If any baseline directory is missing required files
        ValueError: If baseline_dirs is empty or sample indices don't intersect
    """
    if not baseline_dirs:
        raise ValueError("baseline_dirs cannot be empty")
    
    logger.info(f"\n{'='*80}")
    logger.info("CREATING MAJORITY-VOTED BASELINE")
    logger.info(f"{'='*80}")
    logger.info(f"Loading {len(baseline_dirs)} baseline run(s)...")
    
    # Load all baselines
    all_baseline_evaluations = []
    common_indices = None
    
    for i, baseline_dir in enumerate(baseline_dirs, 1):
        logger.info(f"\n  Run {i}/{len(baseline_dirs)}: {baseline_dir}")
        
        # Load baseline_evaluation.pkl
        baseline_evaluation_file = os.path.join(baseline_dir, "baseline_evaluation.pkl")
        if not os.path.exists(baseline_evaluation_file):
            raise FileNotFoundError(f"baseline_evaluation.pkl not found in {baseline_dir}")
        
        with open(baseline_evaluation_file, 'rb') as f:
            baseline_eval = pickle.load(f)
        
        logger.info(f"    ✓ Loaded {len(baseline_eval)} samples")
        all_baseline_evaluations.append(baseline_eval)
        
        # Track common indices (intersection)
        indices_set = set(baseline_eval.keys())
        if common_indices is None:
            common_indices = indices_set
        else:
            common_indices = common_indices.intersection(indices_set)
    
    logger.info(f"\nCommon samples across all runs: {len(common_indices)}")
    
    # Load ALL baseline runs' texts, outputs, and evaluations
    # (answers/outputs differ due to LLM non-determinism, so we need to select matching ones)
    logger.info(f"Loading all baseline texts, outputs, and evaluations...")
    all_baseline_data = []
    for baseline_dir in baseline_dirs:
        texts, outputs, _, evals = load_baseline_data(baseline_dir, logger, num_samples=None)
        all_baseline_data.append((texts, outputs, evals))
    
    # Use prompts from first baseline (prompts are identical across runs)
    prompts = all_baseline_data[0][0]  # Placeholder to get first dir
    baseline_texts_first, baseline_outputs_first, prompts, _ = load_baseline_data(baseline_dirs[0], logger, num_samples=None)
    
    # Intersect with common indices
    common_indices_sorted = sorted(common_indices)
    
    # Further filter to num_samples if specified
    if num_samples is not None and len(common_indices_sorted) > num_samples:
        common_indices_sorted = common_indices_sorted[:num_samples]
        logger.info(f"Filtered to first {num_samples} samples")
    
    # Apply majority voting
    logger.info(f"\nApplying per-sample majority voting...")
    baseline_evaluation_voted = {}
    baseline_texts_voted = {}
    baseline_outputs_voted = {}
    prompts_voted = {}
    
    for idx in common_indices_sorted:
        # Collect votes for this sample from all runs
        votes = []
        for run_idx, (texts, outputs, evals) in enumerate(all_baseline_data):
            score = evals.get(idx, None)
            if score is not None and score in [0, 1]:
                votes.append((score, run_idx))  # Store score and which run it came from
        
        if not votes:
            logger.warning(f"  Sample {idx}: No valid votes found, skipping")
            continue
        
        # Calculate vote ratios
        vote_count_1 = sum(1 for v, _ in votes if v == 1)
        vote_count_0 = len(votes) - vote_count_1
        
        # Majority voting: highest ratio wins (ties favor label 1)
        if vote_count_1 >= vote_count_0:
            majority_vote = 1
        else:
            majority_vote = 0
        
        baseline_evaluation_voted[idx] = majority_vote
        
        # Select answer and output from a baseline run with matching majority vote
        # This ensures consistency: if majority voted hallucination=1, 
        # we use answer/output from a run that also evaluated hallucination=1
        selected_run_idx = None
        for score, run_idx in votes:
            if score == majority_vote:
                selected_run_idx = run_idx
                break
        
        if selected_run_idx is not None:
            baseline_texts_voted[idx] = all_baseline_data[selected_run_idx][0][idx]
            baseline_outputs_voted[idx] = all_baseline_data[selected_run_idx][1][idx]
        else:
            # Fallback to first run if no exact match (shouldn't happen)
            baseline_texts_voted[idx] = all_baseline_data[0][0].get(idx, "")
            baseline_outputs_voted[idx] = all_baseline_data[0][1].get(idx, "")
        
        prompts_voted[idx] = prompts[idx]
    
    logger.info(f"✓ Majority voting completed for {len(baseline_evaluation_voted)} samples")
    
    # Log vote distribution
    halluc_count = sum(1 for score in baseline_evaluation_voted.values() if score == 1)
    halluc_rate = halluc_count / len(baseline_evaluation_voted) if baseline_evaluation_voted else 0
    logger.info(f"  Hallucinations (label=1): {halluc_count}/{len(baseline_evaluation_voted)} ({halluc_rate*100:.2f}%)")
    logger.info(f"  No hallucinations (label=0): {len(baseline_evaluation_voted) - halluc_count}/{len(baseline_evaluation_voted)} ({(1-halluc_rate)*100:.2f}%)")
    
    logger.info(f"\n✓ Successfully created voted baseline for {len(baseline_evaluation_voted)} samples")
    logger.info(f"  Answers/outputs selected from runs matching majority-voted labels for consistency")
    return baseline_texts_voted, baseline_outputs_voted, prompts_voted, baseline_evaluation_voted


def save_baseline_detailed_results(baseline_texts: Dict, baseline_outputs: Dict, prompts: Dict,
                                   baseline_evaluation: Dict, run_dir: str, logger) -> str:
    """Save detailed baseline results to JSON file."""

    detailed_results = {
        'metadata': {
            'num_samples': len(baseline_texts),
            'timestamp': pd.Timestamp.now().isoformat(),
        },
        'samples': []
    }

    for idx in sorted(prompts.keys()):
        sample_result = {
            'sample_idx': int(idx),
            'qa_prompt': prompts[idx]['qa_prompt'],
            'question': prompts[idx]['question'],
            'context': prompts[idx]['context'],
            'right_answer': prompts[idx]['right_answer'],
            'baseline_output': baseline_outputs[idx],
            'baseline_answer': baseline_texts[idx],
            'hallucination': int(baseline_evaluation.get(idx, -1)),
        }
        detailed_results['samples'].append(sample_result)

    detailed_file = os.path.join(run_dir, "baseline_results_detailed.json")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    logger.info(f"✓ Saved baseline detailed results: {detailed_file}")

    return detailed_file


# ================================================================
# Grid Search Result Aggregation Functions
# ================================================================

def aggregate_grid_results(stage1_csv: str, stage2_csv: str = None):
    """
    Aggregate results from Stage 1 and optionally Stage 2.
    
    Args:
        stage1_csv: Path to Stage 1 results CSV
        stage2_csv: Path to Stage 2 results CSV (optional)
        
    Returns:
        DataFrame with combined results
    """
    dfs = []
    
    if os.path.exists(stage1_csv):
        df1 = pd.read_csv(stage1_csv)
        dfs.append(df1)
    
    if stage2_csv and os.path.exists(stage2_csv):
        df2 = pd.read_csv(stage2_csv)
        dfs.append(df2)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicates, keeping the one with higher reduction
    combined = combined.sort_values('relative_reduction', ascending=False)
    combined = combined.drop_duplicates(subset=['k', 'alpha'], keep='first')
    
    return combined


def identify_stage2_region(stage1_results: pd.DataFrame, margin_k: int = 5, 
                          margin_alpha: float = 0.5) -> Tuple[List[int], List[float]]:
    """
    Identify Stage 2 fine-grained region based on Stage 1 best results.
    
    Args:
        stage1_results: Stage 1 results DataFrame
        margin_k: K margin around best (steps of 1)
        margin_alpha: Alpha margin around best (steps of 0.1)
        
    Returns:
        Tuple of (k_values_fine, alpha_values_fine)
    """
    if len(stage1_results) == 0:
        return [], []
    
    # Find best combo
    best_idx = stage1_results['relative_reduction'].idxmax()
    best_k = int(stage1_results.loc[best_idx, 'k'])
    best_alpha = float(stage1_results.loc[best_idx, 'alpha'])
    best_reduction = float(stage1_results.loc[best_idx, 'relative_reduction'])
    
    # Define fine region
    k_fine = list(range(max(10, best_k - margin_k), min(300, best_k + margin_k + 1)))
    alpha_fine = np.arange(max(0.1, best_alpha - margin_alpha), 
                           min(10.0, best_alpha + margin_alpha + 0.05), 0.1)
    alpha_fine = [round(a, 1) for a in alpha_fine]
    
    return k_fine, alpha_fine


def generate_grid_heatmap(results_df: pd.DataFrame, output_path: str, 
                         stage_name: str = "Combined"):
    """
    Generate heatmap visualization from grid results.
    
    Args:
        results_df: DataFrame with k, alpha, relative_reduction
        output_path: Output PNG path
        stage_name: Name for title
    """
    if len(results_df) == 0:
        return
    
    # Pivot to create matrix
    pivot_df = results_df.pivot_table(index='k', columns='alpha', 
                                      values='relative_reduction', aggfunc='max')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f'{a:.1f}' for a in pivot_df.columns], rotation=45)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    
    ax.set_xlabel('Alpha (Steering Strength)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-K Heads', fontsize=12, fontweight='bold')
    ax.set_title(f'{stage_name} Grid Search: Relative Hallucination Reduction (%)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Reduction (%)', fontsize=11, fontweight='bold')
    
    # Add value annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color='white' if val > 15 else 'black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_best_hyperparams_report(combined_df: pd.DataFrame, stage1_df: pd.DataFrame,
                                 stage2_df: pd.DataFrame, output_path: str, 
                                 baseline_rate: float):
    """
    Save final best hyperparameters recommendation report.
    
    Args:
        combined_df: Combined results from both stages
        stage1_df: Stage 1 results
        stage2_df: Stage 2 results
        output_path: Output file path
        baseline_rate: Baseline hallucination rate (%)
    """
    if len(combined_df) == 0:
        return
    
    # Find best combos
    best_idx = combined_df['relative_reduction'].idxmax()
    best_k_global = int(combined_df.loc[best_idx, 'k'])
    best_alpha_global = float(combined_df.loc[best_idx, 'alpha'])
    best_reduction_global = float(combined_df.loc[best_idx, 'relative_reduction'])
    
    # Get top 5
    top5 = combined_df.nlargest(5, 'relative_reduction')
    
    # Stage 1 best
    if len(stage1_df) > 0:
        best_idx_s1 = stage1_df['relative_reduction'].idxmax()
        best_k_s1 = int(stage1_df.loc[best_idx_s1, 'k'])
        best_alpha_s1 = float(stage1_df.loc[best_idx_s1, 'alpha'])
        best_reduction_s1 = float(stage1_df.loc[best_idx_s1, 'relative_reduction'])
    else:
        best_k_s1 = best_k_global
        best_alpha_s1 = best_alpha_global
        best_reduction_s1 = best_reduction_global
    
    # Stage 2 best (if exists)
    if len(stage2_df) > 0:
        best_idx_s2 = stage2_df['relative_reduction'].idxmax()
        best_k_s2 = int(stage2_df.loc[best_idx_s2, 'k'])
        best_alpha_s2 = float(stage2_df.loc[best_idx_s2, 'alpha'])
        best_reduction_s2 = float(stage2_df.loc[best_idx_s2, 'relative_reduction'])
        improvement_s1_s2 = best_reduction_s2 - best_reduction_s1
    else:
        best_k_s2 = best_k_s1
        best_alpha_s2 = best_alpha_s1
        best_reduction_s2 = best_reduction_s1
        improvement_s1_s2 = 0
    
    steered_rate = baseline_rate * (100 - best_reduction_global) / 100
    absolute_reduction = baseline_rate - steered_rate
    
    with open(output_path, 'w') as f:
        f.write("╔" + "═" * 78 + "╗\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("║" + "OPTIMAL ITI STEERING CONFIGURATION".center(78) + "║\n")
        f.write("║" + " " * 78 + "║\n")
        f.write("╚" + "═" * 78 + "╝\n\n")
        
        f.write(f"GLOBAL BEST: k={best_k_global}, α={best_alpha_global:.2f}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hallucination Reduction:\n")
        f.write(f"  - Baseline Rate: {baseline_rate:.2f}%\n")
        f.write(f"  - Steered Rate (k={best_k_global}, α={best_alpha_global:.2f}): {steered_rate:.2f}%\n")
        f.write(f"  - Absolute Reduction: {absolute_reduction:.2f} percentage points\n")
        f.write(f"  - Relative Reduction: {best_reduction_global:.2f}%\n\n")
        
        f.write("Top-5 Configurations:\n")
        for rank, (idx, row) in enumerate(top5.iterrows(), 1):
            f.write(f"  {rank}. k={int(row['k'])}, α={float(row['alpha']):.2f}: {row['relative_reduction']:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Stage Comparison:\n")
        f.write(f"  Stage 1 Best: k={best_k_s1}, α={best_alpha_s1:.1f} ({best_reduction_s1:.2f}%)\n")
        if len(stage2_df) > 0:
            f.write(f"  Stage 2 Best: k={best_k_s2}, α={best_alpha_s2:.2f} ({best_reduction_s2:.2f}%)\n")
            f.write(f"  Improvement (S1→S2): +{improvement_s1_s2:.2f}%\n")
        else:
            f.write(f"  Stage 2: Not executed\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Recommendation Command:\n")
        f.write(f"python sweep_orchestrator.py \\\n")
        f.write(f"  --h5-dir ./data/ITI/activations/... \\\n")
        f.write(f"  --baseline-dir ./data/ITI/baseline_results/... \\\n")
        f.write(f"  --k-value {best_k_global} --alpha-value {best_alpha_global:.2f}\n")
