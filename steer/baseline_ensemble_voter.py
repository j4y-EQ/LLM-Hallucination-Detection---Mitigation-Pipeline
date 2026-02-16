#!/usr/bin/env python
"""
Baseline Ensemble Voter - Creates majority-voted baseline from multiple runs.

This script:
1. Auto-detects multiple BASELINE_*_YYYYMMDD_HHMMSS directories in a parent directory
2. Loads baseline_evaluation.pkl from each run
3. Applies per-sample majority voting (highest ratio label wins; ties favor 0)
4. Saves new BASELINE_ENSEMBLE_VOTED_YYYYMMDD_HHMMSS directory with voted results

Usage:
    python -m steer.baseline_ensemble_voter \
        --parent-dir ./data/baseline_results_llama/individual/nqswap \
        --output-dir ./data/baseline_results_llama/ensembled_nqswap

Or with explicit baseline directories:
    python -m steer.baseline_ensemble_voter --baseline-dirs \
        ./data/baseline_results/BASELINE_nqswap_20251120_030454 \
        ./data/baseline_results/BASELINE_nqswap_20251120_040521 \
        ./data/baseline_results/BASELINE_nqswap_20251120_050632

Output:
    Creates new directory with identical structure to single baseline run:
    - baseline_config.json (preserves dataset_format, adds voting_metadata)
    - baseline_texts.pkl
    - baseline_full_outputs.pkl
    - baseline_prompts.pkl
    - baseline_evaluation.pkl (voted)
    - baseline_results_detailed.json
"""

import os
import sys
import argparse
import json
import pickle
import glob
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from steer.utils.steer_files_utils import create_voting_baseline, save_baseline_data, save_baseline_detailed_results
from logger import consolidated_logger as pipeline_logger

# Parse arguments
parser = argparse.ArgumentParser(
    description='Create majority-voted baseline from multiple baseline runs')
parser.add_argument('--parent-dir', type=str, default=None,
                    help='Parent directory to auto-detect BASELINE_* directories')
parser.add_argument('--baseline-dirs', type=str, nargs='+', default=None,
                    help='Explicit list of baseline directories (use instead of --parent-dir)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory for voted baseline (default: same as parent-dir)')
parser.add_argument('--num-samples', type=int, default=None,
                    help='Optional maximum number of samples to use')
args = parser.parse_args()

# Determine baseline directories to use
baseline_dirs = None

if args.baseline_dirs:
    baseline_dirs = args.baseline_dirs
    logger.info(f"Using explicitly provided baseline directories ({len(baseline_dirs)} dirs)")

elif args.parent_dir:
    if not os.path.isdir(args.parent_dir):
        logger.error(f"Parent directory not found: {args.parent_dir}")
        sys.exit(1)
    
    # Auto-detect BASELINE_* directories
    pattern = os.path.join(args.parent_dir, "BASELINE_*_*")
    matching_dirs = sorted(glob.glob(pattern))
    
    # Filter to valid baseline directories (have required pickle files)
    valid_baseline_dirs = []
    for d in matching_dirs:
        baseline_eval_file = os.path.join(d, "baseline_evaluation.pkl")
        if os.path.exists(baseline_eval_file):
            valid_baseline_dirs.append(d)
    
    baseline_dirs = valid_baseline_dirs
    logger.info(f"Auto-detected {len(baseline_dirs)} valid baseline run(s) in {args.parent_dir}")
    
    if not baseline_dirs:
        logger.error(f"No valid baseline directories found in {args.parent_dir}")
        logger.error(f"Searched pattern: {pattern}")
        sys.exit(1)

else:
    logger.error("Either --parent-dir or --baseline-dirs must be specified")
    parser.print_help()
    sys.exit(1)

# Verify minimum 2 baselines
if len(baseline_dirs) < 2:
    logger.error(f"Need at least 2 baseline runs to vote, but found {len(baseline_dirs)}")
    sys.exit(1)

# Determine output directory
output_dir = args.output_dir or (args.parent_dir if args.parent_dir else None)
if not output_dir:
    logger.error("Could not determine output directory")
    sys.exit(1)

if not os.path.isdir(output_dir):
    logger.info(f"Output directory does not exist, creating: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# Create voted baseline directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
voted_baseline_name = f"BASELINE_ENSEMBLE_VOTED_{timestamp}"
voted_baseline_dir = os.path.join(output_dir, voted_baseline_name)
os.makedirs(voted_baseline_dir, exist_ok=True)

# Setup logger to output to this directory
pipeline_logger.set_output_directory(voted_baseline_dir)

logger.info("\n" + "="*80)
logger.info("BASELINE ENSEMBLE VOTER")
logger.info("="*80)
logger.info(f"Voted Baseline Name: {voted_baseline_name}")
logger.info(f"Output Directory: {voted_baseline_dir}")
logger.info(f"Number of baseline runs to ensemble: {len(baseline_dirs)}")
logger.info(f"Optional num_samples limit: {args.num_samples}")

logger.info(f"\nBaseline directories:")
for i, d in enumerate(baseline_dirs, 1):
    logger.info(f"  {i}. {os.path.basename(d)}")

# Load config from first baseline (all baselines should have same dataset_format)
baseline_config = None
dataset_format = None

# Try to load config from first baseline
baseline_config_file = os.path.join(baseline_dirs[0], "baseline_config.json")
if os.path.exists(baseline_config_file):
    with open(baseline_config_file, 'r') as f:
        baseline_config = json.load(f)
    logger.info(f"✓ Loaded baseline config from: {baseline_config_file}")
    dataset_format = baseline_config.get('dataset_format')
else:
    logger.warning(f"baseline_config.json not found in {baseline_dirs[0]}")

# If dataset_format not found in first baseline, try other baselines
if not dataset_format:
    logger.warning(f"dataset_format not found in first baseline, searching others...")
    for baseline_dir in baseline_dirs[1:]:
        baseline_config_file = os.path.join(baseline_dir, "baseline_config.json")
        if os.path.exists(baseline_config_file):
            with open(baseline_config_file, 'r') as f:
                candidate_config = json.load(f)
            candidate_format = candidate_config.get('dataset_format')
            if candidate_format:
                dataset_format = candidate_format
                logger.info(f"✓ Found dataset_format '{dataset_format}' in: {baseline_config_file}")
                break
        
# Validate dataset_format was found
if not dataset_format:
    logger.error(f"Could not find 'dataset_format' in any baseline config files")
    logger.error(f"Searched {len(baseline_dirs)} baseline directories")
    sys.exit(1)
else:
    logger.info(f"✓ Using dataset_format: {dataset_format}")

# Create majority-voted baseline
try:
    baseline_texts, baseline_outputs, prompts, baseline_evaluation_voted = create_voting_baseline(
        baseline_dirs=baseline_dirs,
        logger=logger,
        num_samples=args.num_samples
    )
except Exception as e:
    logger.error(f"Error during majority voting: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

# Save voted baseline data
logger.info(f"\n{'='*80}")
logger.info("SAVING VOTED BASELINE DATA")
logger.info(f"{'='*80}")

try:
    save_baseline_data(
        baseline_texts=baseline_texts,
        baseline_outputs=baseline_outputs,
        prompts=prompts,
        baseline_evaluation=baseline_evaluation_voted,
        run_dir=voted_baseline_dir,
        logger=logger
    )
    
    save_baseline_detailed_results(
        baseline_texts=baseline_texts,
        baseline_outputs=baseline_outputs,
        prompts=prompts,
        baseline_evaluation=baseline_evaluation_voted,
        run_dir=voted_baseline_dir,
        logger=logger
    )
except Exception as e:
    logger.error(f"Error saving baseline data: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

# Save baseline config with voting metadata
# The baseline_config.json is critical for steering_experiment.py to work correctly
voting_config = {
    'voting_type': 'per-sample-majority',
    'num_runs_ensembled': len(baseline_dirs),
    'baseline_dirs': baseline_dirs,
    'num_samples_voted': len(baseline_evaluation_voted),
    'hallucination_rate_voted': sum(1 for s in baseline_evaluation_voted.values() if s == 1) / len(baseline_evaluation_voted) if baseline_evaluation_voted else 0,
    'timestamp': timestamp,
    'tie_resolution': 'favor_label_1_hallucination',
}

# ALWAYS include dataset_format in merged config - this is CRITICAL for steering_experiment.py
# Create merged config that preserves original config and guarantees dataset_format
if baseline_config:
    merged_config = baseline_config.copy()
else:
    merged_config = {}

# ALWAYS ensure dataset_format is present (may override if inconsistent)
merged_config['dataset_format'] = dataset_format

# Add voting metadata
merged_config['voting_metadata'] = voting_config

# Validate merged config has required fields
if 'dataset_format' not in merged_config:
    logger.error("CRITICAL: dataset_format is missing from merged config!")
    sys.exit(1)

config_file = os.path.join(voted_baseline_dir, "baseline_config.json")
with open(config_file, 'w') as f:
    json.dump(merged_config, f, indent=2)
logger.info(f"✓ Saved baseline config: {config_file}")
logger.info(f"  - dataset_format: {merged_config['dataset_format']}")
logger.info(f"  - voting_metadata: included")

# Log summary
logger.info(f"\n{'='*80}")
logger.info("ENSEMBLE VOTING COMPLETE")
logger.info(f"{'='*80}")
logger.info(f"Voted baseline directory: {voted_baseline_dir}")
logger.info(f"Total samples: {len(baseline_evaluation_voted)}")
logger.info(f"Hallucination rate: {merged_config.get('voting_metadata', {}).get('hallucination_rate_voted', 0)*100:.2f}%")
logger.info(f"\nSaved files:")
logger.info(f"  ✓ baseline_config.json (with dataset_format and voting metadata)")
logger.info(f"  ✓ baseline_texts.pkl")
logger.info(f"  ✓ baseline_full_outputs.pkl")
logger.info(f"  ✓ baseline_prompts.pkl")
logger.info(f"  ✓ baseline_evaluation.pkl")
logger.info(f"  ✓ baseline_results_detailed.json")
logger.info(f"\nConsistency guarantee:")
logger.info(f"  Each sample answer/output matches its hallucination label")
logger.info(f"\nUse directly with steering_experiment.py:")
logger.info(f"  python -m pipeline.steer.ITI.steering_experiment \\")
logger.info(f"    --baseline-dir {voted_baseline_dir} \\")
logger.info(f"    --iti-config-path <config.pkl> \\")
logger.info(f"    --steering-strength 0.5 1.0 2.0")
logger.info(f"{'='*80}\n")

print(f"\n✓ Voted baseline saved to: {voted_baseline_dir}")
