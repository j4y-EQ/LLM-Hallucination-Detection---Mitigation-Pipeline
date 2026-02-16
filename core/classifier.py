"""Activation Classification Analysis for Hallucination Detection.

Train XGBoost classifiers on captured transformer activations to predict answer
correctness (hallucination detection). Implements GPU-accelerated parallel training
with industrial-strength checkpointing and resume capability.

LABELING SYSTEM:
    - Label 1 = HALLUCINATION (incorrect answer)
    - Label 0 = NO HALLUCINATION (correct answer)
    - Label 2 = API FAILURE (excluded from training)

CONFUSION MATRIX INTERPRETATION:
    - True Positive (TP):  Predicted 1, Actual 1 (correctly identified hallucination)
    - False Positive (FP): Predicted 1, Actual 0 (incorrectly predicted hallucination)
    - True Negative (TN):  Predicted 0, Actual 0 (correctly identified non-hallucination)
    - False Negative (FN): Predicted 0, Actual 1 (missed hallucination)

FEATURES:
    - GPU-accelerated XGBoost training with multi-GPU support
    - Parallel processing with tqdm progress bars
    - Atomic file operations with corruption recovery
    - Automatic imbalance detection and handling
    - Threshold optimization (quantile or fixed modes)
    - Lock-free checkpointing with perfect resume capability
    - Data leakage prevention: train/test splits use only training data

CHECKPOINTING:
    1. First run: Leave CONTINUE_FROM_RUN = "" to start fresh
    2. If interrupted: Set CONTINUE_FROM_RUN = "run_id" to resume
       Example: CONTINUE_FROM_RUN = "run_1753945404_chunk_0"
    3. Checkpoint files: ./data/classifier/classifier_run_*/checkpoint_*.json
    4. Lock-free: Each work unit writes to unique files, no race conditions
    5. Resumable: Work units get consistent IDs for perfect resumability

REQUIRED ARGUMENTS:
    --experiment-id: Experiment ID for this classifier run (e.g., llamainstruct_repliqa)
    --gpu-ids: Comma-separated GPU IDs (e.g., 0,1,2,3)
    --num-gpus: Number of GPUs to use (must match --gpu-ids count)

Example:
    $ python classifier.py --experiment-id llamainstruct_repliqa --gpu-ids 0,1,2,3 --num-gpus 4
"""

# ================================================================
# Activation Classification Analysis
# Load saved activation data and train classifiers to predict answer correctness
#
# LABELING SYSTEM:
# - Label 1 = HALLUCINATION (incorrect answer)
# - Label 0 = NO HALLUCINATION (correct answer)
#
# CONFUSION MATRIX INTERPRETATION:
# - True Positive (TP):  Predicted 1, Actual 1 (correctly identified hallucination)
# - False Positive (FP): Predicted 1, Actual 0 (incorrectly predicted hallucination)
# - True Negative (TN):  Predicted 0, Actual 0 (correctly identified non-hallucination)
# - False Negative (FN): Predicted 0, Actual 1 (incorrectly predicted non-hallucination)
#
# ENHANCED FEATURES:
# - tqdm progress bars for all processing loops (sequential and parallel)
# - Sophisticated parallel processing with tqdm integration
# - Enhanced error handling with detailed context and str() conversion
# - Exclusion of attention pattern hooks for simplified processing
# - Robust checkpointing with resume capability
# - Data leakage prevention: train/test splits, CV, and strategy selection use only training data
#
# LOCK-FREE CHECKPOINTING USAGE:
# 1. First run: Leave CONTINUE_FROM_RUN = "" (empty) to start fresh
# 2. If interrupted: Set CONTINUE_FROM_RUN = "run_id" to resume from that run
#    Example: CONTINUE_FROM_RUN = "run_1753945404_chunk_0"
# 3. Group checkpoint files: ./data/classifier/classifier_run_*/checkpoint_run_id_group_L01_hook_scheme.json
# 4. Master checkpoint files: ./data/classifier/classifier_run_*/checkpoint_run_id.json
# 5. Progress is saved immediately after each group completes
# 6. LOCK-FREE: Each group writes to unique files based on layer/hook/scheme, no race conditions possible
# 7. RESUMABLE: Groups always get the same ID regardless of when processed, ensuring perfect resumability
# ================================================================

# ## Required Arguments

# ### 1. `--experiment-id` (REQUIRED)
# The experiment ID to use for this classifier run.
# ```bash
# --experiment-id llamainstruct_repliqa
# ```

# ### 2. `--gpu-ids` (REQUIRED)
# Comma-separated list of GPU IDs to use.
# ```bash
# --gpu-ids 0,1,2,3
# ```

# ### 3. `--num-gpus` (REQUIRED)
# Number of GPUs to use (must match the count of GPU IDs provided).
# ```bash
# --num-gpus 4
# ```

# ```
# ================================================================
# COMMAND LINE ARGUMENT PARSING
# ================================================================
import argparse
import sys

# Parse command line arguments FIRST, before any other imports
parser = argparse.ArgumentParser(
    description='Train hallucination detection classifiers on activation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--experiment-id',
    type=str,
    required=True,
    help='Experiment ID to use for this classifier run (REQUIRED, no default)'
)

parser.add_argument(
    '--gpu-ids',
    type=str,
    required=True,
    help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). REQUIRED, no default'
)

parser.add_argument(
    '--num-gpus',
    type=int,
    required=True,
    help='Number of GPUs to use (must match length of --gpu-ids). REQUIRED, no default'
)

args = parser.parse_args()

# Validate GPU configuration
gpu_id_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
if len(gpu_id_list) != args.num_gpus:
    print(f"ERROR: --num-gpus ({args.num_gpus}) does not match number of GPU IDs provided ({len(gpu_id_list)})")
    sys.exit(1)

# Override config values with command line arguments
EXPERIMENT_ID = args.experiment_id
GPU_IDS = gpu_id_list
NUM_GPUS = args.num_gpus
HANDLE_IMBALANCE = None  # Will be auto-detected based on data distribution

print("=" * 70)
print("CLASSIFIER COMMAND LINE CONFIGURATION:")
print("=" * 70)
print(f"  Experiment ID:      {EXPERIMENT_ID}")
print(f"  GPU IDs:            {GPU_IDS}")
print(f"  Number of GPUs:     {NUM_GPUS}")
print(f"  Handle Imbalance:   {HANDLE_IMBALANCE} (auto-detect from data)")
print("=" * 70)

# ================================================================
# Import the consolidated logger
from logger import consolidated_logger as logger

# Import configurations from config.py
# NOTE: EXPERIMENT_ID, GPU_IDS, NUM_GPUS, HANDLE_IMBALANCE are set from command-line args above
from config import (
    # Paths
    ACTIVATIONS_BASE_DIR, CLASSIFIER_BASE_DIR,
    # Production mode
    PRODUCTION_MAX_BATCHES, PRODUCTION_MAX_BATCH_FILES,
    # Class imbalance
    IMBALANCE_THRESHOLD,
    # Custom scoring
    CUSTOM_SCORING,
    # Random seeds
    GLOBAL_RANDOM_SEED,
    FINALIST_COUNT,
    PROBE_THRESHOLD_MODE,
    PROBE_QUANTILES,
    PROBE_FIXED_THRESHOLDS,
    LINALG_THREADS,
    # GPU configuration (DEVICE_ID kept for backwards compatibility if needed)
    DEVICE_ID,
    # Layer configuration
    SKIP_LAYERS


)

# Import custom scoring utilities from helpers
from helpers.custom_scoring import attach_custom_score

def get_current_logger():
    """Get the current logger - now using consolidated logger"""
    return logger

# 1. Setup and Imports
import subprocess
import sys

# Install required packages
def install_package(package):
    """
    Install a Python package using pip.
    
    Args:
        package (str): Name of the package to install (e.g., 'scikit-learn', 'xgboost')
    
    Raises:
        subprocess.CalledProcessError: If pip installation fails
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install missing packages
try:
    import sklearn
except ImportError:
    install_package("scikit-learn")

try:
    import h5py
except ImportError:
    install_package("h5py")

try:
    import psutil
except ImportError:
    install_package("psutil")

try:
    import imblearn
except ImportError:
    install_package("imbalanced-learn")

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    install_package("xgboost")
    from xgboost import XGBClassifier

# Import all required libraries
import os
import glob
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import time
import hashlib
import copy
import random
import re
import base64

# Additional imports for ML and processing

# Atomic loading functions (matching llama_aistack2.py patterns)
def atomic_load_pickle(filepath):
    """Load pickle file with automatic corruption recovery.
    
    Implements a robust loading strategy with automatic fallback to backup files
    if the main file is corrupted. If backup is used, it automatically restores
    the main file for consistency.
    
    Args:
        filepath (str): Path to the pickle file to load.
        
    Returns:
        object: The unpickled data, or None if both main and backup files are corrupted.
        
    Notes:
        - Tries main file first, falls back to .backup file if corrupted
        - Automatically restores backup as main file after successful recovery
        - Returns None if no valid file is found (caller must handle this)
    """
    backup_filepath = filepath + '.backup'

    # Try main file first
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            get_current_logger().warning(f"Main file corrupted: {e}")

    # Fallback to backup
    if os.path.exists(backup_filepath):
        try:
            get_current_logger().info(f"Recovering from backup: {backup_filepath}")
            with open(backup_filepath, 'rb') as f:
                data = pickle.load(f)

            # Restore backup as main file
            atomic_save_pickle(data, filepath)
            return data

        except Exception as e:
            get_current_logger().error(f"Backup also corrupted: {e}")

    return None

def atomic_save_pickle(data, filepath):
    """Atomically save data to pickle file with backup and rollback.
    
    Implements atomic write-then-rename strategy to prevent corruption:
    1. Write to temporary file
    2. Backup existing file (if exists)
    3. Atomically rename temp to target
    4. Clean up backup after success
    
    Args:
        data (object): Python object to pickle.
        filepath (str): Target file path for the pickle file.
        
    Returns:
        bool: True if save succeeded, False if failed.
        
    Notes:
        - Uses HIGHEST_PROTOCOL for maximum efficiency
        - Automatically rolls back to backup on failure
        - Cleans up temporary files even on error
    """
    temp_filepath = filepath + '.tmp'
    backup_filepath = filepath + '.backup'

    try:
        # Step 1: Write to temporary file
        with open(temp_filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Step 2: Backup existing file (if exists)
        if os.path.exists(filepath):
            if os.path.exists(backup_filepath):
                os.remove(backup_filepath)
            os.rename(filepath, backup_filepath)

        # Step 3: ATOMIC RENAME
        os.rename(temp_filepath, filepath)

        # Step 4: Clean up backup after successful save
        if os.path.exists(backup_filepath):
            os.remove(backup_filepath)
        return True

    except Exception as e:
        # Cleanup on failure
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        # Restore backup if needed
        if os.path.exists(backup_filepath) and not os.path.exists(filepath):
            os.rename(backup_filepath, filepath)

        print(f"ERROR: Atomic save failed for {filepath}: {e}")
        return False

# ML imports
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                           recall_score, confusion_matrix, auc,
                           classification_report, balanced_accuracy_score,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import itertools
import gc
import psutil

# Parallel processing imports
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock

# Add this import near the top of the file with other imports
from threadpoolctl import threadpool_limits

# ================================================================
# IMAGE EMBEDDING UTILITIES
# ================================================================

def embed_image_as_base64(image_path):
    """
    Convert an image file to a base64 data URI for embedding in HTML.
    Returns the data URI string if file exists, otherwise returns the original path.
    """
    if not os.path.exists(image_path):
        get_current_logger().warning(f"Image file not found for embedding: {image_path}")
        return image_path  # Return original path if file doesn't exist

    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            encoded_data = base64.b64encode(image_data).decode('utf-8')

            # Determine MIME type based on file extension
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.gif'):
                mime_type = 'image/gif'
            elif image_path.lower().endswith('.svg'):
                mime_type = 'image/svg+xml'
            else:
                mime_type = 'image/png'  # Default fallback

            return f"data:{mime_type};base64,{encoded_data}"
    except Exception as e:
        get_current_logger().error(f"Failed to embed image {image_path}: {e}")
        return image_path  # Return original path on error

# ================================================================
# ATOMIC & CHECKPOINTING UTILITIES (Phase 1 Implementation)
# ================================================================

class RobustCheckpointer:
    """
    Manages the state of a run by tracking completed work units.
    This enables robust stopping and resuming of the analysis.
    """
    def __init__(self, classifier_run_dir):
        self.run_dir = classifier_run_dir
        self.results_dir = os.path.join(self.run_dir, "work_unit_results")
        self.checkpoint_file = os.path.join(self.run_dir, "completed_work_units.json")
        os.makedirs(self.results_dir, exist_ok=True)

    def _work_unit_to_str(self, work_unit):
        """Converts a work unit tuple to a standardized string for lookups."""
        return f"L{work_unit[0]}_{work_unit[1]}_{work_unit[2]}"

    def save_work_unit_result(self, work_unit, result_data):
        """Atomically saves the result for a single work unit."""
        unit_str = self._work_unit_to_str(work_unit)
        result_filepath = os.path.join(self.results_dir, f"{unit_str}.json")
        atomic_save_json(result_data, result_filepath)
        
        # Also update the main checkpoint file for quick lookups
        completed = self.load_completed_work_units()
        completed.add(unit_str)
        atomic_save_json(list(completed), self.checkpoint_file)

    def load_completed_work_units(self):
        """Loads the set of completed work unit strings for fast filtering."""
        if not os.path.exists(self.checkpoint_file):
            return set()
        try:
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            # If checkpoint is corrupted, rebuild it from saved result files
            get_current_logger().warning("Checkpoint file corrupted. Rebuilding from result files...")
            completed = {os.path.splitext(f)[0] for f in os.listdir(self.results_dir) if f.endswith('.json')}
            atomic_save_json(list(completed), self.checkpoint_file)
            return completed

# ================================================================
# DATA DISCOVERY & LOADING (Phase 1 Implementation)
# ================================================================

def generate_work_units(chunk_directories):
    """Discover all unique (layer, hook, scheme) combinations from activation files.
    
    Scans HDF5 metadata only (not activation data) to identify all work units
    (layer/hook/scheme combinations) available for classifier training. Handles
    both consolidated and individual batch files, with smart duplicate detection.
    
    Args:
        chunk_directories (list of str): List of chunk directory paths containing HDF5 files.
        
    Returns:
        tuple: (work_units, all_h5_files) where:
            - work_units: set of (layer, hook, scheme) tuples
            - all_h5_files: list of file paths to process
            
    Raises:
        FileNotFoundError: If no HDF5 activation files found in any chunk directory.
        
    Notes:
        - Filters out layers in SKIP_LAYERS global variable
        - Excludes attention patterns (hook_pattern, hook_attn_scores)
        - Prioritizes consolidated files over individual batches
        - Only reads HDF5 metadata, not activation arrays (fast)
    """
    get_current_logger().info("Discovering all work units from activation files...")
    work_units = set()
    all_h5_files = []

    get_current_logger().info(f"Will skip layers: {sorted(SKIP_LAYERS)}")

    for chunk_dir in chunk_directories:
        # Consolidate file discovery logic here
        consolidated = glob.glob(os.path.join(chunk_dir, "consolidated_activations_*.h5"))
        all_h5_files.extend(consolidated)
        
        consolidated_indices = set()
        for f in consolidated:
            match = re.search(r'consolidated_activations_(\d+)_(\d+).h5', os.path.basename(f))
            if match:
                consolidated_indices.update(range(int(match.group(1)), int(match.group(2))))
        
        individual = glob.glob(os.path.join(chunk_dir, "*_batch_*_activations_*.h5"))
        for f in individual:
            match = re.search(r'_batch_(\d+)_activations', os.path.basename(f))
            if match and int(match.group(1)) not in consolidated_indices:
                all_h5_files.append(f)

    if not all_h5_files:
        raise FileNotFoundError("No HDF5 activation files found in any chunk directory.")

    # Scan files for metadata
    for file_path in tqdm(all_h5_files, desc="Scanning H5 metadata", unit="file"):
        try:
            with h5py.File(file_path, 'r') as f:
                for layer_key in f.keys():
                    if not layer_key.startswith('layer_'): continue
                    
                    parts = layer_key.split('_', 2)
                    if len(parts) < 3: continue
                    
                    layer_num, hook_name = int(parts[1]), parts[2]

                    # Skip specified layers
                    if layer_num in SKIP_LAYERS:
                        continue

                    # Exclude attention patterns as they are not used for classification
                    if 'attn.hook_pattern' in hook_name or 'attn.hook_attn_scores' in hook_name:
                        continue

                    for scheme in f[layer_key].keys():
                        work_units.add((layer_num, hook_name, scheme))
        except Exception as e:
            get_current_logger().warning(f"Could not read metadata from {os.path.basename(file_path)}: {e}")

    work_units = sorted(list(work_units))

    # Log information about skipped layers
    if SKIP_LAYERS:
        get_current_logger().info(f"Skipped layers: {sorted(SKIP_LAYERS)}")
        get_current_logger().info(f"Discovered {len(work_units)} unique work units to process (excluding skipped layers)")
    else:
        get_current_logger().info(f"Discovered {len(work_units)} unique work units to process")

    # Apply production mode limits
    if PRODUCTION_MAX_BATCHES is not None or PRODUCTION_MAX_BATCH_FILES is not None:
        original_work_units = len(work_units)
        original_files = len(all_h5_files)

        if PRODUCTION_MAX_BATCHES is not None and len(work_units) > PRODUCTION_MAX_BATCHES:
            work_units = work_units[:PRODUCTION_MAX_BATCHES]

        if PRODUCTION_MAX_BATCH_FILES is not None and len(all_h5_files) > PRODUCTION_MAX_BATCH_FILES:
            all_h5_files = all_h5_files[:PRODUCTION_MAX_BATCH_FILES]

        # Report the production limits applied
        if (PRODUCTION_MAX_BATCHES is not None and len(work_units) < original_work_units) or \
           (PRODUCTION_MAX_BATCH_FILES is not None and len(all_h5_files) < original_files):
            logger.info("PRODUCTION MODE LIMITS APPLIED:")
            if PRODUCTION_MAX_BATCHES is not None:
                logger.info(f"Work Units: {original_work_units} → {len(work_units)} ({PRODUCTION_MAX_BATCHES} limit)")
            if PRODUCTION_MAX_BATCH_FILES is not None:
                logger.info(f"H5 Files: {original_files} → {len(all_h5_files)} ({PRODUCTION_MAX_BATCH_FILES} limit)")

    return work_units, all_h5_files

def load_single_activation_group(work_unit, all_h5_files):
    """Lazily load activation data for a single work unit across all HDF5 files.
    
    Performs targeted loading of activations for one specific (layer, hook, scheme)
    combination by scanning all HDF5 files and combining data. This is the core
    worker function for parallel activation loading.
    
    Args:
        work_unit (tuple): (layer, hook, scheme) identifying the activation group to load.
        all_h5_files (list of str): List of HDF5 file paths to search.
        
    Returns:
        dict: Activation group with keys:
            - 'layer': int, layer number
            - 'hook': str, hook name
            - 'scheme': str, aggregation scheme
            - 'activations': np.ndarray, combined flattened activations
            - 'row_indices': np.ndarray, corresponding row indices
            - 'evaluation_data': dict (empty, labels come from pickle results)
            
    Raises:
        ValueError: If no activation data found for this work unit.
        
    Notes:
        - Validates shape consistency between activations and row_indices
        - Warns on missing datasets or shape mismatches but continues
        - Uses concatenate_and_flatten_streaming() for memory efficiency
        - Called by worker processes in parallel processing pool
    """
    layer, hook, scheme = work_unit
    key_str = f"layer_{layer}_{hook}"

    activations_list = []
    row_indices_list = []
    # Evaluation arrays in HDF5 are no longer used; keep only activations and indices

    # Iterate through all files to find chunks of data for this specific work unit
    for file_path in all_h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                if key_str in f and scheme in f[key_str]:
                    data = f[key_str][scheme]

                    # Validate HDF5 data integrity
                    if 'activations' not in data or 'row_indices' not in data:
                        get_current_logger().warning(f"Missing required datasets in {file_path}")
                        continue

                    # Check for shape consistency
                    activations_shape = data['activations'].shape
                    row_indices_shape = data['row_indices'].shape

                    if activations_shape[0] != row_indices_shape[0]:
                        get_current_logger().warning(f"Shape mismatch in {file_path}: activations={activations_shape}, row_indices={row_indices_shape}")
                        continue

                    # Load all available samples for this work unit

                    # Load all available samples for this work unit
                    activations_list.append(data['activations'][:])
                    row_indices_list.append(data['row_indices'][:])


        except Exception as e:
            get_current_logger().warning(f"Error reading {file_path}: {e}")
            continue

    if not activations_list:
        raise ValueError(f"No activation data found for work unit {work_unit}")

    # Concatenate and flatten the data from all files for this work unit
    combined_activations = concatenate_and_flatten_streaming(activations_list)
    combined_row_indices = np.concatenate(row_indices_list, axis=0)

    # No evaluation data returned; labels come from pickle results

    return {
        'layer': layer,
        'hook': hook,
        'scheme': scheme,
        'activations': combined_activations,
        'row_indices': combined_row_indices,
        'evaluation_data': {}
    }

def process_single_work_unit(args):
    """
    Main worker function for parallel processing of a single activation group (layer/hook/scheme combination).
    
    This function handles the complete pipeline for one work unit:
    1. Loads activation data for the specified layer/hook/scheme
    2. Aligns features with labels from results DataFrame 
    3. Creates GPU-optimized XGBoost classifier with imbalance handling
    4. Trains and evaluates the classifier with threshold optimization
    5. Saves results and trained model for this work unit
    
    Args:
        args (tuple): Contains (work_unit, all_h5_files, results_df, classifiers, compute_resources)
            - work_unit (tuple): (layer_id, hook_name, token_scheme) identifier
            - all_h5_files (list): List of paths to activation HDF5 files
            - results_df (pd.DataFrame): DataFrame with labels and sample metadata
            - classifiers (dict): Dictionary of classifier objects to train
            - compute_resources (dict): CPU/GPU resource allocation information
    
    Returns:
        tuple: (work_unit, results_dict) where results_dict contains:
            - 'status': 'completed' or 'skipped' or 'failed'
            - 'reason': Explanation if skipped/failed
            - 'best_model_data': Trained model package with metadata (if successful)
            - 'metrics': Performance metrics dictionary
    
    GPU Assignment:
        Automatically assigns GPU from GPU_IDS list based on worker index for parallel multi-GPU training
    
    Data Leakage Prevention:
        All train/test splits and scaling are performed within this function to avoid leakage
    """
    work_unit, all_h5_files, results_df, classifiers, compute_resources = args
    try:
        # MULTI-GPU: Get worker process index and assign GPU from GPU_IDS list
        # This allows parallel workers to use different GPUs
        worker_id = args[0] if isinstance(args[0], int) else hash(work_unit) % len(GPU_IDS)
        gpu_id = GPU_IDS[worker_id % len(GPU_IDS)]
        
        # Log GPU assignment
        logger.info(f"WORKER ASSIGNED: GPU {gpu_id} (from GPU_IDS={GPU_IDS})")
        
        # 1. Lazy Load Data: Load only the data needed for this specific task.
        activation_group = load_single_activation_group(work_unit, all_h5_files)

        # 2. Align Features and Labels
        X, y = align_features_labels(activation_group, results_df)

        if X is None or y is None or len(X) == 0:
            return work_unit, {"status": "skipped", "reason": "No valid data after alignment."}

        # Calculate and display data distribution information
        unique_classes, counts = np.unique(y, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique_classes, counts)}
        hallucinated_count = class_dist.get(1, 0)  # Label 1 = HALLUCINATION
        not_hallucinated_count = class_dist.get(0, 0)  # Label 0 = NO HALLUCINATION
        total_samples = len(y)

        logger.info(f"PROCESSING: L{work_unit[0]:02d} | {work_unit[1]:20s} | {work_unit[2]:15s} | GPU: {gpu_id} | DATA: {total_samples} samples ({hallucinated_count} hallucinated, {not_hallucinated_count} not hallucinated) | FEATURES: {X.shape[1]}")

        # 3. Get GPU-specific classifiers for this work unit with error handling
        try:
            # CRITICAL: Create XGBoost with imbalance parameters calculated from TRAINING DATA ONLY
            X_temp, _, y_temp, _ = train_test_split(
                X, y, test_size=0.2, random_state=GLOBAL_RANDOM_SEED, stratify=y
            )

            # Create imbalance-aware XGBoost using training data only on assigned GPU
            xgb_classifier = {'XGBoost_GPU': create_imbalanced_xgboost(y_temp, gpu_id)}

            logger.info(f"TRAINING: XGBoost GPU training on GPU {gpu_id} for work unit (L{work_unit[0]:02d}_{work_unit[1]}_{work_unit[2]})")

            gc.collect()

            group_results, best_classifier_data = train_xgboost_parallel_optimized(
                xgb_classifier, X, y,
                compute_resources=compute_resources,
                work_unit=work_unit,  # Pass work unit for better logging
                gpu_id=gpu_id  # Pass GPU ID for logging
            )
            
        except Exception as gpu_error:
            logger.error(f"GPU training failed for GPU {gpu_id}: {gpu_error}")
            return work_unit, {"status": "error", "reason": f"GPU training failed on GPU {gpu_id}: {str(gpu_error)}"}
        
        # 4. Package and Return Results
        return work_unit, {
            "status": "success",
            "group_results": group_results,
            "best_classifier_data": best_classifier_data
        }

    except Exception as e:
        import traceback
        error_msg = f"Failed to process work unit {work_unit}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return work_unit, {"status": "error", "reason": error_msg}

# ================================================================
# CHECKPOINT CONFIGURATION
# ================================================================
# Set to a specific classifier run ID to continue (e.g., "classifier_run_1753945404")
# Leave empty to start a new analysis run.
CONTINUE_FROM_RUN = ""

# ================================================================
# PRODUCTION MODE CONFIGURATION
# ================================================================
# Production Mode Limits are now configured in config.py

def print_production_status():
    """Print production mode configuration"""
    logger.info("PRODUCTION MODE: Full dataset processing enabled")
    if PRODUCTION_MAX_BATCHES is not None or PRODUCTION_MAX_BATCH_FILES is not None:
        logger.info("PRODUCTION LIMITS APPLIED:")
        if PRODUCTION_MAX_BATCHES is not None:
            logger.info(f"Work Units: Limited to {PRODUCTION_MAX_BATCHES}")
        if PRODUCTION_MAX_BATCH_FILES is not None:
            logger.info(f"H5 Files: Limited to {PRODUCTION_MAX_BATCH_FILES}")
    else:
        logger.info("No work unit or file limits applied")
    logger.info("=" * 60)

# Initialize production status
print_production_status()


# ================================================================
# 2. Path Configuration
# ================================================================
# Paths are now configured in config.py and imported above

# --- FALLBACK METHOD ---
# These are now populated automatically but can be used as a manual override if EXPERIMENT_ID is blank.
CHUNK_DIRECTORIES = []

# Globals that will be dynamically set in main() after checking for existing runs
RUN_ID = None
CLASSIFIER_DIR = None

# NEW: Global variable to store the classifier directory path for evaluate.py to use
CLASSIFIER_EXPERIMENT_DIR = None  # Will be set to ACTIVATIONS_BASE_DIR/EXPERIMENT_ID/classifier_run_ID

# ================================================================
# CONFIGURATION OPTIONS
# ================================================================
# All configurations are now centralized in config.py and imported above


# ================================================================
# CHECKPOINTING FUNCTIONS
# ================================================================


def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Skip keys that are known to contain trained models
            if k in ['trained_model', 'trained_scaler', 'model', 'scaler']:
                continue
            # Skip keys that start with '_' (internal attributes)
            if k.startswith('_'):
                continue
            try:
                result[k] = make_json_serializable(v)
            except (TypeError, ValueError):
                # If serialization fails, replace with type name
                result[k] = str(type(v).__name__)
        return result
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, list, dict, type(None))):
        # Complex object - replace with type name
        return str(type(obj).__name__)
    else:
        return obj

def atomic_save_json(data, filepath):
    """Atomically save data to JSON file with automatic serialization.
    
    Converts data to JSON-serializable format and writes atomically to prevent
    corruption. Handles numpy arrays, dataframes, and other non-native types.
    
    Args:
        data (dict or list): Data structure to save as JSON.
        filepath (str): Target file path for the JSON file.
        
    Returns:
        bool: True if save succeeded, False if failed.
        
    Notes:
        - Automatically converts non-serializable types via make_json_serializable()
        - Uses indent=2 for human-readable formatting
        - Cleans up temp file on error
    """
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w') as f:
            # Convert to JSON-serializable format
            serializable_data = make_json_serializable(data)
            json.dump(serializable_data, f, indent=2)
        os.rename(temp_filepath, filepath)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return False

def atomic_save_csv(df, filepath):
    """Atomically save pandas DataFrame to CSV file.
    
    Uses atomic write-then-rename to ensure no partial writes or corruption.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Target file path for the CSV file.
        
    Returns:
        bool: True if save succeeded, False if failed.
        
    Notes:
        - Uses os.replace() for atomic rename on all platforms
        - Index is excluded from CSV output
        - Cleans up temp file on error
    """
    temp_file = filepath + '.tmp'
    try:
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, filepath)
        return True
    except Exception as e:
        logger.error(f"Atomic CSV save failed for {filepath}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False


# ================================================================
# CONFIG METADATA MANAGEMENT - AUDIT TRAIL
# ================================================================

def load_generator_metadata(chunk_directories):
    """
    Load generator configuration from chunk directories.
    Generator saves run_metadata.json in each chunk folder.
    
    Args:
        chunk_directories (dict): Dictionary of chunk directories
        
    Returns:
        dict: Generator configuration metadata, or None if not found
    """
    if not chunk_directories:
        logger.warning("No chunk directories provided. Cannot load generator metadata.")
        return None
    
    # Try to load from the first chunk directory
    first_chunk_dir = list(chunk_directories.values())[0]
    metadata_path = os.path.join(first_chunk_dir, 'run_metadata.json')
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Generator metadata not found at: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            generator_config = json.load(f)
        logger.info(f"Loaded generator metadata from: {os.path.relpath(metadata_path)}")
        return generator_config
    except Exception as e:
        logger.error(f"Failed to load generator metadata: {e}")
        return None

def create_classifier_metadata(run_id, generator_config=None):
    """
    Create comprehensive classifier configuration metadata.
    Combines generator config with classifier-specific settings.
    
    Args:
        run_id (str): Classifier run ID
        generator_config (dict): Generator configuration (optional)
        
    Returns:
        dict: Complete classifier metadata
    """
    from datetime import datetime
    
    classifier_config = {
        'classifier_info': {
            'run_id': run_id,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experiment_id': EXPERIMENT_ID,
            'activations_base_dir': ACTIVATIONS_BASE_DIR,
            'classifier_base_dir': CLASSIFIER_BASE_DIR,
        },
        'classifier_settings': {
            'global_random_seed': GLOBAL_RANDOM_SEED,
            'finalist_count': FINALIST_COUNT,
            'imbalance_threshold': IMBALANCE_THRESHOLD,
            'imbalance_handling_enabled': bool(HANDLE_IMBALANCE),
            'probe_threshold_mode': PROBE_THRESHOLD_MODE,
            'probe_quantiles': PROBE_QUANTILES if isinstance(PROBE_QUANTILES, list) else list(PROBE_QUANTILES),
            'probe_fixed_thresholds': PROBE_FIXED_THRESHOLDS if isinstance(PROBE_FIXED_THRESHOLDS, list) else list(PROBE_FIXED_THRESHOLDS),
            'skip_layers': list(SKIP_LAYERS),
            'custom_scoring': CUSTOM_SCORING if isinstance(CUSTOM_SCORING, dict) else {},
            'production_mode': {
                'max_batches': PRODUCTION_MAX_BATCHES,
                'max_batch_files': PRODUCTION_MAX_BATCH_FILES,
            }
        },
        'generator_config': generator_config if generator_config else {},
    }
    
    return classifier_config

def save_classifier_metadata(classifier_dir, metadata):
    """
    Save classifier metadata to classifier run directory.
    
    Args:
        classifier_dir (str): Classifier run directory
        metadata (dict): Metadata to save
    """
    metadata_path = os.path.join(classifier_dir, 'classifier_config_metadata.json')
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved classifier metadata to: {os.path.relpath(metadata_path)}")
        return metadata_path
    except Exception as e:
        logger.error(f"Failed to save classifier metadata: {e}")
        return None

# ================================================================
# Memory Management Helper Functions - SIMPLIFIED
# ================================================================

def detect_compute_resources():
    """
    Detect available compute resources and configure for multi-GPU XGBoost training.
    
    GPU-Optimized Strategy (from command-line args):
    - W = work-unit workers (optimized for NUM_GPUS from --num-gpus)
    - C = classifiers per work-unit (always 1, XGBoost only)
    - T = minimal CPU threads (GPU does the heavy lifting)
    
    Uses GPU_IDS from --gpu-ids to determine which GPUs to use
    """
    # CPU information
    cpu_count = mp.cpu_count()
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    # GPU CONFIGURATION: From command-line arguments
    actual_num_gpus = len(GPU_IDS)  # Use actual number of GPUs specified via --gpu-ids
    configured_num_gpus = NUM_GPUS   # Expected number from --num-gpus
    
    # GPU-OPTIMIZED PARALLELISM: Configure based on GPU_IDS
    W = actual_num_gpus * 1  # 1 worker per GPU for reliable operation
    C = 1  # XGBoost only
    T = 1  # Minimal CPU threads since GPU handles computation
    
    # Ensure we don't exceed available cores, but prioritize GPU utilization
    W = min(W, logical_cores - 2)  # Reserve cores for system
    
    logger.info(f"    GPU-OPTIMIZED PARALLELISM CONFIGURATION (from command-line args):")
    logger.info(f"   CPU: {physical_cores} physical, {logical_cores} logical cores")
    logger.info(f"   Configured GPUs (--num-gpus): {configured_num_gpus}")
    logger.info(f"   Actual GPUs (--gpu-ids): {GPU_IDS} ({actual_num_gpus} GPUs)")
    logger.info(f"   W = {W} work-unit workers ({W//actual_num_gpus if actual_num_gpus > 0 else 0} per GPU)")
    logger.info(f"   C = {C} XGBoost-only training per work-unit")
    logger.info(f"   T = {T} minimal CPU threads (GPU-focused)")
    logger.info(f"   Total compute slots: W×C×T = {W}×{C}×{T} = {W*C*T}")
    logger.info(f"   Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB total")

    return {
        'cpu_count': cpu_count,
        'logical_cores': logical_cores,
        'physical_cores': physical_cores,
        'group_jobs': W,              # GPU-optimized work-unit parallelism
        'optimal_jobs': C,            # XGBoost only
        'ml_algorithm_jobs': T,       # Minimal CPU threads
        'num_gpus': actual_num_gpus,  # Number of GPUs to actually use (from GPU_IDS)
        'gpu_ids': GPU_IDS            # List of specific GPU IDs to use
    }


# ================================================================
# 3. Data Discovery and Loading Functions - STREAMLINED VERSION
# ================================================================

def concatenate_and_flatten_streaming(activations_list):
    """
    Memory-efficient concatenation with streaming to avoid intermediate copies.
    
    This function eliminates the memory-killing pattern of:
    1. Concatenate all chunks into one huge array
    2. Then flatten that huge array (creating another copy)
    
    Instead, it flattens each chunk individually and pre-allocates the final array.
    Memory usage: 3x → 1x (67% reduction)
    """
    if not activations_list:
        return np.array([])
    
    # Pre-calculate total size to avoid reallocation during concatenation
    total_samples = sum(arr.shape[0] for arr in activations_list)
    
    # Flatten first chunk to determine feature dimensions
    first_chunk = activations_list[0]
    if len(first_chunk.shape) > 2:
        # Multi-dimensional: flatten to 2D
        flattened_features = first_chunk.reshape(first_chunk.shape[0], -1).shape[1]
    else:
        # Already 2D
        flattened_features = first_chunk.shape[1]
    
    # Pre-allocate final output array (single allocation, no intermediate copies)
    result = np.empty((total_samples, flattened_features), dtype=first_chunk.dtype)
    
    # Fill output array in-place, flattening each chunk individually
    current_idx = 0
    for chunk in activations_list:
        if len(chunk.shape) > 2:
            # Flatten chunk (small memory footprint per chunk)
            flattened_chunk = chunk.reshape(chunk.shape[0], -1)
        else:
            flattened_chunk = chunk
            
        # Copy flattened chunk directly into final array
        end_idx = current_idx + flattened_chunk.shape[0]
        result[current_idx:end_idx] = flattened_chunk
        current_idx = end_idx
    
    return result

def discover_experiment_chunks(experiment_dir):
    """
    Discovers all chunk folders (e.g., 'chunk_0', 'chunk_1') within a
    single, specified experiment directory.
    """
    if not experiment_dir or not os.path.isdir(experiment_dir):
        logger.error(f"ERROR: Experiment directory not found or not specified: {experiment_dir}")
        logger.error("Please set the EXPERIMENT_ID at the top of the script.")
        return {}

    chunk_pattern = os.path.join(experiment_dir, "chunk_*")
    chunk_dirs = [d for d in glob.glob(chunk_pattern) if os.path.isdir(d)]

    if not chunk_dirs:
        logger.error(f"ERROR: No chunk directories found in {experiment_dir}")
        return {}

    logger.info(f"SUCCESS: Found {len(chunk_dirs)} data chunks for experiment {EXPERIMENT_ID}.")
    # Return a dictionary where keys are chunk names and values are full paths
    return {os.path.basename(d): d for d in chunk_dirs}

def load_multi_chunk_results_data(chunk_directories, run_info):
    """
    MODIFIED: Robustly loads results data from multiple chunk directories by
    prioritizing the most complete result files available.
    """
    logger.info(f"Loading results data from {len(chunk_directories)} chunk directories...")
    all_results = []
    
    for chunk_dir in chunk_directories:


        logger.info(f"Processing results from chunk: {os.path.basename(chunk_dir)}")
        
        files_to_load = []
        # 1. Prioritize the single, final 'all_results.pkl' file.
        final_results_file = os.path.join(chunk_dir, "all_results.pkl")
        


        if os.path.exists(final_results_file):
            files_to_load = [final_results_file]
            logger.info(f"Found final summary file: {os.path.basename(final_results_file)}")
        else:
            # 2. If not found, look for consolidated files.
            consolidated_pattern = os.path.join(chunk_dir, "consolidated_results_*.pkl")

            consolidated_files = glob.glob(consolidated_pattern)
            if consolidated_files:
                files_to_load = consolidated_files
                logger.info(f"Found {len(files_to_load)} consolidated result files.")
            else:
                # 3. As a last resort, look for individual batch files.
                batch_pattern = os.path.join(chunk_dir, "batch_*_results.pkl")

                batch_files = glob.glob(batch_pattern)
                if batch_files:
                    files_to_load = batch_files
                    logger.info(f"Found {len(files_to_load)} individual batch result files.")

        if not files_to_load:
            logger.warning(f"No result files of any kind found in chunk directory: {chunk_dir}")

            continue
            
        # Load the data from the identified files with atomic loading
        for pickle_file in files_to_load:
            try:
                # Use atomic loading pattern with corruption recovery
                results_data = atomic_load_pickle(pickle_file)
                if results_data is None:
                    logger.warning(f"Failed to load {os.path.basename(pickle_file)} - skipping")
                    continue

                # Results can be a single list (from batch/consolidated) or a dict in some cases
                if isinstance(results_data, list):
                    all_results.extend(results_data)
                elif isinstance(results_data, dict):
                    all_results.append(results_data)

            except Exception as e:
                logger.warning(f"Error loading {os.path.basename(pickle_file)}: {e}")
                continue
    
    if not all_results:
        logger.error("No results loaded from any chunk directory")
        return pd.DataFrame()

    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(all_results)

    # VALIDATION: Ensure row_idx exists and normalize label columns
    required_cols = ['row_idx']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in results data: {missing_cols}")
        logger.debug(f"Available columns: {list(results_df.columns)}")
        return pd.DataFrame()

    # Normalize label columns to ensure is_hallucination exists
    if 'is_hallucination' not in results_df.columns:
        if 'is_correct' in results_df.columns:
            # Map: correct(1) → hallucination(0); incorrect(0) → hallucination(1); keep 2
            results_df['is_hallucination'] = results_df['is_correct'].map(lambda v: 0 if v == 1 else (1 if v == 0 else 2))
        else:
            logger.error("Missing both is_hallucination and is_correct in results data")
            logger.debug(f"Available columns: {list(results_df.columns)}")
            return pd.DataFrame()

    # Drop duplicates just in case a mix of consolidated/batch files caused overlap
    initial_count = len(results_df)
    results_df = results_df.drop_duplicates(subset=['row_idx'])
    duplicates_removed = initial_count - len(results_df)

    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate records")

    # Report data quality metrics
    evaluation_counts = results_df['is_hallucination'].value_counts()
    logger.info(f"Loaded and combined {len(results_df)} unique result records.")
    logger.info(f"Evaluation distribution: {dict(evaluation_counts)}")

    # Store class distribution for later use
    class_distribution = {
        'total': len(results_df),
        'hallucination': int(evaluation_counts.get(1, 0)),
        'non_hallucination': int(evaluation_counts.get(0, 0)),
        'unlabeled': int(evaluation_counts.get(2, 0))
    }

    return results_df, class_distribution

def align_features_labels(activation_group, results_df):
    """Align activation features with hallucination labels and filter invalid samples.
    
    Matches activations to their corresponding labels from results DataFrame using
    row indices. Implements robust filtering to exclude API failures (label=2) and
    missing data from training. Performs vectorized operations for efficiency.
    
    Args:
        activation_group (dict): Activation data with keys:
            - 'layer': int, layer number
            - 'hook': str, hook name
            - 'scheme': str, aggregation scheme
            - 'activations': np.ndarray, activation features
            - 'row_indices': np.ndarray, corresponding row IDs
        results_df (pd.DataFrame): Evaluation results with columns:
            - 'row_idx': int, sample identifier (required)
            - 'is_hallucination' or 'is_correct': int, label (0, 1, or 2 for API fail)
            
    Returns:
        tuple: (valid_activations, labels) where:
            - valid_activations: np.ndarray, filtered activation features
            - labels: np.ndarray, corresponding binary labels (0=correct, 1=hallucination)
            Returns (empty array, empty array) if no valid samples remain.
            
    Raises:
        ValueError: If shape mismatch or missing required columns in results_df.
        
    Notes:
        - Excludes API failures (label=2) from training data
        - Normalizes is_correct to is_hallucination format if needed
        - Logs statistics on filtered samples (API failures, not found)
        - Uses vectorized operations via dict lookup for performance
    """

    work_unit = (activation_group['layer'], activation_group['hook'], activation_group['scheme'])

    activations = activation_group['activations']
    row_indices = activation_group['row_indices']

    # VALIDATION: Ensure data consistency
    if len(activations) != len(row_indices):
        raise ValueError(f"Shape mismatch: activations ({len(activations)}) vs row_indices ({len(row_indices)})")

    if len(activations) == 0:
        get_current_logger().warning("Empty activation group")
        return np.array([]), np.array([])

    # Validate required columns exist in results_df (row_idx required; prefer is_hallucination)
    required_cols = ['row_idx']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in results_df: {missing_cols}")

    # OPTIMIZED: Use vectorized operations instead of loops
    # Create a mapping for faster lookups. Prefer is_hallucination; fallback to is_correct
    has_h = 'is_hallucination' in results_df.columns
    has_c = 'is_correct' in results_df.columns
    source_col = 'is_hallucination' if has_h else 'is_correct'
    results_dict = results_df.set_index('row_idx')[source_col].to_dict()

    # Vectorized label assignment with API failure filtering

    # Vectorized label assignment with API failure filtering
    labels = []
    valid_mask = []
    api_failures_skipped = 0
    not_found_skipped = 0

    for i, row_idx in enumerate(row_indices):
        if row_idx in results_dict:
            v = results_dict[row_idx]

            # Normalize to hallucination label if source was is_correct
            if source_col == 'is_correct' and v in (0, 1):
                v = 0 if v == 1 else 1

            # Only process valid evaluations (0 or 1), exclude API failures (2)
            if v in (0, 1):
                label = v  # 1 = hallucination, 0 = non-hallucination
                labels.append(label)
                valid_mask.append(i)
            elif v == 2:
                # Skip API failures - don't include in training data
                api_failures_skipped += 1
                continue
            else:
                # Skip any other unexpected values
                get_current_logger().warning(f"Unexpected is_correct value {v} for row {row_idx}, skipping")
                continue
        else:
            # Row index not found in results
            not_found_skipped += 1
            continue

    # Report filtering statistics

    # Report API failures that were excluded from training
    if api_failures_skipped > 0:
        logger.info(f"Excluded {api_failures_skipped} API failures from training for this activation group")

    if not valid_mask:
        logger.debug(f"No valid samples after alignment for {work_unit}")
        return np.array([]), np.array([])

    # Filter activations to only include valid indices
    valid_activations = activations[valid_mask]
    labels = np.array(labels)

    return valid_activations, labels

# ================================================================
# AUTO-DETECTION OF IMBALANCE HANDLING (Phase 1)
# ================================================================

def auto_detect_imbalance_handling(class_distribution, threshold=IMBALANCE_THRESHOLD):
    """
    Automatically decide if imbalance handling should be enabled based on
    hallucination fraction from the pre-computed class distribution.
    
    Decision Logic:
    - Use pre-computed class distribution counts (hallucination, non_hallucination, unlabeled)
    - Calculate hallucination fraction from labeled samples only
    - If hallucination fraction < threshold → enable imbalance handling
    - Otherwise → disable imbalance handling
    
    Args:
        class_distribution (dict): Pre-computed class distribution dict with keys:
            - 'hallucination': count of hallucinations (label 1)
            - 'non_hallucination': count of non-hallucinations (label 0)
            - 'unlabeled': count of unlabeled/API failures (label 2)
            - 'total': total samples
        threshold (float): Hallucination fraction threshold (default 0.40 from config)
        
    Returns:
        bool: True if imbalance handling should be enabled, False otherwise
    """
    global HANDLE_IMBALANCE
    
    try:
        # Validate class_distribution has required keys
        if not isinstance(class_distribution, dict):
            logger.warning("class_distribution is not a dictionary. Defaulting to False.")
            HANDLE_IMBALANCE = False
            return False
        
        required_keys = ['hallucination', 'non_hallucination', 'unlabeled']
        missing_keys = [k for k in required_keys if k not in class_distribution]
        if missing_keys:
            logger.warning(f"class_distribution missing keys: {missing_keys}. Defaulting to False.")
            HANDLE_IMBALANCE = False
            return False
        
        # Extract counts from class_distribution
        n_hallucination = int(class_distribution['hallucination'])
        n_non_hallucination = int(class_distribution['non_hallucination'])
        n_unlabeled = int(class_distribution['unlabeled'])
        total_labeled = n_hallucination + n_non_hallucination
        
        if total_labeled == 0:
            logger.warning("No labeled samples available in class_distribution. Defaulting to False.")
            HANDLE_IMBALANCE = False
            return False
        
        # Calculate hallucination fraction: what proportion of labeled samples are hallucinations?
        # If hallucinations are < threshold (e.g., 0.30), the minority class needs special handling
        hallucination_fraction = n_hallucination / total_labeled
        
        # Calculate imbalance ratio: how many times more non-hallucinations than hallucinations?
        # Example: 700 non-halluc / 300 halluc = 2.33x imbalance
        # Higher ratio = more severe imbalance
        if n_hallucination > 0:
            imbalance_ratio = n_non_hallucination / n_hallucination
        else:
            imbalance_ratio = float('inf')
        
        # Make decision: enable imbalance handling if hallucination_fraction < threshold
        # Logic: If hallucinations are rare (< 30%), we need SMOTE/class_weight to boost minority class
        auto_enable = hallucination_fraction < float(threshold)
        HANDLE_IMBALANCE = bool(auto_enable)
        
        # Log decision with detailed reasoning
        logger.info("=" * 70)
        logger.info("AUTO-DETECTION OF IMBALANCE HANDLING:")
        logger.info("=" * 70)
        logger.info(f"  Total labeled samples:           {total_labeled:,}")
        logger.info(f"  Hallucinations (label 1):        {n_hallucination:,} ({hallucination_fraction*100:.2f}%)")
        logger.info(f"  Non-Hallucinations (label 0):    {n_non_hallucination:,} ({(1-hallucination_fraction)*100:.2f}%)")
        
        if n_unlabeled > 0:
            logger.info(f"  Unlabeled/API Failures (label 2): {n_unlabeled:,} (excluded from decision)")
        
        logger.info(f"  Imbalance ratio (majority:minority): {imbalance_ratio:.2f}x")
        logger.info(f"  Hallucination threshold:        {threshold:.3f}")
        logger.info(f"  Decision: hallucination_fraction ({hallucination_fraction:.3f}) < threshold ({threshold})?")
        logger.info(f"  AUTO-DECISION: HANDLE_IMBALANCE = {HANDLE_IMBALANCE}")
        
        if HANDLE_IMBALANCE:
            logger.info("  ✓ IMBALANCE HANDLING ENABLED (using PR-AUC metric and scale_pos_weight)")
        else:
            logger.info("  ✓ IMBALANCE HANDLING DISABLED (using standard ROC-AUC metric)")
        
        logger.info("=" * 70)
        
        return HANDLE_IMBALANCE
        
    except Exception as e:
        logger.warning(f"Auto-detection of imbalance handling failed: {str(e)}")
        logger.warning(f"Defaulting HANDLE_IMBALANCE to False")
        HANDLE_IMBALANCE = False
        return False

# ================================================================
# 4. Classifier Definitions
# ================================================================

def calculate_scale_pos_weight(y):
    """
    Calculate scale_pos_weight for XGBoost binary classification.
    Adaptive to whichever class is positive (always 1) and majority/minority.

    Args:
        y: Target labels where 1 is always the positive class

    Returns:
        float: XGBoost-recommended scale_pos_weight = n_neg / n_pos
    """
    unique, counts = np.unique(y, return_counts=True)
    cnt = dict(zip(unique, counts))
    n_pos = cnt.get(1, 0)           # positive = 1 (hallucinations)
    n_neg = len(y) - n_pos          # negative = everything else (non-hallucinations)

    if n_pos == 0:
        logger.warning("No positive samples (label 1) found in training data!")
        return 1.0

    scale_pos_weight = n_neg / n_pos

    # Enhanced logging for verification
    logger.info("=" * 50)
    logger.info("SCALE_POS_WEIGHT CALCULATION VERIFICATION:")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(y)}")
    logger.info(f"Label 0 (Non-hallucinations): {n_neg} samples ({n_neg/len(y)*100:.1f}%)")
    logger.info(f"Label 1 (Hallucinations): {n_pos} samples ({n_pos/len(y)*100:.1f}%)")
    logger.info(f"Imbalance ratio: {n_neg/n_pos:.2f} (majority:minority)")
    logger.info(f"XGBoost scale_pos_weight = n_neg/n_pos = {n_neg}/{n_pos} = {scale_pos_weight:.3f}")
    logger.info("=" * 50)

    # Verify the calculation makes sense
    if scale_pos_weight < 1.0:
        logger.warning(f"UNEXPECTED: scale_pos_weight ({scale_pos_weight:.3f}) < 1.0!")
        logger.warning("This means label 1 is actually the MAJORITY class, not minority!")
        logger.warning("Double-check your label assignment: Label 1 should be HALLUCINATIONS (minority)")

    return scale_pos_weight

def create_imbalanced_xgboost(y_train, gpu_id=0):
    """Create XGBoost classifier with imbalance parameters calculated from training data only"""
    
    # Only calculate scale_pos_weight if HANDLE_IMBALANCE is True
    if HANDLE_IMBALANCE:
        scale_pos_weight = calculate_scale_pos_weight(y_train)
        logger.info(f"IMBALANCE CALCULATION (training data only): scale_pos_weight = {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = None
        logger.info(f"IMBALANCE HANDLING DISABLED: scale_pos_weight will not be used")

    return create_imbalanced_xgboost_from_params(
        scale_pos_weight=scale_pos_weight,
        gpu_id=gpu_id
    )

def create_imbalanced_xgboost_from_params(scale_pos_weight=None, gpu_id=0, n_estimators=None, learning_rate=None, max_bin=None):
    """Create XGBoost classifier with unified configuration - supports both imbalance-aware and standard modes"""

    # Determine eval_metric. Use PR-AUC ('aucpr') when handling class imbalance
    # to better reflect performance on the minority class; otherwise use ROC-AUC ('auc').
    # NOTE: Avoid unsupported custom metric names. Custom scoring is handled
    # separately by `attach_custom_score` on Python-side after predictions.
    eval_metric = 'aucpr' if HANDLE_IMBALANCE else 'auc'

    xgb_params = {
        'random_state': GLOBAL_RANDOM_SEED,

        "objective":"binary:logistic",
        'eval_metric': eval_metric,

        'n_estimators': 4000,
         'early_stopping_rounds': 50,
        'max_depth': 8,
        'learning_rate': 0.50,

        'n_jobs': 1,
        'device': f'cuda:{gpu_id}',
        'tree_method': 'hist',

        'max_bin': 256,
        'colsample_bytree': 0.8,
        'colsample_bynode':0.80,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'max_delta_step': 1.0,
    }

    # Override defaults with provided parameters
    if n_estimators is not None:
        xgb_params['n_estimators'] = n_estimators
    if learning_rate is not None:
        xgb_params['learning_rate'] = learning_rate
    if max_bin is not None:
        xgb_params['max_bin'] = max_bin

    # Only add scale_pos_weight if HANDLE_IMBALANCE is True
    if HANDLE_IMBALANCE and scale_pos_weight is not None:
        xgb_params['scale_pos_weight'] = scale_pos_weight
        logger.info(f"XGBoost CONFIG: Imbalance-aware mode enabled (HANDLE_IMBALANCE=True)")
        logger.info(f"XGBoost CONFIG: scale_pos_weight={scale_pos_weight:.2f}")

    if HANDLE_IMBALANCE and scale_pos_weight is not None:
        # VERIFICATION: Ensure the parameter is actually being applied
        logger.info("=" * 50)
        logger.info("XGBoost SCALE_POS_WEIGHT VERIFICATION:")
        logger.info("=" * 50)
        logger.info(f"scale_pos_weight parameter set to: {scale_pos_weight:.3f}")
        logger.info(f"This means XGBoost will give {scale_pos_weight:.1f}x more weight to positive samples (label 1)")
        logger.info(f"Expected behavior: Better recall for hallucinations (label 1) at cost of precision")
        logger.info("=" * 50)
    else:
        if HANDLE_IMBALANCE and scale_pos_weight is None:
            logger.warning("XGBoost CONFIG: HANDLE_IMBALANCE=True but scale_pos_weight is None")
        logger.info("XGBoost CONFIG: Standard mode (no imbalance correction)")
        logger.info("WARNING: Without scale_pos_weight, XGBoost may under-perform on minority class (hallucinations)")

    # Log chosen eval metric after configuring imbalance handling
    logger.info(f"XGBoost CONFIG: Using {eval_metric} as eval_metric for early stopping/evaluation")

    # Create the classifier
    classifier = XGBClassifier(**xgb_params)

    # FINAL VERIFICATION: Check that the parameter was actually applied
    applied_scale_pos_weight = classifier.get_params().get('scale_pos_weight')
    applied_eval_metric = classifier.get_params().get('eval_metric')
    
    logger.info(f"Applied eval_metric: {applied_eval_metric}")
    
    if HANDLE_IMBALANCE and scale_pos_weight is not None:
        if abs(applied_scale_pos_weight - scale_pos_weight) > 1e-6:
            logger.error(f"PARAMETER MISMATCH: Requested scale_pos_weight={scale_pos_weight:.3f}, but classifier has {applied_scale_pos_weight:.3f}")
        else:
            logger.info(f"VERIFIED: XGBoost classifier has scale_pos_weight={applied_scale_pos_weight:.3f}")
    else:
        if applied_scale_pos_weight is not None:
            logger.warning(f"UNEXPECTED: Classifier has scale_pos_weight={applied_scale_pos_weight} even though imbalance handling is disabled")

    return classifier

def get_classifiers(class_weights=None, compute_resources=None, gpu_id=None, scale_pos_weight=None):
    """
    Define XGBoost classifier optimized for multi-GPU training.

    Multi-GPU Configuration (from config.py):
    - Uses gpu_hist for maximum GPU utilization
    - Minimal CPU threads since GPU handles computation
    - Dynamic GPU assignment from GPU_IDS list
    - Each worker process gets its own GPU from GPU_IDS

    Args:
        gpu_id: GPU ID to use (from GPU_IDS in config.py)
        scale_pos_weight: If provided, uses create_imbalanced_xgboost logic.
                         If None, uses default XGBoost configuration.
    """
    if class_weights is None:
        class_weights = 'balanced'  # Default to balanced class weights

    if compute_resources is None:
        compute_resources = {'ml_algorithm_jobs': 2, 'num_gpus': len(GPU_IDS)}

    # MULTI-GPU: Minimal CPU threads, let GPU do the work
    T = compute_resources.get('ml_algorithm_jobs', 2)
    num_gpus = compute_resources.get('num_gpus', len(GPU_IDS))

    # If gpu_id not specified, use GPU 0 from GPU_IDS (fallback)
    if gpu_id is None:
        gpu_id = GPU_IDS[0] if GPU_IDS else 0

    logger.info(f"MULTI-GPU CONFIG: XGBoost using GPU {gpu_id} (from GPU_IDS={GPU_IDS}), T={T} CPU threads, Total GPUs: {num_gpus}")

    # XGBoost Multi-GPU Configuration (from config.py GPU_IDS)
    xgb_classifier = {
        'XGBoost_GPU': create_imbalanced_xgboost_from_params(
            scale_pos_weight=scale_pos_weight,
            gpu_id=gpu_id,
        ),
    }

    # Return XGBoost classifier
    classifiers = xgb_classifier

    return classifiers



# ================================================================
# 4.5. Custom Scoring Helper Functions
# ================================================================

# ================================================================
# 4.5. Custom Scoring Helper Functions (now imported from helpers)
# ================================================================
# attach_custom_score is now imported from helpers.custom_scoring
# See: helpers/custom_scoring.py for implementation details

# ================================================================
# 5. Training and Evaluation Functions
# ================================================================

def train_xgboost_parallel_optimized(classifiers, X, y, compute_resources, work_unit=None, gpu_id=0):
    """
    Train and evaluate XGBoost classifiers with GPU acceleration and threshold optimization.
    
    Performs single-pass training with proxy threshold optimization on validation set.
    Uses 60/20/20 train/val/test split to prevent data leakage. All scaling is performed
    after split to ensure validation/test data never influences training.
    
    Args:
        classifiers (dict): Dictionary of classifier objects to train (typically {'XGBoost_GPU': xgb_classifier})
        X (np.ndarray): Feature matrix of shape (n_samples, n_features) - unscaled activations
        y (np.ndarray): Target labels (0=correct, 1=hallucination)
        compute_resources (dict): Resource allocation info including logical_cores, group_jobs
        work_unit (tuple, optional): (layer, hook, scheme) identifier for logging. Defaults to None.
        gpu_id (int, optional): GPU device ID for training. Defaults to 0.
    
    Returns:
        dict: Results dictionary containing:
            - For each classifier name:
                - 'accuracy': Test set accuracy
                - 'f1': F1 score
                - 'auroc': Area under ROC curve
                - 'confusion_matrix': 2x2 confusion matrix 
                - 'optimal_threshold': Best threshold from validation set
                - 'custom_score': Custom scoring metric (if HANDLE_IMBALANCE=True)
                - 'trained_model': Fitted classifier object
                - 'scaler': Fitted StandardScaler object
                - 'hyperparameters': Model hyperparameters
    
    Data Split Strategy:
        - 60% training (for model fitting)
        - 20% validation (for threshold optimization)
        - 20% test (for final evaluation)
    
    Threshold Optimization:
        Performs proxy optimization on validation set using probe quantiles/fixed thresholds
        to find optimal decision boundary for imbalanced classification.
    
    Notes:
        - StandardScaler fitted only on training data to prevent leakage
        - LINALG threads automatically configured based on compute_resources
        - Supports custom scoring for imbalanced datasets via HANDLE_IMBALANCE config
        - Expensive full threshold sweep removed for performance
    """

    # Prepare data splits: train/val/test (60%/20%/20%) to prevent data leakage
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=GLOBAL_RANDOM_SEED, stratify=y
    )

    # Second split: separate train and validation from temp (75%/25% of remaining = 60%/20% overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=GLOBAL_RANDOM_SEED, stratify=y_temp
    )

    # Log data split sizes for transparency
    logger.info(f"DATA SPLIT SIZES: Train: {len(X_train)} ({len(X_train)/len(X):.1%}) | "
                f"Val: {len(X_val)} ({len(X_val)/len(X):.1%}) | "
                f"Test: {len(X_test)} ({len(X_test)/len(X):.1%}) | "
                f"Total: {len(X)}")

    # Log class distribution for imbalance analysis (training data only)
    unique_classes, counts = np.unique(y_train, return_counts=True)
    if len(counts) == 2:
        imbalance_ratio = max(counts) / min(counts)
        logger.info(f"TRAINING DATA CLASS DISTRIBUTION: {dict(zip(unique_classes, counts))} (ratio: {imbalance_ratio:.1f})")

    # Scale features AFTER split to prevent data leakage
    # Fit scaler ONLY on training data, then transform val and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute per-process linalg threads
    total_cores = compute_resources.get('logical_cores', 96)
    n_workers   = compute_resources.get('group_jobs', 1)
    linalg_threads = max(1, min(LINALG_THREADS, total_cores) // max(1, n_workers))

    logger.info(f"LINALG threads per process: {linalg_threads}; workers: {n_workers}")

    # Use baseline data with StandardScaler only
    original_dim = X_train_scaled.shape[1]

    logger.info("Using baseline preprocessing with StandardScaler only")

    # Track the best classifier for this work unit using custom scoring at optimal threshold
    best_classifier_data = None
    best_selector_score = 0.0

    # Train XGBoost on baseline data
    results = {}
    xgb_results = {}  # Store XGBoost results for threshold-aware selection

    # Loop over the provided classifiers (will only be XGBoost_GPU in this setup)
    for clf_name, clf in classifiers.items():
        start_time = time.time()
        try:
            # Use baseline data
            X_train_var = X_train_scaled
            X_val_var = X_val_scaled
            X_test_var = X_test_scaled
            preprocessing = 'StandardScaler only'

            # ENHANCED DIAGNOSTIC LOGGING
            logger.info(f"DIAGNOSTICS [{clf_name}] [{preprocessing}]:")
            logger.info(f"  - X_train shape: {X_train_var.shape}, dtype: {X_train_var.dtype}")
            logger.info(f"  - y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            logger.info(f"  - XGBoost Device Param: {clf.get_params().get('device')}")
            logger.info(f"  - XGBoost Tree Method: {clf.get_params().get('tree_method')}")
            logger.info(f"  - XGBoost GPU ID: {clf.get_params().get('gpu_id')}")
            logger.info(f"  - XGBoost scale_pos_weight: {clf.get_params().get('scale_pos_weight')}")

            # Verify class distribution in training data
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            train_class_dist = dict(zip(unique_train, counts_train))
            logger.info(f"  - Training class distribution: {train_class_dist}")

            # Expected scale_pos_weight calculation
            n_pos_train = train_class_dist.get(1, 0)
            n_neg_train = train_class_dist.get(0, 0)
            expected_spw = n_neg_train / n_pos_train if n_pos_train > 0 else 1.0
            # Defensive handling: XGBoost may have scale_pos_weight explicitly set to None
            # or not include the parameter at all. Normalize to a numeric value for safe
            # arithmetic and logging (default to 1.0 when absent/None).
            actual_spw_raw = clf.get_params().get('scale_pos_weight', None)
            try:
                actual_spw = float(actual_spw_raw) if actual_spw_raw is not None else 1.0
            except Exception:
                actual_spw = 1.0

            if abs(expected_spw - actual_spw) > 1e-6:
                logger.warning(f"SCALE_POS_WEIGHT MISMATCH: Expected {expected_spw:.3f}, Actual {actual_spw:.3f}")
            else:
                logger.info(f"✓ SCALE_POS_WEIGHT VERIFIED: {actual_spw:.3f} matches expected calculation")

            # GPU training progress with work unit context
            work_unit_str = f"L{work_unit[0]:02d}_{work_unit[1]}_{work_unit[2]}" if work_unit else "unknown"
            logger.info(f"GPU TRAINING [{work_unit_str}] [GPU-{gpu_id}] [{clf_name}] [{preprocessing}] Starting XGBoost GPU training...")
            clf.fit(X_train_var, y_train, eval_set=[(X_val_var, y_val)])
            logger.info(f"GPU TRAINING [{work_unit_str}] [GPU-{gpu_id}] [{clf_name}] [{preprocessing}] Training complete, making predictions...")
            y_pred = clf.predict(X_test_var)
            y_pred_proba = clf.predict_proba(X_test_var)[:, 1]
            y_train_proba = clf.predict_proba(X_train_var)[:, 1]

            # POST-TRAINING VERIFICATION: Check if imbalance handling is working
            if clf.get_params().get('scale_pos_weight') is not None:
                logger.info("=" * 50)
                logger.info("POST-TRAINING IMBALANCE VERIFICATION:")
                logger.info("=" * 50)

                # Check training predictions distribution
                train_pred_pos = np.sum(y_train_proba > 0.5)
                train_actual_pos = np.sum(y_train == 1)
                logger.info(f"Training predictions: {train_pred_pos}/{len(y_train)} positive (threshold=0.5)")
                logger.info(f"Training actual: {train_actual_pos}/{len(y_train)} positive")
                logger.info(f"Training prediction ratio: {train_pred_pos/train_actual_pos:.2f}x predicted vs actual positives")

                # Check test predictions distribution
                test_pred_pos = np.sum(y_pred_proba > 0.5)
                test_actual_pos = np.sum(y_test == 1)
                logger.info(f"Test predictions: {test_pred_pos}/{len(y_test)} positive (threshold=0.5)")
                logger.info(f"Test actual: {test_actual_pos}/{len(y_test)} positive")
                logger.info(f"Test prediction ratio: {test_pred_pos/test_actual_pos:.2f}x predicted vs actual positives")

                # Expected behavior with scale_pos_weight: More positive predictions
                # Normalize scale_pos_weight to float to avoid None arithmetic
                spw_raw = clf.get_params().get('scale_pos_weight', None)
                try:
                    scale_pos_weight = float(spw_raw) if spw_raw is not None else 1.0
                except Exception:
                    scale_pos_weight = 1.0

                # Avoid division by zero when no positive samples
                actual_ratio = (train_pred_pos / train_actual_pos) if train_actual_pos > 0 else float('inf')
                expected_ratio = scale_pos_weight * 0.8  # Some tolerance for learning dynamics
                if actual_ratio > expected_ratio:
                    logger.info(f"✓ IMBALANCE HANDLING WORKING: Positive predictions increased by {actual_ratio:.1f}x")
                    logger.info(f"  (Expected ~{expected_ratio:.1f}x with scale_pos_weight={scale_pos_weight:.1f})")
                else:
                    logger.warning(f"IMBALANCE HANDLING MAY NOT BE WORKING: Only {actual_ratio:.1f}x positive predictions")
                    logger.warning(f"  (Expected ~{expected_ratio:.1f}x with scale_pos_weight={scale_pos_weight:.1f})")
                logger.info("=" * 50)

            # Compute metrics (optimized set)
            auroc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Per-class F1 scores
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
            f1_hallucinated = f1_per_class[1] if len(f1_per_class) > 1 else 0
            f1_not_hallucinated = f1_per_class[0] if len(f1_per_class) > 0 else 0

            # Additional metrics
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

            # Per-class precision and recall
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            
            precision_hallucinated = precision_per_class[1] if len(precision_per_class) > 1 else 0
            precision_not_hallucinated = precision_per_class[0] if len(precision_per_class) > 0 else 0
            recall_hallucinated = recall_per_class[1] if len(recall_per_class) > 1 else 0
            recall_not_hallucinated = recall_per_class[0] if len(recall_per_class) > 0 else 0

            # Initialize confusion matrix variables (will be set with placeholder threshold below)
            tn = fp = fn = tp = 0

            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)

            # Cache validation and test probabilities for later threshold optimization
            proba_val = clf.predict_proba(X_val_var)[:, 1]
            proba_test = clf.predict_proba(X_test_var)[:, 1]

            # Stash these values to add to metrics later (fix KeyError)
            cached_val_proba = proba_val
            cached_test_proba = proba_test
            cached_y_val = y_val
            cached_y_test = y_test
            
            logger.info(f"THRESHOLD OPT: Cached validation and test probabilities for later optimization")

            # We'll evaluate test metrics after selecting the best model with its optimal threshold
            # For now, just use 0.5 as a placeholder for metrics calculation
            placeholder_threshold = 0.5
            logger.info(f"PLACEHOLDER EVAL: Calculating initial metrics with threshold {placeholder_threshold:.3f}")
            y_test_pred_placeholder = (y_pred_proba >= placeholder_threshold).astype(int)

            # Calculate metrics using placeholder threshold on test set
            cm_placeholder = confusion_matrix(y_test, y_test_pred_placeholder)

            # VERIFICATION: Ensure proper confusion matrix interpretation
            # Label 0 = NO HALLUCINATION, Label 1 = HALLUCINATION
            # cm[i,j] = count of samples with true label i and predicted label j

            # Handle different confusion matrix shapes and update metrics
            if cm_placeholder.shape == (2, 2):
                tn_opt, fp_opt, fn_opt, tp_opt = cm_placeholder.ravel()
                # Verify interpretation:
                # tn = cm[0,0] = predicted 0, actual 0 = correctly predicted non-hallucinations
                # fp = cm[0,1] = predicted 1, actual 0 = incorrectly predicted hallucinations (was non-hallucination)
                # fn = cm[1,0] = predicted 0, actual 1 = incorrectly predicted non-hallucinations (was hallucination)
                # tp = cm[1,1] = predicted 1, actual 1 = correctly predicted hallucinations

                # Update metrics with placeholder threshold performance
                precision_per_class_opt = precision_score(y_test, y_test_pred_placeholder, average=None, zero_division=0)
                recall_per_class_opt = recall_score(y_test, y_test_pred_placeholder, average=None, zero_division=0)

                precision_hallucinated = precision_per_class_opt[1] if len(precision_per_class_opt) > 1 else 0
                precision_not_hallucinated = precision_per_class_opt[0] if len(precision_per_class_opt) > 0 else 0
                recall_hallucinated = recall_per_class_opt[1] if len(recall_per_class_opt) > 1 else 0
                recall_not_hallucinated = recall_per_class_opt[0] if len(recall_per_class_opt) > 0 else 0

                # Update confusion matrix metrics
                tp, fp, tn, fn = tp_opt, fp_opt, tn_opt, fn_opt

            elif cm_placeholder.shape == (1, 1):
                # Only one class in test set
                if np.unique(y_test)[0] == 0:  # Only hallucinations
                    tn, fp, fn, tp = cm_placeholder[0, 0], 0, 0, 0
                else:  # Only non-hallucinations
                    tn, fp, fn, tp = 0, 0, 0, cm_placeholder[0, 0]
            else:
                # Fallback for unexpected shapes
                tn = fp = fn = tp = 0

            # VERIFICATION: Check that TP+FP+TN+FN = total test samples
            total_cm = tp + fp + tn + fn
            if total_cm != len(y_test):
                logger.warning(f"CONFUSION MATRIX WARNING: {clf_name}")
                logger.warning(f"CM sum ({total_cm}) != test size ({len(y_test)})")
                logger.warning(f"TP:{tp}, FP:{fp}, TN:{tn}, FN:{fn}")

            # VERIFICATION: Ensure labels are properly interpreted
            actual_non_hallucinations = np.sum(y_test == 0)  # Count of actual label 0 (non-hallucinations)
            actual_hallucinations = np.sum(y_test == 1)  # Count of actual label 1 (hallucinations)

            # Check if confusion matrix matches actual class counts
            cm_non_hallucinations = tn + fp  # TN + FP should equal actual non-hallucinations
            cm_hallucinations = tp + fn  # TP + FN should equal actual hallucinations

            if cm_non_hallucinations != actual_non_hallucinations:
                logger.warning(f"NON-HALLUCINATION COUNT MISMATCH: {clf_name}")
                logger.warning(f"Actual: {actual_non_hallucinations}, CM: {cm_non_hallucinations}")

            if cm_hallucinations != actual_hallucinations:
                logger.warning(f"HALLUCINATION COUNT MISMATCH: {clf_name}")
                logger.warning(f"Actual: {actual_hallucinations}, CM: {cm_hallucinations}")

            # AUTOMATIC OVERFITTING DETECTION: Calculate training performance
            y_train_proba = clf.predict_proba(X_train_var)[:, 1]
            train_auc = roc_auc_score(y_train, y_train_proba)

            # Simple train-test gap overfitting detection (using placeholder threshold)
            train_test_gap = train_auc - roc_auc_score(y_test, y_pred_proba)
            is_overfitting = (
                train_test_gap > 0.1 or     # 10% train-test gap
                train_auc > 0.98 or         # Suspiciously perfect training
                roc_auc_score(y_test, y_pred_proba) < 0.52  # Poor test performance
            )
            overfitting_confidence = min(1.0, max(0.0, train_test_gap * 10 + (max(0, train_auc - 0.95) * 20)))

            # Calculate ACTUAL class distribution (not assumed)
            unique_classes, counts = np.unique(y, return_counts=True)
            class_dist = {int(k): int(v) for k, v in zip(unique_classes, counts)}
            imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1.0

            # Determine actual minority/majority classes
            if len(counts) >= 2:
                minority_class = int(unique_classes[np.argmin(counts)])
                majority_class = int(unique_classes[np.argmax(counts)])
            else:
                minority_class = majority_class = int(unique_classes[0]) if len(unique_classes) > 0 else 0

            # Calculate single-point ROC metrics from test set performance at optimal threshold
            tpr_single = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
            fpr_single = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
            tnr_single = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
            fnr_single = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate (Miss rate)

            metrics = {
                'auroc': auroc,
                'train_auc': train_auc,  # Added for overfitting detection
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'f1_hallucinated': f1_hallucinated,
                'f1_not_hallucinated': f1_not_hallucinated,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'precision_hallucinated': precision_hallucinated,
                'precision_not_hallucinated': precision_not_hallucinated,
                'recall_hallucinated': recall_hallucinated,
                'recall_not_hallucinated': recall_not_hallucinated,
                'pr_auc': pr_auc,
                'n_train': len(X_train_var),
                'n_test': len(X_test_var),
                'imbalance_ratio': imbalance_ratio,
                # Record imbalance handling strategy used for this training run
                'imbalance_strategy': 'scale_pos_weight' if HANDLE_IMBALANCE else 'none',
                'imbalance_handling_applied': bool(HANDLE_IMBALANCE),

                # ADDED: Preprocessing information for comparison
                'preprocessing_method': preprocessing,
                'n_features_original': original_dim,
                'n_features_processed': X_train_var.shape[1],

                # THRESHOLD OPTIMIZATION RESULTS
                'optimal_threshold': 0.5,  # placeholder for non-finalists
                'custom_score_at_optimal': None,
                'threshold_analysis_performed': False,  # Will be set to True for finalists after Stage-2

                # AUTOMATIC OVERFITTING DETECTION METRICS
                'is_overfitting': bool(is_overfitting),
                'overfitting_confidence': float(overfitting_confidence),
                'train_test_gap': float(train_test_gap),
                'overfitting_method': 'train_test_gap',

                # ADDED: Confusion Matrix Metrics
                'true_positives': int(tp),      # Correctly predicted as hallucination
                'false_positives': int(fp),     # Incorrectly predicted as hallucination
                'true_negatives': int(tn),      # Correctly predicted as non-hallucination
                'false_negatives': int(fn),     # Incorrectly predicted as non-hallucination

                # ADDED: Class distribution info
                'total_not_hallucinated': int(class_dist.get(0, 0)),
                'total_hallucinated': int(class_dist.get(1, 0)),
                'minority_class': minority_class,
                'majority_class': majority_class,
                'class_distribution': str(class_dist),  # For debugging

                # ADDED: Store trained model and scaler for inference
                'trained_model': clf,
                'trained_scaler': scaler,

                # Cached values for threshold optimization
                'val_proba': cached_val_proba,
                'test_proba': cached_test_proba,
                'y_val': cached_y_val,
                'y_test': cached_y_test
            }

            # Apply custom scoring for model selection
            attach_custom_score(metrics)
            selector_score = metrics['custom_final_score']

            logger.info(f"TRAINING [{work_unit_str}] [{clf_name}] Training complete, metrics calculated")
            logger.info(f"{'':40s} | AUROC: {metrics['auroc']:.3f} | Custom Score: {selector_score:.3f}")
            results[clf_name] = metrics
            training_time = time.time() - start_time

            # Store XGBoost results for threshold-aware selection later
            xgb_results[clf_name] = metrics

            # Format output with GPU info
            clf_display = clf_name.replace('_GPU', '').replace('XGBoost', 'XGB')
            logger.info(f"SUCCESS [{work_unit_str}] GPU-{gpu_id} {clf_display:15s} | AUROC: {metrics['auroc']:.3f} | Bal.Acc: {metrics['balanced_accuracy']:.3f} | PR-AUC: {metrics['pr_auc']:.3f} | Time: {training_time:.1f}s")
            logger.info(f"{'':40s} | F1-Macro: {metrics['f1_macro']:.3f} | F1-Weighted: {metrics['f1_weighted']:.3f}")
            logger.info(f"{'':40s} | F1-Halluc: {metrics['f1_hallucinated']:.3f} | F1-NotHalluc: {metrics['f1_not_hallucinated']:.3f}")
            logger.info(f"{'':40s} | Prec-Halluc: {metrics.get('precision_hallucinated', 0):.3f} | Rec-Halluc: {metrics.get('recall_hallucinated', 0):.3f}")
            logger.info(f"{'':40s} | Custom: {metrics.get('custom_final_score', 0):.3f} | F1-beta: {metrics.get('f1_beta', 0):.3f} | MCC: {metrics.get('mcc', 0):.3f}")

        except Exception as e:
            logger.error(f"ERROR {clf_name}: Failed to train - {e}")

    # ---- MODIFIED: Only perform Stage-1 proxy optimization ----
    # Stage-1: Fast, approximate threshold optimization using validation set
    # Purpose: Find rough optimal threshold without expensive test set evaluation
    logger.info("STAGE-1 (PROXY ONLY): Evaluating approximate custom score for this work unit...")
    
    for name, m in results.items():
        yv = m['y_val']; pv = m['val_proba']
        # Generate threshold candidates based on configured mode
        if PROBE_THRESHOLD_MODE == "quantiles":
            # Quantile mode: Use model-specific quantiles of its validation scores
            # Adapts to each model's probability distribution
            qs = np.quantile(pv, PROBE_QUANTILES)
            candidates = np.unique(np.clip(qs, 0.0, 1.0))
        else:
            # Fixed mode: Use pre-defined threshold values (e.g., 0.3, 0.4, 0.5, 0.6, 0.7)
            # Same thresholds tested for all models
            candidates = np.array(PROBE_FIXED_THRESHOLDS, dtype=float)

        best_proxy_score = -1e18
        best_proxy_t = 0.5

        # Test each threshold candidate to find the one maximizing custom score
        for t in candidates:
            # Apply threshold: predict hallucination (1) if probability >= t
            y_pred = (pv >= t).astype(int)
            # Calculate confusion matrix for this threshold
            cm = confusion_matrix(yv, y_pred, labels=[0,1])
            tn=fp=fn=tp=0
            if cm.shape == (2,2): tn, fp, fn, tp = cm.ravel()

            # Build minimal metrics dict for attach_custom_score
            # Only include confusion matrix and precision/recall (no AUROC, etc.)
            proxy_metrics = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
            }
            prec = precision_score(yv, y_pred, average=None, zero_division=0)
            rec  = recall_score(yv, y_pred, average=None, zero_division=0)
            if len(prec)==2:
                proxy_metrics['precision_not_hallucinated'] = float(prec[0])
                proxy_metrics['precision_hallucinated']     = float(prec[1])
            if len(rec)==2:
                proxy_metrics['recall_not_hallucinated']    = float(rec[0])
                proxy_metrics['recall_hallucinated']        = float(rec[1])

            attach_custom_score(proxy_metrics)
            s = proxy_metrics.get('custom_final_score', -1e18)
            if s > best_proxy_score:
                best_proxy_score, best_proxy_t = s, float(t)

        m['proxy_custom_score'] = float(best_proxy_score)
        m['proxy_threshold']    = float(best_proxy_t)
        # Set final scores to proxy scores for now
        m['optimal_threshold'] = float(best_proxy_t)
        m['custom_score_at_optimal'] = float(best_proxy_score)
        m['is_finalist'] = False # This will be determined globally later
        m['threshold_analysis_performed'] = False

        logger.info(f"PROXY: {name} | Best Proxy Score: {best_proxy_score:.4f} | Proxy Threshold: {best_proxy_t:.3f}")

    # ---- REMOVED: Stage-2 and Winner selection logic is now handled globally ----

    # Find the best classifier from this work unit
    if not results:
        return {}, None
    
    # Use custom score if HANDLE_IMBALANCE is True, otherwise use AUROC
    selector_key = 'proxy_custom_score' if HANDLE_IMBALANCE else 'auroc'
    best_clf_name = max(results.keys(), key=lambda k: results[k].get(selector_key, -1e18))
    
    best_classifier_data = {
        'model': results[best_clf_name]['trained_model'],
        'scaler': results[best_clf_name]['trained_scaler'],
        'classifier_name': best_clf_name,
        'auroc': results[best_clf_name].get('auroc', None),
        'custom_final_score': results[best_clf_name].get('proxy_custom_score', 0.0),
        'optimal_threshold': results[best_clf_name].get('proxy_threshold', 0.5),
        'n_features': X.shape[1] if len(X.shape) > 1 else 1,
        'metrics': results[best_clf_name],
        'selection_method': 'custom_score' if HANDLE_IMBALANCE else 'auroc'
    }

    selector_value = best_classifier_data.get('custom_final_score' if HANDLE_IMBALANCE else 'auroc', 0.0)
    logger.info(f"WORK UNIT BEST: {best_clf_name} ({selector_key.upper()}: {selector_value:.4f})")
    
    return results, best_classifier_data

# ================================================================
# 6. Advanced Visualization and Analysis Functions
# ================================================================

def create_comprehensive_visualizations(results_df, output_dir):
    """Create comprehensive visualizations for the results - STREAMLINED with threshold analysis"""

    # Set style for better plots
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")

    # 1. Individual Performance Heatmaps

    # AUROC Heatmap by Layer and Hook
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot_auroc = results_df.pivot_table(values='auroc', index='layer', columns='hook', aggfunc='mean')
    sns.heatmap(pivot_auroc, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('AUROC by Layer and Hook Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_auroc_layer_hook.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Accuracy Heatmap by Layer and Hook
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot_acc = results_df.pivot_table(values='accuracy', index='layer', columns='hook', aggfunc='mean')
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title('Accuracy by Layer and Hook Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_accuracy_layer_hook.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # F1 Macro Heatmap by Layer and Token Scheme
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot_f1 = results_df.pivot_table(values='f1_macro', index='layer', columns='scheme', aggfunc='mean')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('F1 Macro by Layer and Token Scheme', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_f1_layer_scheme.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # AUROC Heatmap by Layer and Token Scheme
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot_auroc_scheme = results_df.pivot_table(values='auroc', index='layer', columns='scheme', aggfunc='mean')
    sns.heatmap(pivot_auroc_scheme, annot=True, fmt='.3f', cmap='plasma', ax=ax)
    ax.set_title('AUROC by Layer and Token Scheme', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_auroc_layer_scheme.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Custom Final Score Heatmap by Layer and Hook
    fig, ax = plt.subplots(figsize=(12, 10))
    if 'custom_final_score' in results_df.columns:
        pivot_custom = results_df.pivot_table(values='custom_final_score', index='layer', columns='hook', aggfunc='mean')
        sns.heatmap(pivot_custom, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('Custom Score by Layer and Hook Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_custom_score_layer_hook.png'), dpi=300, bbox_inches='tight')
    else:
        ax.text(0.5, 0.5, 'Custom Score\nNot Available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Custom Score by Layer and Hook Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_custom_score_layer_hook.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Custom Final Score Heatmap by Layer and Token Scheme
    fig, ax = plt.subplots(figsize=(12, 10))
    if 'custom_final_score' in results_df.columns:
        pivot_custom_scheme = results_df.pivot_table(values='custom_final_score', index='layer', columns='scheme', aggfunc='mean')
        sns.heatmap(pivot_custom_scheme, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('Custom Score by Layer and Token Scheme', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_custom_score_layer_scheme.png'), dpi=300, bbox_inches='tight')
    else:
        ax.text(0.5, 0.5, 'Custom Score\nNot Available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Custom Score by Layer and Token Scheme', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_custom_score_layer_scheme.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Performance Trends Across Layers (3x2 layout for 5 subplots)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    layer_stats = results_df.groupby('layer').agg({
        'auroc': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'f1_macro': ['mean', 'std']
    }).round(3)

    layers = layer_stats.index

    # AUROC trend
    axes[0,0].errorbar(layers, layer_stats[('auroc', 'mean')],
                      yerr=layer_stats[('auroc', 'std')], marker='o', capsize=5)
    axes[0,0].set_title('AUROC Across Layers', fontweight='bold')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('AUROC')
    axes[0,0].grid(True, alpha=0.3)

    # Accuracy trend
    axes[0,1].errorbar(layers, layer_stats[('accuracy', 'mean')],
                      yerr=layer_stats[('accuracy', 'std')], marker='s', capsize=5, color='orange')
    axes[0,1].set_title('Accuracy Across Layers', fontweight='bold')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].grid(True, alpha=0.3)

    # F1 trend
    axes[1,0].errorbar(layers, layer_stats[('f1_macro', 'mean')],
                      yerr=layer_stats[('f1_macro', 'std')], marker='^', capsize=5, color='green')
    axes[1,0].set_title('F1 Macro Across Layers', fontweight='bold')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('F1 Macro Score')
    axes[1,0].grid(True, alpha=0.3)

    # Combined metrics
    axes[1,1].plot(layers, layer_stats[('auroc', 'mean')], marker='o', label='AUROC')
    axes[1,1].plot(layers, layer_stats[('accuracy', 'mean')], marker='s', label='Accuracy')
    axes[1,1].plot(layers, layer_stats[('f1_macro', 'mean')], marker='^', label='F1 Macro')

    # Add custom score to combined metrics if available
    if 'custom_final_score' in results_df.columns:
        custom_stats_combined = results_df.groupby('layer')['custom_final_score'].agg(['mean'])
        axes[1,1].plot(layers, custom_stats_combined['mean'], marker='D', linewidth=3, color='purple', label='Custom Score')

    axes[1,1].set_title('All Metrics Across Layers', fontweight='bold')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Score')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # Custom Score trend
    if 'custom_final_score' in results_df.columns:
        custom_stats = results_df.groupby('layer')['custom_final_score'].agg(['mean', 'std'])
        axes[2,0].errorbar(layers, custom_stats['mean'],
                          yerr=custom_stats['std'], marker='D', capsize=5, color='purple')
        axes[2,0].set_title('Custom Score Across Layers', fontweight='bold')
        axes[2,0].set_xlabel('Layer')
        axes[2,0].set_ylabel('Custom Score')
        axes[2,0].grid(True, alpha=0.3)
    else:
        axes[2,0].text(0.5, 0.5, 'Custom Score\nNot Available', ha='center', va='center', transform=axes[2,0].transAxes)
        axes[2,0].set_title('Custom Score Across Layers', fontweight='bold')

    # Hide the last subplot
    axes[2,1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_trends.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Custom Score vs Traditional Metrics Scatter Plot
    if 'custom_final_score' in results_df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Check if MCC gate values are available for sizing
        if 'gate_mcc_gamma' in results_df.columns:
            # Use MCC gate for point size (scale to reasonable sizes)
            sizes = results_df['gate_mcc_gamma'] * 100 + 20  # Scale MCC gate to pixel sizes
            size_label = 'MCC Gate Value'
        else:
            # Fallback to fixed size if MCC gate not available
            sizes = 50
            size_label = 'Fixed Size'

        # Scatter plot: Custom Score vs AUROC
        scatter = ax.scatter(results_df['custom_final_score'], results_df['auroc'],
                           c=results_df['f1_macro'], cmap='viridis', alpha=0.7, s=sizes)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Macro Score')

        # Add reference line (y=x)
        min_val = min(results_df['custom_final_score'].min(), results_df['auroc'].min())
        max_val = max(results_df['custom_final_score'].max(), results_df['auroc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal Performance')

        ax.set_xlabel('Custom Final Score', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Custom Score vs AUROC\n(Color: F1 Macro, Size: {size_label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add correlation coefficient
        corr = results_df['custom_final_score'].corr(results_df['auroc'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add custom score range info
        custom_range = results_df['custom_final_score'].max() - results_df['custom_final_score'].min()
        ax.text(0.05, 0.90, f'Custom Score Range: {custom_range:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'custom_vs_traditional_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


    # 5. Token Scheme Analysis (Traditional Metrics Only)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for plotting
    base_metrics = ['auroc', 'accuracy', 'f1_macro']
    scheme_performance = results_df.groupby('scheme')[base_metrics].mean()

    x = np.arange(len(scheme_performance.index))
    width = 0.25  # Standard width for 3 bars

    # Plot traditional metrics only
    ax.bar(x - width, scheme_performance['auroc'], width, label='AUROC', alpha=0.8, color='lightcoral')
    ax.bar(x, scheme_performance['accuracy'], width, label='Accuracy', alpha=0.8, color='skyblue')
    ax.bar(x + width, scheme_performance['f1_macro'], width, label='F1 Macro', alpha=0.8, color='lightgreen')

    ax.set_xlabel('Token Scheme', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_title('Performance by Token Scheme (Traditional Metrics)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scheme_performance.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_scheme_analysis_traditional.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 6. Custom Score Token Scheme Analysis (Separate Graph)
    if 'custom_final_score' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        custom_by_scheme = results_df.groupby('scheme')['custom_final_score'].mean().sort_values(ascending=False)

        bars = ax.bar(range(len(custom_by_scheme)), custom_by_scheme.values,
                     color='purple', alpha=0.8, width=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, custom_by_scheme.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Token Scheme', fontsize=12)
        ax.set_ylabel('Custom Score (Hallucination Detection)', fontsize=12)
        ax.set_title('Custom Score by Token Scheme\n(Hallucination Detection Performance)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(custom_by_scheme)))
        ax.set_xticklabels(custom_by_scheme.index, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add performance range annotation
        score_range = custom_by_scheme.max() - custom_by_scheme.min()
        ax.text(0.02, 0.98, f'Score Range: {score_range:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'token_scheme_analysis_custom_score.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 6. Best Classifier Multi-Metric Comparison
    if 'custom_final_score' in results_df.columns:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Find best classifier metrics
        best_idx = results_df['custom_final_score'].idxmax()
        best_metrics = results_df.loc[best_idx]

        # Metrics to display
        metric_names = ['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'precision_hallucinated',
                       'recall_hallucinated', 'precision_not_hallucinated', 'recall_not_hallucinated',
                       'custom_final_score']
        metric_labels = ['AUROC', 'Accuracy', 'F1 (Halluc)', 'F1 (No Halluc)', 'Precision (Halluc)',
                        'Recall (Halluc)', 'Precision (No Halluc)', 'Recall (No Halluc)', 'Custom Score']

        # Multi-metric radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Normalize metrics to 0-1 scale for comparison
        metric_values = []
        for metric in metric_names:
            if metric in best_metrics.index:
                value = best_metrics[metric]
                # Normalize based on typical ranges
                if metric == 'auroc':
                    normalized = value  # Already 0-1
                elif metric == 'accuracy':
                    normalized = value  # Already 0-1
                elif 'f1' in metric:
                    normalized = value  # Already 0-1
                elif 'precision' in metric or 'recall' in metric:
                    normalized = value  # Already 0-1
                elif metric == 'custom_final_score':
                    # Normalize custom score to 0-1 based on its range
                    custom_range = results_df['custom_final_score'].max() - results_df['custom_final_score'].min()
                    if custom_range > 0:
                        normalized = (value - results_df['custom_final_score'].min()) / custom_range
                    else:
                        normalized = 0.5
                else:
                    normalized = value
                metric_values.append(normalized)
            else:
                metric_values.append(0)

        metric_values += metric_values[:1]  # Close the circle

        # Plot radar chart
        ax1.plot(angles, metric_values, 'o-', linewidth=2, label='Best Classifier', color='purple')
        ax1.fill(angles, metric_values, alpha=0.25, color='purple')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metric_labels, fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.set_title('Best Classifier - Multi-Metric Performance Profile', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Custom Score Distribution Analysis
        custom_scores = results_df['custom_final_score'].dropna()

        # Histogram
        ax2.hist(custom_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(best_metrics['custom_final_score'], color='red', linestyle='--', linewidth=2,
                   label=f'Best: {best_metrics["custom_final_score"]:.3f}')
        ax2.set_xlabel('Custom Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Custom Score Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Trade-off Analysis: Custom Score vs AUROC
        scatter = ax3.scatter(results_df['custom_final_score'], results_df['auroc'],
                             c=results_df['f1_macro'], cmap='viridis', alpha=0.6, s=60)

        # Highlight best classifier
        ax3.scatter([best_metrics['custom_final_score']], [best_metrics['auroc']],
                   color='red', s=200, marker='*', edgecolor='black', linewidth=2,
                   label='Best Classifier')

        # Add reference lines
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='AUROC = 0.5')
        ax3.axvline(x=results_df['custom_final_score'].median(), color='purple', linestyle='--', alpha=0.5,
                   label='Median Custom Score')

        ax3.set_xlabel('Custom Score (Hallucination Detection)')
        ax3.set_ylabel('AUROC (General Performance)')
        ax3.set_title('Custom Score vs AUROC Trade-off Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('F1 Macro Score')

        # Performance Summary Table
        ax4.axis('off')
        summary_text = f"Best Classifier Summary:\n"
        summary_text += f"Layer: {best_metrics['layer']}\n"
        summary_text += f"Hook: {best_metrics['hook']}\n"
        summary_text += f"Scheme: {best_metrics['scheme']}\n"
        summary_text += f"AUROC: {best_metrics['auroc']:.3f}\n"
        summary_text += f"Custom Score: {best_metrics['custom_final_score']:.3f}\n"
        summary_text += f"F1 Macro: {best_metrics['f1_macro']:.3f}"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_classifier_multi_metric_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

def create_statistical_analysis(results_df, output_dir):
    """Perform statistical analysis of results"""

    # 1. Statistical significance tests between classifiers
    classifiers = results_df['classifier'].unique()
    metrics = ['auroc', 'accuracy', 'f1_macro']

    significance_results = []

    for metric in metrics:
        # Convert metric names to more readable format for display
        display_name = metric.upper()
        if metric == 'f1_macro':
            display_name = 'F1_MACRO'
        elif metric == 'auroc':
            display_name = 'AUROC'

        logger.info(f"STATISTICAL ANALYSIS: Statistical Analysis for {display_name}:")
        logger.info("=" * 50)

        for clf1, clf2 in itertools.combinations(classifiers, 2):
            group1 = results_df[results_df['classifier'] == clf1][metric]
            group2 = results_df[results_df['classifier'] == clf2][metric]

            if len(group1) > 5 and len(group2) > 5:  # Enough samples for test
                stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                significance_results.append({
                    'metric': metric,
                    'classifier_1': clf1,
                    'classifier_2': clf2,
                    'mean_1': group1.mean(),
                    'mean_2': group2.mean(),
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(group1.mean() - group2.mean())
                })

                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                logger.info(f"{clf1} vs {clf2}: p={p_value:.4f} {significance} (Δ={abs(group1.mean() - group2.mean()):.3f})")

    # Save statistical results
    sig_df = pd.DataFrame(significance_results)
    sig_df.to_csv(os.path.join(output_dir, 'statistical_analysis.csv'), index=False)

    # 2. Best performing combinations analysis
    logger.info(f"TOP PERFORMERS: TOP PERFORMING COMBINATIONS:")
    logger.info("=" * 50)

def generate_class_distribution_html(class_dist, border_color="#34495e"):
    """
    Generate HTML table for class distribution from precomputed class_dist dictionary.
    This eliminates code duplication between comprehensive and finalist reports.
    
    Args:
        class_dist (dict): Dictionary with keys 'total', 'hallucination', 'non_hallucination', 'unlabeled'
        border_color (str): Color for the total row border (default for comprehensive report)
        
    Returns:
        str: HTML string for the class distribution table
    """
    if class_dist is None:
        return ""
    
    total_samples = class_dist['total']
    hallucination_count = class_dist['hallucination']
    non_hallucination_count = class_dist['non_hallucination']
    unlabeled_count = class_dist['unlabeled']
    
    hallucination_pct = (hallucination_count / total_samples * 100) if total_samples > 0 else 0
    non_hallucination_pct = (non_hallucination_count / total_samples * 100) if total_samples > 0 else 0
    unlabeled_pct = (unlabeled_count / total_samples * 100) if total_samples > 0 else 0
    
    return f"""
        <div class="stats-box">
            <h2>Dataset Class Distribution</h2>
            <table class="class-dist-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Hallucination</td>
                        <td>{hallucination_count:,}</td>
                        <td>{hallucination_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Non-Hallucination</td>
                        <td>{non_hallucination_count:,}</td>
                        <td>{non_hallucination_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Unlabeled/API Failure</td>
                        <td>{unlabeled_count:,}</td>
                        <td>{unlabeled_pct:.2f}%</td>
                    </tr>
                    <tr style="border-top: 2px solid {border_color}; font-weight: bold;">
                        <td>Total</td>
                        <td>{total_samples:,}</td>
                        <td>100.00%</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

def create_comprehensive_analysis_report(results_df, output_dir, class_dist=None):
    """
    Create a comprehensive HTML report with all analyses, visualizations, and threshold optimization.
    
    This function generates the main analysis report for classifier experiments, combining
    statistical summaries, performance rankings, heatmaps, and threshold analysis plots
    into a single self-contained HTML file with embedded images.
    
    Args:
        results_df (pd.DataFrame): Complete results with columns:
                                   - layer, hook, scheme, classifier: Configuration
                                   - auroc, accuracy, f1_*: Performance metrics
                                   - custom_final_score, custom_score_at_optimal: Custom metrics
                                   - optimal_threshold: Best threshold for custom score
        output_dir (str): Directory containing visualization PNGs and where HTML will be saved
        class_dist (Dict, optional): Class distribution statistics:
                                     {'hallucinated': count, 'not_hallucinated': count, ...}
    
    Report Sections:
        1. Executive Summary: Best scores, experiment overview, configuration summary
        2. Best Configuration: Top performer with threshold optimization plot
        3. Top 10 Performers: Ranked table of best configurations
        4. Category Summaries: Aggregated stats by classifier/hook/scheme
        5. Performance Heatmaps: 6 heatmaps showing metrics across layers/hooks/schemes
        6. Detailed Graphs: Comprehensive visualizations gallery
        7. Class Distribution: Training data balance (if provided)
    
    Output Files:
        - comprehensive_report.html: Main report (self-contained with embedded images)
        
    Selection Logic:
        - If HANDLE_IMBALANCE=True: Best = max(custom_score_at_optimal or custom_final_score)
        - If HANDLE_IMBALANCE=False: Best = max(auroc)
        - Consistent with training phase optimization target
    
    Image Embedding:
        All visualization PNGs from output_dir are embedded as base64 within HTML
        for portability (no external file dependencies).
    
    Example:
        >>> results = pd.read_csv('./classifier_run/comprehensive_metrics.csv')
        >>> create_comprehensive_analysis_report(results, './classifier_run', class_dist)
        >>> # Opens comprehensive_report.html in browser to view results
    """

    # Determine best performer based on HANDLE_IMBALANCE setting (consistent with training phase)
    if HANDLE_IMBALANCE:
        # When imbalance handling is enabled, use custom scores for selection
        if 'custom_score_at_optimal' in results_df.columns and not results_df['custom_score_at_optimal'].isnull().all():
            best_metric = 'custom_score_at_optimal'
            best_value = results_df[best_metric].max()
            best_idx = results_df[best_metric].idxmax()
        elif 'custom_final_score' in results_df.columns:
            best_metric = 'custom_final_score'
            best_value = results_df[best_metric].max()
            best_idx = results_df[best_metric].idxmax()
        else:
            best_metric = 'auroc'
            best_value = results_df[best_metric].max()
            best_idx = results_df[best_metric].idxmax()
    else:
        # When imbalance handling is disabled, use AUROC for selection
        best_metric = 'auroc'
        best_value = results_df[best_metric].max()
        best_idx = results_df[best_metric].idxmax()

    # Helper function to create embedded image tags
    def create_embedded_img_tag(image_filename, alt_text="", style=""):
        """Create an HTML img tag with embedded base64 image data"""
        image_path = os.path.join(output_dir, image_filename)
        embedded_src = embed_image_as_base64(image_path)
        style_attr = f' style="{style}"' if style else ""
        return f'<img src="{embedded_src}" alt="{alt_text}"{style_attr}>'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Activation Classification Analysis Report - {EXPERIMENT_ID}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #34495e; color: white; }}
            .metric {{ font-weight: bold; color: #3498db; }}
            .highlight {{ background-color: #fff3cd; }}
            .graph-section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .graph-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            .stats-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 6px; margin: 10px 0; }}
            .custom-score {{ color: #9b59b6; font-weight: bold; }}
            .best-performance {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 4px; margin: 10px 0; }}
            .class-dist-table {{ width: auto; max-width: 800px; margin: 20px auto; }}
            .class-dist-table td {{ text-align: right; padding: 10px 15px; }}
            .class-dist-table th {{ text-align: center; padding: 10px 15px; }}
            .class-dist-table td:first-child {{ text-align: left; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Activation Classification Analysis Report - {EXPERIMENT_ID}</h1>
"""

    # Add class distribution table if class_dist is available
    html_content += generate_class_distribution_html(class_dist, border_color="#34495e")
    
    # Always add Executive Summary
    html_content += f"""
        <div class="stats-box">
            <h2>Executive Summary</h2>
            <ul>
                <li><strong>Experiment ID:</strong> {EXPERIMENT_ID}</li>
                <li><strong>Total Experiments:</strong> {len(results_df)}</li>
                <li><strong>Best AUROC:</strong> {results_df['auroc'].max():.3f}</li>
                <li><strong>Average AUROC:</strong> {results_df['auroc'].mean():.3f} ± {results_df['auroc'].std():.3f}</li>
"""

    # Determine which custom score column to use for executive summary
    custom_score_column = None
    if 'custom_score_at_optimal' in results_df.columns and not results_df['custom_score_at_optimal'].isnull().all():
        custom_score_column = 'custom_score_at_optimal'
        custom_score_label = 'Custom Score (at Optimal Threshold)'
    elif 'custom_final_score' in results_df.columns:
        custom_score_column = 'custom_final_score'
        custom_score_label = 'Custom Score'

    if custom_score_column:
        valid_scores = results_df[custom_score_column].dropna()
        html_content += f"""
                <li><strong>Best {custom_score_label}:</strong> <span class="custom-score">{valid_scores.max():.3f}</span></li>
                <li><strong>Average {custom_score_label}:</strong> <span class="custom-score">{valid_scores.mean():.3f} ± {valid_scores.std():.3f}</span></li>
"""

    html_content += f"""
                <li><strong>Layers Analyzed:</strong> {results_df['layer'].nunique()}</li>
                <li><strong>Hook Types:</strong> {', '.join(results_df['hook'].unique())}</li>
                <li><strong>Token Schemes:</strong> {', '.join(results_df['scheme'].unique())}</li>
                <li><strong>Classifiers:</strong> {', '.join(results_df['classifier'].unique())}</li>
            </ul>
        </div>

        <div class="best-performance">
            <h2>Best Performance</h2>
            <p><strong>Best Configuration:</strong> Layer {results_df.loc[best_idx, 'layer']}, Hook: {results_df.loc[best_idx, 'hook']}, Scheme: {results_df.loc[best_idx, 'scheme']}</p>
            <p><strong>Best Score:</strong> {best_value:.3f} ({best_metric.upper()})</p>
        <p><strong>Optimal Threshold:</strong> {results_df.loc[best_idx, 'optimal_threshold']:.3f} (Custom Score Maximizing)</p>
        </div>

        <div class="graph-section">
            <h2>Best Configuration - Threshold Optimization Analysis</h2>
            <div>
                {create_embedded_img_tag(f"threshold_optimization_L{results_df.loc[best_idx, 'layer']:02d}_{results_df.loc[best_idx, 'hook']}_{results_df.loc[best_idx, 'scheme']}.png", "Threshold Optimization for Best Configuration", "max-width: 100%; height: auto;")}
                <p><em>Detailed threshold optimization analysis for the best performing configuration (Layer {results_df.loc[best_idx, 'layer']}, {results_df.loc[best_idx, 'hook']}, {results_df.loc[best_idx, 'scheme']}). Shows how custom score, precision, recall, and other metrics vary across different threshold values. The optimal threshold ({results_df.loc[best_idx, 'optimal_threshold']:.3f}) maximizes the custom score for hallucination detection.</em></p>
            </div>
        </div>

        <h2>Top 10 Performers</h2>
        {results_df.nlargest(10, best_metric)[['layer', 'hook', 'scheme', 'classifier', 'auroc', 'optimal_threshold', 'custom_score_at_optimal', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated']].to_html(classes='highlight', index=False)}

        <h2>Performance by Category</h2>

        <h3>By Classifier</h3>
        {results_df.groupby('classifier')[['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'custom_final_score']].agg(['mean', 'std']).round(3).to_html()}

        <h3>By Hook Type</h3>
        {results_df.groupby('hook')[['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'custom_final_score']].agg(['mean', 'std']).round(3).to_html()}

        <h3>By Token Scheme</h3>
        {results_df.groupby('scheme')[['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'custom_final_score']].agg(['mean', 'std']).round(3).to_html()}

        <div class="graph-section">
            <h2>Performance Heatmaps</h2>
            <div class="graph-grid">
                <div>
                    <h3>AUROC by Layer and Hook Type</h3>
                    {create_embedded_img_tag("heatmap_auroc_layer_hook.png", "AUROC Heatmap")}
                </div>
                <div>
                    <h3>Accuracy by Layer and Hook Type</h3>
                    {create_embedded_img_tag("heatmap_accuracy_layer_hook.png", "Accuracy Heatmap")}
                </div>
                <div>
                    <h3>Custom Score by Layer and Hook Type</h3>
                    {create_embedded_img_tag("heatmap_custom_score_layer_hook.png", "Custom Score Heatmap")}
                </div>
                <div>
                    <h3>F1 Macro by Layer and Token Scheme</h3>
                    {create_embedded_img_tag("heatmap_f1_layer_scheme.png", "F1 Heatmap")}
                </div>
                <div>
                    <h3>AUROC by Layer and Token Scheme</h3>
                    {create_embedded_img_tag("heatmap_auroc_layer_scheme.png", "AUROC Scheme Heatmap")}
                </div>
                <div>
                    <h3>Custom Score by Layer and Token Scheme</h3>
                    {create_embedded_img_tag("heatmap_custom_score_layer_scheme.png", "Custom Score Scheme Heatmap")}
                </div>
            </div>
        </div>

        <div class="graph-section">
            <h2>Layer Performance Trends</h2>
            <div>
                {create_embedded_img_tag("layer_trends.png", "Layer Trends", "max-width: 100%; height: auto;")}
                <p><em>Individual metric trends across transformer layers with error bars, plus combined view including custom score.</em></p>
            </div>
        </div>

        <div class="graph-section">
            <h2>Custom Score vs Traditional Metrics</h2>
            <div>
                {create_embedded_img_tag("custom_vs_traditional_scatter.png", "Custom vs Traditional Scatter", "max-width: 100%; height: auto;")}
                <p><em>Scatter plot comparing custom score with AUROC. Point colors represent F1 Macro performance, point sizes represent MCC gate values.</em></p>
            </div>
        </div>

        <div class="graph-section">
            <h2>Token Scheme Analysis - Traditional Metrics</h2>
            <div>
                {create_embedded_img_tag("token_scheme_analysis_traditional.png", "Token Scheme Analysis Traditional", "max-width: 100%; height: auto;")}
                <p><em>Comparison of AUROC, Accuracy, and F1 Macro across different token schemes.</em></p>
            </div>
        </div>

        <div class="graph-section">
            <h2>Token Scheme Analysis - Custom Score</h2>
            <div>
                {create_embedded_img_tag("token_scheme_analysis_custom_score.png", "Token Scheme Analysis Custom Score", "max-width: 100%; height: auto;")}
                <p><em>Custom score performance across different token schemes, sorted by hallucination detection effectiveness.</em></p>
            </div>
        </div>

        <div class="graph-section">
            <h2>Best Classifier Multi-Metric Analysis</h2>
            <div>
                {create_embedded_img_tag("best_classifier_multi_metric_analysis.png", "Best Classifier Multi-Metric Analysis", "max-width: 100%; height: auto;")}
                <p><em>Comprehensive analysis of the best classifier including: (1) Multi-metric performance profile, (2) Custom score distribution with best classifier highlighted, (3) Custom score vs AUROC trade-off analysis, (4) Performance summary.</em></p>
            </div>
        </div>

        <h2>Key Insights</h2>
        <ul>
            <li><strong>Best Layer:</strong> Layer {results_df.loc[best_idx, 'layer']} ({best_metric.upper()}: {best_value:.3f})</li>
            <li><strong>Best Hook:</strong> {results_df.loc[best_idx, 'hook']}</li>
            <li><strong>Best Token Scheme:</strong> {results_df.loc[best_idx, 'scheme']}</li>
            <li><strong>Best Classifier:</strong> {results_df.loc[best_idx, 'classifier']}</li>
"""


    if 'custom_final_score' in results_df.columns:
        custom_best_idx = results_df['custom_final_score'].idxmax()
        html_content += f"""
            <li><strong>Custom Score Best:</strong> <span class="custom-score">Layer {results_df.loc[custom_best_idx, 'layer']}, Hook: {results_df.loc[custom_best_idx, 'hook']}, Scheme: {results_df.loc[custom_best_idx, 'scheme']} ({results_df.loc[custom_best_idx, 'custom_final_score']:.3f})</span></li>
"""

    html_content += f"""
        </ul>

        <div class="stats-box">
            <p><em>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            <p><em>Analysis includes {len(results_df)} experiments across {results_df['layer'].nunique()} layers, {len(results_df['hook'].unique())} hook types, and {len(results_df['scheme'].unique())} token schemes.</em></p>
"""


    html_content += f"""
        </div>
    </body>
    </html>
    """

    with open(os.path.join(output_dir, 'comprehensive_report.html'), 'w') as f:
        f.write(html_content)

    logger.info(f"Comprehensive HTML report saved to: {os.path.join(output_dir, 'comprehensive_report.html')}")

# ================================================================
# 7. Two-Level Parallel Processing Functions
# ================================================================

# ================================================================
# 6.5. ROC Curve Data Saving Functions
# ================================================================

# ROC curve data saving function removed

# ================================================================
# 6.6. Classifier Saving Functions - MINIMAL INFERENCE SUPPORT
# ================================================================
# 
# FIXED: ProcessPoolExecutor issue where trained_classifiers_storage was empty
# SOLUTION: Pass trained models through return values instead of global storage
# - process_single_activation_group() returns best_classifier_data
# - Main loop tracks best_overall_model across all groups  
# - save_best_classifier() accepts trained_model_data directly
# ================================================================

def save_best_classifier(results_df, output_dir, trained_model_data=None, selection_metric=None):
    """
    Saves the best performing classifier and associated metadata.
    MODIFIED: Now accepts a 'selection_metric' to determine the best model.
    """
    logger.info("Finding and saving the best overall classifier...")
    logger.info("=" * 60)

    # Initialize variables
    selection_value = 0.0

    if results_df.empty:
        logger.error("No results to analyze. Cannot save the best classifier.")
        return None

    if trained_model_data is None or 'model' not in trained_model_data:
        logger.info("No trained model object provided. Will reconstruct from best result in data.")
        # Find the best result from the results DataFrame using custom scoring
        if results_df.empty:
            logger.error("No results available to determine best classifier.")
            return None

        # --- MODIFIED: Use the provided selection_metric ---
        if selection_metric and selection_metric in results_df.columns:
            best_idx = results_df[selection_metric].idxmax()
            selection_value = results_df.loc[best_idx, selection_metric]
            logger.info(f"Selecting best model using '{selection_metric}' (value: {selection_value:.4f})")
        # Fallback to original logic if no metric is provided
        elif 'custom_final_score' in results_df.columns:
            best_idx = results_df['custom_final_score'].idxmax()
            selection_metric = 'custom_final_score'
            selection_value = results_df.loc[best_idx, 'custom_final_score']
            logger.info(f"Selecting best model using custom scoring (value: {selection_value:.4f})")
        else:
            best_idx = results_df['auroc'].idxmax()
            selection_metric = 'auroc'
            selection_value = results_df.loc[best_idx, 'auroc']
            logger.info(f"Selecting best model using AUROC (value: {selection_value:.4f}) - custom score not available")

        best_result_row = results_df.loc[best_idx]
        
        # Create a trained_model_data-like structure from the best result
        trained_model_data = {
            'layer': best_result_row['layer'],
            'hook': best_result_row['hook'], 
            'scheme': best_result_row['scheme'],
            'classifier': best_result_row['classifier'],
            'imbalance_strategy': best_result_row.get('imbalance_strategy', 'none'),
            'auroc': best_result_row['auroc'],
            'chunk_id': best_result_row.get('chunk_id', 'unknown'),
            # Note: 'model' key is intentionally missing - will be reconstructed below
        }
        logger.info(f"Best configuration from results: {trained_model_data.get('classifier', trained_model_data.get('classifier_name', 'Unknown'))} with AUROC {trained_model_data['auroc']:.4f}")
        logger.info(f"Layer {trained_model_data['layer']}, Hook {trained_model_data['hook']}, Scheme {trained_model_data['scheme']}")

    best_result = trained_model_data

    logger.info(f"Best model identified: {best_result.get('classifier', best_result.get('classifier_name', 'Unknown'))} from Layer {best_result['layer']}, Hook {best_result['hook']}")
    logger.info(f"AUROC: {best_result['auroc']:.4f}")
    logger.info(f"Chunk ID: {best_result.get('chunk_id', 'N/A')}")
    logger.info(f"NOTE: Model selection was based on held-out test set performance.")
    logger.info(f"For production use, consider nested cross-validation for unbiased selection.")

    # --- DATA LEAKAGE FIX: Re-train the best model on the full dataset for its group ---
    logger.info("Re-training the best model on the full dataset for its activation group...")

    # Step 1: Load activation data for the best (layer, hook, scheme) combination from ALL chunks
    try:
        experiment_path = os.path.join(ACTIVATIONS_BASE_DIR, EXPERIMENT_ID)
        logger.info(f"Loading activation data for best combination from all available chunks...")

        # Get all chunk directories
        chunk_directories = discover_experiment_chunks(experiment_path)
        if not chunk_directories:
            raise FileNotFoundError(f"No chunk directories found in {experiment_path}")
        
        chunk_directory_paths = list(chunk_directories.values())

        # Efficiently get all H5 files from all chunks
        _, all_h5_files = generate_work_units(chunk_directory_paths)

        # Define the single work unit to load
        work_unit = (best_result['layer'], best_result['hook'], best_result['scheme'])
        logger.info(f"Performing targeted load for best work unit: {work_unit}")
        
        # Use the targeted loader to get data from all H5 files at once
        combined_group = load_single_activation_group(work_unit, all_h5_files)
        
        if not combined_group or 'activations' not in combined_group or len(combined_group['activations']) == 0:
            raise ValueError(f"Could not find activation data for {work_unit} in any chunk")

        # Efficiently load all results data once for alignment
        logger.info("Loading all results data for alignment...")
        combined_results_df, _ = load_multi_chunk_results_data(chunk_directory_paths, {})
        if combined_results_df.empty:
            raise ValueError("Failed to load any results data for alignment.")

        # Align features and labels
        logger.info("Aligning features and labels from combined dataset...")
        X, y = align_features_labels(combined_group, combined_results_df)

        if X is None or len(X) == 0:
            raise ValueError("Could not align features and labels from combined data")

        logger.info(f"Successfully loaded and aligned {len(X)} samples for the target work unit.")

    except Exception as e:
        logger.error(f"Failed to load data for final model training: {e}")
        return None
    
    # Step 2: Create a pipeline to prevent data leakage
    logger.info("Creating training pipeline to prevent data leakage...")
    imbalance_strategy = best_result['imbalance_strategy']

    # Calculate scale_pos_weight for the full dataset if using XGBoost
    scale_pos_weight = None
    if best_result.get('classifier', best_result.get('classifier_name', '')).startswith('XGBoost'):
        scale_pos_weight = calculate_scale_pos_weight(y)
        logger.info(f"FINAL MODEL: Calculated scale_pos_weight = {scale_pos_weight:.2f} from full dataset")

    # Use a fresh instance of the classifier with the same hyperparameters
    # The 'model' in best_result is already trained on a fold, so we get a new one.
    # For final training on full dataset, disable early stopping since we don't have a validation set
    classifiers_map = get_classifiers(scale_pos_weight=scale_pos_weight, gpu_id=0)
    final_classifier = classifiers_map[best_result.get('classifier', best_result.get('classifier_name', 'XGBoost_GPU'))]

    # For final model training on full dataset, disable early stopping
    if hasattr(final_classifier, 'set_params'):
        final_classifier.set_params(early_stopping_rounds=None)

    steps = []
    # Always scale the data
    steps.append(('scaler', StandardScaler()))
    

    # Add sampler to pipeline if strategy is not 'none'
    if imbalance_strategy == 'smote':
        sampler = SMOTE(random_state=GLOBAL_RANDOM_SEED)
        steps.append(('sampler', sampler))
        logger.info("Pipeline using: StandardScaler -> SMOTE -> Classifier")
    elif imbalance_strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=GLOBAL_RANDOM_SEED)
        steps.append(('sampler', sampler))
        logger.info("Pipeline using: StandardScaler -> RandomUnderSampler -> Classifier")
    else:
        logger.info("Pipeline using: StandardScaler -> Classifier")

    steps.append(('classifier', final_classifier))
    
    # Use imblearn's pipeline to handle samplers correctly
    pipeline = ImbPipeline(steps)
    
    # Step 3: Train the pipeline on the full, original data for that group
    logger.info("Fitting final pipeline on the full group dataset...")
    # Compute per-process linalg threads
    total_cores = psutil.cpu_count(logical=True)
    n_workers = 1  # Assuming single process for saving
    linalg_threads = max(1, min(LINALG_THREADS, total_cores))

    logger.info(f"LINALG threads for final model: {linalg_threads}")

    # Wrap pipeline fit with threadpool_limits
    with threadpool_limits(limits=linalg_threads, user_api='blas'):
        pipeline.fit(X, y)
    logger.info("Training complete.")
    
    # Extract the fitted components for saving
    fitted_classifier = pipeline.named_steps['classifier']
    fitted_scaler = pipeline.named_steps['scaler']

    # Final package for saving, maintaining the original format
    classifier_package = {
        'model': fitted_classifier,
        'scaler': fitted_scaler,
        'metadata': {
            'run_id': RUN_ID,
            'layer': best_result['layer'],
            'hook': best_result['hook'],
            'scheme': best_result['scheme'],
            'classifier': best_result.get('classifier', best_result.get('classifier_name', 'Unknown')),
            'classifier_name': best_result.get('classifier', best_result.get('classifier_name', 'Unknown')),  # ADDED for inference compatibility
            'imbalance_strategy': imbalance_strategy,
            'n_features': X.shape[1],
            'optimal_threshold': best_result_row.get('optimal_threshold', 0.5),
            'custom_score_at_optimal': best_result_row.get('custom_score_at_optimal', 0.0),
            'selection_method': 'custom_only_finalists_exact_cutpoints',
            'proxy_custom_score': best_result_row.get('proxy_custom_score', None),
            'is_finalist': best_result_row.get('is_finalist', True),
            'performance': {
                'auroc': best_result_row.get('auroc', 0.0),
                'custom_final_score': best_result_row.get('custom_final_score', 0.0),
                'accuracy': best_result_row.get('accuracy', 0.0),
                'n_train': X.shape[0]
            },
            'selection_info': {
                'selection_metric': selection_metric,
                'selection_value': selection_value,
                'selection_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save the final, properly trained model package
    classifier_filename = f"best_classifier_{RUN_ID}.pkl"
    classifier_path = os.path.join(output_dir, classifier_filename)
    
    try:
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Best classifier saved to: {classifier_path}")
        logger.info(f"Model: {best_result.get('classifier', best_result.get('classifier_name', 'Unknown'))}")
        logger.info(f"Selection Metric: {selection_metric}")
        logger.info(f"Selection Value: {selection_value:.4f}")
        if 'custom_final_score' in best_result:
            logger.info(f"Custom Score: {best_result.get('custom_final_score', 0.0):.4f}")
        logger.info(f"AUROC: {best_result.get('auroc', 0.0):.4f}")
        
        logger.info(f"Expected input: {X.shape[1]} features")
        logger.info(f"Ready for inference!")

        return classifier_path

    except Exception as e:
        logger.error(f"Failed to save classifier: {e}")
        return None

# 7. Main Processing Loop
    # ================================================================
def process_all_chunks_parallel(chunk_directories, results_df, classifiers, compute_resources, run_id):
    """
    Orchestrates the parallel processing of all work units using the
    Work-Queue pattern. This is the main driver for the analysis.
    """
    # 1. Setup checkpointer and generate the full work queue
    # Use CLASSIFIER_DIR (passed via global) which is inside the experiment directory
    checkpointer = RobustCheckpointer(CLASSIFIER_DIR)
    all_work_units, all_h5_files = generate_work_units(chunk_directories)

    # 2. Filter out already completed work
    completed_units = checkpointer.load_completed_work_units()
    remaining_work = [
        unit for unit in all_work_units 
        if checkpointer._work_unit_to_str(unit) not in completed_units
    ]

    logger.info(f"Total work units discovered: {len(all_work_units)}")
    logger.info(f"Work units already completed: {len(completed_units)}")
    logger.info(f"Work units remaining to process: {len(remaining_work)}")

    if not remaining_work:
        logger.info("All work has been completed in a previous run.")
        return

    logger.info(f"Will process {len(remaining_work)} work units in parallel batches")

    # 3. Prepare arguments for the parallel workers
    args_list = [(unit, all_h5_files, results_df, classifiers, compute_resources) for unit in remaining_work]
    n_workers = compute_resources.get('group_jobs', os.cpu_count() // 2)

    # 4. Execute work in parallel and save results as they complete
    logger.info(f"Starting parallel processing of {len(remaining_work)} work units...")
    logger.info(f"Using ProcessPoolExecutor with max_workers={n_workers}")

    # Track global best classifier across all work units using custom scoring
    global_best_classifier = None
    global_best_selector_score = 0.0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_unit = {executor.submit(process_single_work_unit, args): args[0] for args in args_list}
        logger.info(f"Submitted {len(future_to_unit)} work units to {n_workers} worker processes")
        
        progress = tqdm(
            as_completed(future_to_unit),
            total=len(remaining_work),
            desc="Processing work units",
            unit="unit",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        completed_work_units = 0
        failed_work_units = 0
        for future in progress:
            try:
                work_unit, result_data = future.result()
                completed_work_units += 1

                # Calculate and display remaining work units
                remaining_count = len(remaining_work) - progress.n - 1
                layer_num, hook_name, scheme = work_unit

                work_unit_progress = f"[{completed_work_units}/{len(remaining_work)}]"
                progress.set_postfix(
                    layer=f"L{layer_num:02d}",
                    hook=hook_name[:15],
                    remaining=f"{remaining_count} left",
                    progress=work_unit_progress
                )
            except Exception as e:
                failed_work_units += 1
                logger.error(f"Worker process failed: {e}")
                # Continue processing other work units
                continue

            # Atomically save the result and update the checkpoint file immediately
            if result_data.get('status') == 'success':
                checkpointer.save_work_unit_result(work_unit, result_data)
                logger.info(f"{work_unit_progress} CHECKPOINT: Saved work unit L{layer_num:02d}_{hook_name}_{scheme}")
                
            else:
                logger.warning(f"{work_unit_progress} WARNING: Work unit L{layer_num:02d}_{hook_name}_{scheme} failed: {result_data.get('reason', 'Unknown error')}")

    # --- REMOVED: Immediate saving is no longer done here. The final model is selected and saved after aggregation. ---

    logger.info(f"Processing complete. Summary:")
    logger.info(f"Completed successfully: {completed_work_units}")
    logger.info(f"Failed: {failed_work_units}")
    logger.info(f"Total processed: {completed_work_units + failed_work_units}")

    # Production mode completed
    logger.info("=" * 60)
    logger.info("GPU-OPTIMIZED EXECUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successfully processed {completed_work_units} work units across 4 GPUs.")
    if global_best_classifier:
        logger.info(f"Best model: {global_best_classifier.get('classifier_name', 'Unknown')}")
        logger.info(f"Custom Score: {global_best_selector_score:.4f}")
        logger.info(f"AUROC: {global_best_classifier.get('auroc', 0):.4f}")
    if failed_work_units > 0:
        logger.warning(f"Note: {failed_work_units} work units failed and can be retried.")
    logger.info("=" * 60)

def main():
    """
    Main entry point for the classifier analysis script.
    Orchestrates the new Work-Queue parallel processing architecture.
    
    NEW DIRECTORY STRUCTURE:
    - Classifier run directory is created INSIDE the experiment directory:
      ACTIVATIONS_BASE_DIR/EXPERIMENT_ID/classifier_run_ID/
    - This places classifier results alongside chunk folders (chunk_0, chunk_1, etc.)
    """
    global RUN_ID, CLASSIFIER_DIR, CLASSIFIER_EXPERIMENT_DIR, run_metadata, CONTINUE_FROM_RUN, logger

    # --- 1. SETUP & CONFIGURATION ---
    # Dynamic run ID and directory setup for resumability
    # NEW: Create classifier directory INSIDE the experiment directory, not in CLASSIFIER_BASE_DIR
    experiment_base_dir = os.path.join(ACTIVATIONS_BASE_DIR, EXPERIMENT_ID)
    
    # Determine RUN_ID and setup directory
    if CONTINUE_FROM_RUN:
        # Look for the run inside the experiment directory
        run_dir = os.path.join(experiment_base_dir, CONTINUE_FROM_RUN)
        if os.path.isdir(run_dir):
            RUN_ID = CONTINUE_FROM_RUN
            CLASSIFIER_DIR = run_dir
            CLASSIFIER_EXPERIMENT_DIR = run_dir  # NEW: Store for evaluate.py
            logger.info(f"RESUMING: Continuing from specified run: {RUN_ID}")
            logger.info(f"Classifier directory: {CLASSIFIER_DIR}")
        else:
            logger.warning(f"Specified run '{CONTINUE_FROM_RUN}' not found in {experiment_base_dir}. Starting a new run.")
            CONTINUE_FROM_RUN = ""

    if not CONTINUE_FROM_RUN:
        RUN_ID = f"classifier_run_{int(time.time())}"
        logger.info(f"STARTING: Creating new run: {RUN_ID}")

        # NEW: Create inside experiment directory, not CLASSIFIER_BASE_DIR
        CLASSIFIER_DIR = os.path.join(experiment_base_dir, RUN_ID)
        CLASSIFIER_EXPERIMENT_DIR = CLASSIFIER_DIR  # NEW: Store for evaluate.py
        os.makedirs(CLASSIFIER_DIR, exist_ok=True)
        logger.info(f"Classifier directory created: {CLASSIFIER_DIR}")

    # Redirect logger to the correct run-specific directory
    logger.set_output_directory(CLASSIFIER_DIR)
    logger.info(f"Logger setup complete. All logging now directed to: {CLASSIFIER_DIR}")
    logger.info(f"XGBoost version being used: {xgboost.__version__}")

    # Standard setup calls from the original main function
    compute_resources = detect_compute_resources()
    
    # MULTI-GPU OPTIMIZED: Set environment variables for multi-GPU training
    T = compute_resources.get('ml_algorithm_jobs', 2)
    logger.info("=" * 80)
    logger.info("MULTI-GPU CONFIGURATION (from config.py)")
    logger.info("=" * 80)
    logger.info(f"NUM_GPUS (configured): {NUM_GPUS}")
    logger.info(f"GPU_IDS (to use): {GPU_IDS}")
    logger.info(f"Actual GPUs available: {len(GPU_IDS)}")
    logger.info("Setting environment variables for multi-GPU training")
    logger.info(f"OMP_NUM_THREADS={T} (minimal for GPU workloads)")
    logger.info(f"CUDA_VISIBLE_DEVICES will be set per worker for dynamic GPU assignment")
    logger.info("=" * 80)

    os.environ['OMP_NUM_THREADS'] = str(T)
    os.environ['OPENBLAS_NUM_THREADS'] = str(T)
    os.environ['MKL_NUM_THREADS'] = str(T)
    os.environ['NUMEXPR_NUM_THREADS'] = str(T)
    # NOTE: Do NOT set CUDA_VISIBLE_DEVICES globally - let each worker set it independently
    
    logger.info("=" * 60)
    logger.info(f"Starting Classifier Analysis Run: {RUN_ID}")
    logger.info(f"Experiment Directory: {experiment_base_dir}")
    logger.info(f"Classifier Directory: {CLASSIFIER_DIR}")
    logger.info("=" * 60)

    # --- 2. LOAD SHARED DATA ---
    # This data is loaded once and shared read-only with all worker processes.
    logger.info("Loading shared results data (labels) for all workers...")
    CHUNK_DIRECTORIES = discover_experiment_chunks(experiment_base_dir)
    if not CHUNK_DIRECTORIES: return

    # --- 2.5. LOAD GENERATOR CONFIG ---
    logger.info("Loading configuration audit trail...")
    generator_config = load_generator_metadata(CHUNK_DIRECTORIES)

    results_df, class_distribution = load_multi_chunk_results_data(list(CHUNK_DIRECTORIES.values()), {})
    if results_df.empty:
        logger.error("No results data (labels) found. Cannot proceed.")
        return

    # --- 2.5a. AUTO-DETECT IMBALANCE HANDLING (BEFORE creating classifier metadata) ---
    logger.info("Auto-detecting class imbalance handling based on data distribution...")
    auto_detect_imbalance_handling(class_distribution, threshold=IMBALANCE_THRESHOLD)
    logger.info(f"Imbalance handling decision: HANDLE_IMBALANCE = {HANDLE_IMBALANCE}")
    
    # --- 2.5b. CREATE CLASSIFIER METADATA (AFTER auto-detecting imbalance) ---
    logger.info("Creating classifier metadata with auto-detected imbalance decision...")
    classifier_metadata = create_classifier_metadata(RUN_ID, generator_config)
    save_classifier_metadata(CLASSIFIER_DIR, classifier_metadata)
    logger.info("Classifier metadata created and saved with imbalance decision.")

    # Note: Classifiers are now created per work unit with GPU-specific configuration
    # This allows dynamic GPU assignment and avoids GPU conflicts
    
    # --- 3. ORCHESTRATE PARALLEL PROCESSING ---
    # This is the core call to our new, robust, memory-efficient system.
    process_all_chunks_parallel(
        chunk_directories=list(CHUNK_DIRECTORIES.values()),
        results_df=results_df,
        classifiers=None,  # Created per work unit now
        compute_resources=compute_resources,
        run_id=RUN_ID
    )

    # --- 4. AGGREGATE & VISUALIZE (Phase 4) ---
    # This final step runs after all parallel workers are done.
    logger.info("All parallel processing complete. Starting final result aggregation...")
    aggregate_and_visualize_results(CLASSIFIER_DIR)

    logger.info("End-to-end analysis run has completed.")
    logger.info("All remaining work units have been processed.")

def aggregate_and_visualize_results(classifier_run_dir):
    """
    Phase 4: Aggregates all individual work unit results and runs the
    final analysis and visualization suite.
    """
    logger.info("Aggregating results from all completed work units...")
    results_dir = os.path.join(classifier_run_dir, "work_unit_results")
    if not os.path.isdir(results_dir):
        logger.error("No results directory found. Cannot generate final report.")
        return

    all_results = []
    result_files = glob.glob(os.path.join(results_dir, "*.json"))

    for file_path in tqdm(result_files, desc="Aggregating results"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if data.get('status') == 'success' and 'group_results' in data:
                # Extract work unit metadata from filename
                filename = os.path.basename(file_path)
                # Expected format: L{layer:02d}_{hook}_{scheme}.json
                if filename.startswith('L') and filename.endswith('.json'):
                    # Remove 'L' prefix and '.json' suffix
                    content = filename[1:-5]  # Remove L and .json

                    # Define the 4 expected token schemes
                    known_schemes = ['first_generated', 'last_generated', 'bos_token', 'last_prompt_token']

                    # Find which scheme is at the end of the filename
                    scheme = None
                    for s in known_schemes:
                        if content.endswith('_' + s):
                            scheme = s
                            # Remove the scheme and underscore from the end
                            content = content[:-len('_' + s)]
                            break

                    if scheme is not None:
                        # Extract layer number from the beginning
                        layer_str = 'L' + content.split('_')[0]

                        # Everything after the layer number is the hook
                        hook_part = content[len(layer_str[1:]):]  # Remove layer number
                        if hook_part.startswith('_'):
                            hook_part = hook_part[1:]  # Remove leading underscore
                        hook = hook_part

                        # Extract layer number
                        try:
                            layer = int(layer_str[1:])  # Remove 'L' prefix
                        except ValueError:
                            layer = layer_str  # Keep as string if not numeric
                    else:
                        logger.warning(f"Could not identify scheme in filename: {filename}")
                        # Fallback: try original parsing
                        parts = filename[:-5].split('_', 2)
                        if len(parts) >= 3 and parts[0].startswith('L'):
                            layer_str = parts[0]
                            hook = parts[1]
                            scheme = '_'.join(parts[2:])
                            try:
                                layer = int(layer_str[1:])
                            except ValueError:
                                layer = layer_str
                        else:
                            continue

                    # Add metadata to each classifier result
                    for classifier_name, metrics in data['group_results'].items():
                        # Create a copy of metrics with added metadata
                        enriched_metrics = metrics.copy()
                        enriched_metrics.update({
                            'layer': layer,
                            'hook': hook,
                            'scheme': scheme,
                            'classifier': classifier_name
                        })
                        all_results.append(enriched_metrics)
                else:
                    logger.warning(f"Unexpected filename format: {filename}")
                    # Fallback: just extend with values without metadata
                    all_results.extend(data['group_results'].values())
        except Exception as e:
            logger.warning(f"Could not read or parse result file {os.path.basename(file_path)}: {e}")

    if not all_results:
        logger.error("No valid results found to aggregate. Aborting final analysis.")
        return

    results_df = pd.DataFrame(all_results)
    logger.info(f"Aggregated {len(results_df)} total experiment results from {len(result_files)} work units.")

    # Debug: Check available custom score columns
    custom_columns = [col for col in results_df.columns if 'custom' in col.lower() or col in ['mcc', 'f0_beta', 'f1_1', 'gate_mcc_gamma']]
    logger.info(f"Available custom score columns: {custom_columns}")

    if 'custom_final_score' in results_df.columns:
        logger.info(f"Custom score range: {results_df['custom_final_score'].min():.3f} - {results_df['custom_final_score'].max():.3f}")
        logger.info(f"Custom score mean: {results_df['custom_final_score'].mean():.3f} ± {results_df['custom_final_score'].std():.3f}")

    # --- Run the original, existing analysis and visualization suite ---
    logger.info("Running final analysis and generating reports...")
    try:
        # Save comprehensive CSV
        final_csv_path = os.path.join(classifier_run_dir, "comprehensive_metrics.csv")
        atomic_save_csv(results_df, final_csv_path)
        logger.info(f"Comprehensive metrics saved to: {os.path.basename(final_csv_path)}")

        # === PHASE 4A: Create visualizations for ALL work units ===
        logger.info("Creating comprehensive visualizations for ALL work units...")
        create_comprehensive_visualizations(results_df, classifier_run_dir)
        logger.info("Comprehensive visualizations (all work units) complete.")

        # Create statistical analysis for all work units
        create_statistical_analysis(results_df, classifier_run_dir)
        logger.info("Statistical analysis (all work units) complete.")

        # Load label distribution data from chunks
        chunk_dirs = discover_experiment_chunks(os.path.dirname(classifier_run_dir))
        labels_df = None
        class_dist = None
        if chunk_dirs:
            try:
                labels_df, class_dist = load_multi_chunk_results_data(list(chunk_dirs.values()), {})
                logger.info(f"Loaded label distribution data: {len(labels_df)} samples")
            except Exception as e:
                logger.warning(f"Could not load label distribution data: {e}")

        # Create HTML report for all work units
        create_comprehensive_analysis_report(results_df, classifier_run_dir, class_dist)
        logger.info("Comprehensive HTML report (all work units) generated.")

        # === PHASE 4B: Finalist optimization and finalist-specific analysis ===
        logger.info("Starting finalist optimization and finalist-specific analysis...")
        finalist_output_dir = os.path.join(classifier_run_dir, "finalist_analysis")
        os.makedirs(finalist_output_dir, exist_ok=True)
        updated_results_df = optimize_finalists_and_save(results_df, classifier_run_dir, finalist_output_dir)

        if updated_results_df is not None:

            logger.info("Creating finalist-specific visualizations...")
            create_finalist_visualizations(updated_results_df, finalist_output_dir)
            logger.info("Finalist visualizations generated.")

            # Create statistical analysis for finalists
            create_statistical_analysis(updated_results_df, finalist_output_dir)
            logger.info("Statistical analysis (finalists) complete.")

            # Create HTML report for finalists
            create_finalist_analysis_report(updated_results_df, finalist_output_dir, class_dist)
            logger.info("Finalist HTML report generated.")

        logger.info("Best classifier model saved.")

    except Exception as e:
        import traceback
        logger.error(f"An error occurred during the final analysis phase: {e}")
        logger.error(traceback.format_exc())

def optimize_finalists_and_save(results_df, classifier_run_dir, finalist_output_dir):
    """
    NEW: Performs the expensive, exact threshold optimization ONLY for the top F finalists.
    This implements the global optimization strategy.
    """
    logger.info("=" * 60)
    logger.info("STARTING PHASE 3: FINALIST OPTIMIZATION")
    logger.info("=" * 60)

    if 'proxy_custom_score' not in results_df.columns:
        logger.error("Missing 'proxy_custom_score' in results. Cannot select finalists. Falling back to old save method.")
        save_best_classifier(results_df, classifier_run_dir)
        return

    # 1. Select Top F Finalists based on their proxy scores
    finalists_df = results_df.nlargest(FINALIST_COUNT, 'proxy_custom_score').copy()
    logger.info(f"Selected {len(finalists_df)} finalists for exact threshold optimization based on proxy scores.")

    finalist_results = []
    
    # Lazily load shared data only if there are finalists
    if not finalists_df.empty:
        logger.info("Loading shared data needed for finalist optimization...")
        experiment_path = os.path.join(ACTIVATIONS_BASE_DIR, EXPERIMENT_ID)
        chunk_directories = list(discover_experiment_chunks(experiment_path).values())
        all_h5_files = generate_work_units(chunk_directories)[1]
        results_data_df, _ = load_multi_chunk_results_data(chunk_directories, {})
        logger.info("Shared data loaded.")

    # 2. Loop through finalists, re-train, and perform exact optimization
    for idx, finalist_row in tqdm(finalists_df.iterrows(), total=len(finalists_df), desc="Optimizing Finalists"):
        work_unit = (finalist_row['layer'], finalist_row['hook'], finalist_row['scheme'])
        logger.info(f"Optimizing Finalist {idx+1}/{len(finalists_df)}: L{work_unit[0]}_{work_unit[1]}_{work_unit[2]}")
        
        try:
            # a. Load data for this specific work unit
            activation_group = load_single_activation_group(work_unit, all_h5_files)
            X, y = align_features_labels(activation_group, results_data_df)
            
            if X is None or len(X) == 0:
                logger.warning(f"Skipping finalist {work_unit} due to no data after alignment.")
                continue

            # b. Re-split the data IDENTICALLY to how it was done in the worker
            X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_RANDOM_SEED, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=GLOBAL_RANDOM_SEED, stratify=y_temp)

            # c. Re-train the model to get the identical trained object
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            scale_pos_weight = calculate_scale_pos_weight(y_train)
            clf = create_imbalanced_xgboost_from_params(scale_pos_weight=scale_pos_weight, gpu_id=0) # CUDA_VISIBLE_DEVICES remaps GPU to 0
            clf.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])

            # d. Generate validation probabilities
            proba_val = clf.predict_proba(X_val_scaled)[:, 1]

            # e. Perform the expensive, exact threshold sweep on validation probabilities
            cuts = np.r_[0.0, np.sort(np.unique(proba_val)), 1.0]
            best_s = -1e18
            best_t = 0.5

            # Collect threshold-performance data for visualization
            threshold_sweep_data = []

            for t in cuts:
                y_pred = (proba_val >= t).astype(int)
                cm = confusion_matrix(y_val, y_pred, labels=[0,1])
                tn, fp, fn, tp = (0,0,0,0) if cm.shape != (2,2) else cm.ravel()

                met = {'true_positives': int(tp), 'false_positives': int(fp), 'true_negatives': int(tn), 'false_negatives': int(fn)}
                prec = precision_score(y_val, y_pred, average=None, zero_division=0)
                rec  = recall_score(y_val, y_pred, average=None, zero_division=0)
                if len(prec)==2:
                    met.update({'precision_not_hallucinated': float(prec[0]), 'precision_hallucinated': float(prec[1])})
                if len(rec)==2:
                    met.update({'recall_not_hallucinated': float(rec[0]), 'recall_hallucinated': float(rec[1])})

                attach_custom_score(met)
                s = met.get('custom_final_score', -1e18)

                # Store all metrics for this threshold
                threshold_sweep_data.append({
                    'threshold': float(t),
                    'custom_score': float(s),
                    'precision_halluc': float(met.get('precision_hallucinated', 0)),
                    'recall_halluc': float(met.get('recall_hallucinated', 0)),
                    'precision_no_halluc': float(met.get('precision_not_hallucinated', 0)),
                    'recall_no_halluc': float(met.get('recall_not_hallucinated', 0)),
                    'f1_halluc': 2 * met.get('precision_hallucinated', 0) * met.get('recall_hallucinated', 0) /
                                (met.get('precision_hallucinated', 0) + met.get('recall_hallucinated', 0)) if
                                (met.get('precision_hallucinated', 0) + met.get('recall_hallucinated', 0)) > 0 else 0,
                    'f1_no_halluc': 2 * met.get('precision_not_hallucinated', 0) * met.get('recall_not_hallucinated', 0) /
                                   (met.get('precision_not_hallucinated', 0) + met.get('recall_not_hallucinated', 0)) if
                                   (met.get('precision_not_hallucinated', 0) + met.get('recall_not_hallucinated', 0)) > 0 else 0,
                    'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                    'mcc': met.get('mcc', 0)
                })

                if s > best_s:
                    best_s, best_t = s, float(t)

            # Create threshold optimization visualization
            create_threshold_optimization_plot(threshold_sweep_data, work_unit, finalist_output_dir)
            
            # f. Store the true optimal threshold and score
            finalist_row['optimal_threshold'] = best_t
            finalist_row['custom_score_at_optimal'] = best_s
            finalist_row['is_finalist'] = True
            finalist_row['threshold_analysis_performed'] = True
            finalist_results.append(finalist_row)
            logger.info(f"Finalist {work_unit} | True Optimal Score: {best_s:.4f} @ Threshold: {best_t:.4f}")

        except Exception as e:
            logger.error(f"Failed to optimize finalist {work_unit}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if not finalist_results:
        logger.error("No finalists were successfully optimized. Cannot save best model.")
        return None

    # 3. Create a new DataFrame with the updated, optimized finalist results
    optimized_finalists_df = pd.DataFrame(finalist_results)

    # 4. Find the final winner and save it
    logger.info("Finding winner from optimized finalists and saving the final model...")
    save_best_classifier(optimized_finalists_df, classifier_run_dir, selection_metric='custom_score_at_optimal')

    # Return the optimized finalists DataFrame so visualizations can use the updated data
    return optimized_finalists_df


def create_finalist_visualizations(results_df, output_dir):
    """Create visualizations specifically for the top 10 finalists with optimized thresholds"""
    # Set style for better plots
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")

    # Only create the most relevant visualizations for finalists
    # 1. Best Classifier Multi-Metric Analysis (most important for finalists)
    if 'custom_final_score' in results_df.columns:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Find best classifier metrics
        best_idx = results_df['custom_final_score'].idxmax()
        best_metrics = results_df.loc[best_idx]

        # Metrics to display
        metric_names = ['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'precision_hallucinated',
                       'recall_hallucinated', 'precision_not_hallucinated', 'recall_not_hallucinated',
                       'custom_final_score']
        metric_labels = ['AUROC', 'Accuracy', 'F1 (Halluc)', 'F1 (No Halluc)', 'Precision (Halluc)',
                        'Recall (Halluc)', 'Precision (No Halluc)', 'Recall (No Halluc)', 'Custom Score']

        # Multi-metric radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Normalize metrics to 0-1 scale for comparison
        metric_values = []
        for metric in metric_names:
            if metric in best_metrics.index:
                value = best_metrics[metric]
                # Normalize based on typical ranges
                if metric == 'auroc':
                    normalized = value  # Already 0-1
                elif metric == 'accuracy':
                    normalized = value  # Already 0-1
                elif 'f1' in metric:
                    normalized = value  # Already 0-1
                elif 'precision' in metric or 'recall' in metric:
                    normalized = value  # Already 0-1
                elif metric == 'custom_final_score':
                    # Normalize custom score to 0-1 based on its range
                    custom_range = results_df['custom_final_score'].max() - results_df['custom_final_score'].min()
                    if custom_range > 0:
                        normalized = (value - results_df['custom_final_score'].min()) / custom_range
                    else:
                        normalized = 0.5
                else:
                    normalized = value
                metric_values.append(normalized)
            else:
                metric_values.append(0)

        metric_values += metric_values[:1]  # Close the circle

        # Plot radar chart
        ax1.plot(angles, metric_values, 'o-', linewidth=2, label='Best Classifier', color='purple')
        ax1.fill(angles, metric_values, alpha=0.25, color='purple')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metric_labels, fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.set_title('Best Finalist - Multi-Metric Performance Profile', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Custom Score Distribution Analysis
        custom_scores = results_df['custom_final_score'].dropna()

        # Histogram
        ax2.hist(custom_scores, bins=10, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(best_metrics['custom_final_score'], color='red', linestyle='--', linewidth=2,
                   label=f'Best: {best_metrics["custom_final_score"]:.3f}')
        ax2.set_xlabel('Custom Score (Finalists)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Finalist Custom Score Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Trade-off Analysis: Custom Score vs AUROC
        scatter = ax3.scatter(results_df['custom_final_score'], results_df['auroc'],
                             c=results_df['f1_macro'], cmap='viridis', alpha=0.6, s=100)

        # Highlight best classifier
        ax3.scatter([best_metrics['custom_final_score']], [best_metrics['auroc']],
                   color='red', s=200, marker='*', edgecolor='black', linewidth=2,
                   label='Best Finalist')

        # Add reference lines
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='AUROC = 0.5')
        ax3.axvline(x=results_df['custom_final_score'].median(), color='purple', linestyle='--', alpha=0.5,
                   label='Median Custom Score')

        ax3.set_xlabel('Custom Score (Hallucination Detection)')
        ax3.set_ylabel('AUROC (General Performance)')
        ax3.set_title('Finalist Trade-off Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('F1 Macro Score')

        # Performance Summary Table
        ax4.axis('off')
        summary_text = f"Best Finalist Summary:\n"
        summary_text += f"Layer: {best_metrics['layer']}\n"
        summary_text += f"Hook: {best_metrics['hook']}\n"
        summary_text += f"Scheme: {best_metrics['scheme']}\n"
        summary_text += f"AUROC: {best_metrics['auroc']:.3f}\n"
        summary_text += f"Custom Score: {best_metrics['custom_final_score']:.3f}\n"
        summary_text += f"F1 Macro: {best_metrics['f1_macro']:.3f}"
        summary_text += f"Optimal Threshold: {best_metrics.get('optimal_threshold', 'N/A')}"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_finalist_multi_metric_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 2. Custom Score vs Traditional Metrics Scatter Plot (finalists only)
    if 'custom_final_score' in results_df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Check if MCC gate values are available for sizing
        if 'gate_mcc_gamma' in results_df.columns:
            # Use MCC gate for point size (scale to reasonable sizes)
            sizes = results_df['gate_mcc_gamma'] * 100 + 20  # Scale MCC gate to pixel sizes
            size_label = 'MCC Gate Value'
        else:
            # Fallback to fixed size if MCC gate not available
            sizes = 50
            size_label = 'Fixed Size'

        # Scatter plot: Custom Score vs AUROC for finalists
        scatter = ax.scatter(results_df['custom_final_score'], results_df['auroc'],
                           c=results_df['f1_macro'], cmap='viridis', alpha=0.7, s=sizes)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Macro Score')

        # Add reference line (y=x)
        min_val = min(results_df['custom_final_score'].min(), results_df['auroc'].min())
        max_val = max(results_df['custom_final_score'].max(), results_df['auroc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal Performance')

        ax.set_xlabel('Custom Score (Finalists)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Finalist Custom Score vs AUROC\n(Color: F1 Macro, Size: {size_label})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add correlation coefficient
        corr = results_df['custom_final_score'].corr(results_df['auroc'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add custom score range info
        custom_range = results_df['custom_final_score'].max() - results_df['custom_final_score'].min()
        ax.text(0.05, 0.90, f'Custom Score Range: {custom_range:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'finalist_custom_vs_traditional_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 3. Custom Score Token Scheme Analysis (Finalists)
    if 'custom_final_score' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        custom_by_scheme = results_df.groupby('scheme')['custom_final_score'].mean().sort_values(ascending=False)

        bars = ax.bar(range(len(custom_by_scheme)), custom_by_scheme.values,
                     color='purple', alpha=0.8, width=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, custom_by_scheme.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Token Scheme', fontsize=12)
        ax.set_ylabel('Custom Score (Hallucination Detection)', fontsize=12)
        ax.set_title('Finalist Custom Score by Token Scheme\n(Hallucination Detection Performance)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(custom_by_scheme)))
        ax.set_xticklabels(custom_by_scheme.index, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add performance range annotation
        score_range = custom_by_scheme.max() - custom_by_scheme.min()
        ax.text(0.02, 0.98, f'Score Range: {score_range:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'finalist_token_scheme_analysis_custom_score.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_threshold_optimization_plot(threshold_data, work_unit, output_dir):
    """Create a comprehensive visualization of threshold optimization for a finalist model"""
    # Set style for better plots
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(threshold_data)

    # Find optimal threshold
    optimal_idx = df['custom_score'].idxmax()
    optimal_threshold = df.loc[optimal_idx, 'threshold']
    optimal_score = df.loc[optimal_idx, 'custom_score']

    # Create a 2x2 subplot for comprehensive threshold analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Threshold Optimization Analysis\n{work_unit[1]} - {work_unit[2]} (Layer {work_unit[0]})',
                 fontsize=16, fontweight='bold')

    # 1. Custom Score vs Threshold (main optimization target)
    ax1.plot(df['threshold'], df['custom_score'], 'b-', linewidth=2, alpha=0.8)
    ax1.scatter([optimal_threshold], [optimal_score], color='red', s=150, marker='*',
                edgecolor='black', linewidth=2, label=f'Optimal: {optimal_threshold:.4f}')
    ax1.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axhline(y=optimal_score, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.fill_between(df['threshold'], df['custom_score'], alpha=0.3, color='blue')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Custom Score', fontsize=12)
    ax1.set_title('Custom Score vs Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add optimal point annotation
    ax1.annotate(f'Optimal\n{optimal_threshold:.4f}\n{optimal_score:.3f}',
                xy=(optimal_threshold, optimal_score), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                fontsize=10, ha='left')

    # 2. Precision-Recall Trade-off for Hallucinations
    ax2.plot(df['threshold'], df['precision_halluc'], 'r-', linewidth=2, label='Precision (Halluc)', alpha=0.8)
    ax2.plot(df['threshold'], df['recall_halluc'], 'g-', linewidth=2, label='Recall (Halluc)', alpha=0.8)
    ax2.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.scatter([optimal_threshold], [df.loc[optimal_idx, 'precision_halluc']], color='red', s=50, marker='o')
    ax2.scatter([optimal_threshold], [df.loc[optimal_idx, 'recall_halluc']], color='red', s=50, marker='o')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Hallucination Detection: Precision vs Recall', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add F1 score for hallucinations
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['threshold'], df['f1_halluc'], 'purple', linewidth=1.5, linestyle='--', label='F1 (Halluc)', alpha=0.7)
    ax2_twin.set_ylabel('F1 Score (Halluc)', fontsize=10, color='purple')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    ax2_twin.legend(loc='upper right')

    # 3. Non-Hallucination Performance
    ax3.plot(df['threshold'], df['precision_no_halluc'], 'orange', linewidth=2, label='Precision (No Halluc)', alpha=0.8)
    ax3.plot(df['threshold'], df['recall_no_halluc'], 'cyan', linewidth=2, label='Recall (No Halluc)', alpha=0.8)
    ax3.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.scatter([optimal_threshold], [df.loc[optimal_idx, 'precision_no_halluc']], color='red', s=50, marker='o')
    ax3.scatter([optimal_threshold], [df.loc[optimal_idx, 'recall_no_halluc']], color='red', s=50, marker='o')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Non-Hallucination Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Add F1 score for non-hallucinations
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['threshold'], df['f1_no_halluc'], 'brown', linewidth=1.5, linestyle='--', label='F1 (No Halluc)', alpha=0.7)
    ax3_twin.set_ylabel('F1 Score (No Halluc)', fontsize=10, color='brown')
    ax3_twin.tick_params(axis='y', labelcolor='brown')
    ax3_twin.legend(loc='upper right')

    # 4. Overall Performance Metrics
    ax4.plot(df['threshold'], df['accuracy'], 'navy', linewidth=2, label='Accuracy', alpha=0.8)
    ax4.plot(df['threshold'], df['mcc'], 'darkgreen', linewidth=2, label='MCC', alpha=0.8)
    ax4.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.scatter([optimal_threshold], [df.loc[optimal_idx, 'accuracy']], color='red', s=50, marker='o')
    ax4.scatter([optimal_threshold], [df.loc[optimal_idx, 'mcc']], color='red', s=50, marker='o')
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add threshold distribution info
    ax4.text(0.02, 0.98, f'Optimal Threshold: {optimal_threshold:.4f}\nCustom Score: {optimal_score:.3f}\nThresholds Tested: {len(df)}',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
             verticalalignment='top')

    plt.tight_layout()
    filename = f"threshold_optimization_L{work_unit[0]:02d}_{work_unit[1]}_{work_unit[2]}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Threshold optimization plot saved: {filename}")

def create_finalist_analysis_report(results_df, output_dir, class_dist=None):
    """Create an HTML report specifically for the top 10 finalists"""
    # Determine best performer based on the most accurate score available
    if 'custom_score_at_optimal' in results_df.columns and not results_df['custom_score_at_optimal'].isnull().all():
        best_metric = 'custom_score_at_optimal'
        best_value = results_df[best_metric].max()
        best_idx = results_df[best_metric].idxmax()
    elif 'custom_final_score' in results_df.columns:
        best_metric = 'custom_final_score'
        best_value = results_df[best_metric].max()
        best_idx = results_df[best_metric].idxmax()
    else:
        best_metric = 'auroc'
        best_value = results_df[best_metric].max()
        best_idx = results_df[best_metric].idxmax()

    # Helper function to create embedded image tags
    def create_embedded_img_tag(image_filename, alt_text="", style=""):
        """Create an HTML img tag with embedded base64 image data"""
        image_path = os.path.join(output_dir, image_filename)
        embedded_src = embed_image_as_base64(image_path)
        style_attr = f' style="{style}"' if style else ""
        return f'<img src="{embedded_src}" alt="{alt_text}"{style_attr}>'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top 10 Finalists Analysis Report - {EXPERIMENT_ID}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
            h2 {{ border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #9b59b6; color: white; }}
            .metric {{ font-weight: bold; color: #9b59b6; }}
            .highlight {{ background-color: #fff3cd; }}
            .graph-section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .graph-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            .stats-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 6px; margin: 10px 0; }}
            .custom-score {{ color: #9b59b6; font-weight: bold; }}
            .best-performance {{ background-color: #e8d5f0; border: 1px solid #9b59b6; padding: 10px; border-radius: 4px; margin: 10px 0; }}
            .finalist-badge {{ background-color: #9b59b6; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; }}
            .class-dist-table {{ width: auto; max-width: 800px; margin: 20px auto; }}
            .class-dist-table td {{ text-align: right; padding: 10px 15px; }}
            .class-dist-table th {{ text-align: center; padding: 10px 15px; }}
            .class-dist-table td:first-child {{ text-align: left; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Top 10 Finalists Analysis Report - {EXPERIMENT_ID}</h1>
"""

    # Add class distribution table if class_dist is available (purple border for finalist theme)
    html_content += generate_class_distribution_html(class_dist, border_color="#9b59b6")
    
    # Always add Executive Summary
    html_content += f"""
        <div class="stats-box">
            <h2>Executive Summary</h2>
            <ul>
                <li><strong>Experiment ID:</strong> {EXPERIMENT_ID}</li>
                <li><strong>Total Finalists:</strong> {len(results_df)}</li>
                <li><strong>Best AUROC:</strong> {results_df['auroc'].max():.3f}</li>
                <li><strong>Average AUROC:</strong> {results_df['auroc'].mean():.3f} ± {results_df['auroc'].std():.3f}</li>
"""

    if 'custom_final_score' in results_df.columns:
        html_content += f"""
                <li><strong>Best Custom Score:</strong> <span class="custom-score">{results_df['custom_final_score'].max():.3f}</span></li>
                <li><strong>Average Custom Score:</strong> <span class="custom-score">{results_df['custom_final_score'].mean():.3f} ± {results_df['custom_final_score'].std():.3f}</span></li>
"""

    html_content += f"""
                <li><strong>Layers Analyzed:</strong> {results_df['layer'].nunique()}</li>
                <li><strong>Hook Types:</strong> {', '.join(results_df['hook'].unique())}</li>
                <li><strong>Token Schemes:</strong> {', '.join(results_df['scheme'].unique())}</li>
                <li><strong>Classifiers:</strong> {', '.join(results_df['classifier'].unique())}</li>
            </ul>
        </div>

        <div class="best-performance">
            <h2>Best Finalist Performance</h2>
            <p><strong>Best Configuration:</strong> Layer {results_df.loc[best_idx, 'layer']}, Hook: {results_df.loc[best_idx, 'hook']}, Scheme: {results_df.loc[best_idx, 'scheme']}</p>
            <p><strong>Best Score:</strong> {best_value:.3f} ({best_metric.upper()})</p>
            <p><strong>Optimal Threshold:</strong> {results_df.loc[best_idx, 'optimal_threshold']:.3f} (Custom Score Maximizing)</p>
            <p><strong>Selection Method:</strong> Top 10 finalists with exact threshold optimization</p>
        </div>

        <h2>All Finalists</h2>
        <div style="overflow-x: auto;">
            {results_df[['layer', 'hook', 'scheme', 'classifier', 'auroc', 'optimal_threshold', 'custom_score_at_optimal', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated']].to_html(classes='highlight', index=False)}
        </div>

        <h2>Finalist Performance by Category</h2>

        <h3>By Classifier</h3>
        {results_df.groupby('classifier')[['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'custom_final_score']].agg(['mean', 'std']).round(3).to_html()}

        <h3>By Hook Type</h3>
        {results_df.groupby('hook')[['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'custom_final_score']].agg(['mean', 'std']).round(3).to_html()}

        <h3>By Token Scheme</h3>
        {results_df.groupby('scheme')[['auroc', 'accuracy', 'f1_hallucinated', 'f1_not_hallucinated', 'custom_final_score']].agg(['mean', 'std']).round(3).to_html()}

        <div class="graph-section">
            <h2>Finalist Visualizations</h2>
            <div class="graph-grid">
                <div>
                    <h3>Best Finalist Multi-Metric Analysis</h3>
                    {create_embedded_img_tag("best_finalist_multi_metric_analysis.png", "Best Finalist Multi-Metric Analysis")}
                    <p><em>Comprehensive analysis of the best finalist including performance profile, distribution analysis, and trade-off analysis.</em></p>
                </div>
                <div>
                    <h3>Finalist Custom Score vs AUROC</h3>
                    {create_embedded_img_tag("finalist_custom_vs_traditional_scatter.png", "Finalist Custom vs Traditional Scatter")}
                    <p><em>Scatter plot comparing custom score with AUROC for all finalists. Point colors represent F1 Macro performance.</em></p>
                </div>
                <div>
                    <h3>Finalist Custom Score by Token Scheme</h3>
                    {create_embedded_img_tag("finalist_token_scheme_analysis_custom_score.png", "Finalist Token Scheme Analysis")}
                    <p><em>Custom score performance across different token schemes for the top finalists.</em></p>
                </div>
            </div>

            <h3>Individual Threshold Optimization Analysis</h3>
            <p><em>Each finalist gets a detailed threshold optimization plot showing how performance metrics change across all possible thresholds:</em></p>
            <div class="graph-grid">
"""

    # Add threshold optimization plots for each finalist
    for idx, row in results_df.iterrows():
        layer, hook, scheme = row['layer'], row['hook'], row['scheme']
        filename = f"threshold_optimization_L{layer:02d}_{hook}_{scheme}.png"
        html_content += f"""
                <div>
                    <h4>Layer {layer} - {hook} - {scheme}</h4>
                    {create_embedded_img_tag(filename, f"Threshold Optimization L{layer:02d}_{hook}_{scheme}")}
                    <p><em>Optimal threshold: {row.get('optimal_threshold', 'N/A'):.4f}, Custom score: {row.get('custom_score_at_optimal', 'N/A'):.3f}</em></p>
                </div>
"""

    html_content += """
            </div>
        </div>

        <h2>Key Finalist Insights</h2>
        <ul>
            <li><strong>Best Layer:</strong> Layer {results_df.loc[best_idx, 'layer']} ({best_metric.upper()}: {best_value:.3f})</li>
            <li><strong>Best Hook:</strong> {results_df.loc[best_idx, 'hook']}</li>
            <li><strong>Best Token Scheme:</strong> {results_df.loc[best_idx, 'scheme']}</li>
            <li><strong>Best Classifier:</strong> {results_df.loc[best_idx, 'classifier']}</li>
"""

    if 'custom_final_score' in results_df.columns:
        custom_best_idx = results_df['custom_final_score'].idxmax()
        html_content += f"""
            <li><strong>Custom Score Best:</strong> <span class="custom-score">Layer {results_df.loc[custom_best_idx, 'layer']}, Hook: {results_df.loc[custom_best_idx, 'hook']}, Scheme: {results_df.loc[custom_best_idx, 'scheme']} ({results_df.loc[custom_best_idx, 'custom_final_score']:.3f})</span></li>
"""

    html_content += f"""
        </ul>

        <div class="stats-box">
            <p><em>Finalist report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            <p><em>Analysis includes {len(results_df)} top-performing models with exact threshold optimization across {results_df['layer'].nunique()} layers, {len(results_df['hook'].unique())} hook types, and {len(results_df['scheme'].unique())} token schemes.</em></p>
            <p><em>These finalists were selected from hundreds of work units based on proxy custom scoring and then received expensive exact threshold optimization.</em></p>
        </div>
    </body>
    </html>
    """

    with open(os.path.join(output_dir, 'finalist_analysis_report.html'), 'w') as f:
        f.write(html_content)

    logger.info(f"Finalist HTML report saved to: {os.path.join(output_dir, 'finalist_analysis_report.html')}")

# ================================================================
# 8. Execute Main Function
# ================================================================

if __name__ == "__main__":
    main()
