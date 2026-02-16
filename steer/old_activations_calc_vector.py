#!/usr/bin/env python3
"""
Build ITI Intervention Vectors from Generator Activations
==========================================================
Converts activations from generator.py into ITI steering vectors compatible with steering_experiment.py.

Workflow:
1. Loads batch-wise HDF5 activations from generator (organized by hook/layer/scheme)
2. Extracts hallucination labels from results pickle files
3. Unflattens attn.hook_z activations: [n_samples, 4096] → [n_samples, 32, 128]
4. Trains logistic regression classifiers per (layer, head) with parallelization
5. Computes steering directions (mean_faithful - mean_unfaithful per head)
6. Ranks heads by validation accuracy and selects top-K
7. Saves ITI intervention config (.pkl) compatible with steering_experiment.py
8. Outputs detailed analysis: top heads per layer, validation accuracies, projection statistics

Usage (defaults to attn.hook_z + last_generated, auto-discovers chunks):
    python -m pipeline.steer.ITI.old_activations_calc_vector --top-k-values 625 630 635 640 645 650 655 660 665 670 675 680 685 690 695 700 705 710 715 720 725 730 735 740 745 750 755 760 765 770 775 780 785 790 795 800 805 810 815 815 820  --output-dir ./data/ITI_old_activations/iti_steering_vectors

With custom hook and scheme:
python -m pipeline.steer.ITI.old_activations_calc_vector        --experiment-dir /home/jovyan/workspace-enqi/data/activations/llama_squadcoqahalu \
        --hook "attn.hook_z" \
        --scheme "last_generated" \
        --top-k 45 \
        --output-dir ./iti_steering_vectors

IMPORTANT: The script automatically discovers:
- All chunk_* directories within the experiment directory
- H5 files within each chunk (both consolidated and individual batch files)
- This matches the pattern used by classifier.py for multi-chunk experiments
"""

import os
import sys
import argparse
import glob
import pickle
import h5py
import json
import numpy as np
import pandas as pd
import re
from typing import Dict, Tuple, Optional, List
from tqdm.auto import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import configuration and utilities from pipeline
from config import ACTIVATIONS_BASE_DIR, ROOT
from logger import consolidated_logger as logger

# ================================================================
# Utilities and Constants
# ================================================================

# Default configuration for Llama-3-8B attention heads
DEFAULT_N_HEADS = 32
DEFAULT_D_HEAD = 128
DEFAULT_FLATTENED_SIZE = DEFAULT_N_HEADS * DEFAULT_D_HEAD  # 4096
DEFAULT_HOOK = "attn.hook_z"
DEFAULT_SCHEME = "last_generated"
DEFAULT_TOP_K = 30
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_SEED = 42
DEFAULT_ACTIVATIONS_DIR = "/home/jovyan/workspace-enqi/data/activations/llama_squadcoqahalu"
DEFAULT_RESULTS_DIR = "/home/jovyan/workspace-enqi/data/activations/llama_squadcoqahalu"

def atomic_load_pickle(filepath):
    """Load pickle file with automatic corruption recovery"""
    backup_filepath = filepath + '.backup'

    # Try main file first
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.warning(f"Corrupted pickle file {filepath}: {e}")

    # Fallback to backup
    if os.path.exists(backup_filepath):
        try:
            with open(backup_filepath, 'rb') as f:
                logger.info(f"Loaded from backup: {backup_filepath}")
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.error(f"Backup file also corrupted {backup_filepath}: {e}")

    return None


def unflatten_attention_activations(activations: np.ndarray, n_heads: int = DEFAULT_N_HEADS, d_head: int = DEFAULT_D_HEAD) -> np.ndarray:
    """
    Unflatten attention hook_z activations from [n_samples, 4096] to [n_samples, n_heads, d_head].
    
    CRITICAL: Generator.py saves attn.hook_z with shape [batch, seq, n_heads, d_head],
    then extracts at token position resulting in [n_samples, n_heads, d_head],
    then stores to HDF5 flattened as [n_samples, n_heads*d_head].
    
    This function reverses the flattening to recover per-head structure.
    
    Args:
        activations: [n_samples, n_heads*d_head] array (e.g., [1000, 4096])
        n_heads: Number of attention heads (default: 32 for Llama-3-8B)
        d_head: Dimension per head (default: 128 for Llama-3-8B)
    
    Returns:
        activations_unflattened: [n_samples, n_heads, d_head] array
    """
    if activations.ndim != 2:
        raise ValueError(f"Expected 2D activations, got shape: {activations.shape}")
    
    expected_flattened_size = n_heads * d_head
    if activations.shape[1] != expected_flattened_size:
        raise ValueError(
            f"Activation second dimension ({activations.shape[1]}) does not match "
            f"expected flattened size ({expected_flattened_size} = {n_heads} heads × {d_head} d_head)"
        )
    
    # Reshape: [n_samples, n_heads*d_head] → [n_samples, n_heads, d_head]
    activations_unflattened = activations.reshape(activations.shape[0], n_heads, d_head)
    
    logger.debug(f"Unflattened activations: {activations.shape} → {activations_unflattened.shape}")
    
    return activations_unflattened


def extract_per_head_activations(activations_3d: np.ndarray, layer_idx: int) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Extract per-head activations from [n_samples, n_heads, d_head] into dict indexed by (layer, head).
    
    Args:
        activations_3d: [n_samples, n_heads, d_head] array
        layer_idx: Layer index for the key
    
    Returns:
        per_head_dict: {(layer_idx, head_idx): [n_samples, d_head]}
    """
    if activations_3d.ndim != 3:
        raise ValueError(f"Expected 3D activations, got shape: {activations_3d.shape}")
    
    n_samples, n_heads, d_head = activations_3d.shape
    per_head_dict = {}
    
    for head_idx in range(n_heads):
        per_head_dict[(layer_idx, head_idx)] = activations_3d[:, head_idx, :]  # [n_samples, d_head]
    
    return per_head_dict

def discover_experiment_chunks(experiment_dir):
    """Discovers all chunk folders within experiment directory.
    
    Args:
        experiment_dir: Absolute path to the experiment directory
        
    Returns:
        Dictionary: {chunk_name: chunk_full_path} or empty dict if no chunks found
    """
    if not experiment_dir or not os.path.isdir(experiment_dir):
        logger.error(f"ERROR: Experiment directory not found: {experiment_dir}")
        return {}

    chunk_pattern = os.path.join(experiment_dir, "chunk_*")
    chunk_dirs = [d for d in glob.glob(chunk_pattern) if os.path.isdir(d)]

    if not chunk_dirs:
        logger.error(f"ERROR: No chunk directories found in {experiment_dir}")
        return {}

    logger.info(f"Found {len(chunk_dirs)} data chunks in experiment.")
    return {os.path.basename(d): d for d in sorted(chunk_dirs)}

def load_multi_chunk_results_data(chunk_directories, experiment_id):
    """Loads results data (labels) from multiple chunk directories"""
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
            logger.warning(f"No result files found in chunk directory: {chunk_dir}")
            continue
            
        # Load the data from the identified files with atomic loading
        for pickle_file in files_to_load:
            try:
                results_data = atomic_load_pickle(pickle_file)
                if results_data is None:
                    logger.warning(f"Failed to load {os.path.basename(pickle_file)} - skipping")
                    continue

                if isinstance(results_data, list):
                    all_results.extend(results_data)
                elif isinstance(results_data, dict):
                    all_results.append(results_data)

            except Exception as e:
                logger.warning(f"Error loading {os.path.basename(pickle_file)}: {e}")
                continue
    
    if not all_results:
        logger.error("No results loaded from any chunk directory")
        return pd.DataFrame(), {}

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Normalize label columns
    if 'is_hallucination' not in results_df.columns:
        if 'is_correct' in results_df.columns:
            results_df['is_hallucination'] = results_df['is_correct'].map(lambda v: 0 if v == 1 else (1 if v == 0 else 2))
        else:
            logger.error("Missing both is_hallucination and is_correct in results data")
            return pd.DataFrame(), {}

    # Drop duplicates
    initial_count = len(results_df)
    results_df = results_df.drop_duplicates(subset=['row_idx'])
    duplicates_removed = initial_count - len(results_df)

    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate records")

    evaluation_counts = results_df['is_hallucination'].value_counts()
    logger.info(f"Loaded {len(results_df)} unique result records.")
    logger.info(f"Evaluation distribution: {dict(evaluation_counts)}")

    class_distribution = {
        'total': len(results_df),
        'hallucination': int(evaluation_counts.get(1, 0)),
        'non_hallucination': int(evaluation_counts.get(0, 0)),
        'api_failures': int(evaluation_counts.get(2, 0))
    }

    return results_df, class_distribution

def discover_available_h5_files(chunk_directories):
    """Discover all HDF5 activation files across chunk directories.
    
    Inspired by classifier.py, searches for both:
    - Consolidated files: consolidated_activations_*.h5
    - Individual batch files: *_batch_*_activations_*.h5
    
    Args:
        chunk_directories: Dictionary or list of chunk directory paths
        
    Returns:
        List of absolute paths to H5 files
    """
    h5_files = []
    
    # Handle both dict and list inputs
    chunk_dirs = chunk_directories.values() if isinstance(chunk_directories, dict) else chunk_directories
    
    for chunk_dir in chunk_dirs:
        if not os.path.isdir(chunk_dir):
            logger.warning(f"Chunk directory not found: {chunk_dir}")
            continue
        
        # 1. Look for consolidated files first
        consolidated = glob.glob(os.path.join(chunk_dir, "consolidated_activations_*.h5"))
        h5_files.extend(consolidated)
        
        # Track which batch indices are already covered by consolidated files
        consolidated_indices = set()
        for f in consolidated:
            match = re.search(r'consolidated_activations_(\d+)_(\d+).h5', os.path.basename(f))
            if match:
                consolidated_indices.update(range(int(match.group(1)), int(match.group(2))))
        
        # 2. Look for individual batch files that aren't covered by consolidated files
        individual = glob.glob(os.path.join(chunk_dir, "*_batch_*_activations_*.h5"))
        individual_count = 0
        for f in individual:
            match = re.search(r'_batch_(\d+)_activations', os.path.basename(f))
            if match and int(match.group(1)) not in consolidated_indices:
                h5_files.append(f)
                individual_count += 1
        
        logger.debug(f"Found {len(consolidated)} consolidated + {individual_count} individual H5 files in {os.path.basename(chunk_dir)}")
    
    logger.info(f"Total H5 files discovered: {len(h5_files)} across {len(list(chunk_dirs))} chunks")
    return h5_files


def discover_available_layers_and_hooks(h5_files: List[str]) -> Tuple[set, set]:
    """
    Discover all available layers and hooks in HDF5 files by scanning metadata.
    
    Args:
        h5_files: List of HDF5 file paths
    
    Returns:
        Tuple of (available_layers, available_hooks) as sorted lists
    """
    available_layers = set()
    available_hooks = set()
    
    for h5_file in h5_files[:5]:  # Sample first 5 files for efficiency
        try:
            with h5py.File(h5_file, 'r') as f:
                for hook_key in f.keys():
                    # Parse hook_key like "layer_10_attn.hook_z"
                    parts = hook_key.split('_', 2)
                    if len(parts) >= 2 and parts[0] == 'layer':
                        try:
                            layer_idx = int(parts[1])
                            available_layers.add(layer_idx)
                            if len(parts) >= 3:
                                hook_name = parts[2]
                                available_hooks.add(hook_name)
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug(f"Error scanning {h5_file}: {e}")
            continue
    
    return sorted(list(available_layers)), sorted(list(available_hooks))


# ================================================================
# Head Classifier and Training
# ================================================================

class HeadClassifier:
    """Trains and stores logistic regression classifier for a single (layer, head) pair"""
    
    def __init__(self, layer: int, head: int, seed: int = DEFAULT_SEED):
        self.layer = layer
        self.head = head
        self.seed = seed
        self.model = LogisticRegression(
            max_iter=100,
            random_state=seed,
            solver='saga',
            penalty='l1',
            C=0.1,
            n_jobs=-1,
            verbose=0
        )
        self.train_accuracy = None
        self.val_accuracy = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train classifier on training set and evaluate on validation set.
        
        Args:
            X_train: (n_train_samples, 128) training activations
            y_train: (n_train_samples,) binary labels (0=faithful, 1=unfaithful)
            X_val: (n_val_samples, 128) validation activations
            y_val: (n_val_samples,) binary labels
            
        Returns:
            dict with training and validation accuracies
        """
        self.model.fit(X_train, y_train)
        self.train_accuracy = self.model.score(X_train, y_train)
        self.val_accuracy = self.model.score(X_val, y_val)
        
        return {
            'layer': self.layer,
            'head': self.head,
            'train_acc': self.train_accuracy,
            'val_acc': self.val_accuracy,
        }


def _train_single_head(args):
    """
    Standalone function for multiprocessing. Trains a single head classifier.
    
    Args:
        args: tuple of (layer, head, activations, row_indices, labels_dict, val_split, seed)
    
    Returns:
        tuple of (layer, head, val_accuracy, classifier) or None if failed
    """
    layer, head, activations, row_indices, labels_dict, val_split, seed = args
    
    try:
        # Get labels for all samples
        labels = np.array([labels_dict.get(int(idx), -1) for idx in row_indices])
        
        # Filter out samples with no label (label == -1)
        valid_mask = labels != -1
        if not np.any(valid_mask):
            return None
        
        activations = activations[valid_mask]
        labels = labels[valid_mask]
        
        # Ensure we have both classes
        if len(np.unique(labels)) < 2:
            return None
        
        # Split with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            activations, labels,
            test_size=val_split,
            random_state=seed,
            stratify=labels
        )
        
        if len(X_train) < 5 or len(X_val) < 5:
            return None
        
        # Train classifier
        classifier = HeadClassifier(layer, head, seed=seed)
        result = classifier.train(X_train, y_train, X_val, y_val)
        
        return (layer, head, result['val_acc'], classifier)
        
    except Exception as e:
        logger.debug(f"Error training head ({layer}, {head}): {e}")
        return None

# ================================================================
# ITI Intervention Builder
# ================================================================

class ITIInterventionBuilder:
    """
    Builds ITI intervention vectors from generator activations.
    
    Pipeline:
    1. Load batch-wise HDF5 activations (organized by hook/layer/scheme)
    2. Extract labels from results pickle files
    3. Unflatten attn.hook_z: [n_samples, 4096] → [n_samples, 32, 128]
    4. Organize activations by (layer, head)
    5. Train logistic regression per (layer, head) with parallelization
    6. Compute directions: mean_faithful - mean_unfaithful
    7. Select top-K heads by validation accuracy
    8. Build and save intervention config pickle
    """
    
    def __init__(self, hook: str = DEFAULT_HOOK, scheme: str = DEFAULT_SCHEME, 
                 top_k: int = DEFAULT_TOP_K, seed: int = DEFAULT_SEED,
                 n_workers: int = -1):
        """
        Args:
            hook: Hook name (default: "attn.hook_z")
            scheme: Token scheme (default: "last_generated")
            top_k: Number of top heads to select
            seed: Random seed for reproducibility
            n_workers: Number of workers for parallelization (-1 = all cores)
        """
        self.hook = hook
        self.scheme = scheme
        self.top_k = top_k
        self.seed = seed
        self.n_workers = n_workers
        
        # Storage
        self.activations_by_layer_head = {}  # {(layer, head): [n_samples, d_head]}
        self.row_indices_by_layer_head = {}  # {(layer, head): [n_samples]}
        self.labels_dict = {}  # {row_idx: label}
        self.classifiers_by_layer_head = {}  # {(layer, head): HeadClassifier}
        self.validation_accuracies = []  # [(layer, head, val_accuracy), ...], sorted by accuracy descending
        self.directions_by_layer_head = {}  # {(layer, head): unnormalized_direction}
        self.projection_stds_by_head = {}  # {(layer, head): projection_std}
        self.top_heads_by_layer = {}  # {layer: [(head, accuracy), ...]}
        
        logger.info(f"ITI Builder initialized: hook={hook}, scheme={scheme}, top_k={top_k}")
    
    def load_labels_from_results(self, chunk_dirs: List[str]):
        """Load hallucination labels from results pickle files"""
        logger.info(f"Loading labels from {len(chunk_dirs)} chunk directories...")
        
        for chunk_dir in chunk_dirs:
            final_results_file = os.path.join(chunk_dir, "all_results.pkl")
            
            if os.path.exists(final_results_file):
                results_data = atomic_load_pickle(final_results_file)
                if results_data:
                    if isinstance(results_data, list):
                        for item in results_data:
                            if 'row_idx' in item and 'is_correct' in item:
                                # Map is_correct to is_hallucination: correct=0, incorrect=1
                                self.labels_dict[item['row_idx']] = 0 if item['is_correct'] else 1
                    logger.debug(f"Loaded {len(results_data)} results from {os.path.basename(chunk_dir)}")
        
        if not self.labels_dict:
            raise ValueError("No labels loaded from results files")
        
        label_counts = {}
        for label in self.labels_dict.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"Loaded {len(self.labels_dict)} labels:")
        logger.info(f"  - Faithful (0): {label_counts.get(0, 0)}")
        logger.info(f"  - Unfaithful (1): {label_counts.get(1, 0)}")
    
    def load_activations_from_h5(self, h5_files: List[str], start_layer: int = None, end_layer: int = None):
        """
        Load activations from HDF5 files and organize by (layer, head).
        Handles unflatten for attn.hook_z.
        """
        logger.info(f"Loading activations from {len(h5_files)} H5 files...")
        logger.info(f"Hook: {self.hook}, Scheme: {self.scheme}")
        
        samples_loaded = 0
        files_processed = 0
        
        for h5_file in tqdm(h5_files, desc="Loading H5 files"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    for hook_key in f.keys():
                        # Parse hook_key like "layer_10_attn.hook_z"
                        parts = hook_key.split('_', 2)
                        if len(parts) < 3:
                            continue
                        
                        try:
                            layer_idx = int(parts[1])
                            hook_name = parts[2]
                        except ValueError:
                            continue
                        
                        # Filter by hook and layer range
                        if hook_name != self.hook:
                            continue
                        if start_layer is not None and layer_idx < start_layer:
                            continue
                        if end_layer is not None and layer_idx > end_layer:
                            continue
                        
                        # Check if scheme exists
                        if self.scheme not in f[hook_key]:
                            continue
                        
                        scheme_group = f[hook_key][self.scheme]
                        
                        # Load data
                        activations = scheme_group['activations'][:]
                        row_indices = scheme_group['row_indices'][:]
                        
                        if activations.shape[0] != len(row_indices):
                            logger.warning(f"Shape mismatch in {hook_key}/{self.scheme}")
                            continue
                        
                        # Unflatten for attn.hook_z: [n_samples, 4096] → [n_samples, 32, 128]
                        if self.hook == "attn.hook_z" and activations.ndim == 2:
                            activations = unflatten_attention_activations(activations, DEFAULT_N_HEADS, DEFAULT_D_HEAD)
                        
                        # Extract per-head activations
                        per_head_dict = extract_per_head_activations(activations, layer_idx)
                        
                        # Store in main storage
                        for (layer, head), head_acts in per_head_dict.items():
                            key = (layer, head)
                            if key not in self.activations_by_layer_head:
                                self.activations_by_layer_head[key] = []
                                self.row_indices_by_layer_head[key] = []
                            
                            self.activations_by_layer_head[key].append(head_acts)
                            self.row_indices_by_layer_head[key].extend(row_indices)
                        
                        samples_loaded += len(row_indices)
                        files_processed += 1
                        
            except Exception as e:
                logger.debug(f"Error reading {h5_file}: {e}")
                continue
        
        # Convert lists to numpy arrays
        for key in self.activations_by_layer_head:
            self.activations_by_layer_head[key] = np.vstack(self.activations_by_layer_head[key])
            self.row_indices_by_layer_head[key] = np.array(self.row_indices_by_layer_head[key])
        
        logger.info(f"Loaded {samples_loaded} samples from {files_processed} files")
        logger.info(f"Organized into {len(self.activations_by_layer_head)} (layer, head) pairs")
    
    def train_head_classifiers(self, n_workers: int = None, val_split: float = DEFAULT_VALIDATION_SPLIT):
        """Train classifiers for all (layer, head) pairs with parallelization"""
        if n_workers is None:
            n_workers = self.n_workers
        
        logger.info(f"\nTraining {len(self.activations_by_layer_head)} head classifiers...")
        logger.info(f"Using {n_workers if n_workers > 0 else 'all'} workers for parallelization")
        
        # Prepare arguments for parallel training
        train_args = []
        for (layer, head), activations in self.activations_by_layer_head.items():
            row_indices = self.row_indices_by_layer_head[(layer, head)]
            train_args.append((layer, head, activations, row_indices, self.labels_dict, val_split, self.seed))
        
        # Train in parallel
        successful_trainings = 0
        validation_accuracies = []  # List of tuples: (layer, head, val_acc)
        
        with ProcessPoolExecutor(max_workers=n_workers if n_workers > 0 else None) as executor:
            futures = {executor.submit(_train_single_head, args): args for args in train_args}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Training classifiers"):
                result = future.result()
                if result:
                    layer, head, val_acc, classifier = result
                    self.classifiers_by_layer_head[(layer, head)] = classifier
                    validation_accuracies.append((layer, head, val_acc))
                    successful_trainings += 1
        
        # Sort by accuracy (descending), matching ITI reference
        self.validation_accuracies = sorted(validation_accuracies, key=lambda x: x[2], reverse=True)
        
        logger.info(f"Successfully trained {successful_trainings}/{len(self.activations_by_layer_head)} classifiers")
    
    def compute_directions(self):
        """Compute steering directions: mean_faithful - mean_unfaithful per head (UNNORMALIZED)"""
        logger.info(f"\nComputing steering directions for {len(self.activations_by_layer_head)} heads...")
        
        for (layer, head), activations in tqdm(self.activations_by_layer_head.items(), desc="Computing directions"):
            row_indices = self.row_indices_by_layer_head[(layer, head)]
            
            # Get labels
            labels = np.array([self.labels_dict.get(int(idx), -1) for idx in row_indices])
            
            # Separate by label
            # label 0 = Faithful, label 1 = Unfaithful
            faithful_mask = labels == 0
            unfaithful_mask = labels == 1
            
            if not (np.any(faithful_mask) and np.any(unfaithful_mask)):
                continue
            
            # Compute means
            mean_faithful = np.mean(activations[faithful_mask], axis=0)
            mean_unfaithful = np.mean(activations[unfaithful_mask], axis=0)
            
            # Direction: what to add to reduce hallucination (UNNORMALIZED - normalization happens in build_intervention_vectors)
            direction = mean_faithful - mean_unfaithful
            
            self.directions_by_layer_head[(layer, head)] = direction
        
        logger.info(f"Computed directions for {len(self.directions_by_layer_head)} heads")
    
    def select_top_k_heads(self):
        """Select top-K heads by validation accuracy and organize by layer"""
        logger.info(f"\nSelecting top-{self.top_k} heads by validation accuracy...")
        
        # validation_accuracies is already sorted by accuracy descending
        if not self.validation_accuracies:
            logger.error("No validation accuracies found. Did training fail?")
            return
        
        top_k_heads = self.validation_accuracies[:self.top_k]
        
        logger.info(f"Top-K selection summary:")
        logger.info(f"  - Total heads trained: {len(self.validation_accuracies)}")
        logger.info(f"  - Selected top-K: {len(top_k_heads)}")
        if top_k_heads:
            logger.info(f"  - Accuracy range: {top_k_heads[-1][2]:.4f} - {top_k_heads[0][2]:.4f}")
        
        # Organize by layer
        self.top_heads_by_layer = {}
        for layer, head, accuracy in top_k_heads:
            if layer not in self.top_heads_by_layer:
                self.top_heads_by_layer[layer] = []
            
            self.top_heads_by_layer[layer].append((head, accuracy))
        
        # Log per-layer summary
        logger.info(f"\nTop heads per layer:")
        for layer in sorted(self.top_heads_by_layer.keys()):
            heads = self.top_heads_by_layer[layer]
            logger.info(f"  Layer {layer}: {len(heads)} heads - accuracies: {[f'{h[1]:.4f}' for h in heads]}")
    
    def build_intervention_config_for_k(self, k: int) -> Dict:
        """Build intervention config for a specific K value using pre-trained classifiers"""
        
        # Select top-K from pre-computed validation accuracies
        top_k_heads = self.validation_accuracies[:k]
        
        intervention_vectors = {}
        projection_stds_by_head = {}
        
        for layer, head, accuracy in top_k_heads:
            direction = self.directions_by_layer_head.get((layer, head))
            if direction is None:
                logger.debug(f"Skipping layer {layer} head {head}: No direction vector")
                continue
            
            try:
                # Normalize direction to unit length
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-10:
                    logger.warning(f"Zero direction for layer {layer} head {head}")
                    continue
                
                normalized_direction = direction / direction_norm
                
                # Project ALL tuning activations onto this normalized direction
                activations = self.activations_by_layer_head[(layer, head)]
                projections = activations @ normalized_direction  # Element-wise dot product
                
                # Compute std of projections
                projection_std = np.std(projections)
                
                if projection_std < 1e-10:
                    logger.warning(f"Zero std for layer {layer} head {head}")
                    continue
                
                # Store for analysis outputs
                projection_stds_by_head[(layer, head)] = projection_std
                
                layer_key = f'layer_{layer}'
                if layer_key not in intervention_vectors:
                    intervention_vectors[layer_key] = []
                
                intervention_vectors[layer_key].append({
                    'head': head,
                    'direction': normalized_direction,  # normalized to unit length
                    'projection_std': float(projection_std),
                    'accuracy': float(accuracy),
                })
                
            except Exception as e:
                logger.warning(f"Error building vector for layer {layer} head {head}: {e}")
                continue
        
        config = {
            'hook': self.hook,
            'scheme': self.scheme,
            'top_k': k,
            'seed': self.seed,
            'n_heads': DEFAULT_N_HEADS,
            'd_head': DEFAULT_D_HEAD,
            'intervention_vectors': intervention_vectors,
        }
        
        return config, projection_stds_by_head
    
    def build_intervention_config(self) -> Dict:
        """Build intervention config for steering_experiment.py (uses self.top_k)"""
        logger.info(f"\nBuilding intervention config for K={self.top_k}...")
        config, projection_stds = self.build_intervention_config_for_k(self.top_k)
        self.projection_stds_by_head = projection_stds
        logger.info(f"Config built with {len(config['intervention_vectors'])} layers")
        return config
    
    def save_config(self, output_path: str):
        """Save intervention config to pickle file"""
        config = self.build_intervention_config()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved intervention config to {output_path}")
        return config
    
    def save_configs_for_multiple_k(self, output_dir: str, k_values: List[int]):
        """Save intervention configs for multiple K values without retraining
        
        Args:
            output_dir: Output directory (will create subdirectories for each k)
            k_values: List of K values to generate configs for
        
        Returns:
            Dictionary mapping k values to their config file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_configs = {}
        
        logger.info(f"\nSaving configs for K values: {k_values}")
        
        for k in sorted(k_values):
            # Skip if k is larger than available heads
            if k > len(self.validation_accuracies):
                logger.warning(f"Skipping K={k}: Only {len(self.validation_accuracies)} heads available")
                continue
            
            try:
                logger.info(f"  Building config for K={k}...")
                config, projection_stds = self.build_intervention_config_for_k(k)
                
                # Create subdirectory for this k value
                k_output_dir = os.path.join(output_dir, f'k_{k}')
                os.makedirs(k_output_dir, exist_ok=True)
                
                # Save config pickle
                config_path = os.path.join(k_output_dir, 'iti_intervention_config.pkl')
                with open(config_path, 'wb') as f:
                    pickle.dump(config, f)
                
                logger.info(f"    Saved config: {config_path}")
                
                # Save per-k analysis
                k_analysis_dir = os.path.join(k_output_dir, 'analysis')
                os.makedirs(k_analysis_dir, exist_ok=True)
                
                # Save top heads for this k
                top_k_data = []
                for layer, head, accuracy in self.validation_accuracies[:k]:
                    top_k_data.append({
                        'layer': layer,
                        'head': head,
                        'accuracy': accuracy,
                        'projection_std': projection_stds.get((layer, head), 0.0),
                    })
                
                top_heads_df = pd.DataFrame(top_k_data)
                top_heads_csv = os.path.join(k_analysis_dir, 'top_heads.csv')
                top_heads_df.to_csv(top_heads_csv, index=False)
                logger.info(f"    Saved analysis: {top_heads_csv}")
                
                saved_configs[k] = config_path
                
            except Exception as e:
                logger.error(f"  Error building config for K={k}: {e}")
                continue
        
        logger.info(f"\nSuccessfully saved {len(saved_configs)} configs")
        return saved_configs
    
    def save_analysis_outputs(self, output_dir: str):
        """Save detailed analysis outputs: CSVs, JSON report, etc."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving analysis outputs to {output_dir}")
        
        # 1. Top heads per layer CSV
        top_heads_data = []
        for layer in sorted(self.top_heads_by_layer.keys()):
            for head, accuracy in self.top_heads_by_layer[layer]:
                projection_std = self.projection_stds_by_head.get((layer, head), 0.0)
                top_heads_data.append({
                    'layer': layer,
                    'head': head,
                    'accuracy': accuracy,
                    'projection_std': projection_std,
                })
        
        top_heads_df = pd.DataFrame(top_heads_data)
        top_heads_csv = os.path.join(output_dir, 'top_heads_per_layer.csv')
        top_heads_df.to_csv(top_heads_csv, index=False)
        logger.info(f"Saved top heads to {top_heads_csv}")
        
        # 2. Summary statistics CSV
        summary_stats = {
            'metric': [
                'total_heads_trained',
                'successful_trainings',
                'top_k_selected',
                'mean_accuracy_topk',
                'min_accuracy_topk',
                'max_accuracy_topk',
                'layers_count',
                'hook',
                'scheme',
            ],
            'value': [
                len(self.activations_by_layer_head),
                len(self.classifiers_by_layer_head),
                len(top_heads_data),
                float(np.mean([h['accuracy'] for h in top_heads_data])) if top_heads_data else 0.0,
                float(np.min([h['accuracy'] for h in top_heads_data])) if top_heads_data else 0.0,
                float(np.max([h['accuracy'] for h in top_heads_data])) if top_heads_data else 0.0,
                len(self.top_heads_by_layer),
                self.hook,
                self.scheme,
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_csv = os.path.join(output_dir, 'summary_statistics.csv')
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved summary to {summary_csv}")
        
        # 3. Full report JSON
        report = {
            'config': {
                'hook': self.hook,
                'scheme': self.scheme,
                'top_k': self.top_k,
                'seed': self.seed,
                'n_heads': DEFAULT_N_HEADS,
                'd_head': DEFAULT_D_HEAD,
            },
            'statistics': summary_stats,
            'top_heads_per_layer': {
                str(layer): [
                    {
                        'head': int(head),
                        'accuracy': float(acc),
                        'projection_std': float(self.projection_stds_by_head.get((layer, head), 0.0)),
                    }
                    for head, acc in heads
                ]
                for layer, heads in self.top_heads_by_layer.items()
            },
            'all_accuracies': {
                f"{layer}_{head}": float(acc)
                for layer, head, acc in self.validation_accuracies
            }
        }
        
        report_json = os.path.join(output_dir, 'detailed_report.json')
        with open(report_json, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved detailed report to {report_json}")


# ================================================================
# Main Function & CLI
# ================================================================

# ================================================================
# Main Function & CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build ITI steering vectors from generator activations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--experiment-dir',
        type=str,
        default=DEFAULT_ACTIVATIONS_DIR,
        help='Absolute path to experiment directory (contains chunk_0, chunk_1, etc.)'
    )
    
    parser.add_argument(
        '--hook',
        type=str,
        default=DEFAULT_HOOK,
        help=f'Hook name to process (default: {DEFAULT_HOOK})'
    )
    
    parser.add_argument(
        '--scheme',
        type=str,
        default=DEFAULT_SCHEME,
        help=f'Token scheme to process (default: {DEFAULT_SCHEME})'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=DEFAULT_TOP_K,
        help=f'Number of top heads to select (default: {DEFAULT_TOP_K}) [DEPRECATED: use --top-k-values]'
    )
    
    parser.add_argument(
        '--top-k-values',
        type=int,
        nargs='+',
        default=None,
        help='Space-separated K values to save (e.g., 24 28 30 48). If not provided, uses --top-k value.'
    )
    
    parser.add_argument(
        '--start-layer',
        type=int,
        default=None,
        help='Start layer index (auto-detect if not provided)'
    )
    
    parser.add_argument(
        '--end-layer',
        type=int,
        default=None,
        help='End layer index (auto-detect if not provided)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./iti_steering_vectors',
        help='Output directory for intervention config and analysis'
    )
    
    parser.add_argument(
        '--n-workers',
        type=int,
        default=-1,
        help='Number of workers for parallelization (-1 = all cores)'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help=f'Validation split ratio (default: {DEFAULT_VALIDATION_SPLIT})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed (default: {DEFAULT_SEED})'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info("ITI INTERVENTION VECTOR BUILDER - FROM GENERATOR ACTIVATIONS")
        logger.info(f"{'='*80}")
        logger.info(f"Experiment directory: {args.experiment_dir}")
        logger.info(f"Hook: {args.hook}")
        logger.info(f"Scheme: {args.scheme}")
        logger.info(f"Top-K: {args.top_k}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"{'='*80}\n")
        
        # Validate experiment directory exists
        if not os.path.isdir(args.experiment_dir):
            raise ValueError(f"Experiment directory not found: {args.experiment_dir}")
        
        # STEP 1: Discover chunk directories (matching classifier.py pattern)
        logger.info("STEP 1: Discovering chunk directories...")
        chunk_directories = discover_experiment_chunks(args.experiment_dir)
        if not chunk_directories:
            raise ValueError(f"No chunk directories found in {args.experiment_dir}")
        
        logger.info(f"Found {len(chunk_directories)} chunks: {list(chunk_directories.keys())}")
        
        # STEP 2: Discover H5 files within all chunks (matching classifier.py pattern)
        logger.info("STEP 2: Discovering H5 activation files within chunks...")
        h5_files = discover_available_h5_files(chunk_directories)
        if not h5_files:
            raise ValueError(f"No H5 files found in any chunk directory. Check file naming conventions.")
        
        logger.info(f"Found {len(h5_files)} H5 activation files")
        # Auto-detect layers if not specified
        if args.start_layer is None or args.end_layer is None:
            available_layers, available_hooks = discover_available_layers_and_hooks(h5_files)
            logger.info(f"Auto-detected layers: {available_layers}")
            logger.info(f"Auto-detected hooks: {available_hooks}")
            
            if args.start_layer is None and available_layers:
                args.start_layer = available_layers[0]
            if args.end_layer is None and available_layers:
                args.end_layer = available_layers[-1]
            
            logger.info(f"Processing layers {args.start_layer} to {args.end_layer}")
        
        # Initialize builder
        builder = ITIInterventionBuilder(
            hook=args.hook,
            scheme=args.scheme,
            top_k=args.top_k,
            seed=args.seed,
            n_workers=args.n_workers
        )
        
        # Load labels from ALL chunk directories
        logger.info(f"\n{'='*80}")
        logger.info("LOADING LABELS")
        logger.info(f"{'='*80}")
        builder.load_labels_from_results(list(chunk_directories.values()))
        
        # Load activations
        logger.info(f"\n{'='*80}")
        logger.info("LOADING ACTIVATIONS")
        logger.info(f"{'='*80}")
        builder.load_activations_from_h5(h5_files, start_layer=args.start_layer, end_layer=args.end_layer)
        
        # Train classifiers
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING CLASSIFIERS")
        logger.info(f"{'='*80}")
        builder.train_head_classifiers(n_workers=args.n_workers, val_split=args.val_split)
        
        # Compute directions
        logger.info(f"\n{'='*80}")
        logger.info("COMPUTING DIRECTIONS")
        logger.info(f"{'='*80}")
        builder.compute_directions()
        
        # Select top-K heads
        logger.info(f"\n{'='*80}")
        logger.info("SELECTING TOP-K HEADS")
        logger.info(f"{'='*80}")
        builder.select_top_k_heads()
        
        # Save outputs
        logger.info(f"\n{'='*80}")
        logger.info("SAVING OUTPUTS")
        logger.info(f"{'='*80}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        config_path = None
        analysis_dir = None
        
        # Parse k values to save
        if args.top_k_values:
            try:
                k_values = args.top_k_values  # Already parsed as list of ints
                logger.info(f"Saving configs for multiple K values: {k_values}")
                builder.save_configs_for_multiple_k(args.output_dir, k_values)
                logger.info(f"Saved {len(k_values)} configs to {args.output_dir}")
            except ValueError as e:
                logger.error(f"Invalid --top-k-values format: {e}")
                return 1
        else:
            # Fallback to single k value
            logger.info(f"Saving config for K={args.top_k}")
            config_path = os.path.join(args.output_dir, 'iti_intervention_config.pkl')
            builder.save_config(config_path)
            
            # Save analysis outputs
            analysis_dir = os.path.join(args.output_dir, 'analysis')
            builder.save_analysis_outputs(analysis_dir)
        
        logger.info(f"\n{'='*80}")
        logger.info("BUILD COMPLETE!")
        logger.info(f"{'='*80}")
        if config_path:
            logger.info(f"Intervention config: {config_path}")
        if analysis_dir:
            logger.info(f"Analysis outputs: {analysis_dir}/")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
