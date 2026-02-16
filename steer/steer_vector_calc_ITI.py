"""
Build ITI Intervention Vectors from Hallucination Activations
===============================================================
Analyzes activations captured by grab_activation_ITI.py by:
1. Loading HDF5 files with pre-extracted activations (already [32, 128] per layer)
2. Organizing activations by (layer, head) and hallucination status
3. Training logistic regression classifiers per head (parallel) - ONCE
4. Computing center-of-mass directions for faithful vs unfaithful
5. Selecting top-K heads for MULTIPLE K values and building intervention vectors
6. Saving separate intervention configs for each K value

Usage:
    # Build ITI for multiple K values from HDF5 activations
    python -m steer.steer_vector_calc_ITI \
        --h5-dir ./data/ITI/activations/ITI_ACTIVATIONS_faitheval_20251120_054057 \
        --top-k 10 50 100 500 1000 \
        --output-dir ./data/ITI/steering_vector/llama/default_setting

"""

import os
import sys
import argparse
import glob
import pickle
import h5py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from tqdm.auto import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# Filter convergence warnings from Logistic Regression
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# Path to HDF5 files (will be set by CLI argument)
H5_DIR = None

# Add pipeline to path
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), '..', 'pipeline')
if os.path.isdir(PIPELINE_DIR):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import ROOT
    from logger import consolidated_logger as logger
else:
    # Fallback if the script is not in the expected directory
    print("Warning: Could not find 'pipeline' directory. Using fallback logging.")
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # Basic logger
    import logging
    logger = logging.getLogger('compare_means')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Basic config
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ================================================================
# Utilities (copied from classifier.py and analyse_saved_activations.py)
# ================================================================

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

def discover_h5_files(h5_dir):
    """Discover all HDF5 activation files in the given directory"""
    if not os.path.isdir(h5_dir):
        logger.error(f"ERROR: H5 directory not found: {h5_dir}")
        return []

    h5_pattern = os.path.join(h5_dir, "*.h5")
    h5_files = glob.glob(h5_pattern)

    if not h5_files:
        logger.error(f"ERROR: No H5 files found in {h5_dir}")
        return []

    logger.info(f"Found {len(h5_files)} H5 files in {h5_dir}")
    return h5_files

def extract_labels_from_h5_metadata(h5_files):
    """Extract hallucination labels directly from H5 file metadata, filtering out API failures"""
    labels_dict = {}  # row_idx -> is_hallucination label
    total_samples = 0
    api_failures_filtered = 0
    
    for h5_file in tqdm(h5_files, desc="Extracting labels from H5 metadata"):
        try:
            with h5py.File(h5_file, 'r') as f:
                for sample_group_name in f.keys():
                    if sample_group_name.startswith('sample_'):
                        try:
                            sample_group = f[sample_group_name]
                            if 'metadata' in sample_group:
                                metadata = sample_group['metadata']
                                row_idx = int(metadata.attrs.get('row_idx', -1))
                                is_hallucination = int(metadata.attrs.get('is_hallucination', -1))
                                
                                if row_idx != -1 and is_hallucination != -1:
                                    # Filter out API failures (label=2), keep only 0 and 1
                                    if is_hallucination in [0, 1]:
                                        labels_dict[row_idx] = is_hallucination
                                        total_samples += 1
                                    elif is_hallucination == 2:
                                        api_failures_filtered += 1
                        except Exception as e:
                            logger.debug(f"Error extracting label from {sample_group_name}: {e}")
                            continue
        except Exception as e:
            logger.warning(f"Error reading {h5_file}: {e}")
            continue
    
    # Compute statistics
    if labels_dict:
        label_counts = {}
        for label in labels_dict.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"Extracted {total_samples} labels from H5 metadata (after filtering API failures):")
        logger.info(f"  - Faithful (0): {label_counts.get(0, 0)}")
        logger.info(f"  - Unfaithful (1): {label_counts.get(1, 0)}")
        logger.info(f"  - API Failures (2) FILTERED OUT: {api_failures_filtered}")
    
    return labels_dict

def discover_available_h5_files(chunk_directories):
    """Discover all HDF5 activation files across chunk directories"""
    h5_files = []
    h5_pattern = "*.h5"
    
    for chunk_dir in chunk_directories:
        chunk_h5_files = glob.glob(os.path.join(chunk_dir, h5_pattern))
        h5_files.extend(chunk_h5_files)
        logger.debug(f"Found {len(chunk_h5_files)} H5 files in {os.path.basename(chunk_dir)}")
    
    logger.info(f"Total H5 files discovered: {len(h5_files)}")
    return h5_files

def discover_available_layers_and_hooks(h5_files: List[str]) -> Tuple[set, set]:
    """
    Discover all available layers and hooks in the HDF5 files by scanning metadata.
    Returns tuple of (available_layers, available_hooks)
    """
    available_layers = set()
    available_hooks = set()
    
    for h5_file in h5_files[:5]:  # Sample first 5 files for efficiency
        try:
            with h5py.File(h5_file, 'r') as f:
                for key in f.keys():
                    if key.startswith('layer_'):
                        # Parse layer_L_hookname format
                        parts = key.split('_', 2)
                        if len(parts) >= 3:
                            try:
                                layer = int(parts[1])
                                hook = parts[2]
                                available_layers.add(layer)
                                available_hooks.add(hook)
                            except ValueError:
                                pass
        except Exception as e:
            logger.debug(f"Error scanning {h5_file}: {e}")
            continue
    
    return sorted(list(available_layers)), sorted(list(available_hooks))

# ================================================================
# NEW: ITI-Specific Components
# ================================================================

class HeadClassifier:
    """
    Trains and stores logistic regression classifier for a single attention head.
    
    This class implements the core hallucination detection for one (layer, head) pair using
    linear classification on head activations. The trained weight vector becomes the
    steering direction for ITI intervention.
    
    Mathematical Foundation:
        - Input: Activation vectors x ∈ ℝ^d_head from attention head (typically d_head=128)
        - Model: Logistic regression P(hallucination|x) = σ(w^T x + b)
        - Steering direction: w (normalized weight vector)
        - Intervention: Add α * w to activations during generation
    
    Training Strategy:
        - Balanced class weighting to handle hallucination imbalance
        - L2 regularization (default) for stability
        - Max 1000 iterations for convergence
        - Train/validation split for head ranking
    
    Attributes:
        layer (int): Layer index (0-based)
        head (int): Head index within layer (0-based)
        seed (int): Random seed for reproducibility
        model (LogisticRegression): Sklearn classifier instance
        train_accuracy (float): Training set accuracy (0-1)
        val_accuracy (float): Validation set accuracy (0-1)
    
    Example:
        >>> classifier = HeadClassifier(layer=15, head=8, seed=42)
        >>> metrics = classifier.train(X_train, y_train, X_val, y_val)
        >>> print(f"Validation accuracy: {metrics['val_accuracy']:.3f}")
        >>> direction = classifier.get_direction()
        >>> # direction is the steering vector for ITI
    """
    
    def __init__(self, layer: int, head: int, seed: int = 42):
        self.layer = layer
        self.head = head
        self.seed = seed
        # Using saga solver as it's good for large datasets, l1 penalty for sparsity
        self.model = LogisticRegression(
            max_iter=1000,  # Increased for convergence
            class_weight='balanced',  # Handle class imbalance
            random_state=seed, 
        )
        self.train_accuracy = None
        self.val_accuracy = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train classifier on training set and evaluate on validation set
        
        Args:
            X_train: (n_train_samples, 128) training activations
            y_train: (n_train_samples,) binary labels
            X_val: (n_val_samples, 128) validation activations
            y_val: (n_val_samples,) binary labels
            
        Returns:
            dict with training and validation accuracies
        """
        # Train on training set
        self.model.fit(X_train, y_train)
        
        # Evaluate
        self.train_accuracy = self.model.score(X_train, y_train)
        self.val_accuracy = self.model.score(X_val, y_val)
        
        return {
            'layer': self.layer,
            'head': self.head,
            'train_acc': self.train_accuracy,
            'val_acc': self.val_accuracy,
        }

# Add this standalone function OUTSIDE the class (required for multiprocessing)
def _train_single_head(args):
    """
    Standalone function for multiprocessing.
    Trains a single head classifier.
    
    Args:
        args: tuple of (layer, head, activations, row_indices, labels_dict, val_split, seed)
    
    Returns:
        tuple of (layer, head, val_accuracy) or None if failed
    """
    layer, head, activations, row_indices, labels_dict, val_split, seed = args
    
    try:
        # Get labels
        labels = np.array([labels_dict.get(idx) for idx in row_indices])
        
        # Ensure we have both classes
        if len(np.unique(labels)) < 2:
            return None
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            activations, labels,
            test_size=val_split,
            random_state=seed,
            stratify=labels
        )
        
        # Scale activations 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        if len(X_train) < 5 or len(X_val) < 5:
            return None
        
        # Train
        classifier = HeadClassifier(layer, head, seed=seed)
        result = classifier.train(X_train, y_train, X_val, y_val)
        
        return (layer, head, result['val_acc'], classifier)
        
    except Exception as e:
        return None


class ITIInterventionBuilder:
    """
    Builds ITI (Inference-Time Intervention) steering vectors from hallucination activations.
    
    This class orchestrates the complete pipeline for creating steering interventions:
    1. **Head Training**: Train classifiers for all (layer, head) pairs in parallel
    2. **Head Ranking**: Rank heads by validation accuracy (hallucination detection quality)
    3. **Direction Computation**: Extract normalized weight vectors as steering directions
    4. **Top-K Selection**: Select top-K most predictive heads for intervention
    5. **Config Export**: Save intervention config for use in steering experiments
    
    Algorithm Overview:
        For each attention head:
        - Train logistic regression: P(hallucination|activation) = σ(w^T x + b)
        - Computing mean directions: μ_faithful, μ_unfaithful
        - Steering direction: v = normalize(μ_faithful - μ_unfaithful)
        OR use classifier weight: v = normalize(w)
        
        Head selection:
        - Rank heads by validation accuracy
        - Select top-K heads (K determined by user, e.g., 1%, 5%, 10% of total heads)
        - Higher accuracy → stronger steering signal
    
    Intervention Application (in steering_experiment.py):
        - During generation, add α * v to selected head activations
        - α (steering strength) controls intervention magnitude
        - Positive α: Steer toward faithful behavior
        - Negative α: Steer toward unfaithful (for validation)
    
    Attributes:
        hook (str): Activation hook name (e.g., 'attn.hook_z')
        scheme (str): Token position scheme (e.g., 'last_generated')
        layers_to_analyze (List[int]): Layer indices to process
        top_k (int): Number of heads to select for intervention
        seed (int): Random seed for reproducibility
        head_classifiers (Dict): Trained classifiers keyed by (layer, head)
        validation_accuracies (List): Ranked list of (layer, head, accuracy)
        intervention_vectors (Dict): Final steering vectors by layer
    
    Example:
        >>> builder = ITIInterventionBuilder(
        ...     hook='attn.hook_z',
        ...     scheme='last_generated',
        ...     layers_to_analyze=range(32),
        ...     top_k=50,
        ...     seed=42
        ... )
        >>> builder.train_all_heads(activations, labels)
        >>> builder.rank_heads_by_validation()
        >>> builder.build_intervention_config()
        >>> builder.save_config('./steering_vector_top50.pkl')
    """
    
    def __init__(self, hook: str, scheme: str, 
                 layers_to_analyze: List[int], top_k: int = 30, seed: int = 42,
                 n_heads: int = None, d_head: int = None):
        """
        Args:
            hook: Hook name (e.g., "attn.hook_z")
            scheme: Activation scheme (e.g., "last_generated")
            layers_to_analyze: List of layer indices to process
            top_k: Number of top heads to select for intervention
            seed: Random seed for reproducibility
            n_heads: Number of heads (detected from data if None)
            d_head: Head dimension (detected from data if None)
        """
        self.hook = hook
        self.scheme = scheme
        self.layers_to_analyze = layers_to_analyze
        self.top_k = top_k
        self.seed = seed
        
        # Storage for intermediate results
        self.head_classifiers = {}  # (layer, head) -> HeadClassifier
        self.validation_accuracies = []  # List of (layer, head, accuracy)
        
        # Final intervention vectors
        self.intervention_vectors = {}  # layer_name -> [(head_id, direction, std)]
        
        # Model dimensions (will be detected from data if not provided)
        self.n_heads = n_heads
        self.d_head = d_head
        
    def train_head_classifiers(self, 
                             activations_by_layer_head: Dict, 
                             row_indices_by_layer_head: Dict,
                             labels_dict: Dict[int, int], 
                             val_split: float = 0.2,
                             n_workers: int = -1):
        """
        *** PARALLELIZED IMPLEMENTATION ***
        Train logistic regression classifiers for each (layer, head) pair using multiprocessing.
        
        Args:
            activations_by_layer_head: Dict mapping (layer, head) -> (n_samples, 128) activations
            row_indices_by_layer_head: Dict mapping (layer, head) -> (n_samples,) row indices
            labels_dict: Dict mapping row_idx -> label (0 or 1)
            val_split: Fraction of data to use for validation
            n_workers: Number of worker processes (-1 = use all cores)
        """
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 1: Training Logistic Regression Classifiers (PARALLELIZED)")
        logger.info(f"{'='*80}")
        
        # Prepare arguments for each head
        # NOTE: On Windows, passing large numpy arrays to multiprocessing can cause hangs
        # We pass smaller data and batch submissions to avoid serialization issues
        work_items = []
        for (layer, head), activations in activations_by_layer_head.items():
            row_indices = row_indices_by_layer_head[(layer, head)]
            work_items.append((layer, head, activations, row_indices, labels_dict, val_split, self.seed))
        
        logger.info(f"Submitting {len(work_items)} heads for parallel training...")
        logger.info(f"Using {n_workers if n_workers > 0 else 'all available'} worker processes")
        
        all_accuracies = []
        n_workers_actual = min(n_workers if n_workers > 0 else os.cpu_count(), 8)  # Cap at 8 to reduce memory overhead
        logger.info(f"Actual workers: {n_workers_actual}")
        
        # Train in parallel with chunking to avoid Windows multiprocessing hangs
        chunk_size = 50  # Process in smaller batches
        with ProcessPoolExecutor(max_workers=n_workers_actual) as executor:
            futures = {}
            
            # Submit in chunks to reduce memory pressure
            for i in range(0, len(work_items), chunk_size):
                chunk = work_items[i:i+chunk_size]
                logger.info(f"Submitting chunk {i//chunk_size + 1}/{(len(work_items) + chunk_size - 1)//chunk_size} ({len(chunk)} heads)...")
                for item in chunk:
                    future = executor.submit(_train_single_head, item)
                    futures[future] = item
            
            logger.info(f"All {len(futures)} tasks submitted. Waiting for completion...")
            
            # Process results as they complete
            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Training classifiers", unit="head"):
                try:
                    result = future.result(timeout=300)  # 5 min timeout per head
                    
                    if result is not None:
                        layer, head, val_acc, classifier = result
                        self.head_classifiers[(layer, head)] = classifier
                        self.validation_accuracies.append((layer, head, val_acc))
                        all_accuracies.append({'layer': layer, 'head': head, 'val_acc': val_acc})
                    
                    completed += 1
                    if completed % 100 == 0:
                        logger.info(f"Completed {completed}/{len(futures)} heads ({100*completed/len(futures):.1f}%)")
                        
                except TimeoutError:
                    item = futures[future]
                    logger.warning(f"Timeout training layer {item[0]} head {item[1]}")
                except Exception as e:
                    item = futures[future]
                    logger.warning(f"Error training layer {item[0]} head {item[1]}: {e}")
        
        logger.info(f"Trained {len(self.head_classifiers)} head classifiers")
        
        # Sort by validation accuracy
        self.validation_accuracies.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Validation accuracies sorted. Top 10 heads: {self.validation_accuracies[:10]}")
        
        logger.info(f"All heads sorted by validation accuracy:")
        for i, (layer, head, acc) in enumerate(self.validation_accuracies, 1):
            logger.info(f"  {i:4d}. Layer {layer:2d} Head {head:2d}: {acc:.4f}")
        
        logger.info(f"\nTop 10 Heads by Validation Accuracy:")
        for i, (layer, head, acc) in enumerate(self.validation_accuracies[:10], 1):
            logger.info(f"  {i:2d}. Layer {layer:2d} Head {head:2d}: {acc:.4f}")
    
    def compute_directions(self, activations_by_layer_head: Dict, row_indices_by_layer_head: Dict,
                           labels_dict: Dict[int, int]):
        """
        Compute center-of-mass directions for each (layer, head) pair
        
        Args:
            activations_by_layer_head: Dict mapping (layer, head) -> activations
            row_indices_by_layer_head: Dict mapping (layer, head) -> row indices
            labels_dict: Dict mapping row_idx -> label (0=Faithful, 1=Unfaithful)
        """
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2: Computing Center-of-Mass Directions")
        logger.info(f"{'='*80}")
        
        directions_by_layer_head = {}  # (layer, head) -> direction vector
        
        for (layer, head), activations in tqdm(activations_by_layer_head.items(),
                                               desc="Computing directions", unit="head"):
            try:
                row_indices = row_indices_by_layer_head[(layer, head)]
                
                # Separate activations by label
                # label 0 = non_hallucination / Faithful
                # label 1 = hallucination / Unfaithful
                correct_mask = np.array([labels_dict.get(idx, -1) == 0 for idx in row_indices])
                incorrect_mask = np.array([labels_dict.get(idx, -1) == 1 for idx in row_indices])
                
                if np.sum(correct_mask) < 2 or np.sum(incorrect_mask) < 2:
                    logger.debug(f"Skipping layer {layer} head {head}: insufficient labeled samples for direction")
                    continue
                
                correct_activations = activations[correct_mask]
                incorrect_activations = activations[incorrect_mask]
                
                # Compute means
                mean_correct = np.mean(correct_activations, axis=0)
                mean_incorrect = np.mean(incorrect_activations, axis=0)
                
                logger.info(f"Layer {layer} Head {head}: correct samples {len(correct_activations)}, incorrect samples {len(incorrect_activations)}")
                logger.info(f"Layer {layer} Head {head}: mean_correct norm {np.linalg.norm(mean_correct):.6f}, mean_incorrect norm {np.linalg.norm(mean_incorrect):.6f}")
                
                # Direction: Faithful - Unfaithful (Correct - Incorrect)
                # This is the direction to *add* to be more faithful
                direction = mean_correct - mean_incorrect
                
                logger.info(f"Layer {layer} Head {head}: direction norm {np.linalg.norm(direction):.6f}")
                
                directions_by_layer_head[(layer, head)] = direction
                
            except Exception as e:
                logger.debug(f"Error computing direction for layer {layer} head {head}: {e}")
                continue
        
        logger.info(f"Computed directions for {len(directions_by_layer_head)} heads")
        return directions_by_layer_head
    
    def build_intervention_vectors_for_k(self, k: int, directions_by_layer_head: Dict,
                                          activations_by_layer_head: Dict,
                                          row_indices_by_layer_head: Dict,
                                          labels_dict: Dict[int, int]) -> Dict:
        """
        Select top K heads and build normalized intervention vectors with projection std
        
        Args:
            k: Number of top heads to select
            directions_by_layer_head: Dict mapping (layer, head) -> direction
            activations_by_layer_head: Dict mapping (layer, head) -> activations (tuning set)
            row_indices_by_layer_head: Dict mapping (layer, head) -> row indices
            labels_dict: Dict mapping row_idx -> label
            
        Returns:
            Dict mapping layer_name -> list of intervention vectors for this K
        """
        intervention_vectors = {}
        
        # Get top K heads by validation accuracy
        if not self.validation_accuracies:
             logger.error("No validation accuracies found. Did training fail?")
             return intervention_vectors
             
        top_k_heads = self.validation_accuracies[:k]
        logger.info(f"Selected top {k} heads: {[(layer, head, acc) for layer, head, acc in top_k_heads]}")
        logger.debug(f"Selected top {len(top_k_heads)} heads for K={k}")
        
        # Group by layer
        heads_by_layer = {}
        for layer, head, acc in top_k_heads:
            if layer not in heads_by_layer:
                heads_by_layer[layer] = []
            heads_by_layer[layer].append((head, acc))
        
        # Build intervention vectors
        for layer in heads_by_layer:
            # The key must match the hook name in the model for intervention
            # e.g., "model.layers.10.self_attn.hook_z"
            # For now, we just use "layer_10" as a generic key
            intervention_vectors[f"layer_{layer}"] = []
            
            for head, acc in heads_by_layer[layer]:
                if (layer, head) not in directions_by_layer_head:
                    logger.debug(f"Skipping Layer {layer} Head {head}: No direction vector computed.")
                    continue
                
                try:
                    direction = directions_by_layer_head[(layer, head)]
                    
                    # Normalize direction to unit length
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm < 1e-10:
                        logger.warning(f"Zero direction for layer {layer} head {head}")
                        continue
                    
                    normalized_direction = direction / direction_norm
                    
                    logger.info(f"Layer {layer} Head {head}: direction norm before normalization {direction_norm:.6f}, after {np.linalg.norm(normalized_direction):.6f}")
                    
                    # Project *all* tuning activations (both classes) onto this direction
                    activations = activations_by_layer_head[(layer, head)]
                    projections = activations @ normalized_direction  # Shape: (n_samples,)
                    
                    # Compute std of projections
                    projection_std = np.std(projections)
                    
                    logger.info(f"Layer {layer} Head {head}: projections shape {projections.shape}, std {projection_std:.6f}")
                    
                    if projection_std < 1e-10:
                        logger.warning(f"Zero std for layer {layer} head {head}")
                        continue

                    # Store intervention vector
                    intervention_vectors[f"layer_{layer}"].append({
                        'head': head,
                        'direction': normalized_direction,  # (128,)
                        'projection_std': projection_std,
                        'accuracy': acc,
                    })
                    
                    logger.debug(f"Layer {layer} Head {head}: std={projection_std:.6f}, acc={acc:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Error building vector for layer {layer} head {head}: {e}")
                    continue
        
        total_heads = sum(len(v) for v in intervention_vectors.values())
        logger.debug(f"Built intervention vectors for {total_heads} heads with K={k}")
        
        return intervention_vectors
    
    def build_intervention_vectors(self, directions_by_layer_head: Dict,
                                   activations_by_layer_head: Dict,
                                   row_indices_by_layer_head: Dict,
                                   labels_dict: Dict[int, int]):
        """
        Build intervention vectors for the configured top_k value (legacy method)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 3: Building Top-K={self.top_k} Intervention Vectors")
        logger.info(f"{'='*80}")
        
        self.intervention_vectors = self.build_intervention_vectors_for_k(
            self.top_k,
            directions_by_layer_head,
            activations_by_layer_head,
            row_indices_by_layer_head,
            labels_dict
        )
        
        logger.info(f"Built intervention vectors for {sum(len(v) for v in self.intervention_vectors.values())} heads")
        
        return self.intervention_vectors
    
    def save_intervention_config(self, output_path: str, intervention_vectors: Dict = None, k: int = None):
        """Save intervention configuration for deployment
        
        Args:
            output_path: Path to save the pickle file
            intervention_vectors: Intervention vectors to save (if None, uses self.intervention_vectors)
            k: K value used to build these vectors (for logging)
        """
        if intervention_vectors is None:
            intervention_vectors = self.intervention_vectors
        
        if k is None:
            k = self.top_k
        
        config = {
            'hook': self.hook,
            'scheme': self.scheme,
            'top_k': k,
            'seed': self.seed,
            'intervention_vectors': intervention_vectors,
            'n_heads': self.n_heads,
            'd_head': self.d_head,
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved intervention config (K={k}) to {output_path}")
        return output_path
    
    def save_accuracy_heatmap(self, output_dir: str):
        """
        Create and save a heatmap of validation accuracies per layer and head.
        
        Args:
            output_dir: Directory to save the heatmap figure
        """
        if not self.validation_accuracies:
            logger.warning("No validation accuracies available for heatmap")
            return None
        
        # Create a matrix: layers x heads, filled with NaN initially
        layers = sorted(list(set(layer for layer, head, _ in self.validation_accuracies)))
        
        if not layers:
            logger.warning("No layers found in validation accuracies")
            return None
        
        min_layer = min(layers)
        max_layer = max(layers)
        n_layers = max_layer - min_layer + 1
        
        # Detect number of heads from data (max head index + 1)
        max_head = max(head for layer, head, _ in self.validation_accuracies)
        n_heads = max_head + 1
        logger.info(f"Detected {n_heads} heads from validation accuracies")
        
        # Initialize heatmap matrix with NaN
        heatmap_data = np.full((n_layers, n_heads), np.nan)
        
        # Fill in the accuracies
        for layer, head, accuracy in self.validation_accuracies:
            layer_idx = layer - min_layer
            heatmap_data[layer_idx, head] = accuracy
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(8, n_layers * 0.4)))
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            xticklabels=list(range(n_heads)),
            yticklabels=[min_layer + i for i in range(n_layers)],
            cmap='RdYlGn',
            vmin=0.0,
            vmax=1.0,
            cbar_kws={'label': 'Validation Accuracy'},
            ax=ax,
            square=False,
            linewidths=0.1,
            linecolor='gray'
        )
        
        ax.set_xlabel('Head Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_title('Validation Accuracy Heatmap per Layer and Head\n(ITI Classifier Discriminative Power)', 
                     fontsize=14, fontweight='bold')
        
        # Rotate tick labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'validation_accuracy_heatmap.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved validation accuracy heatmap to {output_path}")
        plt.close(fig)
        
        return output_path
    
    def generate_insight_plots(self, output_dir: str, directions_by_layer_head: Dict,
                                activations_by_layer_head: Dict, row_indices_by_layer_head: Dict,
                                labels_dict: Dict[int, int], intervention_vectors: Dict = None):
        """
        Generate comprehensive insight plots for ITI analysis
        
        Args:
            output_dir: Directory to save plots
            directions_by_layer_head: Dict mapping (layer, head) -> direction vector
            activations_by_layer_head: Dict mapping (layer, head) -> activations
            row_indices_by_layer_head: Dict mapping (layer, head) -> row indices
            labels_dict: Dict mapping row_idx -> label
            intervention_vectors: Dict of intervention vectors (for std vs accuracy plot)
        """
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING INSIGHT VISUALIZATIONS")
        logger.info(f"{'='*80}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribution of Validation Accuracies
        logger.info("Generating validation accuracy distribution...")
        try:
            accuracies = [acc for _, _, acc in self.validation_accuracies]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(accuracies, bins=30, kde=True, color='blue', ax=ax)
            ax.axvline(np.mean(accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(accuracies):.4f}')
            ax.axvline(np.median(accuracies), color='green', linestyle='--', label=f'Median: {np.median(accuracies):.4f}')
            ax.set_xlabel('Validation Accuracy', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Head Classifier Accuracies', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'accuracy_distribution.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved accuracy distribution to {output_path}")
        except Exception as e:
            logger.error(f"Error generating accuracy distribution: {e}")
        
        # 2. Cumulative Accuracy vs. K (Top-K Selection)
        logger.info("Generating top-K accuracy curve...")
        try:
            k_values = list(range(1, min(len(self.validation_accuracies) + 1, 500)))
            avg_accuracies = [np.mean([acc for _, _, acc in self.validation_accuracies[:k]]) for k in k_values]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(k_values, avg_accuracies, marker='o', markersize=3, linewidth=2)
            ax.set_xlabel('K (Number of Top Heads)', fontsize=12)
            ax.set_ylabel('Average Validation Accuracy', fontsize=12)
            ax.set_title('Average Accuracy of Top-K Heads', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'top_k_accuracy_curve.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved top-K accuracy curve to {output_path}")
        except Exception as e:
            logger.error(f"Error generating top-K accuracy curve: {e}")
        
        # 3. Layer-Wise Accuracy Box Plot
        logger.info("Generating layer-wise accuracy box plot...")
        try:
            layer_data = {}
            for layer, head, acc in self.validation_accuracies:
                if layer not in layer_data:
                    layer_data[layer] = []
                layer_data[layer].append(acc)
            
            sorted_layers = sorted(layer_data.keys())
            fig, ax = plt.subplots(figsize=(14, 6))
            bp = ax.boxplot([layer_data[layer] for layer in sorted_layers],
                           labels=sorted_layers,
                           patch_artist=True,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_xlabel('Layer Index', fontsize=12)
            ax.set_ylabel('Validation Accuracy', fontsize=12)
            ax.set_title('Accuracy Distribution per Layer (Box Plot)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'layer_accuracy_boxplot.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved layer-wise accuracy box plot to {output_path}")
        except Exception as e:
            logger.error(f"Error generating layer-wise box plot: {e}")
        
        # 4. Direction Vector Norms Distribution
        logger.info("Generating direction vector norms distribution...")
        try:
            norms = [np.linalg.norm(direction) for direction in directions_by_layer_head.values()]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(norms, bins=30, kde=True, color='green', ax=ax)
            ax.axvline(np.mean(norms), color='red', linestyle='--', label=f'Mean: {np.mean(norms):.4f}')
            ax.axvline(np.median(norms), color='orange', linestyle='--', label=f'Median: {np.median(norms):.4f}')
            ax.set_xlabel('Direction Vector Norm (L2)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Direction Vector Norms', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'direction_norms_distribution.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved direction norms distribution to {output_path}")
        except Exception as e:
            logger.error(f"Error generating direction norms distribution: {e}")
        
        # 5. Projection STD vs. Accuracy Scatter Plot
        if intervention_vectors:
            logger.info("Generating projection STD vs accuracy scatter plot...")
            try:
                accuracies_list = []
                stds_list = []
                layers_list = []
                for layer_key, layer_vecs in intervention_vectors.items():
                    layer_idx = int(layer_key.split('_')[1])
                    for head_info in layer_vecs:
                        accuracies_list.append(head_info['accuracy'])
                        stds_list.append(head_info['projection_std'])
                        layers_list.append(layer_idx)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(accuracies_list, stds_list, c=layers_list, 
                                    alpha=0.7, cmap='viridis', s=50)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Layer Index', fontsize=11)
                ax.set_xlabel('Validation Accuracy', fontsize=12)
                ax.set_ylabel('Projection STD', fontsize=12)
                ax.set_title('Accuracy vs. Projection STD for Selected Heads', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                output_path = os.path.join(output_dir, 'accuracy_vs_std_scatter.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved accuracy vs STD scatter plot to {output_path}")
            except Exception as e:
                logger.error(f"Error generating accuracy vs STD scatter: {e}")
        
        # 6. Head Activation Projection Histograms (Top 5 heads)
        logger.info("Generating head activation projection histograms (top 5 heads)...")
        try:
            top_5_heads = self.validation_accuracies[:5]
            for rank, (layer, head, acc) in enumerate(top_5_heads, 1):
                try:
                    if (layer, head) not in directions_by_layer_head:
                        continue
                    if (layer, head) not in activations_by_layer_head:
                        continue
                    
                    direction = directions_by_layer_head[(layer, head)]
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm < 1e-10:
                        continue
                    normalized_direction = direction / direction_norm
                    
                    activations = activations_by_layer_head[(layer, head)]
                    row_indices = row_indices_by_layer_head[(layer, head)]
                    labels = np.array([labels_dict.get(idx, -1) for idx in row_indices])
                    
                    projections = activations @ normalized_direction
                    
                    faithful_projections = projections[labels == 0]
                    unfaithful_projections = projections[labels == 1]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(faithful_projections, bins=30, alpha=0.6, label=f'Faithful (n={len(faithful_projections)})', color='blue')
                    ax.hist(unfaithful_projections, bins=30, alpha=0.6, label=f'Unfaithful (n={len(unfaithful_projections)})', color='red')
                    ax.axvline(np.mean(faithful_projections), color='blue', linestyle='--', alpha=0.8)
                    ax.axvline(np.mean(unfaithful_projections), color='red', linestyle='--', alpha=0.8)
                    ax.set_xlabel('Projection onto Direction Vector', fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'Rank {rank}: Layer {layer} Head {head} (Acc={acc:.4f})', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'projection_histogram_rank{rank}_layer{layer}_head{head}.png')
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    logger.debug(f"Error generating projection histogram for layer {layer} head {head}: {e}")
                    continue
            logger.info(f"Saved projection histograms for top 5 heads")
        except Exception as e:
            logger.error(f"Error generating projection histograms: {e}")
        
        # 7. PCA of Direction Vectors
        logger.info("Generating PCA of direction vectors...")
        try:
            if len(directions_by_layer_head) >= 2:
                directions_list = []
                layer_labels = []
                head_labels = []
                for (layer, head), direction in directions_by_layer_head.items():
                    directions_list.append(direction)
                    layer_labels.append(layer)
                    head_labels.append(head)
                
                directions_array = np.array(directions_list)
                pca = PCA(n_components=2)
                reduced_pca = pca.fit_transform(directions_array)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(reduced_pca[:, 0], reduced_pca[:, 1], 
                                    c=layer_labels, alpha=0.7, cmap='viridis', s=50)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Layer Index', fontsize=11)
                ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
                ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
                ax.set_title('PCA of Direction Vectors (All Heads)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                output_path = os.path.join(output_dir, 'direction_vectors_pca.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved PCA plot to {output_path}")
        except Exception as e:
            logger.error(f"Error generating PCA plot: {e}")
        
        # 8. t-SNE of Direction Vectors (if enough samples)
        logger.info("Generating t-SNE of direction vectors...")
        try:
            if len(directions_by_layer_head) >= 30:
                directions_list = []
                layer_labels = []
                for (layer, head), direction in directions_by_layer_head.items():
                    directions_list.append(direction)
                    layer_labels.append(layer)
                
                directions_array = np.array(directions_list)
                # Use perplexity based on sample size
                perplexity = min(30, len(directions_list) // 3)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                reduced_tsne = tsne.fit_transform(directions_array)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], 
                                    c=layer_labels, alpha=0.7, cmap='viridis', s=50)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Layer Index', fontsize=11)
                ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
                ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
                ax.set_title('t-SNE of Direction Vectors (All Heads)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                output_path = os.path.join(output_dir, 'direction_vectors_tsne.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved t-SNE plot to {output_path}")
            else:
                logger.info(f"Skipping t-SNE: need at least 30 samples, have {len(directions_by_layer_head)}")
        except Exception as e:
            logger.error(f"Error generating t-SNE plot: {e}")
        
        logger.info(f"{'='*80}")
        logger.info("INSIGHT VISUALIZATIONS COMPLETE")
        logger.info(f"{'='*80}")
    
    def print_summary(self, intervention_vectors: Dict = None, k: int = None):
        """Print summary of intervention configuration
        
        Args:
            intervention_vectors: Vectors to summarize (if None, uses self.intervention_vectors)
            k: K value (for logging purposes)
        """
        if intervention_vectors is None:
            intervention_vectors = self.intervention_vectors
        if k is None:
            k = self.top_k
        
        logger.info(f"\n{'='*80}")
        logger.info(f"INTERVENTION CONFIGURATION SUMMARY (K={k})")
        logger.info(f"{'='*80}")
        logger.info(f"Top K heads selected: {k}")
        logger.info(f"Total heads in intervention: {sum(len(v) for v in intervention_vectors.values())}")
        
        for layer_key in sorted(intervention_vectors.keys(), key=lambda x: int(x.split('_')[1])):
            heads = intervention_vectors[layer_key]
            logger.info(f"\n{layer_key}: ({len(heads)} heads)")
            # Sort heads by head index
            for head_info in sorted(heads, key=lambda x: x['head']):
                logger.info(f"  Head {head_info['head']:2d}: "
                            f"accuracy={head_info['accuracy']:.4f}, "
                            f"projection_std={head_info['projection_std']:.6f}")

# ================================================================
# Updated Main Analysis Class
# ================================================================

class ActivationMeanAnalyzer:
    """
    Analyzes hallucinated vs non-hallucinated activations from HDF5 files.
    Builds ITI intervention vectors for inference-time application.
    
    Uses activations pre-extracted by grab_activation_ITI.py:
    - Each sample's [32, 128] activations per layer stored in HDF5
    - Labels (is_hallucination) stored in metadata
    - Organized by (layer, head) for ITI building
    """
    
    def __init__(self, h5_dir: str, top_k: int = 30, start_layer: int = None, end_layer: int = None):
        self.h5_dir = h5_dir
        self.top_k = top_k
        self.start_layer = start_layer
        self.end_layer = end_layer
        
        # Hook and scheme (from HDF5 structure)
        self.hook = "attn.hook_z"
        self.scheme = "last_generated"
        
        # For all_layers mode, this will be populated
        self.layers_to_analyze = []
        
        # Data storage
        self.h5_files = []
        
        # For ITI: store by (layer, head)
        self.activations_by_layer_head = {}
        self.row_indices_by_layer_head = {}
        self.labels_dict = {}
        
        # Detected model dimensions
        self.detected_n_heads = None
        self.detected_d_head = None
        
        # ITI builder
        self.iti_builder = None
        
    def discover_data(self):
        """Discover and load activation data from HDF5 files"""
        logger.info(f"\n{'='*80}")
        logger.info(f"BUILDING ITI INTERVENTION VECTORS")
        logger.info(f"{'='*80}")
        logger.info(f"H5 Directory: {self.h5_dir}")
        logger.info(f"Top-K: {self.top_k}")
        if self.start_layer is not None:
            logger.info(f"Layer range: {self.start_layer} - {self.end_layer}")
        logger.info(f"{'='*80}")
        
        # Validate h5_dir exists
        if not os.path.isdir(self.h5_dir):
            raise ValueError(f"H5 directory not found: {self.h5_dir}")
        
        # Discover H5 files
        self.h5_files = discover_h5_files(self.h5_dir)
        if not self.h5_files:
            raise ValueError(f"No H5 files found in {self.h5_dir}")
            
        # Extract labels from H5 metadata
        self.labels_dict = extract_labels_from_h5_metadata(self.h5_files)
        
        if not self.labels_dict:
            raise ValueError("No labels extracted from H5 files")
        
        logger.info(f"Data discovery complete:")
        logger.info(f"  - H5 files: {len(self.h5_files)}")
        logger.info(f"  - Total samples: {len(self.labels_dict)}")
        
        # Auto-discover available layers if not specified
        if self.start_layer is None or self.end_layer is None:
            self.layers_to_analyze = self._discover_layers_from_h5()
            if self.start_layer is not None:
                self.layers_to_analyze = [l for l in self.layers_to_analyze if l >= self.start_layer and l <= self.end_layer]
        else:
            self.layers_to_analyze = list(range(self.start_layer, self.end_layer + 1))
        
        logger.info(f"Will analyze {len(self.layers_to_analyze)} layers: {self.layers_to_analyze}")
        
    def _discover_layers_from_h5(self):
        """Discover available layers from HDF5 files"""
        layers = set()
        for h5_file in self.h5_files[:5]:  # Sample first 5 files
            try:
                with h5py.File(h5_file, 'r') as f:
                    for sample_group_name in f.keys():
                        if sample_group_name.startswith('sample_'):
                            sample_group = f[sample_group_name]
                            if 'activations_last_gen_token' in sample_group:
                                acts_group = sample_group['activations_last_gen_token']
                                for layer_name in acts_group.keys():
                                    if layer_name.startswith('layer_'):
                                        layer_idx = int(layer_name.split('_')[1])
                                        layers.add(layer_idx)
                            break  # Only need one sample per file
            except Exception as e:
                logger.debug(f"Error discovering layers from {h5_file}: {e}")
                continue
        return sorted(list(layers))
        
    def load_activations_for_iti(self):
        """
        Load all layer activations organized by (layer, head) for ITI building.
        
        From grab_activation_ITI.py HDF5 format:
        - Each sample_{idx}/activations_last_gen_token/layer_{L} is [32, 128]
        - Reshape to [32*128] = [4096] then organize by head
        """
        logger.info(f"\n{'='*80}")
        logger.info("LOADING ACTIVATIONS FOR ITI (By Layer and Head)")
        logger.info(f"{'='*80}")
        
        # Will be auto-detected from first activation
        detected_n_heads = None
        detected_d_head = None
        
        total_loaded = 0
        samples_loaded = set()  # Track unique samples loaded
        
        # Note: We'll initialize dicts dynamically after detecting dimensions
        # to avoid wasting memory on unused head slots
        
        # Load from all HDF5 files
        for h5_file in tqdm(self.h5_files, desc="Loading activations by (layer, head)"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    for sample_group_name in f.keys():
                        if not sample_group_name.startswith('sample_'):
                            continue
                        
                        try:
                            sample_group = f[sample_group_name]
                            
                            # Get metadata
                            if 'metadata' not in sample_group:
                                continue
                            metadata = sample_group['metadata']
                            row_idx = int(metadata.attrs.get('row_idx', -1))
                            
                            if row_idx == -1 or row_idx not in self.labels_dict:
                                continue
                            
                            # Get activations
                            if 'activations_last_gen_token' not in sample_group:
                                continue
                            acts_group = sample_group['activations_last_gen_token']
                            
                            # Track this sample
                            samples_loaded.add(row_idx)
                            
                            # Load activations for each layer
                            for layer_name in acts_group.keys():
                                if not layer_name.startswith('layer_'):
                                    continue
                                
                                layer_idx = int(layer_name.split('_')[1])
                                
                                # Only load specified layers
                                if layer_idx not in self.layers_to_analyze:
                                    continue
                                
                                # Load activation (shape depends on model: [n_heads, d_head])
                                activation = acts_group[layer_name][:]  # Shape: [n_heads, d_head]
                                
                                # Auto-detect dimensions from first activation
                                if detected_n_heads is None:
                                    detected_n_heads = activation.shape[0]
                                    detected_d_head = activation.shape[1]
                                    logger.info(f"Auto-detected from first activation: n_heads={detected_n_heads}, d_head={detected_d_head}")
                                    logger.info(f"Loading first sample: layer {layer_idx}, activation shape: {activation.shape}")
                                
                                # Increment counter (counts layer-sample combinations)
                                total_loaded += 1
                                
                                # Split by head and store (dynamically initialize dict keys as needed)
                                actual_n_heads = activation.shape[0]
                                for head_idx in range(actual_n_heads):
                                    head_activation = activation[head_idx]  # [d_head]
                                    key_tuple = (layer_idx, head_idx)
                                    
                                    # Initialize dict entry if not exists
                                    if key_tuple not in self.activations_by_layer_head:
                                        self.activations_by_layer_head[key_tuple] = []
                                        self.row_indices_by_layer_head[key_tuple] = []
                                    
                                    self.activations_by_layer_head[key_tuple].append(head_activation)
                                    self.row_indices_by_layer_head[key_tuple].append(row_idx)
                                
                                total_loaded += 1
                        
                        except Exception as e:
                            logger.debug(f"Error loading sample from {sample_group_name}: {e}")
                            continue
            
            except Exception as e:
                logger.warning(f"Error reading {h5_file}: {e}")
                continue
        
        # Convert lists to numpy arrays
        empty_keys = []
        for key_tuple in self.activations_by_layer_head:
            if self.activations_by_layer_head[key_tuple]:
                self.activations_by_layer_head[key_tuple] = np.array(self.activations_by_layer_head[key_tuple])
                self.row_indices_by_layer_head[key_tuple] = np.array(self.row_indices_by_layer_head[key_tuple])
            else:
                empty_keys.append(key_tuple)
        
        # Clean up empty keys
        for key in empty_keys:
            del self.activations_by_layer_head[key]
            del self.row_indices_by_layer_head[key]

        logger.info(f"Loaded {len(self.activations_by_layer_head)} (layer, head) pairs with data")
        logger.info(f"Total samples loaded: {len(samples_loaded)}")
        logger.info(f"Total layer-sample combinations: {total_loaded}")
        
        # Store detected dimensions
        self.detected_n_heads = detected_n_heads
        self.detected_d_head = detected_d_head
        
        if self.detected_n_heads is not None:
            logger.info(f"\nDetected model dimensions:")
            logger.info(f"  - Number of heads: {self.detected_n_heads}")
            logger.info(f"  - Head dimension: {self.detected_d_head}")
        
        # Print summary
        layer_counts = {}
        for (layer, head) in self.activations_by_layer_head.keys():
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1
        
        # Report using auto-detected dimensions
        if detected_n_heads is not None:
            for layer in sorted(layer_counts.keys()):
                logger.info(f"  - Layer {layer:2d}: {layer_counts[layer]}/{detected_n_heads} heads loaded")
        else:
            logger.warning("No activations loaded - could not detect dimensions")
    
    def build_iti_intervention(self) -> ITIInterventionBuilder:
        """Build ITI intervention vectors"""
        
        logger.info(f"\n{'='*80}")
        logger.info("BUILDING ITI INTERVENTION VECTORS")
        logger.info(f"{'='*80}")
        
        # Create builder with detected dimensions
        self.iti_builder = ITIInterventionBuilder(
            self.hook,
            self.scheme,
            self.layers_to_analyze,
            top_k=self.top_k,
            n_heads=self.detected_n_heads,
            d_head=self.detected_d_head
        )
        
        # Phase 1: Train classifiers
        self.iti_builder.train_head_classifiers(
            self.activations_by_layer_head,
            self.row_indices_by_layer_head,
            self.labels_dict
        )
        
        # Phase 2: Compute directions
        directions = self.iti_builder.compute_directions(
            self.activations_by_layer_head,
            self.row_indices_by_layer_head,
            self.labels_dict
        )
        
        # Phase 3: Build intervention vectors
        self.iti_builder.build_intervention_vectors(
            directions,
            self.activations_by_layer_head,
            self.row_indices_by_layer_head,
            self.labels_dict
        )
        
        # Print summary
        self.iti_builder.print_summary()
        
        return self.iti_builder
        
    def compute_mean_difference(self):
        """Compute mean activations and their difference"""
        
        logger.info(f"\n{'='*80}")
        logger.info("MEAN ACTIVATION ANALYSIS")
        logger.info(f"{'='*80}")
        
        if self.hallucinated_activations.size == 0 or self.non_hallucinated_activations.size == 0:
            logger.error("Insufficient data for mean analysis (one class is empty).")
            return None

        # Compute means
        mean_hallucinated = np.mean(self.hallucinated_activations, axis=0)
        mean_non_hallucinated = np.mean(self.non_hallucinated_activations, axis=0)
        
        # Compute difference: non_hallucinated - hallucinated (Faithful - Unfaithful)
        mean_difference = mean_non_hallucinated - mean_hallucinated
        
        logger.info(f"\nMean Activation Shapes:")
        logger.info(f"  - Mean hallucinated shape:      {mean_hallucinated.shape}")
        logger.info(f"  - Mean non-hallucinated shape:  {mean_non_hallucinated.shape}")
        logger.info(f"  - Difference shape:             {mean_difference.shape}")
        
        # Statistics about the difference
        logger.info(f"\nDifference Statistics (Faithful - Unfaithful):")
        logger.info(f"  - Mean of difference:     {np.mean(mean_difference):.8f}")
        logger.info(f"  - Std of difference:      {np.std(mean_difference):.8f}")
        logger.info(f"  - Min of difference:      {np.min(mean_difference):.8f}")
        logger.info(f"  - Max of difference:      {np.max(mean_difference):.8f}")
        logger.info(f"  - Median of difference:   {np.median(mean_difference):.8f}")
        logger.info("")
        logger.info("Interpretation of difference:")
        logger.info("  - Positive value: Feature is STRONGER in FAITHFUL answers")
        logger.info("  - Negative value: Feature is WEAKER in FAITHFUL answers (Stronger in Unfaithful)")
        
        # Statistics about mean values
        logger.info(f"\nHallucinated Mean Activation (Label 1 - Unfaithful Answers):")
        logger.info(f"  - Mean:   {np.mean(mean_hallucinated):.8f}")
        logger.info(f"  - Std:    {np.std(mean_hallucinated):.8f}")
        
        logger.info(f"\nNon-Hallucinated Mean Activation (Label 0 - Faithful Answers):")
        logger.info(f"  - Mean:   {np.mean(mean_non_hallucinated):.8f}")
        logger.info(f"  - Std:    {np.std(mean_non_hallucinated):.8f}")
        
        # Find features with largest differences (flatten if multi-dimensional)
        mean_diff_flat = mean_difference.flatten()
        abs_diff = np.abs(mean_diff_flat)
        top_10_indices = np.argsort(abs_diff)[-10:][::-1]
        
        logger.info(f"\nTop 10 Features with Largest Absolute Differences:")
        logger.info("-" * 80)
        
        # Pre-flatten mean activations for easier indexing
        mean_non_hal_flat = mean_non_hallucinated.flatten()
        mean_hal_flat = mean_hallucinated.flatten()

        for rank, idx in enumerate(top_10_indices, 1):
            logger.info(f"  {rank:2d}. Feature {idx:4d}: "
                        f"diff={mean_diff_flat[idx]:12.8f}, "
                        f"faithful_mean={mean_non_hal_flat[idx]:12.8f}, "
                        f"unfaithful_mean={mean_hal_flat[idx]:12.8f}")
        
        # Count positive vs negative differences
        positive_diffs = np.sum(mean_diff_flat > 0)
        negative_diffs = np.sum(mean_diff_flat < 0)
        
        logger.info(f"\nDifference Direction (Faithful - Unfaithful):")
        logger.info(f"  - Features with POSITIVE difference: {positive_diffs} ({100*positive_diffs/len(mean_diff_flat):.1f}%)")
        logger.info(f"  - Features with NEGATIVE difference: {negative_diffs} ({100*negative_diffs/len(mean_diff_flat):.1f}%)")
        
        return {
            'mean_hallucinated': mean_hallucinated,
            'mean_non_hallucinated': mean_non_hallucinated,
            'mean_difference': mean_difference,
        }
        
    def save_results(self, output_dir: str, analysis_results: dict, layer: int = None):
        """Save analysis results to disk"""
        if layer is None:
            layer = self.layer
            
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"\nSaving results for Layer {layer} to {output_dir}")
        
        # Save mean activations
        means_data = {
            'layer': layer,
            'hook': self.hook,
            'scheme': self.scheme,
            'hallucinated_count': len(self.hallucinated_activations),
            'non_hallucinated_count': len(self.non_hallucinated_activations),
            'mean_hallucinated': analysis_results['mean_hallucinated'],
            'mean_non_hallucinated': analysis_results['mean_non_hallucinated'],
            'mean_difference': analysis_results['mean_difference'],
            'hallucinated_indices': self.hallucinated_indices,
            'non_hallucinated_indices': self.non_hallucinated_indices,
            'activation_shape': self.hallucinated_activations.shape if len(self.hallucinated_activations) > 0 else None,
        }
        
        # Save as pickle
        output_pkl = os.path.join(output_dir, f'layer{layer}_{self.hook}_{self.scheme}_means.pkl')
        with open(output_pkl, 'wb') as f:
            pickle.dump(means_data, f)
        logger.info(f"Saved to {output_pkl}")
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f'layer{layer}_{self.hook}_{self.scheme}_mean_hallucinated.npy'),
                analysis_results['mean_hallucinated'])
        np.save(os.path.join(output_dir, f'layer{layer}_{self.hook}_{self.scheme}_mean_non_hallucinated.npy'),
                analysis_results['mean_non_hallucinated'])
        np.save(os.path.join(output_dir, f'layer{layer}_{self.hook}_{self.scheme}_mean_difference.npy'),
                analysis_results['mean_difference'])

# ================================================================
# Main Function & CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build ITI intervention vectors from HDF5 activations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--h5-dir',
        type=str,
        required=True,
        help='Directory containing HDF5 activation files from grab_activation_ITI.py'
    )
    
    parser.add_argument(
        '--start-layer',
        type=int,
        default=None,
        help='Start layer index (default: auto-discover from H5 files)'
    )
    
    parser.add_argument(
        '--end-layer',
        type=int,
        default=None,
        help='End layer index (default: auto-discover from H5 files)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        nargs='+',
        default=[30],
        help='One or more K values to select for ITI intervention (e.g., --top-k 30 100 200)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./iti_results',
        help='Output directory to save ITI intervention configs'
    )
    
    args = parser.parse_args()
    
    # Ensure top_k is a list and sort it
    if isinstance(args.top_k, int):
        args.top_k = [args.top_k]
    args.top_k = sorted(list(set(args.top_k)))  # Remove duplicates and sort
    
    try:
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Set up file logging
        log_file = os.path.join(args.output_dir, 'steer_vector_calc_ITI.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
        
        # Initialize analyzer with HDF5 directory (use max K for initialization)
        analyzer = ActivationMeanAnalyzer(
            h5_dir=args.h5_dir,
            top_k=max(args.top_k),
            start_layer=args.start_layer,
            end_layer=args.end_layer
        )
        
        # Discover data from HDF5 files
        analyzer.discover_data()
        
        # Load all activations organized by (layer, head) for ITI
        analyzer.load_activations_for_iti()
        
        # Build ITI intervention vectors - Train classifiers ONCE
        iti_builder = analyzer.build_iti_intervention()
        
        # Generate and save validation accuracy heatmap
        iti_builder.save_accuracy_heatmap(args.output_dir)
        
        # Now build intervention vectors for each K value
        logger.info(f"\n{'='*80}")
        logger.info(f"GENERATING INTERVENTION CONFIGS FOR K VALUES: {args.top_k}")
        logger.info(f"{'='*80}")
        
        # Get directions (computed during training, need to recompute for consistency)
        directions = iti_builder.compute_directions(
            analyzer.activations_by_layer_head,
            analyzer.row_indices_by_layer_head,
            analyzer.labels_dict
        )
        
        # Generate insight plots (before building specific K configs)
        logger.info("\nGenerating comprehensive insight visualizations...")
        iti_builder.generate_insight_plots(
            args.output_dir,
            directions,
            analyzer.activations_by_layer_head,
            analyzer.row_indices_by_layer_head,
            analyzer.labels_dict,
            intervention_vectors=None  # Will be populated after first K build
        )
        
        results_summary = []
        
        # Generate pickle files for each K value
        for k in args.top_k:
            logger.info(f"\n{'='*80}")
            logger.info(f"Building and saving configuration for K={k}")
            logger.info(f"{'='*80}")
            
            # Build vectors for this K
            intervention_vectors = iti_builder.build_intervention_vectors_for_k(
                k,
                directions,
                analyzer.activations_by_layer_head,
                analyzer.row_indices_by_layer_head,
                analyzer.labels_dict
            )
            
            # Print summary for this K
            iti_builder.print_summary(intervention_vectors, k)
            
            # Generate STD vs accuracy plot for first K value
            if len(results_summary) == 0 and intervention_vectors:
                logger.info(f"Generating STD vs accuracy plot for K={k}...")
                try:
                    accuracies_list = []
                    stds_list = []
                    layers_list = []
                    for layer_key, layer_vecs in intervention_vectors.items():
                        layer_idx = int(layer_key.split('_')[1])
                        for head_info in layer_vecs:
                            accuracies_list.append(head_info['accuracy'])
                            stds_list.append(head_info['projection_std'])
                            layers_list.append(layer_idx)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(accuracies_list, stds_list, c=layers_list, 
                                        alpha=0.7, cmap='viridis', s=50)
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Layer Index', fontsize=11)
                    ax.set_xlabel('Validation Accuracy', fontsize=12)
                    ax.set_ylabel('Projection STD', fontsize=12)
                    ax.set_title(f'Accuracy vs. Projection STD (K={k} Selected Heads)', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    output_path = os.path.join(args.output_dir, f'accuracy_vs_std_scatter_k{k}.png')
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved accuracy vs STD scatter plot to {output_path}")
                except Exception as e:
                    logger.error(f"Error generating STD vs accuracy plot: {e}")
            
            # Check if K exceeds available heads
            total_heads = sum(len(v) for v in intervention_vectors.values())
            total_available_heads = len(iti_builder.validation_accuracies)
            
            if k > total_available_heads:
                logger.warning(f"SKIPPING K={k}: Requested K ({k}) exceeds total available heads ({total_available_heads})")
                logger.warning(f"  Only {total_available_heads} heads are available. Pickle file will NOT be saved.")
                results_summary.append({
                    'k': k,
                    'total_heads': total_heads,
                    'output_path': 'SKIPPED (K exceeds available heads)',
                    'skipped': True
                })
                continue
            
            # Save config for this K
            output_filename = f'iti_intervention_config_top{k}.pkl'
            iti_config_path = os.path.join(args.output_dir, output_filename)
            iti_builder.save_intervention_config(iti_config_path, intervention_vectors, k)
            
            results_summary.append({
                'k': k,
                'total_heads': total_heads,
                'output_path': iti_config_path,
                'skipped': False
            })
        
        # Print final summary
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY - ALL CONFIGURATIONS GENERATED")
        logger.info(f"{'='*80}")
        for result in results_summary:
            if result.get('skipped', False):
                logger.info(f"K={result['k']:4d}: {result['total_heads']:3d} heads -> SKIPPED (exceeds available heads)")
            else:
                logger.info(f"K={result['k']:4d}: {result['total_heads']:3d} heads -> {result['output_path']}")
        
        logger.info("\n" + "="*80)
        logger.info("ITI INTERVENTION BUILDING COMPLETE!")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"ITI building failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())