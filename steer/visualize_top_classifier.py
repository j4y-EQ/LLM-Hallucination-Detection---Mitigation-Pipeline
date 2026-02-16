"""
Visualize Top-1 ITI Classifier with Decision Boundary
======================================================
Creates comprehensive visualizations of the best-performing ITI head classifier:
1. 2D projection (PCA) of faithful vs unfaithful activations
2. Decision boundary from the logistic regression classifier
3. Direction vector (mean faithful - mean unfaithful)
4. Individual sample points colored by class

METHODOLOGY ALIGNMENT WITH steer_vector_calc_ITI.py:
----------------------------------------------------
This visualization script EXACTLY matches the ITI steering vector calculation:

1. Direction Computation (UNSCALED space):
   - direction = mean(faithful_activations) - mean(unfaithful_activations)
   - normalized_direction = direction / ||direction||
   - This matches steer_vector_calc_ITI.py lines 467-473

2. Classifier Training (SCALED space):
   - Activations are scaled with StandardScaler before training
   - LogisticRegression with balanced class weights
   - This matches steer_vector_calc_ITI.py _train_single_head() function

3. Visualization Approach:
   - Direction computed in UNSCALED space (raw activations)
   - Also computed in SCALED space for proper PCA projection
   - PCA is fit on SCALED activations (same as classifier sees)
   - Direction vector projected through same PCA for visualization

The graphs show:
- How well the classifier separates faithful from unfaithful activations
- The geometric relationship between the decision boundary and steering direction
- Whether the ITI intervention direction aligns with the classifier decision
- Diagnostic metrics (ROC, PR curves) to verify no overfitting

Usage:
    # Visualize with model preset (auto h5_dir)
    python -m pipeline.steer.ITI.visualize_top_classifier --model llama --output-dir ./data/visualizations/llama/worst
    python -m pipeline.steer.ITI.visualize_top_classifier --model gemma --output-dir ./data/visualizations/gemma/worst
    python -m pipeline.steer.ITI.visualize_top_classifier --model qwen --output-dir ./data/visualizations/qwen/worst

    
    # Force auto-discovery of best head
    python -m pipeline.steer.ITI.visualize_top_classifier --model llama --all --output-dir ./visualizations/
    
    # Manual h5_dir specification
    python -m pipeline.steer.ITI.visualize_top_classifier --h5-dir ./data/ITI/activations/qwen_2.5_7b_instruct --output-dir ./data/visualizations/
"""

import os
import sys
import argparse
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm
import glob
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualize_top_classifier')

# Model-specific layer/head configurations (hardcoded) BEST
# MODEL_CONFIGS = {
#     'gemma': {'layer': 22, 'head': 2, 'h5_dir': './data/ITI/activations/gemma'},
#     'llama': {'layer': 12, 'head': 28, 'h5_dir': './data/ITI/activations/llama'},
#     'qwen': {'layer': 18, 'head': 13, 'h5_dir': './data/ITI/activations/qwen_2.5_7b_instruct'},
# }

# ONE OF THE WORST
MODEL_CONFIGS = {
    'gemma': {'layer': 1, 'head': 1, 'h5_dir': './data/ITI/activations/gemma'},
    'llama': {'layer': 1, 'head': 1, 'h5_dir': './data/ITI/activations/llama'},
    'qwen': {'layer': 1, 'head': 1, 'h5_dir': './data/ITI/activations/qwen_2.5_7b_instruct'},
}


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
                            
                            # Extract metadata from metadata subgroup
                            if 'metadata' not in sample_group:
                                continue
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


def load_activations_for_head(h5_files, target_layer, target_head, labels_dict):
    """
    Load all activations for a specific (layer, head) pair
    
    Args:
        h5_files: List of HDF5 file paths
        target_layer: Layer index to load
        target_head: Head index to load
        labels_dict: Dict mapping row_idx -> label
        
    Returns:
        activations: (n_samples, 128) array
        labels: (n_samples,) array
        row_indices: (n_samples,) array
    """
    activations_list = []
    labels_list = []
    row_indices_list = []
    
    logger.info(f"Loading activations for Layer {target_layer}, Head {target_head}...")
    
    for h5_file in tqdm(h5_files, desc="Loading activations"):
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
                        
                        # Skip if no label
                        if row_idx == -1 or row_idx not in labels_dict:
                            continue
                        
                        # Load layer activation
                        if 'activations_last_gen_token' not in sample_group:
                            continue
                        acts_group = sample_group['activations_last_gen_token']
                        
                        layer_key = f'layer_{target_layer}'
                        if layer_key not in acts_group:
                            continue
                        
                        layer_activation = acts_group[layer_key][:]  # Shape: [32, 128]
                        
                        # Extract specific head
                        head_activation = layer_activation[target_head]  # Shape: [128]
                        
                        activations_list.append(head_activation)
                        labels_list.append(labels_dict[row_idx])
                        row_indices_list.append(row_idx)
                    except Exception as e:
                        logger.debug(f"Error processing {sample_group_name}: {e}")
                        continue
                    
        except Exception as e:
            logger.warning(f"Error reading {h5_file}: {e}")
            continue
    
    if not activations_list:
        logger.error(f"No activations found for Layer {target_layer}, Head {target_head}")
        return None, None, None
    
    activations = np.array(activations_list)
    labels = np.array(labels_list)
    row_indices = np.array(row_indices_list)
    
    logger.info(f"Loaded {len(activations)} samples for Layer {target_layer}, Head {target_head}")
    logger.info(f"  - Faithful (0): {np.sum(labels == 0)}")
    logger.info(f"  - Unfaithful (1): {np.sum(labels == 1)}")
    
    return activations, labels, row_indices


def train_classifier_with_split(activations, labels, val_split=0.2, seed=42):
    """
    Train a logistic regression classifier with train/val split
    
    Args:
        activations: (n_samples, 128) array
        labels: (n_samples,) array
        val_split: Validation split ratio
        seed: Random seed
        
    Returns:
        dict with classifier, scaler, and split data
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        activations, labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels
    )
    
    # Scale activations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train classifier
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=seed
    )
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = classifier.score(X_train_scaled, y_train)
    val_acc = classifier.score(X_val_scaled, y_val)
    
    logger.info(f"Classifier trained:")
    logger.info(f"  - Train accuracy: {train_acc:.4f}")
    logger.info(f"  - Val accuracy:   {val_acc:.4f}")
    
    return {
        'classifier': classifier,
        'scaler': scaler,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'train_acc': train_acc,
        'val_acc': val_acc
    }


def compute_direction_vector(activations, labels, scaler=None):
    """
    Compute center-of-mass direction: mean(faithful) - mean(unfaithful)
    This matches the methodology in steer_vector_calc_ITI.py
    
    Args:
        activations: (n_samples, 128) array (UNSCALED)
        labels: (n_samples,) array
        scaler: Optional StandardScaler to compute scaled version
        
    Returns:
        direction_normalized: (128,) normalized direction vector (UNSCALED space) - THETA
        mean_faithful: (128,) mean of faithful activations (UNSCALED)
        mean_unfaithful: (128,) mean of unfaithful activations (UNSCALED)
        direction_normalized_scaled: (128,) direction in SCALED space (if scaler provided)
        projection_std: float - SIGMA (std of projections)
        steering_vector: (128,) actual steering vector = theta * sigma (UNSCALED space)
    """
    faithful_mask = labels == 0
    unfaithful_mask = labels == 1
    
    # Compute means in UNSCALED space (matches steer_vector_calc_ITI.py)
    mean_faithful = np.mean(activations[faithful_mask], axis=0)
    mean_unfaithful = np.mean(activations[unfaithful_mask], axis=0)
    
    # Direction: faithful - unfaithful (UNSCALED space)
    direction = mean_faithful - mean_unfaithful
    
    # Normalize in UNSCALED space (THETA)
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm
    
    logger.info(f"Direction vector computed (UNSCALED space):")
    logger.info(f"  - Direction norm (before normalization): {direction_norm:.6f}")
    logger.info(f"  - Direction norm (after normalization):  {np.linalg.norm(direction_normalized):.6f}")
    
    # Compute projection std (SIGMA) - matches steer_vector_calc_ITI.py
    projections = activations @ direction_normalized
    projection_std = np.std(projections)
    logger.info(f"  - Projection std (SIGMA): {projection_std:.6f}")
    
    # Compute actual steering vector: theta * sigma (matches steering_experiment.py antidote calculation)
    steering_vector = direction_normalized * projection_std
    logger.info(f"  - Steering vector norm (theta * sigma): {np.linalg.norm(steering_vector):.6f}")
    
    # If scaler provided, compute direction in SCALED space
    direction_normalized_scaled = None
    steering_vector_scaled = None
    if scaler is not None:
        # Transform activations to scaled space
        activations_scaled = scaler.transform(activations)
        mean_faithful_scaled = np.mean(activations_scaled[faithful_mask], axis=0)
        mean_unfaithful_scaled = np.mean(activations_scaled[unfaithful_mask], axis=0)
        
        # Direction in scaled space
        direction_scaled = mean_faithful_scaled - mean_unfaithful_scaled
        direction_scaled_norm = np.linalg.norm(direction_scaled)
        direction_normalized_scaled = direction_scaled / direction_scaled_norm
        
        # Projection std in scaled space
        projections_scaled = activations_scaled @ direction_normalized_scaled
        projection_std_scaled = np.std(projections_scaled)
        
        # Steering vector in scaled space
        steering_vector_scaled = direction_normalized_scaled * projection_std_scaled
        
        logger.info(f"Direction vector computed (SCALED space):")
        logger.info(f"  - Direction norm (before normalization): {direction_scaled_norm:.6f}")
        logger.info(f"  - Direction norm (after normalization):  {np.linalg.norm(direction_normalized_scaled):.6f}")
        logger.info(f"  - Projection std (SIGMA): {projection_std_scaled:.6f}")
        logger.info(f"  - Steering vector norm (theta * sigma): {np.linalg.norm(steering_vector_scaled):.6f}")
    
    return direction_normalized, mean_faithful, mean_unfaithful, direction_normalized_scaled, projection_std, steering_vector, steering_vector_scaled


def visualize_classifier_2d(activations, labels, classifier_result, direction_info, 
                            layer, head, output_path, use_validation_only=False):
    """
    Create 2D visualizations: decision boundary and steering vector as SEPARATE images
    
    Args:
        activations: (n_samples, 128) full dataset
        labels: (n_samples,) full labels
        classifier_result: Dict from train_classifier_with_split()
        direction_info: Tuple (direction_normalized, mean_faithful, mean_unfaithful, direction_normalized_scaled, projection_std, steering_vector, steering_vector_scaled)
        layer: Layer index
        head: Head index
        output_path: Path to save figure (will create 2 files: _boundary.png and _steering.png)
        use_validation_only: If True, only plot validation set
    """
    classifier = classifier_result['classifier']
    scaler = classifier_result['scaler']
    direction_normalized, mean_faithful, mean_unfaithful, direction_normalized_scaled, projection_std, steering_vector, steering_vector_scaled = direction_info
    
    # Determine which data to plot
    if use_validation_only:
        X_plot = classifier_result['X_val']
        y_plot = classifier_result['y_val']
        title_suffix = " (Validation Set)"
    else:
        X_plot = activations
        y_plot = labels
        title_suffix = " (Full Dataset)"
    
    # Scale data
    X_plot_scaled = scaler.transform(X_plot)
    
    # Apply PCA to reduce to 2D
    logger.info("Applying PCA for 2D projection...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_plot_scaled)
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
    logger.info(f"  - PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    logger.info(f"  - PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    
    # Project actual steering vector (theta * sigma) to 2D
    steering_vector_2d = pca.transform(steering_vector_scaled.reshape(1, -1)).flatten()
    
    logger.info(f"Steering vector in 2D PCA space:")
    logger.info(f"  - Steering vector 2D norm (theta * sigma): {np.linalg.norm(steering_vector_2d):.6f}")
    
    # Project means to 2D
    mean_faithful_scaled = scaler.transform(mean_faithful.reshape(1, -1))
    mean_unfaithful_scaled = scaler.transform(mean_unfaithful.reshape(1, -1))
    mean_faithful_2d = pca.transform(mean_faithful_scaled).flatten()
    mean_unfaithful_2d = pca.transform(mean_unfaithful_scaled).flatten()
    
    # Get axis limits
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    
    # Prepare output paths
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)
    output_name_no_ext = os.path.splitext(output_basename)[0]
    
    # ============================================================
    # IMAGE 1: Decision Boundary
    # ============================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create mesh for decision boundary
    h = 0.02  # Step size in mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Project mesh points back to 128D, then get classifier predictions
    mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_128d = pca.inverse_transform(mesh_points_2d)
    
    # Predict on mesh
    Z = classifier.predict(mesh_points_128d)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary (contour)
    ax1.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['#ffcccc', '#ccccff'])
    ax1.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])
    
    # Plot data points
    faithful_mask = y_plot == 0
    unfaithful_mask = y_plot == 1
    
    ax1.scatter(X_2d[faithful_mask, 0], X_2d[faithful_mask, 1], 
               c='blue', label='Faithful', alpha=0.7, s=80, edgecolors='k', linewidth=0.8)
    ax1.scatter(X_2d[unfaithful_mask, 0], X_2d[unfaithful_mask, 1], 
               c='red', label='Unfaithful', alpha=0.7, s=80, edgecolors='k', linewidth=0.8)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=16, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=16, fontweight='bold')
    ax1.set_title(f'Classifier Decision Boundary\nLayer {layer}, Head {head}{title_suffix}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=14, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    # Save decision boundary figure
    os.makedirs(output_dir, exist_ok=True)
    output_path_boundary = os.path.join(output_dir, f'{output_name_no_ext}_boundary.png')
    fig1.savefig(output_path_boundary, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    
    logger.info(f"Saved decision boundary to {output_path_boundary}")
    
    # ============================================================
    # IMAGE 2: Steering Vector (ONLY)
    # ============================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot data points
    ax2.scatter(X_2d[faithful_mask, 0], X_2d[faithful_mask, 1], 
               c='blue', label='Faithful', alpha=0.7, s=80, edgecolors='k', linewidth=0.8)
    ax2.scatter(X_2d[unfaithful_mask, 0], X_2d[unfaithful_mask, 1], 
               c='red', label='Unfaithful', alpha=0.7, s=80, edgecolors='k', linewidth=0.8)
    
    # Plot ONLY the STEERING VECTOR (theta * sigma) from origin
    scale_for_steering = np.max([x_max - x_min, y_max - y_min]) * 0.4
    origin = np.array([0, 0])
    # Normalize steering vector for visualization, then scale for visibility
    steering_vector_2d_normalized = steering_vector_2d / np.linalg.norm(steering_vector_2d)
    steering_end = origin + steering_vector_2d_normalized * scale_for_steering
    
    arrow_steering = FancyArrowPatch(
        origin, steering_end,
        arrowstyle='->', mutation_scale=60, linewidth=8, linestyle='-',
        color='magenta', label='Steering Vector (θ × σ)', zorder=12,
        edgecolor='purple', facecolor='magenta'
    )
    ax2.add_patch(arrow_steering)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=16, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_[1]*100:.1f}% variance)', fontsize=16, fontweight='bold')
    ax2.set_title(f'Steering Vector (ITI Intervention)\nLayer {layer}, Head {head}{title_suffix}', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Position legend to avoid overlap with steering vector
    # Place it in upper left if steering vector points right/down, otherwise adjust
    if steering_end[0] > 0:  # Pointing right
        legend_loc = 'upper left'
    else:  # Pointing left
        legend_loc = 'upper right'
    
    ax2.legend(loc=legend_loc, fontsize=14, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    # Save steering vector figure
    output_path_steering = os.path.join(output_dir, f'{output_name_no_ext}_steering.png')
    fig2.savefig(output_path_steering, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    
    logger.info(f"Saved steering vector to {output_path_steering}")
    
    return pca


def visualize_classifier_with_projections(activations, labels, classifier_result, 
                                         direction_info, layer, head, output_path):
    """
    Create 3 separate 1D projection histograms showing classifier decision boundary
    
    Args:
        activations: (n_samples, 128) array
        labels: (n_samples,) array
        classifier_result: Dict from train_classifier_with_split()
        direction_info: Tuple (direction_normalized, mean_faithful, mean_unfaithful, direction_normalized_scaled, projection_std, steering_vector, steering_vector_scaled)
        layer: Layer index
        head: Head index
        output_path: Path to save figures (will create 3 files: _direction.png, _steering.png, _classifier.png)
    """
    classifier = classifier_result['classifier']
    scaler = classifier_result['scaler']
    direction_normalized, mean_faithful, mean_unfaithful, direction_normalized_scaled, projection_std, steering_vector, steering_vector_scaled = direction_info
    
    # Scale activations
    X_scaled = scaler.transform(activations)
    
    # Project onto direction vector in scaled space
    # Use the direction computed in SCALED space
    projections = X_scaled @ direction_normalized_scaled
    
    # Project onto actual steering vector (theta * sigma) in scaled space
    steering_projections = X_scaled @ steering_vector_scaled
    
    # Get classifier coefficients (these define the decision boundary)
    # For logistic regression: decision_function = X @ coef + intercept
    classifier_coef = classifier.coef_[0]  # Shape: (128,)
    classifier_intercept = classifier.intercept_[0]
    
    # Project onto classifier normal vector
    classifier_projections = X_scaled @ classifier_coef
    
    # Prepare output paths
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)
    output_name_no_ext = os.path.splitext(output_basename)[0]
    
    faithful_mask = labels == 0
    unfaithful_mask = labels == 1
    bins = 50
    
    # ============================================================
    # IMAGE 1: Projection onto direction vector (theta)
    # ============================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    
    faithful_proj = projections[faithful_mask]
    unfaithful_proj = projections[unfaithful_mask]
    
    # Histogram
    ax1.hist(faithful_proj, bins=bins, alpha=0.6, label='Faithful (0)', color='blue', edgecolor='black')
    ax1.hist(unfaithful_proj, bins=bins, alpha=0.6, label='Unfaithful (1)', color='red', edgecolor='black')
    
    # Mark means
    ax1.axvline(np.mean(faithful_proj), color='darkblue', linestyle='--', linewidth=3, label='Mean Faithful')
    ax1.axvline(np.mean(unfaithful_proj), color='darkred', linestyle='--', linewidth=3, label='Mean Unfaithful')
    
    ax1.set_xlabel('Projection onto Direction Vector (θ)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax1.set_title(f'Activation Projections onto Direction Vector (θ)\n'
                 f'Layer {layer}, Head {head}', fontsize=18, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=14, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path_direction = os.path.join(output_dir, f'{output_name_no_ext}_direction.png')
    fig1.savefig(output_path_direction, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    
    logger.info(f"Saved direction projection to {output_path_direction}")
    
    # ============================================================
    # IMAGE 2: Projection onto STEERING vector (theta * sigma)
    # ============================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
    
    faithful_steering_proj = steering_projections[faithful_mask]
    unfaithful_steering_proj = steering_projections[unfaithful_mask]
    
    # Histogram
    ax2.hist(faithful_steering_proj, bins=bins, alpha=0.6, label='Faithful (0)', color='blue', edgecolor='black')
    ax2.hist(unfaithful_steering_proj, bins=bins, alpha=0.6, label='Unfaithful (1)', color='red', edgecolor='black')
    
    # Mark means
    ax2.axvline(np.mean(faithful_steering_proj), color='darkblue', linestyle='--', linewidth=3, label='Mean Faithful')
    ax2.axvline(np.mean(unfaithful_steering_proj), color='darkred', linestyle='--', linewidth=3, label='Mean Unfaithful')
    
    # Add sigma annotation
    ax2.text(0.98, 0.98, f'σ (projection std) = {projection_std:.4f}',
            transform=ax2.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontweight='bold')
    
    ax2.set_xlabel('Projection onto STEERING Vector (θ × σ)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax2.set_title(f'Activation Projections onto STEERING Vector (θ × σ)\n'
                 f'This is the vector actually used in experiments (before multiplying by α)\n'
                 f'Layer {layer}, Head {head}', fontsize=18, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=14, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    output_path_steering = os.path.join(output_dir, f'{output_name_no_ext}_steering.png')
    fig2.savefig(output_path_steering, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    
    logger.info(f"Saved steering projection to {output_path_steering}")
    
    # ============================================================
    # IMAGE 3: Projection onto classifier normal vector
    # ============================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 6))
    
    faithful_clf_proj = classifier_projections[faithful_mask]
    unfaithful_clf_proj = classifier_projections[unfaithful_mask]
    
    # Histogram
    ax3.hist(faithful_clf_proj, bins=bins, alpha=0.6, label='Faithful (0)', color='blue', edgecolor='black')
    ax3.hist(unfaithful_clf_proj, bins=bins, alpha=0.6, label='Unfaithful (1)', color='red', edgecolor='black')
    
    # Decision boundary (where decision_function = 0)
    # decision_function = X @ coef + intercept = 0
    # => projection onto coef = -intercept
    decision_threshold = -classifier_intercept
    ax3.axvline(decision_threshold, color='black', linestyle='-', linewidth=5, label='Decision Boundary')
    
    ax3.set_xlabel('Projection onto Classifier Normal Vector (w)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax3.set_title(f'Activation Projections onto Classifier Normal Vector\n'
                 f'Layer {layer}, Head {head}', fontsize=18, fontweight='bold', pad=20)
    ax3.legend(loc='best', fontsize=14, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=14)
    
    # Add text with classifier accuracy
    train_acc = classifier_result['train_acc']
    val_acc = classifier_result['val_acc']
    ax3.text(0.02, 0.98, f'Train Acc: {train_acc:.4f}\nVal Acc: {val_acc:.4f}',
            transform=ax3.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontweight='bold')
    
    plt.tight_layout()
    
    output_path_classifier = os.path.join(output_dir, f'{output_name_no_ext}_classifier.png')
    fig3.savefig(output_path_classifier, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    
    logger.info(f"Saved classifier projection to {output_path_classifier}")


def visualize_classifier_diagnostics(activations, labels, classifier_result, layer, head, output_dir):
    """
    Create comprehensive diagnostic plots to show classifier is not overfitting
    Creates 4 SEPARATE plots: ROC curves, Precision-Recall curves, Confusion matrices, and Calibration plot
    
    Args:
        activations: (n_samples, 128) array
        labels: (n_samples,) array
        classifier_result: Dict from train_classifier_with_split()
        layer: Layer index
        head: Head index
        output_dir: Directory to save diagnostic figures (will create 4 separate files)
    
    Returns:
        dict with diagnostic metrics
    """
    classifier = classifier_result['classifier']
    scaler = classifier_result['scaler']
    X_train = classifier_result['X_train']
    y_train = classifier_result['y_train']
    X_val = classifier_result['X_val']
    y_val = classifier_result['y_val']
    
    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Get predictions and probabilities
    y_train_pred = classifier.predict(X_train_scaled)
    y_val_pred = classifier.predict(X_val_scaled)
    y_train_proba = classifier.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = classifier.predict_proba(X_val_scaled)[:, 1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # Plot 1: ROC Curves (Train vs Val) - SEPARATE FILE
    # ============================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 9))
    
    # Compute ROC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    # Compute ROC for validation set
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    roc_auc_val = auc(fpr_val, tpr_val)
    
    # Plot ROC curves
    ax1.plot(fpr_train, tpr_train, linewidth=3, label=f'Train (AUC = {roc_auc_train:.3f})', color='blue')
    ax1.plot(fpr_val, tpr_val, linewidth=3, label=f'Validation (AUC = {roc_auc_val:.3f})', color='red')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    ax1.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax1.set_title(f'ROC Curves (Train vs Validation)\\nLayer {layer}, Head {head}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax1.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)
    
    # Add text annotation
    diff = abs(roc_auc_train - roc_auc_val)
    overfitting_status = "✓ No Overfitting" if diff < 0.05 else "⚠ Possible Overfitting"
    ax1.text(0.5, 0.15, f'AUC Difference: {diff:.3f}\\n{overfitting_status}',
            transform=ax1.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if diff < 0.05 else 'lightyellow', alpha=0.9),
            fontweight='bold')
    
    plt.tight_layout()
    output_path_roc = os.path.join(output_dir, f'diagnostic_roc_layer{layer}_head{head}.png')
    fig1.savefig(output_path_roc, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"Saved ROC curve to {output_path_roc}")
    
    # ============================================================
    # Plot 2: Precision-Recall Curves (Train vs Val) - SEPARATE FILE
    # ============================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 9))
    
    # Compute PR curves
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_proba)
    
    # Compute average precision
    from sklearn.metrics import average_precision_score
    ap_train = average_precision_score(y_train, y_train_proba)
    ap_val = average_precision_score(y_val, y_val_proba)
    
    # Plot PR curves
    ax2.plot(recall_train, precision_train, linewidth=3, label=f'Train (AP = {ap_train:.3f})', color='blue')
    ax2.plot(recall_val, precision_val, linewidth=3, label=f'Validation (AP = {ap_val:.3f})', color='red')
    
    # Add baseline
    baseline = np.sum(y_val) / len(y_val)
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=2, label=f'Baseline ({baseline:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=16, fontweight='bold')
    ax2.set_title(f'Precision-Recall Curves\\nLayer {layer}, Head {head}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=14, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path_pr = os.path.join(output_dir, f'diagnostic_pr_layer{layer}_head{head}.png')
    fig2.savefig(output_path_pr, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved Precision-Recall curve to {output_path_pr}")
    
    # ============================================================
    # Plot 3: Confusion Matrices (Train vs Val) - SEPARATE FILE
    # ============================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 9))
    
    # Compute confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    
    # Create combined visualization
    x_positions = [0, 1, 3, 4]
    cm_combined = np.zeros((2, 4))
    cm_combined[:, :2] = cm_train
    cm_combined[:, 2:] = cm_val
    
    # Plot heatmap
    im = ax3.imshow(cm_combined, interpolation='nearest', cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(4):
            text = ax3.text(j, i, int(cm_combined[i, j]),
                           ha="center", va="center", color="white" if cm_combined[i, j] > cm_combined.max()/2 else "black",
                           fontsize=20, fontweight='bold')
    
    # Customize ticks
    ax3.set_xticks([0.5, 3.5])
    ax3.set_xticklabels(['Training', 'Validation'], fontsize=16, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Faithful (0)', 'Unfaithful (1)'], fontsize=14)
    ax3.set_title(f'Confusion Matrices (Train vs Validation)\\nLayer {layer}, Head {head}',
                 fontsize=18, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.ax.tick_params(labelsize=12)
    
    # Add grid lines
    ax3.set_xticks([1.5], minor=True)
    ax3.grid(which='minor', color='red', linestyle='-', linewidth=3)
    
    # Add accuracy annotations
    train_acc = classifier_result['train_acc']
    val_acc = classifier_result['val_acc']
    ax3.text(0.5, -0.15, f'Train Acc: {train_acc:.3f}', ha='center', 
            transform=ax3.transAxes, fontsize=14, fontweight='bold')
    ax3.text(0.5, -0.20, f'Val Acc: {val_acc:.3f}', ha='center',
            transform=ax3.transAxes, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path_cm = os.path.join(output_dir, f'diagnostic_confusion_layer{layer}_head{head}.png')
    fig3.savefig(output_path_cm, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    logger.info(f"Saved Confusion matrix to {output_path_cm}")
    
    # ============================================================
    # Plot 4: Calibration Curve - SEPARATE FILE
    # ============================================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 9))
    
    # Compute calibration curves
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_proba, n_bins=10, strategy='uniform')
    prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_proba, n_bins=10, strategy='uniform')
    
    # Plot calibration curves
    ax4.plot(prob_pred_train, prob_true_train, 'o-', linewidth=3, markersize=10, 
            label='Training', color='blue')
    ax4.plot(prob_pred_val, prob_true_val, 's-', linewidth=3, markersize=10, 
            label='Validation', color='red')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    ax4.set_xlabel('Predicted Probability', fontsize=16, fontweight='bold')
    ax4.set_ylabel('True Probability', fontsize=16, fontweight='bold')
    ax4.set_title(f'Calibration Curves\\nLayer {layer}, Head {head}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax4.legend(loc='best', fontsize=14, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=14)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    # Add explanation text
    ax4.text(0.05, 0.95, 'Well-calibrated classifiers\\nlie close to diagonal',
            transform=ax4.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path_cal = os.path.join(output_dir, f'diagnostic_calibration_layer{layer}_head{head}.png')
    fig4.savefig(output_path_cal, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    logger.info(f"Saved Calibration curve to {output_path_cal}")
    
    # Print classification report
    logger.info(f"\n{'='*80}")
    logger.info(f"CLASSIFICATION METRICS - Layer {layer}, Head {head}")
    logger.info(f"{'='*80}")
    logger.info(f"\nValidation Set Classification Report:")
    logger.info(f"\n{classification_report(y_val, y_val_pred, target_names=['Faithful', 'Unfaithful'])}")
    logger.info(f"\nROC AUC Scores:")
    logger.info(f"  Training:   {roc_auc_train:.4f}")
    logger.info(f"  Validation: {roc_auc_val:.4f}")
    logger.info(f"  Difference: {abs(roc_auc_train - roc_auc_val):.4f}")
    logger.info(f"\nAverage Precision Scores:")
    logger.info(f"  Training:   {ap_train:.4f}")
    logger.info(f"  Validation: {ap_val:.4f}")
    logger.info(f"  Difference: {abs(ap_train - ap_val):.4f}")
    
    return {
        'roc_auc_train': roc_auc_train,
        'roc_auc_val': roc_auc_val,
        'ap_train': ap_train,
        'ap_val': ap_val
    }


def find_top_k_heads_from_config(iti_config_path, k=1):
    
    # Compute ROC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    # Compute ROC for validation set
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    roc_auc_val = auc(fpr_val, tpr_val)
    
    # Plot ROC curves
    ax1.plot(fpr_train, tpr_train, linewidth=3, label=f'Train (AUC = {roc_auc_train:.3f})', color='blue')
    ax1.plot(fpr_val, tpr_val, linewidth=3, label=f'Validation (AUC = {roc_auc_val:.3f})', color='red')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    ax1.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax1.set_title(f'ROC Curves (Train vs Validation)\\nLayer {layer}, Head {head}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax1.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)
    
    # Add text annotation
    diff = abs(roc_auc_train - roc_auc_val)
    overfitting_status = "✓ No Overfitting" if diff < 0.05 else "⚠ Possible Overfitting"
    ax1.text(0.5, 0.15, f'AUC Difference: {diff:.3f}\\n{overfitting_status}',
            transform=ax1.transAxes, fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if diff < 0.05 else 'lightyellow', alpha=0.9),
            fontweight='bold')
    
    # ============================================================
    # Plot 2: Precision-Recall Curves (Train vs Val)
    # ============================================================
    ax2 = axes[0, 1]
    
    # Compute PR curves
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_proba)
    
    # Compute average precision
    from sklearn.metrics import average_precision_score
    ap_train = average_precision_score(y_train, y_train_proba)
    ap_val = average_precision_score(y_val, y_val_proba)
    
    # Plot PR curves
    ax2.plot(recall_train, precision_train, linewidth=3, label=f'Train (AP = {ap_train:.3f})', color='blue')
    ax2.plot(recall_val, precision_val, linewidth=3, label=f'Validation (AP = {ap_val:.3f})', color='red')
    
    # Add baseline
    baseline = np.sum(y_val) / len(y_val)
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=2, label=f'Baseline ({baseline:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=16, fontweight='bold')
    ax2.set_title(f'Precision-Recall Curves\\nLayer {layer}, Head {head}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=14, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # ============================================================
    # Plot 3: Confusion Matrices (Train vs Val)
    # ============================================================
    ax3 = axes[1, 0]
    
    # Compute confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    
    # Create combined visualization
    x_positions = [0, 1, 3, 4]
    cm_combined = np.zeros((2, 4))
    cm_combined[:, :2] = cm_train
    cm_combined[:, 2:] = cm_val
    
    # Plot heatmap
    im = ax3.imshow(cm_combined, interpolation='nearest', cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(4):
            text = ax3.text(j, i, int(cm_combined[i, j]),
                           ha="center", va="center", color="white" if cm_combined[i, j] > cm_combined.max()/2 else "black",
                           fontsize=20, fontweight='bold')
    
    # Customize ticks
    ax3.set_xticks([0.5, 3.5])
    ax3.set_xticklabels(['Training', 'Validation'], fontsize=16, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Faithful (0)', 'Unfaithful (1)'], fontsize=14)
    ax3.set_title(f'Confusion Matrices (Train vs Validation)\\nLayer {layer}, Head {head}',
                 fontsize=18, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.ax.tick_params(labelsize=12)
    
    # Add grid lines
    ax3.set_xticks([1.5], minor=True)
    ax3.grid(which='minor', color='red', linestyle='-', linewidth=3)
    
    # Add accuracy annotations
    train_acc = classifier_result['train_acc']
    val_acc = classifier_result['val_acc']
    ax3.text(0.5, -0.15, f'Train Acc: {train_acc:.3f}', ha='center', 
            transform=ax3.transAxes, fontsize=14, fontweight='bold')
    ax3.text(0.5, -0.20, f'Val Acc: {val_acc:.3f}', ha='center',
            transform=ax3.transAxes, fontsize=14, fontweight='bold')
    
    # ============================================================
    # Plot 4: Calibration Curve
    # ============================================================
    ax4 = axes[1, 1]
    
    # Compute calibration curves
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_proba, n_bins=10, strategy='uniform')
    prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_proba, n_bins=10, strategy='uniform')
    
    # Plot calibration curves
    ax4.plot(prob_pred_train, prob_true_train, 'o-', linewidth=3, markersize=10, 
            label='Training', color='blue')
    ax4.plot(prob_pred_val, prob_true_val, 's-', linewidth=3, markersize=10, 
            label='Validation', color='red')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    ax4.set_xlabel('Predicted Probability', fontsize=16, fontweight='bold')
    ax4.set_ylabel('True Probability', fontsize=16, fontweight='bold')
    ax4.set_title(f'Calibration Curves\\nLayer {layer}, Head {head}', 
                 fontsize=18, fontweight='bold', pad=20)
    ax4.legend(loc='best', fontsize=14, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=14)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    # Add explanation text
    ax4.text(0.05, 0.95, 'Well-calibrated classifiers\\nlie close to diagonal',
            transform=ax4.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved diagnostic visualization to {output_path}")
    
    # Print classification report
    logger.info(f"\n{'='*80}")
    logger.info(f"CLASSIFICATION METRICS - Layer {layer}, Head {head}")
    logger.info(f"{'='*80}")
    logger.info(f"\nValidation Set Classification Report:")
    logger.info(f"\n{classification_report(y_val, y_val_pred, target_names=['Faithful', 'Unfaithful'])}")
    logger.info(f"\nROC AUC Scores:")
    logger.info(f"  Training:   {roc_auc_train:.4f}")
    logger.info(f"  Validation: {roc_auc_val:.4f}")
    logger.info(f"  Difference: {abs(roc_auc_train - roc_auc_val):.4f}")
    logger.info(f"\nAverage Precision Scores:")
    logger.info(f"  Training:   {ap_train:.4f}")
    logger.info(f"  Validation: {ap_val:.4f}")
    logger.info(f"  Difference: {abs(ap_train - ap_val):.4f}")
    
    return {
        'roc_auc_train': roc_auc_train,
        'roc_auc_val': roc_auc_val,
        'ap_train': ap_train,
        'ap_val': ap_val
    }


def find_top_k_heads_from_config(iti_config_path, k=1):
    """
    Load ITI config and extract top-K heads
    
    Args:
        iti_config_path: Path to ITI config pickle file
        k: Rank to extract (1 = best head)
        
    Returns:
        List of tuples: [(layer, head, accuracy), ...]
    """
    if not os.path.exists(iti_config_path):
        logger.error(f"ITI config file not found: {iti_config_path}")
        return None
    
    with open(iti_config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Extract intervention vectors
    intervention_vectors = config['intervention_vectors']
    
    # Collect all heads with their accuracies
    all_heads = []
    for layer_key, heads in intervention_vectors.items():
        layer = int(layer_key.split('_')[1])
        for head_info in heads:
            all_heads.append((layer, head_info['head'], head_info['accuracy']))
    
    # Sort by accuracy (descending)
    all_heads.sort(key=lambda x: x[2], reverse=True)
    
    return all_heads[:k]


def auto_discover_best_head(h5_files, labels_dict, max_layers=None, max_heads_per_layer=None):
    """
    Automatically discover the best performing head by training classifiers on ALL combinations
    
    Args:
        h5_files: List of HDF5 files
        labels_dict: Dict mapping row_idx -> label
        max_layers: Ignored - tests ALL layers
        max_heads_per_layer: Ignored - tests ALL heads
        
    Returns:
        tuple: (best_layer, best_head, best_accuracy)
    """
    logger.info("Auto-discovering best head by testing ALL layer-head combinations (K=1)...")
    
    # First, discover available layers by sampling the H5 files
    available_layers = set()
    with h5py.File(h5_files[0], 'r') as f:
        first_sample = None
        for key in f.keys():
            if key.startswith('sample_'):
                first_sample = key
                break
        
        if first_sample and 'activations_last_gen_token' in f[first_sample]:
            acts_group = f[first_sample]['activations_last_gen_token']
            for layer_key in acts_group.keys():
                if layer_key.startswith('layer_'):
                    layer_idx = int(layer_key.split('_')[1])
                    available_layers.add(layer_idx)
    
    available_layers = sorted(list(available_layers))
    logger.info(f"Found {len(available_layers)} layers: {available_layers}")
    
    # Test ALL layers
    sampled_layers = available_layers
    logger.info(f"Testing ALL {len(sampled_layers)} layers")
    
    # Get number of heads from first layer
    with h5py.File(h5_files[0], 'r') as f:
        for sample_key in f.keys():
            if sample_key.startswith('sample_'):
                sample_group = f[sample_key]
                if 'activations_last_gen_token' in sample_group:
                    acts_group = sample_group['activations_last_gen_token']
                    first_layer_key = f'layer_{sampled_layers[0]}'
                    if first_layer_key in acts_group:
                        layer_acts = acts_group[first_layer_key][:]
                        n_heads = layer_acts.shape[0]
                        logger.info(f"Detected {n_heads} heads per layer")
                        break
                break
    
    # Test ALL heads
    sampled_heads = list(range(n_heads))
    logger.info(f"Testing ALL {len(sampled_heads)} heads per layer")
    logger.info(f"TOTAL COMBINATIONS TO TEST: {len(sampled_layers) * len(sampled_heads)}")
    
    # Test each combination
    best_layer = None
    best_head = None
    best_accuracy = 0.0
    
    from tqdm import tqdm
    total_tests = len(sampled_layers) * len(sampled_heads)
    with tqdm(total=total_tests, desc="Testing head combinations") as pbar:
        for layer in sampled_layers:
            for head in sampled_heads:
                try:
                    # Load activations
                    activations, labels, _ = load_activations_for_head(
                        h5_files, layer, head, labels_dict
                    )
                    
                    if activations is None or len(activations) < 10:
                        pbar.update(1)
                        continue
                    
                    # Train classifier
                    result = train_classifier_with_split(activations, labels, val_split=0.2, seed=42)
                    val_acc = result['val_acc']
                    
                    logger.info(f"Layer {layer}, Head {head}: Val Acc = {val_acc:.4f}")
                    
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        best_layer = layer
                        best_head = head
                        logger.info(f"  -> New best! Layer {best_layer}, Head {best_head}, Acc {best_accuracy:.4f}")
                    
                except Exception as e:
                    logger.debug(f"Error testing layer {layer}, head {head}: {e}")
                
                pbar.update(1)
    
    if best_layer is None:
        logger.error("Failed to find any valid head")
        return None, None, None
    
    logger.info(f"\nBest head found: Layer {best_layer}, Head {best_head}, Val Acc {best_accuracy:.4f}")
    return best_layer, best_head, best_accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Visualize top-K ITI classifier with decision boundary',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--h5-dir',
        type=str,
        default=None,
        help='Directory containing HDF5 activation files (auto-set if --model is used)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualizations',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--top-k-rank',
        type=int,
        default=1,
        help='Rank of the head to visualize (1 = best, 2 = second best, etc.)'
    )
    
    parser.add_argument(
        '--iti-config',
        type=str,
        default=None,
        help='Path to ITI config pickle file (if available, will extract head info from here)'
    )
    
    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='Manually specify layer (if not using --iti-config)'
    )
    
    parser.add_argument(
        '--head',
        type=int,
        default=None,
        help='Manually specify head (if not using --iti-config)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['gemma', 'llama', 'qwen'],
        default=None,
        help='Model type to use hardcoded layer/head config (gemma, llama, qwen)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Force auto-discovery of best head by testing ALL layer-head combinations (overrides model config)'
    )
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Determine h5_dir (from model config or command line)
        if args.model:
            if args.model not in MODEL_CONFIGS:
                logger.error(f"Unknown model: {args.model}. Supported: {list(MODEL_CONFIGS.keys())}")
                return 1
            h5_dir = MODEL_CONFIGS[args.model]['h5_dir']
            logger.info(f"Using h5_dir from model config: {h5_dir}")
        elif args.h5_dir:
            h5_dir = args.h5_dir
        else:
            logger.error("ERROR: Must specify either --model or --h5-dir")
            return 1
        
        # Discover H5 files
        h5_files = discover_h5_files(h5_dir)
        if not h5_files:
            return 1
        
        # Extract labels
        labels_dict = extract_labels_from_h5_metadata(h5_files)
        if not labels_dict:
            return 1
        
        # Determine which head to visualize
        if args.all:
            # Force auto-discovery (overrides everything)
            logger.info("--all flag detected. Auto-discovering best head by testing ALL combinations...")
            target_layer, target_head, expected_acc = auto_discover_best_head(
                h5_files, labels_dict
            )
            
            if target_layer is None:
                logger.error("Failed to auto-discover best head")
                return 1
            
            logger.info(f"Auto-discovered best head: Layer {target_layer}, Head {target_head}, "
                       f"Val Acc {expected_acc:.4f}")
        
        elif args.model:
            # Use hardcoded model-specific config
            config = MODEL_CONFIGS[args.model]
            target_layer = config['layer']
            target_head = config['head']
            logger.info(f"Using hardcoded config for {args.model}: Layer {target_layer}, Head {target_head}")
        
        elif args.iti_config:
            logger.info(f"Loading head info from ITI config: {args.iti_config}")
            top_heads = find_top_k_heads_from_config(args.iti_config, args.top_k_rank)
            if not top_heads:
                logger.error("Failed to load top heads from ITI config")
                return 1
            
            target_layer, target_head, expected_acc = top_heads[0]
            logger.info(f"Selected head from config: Layer {target_layer}, Head {target_head}, "
                       f"Accuracy {expected_acc:.4f}")
        
        elif args.layer is not None and args.head is not None:
            target_layer = args.layer
            target_head = args.head
            logger.info(f"Using manually specified head: Layer {target_layer}, Head {target_head}")
        
        else:
            # Auto-discover best head
            logger.info("No head specified. Auto-discovering best head by testing ALL combinations...")
            target_layer, target_head, expected_acc = auto_discover_best_head(
                h5_files, labels_dict
            )
            
            if target_layer is None:
                logger.error("Failed to auto-discover best head")
                return 1
            
            logger.info(f"Auto-discovered best head: Layer {target_layer}, Head {target_head}, "
                       f"Val Acc {expected_acc:.4f}")
        
        # Load activations for this head
        activations, labels, row_indices = load_activations_for_head(
            h5_files, target_layer, target_head, labels_dict
        )
        
        if activations is None:
            return 1
        
        # Train classifier
        logger.info("\nTraining classifier...")
        classifier_result = train_classifier_with_split(activations, labels)
        
        # Compute direction vector (both unscaled and scaled versions)
        logger.info("\nComputing direction vector...")
        logger.info("NOTE: Computing direction from UNSCALED activations (matches steer_vector_calc_ITI.py)")
        logger.info("      Then computing scaled version for proper PCA projection")
        direction_info = compute_direction_vector(activations, labels, scaler=classifier_result['scaler'])
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        
        # 1. 2D scatter with decision boundary (full dataset)
        output_path_2d_full = os.path.join(
            args.output_dir, 
            f'classifier_2d_layer{target_layer}_head{target_head}_full.png'
        )
        visualize_classifier_2d(
            activations, labels, classifier_result, direction_info,
            target_layer, target_head, output_path_2d_full, use_validation_only=False
        )
        
        # 2. 2D scatter with decision boundary (validation only)
        output_path_2d_val = os.path.join(
            args.output_dir, 
            f'classifier_2d_layer{target_layer}_head{target_head}_validation.png'
        )
        visualize_classifier_2d(
            activations, labels, classifier_result, direction_info,
            target_layer, target_head, output_path_2d_val, use_validation_only=True
        )
        
        # 3. 1D projection histograms
        output_path_1d = os.path.join(
            args.output_dir, 
            f'classifier_projections_layer{target_layer}_head{target_head}.png'
        )
        visualize_classifier_with_projections(
            activations, labels, classifier_result, direction_info,
            target_layer, target_head, output_path_1d
        )
        
        # 4. Diagnostic plots (ROC, PR, Confusion Matrix, Calibration) - 4 SEPARATE FILES
        logger.info("\nGenerating diagnostic plots (4 separate files)...")
        diagnostic_metrics = visualize_classifier_diagnostics(
            activations, labels, classifier_result,
            target_layer, target_head, args.output_dir
        )
        
        logger.info("\n" + "="*80)
        logger.info("VISUALIZATION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Visualized: Layer {target_layer}, Head {target_head}")
        logger.info(f"Validation accuracy: {classifier_result['val_acc']:.4f}")
        logger.info(f"Output directory: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
