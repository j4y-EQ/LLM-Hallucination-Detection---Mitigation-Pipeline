"""Custom Scoring Metrics for Hallucination Detection.

Provides specialized evaluation metrics that emphasize hallucination detection
(high recall for class 1) while maintaining reasonable performance on
non-hallucinations (class 0). Implements configurable multi-component scoring.

SCORING COMPONENTS:
    1. F-beta for class 1 (hallucinations) with beta > 1 to emphasize recall
       - Prioritizes catching all hallucinations over precision
       - Beta typically 1.5-2.0 for 1.5-2x weight on recall vs precision
    
    2. F1 for class 0 (non-hallucinations) for balanced performance
       - Equal weight on precision and recall for correct answers
    
    3. Weighted blend of class scores (arithmetic or geometric mean)
       - Arithmetic: Simple weighted average
       - Geometric: Penalizes extreme imbalance between classes
    
    4. MCC-based gate that penalizes poor overall discrimination
       - Matthews Correlation Coefficient measures overall quality
       - Gate multiplier reduces score if MCC is low

CONFIGURATION:
    Settings in config.py CUSTOM_SCORING dictionary:
    - 'beta_class1': Beta value for F-beta on hallucinations (default: 2.0)
    - 'w_class1': Weight for hallucination score (default: 0.6)
    - 'w_class0': Weight for non-hallucination score (default: 0.4)
    - 'blend_mode': 'arithmetic' or 'geometric' (default: 'arithmetic')
    - 'mcc_gate_enabled': Enable MCC gating (default: True)
    - 'mcc_gate_threshold': MCC threshold for full score (default: 0.3)

USAGE:
    from helpers.custom_scoring import attach_custom_score
    
    metrics = {...}  # Confusion matrix metrics
    attach_custom_score(metrics)  # Adds 'custom_score' field
    
Used by:
    - pipeline/core/classifier.py: Model evaluation and selection
    - pipeline/core/evaluate.py: Performance analysis
"""

# ================================================================
# Custom Scoring Module
# ================================================================
# Centralized custom scoring logic for hallucination detection metrics
#
# This module provides the attach_custom_score function that combines:
# 1. F-beta for class 1 (hallucinations) with beta > 1 to emphasize recall
# 2. F1 for class 0 (non-hallucinations)
# 3. A weighted blend of the two (arithmetic or geometric)
# 4. An MCC-based gate that penalizes models with poor overall discrimination
#
# Used by:
# - pipeline/core/classifier.py: Model evaluation and selection
# - pipeline/core/evaluate.py: Performance analysis
# ================================================================

from config import CUSTOM_SCORING
from logger import consolidated_logger as logger


def attach_custom_score(metrics):
    """
    Attaches a custom scoring metric to the metrics dictionary that emphasizes 
    hallucination detection (class 1) while maintaining reasonable performance on
    non-hallucinations (class 0).

    The score combines:
    1. F-beta for class 1 (hallucinations) with beta > 1 to emphasize recall
    2. F1 for class 0 (non-hallucinations)
    3. A weighted blend of the two (arithmetic or geometric)
    4. An MCC-based gate that penalizes models with poor overall discrimination

    Args:
        metrics: Dictionary containing confusion matrix metrics and per-class precision/recall
                 Will be modified in-place with additional custom scoring fields

    Returns:
        None (modifies metrics dictionary in-place)

    Example:
        >>> metrics = {
        ...     'precision_hallucinated': 0.85,
        ...     'recall_hallucinated': 0.90,
        ...     'precision_not_hallucinated': 0.88,
        ...     'recall_not_hallucinated': 0.82,
        ...     'true_positives': 180,
        ...     'false_positives': 32,
        ...     'true_negatives': 164,
        ...     'false_negatives': 24
        ... }
        >>> attach_custom_score(metrics)
        >>> print(metrics['custom_final_score'])  # Final combined score
        >>> print(metrics['f1_beta'])  # F-beta score component
        >>> print(metrics['mcc'])  # Matthews correlation coefficient
    """
    # Get parameters from config
    beta = CUSTOM_SCORING['beta']
    w = CUSTOM_SCORING['w']
    gamma = CUSTOM_SCORING['gamma']
    blend_method = CUSTOM_SCORING['blend']

    # Extract required metrics (with safe defaults)
    precision_1 = metrics.get('precision_hallucinated', 0.0)
    recall_1 = metrics.get('recall_hallucinated', 0.0)
    precision_0 = metrics.get('precision_not_hallucinated', 0.0)
    recall_0 = metrics.get('recall_not_hallucinated', 0.0)

    # Compute MCC if not already present
    if 'mcc' not in metrics:
        tp = metrics.get('true_positives', 0)
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fn = metrics.get('false_negatives', 0)

        # Matthews Correlation Coefficient (MCC): measures correlation between predictions and actuals
        # MCC formula: (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        # Range: [-1, 1] where 1=perfect prediction, 0=random, -1=total disagreement
        # Safe calculation to avoid division by zero
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        if denominator == 0:
            mcc = 0.0
        else:
            mcc = numerator / denominator
    else:
        mcc = metrics.get('mcc', 0.0)

    # Calculate F-beta for class 1 (hallucinations)
    denominator_1 = beta**2 * precision_1 + recall_1
    if denominator_1 == 0:
        f1_beta = 0.0
    else:
        f1_beta = (1 + beta**2) * precision_1 * recall_1 / denominator_1

    # Calculate F1 for class 0 (non-hallucinations)
    if precision_0 + recall_0 == 0:
        f0_1 = 0.0
    else:
        f0_1 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)

    # Calculate both blend types (always compute both for consistency)
    epsilon = 1e-10
    # Arithmetic blend: weighted average of F-beta (class 1) and F1 (class 0)
    # w controls emphasis: w=0.7 means 70% F-beta, 30% F1
    blend_arith = (w * f1_beta) + ((1-w) * f0_1)
    # Geometric blend: weighted geometric mean, more sensitive to low component scores
    # Use epsilon to avoid log(0) in geometric mean calculation
    blend_geom = (max(f1_beta, epsilon) ** w) * (max(f0_1, epsilon) ** (1-w))

    # Apply MCC gate
    # Gate function: (max(0, MCC))^gamma - take max first, then raise to power
    # Clamp MCC to [0,1] range, then raise to gamma power (default gamma=2.0)
    # This penalizes models with low correlation between predictions and actuals
    gate_mcc = max(0.0, mcc) ** gamma

    # Apply MCC gate to both blend types
    # Multiply blend score by MCC gate: if MCC=0, final_score=0; if MCC=1, final_score=blend
    # This ensures models with poor correlation are strongly penalized
    final_score_arith = blend_arith * gate_mcc
    final_score_geom = blend_geom * gate_mcc

    # Select final score based on configuration
    if blend_method == 'geom':
        custom_final_score = final_score_geom
    else:  # 'arith' (default)
        custom_final_score = final_score_arith

    # Store all components and parameters in the metrics dictionary
    metrics.update({
        # Scores
        'mcc': mcc,
        'f1_beta': f1_beta,
        'f0_1': f0_1,
        'blend_arith': blend_arith,
        'blend_geom': blend_geom,
        'gate_mcc_gamma': gate_mcc,
        'final_score_arith': final_score_arith,
        'final_score_geom': final_score_geom,
        'custom_final_score': custom_final_score,

        # Parameters used
        'custom_params_beta': beta,
        'custom_params_w': w,
        'custom_params_gamma': gamma,
        'custom_params_blend': blend_method
    })

    # Debug: Log custom score calculation
    logger.debug(
        f"Custom Score Components - F1-beta: {f1_beta:.3f}, F0-1: {f0_1:.3f}, MCC: {mcc:.3f}, Gate: {gate_mcc:.3f}, Final: {custom_final_score:.3f}")
