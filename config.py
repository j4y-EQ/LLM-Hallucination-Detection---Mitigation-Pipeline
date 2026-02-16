"""Configuration for Hallucination Detection Pipeline.

Central configuration file containing all hyperparameters, paths, and settings
for the three-stage hallucination detection pipeline (Generation, Classification,
Evaluation). Modify this file to customize pipeline behavior.

CONFIGURATION SECTIONS:

1. MODEL CONFIGURATION:
   - MODEL_NAME: Model identifier (e.g., 'llama3_8b_instruct')
   - HUGGINGFACE_MODEL_ID: HuggingFace model path
   - TRANSFORMER_LENS_MODEL_NAME: TransformerLens model identifier

2. DATASET CONFIGURATION:
   - QA_DATASETS: List of dataset paths with sample counts
   - TOTAL_SAMPLES: Total samples across all datasets
   - CHUNK_SIZE: Samples per chunk for distributed processing

3. GENERATION SETTINGS:
   - MAX_ANSWER_TOKENS: Maximum tokens to generate per answer
   - BATCH_SIZE: Samples per batch (affects GPU memory)
   - FIRST_PERIOD_TRUNCATION: Truncate at first period

4. ACTIVATION CAPTURE:
   - START_LAYER, END_LAYER: Layer range for activation capture
   - SKIP_LAYERS: Layers to exclude from classification
   - ACTIVE_HOOKS: Which hooks to capture (residual, attention, MLP)
   - TOKEN_SCHEMES: Position strategies (bos_token, last_token, etc.)

5. CLASSIFIER TRAINING:
   - CLASSIFIER_BASE_DIR: Output directory for trained classifiers
   - GLOBAL_RANDOM_SEED: Random seed for reproducibility
   - HANDLE_IMBALANCE: Enable imbalance handling (auto-detected)
   - IMBALANCE_THRESHOLD: Threshold for imbalance detection
   - FINALIST_COUNT: Number of top classifiers to evaluate

6. THRESHOLD OPTIMIZATION:
   - PROBE_THRESHOLD_MODE: 'quantile' or 'fixed'
   - PROBE_QUANTILES: Quantile values for threshold search
   - PROBE_FIXED_THRESHOLDS: Fixed threshold values

7. GPU CONFIGURATION:
   - DEVICE_ID: GPU device ID (0, 1, 2, etc.)
   - NUM_GPUS: Number of GPUs for multi-GPU training
   - GPU_IDS: List of GPU IDs for parallel training

8. LOGGING:
   - LOG_MAX_BYTES: Max log file size before rotation
   - LOG_BACKUP_COUNT: Number of rotated log backups
   - DEBUG_VERBOSE: Enable verbose debug logging

9. CUSTOM SCORING:
   - CUSTOM_SCORING: Dictionary with custom metric configuration
   - Beta values, weights, blend modes for hallucination detection

IMPORTANT SETTINGS:
    - EXPERIMENT_ID: Unique identifier for activation/classifier outputs
    - RESULTS_BUFFER_SIZE: Batches to buffer before flushing to disk
    - PRODUCTION_MODE: Limits for testing (MAX_BATCHES, MAX_BATCH_FILES)

USAGE:
    from config import MODEL_NAME, BATCH_SIZE, START_LAYER
    
NOTE:
    This file is imported by all pipeline components. Changes here affect
    generator.py, classifier.py, and evaluate.py.
"""

# ================================================================
# Configuration for Hallucination Detection Pipeline
# ============================================================
import os

# --- PATHS ------------------------------------------------------
ROOT = "./data"

# --- MODEL CONFIGURATION --------------------------------------`
MODEL_NAME = "llama3_8b_instruct"
HUGGINGFACE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
TRANSFORMER_LENS_MODEL_NAME = HUGGINGFACE_MODEL_ID  # Used by TransformerLens for model loading

# --- PATHS & CONSTANTS (DYNAMICALLY SET BASED ON ENVIRONMENT) ---
QA_DATASETS = [
    {'path': os.path.join(ROOT, "squad_clean.csv"), 'samples': 50000}
]

ACTIVATIONS_BASE_DIR = os.path.join(ROOT, "activations")
os.makedirs(ACTIVATIONS_BASE_DIR, exist_ok=True)

# --- EVALUATION / GENERATION -----------------------------------
RANDOM_SEED = 42
MAX_ANSWER_TOKENS = 120
BATCH_SIZE = 8
DEBUG_VERBOSE = False

# --- EXPERIMENT IDENTIFIER --------------------------------------
EXPERIMENT_ID = "squad"

# --- BUFFERED SAVING --------------------------------------------
RESULTS_BUFFER_SIZE = 200

# --- GPU CONFIGURATION -----------------------------------------
DEVICE_ID = 0

# --- CHUNKING ---------------------------------------------------
TOTAL_SAMPLES = sum(d['samples'] for d in QA_DATASETS)
NUM_CHUNKS = 4
CHUNK_ID = 0

CHUNK_SIZE = max(1, TOTAL_SAMPLES // NUM_CHUNKS)

# --- HOOKS / LAYERS --------------------------------------------
START_LAYER =8  # Inclusive. Set to None to auto-detect from model (will use 0)
END_LAYER = 18    # Inclusive. Set to None to auto-detect from model (will use n_layers - 1)
SKIP_LAYERS = {}    # Layers to skip during classification analysis

# Default hooks: Residual stream only
DEFAULT_HOOKS = [
    # "hook_resid_pre",   # Input to each layer
    # "hook_resid_mid",   # After attention, before MLP
    # "hook_resid_post"   # After MLP (layer output)
]

# Additional hooks you can easily enable by uncommenting:
ADDITIONAL_HOOKS = [
    # Residual Stream hooks (3 total)
    # "hook_resid_mid",       # After attention, before MLP
    # "hook_resid_post",      # After MLP (layer output)
    # "hook_resid_pre",       # Input to each layer
    
    # Attention hooks (9 total)
    # "attn.hook_attn_scores", # Raw attention scores (pre-softmax) - HOW STRONG
    # "attn.hook_k",          # Key vectors
    # "attn.hook_pattern",    # Attention patterns (post-softmax) - WHERE model looks
    # "attn.hook_q",          # Query vectors
    # "attn.hook_rot_k",      # Rotary positional embedding for keys
    # "attn.hook_rot_q",      # Rotary positional embedding for queries
    # "attn.hook_v",          # Value vectors
    "attn.hook_z",          # Attention head outputs
    # "hook_attn_out",        # Attention output hook (post-residual connection)

    # # MLP hooks (4 total)
    # "mlp.hook_post",        # MLP output (after activation)
    "mlp.hook_pre",         # MLP input (before activation)
    "mlp.hook_pre_linear",  # MLP pre-linear transformation hook
    
    # # LayerNorm hooks (2 total)
    # "ln1.hook_normalized",  # Pre-attention LayerNorm output
    # "ln2.hook_normalized",  # Pre-MLP LayerNorm output
]

# Token position schemes to capture
TOKEN_SCHEMES = [
    # "bos_token",           # Beginning of sequence token (position 0)
    # "last_prompt_token",   # Last token of the input prompt
    # "first_generated",     # First generated token
    "last_generated"       # Last generated token (at actual end of meaningful answer)
]

# Combine all active hooks from config
ACTIVE_HOOKS = DEFAULT_HOOKS + ADDITIONAL_HOOKS

# Additional runtime variables
SAMPLE_N = CHUNK_SIZE  # Backward compatibility

# --- LOGGING CONFIGURATION -------------------------------------
# Unified logging system with file rotation - all components share the same log files
LOG_MAX_BYTES = 100*1024*1024  # 100 MB per main log file before rotation (increased for unified logs)
LOG_BACKUP_COUNT = 999999  # Keep ALL backup files - no deletion (pipeline_full_YYYYMMDD_HHMMSS.log.X)
LOG_ERROR_MAX_BYTES = 20*1024*1024  # 20 MB for error log file before rotation (increased for unified logs)
LOG_ERROR_BACKUP_COUNT = 999999  # Keep ALL backup files - no deletion (pipeline_errors_YYYYMMDD_HHMMSS.log.X)

# ================================================================
# CLASSIFIER CONFIGURATION
# ================================================================

# --- PATHS FOR CLASSIFIER ---------------------------------------
CLASSIFIER_BASE_DIR = os.path.join(ROOT, "classifier")

# --- PRODUCTION MODE CONFIGURATION -------------------------------
PRODUCTION_MAX_BATCHES = None
PRODUCTION_MAX_BATCH_FILES = None

# --- CLASS IMBALANCE HANDLING ------------------------------------
HANDLE_IMBALANCE = True
IMBALANCE_THRESHOLD = 0.4


# --- CUSTOM SCORING CONFIGURATION --------------------------------
# This configuration controls the custom scoring function for model selection
# - beta: F-beta weight for class 0 (hallucination), higher values emphasize recall
# - w: Weight for class 0 in the blend (higher values emphasize hallucination detection)
# - gamma: MCC gate power (higher values make the gate more strict)
# - blend: Blend method ('arith' for arithmetic mean, 'geom' for geometric mean)
CUSTOM_SCORING = {
    'beta': 2.5,              # F-beta for class 0 (hallucinations), emphasizing recall
    'w': 0.8,                 # Weight for class 0 in blend (0.8 means 80% weight on hallucination)
    'gamma': 1.2,             # MCC gate power (higher values make gate more strict)
    'blend': 'geom'           # Blend method: 'arith' or 'geom'
}

# --- RANDOM SEED CONFIGURATION ----------------------------------
GLOBAL_RANDOM_SEED = RANDOM_SEED

# Constants for threshold optimization and finalist selection
FINALIST_COUNT = 5
PROBE_THRESHOLD_MODE = "quantiles"
PROBE_QUANTILES = [0.50, 0.75, 0.90, 0.96, 0.99]
PROBE_FIXED_THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]

# ================================================================
# EVALUATION-SPECIFIC CONFIGURATION
# ================================================================

SAMPLE_SIZE = 5000
DATASET_CSV_PATH = "./data/repliqa.csv"

MAX_EVAL_WORKERS = 30
SAVE_RESULTS = True
TEMPERATURE = 0.0

ENABLE_MEMORY_CLEARING = True
SPECIFIC_CLASSIFIER_RUN_ID = ""

# --- LINEAR ALGEBRA THREAD CONFIGURATION ---
LINALG_THREADS = 96

# ================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ================================================================

# --- GPU CONFIGURATION FOR HYPERTUNING --------------------------
NUM_GPUS = 1
GPU_IDS = [1]

# --- HYPERPARAMETER TUNING PARAMETERS ---------------------------
MAX_EVALS = 500
N_SPLITS = 5

