"""Question Answering Generation with Activation Capture.

Generates answers to QA questions using LLaMA/GPT models while capturing
transformer activations at multiple layers/hooks for hallucination detection
training. Supports batch processing, automatic Azure GPT evaluation, and
industrial-strength checkpointing.

FEATURES:
    - Multi-hook, multi-layer activation capture during generation
    - Configurable token position schemes (BOS, EOS, last token, etc.)
    - Batch processing with left-padding for efficiency
    - Automatic Azure GPT-4o evaluation of generated answers
    - HDF5-based activation storage with compression
    - Resume capability via checkpoint system
    - Buffered saving for reduced I/O operations

WORKFLOW:
    1. Load QA dataset (SQuAD, Natural Questions, etc.)
    2. Generate answers using LLM with activation hooks
    3. Capture activations at specified layers/hooks/positions
    4. Evaluate answers using Azure GPT-4o
    5. Save results and activations with checkpointing

CONFIGURATION:
    - EXPERIMENT_ID: Unique identifier for this run
    - BATCH_SIZE: Samples per batch (affects GPU memory)
    - START_LAYER, END_LAYER: Layer range for activation capture
    - ACTIVE_HOOKS: Which hooks to capture (residual, attention, MLP)
    - TOKEN_SCHEMES: Position strategies for activation extraction

USAGE:
    $ python generator.py --device-id 0 --chunk-id 0

OUTPUT:
    - Activations: ./data/activations/{EXPERIMENT_ID}/chunk_N/
    - Results: batch_N_results.pkl (includes evaluation labels)
    - Metadata: config_metadata.json (for classifier compatibility)

Dependencies:
    - Azure OpenAI: For answer evaluation
    - TransformerLens: For activation capture
    - HDF5: For efficient activation storage
"""

# --- CRITICAL: SET GPU DEVICE BEFORE TORCH IS IMPORTED ---
import numpy as np
import h5py
import pandas as pd
import torch
import time
import gc
import torch.cuda.memory as memory
import json
from helpers.activation_utils import generate_and_capture_efficiently
from helpers.model_manager import ModelManager
from helpers.token_manager import TokenManager
from helpers.checkpoint import CheckpointManager
from logger import consolidated_logger as logger
from helpers.eval_model import (
    test_evaluator_connectivity,
    judge_answer,
    batch_judge_answers,
    EVAL_MODEL,
    client
)
from helpers.atomic_operations import AtomicOperations
from config import (
    ROOT, QA_DATASETS, ACTIVATIONS_BASE_DIR, RANDOM_SEED, MAX_ANSWER_TOKENS,
    BATCH_SIZE, DEBUG_VERBOSE, EXPERIMENT_ID, RESULTS_BUFFER_SIZE,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_ERROR_MAX_BYTES, LOG_ERROR_BACKUP_COUNT,
    TOTAL_SAMPLES, NUM_CHUNKS, CHUNK_SIZE, START_LAYER, END_LAYER, SKIP_LAYERS,
    DEFAULT_HOOKS, ADDITIONAL_HOOKS, TOKEN_SCHEMES, ACTIVE_HOOKS, SAMPLE_N, MODEL_NAME, HUGGINGFACE_MODEL_ID, TRANSFORMER_LENS_MODEL_NAME
)
import re
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai                              # Azure OpenAI SDK
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import config
import os
import sys
import argparse

# Parse arguments BEFORE importing config to allow override
# python -m pipeline.core.generator --device-id 3 --chunk-id 3
parser = argparse.ArgumentParser(
    description='Run activation generation pipeline')
parser.add_argument('--chunk-id', type=int, default=None,
                    help='Chunk ID to process (overrides config.CHUNK_ID)')
parser.add_argument('--device-id', type=int, default=None,
                    help='GPU device ID (overrides config.DEVICE_ID)')
parser.add_argument('--model-name', type=str, default=None,
                    help='Model name for processing (overrides config.MODEL_NAME)')
parser.add_argument('--huggingface-model-id', type=str, default=None,
                    help='HuggingFace model ID (overrides config.HUGGINGFACE_MODEL_ID)')
parser.add_argument('--first-period-truncation', type=lambda x: (str(x).lower() == 'true'), default=None,
                    help='Truncate generated text at first period (True/False, overrides config.FIRST_PERIOD_TRUNCATION)')
args = parser.parse_args()

# Import config

# Override config values if provided via CLI
if args.chunk_id is not None:
    config.CHUNK_ID = args.chunk_id
    CHUNK_ID = args.chunk_id
else:
    CHUNK_ID = config.CHUNK_ID

if args.device_id is not None:
    config.DEVICE_ID = args.device_id
    DEVICE_ID = args.device_id
else:
    DEVICE_ID = config.DEVICE_ID

if args.model_name is not None:
    config.MODEL_NAME = args.model_name
    MODEL_NAME = args.model_name
else:
    MODEL_NAME = config.MODEL_NAME

if args.huggingface_model_id is not None:
    config.HUGGINGFACE_MODEL_ID = args.huggingface_model_id
    HUGGINGFACE_MODEL_ID = args.huggingface_model_id
    # Also update TRANSFORMER_LENS_MODEL_NAME if it's based on HUGGINGFACE_MODEL_ID
    config.TRANSFORMER_LENS_MODEL_NAME = args.huggingface_model_id
    TRANSFORMER_LENS_MODEL_NAME = args.huggingface_model_id
else:
    HUGGINGFACE_MODEL_ID = config.HUGGINGFACE_MODEL_ID
    TRANSFORMER_LENS_MODEL_NAME = config.TRANSFORMER_LENS_MODEL_NAME

# FIRST_PERIOD_TRUNCATION: Only use CLI argument, do not load from config.py
if args.first_period_truncation is not None:
    FIRST_PERIOD_TRUNCATION = args.first_period_truncation
else:
    FIRST_PERIOD_TRUNCATION = False  # Default to False if not specified

# Set GPU before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
# --- END CRITICAL SECTION ---

# ================================================================
# CONFIGURATION IMPORT
# ================================================================
# Only Azure credentials are optionally loaded from .env for security
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ================================================================
# ACTIVATION CAPTURE CONFIGURATION
# ================================================================
# Combine all active hooks from config
ACTIVE_HOOKS = DEFAULT_HOOKS + ADDITIONAL_HOOKS

# ================================================================
# CHECKPOINT CONTINUATION AND CHUNK CONFIGURATION
# ================================================================
# Log CLI overrides if they were provided
if args.chunk_id is not None:
    logger.info(f"CHUNK_ID overridden via CLI: {args.chunk_id}")
if args.device_id is not None:
    logger.info(f"DEVICE_ID overridden via CLI: {args.device_id}")
if args.model_name is not None:
    logger.info(f"MODEL_NAME overridden via CLI: {args.model_name}")
if args.huggingface_model_id is not None:
    logger.info(f"HUGGINGFACE_MODEL_ID overridden via CLI: {args.huggingface_model_id}")
if args.first_period_truncation is not None:
    logger.info(f"FIRST_PERIOD_TRUNCATION overridden via CLI: {args.first_period_truncation}")

# VALIDATION: Ensure CHUNK_ID is always valid and starts from 0
if CHUNK_ID < 0 or CHUNK_ID >= NUM_CHUNKS:
    raise ValueError(
        f"CHUNK_ID must be between 0 and {NUM_CHUNKS-1}, got {CHUNK_ID}")
logger.info(
    f"CHUNK VALIDATION: Using chunk {CHUNK_ID} (0-based indexing, valid range: 0-{NUM_CHUNKS-1})")


def get_or_create_run_directory():
    """
    Manages directories for experiments, using EXPERIMENT_ID to specify which chunk to resume.

    - If EXPERIMENT_ID is set in config, it uses that.
    - If EXPERIMENT_ID is empty, it creates a new one based on the timestamp.
    - Always creates/uses chunk_{CHUNK_ID} subdirectory.

    Returns:
        tuple: (OUT_DIR, RUN_ID, RUN_TIMESTAMP, IS_CONTINUING)
    """
    global EXPERIMENT_ID, CHUNK_ID, ACTIVATIONS_BASE_DIR

    # --- Step 1: Establish the Experiment ID ---
    if not EXPERIMENT_ID:
        EXPERIMENT_ID = f"exp_{int(time.time())}"
        logger.info(
            f"New experiment started with EXPERIMENT_ID: {EXPERIMENT_ID}")
    else:
        logger.info(f"Using existing EXPERIMENT_ID: {EXPERIMENT_ID}")

    # --- Step 2: Create directory for current chunk ---
    run_id = f"chunk_{CHUNK_ID}"
    out_dir = os.path.join(ACTIVATIONS_BASE_DIR, EXPERIMENT_ID, run_id)

    # Check if directory already exists (resume mode)
    is_continuing = os.path.exists(out_dir)

    if is_continuing:
        logger.info("="*80)
        logger.info("CHECKPOINT CONTINUATION MODE")
        logger.info("="*80)
        logger.info(
            f"Continuing from existing run: {run_id} in experiment {EXPERIMENT_ID}")
        logger.info(f"Directory found: {out_dir}")
    else:
        logger.info("="*80)
        logger.info("NEW RUN MODE (for this chunk)")
        logger.info("="*80)
        logger.info(f"Creating new run for chunk: {run_id}")
        logger.info(f"Directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

    # Extract timestamp from EXPERIMENT_ID if possible
    try:
        run_timestamp = int(EXPERIMENT_ID.split('_')[1])
    except (IndexError, ValueError):
        run_timestamp = int(time.time())  # Fallback

    return out_dir, run_id, run_timestamp, is_continuing


# Get working directory (either continuing or new)
OUT_DIR, RUN_ID, RUN_TIMESTAMP, IS_CONTINUING = get_or_create_run_directory()
logger.info(f"Working directory: {OUT_DIR}")

# Set the logger's output directory to the specific run directory
logger.set_output_directory(OUT_DIR)

# Logger is now using consolidated logger from logger module
logger.info("="*80)
logger.info(
    f"USING CONSOLIDATED LOGGER: Log files will be saved to: {OUT_DIR}")
logger.info("="*80)

SAMPLE_N = CHUNK_SIZE  # Backward compatibility

# ================================================================
# CONFIG METADATA: LOAD OR SAVE FOR EXPERIMENT CONSISTENCY
# ================================================================
# This ensures all chunks in an experiment use the same configuration
# by either loading the saved config (for CHUNK_ID > 0 or resuming CHUNK_ID == 0)
# or saving a new one (for fresh CHUNK_ID == 0 run)

config_metadata_path = os.path.join(
    ACTIVATIONS_BASE_DIR, EXPERIMENT_ID, 'config_metadata.json')

if os.path.exists(config_metadata_path):
    # Load saved configuration - applies to ALL chunks (including resuming CHUNK_ID==0)
    logger.info("="*80)
    logger.info("LOADING SAVED EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    with open(config_metadata_path, 'r') as f:
        saved_config = json.load(f)
    logger.info(
        f"Loaded saved configuration from {config_metadata_path}")
    # Override config.py values with saved configuration to ensure consistency
    DEVICE_ID = saved_config.get('device_id', DEVICE_ID)
    MODEL_NAME = saved_config.get('model_name', MODEL_NAME)
    HUGGINGFACE_MODEL_ID = saved_config.get('huggingface_model_id', HUGGINGFACE_MODEL_ID)
    TRANSFORMER_LENS_MODEL_NAME = saved_config.get('transformer_lens_model_name', TRANSFORMER_LENS_MODEL_NAME)
    QA_DATASETS = saved_config.get('qa_datasets', QA_DATASETS)
    TOTAL_SAMPLES = saved_config.get('total_samples', TOTAL_SAMPLES)
    NUM_CHUNKS = saved_config.get('num_chunks', NUM_CHUNKS)
    CHUNK_SIZE = saved_config.get('chunk_size', CHUNK_SIZE)
    START_LAYER = saved_config.get('start_layer', START_LAYER)
    END_LAYER = saved_config.get('end_layer', END_LAYER)
    SKIP_LAYERS = set(saved_config.get('skip_layers', SKIP_LAYERS))
    RANDOM_SEED = saved_config.get('random_seed', RANDOM_SEED)
    RESULTS_BUFFER_SIZE = saved_config.get(
        'results_buffer_size', RESULTS_BUFFER_SIZE)
    BATCH_SIZE = saved_config.get('batch_size', BATCH_SIZE)
    MAX_ANSWER_TOKENS = saved_config.get(
        'max_answer_tokens', MAX_ANSWER_TOKENS)
    FIRST_PERIOD_TRUNCATION = saved_config.get(
        'first_period_truncation', FIRST_PERIOD_TRUNCATION)
    DEFAULT_HOOKS = saved_config.get('default_hooks', DEFAULT_HOOKS)
    ADDITIONAL_HOOKS = saved_config.get('additional_hooks', ADDITIONAL_HOOKS)
    ACTIVE_HOOKS = saved_config.get('active_hooks', ACTIVE_HOOKS)
    TOKEN_SCHEMES = saved_config.get('token_schemes', TOKEN_SCHEMES)
    ROOT = saved_config.get('root', ROOT)
    ACTIVATIONS_BASE_DIR = saved_config.get('activations_base_dir', ACTIVATIONS_BASE_DIR)
    logger.info("Configuration successfully restored for experiment consistency")
    logger.info("="*80)
elif CHUNK_ID == 0:
    # Save new configuration for this experiment (fresh run, CHUNK_ID==0)
    # This will be loaded by all subsequent chunks and resuming runs
    logger.info("="*80)
    logger.info("SAVING EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    resolved_config = {
        'experiment_id': EXPERIMENT_ID,
        'chunk_id': CHUNK_ID,
        'device_id': DEVICE_ID,
        'model_name': MODEL_NAME,
        'huggingface_model_id': HUGGINGFACE_MODEL_ID,
        'transformer_lens_model_name': TRANSFORMER_LENS_MODEL_NAME,
        'qa_datasets': QA_DATASETS,
        'total_samples': TOTAL_SAMPLES,
        'num_chunks': NUM_CHUNKS,
        'chunk_size': CHUNK_SIZE,
        'start_layer': START_LAYER,  # Will be auto-detected later, save current value
        'end_layer': END_LAYER,  # Will be auto-detected later, save current value
        'skip_layers': list(SKIP_LAYERS),
        'random_seed': RANDOM_SEED,
        'results_buffer_size': RESULTS_BUFFER_SIZE,
        'batch_size': BATCH_SIZE,
        'max_answer_tokens': MAX_ANSWER_TOKENS,
        'first_period_truncation': FIRST_PERIOD_TRUNCATION,
        'default_hooks': DEFAULT_HOOKS,
        'additional_hooks': ADDITIONAL_HOOKS,
        'active_hooks': ACTIVE_HOOKS,
        'token_schemes': TOKEN_SCHEMES,
        'root': ROOT,
        'activations_base_dir': ACTIVATIONS_BASE_DIR,
    }
    
    # Create experiment directory if it doesn't exist
    os.makedirs(os.path.dirname(config_metadata_path), exist_ok=True)
    
    with open(config_metadata_path, 'w') as f:
        json.dump(resolved_config, f, indent=4)
    logger.info(f"Saved resolved configuration to {config_metadata_path}")
    logger.info("="*80)
else:
    # CHUNK_ID > 0 but config_metadata.json doesn't exist - ERROR
    logger.error(
        f"Configuration metadata not found at {config_metadata_path}. Cannot proceed.")
    logger.error(f"CHUNK_ID={CHUNK_ID} requires saved config from CHUNK_ID=0")
    raise FileNotFoundError(
        f"Configuration metadata not found at {config_metadata_path}")

# Global variables for activation collection - now managed by checkpoint_manager
# activation_storage, captured_activations, global_error_log, skipped_data_points, buffer_flush_counter
# are now accessed via checkpoint_manager.activation_storage, etc.

# GPU Memory tracking

# Global inference mode enforcement
INFERENCE_MODE_ENFORCED = False

# ================================================================
# ATOMIC CHECKPOINT SYSTEM - INDUSTRY STANDARD
# ================================================================

# Create a global atomic checkpoint manager with debug settings from the script
atomic_operation_manager = AtomicOperations(debug_verbose=DEBUG_VERBOSE)

# Create a global checkpoint manager with debug settings from the script
checkpoint_manager = CheckpointManager(
    atomic_operation_manager,
    debug_verbose=DEBUG_VERBOSE,
    OUT_DIR=OUT_DIR,
    RUN_TIMESTAMP=RUN_TIMESTAMP,
    CHUNK_SIZE=CHUNK_SIZE,
    BATCH_SIZE=BATCH_SIZE,
    RESULTS_BUFFER_SIZE=RESULTS_BUFFER_SIZE,
    DEVICE_ID=DEVICE_ID
)

# Set function dependencies
checkpoint_manager.batch_judge_answers = batch_judge_answers
checkpoint_manager.tqdm = tqdm


def load_batch_wise_activations(activations_dir: str, model_name: str, layer: int, hook_name: str, token_scheme: str, row_idx: int = None):
    """
    Load activations from batch-wise HDF5 storage (memory-efficient approach).
    
    Loads and concatenates activations from multiple batch files for a specific
    layer/hook/scheme combination. This approach is memory-efficient as it only
    loads necessary data rather than keeping all activations in memory.
    
    Args:
        activations_dir (str): Directory containing batch HDF5 files
        model_name (str): Name of the model (used in filename pattern)
        layer (int): Layer number to load activations from
        hook_name (str): Name of the hook (e.g., 'attn.hook_z', 'mlp.hook_pre')
        token_scheme (str): Token position scheme (e.g., 'last_generated', 'bos_token')
        row_idx (int, optional): If provided, filters for specific row index. Defaults to None.
    
    Returns:
        np.ndarray or None: Combined activations array of shape (n_samples, n_features),
                           or None if no matching data found
    
    File Pattern:
        Searches for files matching: {model_name}_batch_*_activations_*.h5
    
    HDF5 Structure:
        /{hook_key}/{token_scheme}/activations: activation arrays
        /{hook_key}/{token_scheme}/row_indices: corresponding sample indices
    
    Notes:
        - Automatically concatenates data from multiple batch files
        - Handles errors gracefully, skipping corrupted files
        - Logs warnings if no data found for specified configuration
    """
    import glob

    # Find all batch files for this model
    batch_pattern = os.path.join(
        activations_dir, f"{model_name}_batch_*_activations_*.h5")
    batch_files = glob.glob(batch_pattern)

    if not batch_files:
        logger.warning(f"No batch files found for model {model_name}")
        return None

    hook_key = f"layer_{layer}_{hook_name}"
    all_activations = []
    all_row_indices = []

    # Load from all batch files
    for batch_file in sorted(batch_files):
        try:
            with h5py.File(batch_file, 'r') as h5f:
                if hook_key in h5f and token_scheme in h5f[hook_key]:
                    scheme_group = h5f[hook_key][token_scheme]
                    activations = scheme_group['activations'][:]
                    row_indices = scheme_group['row_indices'][:]

                    all_activations.append(activations)
                    all_row_indices.append(row_indices)
        except Exception as e:
            logger.error(f"Error loading {batch_file}: {e}")
            continue

    if not all_activations:
        logger.warning(f"No data found for {hook_key}/{token_scheme}")
        return None

    # Concatenate all batches
    combined_activations = np.concatenate(all_activations, axis=0)
    combined_row_indices = np.concatenate(all_row_indices, axis=0)

    if row_idx is not None:
        # Find specific row
        mask = combined_row_indices == row_idx
        if mask.any():
            return combined_activations[mask][0]  # Return first match
        else:
            return None
    else:
        return combined_activations, combined_row_indices


# 4. Load data & split into unique chunks
dataframes = []
logger.info("Loading and sampling datasets:")
for dataset_config in QA_DATASETS:
    path = dataset_config['path']
    n_samples = dataset_config['samples']
    df = pd.read_csv(path)
    if len(df) < n_samples:
        logger.warning(
            f"Dataset at {path} has {len(df)} rows, but {n_samples} were requested. Using all available rows.")
        n_samples = len(df)
    logger.info(
        f"  - Taking {n_samples} samples from {os.path.basename(path)}")
    dataframes.append(df.sample(n=n_samples, random_state=RANDOM_SEED))

df_all = pd.concat(dataframes, ignore_index=True)
logger.info(f"Total samples combined: {len(df_all)}")

# Ensure we have enough data
if len(df_all) < TOTAL_SAMPLES:
    logger.warning(
        f"Loaded {len(df_all)} samples, which is less than the configured {TOTAL_SAMPLES}. Adjusting total.")
    TOTAL_SAMPLES = len(df_all)
    CHUNK_SIZE = max(1, TOTAL_SAMPLES // NUM_CHUNKS)

# Deterministically shuffle the entire dataset once
df_shuffled = df_all.sample(
    frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Split into chunks - each chunk gets unique samples
chunk_start = CHUNK_ID * CHUNK_SIZE
chunk_end = min((CHUNK_ID + 1) * CHUNK_SIZE, TOTAL_SAMPLES)

# Handle last chunk (might be slightly larger if TOTAL_SAMPLES not divisible by NUM_CHUNKS)
if CHUNK_ID == NUM_CHUNKS - 1:
    chunk_end = TOTAL_SAMPLES

# Keep original global indices to avoid duplicate row_idx values across chunks
df_sample = df_shuffled[chunk_start:chunk_end]

logger.info(f"\n" + "="*60)
logger.info(f"CHUNK PROCESSING CONFIGURATION")
logger.info(f"="*60)
logger.info(f"Processing Chunk {CHUNK_ID}/{NUM_CHUNKS}")
logger.info(
    f"Dataset samples: {chunk_start} to {chunk_end-1} ({len(df_sample)} total)")
logger.info(
    f"Global indices range: {df_shuffled[chunk_start:chunk_end].index.min()} to {df_shuffled[chunk_start:chunk_end].index.max()}")
logger.info(f"Random seed: {RANDOM_SEED} (consistent across all chunks)")
logger.info(f"="*60)

# CHECKPOINT RECOVERY: Filter samples and recover results
sample_list_full = list(df_sample.itertuples())[:SAMPLE_N]

# OPTIMIZATION: Scan all checkpoint files just ONCE at the start.
samples_with_activations, samples_with_results, samples_from_progress = checkpoint_manager.scan_checkpoint_files()
all_completed_samples = samples_with_activations.union(
    samples_with_results).union(samples_from_progress)

# Get completed samples for detailed reporting
checkpoint_manager.print_recovery_summary(
    all_completed_samples, sample_list_full)

sample_list, resume_batch_offset = checkpoint_manager.filter_remaining_samples(
    sample_list_full,
    samples_with_activations,
    samples_with_results,
    samples_from_progress
)

# After setting CUDA_VISIBLE_DEVICES, the visible GPU appears as cuda:0
device = "cuda:0" if torch.cuda.is_available() else "cpu"
results = []

# Load all existing results ONCE at the start if we are resuming a run.
# This list will be appended to as new batches are processed.
if resume_batch_offset > 0:
    logger.info("\n" + "="*60)
    logger.info("RECOVERING PREVIOUS RESULTS")
    logger.info("="*80)
    # Call recover_all_results() once and store them.
    checkpoint_manager.recover_all_results()
    logger.info(
        f"RECOVERED: {len(checkpoint_manager.generation_results)} previous generation results.")

    # CRITICAL: Create the set of already processed samples to avoid duplication.
    already_processed = {r['row_idx']
                         for r in checkpoint_manager.generation_results}
    logger.info(
        f"Will process remaining {len(sample_list)} samples and combine with recovered results.")
    logger.info("="*60)
else:
    already_processed = set()


# Create a unified config dictionary to pass to the ModelManager
config = {
    'DEVICE_ID': DEVICE_ID,
    'HUGGINGFACE_MODEL_ID': HUGGINGFACE_MODEL_ID,
    'TRANSFORMER_LENS_MODEL_NAME': TRANSFORMER_LENS_MODEL_NAME,
    'MODEL_NAME': MODEL_NAME,
}

# --- Orchestration using ModelManager ---
logger.info("Initializing ModelManager...")
model_manager = ModelManager(config)
model_manager.check_initial_gpu_memory()  # Checks GPU state before loading
model_manager.clear_gpu_memory()  # Clears memory before loading

logger.info(f"\n=== Loading {MODEL_NAME} via ModelManager ===")
model_manager.load_model()
model_manager.optimize_for_inference()

# After setting CUDA_VISIBLE_DEVICES, the visible GPU appears as cuda:0
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Start timing
start_time = time.time()
logger.info(f"Starting processing at {time.strftime('%H:%M:%S')}")

# Initialize TokenManager with the loaded model
# Use the transformer-lens model name for model_dir
token_manager = TokenManager(model=model_manager.get_model(
), max_answer_tokens=MAX_ANSWER_TOKENS, model_dir=TRANSFORMER_LENS_MODEL_NAME)
logger.info("TokenManager initialized successfully")

# Model loaded successfully
token_manager.setup_tokenizer_padding(model_manager.get_model())

model_manager.get_model().eval()
logger.info("Model loaded successfully!")

# Debug token IDs to identify conflicts
token_manager.get_token_ids_debug_info(model_manager.get_model())


# ENFORCE INFERENCE MODE: Disable gradients for all model parameters
# This is now handled by model_manager.optimize_for_inference()

# Verify model is on GPU
try:
    model = model_manager.get_model()
    if hasattr(model, 'embed') and hasattr(model.embed, 'W_E'):
        device_name = str(model.embed.W_E.device)
        if 'cuda' in device_name:
            logger.info(f"Model confirmed on GPU: {device_name}")
        else:
            logger.warning(f"Model not on GPU - device: {device_name}")
    else:
        logger.info(
            "Model device verification: Unable to check embedding layer")
except Exception as e:
    logger.error(f"Model device verification failed: {e}")

try:
    logger.info(f"Model has {model_manager.get_model().cfg.n_layers} layers")
except:
    logger.info("Model structure information will be analyzed below...")

# Before proceeding, verify evaluator connectivity (fail fast)
try:
    test_evaluator_connectivity()
except Exception:
    # Hard-stop to ensure nothing continues even in managed runtimes
    logger.critical("Evaluator connectivity test failed. Exiting.")
    os._exit(1)

# --- NEW: DEBUG OUTPUT FOR RESUMED RUNS ---
# If resuming, print a summary of the results that were recovered.
if IS_CONTINUING and resume_batch_offset > 0:
    logger.info("\n" + "="*80)
    logger.info(f"PREVIOUSLY COMPLETED RESULTS (RECOVERED FROM RUN {RUN_ID})")
    logger.info("="*80)

    # Sort results by row index for consistent logging
    recovered_to_show = sorted(
        checkpoint_manager.generation_results, key=lambda x: x['row_idx'])

    # Show first 20 recovered results
    for i, result_item in enumerate(recovered_to_show[:20]):
        logger.info(
            f"\n--- Recovered Item {i+1} (Row Index: {result_item['row_idx']}) ---")
        logger.info(f"  GROUND TRUTH (ref):  '{result_item['right_answer']}'")
        logger.info(
            f"  SAVED ANSWER:        '{result_item['gpt_answer_trim']}'")
        # Check if evaluation was done, default to 'Not Evaluated'
        evaluation_status = 'Not Evaluated'
        if 'is_correct' in result_item:
            evaluation_status = 'CORRECT' if result_item.get(
                'is_correct') else 'INCORRECT'
        logger.info(f"  SAVED EVALUATION:    {evaluation_status}")

    if len(recovered_to_show) > 20:
        logger.info(
            f"\n... and {len(recovered_to_show) - 20} more recovered results (not shown).")

    logger.info("\n" + "="*80)

MAX_CTX = model_manager.get_model().cfg.n_ctx

# Clear activation storage for new model
checkpoint_manager.clear_activation_storage()

# Initialize buffer flush counter based on resume point
checkpoint_manager.buffer_flush_counter = resume_batch_offset // RESULTS_BUFFER_SIZE

# Token constants are now managed by TokenManager

# Layer range validation
n_layers = model_manager.get_model().cfg.n_layers

# Auto-detect layers if not specified in config
if START_LAYER is None:
    start_layer = 0
    logger.info(f"START_LAYER not specified in config - auto-detected from model: {start_layer}")
else:
    start_layer = START_LAYER

if END_LAYER is None:
    end_layer_inclusive = n_layers - 1
    logger.info(f"END_LAYER not specified in config - auto-detected from model: {end_layer_inclusive}")
else:
    end_layer_inclusive = END_LAYER

# Validate layer bounds
if start_layer < 0:
    start_layer = 0
if end_layer_inclusive >= n_layers:
    end_layer_inclusive = n_layers - 1

logger.info(
    f"Capturing activations from layers {start_layer} to {end_layer_inclusive} inclusive (out of {n_layers} total layers)")

# Main batch processing loop with resume support
remaining_samples = sample_list

if DEBUG_VERBOSE:
    logger.debug(
        f"Processing {len(remaining_samples)} remaining samples out of {CHUNK_SIZE} total in chunk")
    if len(remaining_samples) > 0:
        sample_indices = [s.Index for s in remaining_samples]
        logger.debug(f"Sample indices being processed: {sample_indices}")

# Make progress bar reflect total progress in the chunk
total_chunk_batches = (len(sample_list_full) + BATCH_SIZE - 1) // BATCH_SIZE

# Log initial batch processing info
logger.info("\n" + "="*80)
logger.info("STARTING BATCH PROCESSING")
logger.info("="*80)
logger.info(f"Total batches to process: {total_chunk_batches}")
logger.info(f"Batches already completed: {resume_batch_offset}")
logger.info(f"Batches remaining: {total_chunk_batches - resume_batch_offset}")
logger.info(f"Batch size: {BATCH_SIZE} samples per batch")
logger.info(f"Total samples: {len(sample_list_full)}")
logger.info(f"Samples already processed: {len(checkpoint_manager.generation_results)}")
logger.info(f"Samples remaining: {len(sample_list_full) - len(checkpoint_manager.generation_results)}")
logger.info("="*80 + "\n")

progress_bar = tqdm(range(0, len(remaining_samples), BATCH_SIZE),
                    total=total_chunk_batches,
                    initial=resume_batch_offset,
                    unit="batch")

try:
    for batch_start in progress_bar:

        actual_batch_idx = resume_batch_offset + (batch_start // BATCH_SIZE)
        progress_bar.set_description(
            f"Chunk {CHUNK_ID}/{NUM_CHUNKS} | {MODEL_NAME} | Batch {actual_batch_idx+1}/{total_chunk_batches}")
        batch_end = min(batch_start + BATCH_SIZE, len(remaining_samples))
        batch_rows = remaining_samples[batch_start:batch_end]

        batch_prompts = []
        batch_info = []

        valid_items = []
        for idx_in_batch, row in enumerate(batch_rows):
            if row.Index in already_processed:
                continue

            tok_prompt = token_manager.make_tokens_optimized(
                row.knowledge, row.question)

            if tok_prompt is None:
                checkpoint_manager.log_skipped_data_point(
                    "Insufficient context space: prompt + generation tokens exceed model context length",
                    row_data={'knowledge_len': len(
                        row.knowledge), 'question_len': len(row.question)},
                    row_idx=row.Index
                )
                continue

            batch_prompts.append(tok_prompt)

            batch_item_info = {
                'row_idx': row.Index,
                'row': row,
                'prompt_length': tok_prompt.shape[0],
                'padding_length': 0,
            }
            batch_info.append(batch_item_info)
            valid_items.append((idx_in_batch, row))

        if batch_prompts:
            # Use TokenManager for LEFT-padding (identical to evaluate.py)
            batch_tensor, batch_info = token_manager.create_left_padded_batch(
                batch_prompts, batch_info)

            # Debug batch tensor to verify left-padding is correct
            logger.debug(f"Batch tensor shape: {batch_tensor.shape}")
            logger.debug(
                f"Pad token ID: {model_manager.get_model().tokenizer.pad_token_id}")
            for i in range(len(batch_prompts)):
                logger.debug(
                    f"Item {i} - prompt_len: {batch_info[i]['prompt_length']}, padding_len: {batch_info[i]['padding_length']}")
                # Show first 10 and last 10 tokens
                first_10 = batch_tensor[i][:10].tolist()
                last_10 = batch_tensor[i][-10:].tolist()
                logger.debug(f"Item {i} - first 10 tokens: {first_10}")
                logger.debug(f"Item {i} - last 10 tokens: {last_10}")

            full_sequences, batch_activations, batch_results_list, positions_by_scheme = generate_and_capture_efficiently(
                model=model_manager.get_model(),
                token_manager=token_manager,
                batch_prompts_padded=batch_tensor,
                batch_info=batch_info,
                active_hooks=ACTIVE_HOOKS,
                token_schemes=TOKEN_SCHEMES,
                start_layer=start_layer,
                end_layer=end_layer_inclusive,
                max_answer_tokens=MAX_ANSWER_TOKENS,
                model_name=MODEL_NAME,
                logger=logger,
                debug_verbose=DEBUG_VERBOSE,
                first_period_truncation=FIRST_PERIOD_TRUNCATION
            )
            # -----------------------------------------

            # Append results for this batch to the main list
            checkpoint_manager.generation_results.extend(batch_results_list)

            # BUFFERED SAVING: Accumulate in memory buffers instead of saving immediately
            checkpoint_manager.results_buffer.append(batch_results_list)
            checkpoint_manager.activation_buffer.append(batch_activations)

            # Flush buffers when full
            # Check if buffer has accumulated RESULTS_BUFFER_SIZE number of batches
            # Note: Each batch contains BATCH_SIZE samples, so total samples = RESULTS_BUFFER_SIZE * BATCH_SIZE
            current_batches = len(checkpoint_manager.results_buffer)
            if current_batches >= RESULTS_BUFFER_SIZE:
                # Debug output only if enabled
                total_samples = current_batches * BATCH_SIZE
                logger.info(
                    f"Buffer full! Accumulated {current_batches} batches ({total_samples} total samples)")
                logger.debug(
                    f"Each batch has {BATCH_SIZE} samples, buffer size is {RESULTS_BUFFER_SIZE} batches")
                logger.debug("Calling flush_buffers() now...")
                checkpoint_manager.flush_buffers(MODEL_NAME)

            # --- NEW DETAILED DEBUG OUTPUT BLOCK ---
            logger.info("\n" + "="*80)
            logger.info(
                f"BATCH {actual_batch_idx} COMPLETE - RESULTS & ACTIVATION POSITIONS")
            logger.info("="*80)

            for i in range(len(batch_results_list)):
                result_item = batch_results_list[i]
                info = batch_info[i]

                # Safely decode the full sequence for printing
                full_text = token_manager.decode_sequence_with_markers(
                    full_sequences, i)

                logger.info(f"\n--- Item {i} (Row Index: {info['row_idx']}) ---")
                logger.info(
                    f"  GROUND TRUTH (ref):  '{result_item['right_answer']}'")
                logger.info(
                    f"  TEXT FOR EVALUATION: '{result_item['gpt_answer_trim']}'")
                logger.info("-" * 40)
                logger.info(f"  Full Sequence: \"{full_text}\"")
                logger.info("-" * 40)

                # Print token positions only for schemes that are actually being used
                for scheme in TOKEN_SCHEMES:
                    if scheme in positions_by_scheme:
                        pos = positions_by_scheme[scheme][i].item()
                        tok_str = model_manager.get_model().to_string(
                            full_sequences[i, pos:pos+1])
                        logger.info(
                            f"  - {scheme:<18}: Position {pos:<4} | Token: '{tok_str}'")

                # Print detailed generated words with token IDs
                logger.info(f"  Generated Words with Token IDs:")
                padding_len = info['padding_length']
                prompt_len = info['prompt_length']
                start_of_generation = padding_len + prompt_len
                actual_seq_len = full_sequences.shape[1]

                if start_of_generation < actual_seq_len:
                    generated_part_tokens = token_manager.extract_generated_part_tokens(
                        full_sequences, i, start_of_generation, actual_seq_len)

                    # Find where generation actually stops (at EOS token)
                    gen_len = len(generated_part_tokens)
                    for idx, token_id in enumerate(generated_part_tokens):
                        if token_id.item() == model_manager.get_model().tokenizer.eos_token_id:
                            gen_len = idx + 1
                            break

                    # Print each generated word with its token ID
                    for idx in range(gen_len):
                        debug_output = token_manager.decode_individual_token_with_debug(
                            generated_part_tokens, full_sequences, i, start_of_generation, idx)
                        logger.info(debug_output)
                else:
                    logger.info(f"    No generation found (prompt too long)")

            logger.info("\n" + "="*80)

            # BUFFERED: Only update progress display, actual progress saved in flush_buffers()
            progress_bar.set_postfix(
                processed=f"{len(checkpoint_manager.generation_results)}/{len(sample_list_full)}")
            
            # Log progress and time estimates
            batches_completed = actual_batch_idx + 1
            batches_remaining = total_chunk_batches - batches_completed
            if progress_bar.total > 0:
                # Get time estimates from tqdm
                elapsed_time = progress_bar.format_dict.get('elapsed', 0)
                rate = progress_bar.format_dict.get('rate', 0)
                
                if rate and rate > 0:
                    remaining_seconds = batches_remaining / rate
                    hours_remaining = remaining_seconds / 3600
                    minutes_remaining = (remaining_seconds % 3600) / 60
                    seconds_remaining = remaining_seconds % 60
                    
                    logger.info(
                        f"Batch Progress: {batches_completed}/{total_chunk_batches} | "
                        f"Elapsed: {elapsed_time:.1f}s | "
                        f"ETA: {hours_remaining:.1f}h {minutes_remaining:.1f}m {seconds_remaining:.1f}s | "
                        f"Samples Processed: {len(checkpoint_manager.generation_results)}/{len(sample_list_full)}"
                    )

            # --- AGGRESSIVE MEMORY CLEANUP ---
            # Explicitly delete tensors to prevent leaks between batches
            del full_sequences
            del batch_activations
            del batch_results_list
            del positions_by_scheme
            gc.collect()

            # Clear memory
            checkpoint_manager.clear_activation_storage()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

except KeyboardInterrupt:
    logger.warning("\n" + "="*80)
    logger.warning("KEYBOARD INTERRUPT RECEIVED")
    logger.warning("="*80)
    logger.warning("Gracefully shutting down")
    logger.warning("Exiting due to keyboard interrupt.")
    sys.exit(1)

except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        logger.error("\n" + "="*80)
        logger.error("CUDA OUT OF MEMORY ERROR")
        logger.error("="*80)
        logger.error(f"OOM Error: {e}")
        logger.error("Exiting due to out of memory.")
        sys.exit(1)
    else:
        raise

except Exception as e:
    logger.error("\n" + "="*80)
    logger.error(f"UNEXPECTED ERROR DURING BATCH PROCESSING")
    logger.error("="*80)
    logger.error(f"Error Type: {type(e).__name__}")
    logger.error(f"Error Message: {e}")
    import traceback
    logger.error("Full Traceback:")
    logger.error(traceback.format_exc())
    logger.error("Exiting due to error.")
    sys.exit(1)


# Batch processing completed successfully
logger.info("\n" + "="*80)
logger.info("BATCH PROCESSING COMPLETED SUCCESSFULLY")
logger.info("="*80)
logger.info(f"Total batches processed: {total_chunk_batches}")
logger.info(f"Total samples processed: {len(checkpoint_manager.generation_results)}/{len(sample_list_full)}")
logger.info(f"Elapsed time: {time.time() - start_time:.2f} seconds")
logger.info("="*80 + "\n")

logger.info("===== FINAL FLUSH CHECK =====")
logger.info(
    f"Final flush - results_buffer has {len(checkpoint_manager.results_buffer)} items")
logger.info(f"RESULTS_BUFFER_SIZE = {RESULTS_BUFFER_SIZE}")
if checkpoint_manager.results_buffer:
    logger.debug(
        "Calling final flush_buffers() - this should create HDF5 files!")
    checkpoint_manager.flush_buffers(MODEL_NAME)
    logger.debug("Final flush_buffers() completed")
else:
    logger.info("Skipping final flush - results_buffer is empty")
logger.debug("===== END FINAL FLUSH =====")

# The logic to combine results is no longer needed here, as `generation_results`
# now contains both recovered and newly generated results.
final_results = checkpoint_manager.generation_results

# Save final combined and evaluated results (optional, as batches are already saved)
logger.info(f"\nSaving final combined results to all_results.pkl...")
final_results_file = os.path.join(OUT_DIR, "all_results.pkl")
atomic_operation_manager.atomic_save(final_results, final_results_file)

end_time = time.time()
logger.info(f"Finished processing at {time.strftime('%H:%M:%S')}")
logger.info(
    f"Total time for {MODEL_NAME}: {end_time - start_time:.2f} seconds")

# Override chunk_id for the current run
CHUNK_ID = args.chunk_id if args.chunk_id is not None else CHUNK_ID
logger.info(f"Using chunk_id: {CHUNK_ID}")
