"""Classifier-Based Hallucination Detection Evaluation.

Evaluates trained hallucination detection classifiers on new QA datasets by
generating answers, capturing activations, and predicting hallucination
probabilities. Compares classifier predictions against Azure GPT evaluations.

Llama 7B Hallucination Detection on Question-Answering Datasets
============================================================
Uses saved XGBoost/SVM classifiers to detect hallucinations in LLM text generation
on question-answering datasets using TransformerLens activation capture.

Features:
- Loads question-answering dataset from CSV format
- Configurable sample size (SAMPLE_SIZE) - process subset or entire dataset
- Uses same QA prompt format as squad_generator3 halueval2.py
- Loads best classifier from nocheckpoint_nodataleak.py (model + scaler + metadata)
- Uses TransformerLens Llama 7B model (local or downloaded from HuggingFace)
- Hooks attention outputs at Layer 11 (attn.hook_z) - matches training setup
- Generates text until EOS token, or first period (.) if FIRST_PERIOD_TRUNCATION enabled - matches generator.py setup
- Extracts activations from the specific token position used during classifier training
- Predicts hallucination probability using same preprocessing as training
- **NEW**: Batch processing for 5-10x speedup using TransformerLens generation (exactly like generator.py)
- Comprehensive results analysis and display
- Automatic results saving to unique evaluation run directory
- Creates evaluation_run_{timestamp} folder INSIDE the specific classifier run directory
- Saves all outputs (CSV, JSON, HTML, metadata) to the evaluation run subdirectory

PERFORMANCE OPTIONS:
- Batch processing: 5-10x faster using TransformerLens generation (exactly like generator.py)

CONFIGURATION ALIGNMENT (with generator.py):
- FIRST_PERIOD_TRUNCATION: Controls whether to truncate at first period (.) - respects config.py setting
- HANDLE_IMBALANCE: Controls imbalance handling in classifier (set during training) - read from config.py
- Both settings are logged at startup to show active configuration

DEVICE CONFIGURATION (matches generator.py):
Device management is handled by ModelManager using DEVICE_ID from config.py.
- DEVICE_ID in config.py specifies which physical GPU to use
- ModelManager sets CUDA_VISIBLE_DEVICES automatically
- Device appears as cuda:0 after CUDA_VISIBLE_DEVICES is set

QUICK START FOR MAXIMUM SPEED:
1. Set SAMPLE_SIZE = 1000 (instead of 5000 for faster testing)
2. Set BATCH_SIZE = 64 (or higher if GPU memory allows)
3. Set DEVICE_ID in config.py to your preferred GPU (0, 1, 2, etc.)

IMPORTANT: This script EXACTLY matches squad_generator3 SquAd Dataset.py activation capture logic
and nocheckpoint_nodataleak.py preprocessing to ensure perfect classifier compatibility.

CRITICAL PADDING FIX APPLIED:
- Batch processing uses LEFT-padding mechanism for optimal performance
- Position calculation: padding_length + prompt_length + actual_gen_length - 1 (EXACT match)
- Layer 11 attn.hook_z tensor indexing: activations[batch_idx, pos, :, :] (EXACT match)
- Flattening: [heads, d_head] -> [heads * d_head] (matches training data)
- Ensures classifier receives activations from identical tensor positions as training

MODEL LOADING:
- Automatically downloads Llama 7B from HuggingFace if not found locally
- Uses exact same loading system as generator.py for consistency:
  - Approach 1 (prioritized): Direct HookedTransformer loading
  - Approach 2 (fallback): HuggingFace + transformer-lens with TokenManager
- Handles tokenizer pad token issues automatically
- Optimized for inference with gradients disabled
"""

# ================================================================
# CENTRALIZED CONFIGURATION IMPORT
# ================================================================
# Import all configuration from the centralized config file
import sys
import os
import json
import argparse

# ================================================================
# COMMAND-LINE ARGUMENTS PARSING
# ================================================================
# Parse arguments BEFORE importing config to allow override
parser = argparse.ArgumentParser(
    description='Run evaluation pipeline with classifier')
parser.add_argument('--experiment-id', type=str, default=None,
                    help='Experiment ID to load model config from (overrides config.py)')

parser.add_argument('--dataset-csv-path', type=str, default=None,
                    help='Path to evaluation dataset CSV (overrides config.py)')

parser.add_argument('--batch-size', type=int, default=None,
                    help='Batch size for inference (overrides config.py)')

parser.add_argument('--device-id', type=int, default=None,
                    help='GPU device ID (overrides config.py)')

parser.add_argument('--sample-size', type=int, default=5000,
                    help='Number of samples to process (overrides config.py)')

parser.add_argument('--first-period-truncation', type=lambda x: (str(x).lower() == 'true'), default=None,
                    help='Truncate generated text at first period (True/False, overrides generator config)')
args = parser.parse_args()

# --- CRITICAL: SET GPU DEVICE BEFORE TORCH IS IMPORTED ---
# Import config FIRST to get DEVICE_ID, then set CUDA_VISIBLE_DEVICES BEFORE torch
from config import DEVICE_ID as _CONFIG_DEVICE_ID
from config import ACTIVATIONS_BASE_DIR as _CONFIG_ACTIVATIONS_BASE_DIR

# Use device from command line if provided, otherwise use config
_DEVICE_ID = args.device_id if args.device_id is not None else _CONFIG_DEVICE_ID
os.environ["CUDA_VISIBLE_DEVICES"] = str(_DEVICE_ID)
# --- END CRITICAL SECTION ---

# Now safe to import torch and related libraries
import glob
import warnings
from transformer_lens import HookedTransformer
import torch.nn.functional as F
import torch
import numpy as np
import pickle
import gc
import datetime
import time
import pandas as pd
from helpers.eval_model import (
    test_evaluator_connectivity,
    judge_answer,
    batch_judge_answers,
    EVAL_MODEL,
    client
)
from helpers.activation_utils import generate_and_capture_efficiently, find_first_period_token_position
from helpers.model_manager import ModelManager
from helpers.token_manager import TokenManager
from helpers.custom_scoring import attach_custom_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import *

# Import the consolidated logger (needed for package installation checks)
from logger import consolidated_logger as logger

# ================================================================
# LOAD GENERATOR CONFIGURATION
# ================================================================
# Function to load model configuration saved by generator.py

def load_generator_config(experiment_id, activations_base_dir):
    """
    Load model configuration from the generator's saved config_metadata.json
    
    Args:
        experiment_id (str): Experiment ID to load config from
        activations_base_dir (str): Base directory for activations
        
    Returns:
        dict: Configuration dictionary with model settings, or None if not found
    """
    config_metadata_path = os.path.join(
        activations_base_dir, experiment_id, 'config_metadata.json')
    
    if not os.path.exists(config_metadata_path):
        logger.warning(f"Generator config not found at: {config_metadata_path}")
        logger.warning("Will use settings from command-line or config.py")
        return None
    
    try:
        with open(config_metadata_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded generator configuration from: {config_metadata_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load generator config: {e}")
        return None

# ================================================================
# PACKAGE INSTALLATION AND IMPORTS
# ================================================================
# Install required packages if not available in the Colab environment


def install_package(package):
    """
    Install a Python package using pip
    Args:
        package (str): Name of the package to install
    """
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Check for required packages
try:
    import sklearn
except ImportError:
    logger.info("Installing scikit-learn...")
    install_package("scikit-learn")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    logger.info("Installing transformers...")
    install_package("transformers")

logger.info("All packages imported successfully!")

# Core imports
warnings.filterwarnings('ignore')


# Import TokenManager for centralized token handling

# Import ModelManager for centralized model management

# Import the new centralized activation utility

# Import evaluation functions from eval_model


# ================================================================
# MEMORY MANAGEMENT FUNCTIONS
# ================================================================

def clear_activations_memory(activations=None, extractor=None, batch_activations=None):
    """
    Clear activation memory to free up GPU/CPU memory after classifier inference

    Args:
        activations: Single activation tensor to clear
        extractor: ActivationExtractor instance to clear its internal storage
        batch_activations: Batch activation dictionary to clear
    """
    if not ENABLE_MEMORY_CLEARING:
        return

    logger.debug("Clearing activation memory...")

    # Clear single activation tensor
    if activations is not None:
        if hasattr(activations, 'cpu'):
            activations = activations.cpu()
        del activations
        activations = None

    # Clear extractor's internal activation storage
    if extractor is not None and hasattr(extractor, 'activations'):
        extractor.activations.clear()
        logger.debug("Cleared extractor.activations")

    # Clear batch activations dictionary
    if batch_activations is not None:
        batch_activations.clear()
        logger.debug("Cleared batch_activations")

    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache")


def clear_gpu_memory():
    """
    Clear GPU memory to prevent accumulation during long inference runs
    """
    if not ENABLE_MEMORY_CLEARING:
        return

    logger.debug("Clearing GPU memory...")

    # Force Python garbage collection
    gc.collect()

    # Clear CUDA cache if available (like llama_aistack2.py)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Report memory state (after CUDA_VISIBLE_DEVICES, visible GPU appears as cuda:0)
        visible_device_id = 0
        allocated = torch.cuda.memory_allocated(visible_device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(visible_device_id) / 1024**3
        logger.debug(
            f"  GPU Memory after clearing: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# ================================================================
# CONFIGURATION
# ================================================================

# Import default settings from config.py
from config import (
    DEVICE_ID as _CONFIG_DEVICE_ID,
    DATASET_CSV_PATH as _CONFIG_DATASET_CSV_PATH,
    BATCH_SIZE as _CONFIG_BATCH_SIZE,
    SAMPLE_SIZE as _CONFIG_SAMPLE_SIZE,
    CLASSIFIER_BASE_DIR,
    MAX_ANSWER_TOKENS,
    MAX_EVAL_WORKERS,
    SAVE_RESULTS,
    TEMPERATURE,
    ENABLE_MEMORY_CLEARING,
    SPECIFIC_CLASSIFIER_RUN_ID,
    HANDLE_IMBALANCE,
    DEBUG_VERBOSE,
    ACTIVATIONS_BASE_DIR as _CONFIG_ACTIVATIONS_BASE_DIR,
    MODEL_NAME as _CONFIG_MODEL_NAME,
    HUGGINGFACE_MODEL_ID as _CONFIG_HUGGINGFACE_MODEL_ID,
    TRANSFORMER_LENS_MODEL_NAME as _CONFIG_TRANSFORMER_LENS_MODEL_NAME,
)

# Initialize with defaults (but NOT model config - that MUST come from generator)
DEVICE_ID = _DEVICE_ID
DATASET_CSV_PATH = _CONFIG_DATASET_CSV_PATH
BATCH_SIZE = None  # MUST be loaded from generator config or command line
SAMPLE_SIZE = _CONFIG_SAMPLE_SIZE
ACTIVATIONS_BASE_DIR = _CONFIG_ACTIVATIONS_BASE_DIR
MODEL_NAME = None  # MUST be loaded from generator config
HUGGINGFACE_MODEL_ID = None  # MUST be loaded from generator config
TRANSFORMER_LENS_MODEL_NAME = None  # MUST be loaded from generator config
EXPERIMENT_ID = None  # Will be set from args or config
FIRST_PERIOD_TRUNCATION = None  # MUST be loaded from generator config
TRANSFORMER_LENS_MODEL_NAME = None  # MUST be loaded from generator config
EXPERIMENT_ID = None  # Will be set from args or config

# Try to load generator configuration if experiment_id is provided
if args.experiment_id:
    EXPERIMENT_ID = args.experiment_id
    logger.info(f"Loading generator config from experiment: {args.experiment_id}")
    generator_config = load_generator_config(args.experiment_id, _CONFIG_ACTIVATIONS_BASE_DIR)
    
    if generator_config:
        # Load model configuration from generator (REQUIRED - must not be None)
        MODEL_NAME = generator_config.get('model_name', None)
        HUGGINGFACE_MODEL_ID = generator_config.get('huggingface_model_id', None)
        # TRANSFORMER_LENS_MODEL_NAME is the same as HUGGINGFACE_MODEL_ID
        TRANSFORMER_LENS_MODEL_NAME = HUGGINGFACE_MODEL_ID
        
        # Load FIRST_PERIOD_TRUNCATION from generator config (CRITICAL for consistency)
        # Default to False if not found in generator config
        FIRST_PERIOD_TRUNCATION = generator_config.get('first_period_truncation', False)
        logger.info(f"Using FIRST_PERIOD_TRUNCATION from generator config: {FIRST_PERIOD_TRUNCATION}")
        
        # Verify model config was found
        if not MODEL_NAME or not HUGGINGFACE_MODEL_ID:
            logger.error("ERROR: Generator config is incomplete - missing model configuration!")
            logger.error(f"  MODEL_NAME: {MODEL_NAME}")
            logger.error(f"  HUGGINGFACE_MODEL_ID: {HUGGINGFACE_MODEL_ID}")
            sys.exit(1)
        
        # Use batch size from generator config if not overridden by command line
        if args.batch_size is None:
            BATCH_SIZE = generator_config.get('batch_size')
            if BATCH_SIZE is None:
                logger.error("ERROR: batch_size not found in generator config!")
                logger.error("Please specify --batch-size via command line")
                sys.exit(1)
            logger.info(f"Using batch size from generator config: {BATCH_SIZE}")
        
        logger.info("✓ Successfully loaded model configuration from generator config_metadata.json")
    else:
        logger.error("ERROR: Could not load generator config from experiment!")
        logger.error(f"Expected location: {_CONFIG_ACTIVATIONS_BASE_DIR}/{args.experiment_id}/config_metadata.json")
        sys.exit(1)
else:
    # If no experiment_id provided, check if there's one in config.py
    from config import EXPERIMENT_ID as _CONFIG_EXPERIMENT_ID
    if _CONFIG_EXPERIMENT_ID:
        EXPERIMENT_ID = _CONFIG_EXPERIMENT_ID
        logger.info(f"Using experiment ID from config.py: {EXPERIMENT_ID}")
        logger.info(f"Loading generator config from experiment: {EXPERIMENT_ID}")
        generator_config = load_generator_config(EXPERIMENT_ID, _CONFIG_ACTIVATIONS_BASE_DIR)
        
        if generator_config:
            # Load model configuration from generator (REQUIRED - must not be None)
            MODEL_NAME = generator_config.get('model_name', None)
            HUGGINGFACE_MODEL_ID = generator_config.get('huggingface_model_id', None)
            # TRANSFORMER_LENS_MODEL_NAME is the same as HUGGINGFACE_MODEL_ID
            TRANSFORMER_LENS_MODEL_NAME = HUGGINGFACE_MODEL_ID
            
            # Load FIRST_PERIOD_TRUNCATION from generator config (CRITICAL for consistency)
            # Default to False if not found in generator config
            FIRST_PERIOD_TRUNCATION = generator_config.get('first_period_truncation', False)
            logger.info(f"Using FIRST_PERIOD_TRUNCATION from generator config: {FIRST_PERIOD_TRUNCATION}")
            
            # Verify model config was found
            if not MODEL_NAME or not HUGGINGFACE_MODEL_ID:
                logger.error("ERROR: Generator config is incomplete - missing model configuration!")
                logger.error(f"  MODEL_NAME: {MODEL_NAME}")
                logger.error(f"  HUGGINGFACE_MODEL_ID: {HUGGINGFACE_MODEL_ID}")
                sys.exit(1)
            
            # Use batch size from generator config if not overridden by command line
            if args.batch_size is None:
                BATCH_SIZE = generator_config.get('batch_size')
                if BATCH_SIZE is None:
                    logger.error("ERROR: batch_size not found in generator config!")
                    logger.error("Please specify --batch-size via command line")
                    sys.exit(1)
                logger.info(f"Using batch size from generator config: {BATCH_SIZE}")
            
            logger.info("✓ Successfully loaded model configuration from generator config_metadata.json")
        else:
            logger.error("ERROR: Could not load generator config from experiment!")
            logger.error(f"Expected location: {_CONFIG_ACTIVATIONS_BASE_DIR}/{EXPERIMENT_ID}/config_metadata.json")
            sys.exit(1)
    else:
        logger.error("ERROR: --experiment-id argument is required!")
        logger.error("Usage: python evaluate.py --experiment-id <experiment_name> [--dataset-csv-path <path>] [--batch-size <size>]")
        sys.exit(1)

# Override with command-line arguments if provided
if args.dataset_csv_path:
    DATASET_CSV_PATH = args.dataset_csv_path
    logger.info(f"Using dataset path from CLI: {DATASET_CSV_PATH}")

if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
    logger.info(f"Using batch size from CLI: {BATCH_SIZE}")

if args.sample_size is not None:
    SAMPLE_SIZE = args.sample_size
    logger.info(f"Using sample size from CLI: {SAMPLE_SIZE}")

if args.first_period_truncation is not None:
    FIRST_PERIOD_TRUNCATION = args.first_period_truncation
    logger.info(f"FIRST_PERIOD_TRUNCATION overridden via CLI: {FIRST_PERIOD_TRUNCATION}")


# Device configuration - handled by ModelManager (same as generator.py)
# After CUDA_VISIBLE_DEVICES is set by ModelManager, the visible GPU appears as cuda:0
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Classifier configuration
# These values will be automatically updated from the loaded classifier
# No need to hardcode them anymore - they'll be set dynamically
EXPECTED_CONFIG = {
    'layer': None,              # Will be set from loaded classifier
    'hook': None,               # Will be set from loaded classifier
    'scheme': None              # Will be set from loaded classifier
}

# Display configuration for verification
logger.info("Configuration (merged from generator config + CLI args + config.py):")
logger.info(f"  Device: {device}")
logger.info(f"  Physical GPU ID: {DEVICE_ID}")
logger.info(
    f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')} (set by ModelManager)")
logger.info(f"  *** Model name: {MODEL_NAME} (from generator config_metadata.json)")
logger.info(f"  *** HuggingFace model ID: {HUGGINGFACE_MODEL_ID} (from generator config_metadata.json)")
logger.info(f"  *** Transformer-Lens model name: {TRANSFORMER_LENS_MODEL_NAME} (derived from HuggingFace model ID)")
logger.info(f"  Classifier base dir: {CLASSIFIER_BASE_DIR}")
logger.info(f"  Evaluation dataset: {DATASET_CSV_PATH}")
logger.info(f"  Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Entire dataset'}")
logger.info(f"  Batch size: {BATCH_SIZE}")
logger.info(f"  Max answer tokens: {MAX_ANSWER_TOKENS}")
logger.info(f"  GPT-3.5 evaluation: enabled (always)")
logger.info(f"  Save results: {SAVE_RESULTS}")
logger.info(f"  Debug verbose: {DEBUG_VERBOSE}")
logger.info(f"  First period truncation: {FIRST_PERIOD_TRUNCATION}")
logger.info(f"  Handle imbalance: {HANDLE_IMBALANCE}")
if SPECIFIC_CLASSIFIER_RUN_ID:
    logger.info(f"  Target classifier run: {SPECIFIC_CLASSIFIER_RUN_ID}")
else:
    logger.info(f"  Classifier run: Auto-discovery (latest)")

logger.debug(f"  Expected classifier config: {EXPECTED_CONFIG}")

# ================================================================
# MODEL LOADING VIA MODEL_MANAGER
# ================================================================
# Model loading is now handled by ModelManager for consistency with generator.py

# ================================================================
# DATASET LOADING FUNCTIONS
# ================================================================


def load_dataset():
    """
    Load and prepare the question-answering dataset from the CSV file

    Returns:
        pd.DataFrame: Dataset with columns: knowledge, question, right_answer

    Expected CSV format:
        - knowledge: The passage/knowledge text
        - question: The question to be answered
        - right_answer: The correct answer (for evaluation)
    """
    logger.info(f"Loading dataset from: {DATASET_CSV_PATH}")

    try:
        df = pd.read_csv(DATASET_CSV_PATH)
        logger.info(f"Loaded {len(df)} samples from dataset")

        # Check required columns
        required_columns = ['knowledge', 'question', 'right_answer']
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            return None

        # Basic data validation
        logger.info("Dataset statistics:")
        logger.info(f"  - Total samples: {len(df)}")
        logger.info(f"  - Non-null knowledge: {df['knowledge'].notna().sum()}")
        logger.info(f"  - Non-null questions: {df['question'].notna().sum()}")
        logger.info(
            f"  - Non-null answers: {df['right_answer'].notna().sum()}")

        # Remove samples with missing data
        original_length = len(df)
        df = df.dropna(subset=['knowledge', 'question', 'right_answer'])
        if len(df) < original_length:
            logger.info(
                f"Removed {original_length - len(df)} samples with missing data")

        # Sample dataset if requested
        if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
            logger.info(
                f"Sampling {SAMPLE_SIZE} examples from {len(df)} total")
            df = df.sample(
                n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        elif SAMPLE_SIZE is None:
            logger.info(f"Processing entire dataset: {len(df)} samples")

        logger.info(f"Final dataset size: {len(df)} samples")
        return df

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {DATASET_CSV_PATH}")
        logger.error(
            "Please ensure the evaluation dataset file exists at the specified path")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def create_qa_prompt(knowledge, question):
    """
    Create QA prompt in the same format as squad_generator3 halueval2.py

    Args:
        knowledge (str): The context/knowledge text
        question (str): The question to be answered

    Returns:
        str: Formatted prompt for Llama generation

    Format matches squad_generator3 halueval2.py:
    "Context:\n{knowledge}\n\nQuestion: {question}\nAnswer:"
    """
    return f"Context:\n{knowledge}\n\nQuestion: {question}\nAnswer:"

# ================================================================
# GPT-3.5 EVALUATION FUNCTIONS
# ================================================================

# Azure OpenAI client is now imported from eval_model.py (same as generator.py)
# This ensures consistency and centralized configuration

# judge_answer function is now imported from eval_model.py (same as generator.py)
# This ensures consistency and centralized evaluation logic


def batch_evaluate_answers(results, client):
    """
    Evaluate all generated answers in parallel using GPT-3.5
    Uses the centralized batch_judge_answers from eval_model.py (same as generator.py)

    Args:
        results (list): List of inference results
        client: Azure OpenAI client

    Returns:
        list: Updated results with GPT-3.5 evaluation scores
    """
    logger.info(f"Evaluating {len(results)} answers with GPT-3.5...")

    # Prepare evaluation pairs for batch_judge_answers (same format as generator.py)
    evaluation_pairs = []
    for result in results:
        gt_answer = result['correct_answer']
        generated_answer = result['generated_answer']
        evaluation_pairs.append((gt_answer, generated_answer, result))

    # Use centralized batch_judge_answers function from eval_model.py
    batch_results = batch_judge_answers(
        evaluation_pairs, max_workers=MAX_EVAL_WORKERS)

    # Process results and update original results list
    for i, (evaluation_score, original_result) in enumerate(batch_results):
        # evaluation_score: 1 = hallucination, 0 = non-hallucination, 2 = API failure
        results[i]['gpt35_evaluation'] = evaluation_score
        # 🔧 FIX: DON'T overwrite 'is_hallucination' - that's the CLASSIFIER's prediction!
        # Store GPT-3.5 label separately for ground truth comparison
        results[i]['gpt35_is_hallucination'] = evaluation_score
        if evaluation_score in (0, 1):
            results[i]['is_correct'] = 1 - evaluation_score
        else:
            results[i]['is_correct'] = 2

    # Sort back to original order
    results.sort(key=lambda x: x['sample_idx'])

    return results

# ================================================================
# MODEL LOADING FUNCTIONS
# ================================================================
# These functions handle loading the trained classifier and Llama model
# They include extensive error checking and validation


def find_classifier_run(classifier_base_dir):
    """
    Find the appropriate classifier run directory

    Args:
        classifier_base_dir (str): Base directory containing classifier runs

    Returns:
        str: Path to the classifier run directory, or None if not found

    Expected directory structure:
        classifier_base_dir/
        +-- classifier_run_1751729566/
        |   +-- classifier_run_metadata.json
        |   +-- best_classifier_L11_attn.hook_z_last_generated.pkl
        |   +-- ...
        +-- classifier_run_1752475413/
        |   +-- ...
        +-- ...
    """

    # If specific run ID is provided, try to use it first
    if SPECIFIC_CLASSIFIER_RUN_ID:
        specific_path = os.path.join(
            classifier_base_dir, SPECIFIC_CLASSIFIER_RUN_ID)
        if os.path.isdir(specific_path):
            logger.info(
                f"Found specific classifier run: {SPECIFIC_CLASSIFIER_RUN_ID}")

            # Load and display run metadata if available
            metadata_path = os.path.join(
                specific_path, 'classifier_run_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(
                    f"  Created: {metadata.get('start_time', 'Unknown')}")
                logger.info(
                    f"  Best AUROC: {metadata.get('best_auroc', 'Unknown')}")
                logger.info(
                    f"  Total experiments: {metadata.get('total_experiments', 'Unknown')}")
            else:
                logger.info(f"  No metadata found for specific run")

            return specific_path
        else:
            logger.warning(
                f"Specific classifier run '{SPECIFIC_CLASSIFIER_RUN_ID}' not found at {specific_path}")
            logger.info("Available runs:")
            # List available directories for debugging
            if os.path.exists(classifier_base_dir):
                available_dirs = [d for d in os.listdir(classifier_base_dir)
                                  if os.path.isdir(os.path.join(classifier_base_dir, d)) and d.startswith('classifier_run_')]
                for run_dir in sorted(available_dirs):
                    logger.info(f"  - {run_dir}")
            logger.info("Falling back to auto-discovery...")

    # Fallback to finding the latest run by modification time
    logger.info("Searching for latest classifier run...")

    # Look for directories matching the pattern classifier_run_*
    run_pattern = os.path.join(classifier_base_dir, "classifier_run_*")
    run_dirs = glob.glob(run_pattern)

    if not run_dirs:
        logger.error(f"No classifier runs found in {classifier_base_dir}")
        return None

    # Filter to only directories (not files)
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]

    if not run_dirs:
        logger.error(f"No valid classifier run directories found")
        return None

    # Find the most recent run by file modification time
    latest_run = max(run_dirs, key=os.path.getmtime)
    run_id = os.path.basename(latest_run)

    # Load and display run metadata if available
    metadata_path = os.path.join(latest_run, 'classifier_run_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Found latest classifier run: {run_id}")
        logger.info(f"  Created: {metadata.get('start_time', 'Unknown')}")
        logger.info(f"  Best AUROC: {metadata.get('best_auroc', 'Unknown')}")
        logger.info(
            f"  Total experiments: {metadata.get('total_experiments', 'Unknown')}")
    else:
        logger.info(f"Found latest classifier run: {run_id} (no metadata)")

    return latest_run


def find_classifier_run_in_experiment(experiment_dir):
    """
    NEW: Find classifier run directory inside the experiment directory (NEW STRUCTURE).

    Classifiers are now saved in: ACTIVATIONS_BASE_DIR/EXPERIMENT_ID/classifier_run_ID/
    This function looks for them in that location.

    Args:
        experiment_dir (str): Path to the experiment directory (ACTIVATIONS_BASE_DIR/EXPERIMENT_ID/)

    Returns:
        str: Path to the classifier run directory, or None if not found
    """
    logger.info(
        f"Searching for classifier runs in experiment directory: {experiment_dir}")

    if not os.path.exists(experiment_dir):
        logger.debug(f"Experiment directory does not exist: {experiment_dir}")
        return None

    # Look for directories matching the pattern classifier_run_* inside the experiment directory
    run_dirs = [d for d in os.listdir(experiment_dir)
                if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith('classifier_run_')]

    if not run_dirs:
        logger.debug(
            f"No classifier_run_* directories found in {experiment_dir}")
        return None

    # Convert to full paths
    run_dirs = [os.path.join(experiment_dir, d) for d in run_dirs]

    # Find the most recent run by file modification time
    latest_run = max(run_dirs, key=os.path.getmtime)
    run_id = os.path.basename(latest_run)

    logger.info(f"Found classifier run in experiment directory: {run_id}")
    logger.info(f"Path: {latest_run}")

    return latest_run


def find_best_saved_classifier(classifier_run_dir):
    """
    Find the best saved classifier file in the run directory

    Args:
        classifier_run_dir (str): Path to the classifier run directory

    Returns:
        str: Path to the best classifier file, or None if not found

    Expected filename pattern: best_classifier_{RUN_ID}.pkl
    Example: best_classifier_classifier_run_testing_1755246841.pkl
    """

    # Look for best classifier files matching nocheckpoint_nodataleak.py pattern
    pattern = os.path.join(classifier_run_dir, "best_classifier_*.pkl")
    classifier_files = glob.glob(pattern)

    # Debug: Show what files are actually in the directory
    all_files = os.listdir(classifier_run_dir)
    logger.info(f"Files in {classifier_run_dir}:")
    for file in all_files:
        if file.endswith('.pkl'):
            logger.info(f"  - {file}")

    if not classifier_files:
        logger.error(f"No best classifier files found in {classifier_run_dir}")
        logger.error("Expected pattern: best_classifier_*.pkl")
        logger.error(
            "Please run nocheckpoint_nodataleak.py first to train and save a classifier.")
        return None

    # If multiple files exist, take the most recent (shouldn't happen normally)
    latest_classifier = max(classifier_files, key=os.path.getmtime)
    logger.info(
        f"Found best classifier: {os.path.basename(latest_classifier)}")
    return latest_classifier


def load_saved_classifier(classifier_path):
    """
    Load the saved classifier package from nocheckpoint_nodataleak.py

    Args:
        classifier_path (str): Path to the saved classifier pickle file

    Returns:
        dict: Classifier package containing model, scaler, and metadata
        None: If the classifier is invalid or incomplete

    Expected package structure:
    {
        'model': trained_sklearn_model,      # The actual trained classifier
        'scaler': fitted_scaler,             # StandardScaler fitted on training data
        'metadata': {
            'layer': int,                    # Layer number (e.g., 11)
            'hook': str,                     # Hook name (e.g., 'attn.hook_z')
            'scheme': str,                   # Activation scheme (e.g., 'last_generated')
            'classifier_name': str,          # Classifier type (e.g., 'BalancedSVM')
            'performance': {...},            # Training performance metrics
            'training_info': {...},          # Training configuration
            'labeling_system': {...}         # Label interpretation
        }
    }
    """
    with open(classifier_path, 'rb') as f:
        classifier_package = pickle.load(f)

    logger.info("Loaded classifier components:")

    # Check if this is a metadata-only package (indicates training failure)
    if classifier_package['model'] is None or classifier_package['scaler'] is None:
        logger.error("This is a metadata-only classifier package!")
        metadata = classifier_package['metadata']
        logger.warning(
            f"{metadata.get('warning', 'Model and scaler not available')}")
        logger.warning(
            f"{metadata.get('issue', 'Unknown issue during training')}")
        logger.info(f"  Layer: {metadata['layer']}")
        logger.info(f"  Hook: {metadata['hook']}")
        logger.info(f"  Scheme: {metadata['scheme']}")
        logger.info(f"  Classifier: {metadata['classifier_name']}")
        logger.info(f"  AUROC: {metadata['performance']['auroc']:.3f}")

        # ADDED: Explicitly log the optimal threshold and key hyperparameters for verification
        if 'optimal_threshold' in metadata:
            logger.info(
                f"  Optimal Threshold: {metadata['optimal_threshold']:.4f}")
        else:
            logger.warning("  Optimal Threshold: Not found, defaulting to 0.5")

        if 'hyperparameters' in metadata and metadata['hyperparameters'] is not None:
            hp = metadata['hyperparameters']
            n_est = hp.get('n_estimators', 'N/A')
            # Ensure n_est is int if it exists
            if n_est != 'N/A':
                n_est = int(n_est)
            lr = hp.get('learning_rate', 'N/A')
            max_d = hp.get('max_depth', 'N/A')
            if max_d != 'N/A':
                max_d = int(max_d)

            # Format learning rate if it's a float
            lr_str = f"{lr:.4f}" if isinstance(lr, float) else str(lr)

            logger.info(
                f"  Hyperparameters: n_est={n_est}, lr={lr_str}, max_d={max_d}")
        else:
            logger.info("  Hyperparameters: Not found in metadata.")

        logger.info(
            "SOLUTION: Re-run nocheckpoint_nodataleak.py to train and save a complete classifier")
        return None

    # Display classifier information
    logger.info(f"  Model: {type(classifier_package['model']).__name__}")
    logger.info(f"  Scaler: {type(classifier_package['scaler']).__name__}")

    metadata = classifier_package['metadata']
    logger.info(f"  Layer: {metadata['layer']}")
    logger.info(f"  Hook: {metadata['hook']}")
    logger.info(f"  Scheme: {metadata['scheme']}")
    logger.info(
        f"  Classifier: {metadata.get('classifier_name', metadata.get('classifier', 'Unknown'))}")

    # Handle missing performance metrics gracefully
    performance = metadata.get('performance', {})
    auroc = performance.get('auroc', 'N/A')
    accuracy = performance.get('accuracy', 'N/A')
    n_train = performance.get('n_train', 'N/A')

    if isinstance(auroc, (int, float)):
        logger.info(f"  AUROC: {auroc:.3f}")
    else:
        logger.info(f"  AUROC: {auroc}")

    if isinstance(accuracy, (int, float)):
        logger.info(f"  Accuracy: {accuracy:.3f}")
    else:
        logger.info(f"  Accuracy: {accuracy}")

    logger.info(f"  Training samples: {n_train}")

    # Update EXPECTED_CONFIG with the actual values from the loaded classifier
    # This ensures compatibility regardless of what was used during training
    global EXPECTED_CONFIG
    EXPECTED_CONFIG.update({
        'layer': metadata['layer'],
        'hook': metadata['hook'],
        'scheme': metadata['scheme']
    })

    logger.info("CONFIG: Updated EXPECTED_CONFIG to match loaded classifier:")
    logger.info(f"  Layer: {EXPECTED_CONFIG['layer']}")
    logger.info(f"  Hook: {EXPECTED_CONFIG['hook']}")
    logger.info(f"  Scheme: {EXPECTED_CONFIG['scheme']}")

    return classifier_package


# ================================================================
# CONFIG AUDIT TRAIL FUNCTIONS
# ================================================================
# These functions create a complete configuration audit trail from
# generator -> classifier -> evaluate, enabling full reproducibility

def load_classifier_config(classifier_dir):
    """
    Load classifier configuration metadata from the classifier directory

    Args:
        classifier_dir (str): Path to the classifier_run_* directory

    Returns:
        dict: Classifier configuration metadata
        None: If classifier_config_metadata.json not found
    """
    config_path = os.path.join(
        classifier_dir, 'classifier_config_metadata.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(
                f"Successfully loaded classifier config from: {os.path.basename(classifier_dir)}")
            return config
        else:
            logger.warning(f"Classifier config not found at: {config_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading classifier config: {e}")
        return None


def load_all_configs(activations_dir, experiment_id, classifier_run_id):
    """
    Load all configurations from generator, classifier, and current state

    Args:
        activations_dir (str): ACTIVATIONS_BASE_DIR
        experiment_id (str): Experiment identifier
        classifier_run_id (str): Classifier run identifier

    Returns:
        dict: Combined configuration with keys:
            - 'generator_config': Configuration used during generation
            - 'classifier_config': Configuration used during classifier training
            - 'evaluation_config': Current evaluation configuration
    """
    configs = {
        'generator_config': None,
        'classifier_config': None,
        'evaluation_config': None
    }

    # Load generator config from chunk directory
    experiment_dir = os.path.join(activations_dir, experiment_id)
    chunk_dir = os.path.join(experiment_dir, 'chunk_0')

    if os.path.exists(chunk_dir):
        generator_config_path = os.path.join(chunk_dir, 'run_metadata.json')
        try:
            if os.path.exists(generator_config_path):
                with open(generator_config_path, 'r') as f:
                    configs['generator_config'] = json.load(f)
                logger.info(
                    "Successfully loaded generator config from chunk_0")
            else:
                logger.warning(
                    f"Generator config not found at: {generator_config_path}")
        except Exception as e:
            logger.error(f"Error loading generator config: {e}")
    else:
        logger.warning(f"Chunk directory not found: {chunk_dir}")

    # Load classifier config from classifier directory
    classifier_dir = os.path.join(experiment_dir, classifier_run_id)
    if os.path.exists(classifier_dir):
        classifier_config = load_classifier_config(classifier_dir)
        configs['classifier_config'] = classifier_config
    else:
        logger.warning(f"Classifier directory not found: {classifier_dir}")

    # Build evaluation config from current config.py settings
    # Derive dataset name and size from the configured CSV path
    dataset_name = None
    dataset_size = None
    try:
        if DATASET_CSV_PATH:
            dataset_name = os.path.basename(DATASET_CSV_PATH)
            if os.path.exists(DATASET_CSV_PATH):
                # Prefer pandas for an accurate count, fall back to line count
                try:
                    df_tmp = pd.read_csv(DATASET_CSV_PATH)
                    dataset_size = len(df_tmp)
                except Exception:
                    # Fallback: count lines and subtract one for header if present
                    try:
                        with open(DATASET_CSV_PATH, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f)
                        dataset_size = max(0, line_count - 1)
                    except Exception:
                        dataset_size = None
    except Exception:
        dataset_name = None
        dataset_size = None

    configs['evaluation_config'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_name': MODEL_NAME,
        'dataset_name': dataset_name,
        'dataset_size': dataset_size,
        'sample_size': SAMPLE_SIZE,
        'device_id': DEVICE_ID,
        'layer': EXPECTED_CONFIG.get('layer'),
        'hook': EXPECTED_CONFIG.get('hook'),
        'scheme': EXPECTED_CONFIG.get('scheme')
    }

    return configs


def save_audit_metadata(evaluation_dir, all_configs):
    """
    Save comprehensive evaluation metadata including all configurations

    Args:
        evaluation_dir (str): Path to the evaluation_run_* directory
        all_configs (dict): Dictionary containing generator_config, classifier_config, 
                           and evaluation_config from load_all_configs()
    """
    metadata = {
        'evaluation_metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'evaluation_run_id': os.path.basename(evaluation_dir),
            'experiment_id': EXPERIMENT_ID
        },
        'generator_config': all_configs.get('generator_config', {}),
        'classifier_config': all_configs.get('classifier_config', {}),
        'evaluation_config': all_configs.get('evaluation_config', {})
    }

    metadata_path = os.path.join(evaluation_dir, 'evaluation_metadata.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved evaluation metadata to: {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving evaluation metadata: {e}")
        return False

# Model loading is now handled by ModelManager (removed old load_llama_model function)

# ================================================================
# DATA PREPROCESSING FUNCTIONS
# ================================================================
# These functions handle text processing and must exactly match the training setup

# ================================================================
# FALLBACK TOKEN FUNCTIONS (for compatibility)
# ================================================================


def make_tokens_optimized_fallback(model, knowledge, question):
    """
    Fallback tokenization function - simplified version for compatibility
    """
    try:
        prompt = f"Context:\n{knowledge}\n\nQuestion: {question}\nAnswer:"
        return model.to_tokens(prompt)
    except Exception as e:
        logger.error(f"Fallback tokenization failed: {e}")
        return torch.tensor([])


# ================================================================
# INFERENCE FUNCTIONS
# ================================================================
# These functions handle the prediction using the loaded classifier


def predict_hallucination(activations, classifier_package):
    """
    Predict if the generation is a hallucination using the classifier from nocheckpoint_nodataleak.py

    Args:
        activations (np.ndarray): 1D numpy array of activations from the classifier's token scheme
        classifier_package (dict): Dictionary containing 'model' (classifier), 'scaler', and 'metadata'

    Returns:
        tuple: (result, activations_scaled, error)
            - result: Dictionary with prediction results
            - activations_scaled: Scaled activations used for prediction
            - error: Error message if any

    CRITICAL: This function must use the exact same preprocessing as the training script.
    The classifier expects a 1D feature vector that has been scaled using the same scaler
    that was fitted during training.

    The labeling system is:
    - Label 1 = HALLUCINATION (incorrect answer)
    - Label 0 = NO HALLUCINATION (correct answer)
    """

    # Check if activations are available
    if activations is None:
        logger.debug(f"  predict_hallucination: No activations available")
        return None, None, "No activations available"

    # Get model components from the classifier package
    classifier = classifier_package['model']  # The trained sklearn classifier
    # The fitted StandardScaler from training
    scaler = classifier_package['scaler']
    metadata = classifier_package.get('metadata', {})
    # Use saved threshold, default to 0.5
    optimal_threshold = metadata.get('optimal_threshold', 0.5)

    # Reshape activations to match training format
    # sklearn expects shape (n_samples, n_features), we have (n_features,)
    activations_reshaped = activations.reshape(1, -1)  # (1, n_features)

    logger.debug(
        f"  predict_hallucination: Original activations shape: {activations.shape}")
    logger.debug(
        f"  predict_hallucination: Reshaped to: {activations_reshaped.shape}")

    # Scale activations using the same scaler fitted during training
    # This is CRITICAL - the classifier was trained on scaled features
    activations_scaled = scaler.transform(activations_reshaped)

    logger.debug(
        f"  predict_hallucination: Scaled shape: {activations_scaled.shape}")
    logger.debug(
        f"  predict_hallucination: Using optimal threshold: {optimal_threshold:.4f}")

    # Make prediction using the trained classifier's probabilities and the optimal threshold
    prediction_proba = classifier.predict_proba(
        activations_scaled)[0]  # Probability for each class

    # Verify the classifier output is as expected
    if len(prediction_proba) != 2:
        return None, None, f"Expected 2 classes, got {len(prediction_proba)}"

    # Interpret results based on the labeling system from classifier.py:
    # Label 1 = HALLUCINATION (incorrect answer)
    # Label 0 = NO HALLUCINATION (correct answer)
    hallucination_prob = prediction_proba[1]
    prediction = 1 if hallucination_prob >= optimal_threshold else 0

    is_hallucination = (prediction == 1)
    # Probability of class 0 (normal/correct)
    normal_prob = prediction_proba[0]
    confidence = max(prediction_proba)  # Confidence is the highest probability

    # Sanity check: probabilities should sum to 1.0
    prob_sum = hallucination_prob + normal_prob
    if abs(prob_sum - 1.0) > 0.01:
        return None, None, f"Probabilities don't sum to 1: {prob_sum}"

    # Package the results for easy interpretation
    result = {
        # Boolean: True if hallucination detected
        'is_hallucination': is_hallucination,
        # Float: Probability of hallucination (0-1)
        'hallucination_probability': hallucination_prob,
        # Float: Probability of normal text (0-1)
        'normal_probability': normal_prob,
        # Float: Classifier confidence (0-1)
        'confidence': confidence,
        # Int: Raw prediction (0 or 1)
        'prediction': prediction,
        # Array: All class probabilities
        'probabilities': prediction_proba,
        # Int: Raw prediction as integer
        'raw_prediction': int(prediction),
        # String: Human-readable interpretation
        'interpretation': 'HALLUCINATION' if is_hallucination else 'NORMAL'
    }

    return result, activations_scaled, None


def run_batch_inference(model, hf_model, metadata, classifier_package, dataset_df, token_manager=None):
    """
    Run batch inference on the dataset using TransformerLens generation (exactly like generator.py)

    Args:
        model: The loaded HookedTransformer model
        hf_model: The loaded HuggingFace model (deprecated - kept for compatibility)
        metadata: Classifier metadata containing layer, hook, and scheme info
        classifier_package: The loaded classifier package
        dataset_df: Dataset DataFrame with knowledge, question, right_answer columns
        token_manager: TokenManager instance for centralized token handling

    Returns:
        list: List of results for each sample
    """
    logger.info("\n" + "="*80)
    logger.info("DATASET HALLUCINATION DETECTION - BATCH PROCESSING")
    logger.info("="*80)
    logger.info(f"Processing {len(dataset_df)} samples from dataset")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("Generation: TransformerLens-only (exactly like generator.py)")
    logger.info("="*80)

    results = []
    total_samples = len(dataset_df)

    # Convert DataFrame to list for batch processing
    sample_list = list(dataset_df.itertuples())

    # Use the HF model that was loaded together with the HookedTransformer model

    # Main batch processing loop using llama_aistack2.py architecture
    for batch_start in tqdm(range(0, len(sample_list), BATCH_SIZE),
                            desc="Processing batches",
                            total=(len(sample_list) + BATCH_SIZE - 1) // BATCH_SIZE):

        batch_end = min(batch_start + BATCH_SIZE, len(sample_list))
        batch_rows = sample_list[batch_start:batch_end]

        # Process this batch using efficient generation
        batch_results = process_batch_efficient(
            model, hf_model, metadata, classifier_package, batch_rows, batch_start, token_manager
        )

        results.extend(batch_results)

        # Clear GPU memory after each batch to prevent accumulation
        clear_gpu_memory()

        # Print batch progress
        logger.progress(
            f"Completed batch {batch_start//BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1)//BATCH_SIZE}")

    return results


def process_batch_efficient(model, hf_model, metadata, classifier_package, batch_rows, batch_start, token_manager=None):
    """
    Process a single batch using llama_aistack2.py efficient generation and capture

    Args:
        model: The HookedTransformer model
        hf_model: The HuggingFace model for efficient generation
        metadata: Classifier metadata containing layer, hook, and scheme info
        classifier_package: The loaded classifier package
        batch_rows: List of DataFrame rows for this batch
        batch_start: Starting index for this batch
        token_manager: TokenManager instance for centralized token handling

    Returns:
        list: Results for this batch
    """
    # Prepare batch of prompts using llama_aistack2.py pattern
    batch_prompts = []
    batch_info = []

    for idx_in_batch, row in enumerate(batch_rows):
        # Tokenize using token_manager for consistency with generator.py
        if token_manager:
            tok_prompt = token_manager.make_tokens_optimized(
                row.knowledge, row.question)
        else:
            # Fallback to manual tokenization if no token_manager
            tok_prompt = make_tokens_optimized_fallback(
                model, row.knowledge, row.question)
        batch_prompts.append(tok_prompt)

        # Create metadata record for this batch item
        batch_item_info = {
            'row_idx': batch_start + idx_in_batch,
            'row': row,
            'prompt_length': tok_prompt.shape[0],
            'padding_length': 0  # Will be updated during padding
        }
        batch_info.append(batch_item_info)

    if batch_prompts:
        # Use TokenManager for LEFT-padding (identical to generator.py)
        batch_tensor, batch_info = token_manager.create_left_padded_batch(
            batch_prompts, batch_info)

        # Use the new centralized activation capture function
        target_layer = metadata['layer']
        target_hook_base_name = metadata['hook']
        target_scheme = metadata['scheme']

        full_sequences, batch_activations, batch_results_list, positions_by_scheme = generate_and_capture_efficiently(
            model=model,
            token_manager=token_manager,
            batch_prompts_padded=batch_tensor,
            batch_info=batch_info,
            active_hooks=[target_hook_base_name],
            token_schemes=[target_scheme],
            start_layer=target_layer,
            end_layer=target_layer,
            max_answer_tokens=MAX_ANSWER_TOKENS,
            model_name=MODEL_NAME,
            logger=logger,
            debug_verbose=DEBUG_VERBOSE,
            first_period_truncation=FIRST_PERIOD_TRUNCATION
        )

        # Process results using the efficient results from generator.py approach
        batch_results = []
        hook_key = f"layer_{metadata['layer']}_{metadata['hook']}"

        for batch_idx, info in enumerate(batch_info):
            row = info['row']

            # Extract generated text from batch_results_list (from efficient generation)
            if batch_idx < len(batch_results_list):
                generated_answer = batch_results_list[batch_idx].get(
                    'gpt_answer_trim', '')
            else:
                generated_answer = ""

            # Get activations for this sample from batch_activations
            sample_activations = None
            if hook_key in batch_activations and metadata['scheme'] in batch_activations[hook_key]:
                activation_list = batch_activations[hook_key][metadata['scheme']]
                for activation_data in activation_list:
                    if activation_data['row_idx'] == info['row_idx']:
                        sample_activations = activation_data['activations']
                        break

            # Make prediction using same logic as original
            logger.debug(
                f"  Sample {batch_idx}: Making hallucination prediction...")
            logger.debug(
                f"  Sample {batch_idx}: Activations shape: {sample_activations.shape if sample_activations is not None else 'None'}")
            logger.debug(
                f"  Sample {batch_idx}: Activations type: {type(sample_activations)}")

            prediction_result, _, error = predict_hallucination(
                sample_activations, classifier_package)

            if DEBUG_VERBOSE:
                if prediction_result:
                    logger.debug(
                        f"  Sample {batch_idx}: Prediction successful: {prediction_result['interpretation']}")
                else:
                    logger.debug(
                        f"  Sample {batch_idx}: Prediction failed: {error}")

            # Clear sample activations after classifier use to free memory
            clear_activations_memory(activations=sample_activations)

            # Package results (same structure as original)
            result = {
                'sample_idx': info['row_idx'],
                'knowledge': row.knowledge,
                'question': row.question,
                'correct_answer': row.right_answer,
                'generated_answer': generated_answer,
                'full_generated_text': batch_results_list[batch_idx].get('gpt_answer', '') if batch_idx < len(batch_results_list) else '',
                'prompt': create_qa_prompt(row.knowledge, row.question),
                'has_activations': sample_activations is not None,
                'prediction_error': error,
            }

            # Add prediction results if successful (same logic as original)
            if prediction_result:
                result.update({
                    'is_hallucination': prediction_result['is_hallucination'],
                    'hallucination_probability': prediction_result['hallucination_probability'],
                    'normal_probability': prediction_result['normal_probability'],
                    'confidence': prediction_result['confidence'],
                    'prediction': prediction_result['prediction'],
                    'interpretation': prediction_result['interpretation']
                })
            else:
                result.update({
                    'is_hallucination': None,
                    'hallucination_probability': None,
                    'normal_probability': None,
                    'confidence': None,
                    'prediction': None,
                    'interpretation': 'PREDICTION_FAILED'
                })

            batch_results.append(result)

        # Clear batch activations after processing all samples in the batch
        clear_activations_memory(batch_activations=batch_activations)

        return batch_results

    return []

# ================================================================
# Custom Scoring Function (now imported from helpers)
# ================================================================
# attach_custom_score is now imported from helpers.custom_scoring
# See: helpers/custom_scoring.py for implementation details


def analyze_classifier_performance(results):
    """
    Analyze classifier performance against GPT-3.5 ground truth

    Args:
        results (list): Results with both classifier predictions and GPT-3.5 evaluations

    Returns:
        dict: Performance metrics including accuracy, precision, recall, F1, AUROC
    """
    if not results:
        return {}

    logger.info("\n" + "="*80)
    logger.info("CLASSIFIER PERFORMANCE ANALYSIS")
    logger.info("="*80)

    # Filter results with valid predictions and evaluations
    valid_results = [
        r for r in results
        if (r.get('is_correct') is not None and
            r.get('is_hallucination') is not None and
            r.get('hallucination_probability') is not None and
            r.get('gpt35_evaluation') != 2)  # Exclude evaluation failures
    ]

    if not valid_results:
        logger.warning("No valid results for performance analysis")
        return {}

    logger.evaluation(
        f"Analyzing {len(valid_results)} samples with valid predictions and evaluations")

    # 🔧 FIX: Use correct field names for comparison
    # Extract ground truth and predictions
    # y_true: GPT-3.5 ground truth (1 = hallucination/incorrect, 0 = correct)
    y_true = [0 if r['is_correct'] else 1 for r in valid_results]
    
    # y_pred: CLASSIFIER predictions (1 = hallucination, 0 = correct)
    # Use 'is_hallucination' which contains the ORIGINAL classifier prediction (not overwritten)
    y_pred = [1 if r['is_hallucination'] else 0 for r in valid_results]
    
    # y_prob: Classifier's hallucination probability scores
    y_prob = [r['hallucination_probability']
              for r in valid_results]  # Probability of hallucination

    # Calculate performance metrics
    performance = {}

    try:
        performance['accuracy'] = accuracy_score(y_true, y_pred)
        performance['precision'] = precision_score(
            y_true, y_pred, zero_division=0)
        performance['recall'] = recall_score(y_true, y_pred, zero_division=0)
        performance['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Per-class metrics
        performance['precision_per_class'] = precision_score(
            y_true, y_pred, average=None, zero_division=0)
        performance['recall_per_class'] = recall_score(
            y_true, y_pred, average=None, zero_division=0)

        # AUROC calculation
        if len(set(y_true)) > 1:  # Need both classes for AUROC
            performance['auroc'] = roc_auc_score(y_true, y_prob)

            # PR-AUROC (AUPRC)
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_true, y_prob)
            performance['pr_auroc'] = auc(recall_vals, precision_vals)
        else:
            performance['auroc'] = None
            performance['pr_auroc'] = None

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {}

    # Confusion matrix analysis (Labels: 0=Correct, 1=Hallucination)
    # Correctly identified as not hallucination
    tn = sum(1 for i in range(len(y_true))
             if y_true[i] == 0 and y_pred[i] == 0)
    # Correctly identified as hallucination
    tp = sum(1 for i in range(len(y_true))
             if y_true[i] == 1 and y_pred[i] == 1)
    # Hallucinations missed (false negative)
    fn = sum(1 for i in range(len(y_true))
             if y_true[i] == 1 and y_pred[i] == 0)
    # Correct answers flagged as hallucinations (false positive)
    fp = sum(1 for i in range(len(y_true))
             if y_true[i] == 0 and y_pred[i] == 1)

    # Store confusion matrix values at TOP LEVEL for custom_scoring.py to access
    performance['true_positives'] = tp
    performance['true_negatives'] = tn
    performance['false_positives'] = fp
    performance['false_negatives'] = fn
    
    # Also store in nested dictionary for backwards compatibility
    performance['confusion_matrix'] = {
        'true_positives': tp,   # Correctly identified hallucinations
        'true_negatives': tn,   # Correctly identified correct answers
        # False alarms (correct answers classified as hallucinations)
        'false_positives': fp,
        # Missed hallucinations (incorrect answers classified as correct)
        'false_negatives': fn
    }

    # Add per-class precision and recall for custom score
    if len(performance.get('precision_per_class', [])) == 2:
        performance['precision_not_hallucinated'] = performance['precision_per_class'][0]
        performance['precision_hallucinated'] = performance['precision_per_class'][1]
    if len(performance.get('recall_per_class', [])) == 2:
        performance['recall_not_hallucinated'] = performance['recall_per_class'][0]
        performance['recall_hallucinated'] = performance['recall_per_class'][1]

    # Display results
    logger.evaluation(f"\nPerformance Metrics:")
    logger.evaluation(f"  Accuracy:  {performance['accuracy']:.3f}")
    logger.evaluation(f"  Precision: {performance['precision']:.3f}")
    logger.evaluation(f"  Recall:    {performance['recall']:.3f}")
    logger.evaluation(f"  F1 Score:  {performance['f1']:.3f}")
    if performance['auroc'] is not None:
        logger.evaluation(f"  AUROC:     {performance['auroc']:.3f}")
    else:
        logger.evaluation(f"  AUROC:     Cannot calculate (single class)")

    if performance.get('pr_auroc') is not None:
        logger.evaluation(f"  PR-AUROC:  {performance['pr_auroc']:.3f}")
    else:
        logger.evaluation(f"  PR-AUROC:  Cannot calculate (single class)")

    logger.evaluation(f"\nConfusion Matrix:")
    logger.evaluation(
        f"  True Positives (Hallucination -> Hallucination): {tp:3d}")
    logger.evaluation(f"  True Negatives (Correct -> Correct): {tn:3d}")
    logger.evaluation(
        f"  False Positives (Correct -> Hallucination):  {fp:3d} [WARNING]")
    logger.evaluation(
        f"  False Negatives (Hallucination -> Correct): {fn:3d} [WARNING]")

    # Calculate rates
    total = len(valid_results)
    # y_true=1 is a hallucination (incorrect answer)
    hallucinated_answers_gpt = sum(y_true)
    correct_answers_gpt = total - hallucinated_answers_gpt

    logger.evaluation(f"\nDetailed Analysis:")
    logger.evaluation(f"  Total samples: {total}")
    logger.evaluation(
        f"  Incorrect answers/Hallucinations (GPT-3.5): {hallucinated_answers_gpt} ({hallucinated_answers_gpt/total*100:.1f}%)")
    logger.evaluation(
        f"  Correct answers (GPT-3.5): {correct_answers_gpt} ({correct_answers_gpt/total*100:.1f}%)")

    if hallucinated_answers_gpt > 0:
        logger.evaluation(
            f"  Hallucinations properly identified: {tp}/{hallucinated_answers_gpt} ({tp/hallucinated_answers_gpt*100:.1f}%)")
        logger.evaluation(
            f"  Hallucinations missed (misclassified as correct): {fn}/{hallucinated_answers_gpt} ({fn/hallucinated_answers_gpt*100:.1f}%)")

    if correct_answers_gpt > 0:
        logger.evaluation(
            f"  Correct answers properly identified: {tn}/{correct_answers_gpt} ({tn/correct_answers_gpt*100:.1f}%)")
        logger.evaluation(
            f"  Correct answers misclassified (as hallucinations): {fp}/{correct_answers_gpt} ({fp/correct_answers_gpt*100:.1f}%)")

    # Attach custom score to the performance metrics
    attach_custom_score(performance)

    return performance

# ================================================================
# HTML REPORT GENERATION
# ================================================================


def generate_html_report(summary, classifier_performance=None, classifier_run_dir=None):
    """
    Generate an HTML report with the evaluation results

    Args:
        summary (dict): Summary statistics from analyze_results
        classifier_performance (dict): Performance metrics from analyze_classifier_performance
        classifier_run_dir (str): Path to the classifier run directory used

    Returns:
        str: HTML content as a string
    """
    from datetime import datetime

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start building HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hallucination Detection Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 25px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .confusion-item {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .tp {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .tn {{ background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
        .fp {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .fn {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
        .summary-stats {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .stat-row:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            font-weight: 600;
            color: #495057;
        }}
        .stat-value {{
            color: #212529;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Hallucination Detection Evaluation Report</h1>
        
        <div class="timestamp">
            Generated on: {timestamp}
        </div>

        <h2>Classifier Configuration</h2>
        <div class="summary-stats">
            <div class="stat-row">
                <span class="stat-label">Classifier Run Directory:</span>
                <span class="stat-value">{classifier_run_dir if classifier_run_dir else 'N/A'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Model Name:</span>
                <span class="stat-value">{MODEL_NAME if MODEL_NAME else 'N/A'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">HuggingFace Model ID:</span>
                <span class="stat-value">{HUGGINGFACE_MODEL_ID if HUGGINGFACE_MODEL_ID else 'N/A'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Transformer-Lens Model Name:</span>
                <span class="stat-value">{TRANSFORMER_LENS_MODEL_NAME if TRANSFORMER_LENS_MODEL_NAME else 'N/A'}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Evaluation Dataset:</span>
                <span class="stat-value">{DATASET_CSV_PATH if DATASET_CSV_PATH else 'N/A'}</span>
            </div>
        </div>
"""

    # Add classifier performance section if available
    if classifier_performance and classifier_performance.get('accuracy') is not None:
        html_content += f"""
        <h2>📊 Classifier Performance vs GPT-3.5 Ground Truth</h2>
        
        <h3>Overall Performance Metrics</h3>
        <div class="summary-stats">
            <div class="stat-row">
                <span class="stat-label">Accuracy:</span>
                <span class="stat-value">{classifier_performance['accuracy']*100:.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">AUROC:</span>
                <span class="stat-value">{classifier_performance.get('auroc', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">PR-AUROC:</span>
                <span class="stat-value">{classifier_performance.get('pr_auroc', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">F1 Score:</span>
                <span class="stat-value">{classifier_performance.get('f1', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Precision:</span>
                <span class="stat-value">{classifier_performance.get('precision', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Recall:</span>
                <span class="stat-value">{classifier_performance.get('recall', 0):.3f}</span>
            </div>
        </div>
        
        <h3>Confusion Matrix</h3>
        <div class="confusion-matrix">
            <div class="confusion-item tp">
                <strong>True Positives</strong><br>
                <span style="font-size: 1.5em;">{classifier_performance['confusion_matrix']['true_positives']}</span><br>
                <small>Incorrect/Hallucinated → Hallucination</small>
            </div>
            <div class="confusion-item tn">
                <strong>True Negatives</strong><br>
                <span style="font-size: 1.5em;">{classifier_performance['confusion_matrix']['true_negatives']}</span><br>
                <small>Correct → Correct</small>
            </div>
            <div class="confusion-item fp">
                <strong>False Positives</strong><br>
                <span style="font-size: 1.5em;">{classifier_performance['confusion_matrix']['false_positives']}</span><br>
                <small>Correct → Hallucination [WARNING]</small>
            </div>
            <div class="confusion-item fn">
                <strong>False Negatives</strong><br>
                <span style="font-size: 1.5em;">{classifier_performance['confusion_matrix']['false_negatives']}</span><br>
                <small>Incorrect/Hallucinated → Correct [WARNING]</small>
            </div>
        </div>

        <h3>Per-Class Performance</h3>
        <div class="summary-stats">
            <div class="stat-row">
                <span class="stat-label"><strong>Class 0 (Correct Answers):</strong></span>
                <span class="stat-value"></span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Precision:</span>
                <span class="stat-value">{classifier_performance.get('precision_per_class', [None, None])[0]:.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Recall:</span>
                <span class="stat-value">{classifier_performance.get('recall_per_class', [None, None])[0]:.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label"><strong>Class 1 (Hallucinations):</strong></span>
                <span class="stat-value"></span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Precision:</span>
                <span class="stat-value">{classifier_performance.get('precision_per_class', [None, None])[1]:.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Recall:</span>
                <span class="stat-value">{classifier_performance.get('recall_per_class', [None, None])[1]:.3f}</span>
            </div>
        </div>

        <h3>Custom Score Analysis</h3>
        <div class="summary-stats">
            <div class="stat-row">
                <span class="stat-label"><strong>Custom Score Components:</strong></span>
                <span class="stat-value"></span>
            </div>
            <div class="stat-row">
                <span class="stat-label">F-beta (Hallucinations, β={classifier_performance.get('custom_params_beta', 'N/A')}):</span>
                <span class="stat-value">{classifier_performance.get('f1_beta', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">F1 (Correct Answers):</span>
                <span class="stat-value">{classifier_performance.get('f0_1', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">MCC (Matthews Correlation):</span>
                <span class="stat-value">{classifier_performance.get('mcc', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">MCC Gate (γ={classifier_performance.get('custom_params_gamma', 'N/A')}):</span>
                <span class="stat-value">{classifier_performance.get('gate_mcc_gamma', 0):.3f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Blend Method:</span>
                <span class="stat-value">{classifier_performance.get('custom_params_blend', 'N/A')}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Weight (w={classifier_performance.get('custom_params_w', 'N/A')}):</span>
                <span class="stat-value">{classifier_performance.get('custom_params_w', 0):.1f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label"><strong>Final Custom Score:</strong></span>
                <span class="stat-value"><strong>{classifier_performance.get('custom_final_score', 0):.3f}</strong></span>
            </div>
        </div>

        <h3>Detailed Analysis</h3>
        <div class="summary-stats">
"""

        # Calculate totals for detailed analysis
        total_samples = (classifier_performance['confusion_matrix']['true_positives'] +
                         classifier_performance['confusion_matrix']['true_negatives'] +
                         classifier_performance['confusion_matrix']['false_positives'] +
                         classifier_performance['confusion_matrix']['false_negatives'])

        # y_true=1 is a hallucination
        hallucinated_answers_gpt = (classifier_performance['confusion_matrix']['true_positives'] +
                                    classifier_performance['confusion_matrix']['false_negatives'])

        # y_true=0 is a correct answer
        correct_answers_gpt = (classifier_performance['confusion_matrix']['true_negatives'] +
                               classifier_performance['confusion_matrix']['false_positives'])

        tp = classifier_performance['confusion_matrix']['true_positives']
        tn = classifier_performance['confusion_matrix']['true_negatives']
        fp = classifier_performance['confusion_matrix']['false_positives']
        fn = classifier_performance['confusion_matrix']['false_negatives']

        html_content += f"""
            <div class="stat-row">
                <span class="stat-label">Total samples:</span>
                <span class="stat-value">{total_samples}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Incorrect answers/Hallucinations (GPT-3.5):</span>
                <span class="stat-value">{hallucinated_answers_gpt} ({hallucinated_answers_gpt/total_samples*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Correct answers (GPT-3.5):</span>
                <span class="stat-value">{correct_answers_gpt} ({correct_answers_gpt/total_samples*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Hallucinations properly identified:</span>
                <span class="stat-value">{tp}/{hallucinated_answers_gpt} ({tp/hallucinated_answers_gpt*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Hallucinations missed (misclassified as correct):</span>
                <span class="stat-value">{fn}/{hallucinated_answers_gpt} ({fn/hallucinated_answers_gpt*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Correct answers properly identified:</span>
                <span class="stat-value">{tn}/{correct_answers_gpt} ({tn/correct_answers_gpt*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Correct answers misclassified (as hallucinations):</span>
                <span class="stat-value">{fp}/{correct_answers_gpt} ({fp/correct_answers_gpt*100:.1f}%)</span>
            </div>
        </div>
"""

    # Add dataset inference results
    html_content += f"""
        <h2>📈 Dataset Inference Results</h2>
        
        <div class="summary-stats">
            <div class="stat-row">
                <span class="stat-label">Total samples processed:</span>
                <span class="stat-value">{summary['total_samples']}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Successful predictions:</span>
                <span class="stat-value">{summary['successful_predictions']}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Failed predictions:</span>
                <span class="stat-value">{summary['failed_predictions']}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Success rate:</span>
                <span class="stat-value">{summary['success_rate']*100:.1f}%</span>
            </div>
"""

    if 'hallucinations_detected' in summary:
        html_content += f"""
            <div class="stat-row">
                <span class="stat-label">Hallucinations detected:</span>
                <span class="stat-value">{summary['hallucinations_detected']} ({summary['hallucination_rate']*100:.1f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Normal text detected:</span>
                <span class="stat-value">{summary['normal_detected']} ({(1-summary['hallucination_rate'])*100:.1f}%)</span>
            </div>
"""

    if 'average_confidence' in summary and summary['average_confidence'] is not None:
        html_content += f"""
            <div class="stat-row">
                <span class="stat-label">Average confidence:</span>
                <span class="stat-value">{summary['average_confidence']:.3f}</span>
            </div>
"""

    if 'average_hallucination_probability' in summary and summary['average_hallucination_probability'] is not None:
        html_content += f"""
            <div class="stat-row">
                <span class="stat-label">Average hallucination probability:</span>
                <span class="stat-value">{summary['average_hallucination_probability']:.3f}</span>
            </div>
"""

    html_content += """
        </div>
        
        <h2>🎯 Key Insights</h2>
"""

    # Add insights based on performance
    if classifier_performance and classifier_performance.get('accuracy') is not None:
        accuracy = classifier_performance['accuracy']
        auroc = classifier_performance['auroc']
        f1 = classifier_performance['f1']

        if accuracy >= 0.7:
            html_content += """
        <div class="success">
            <strong>✅ Good Performance:</strong> The classifier shows strong accuracy in detecting hallucinations.
        </div>
"""
        elif accuracy >= 0.6:
            html_content += """
        <div class="info">
            <strong>⚠️ Moderate Performance:</strong> The classifier shows reasonable accuracy but has room for improvement.
        </div>
"""
        else:
            html_content += """
        <div class="warning">
            <strong>⚠️ Needs Improvement:</strong> The classifier accuracy is below optimal levels and may need retraining.
        </div>
"""

        if auroc >= 0.8:
            html_content += """
        <div class="success">
            <strong>✅ Excellent Discrimination:</strong> The classifier shows excellent ability to distinguish between hallucinations and normal text.
        </div>
"""
        elif auroc >= 0.7:
            html_content += """
        <div class="info">
            <strong>📊 Good Discrimination:</strong> The classifier shows good ability to distinguish between different types of content.
        </div>
"""
        else:
            html_content += """
        <div class="warning">
            <strong>⚠️ Limited Discrimination:</strong> The classifier may struggle to reliably distinguish between hallucinations and normal text.
        </div>
"""

    html_content += f"""
        <div class="info">
            <strong>📋 Summary:</strong> This evaluation processed {summary['total_samples']} samples with a {summary['success_rate']*100:.1f}% success rate.
            The classifier demonstrates {'strong' if classifier_performance and classifier_performance.get('accuracy', 0) >= 0.7 else 'moderate'} performance
            in detecting hallucinations in generated text. Custom Score: {classifier_performance.get('custom_final_score', 'N/A'):.3f}.
        </div>
        
    </div>
</body>
</html>
"""

    return html_content


def save_html_report(html_content, output_dir=None):
    """
    Save the HTML report to a file

    Args:
        html_content (str): HTML content to save
        output_dir (str): Directory to save the report in (optional, defaults to "./inference_results")

    Returns:
        str: Path to the saved HTML file
    """
    from datetime import datetime

    # Use default directory if none provided
    if output_dir is None:
        output_dir = "./inference_results"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"hallucination_detection_report_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    # Save HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path

# ================================================================
# EVALUATION RUN DIRECTORY MANAGEMENT
# ================================================================


def create_evaluation_run_directory(classifier_run_dir):
    """
    Create a unique evaluation run directory in the specific classifier run directory

    Args:
        classifier_run_dir (str): Specific classifier run directory where the best classifier was found

    Returns:
        str: Path to the created evaluation run directory
    """
    from datetime import datetime
    import os

    # Generate unique timestamp-based directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_run_name = f"evaluation_run_{timestamp}"
    evaluation_run_dir = os.path.join(classifier_run_dir, evaluation_run_name)

    # Create the directory
    os.makedirs(evaluation_run_dir, exist_ok=True)

    logger.info(
        f"Created evaluation run directory inside classifier run: {evaluation_run_dir}")
    return evaluation_run_dir


def save_evaluation_metadata(evaluation_output_dir, classifier_run_dir, summary, classifier_performance=None):
    """
    Save metadata about the evaluation run to a JSON file

    Args:
        evaluation_output_dir (str): Directory where evaluation results are saved
        classifier_run_dir (str): Directory of the classifier that was used
        summary (dict): Summary statistics from analyze_results
        classifier_performance (dict): Performance metrics from analyze_classifier_performance
    """
    from datetime import datetime
    import json
    import os

    metadata = {
        'evaluation_info': {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'classifier_run_used': classifier_run_dir,
            'dataset_path': DATASET_CSV_PATH,
            'sample_size': SAMPLE_SIZE,
            'batch_size': BATCH_SIZE,
            'max_answer_tokens': MAX_ANSWER_TOKENS,
            'model_name': MODEL_NAME,
            'device_id': DEVICE_ID,
            'save_results_enabled': SAVE_RESULTS,
            'debug_verbose': DEBUG_VERBOSE
        },
        'results_summary': summary,
        'classifier_performance': classifier_performance or {}
    }

    # Save metadata to JSON file
    metadata_path = os.path.join(
        evaluation_output_dir, 'evaluation_run_metadata.json')
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Evaluation metadata saved to: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation metadata: {e}")

# ================================================================
# RESULTS ANALYSIS AND SAVING
# ================================================================


def analyze_results(results, evaluation_output_dir=None):
    """
    Analyze dataset inference results and display statistics

    Args:
        results (list): List of inference results from run_batch_inference
        evaluation_output_dir (str): Directory to save evaluation results (optional)

    This function:
    1. Creates comprehensive analysis of hallucination detection performance
    2. Displays detailed statistics and examples
    3. Optionally saves results if SAVE_RESULTS is True
    4. Provides examples of different prediction categories

    Returns:
        dict: Summary statistics
    """

    logger.info("\n" + "="*80)
    logger.info("DATASET INFERENCE RESULTS ANALYSIS")
    logger.info("="*80)

    # Basic statistics
    total_samples = len(results)
    successful_predictions = sum(1 for r in results if r.get(
        'interpretation') not in ['PREDICTION_FAILED', 'PROCESSING_ERROR'])
    failed_predictions = total_samples - successful_predictions

    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Successful predictions: {successful_predictions}")
    logger.info(f"Failed predictions: {failed_predictions}")

    if successful_predictions > 0:
        # Hallucination detection statistics
        hallucination_detected = sum(
            1 for r in results if r.get('is_hallucination') == True)
        normal_detected = sum(1 for r in results if r.get(
            'is_hallucination') == False)

        logger.info(f"\nHallucination Detection Results:")
        logger.info(
            f"  Hallucinations detected: {hallucination_detected} ({hallucination_detected/successful_predictions*100:.1f}%)")
        logger.info(
            f"  Normal text detected: {normal_detected} ({normal_detected/successful_predictions*100:.1f}%)")

        # Confidence statistics
        valid_confidences = [r['confidence']
                             for r in results if r.get('confidence') is not None]
        if valid_confidences:
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
            logger.info(f"  Average confidence: {avg_confidence:.3f}")

        # Probability statistics
        valid_hall_probs = [r['hallucination_probability'] for r in results if r.get(
            'hallucination_probability') is not None]
        if valid_hall_probs:
            avg_hall_prob = sum(valid_hall_probs) / len(valid_hall_probs)
            logger.info(
                f"  Average hallucination probability: {avg_hall_prob:.3f}")

    # Create summary report
    summary = {
        'total_samples': total_samples,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'success_rate': successful_predictions / total_samples if total_samples > 0 else 0,
    }

    if successful_predictions > 0:
        summary.update({
            'hallucinations_detected': hallucination_detected,
            'normal_detected': normal_detected,
            'hallucination_rate': hallucination_detected / successful_predictions,
            'average_confidence': avg_confidence if valid_confidences else None,
            'average_hallucination_probability': avg_hall_prob if valid_hall_probs else None,
        })

    # Conditional saving if SAVE_RESULTS is enabled
    if SAVE_RESULTS:
        import os
        import json
        from datetime import datetime

        # Use provided evaluation output directory or default fallback
        if evaluation_output_dir:
            output_dir = evaluation_output_dir
        else:
            output_dir = "./inference_results"
            os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to CSV
        try:
            df_results = pd.DataFrame(results)
            csv_path = os.path.join(
                output_dir, f"squad_inference_results_{timestamp}.csv")
            df_results.to_csv(csv_path, index=False)
            logger.info(f"\nDetailed results saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")

        # Save results to JSON for complex data preservation
        try:
            json_path = os.path.join(
                output_dir, f"squad_inference_results_{timestamp}.json")

            # Convert NumPy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                """Convert NumPy types to Python native types for JSON serialization"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Convert the results to JSON-serializable format
            json_serializable_results = convert_numpy_types(results)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_serializable_results, f,
                          indent=2, ensure_ascii=False)
            logger.info(f"JSON results saved to: {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")

        # Save summary report
        try:
            summary_path = os.path.join(
                output_dir, f"squad_inference_summary_{timestamp}.json")

            # Convert NumPy types to Python native types for JSON serialization
            json_serializable_summary = convert_numpy_types(summary)

            with open(summary_path, 'w') as f:
                json.dump(json_serializable_summary, f, indent=2)
            logger.info(f"Summary report saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
    else:
        logger.info(f"\nResults not saved (SAVE_RESULTS=False)")
        logger.info(f"Set SAVE_RESULTS=True to save results to Google Drive")

    # Analyze classifier performance if GPT-3.5 evaluation was performed
    classifier_performance = {}
    if any(r.get('is_correct') is not None for r in results):
        classifier_performance = analyze_classifier_performance(results)
        summary['classifier_performance'] = classifier_performance

    # Show example predictions
    logger.info(f"\nExample Predictions:")
    logger.info("-" * 50)

    # Enhanced categories with GPT-3.5 evaluation context
    if any(r.get('is_correct') is not None for r in results):
        categories = {
            'Correct Answers (GPT-3.5) + Normal (Classifier)': [r for r in results if r.get('is_correct') == True and r.get('is_hallucination') == False],
            'Correct Answers (GPT-3.5) + Hallucination (Classifier)': [r for r in results if r.get('is_correct') == True and r.get('is_hallucination') == True],
            'Incorrect Answers (GPT-3.5) + Hallucination (Classifier)': [r for r in results if r.get('is_correct') == False and r.get('is_hallucination') == True],
            'Incorrect Answers (GPT-3.5) + Normal (Classifier)': [r for r in results if r.get('is_correct') == False and r.get('is_hallucination') == False],
            'Failed Predictions': [r for r in results if r.get('interpretation') in ['PREDICTION_FAILED', 'PROCESSING_ERROR']]
        }
    else:
        categories = {
            'Hallucination Detected': [r for r in results if r.get('is_hallucination') == True],
            'Normal Text': [r for r in results if r.get('is_hallucination') == False],
            'Failed Predictions': [r for r in results if r.get('interpretation') in ['PREDICTION_FAILED', 'PROCESSING_ERROR']]
        }

    for category, examples in categories.items():
        if examples:
            logger.debug(f"\n{category} (showing first 2):")
            for i, example in enumerate(examples[:2]):
                logger.debug(
                    f"  {i+1}. Q: {example.get('question', 'N/A')[:70]}...")
                logger.debug(
                    f"     Expected: {example.get('correct_answer', 'N/A')[:50]}...")
                logger.debug(
                    f"     Generated: {example.get('generated_answer', 'N/A')[:50]}...")
                if example.get('hallucination_probability') is not None:
                    logger.debug(
                        f"     Hallucination prob: {example.get('hallucination_probability', 0):.3f}")
                if example.get('is_correct') is not None:
                    logger.debug(
                        f"     GPT-3.5 evaluation: {'[OK] Correct' if example.get('is_correct') else '[ERROR] Incorrect'}")
                logger.debug("")

    return summary

# ================================================================
# MAIN FUNCTION
# ================================================================


def main():
    """
    Main function to orchestrate the dataset inference system.

    This function coordinates the full pipeline:
    1. Loads the question-answering dataset from CSV format
    2. Loads the trained classifier from nocheckpoint_nodataleak.py
    3. Loads the Llama 7B model with TransformerLens (local or downloaded)
    4. Sets up the activation extractor to match training conditions
    5. Runs batch inference on the entire dataset
    6. Analyzes and saves results

    The system will exit with helpful error messages if any step fails.
    """

    # Declare global variables that will be modified
    global device

    logger.info("Llama 7B Hallucination Detection on Question-Answering Dataset")
    logger.info("Using classifier from nocheckpoint_nodataleak.py")
    logger.info("=" * 60)

    # Display current configuration
    logger.info("Configuration:")
    logger.info(
        f"  Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Entire dataset'}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Physical GPU ID: {DEVICE_ID}")
    logger.info(
        f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"  Batch processing: squad_generator3 architecture")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Save results: {SAVE_RESULTS}")
    logger.info(f"  GPT-3.5 evaluation: enabled (always)")
    logger.info(f"  Max answer tokens: {MAX_ANSWER_TOKENS}")
    logger.info(f"  Debug verbose: {DEBUG_VERBOSE}")
    logger.info("=" * 60)

    # STEP 1: Load the dataset
    logger.info("Loading dataset...")
    dataset_df = load_dataset()
    if dataset_df is None:
        logger.error(
            "Failed to load dataset! Please check the file path and format.")
        return

    # STEP 2: Load the trained classifier
    logger.info("Loading saved classifier...")

    # Find the classifier run directory in the experiment folder
    # Classifiers are now saved in: ACTIVATIONS_BASE_DIR/EXPERIMENT_ID/classifier_run_ID/
    experiment_dir = os.path.join(ACTIVATIONS_BASE_DIR, EXPERIMENT_ID)

    classifier_run_dir = find_classifier_run_in_experiment(experiment_dir)

    if not classifier_run_dir:
        logger.error(
            "No classifier runs found in experiment directory! Please run classifier.py first.")
        logger.error(f"Expected location: {experiment_dir}/classifier_run_*/")
        return

    # Find the best classifier file within the run
    classifier_path = find_best_saved_classifier(classifier_run_dir)
    if not classifier_path:
        logger.error(
            "No best classifier found in the run! Please check classifier.py output.")
        return

    # Load the complete classifier package (model + scaler + metadata)
    classifier_package = load_saved_classifier(classifier_path)
    if not classifier_package:
        logger.error(
            "Failed to load classifier package! Please check the saved classifier.")
        return

    # STEP 2.5: Create evaluation run directory in the specific classifier run directory (NEW STRUCTURE)
    # Evaluation results are now saved inside the classifier folder: classifier_run_ID/evaluation_run_TIMESTAMP/
    logger.info("Creating evaluation run directory...")
    evaluation_output_dir = create_evaluation_run_directory(classifier_run_dir)

    # STEP 2.6: Load and save configuration audit trail
    logger.info("Loading configuration audit trail...")
    all_configs = load_all_configs(
        ACTIVATIONS_BASE_DIR, EXPERIMENT_ID, os.path.basename(classifier_run_dir))
    save_audit_metadata(evaluation_output_dir, all_configs)
    logger.info("Configuration audit trail saved successfully")

    # Set logger output directory to the evaluation directory
    logger.set_output_directory(evaluation_output_dir)

    # STEP 3: Load the Llama model using ModelManager (exactly like generator.py)
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

    llama_model = model_manager.get_model()

    # Initialize TokenManager for centralized token handling (exactly like generator.py)
    token_manager = TokenManager(
        model=llama_model, max_answer_tokens=MAX_ANSWER_TOKENS)
    logger.info("TokenManager initialized successfully")

    # Model loaded successfully
    llama_model = token_manager.setup_tokenizer_padding(llama_model)

    # Debug token IDs to identify conflicts
    token_manager.get_token_ids_debug_info(llama_model)

    # Verify model is on GPU
    try:
        if hasattr(llama_model, 'embed') and hasattr(llama_model.embed, 'W_E'):
            device_name = str(llama_model.embed.W_E.device)
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
        logger.info(f"Model has {llama_model.cfg.n_layers} layers")
    except:
        logger.info("Model structure information will be analyzed below...")

    # STEP 4: Extract classifier metadata
    logger.info("Extracting classifier metadata...")
    metadata = classifier_package['metadata']

    # STEP 5: Test evaluator connectivity (always required)
    logger.info("Testing evaluator connectivity...")
    test_evaluator_connectivity()
    logger.info("Evaluator connectivity test passed")

    # STEP 6: Run batch inference on dataset
    logger.info(
        "Running batch inference on dataset (TransformerLens generation - exactly like generator.py)...")
    results = run_batch_inference(
        llama_model, None, metadata, classifier_package, dataset_df, token_manager)

    # STEP 7: Evaluate answers with GPT-3.5 (always enabled)
    logger.info("Evaluating answers with GPT-3.5...")
    results = batch_evaluate_answers(results, client)

    # STEP 8: Analyze results
    logger.info("Analyzing results...")
    summary = analyze_results(results, evaluation_output_dir)

    # STEP 9: Generate HTML report
    logger.info("Generating HTML report...")
    classifier_performance = summary.get('classifier_performance', {})
    html_content = generate_html_report(
        summary, classifier_performance, classifier_run_dir)

    # Save HTML report
    html_path = save_html_report(html_content, evaluation_output_dir)
    logger.info(f"HTML report saved to: {html_path}")

    # STEP 10: Save evaluation metadata
    logger.info("Saving evaluation metadata...")
    save_evaluation_metadata(
        evaluation_output_dir, classifier_run_dir, summary, classifier_performance)

    logger.info("="*60)
    logger.info("DATASET INFERENCE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"Processed {summary['total_samples']} samples")
    logger.info(f"Success rate: {summary['success_rate']*100:.1f}%")
    if 'hallucination_rate' in summary:
        logger.evaluation(
            f"Hallucination detection rate: {summary['hallucination_rate']*100:.1f}%")

    # Display processing method used
    logger.info(
        "Processing Method: BATCH (TransformerLens generation - exactly like generator.py)")
    logger.info("  Expected speedup: 5-10x faster than single-sample processing")
    logger.info("  GPU utilization: Optimized with LEFT-padding")
    logger.info("  Memory efficiency: Batch tensor operations")

    # Display classifier performance if available
    if 'classifier_performance' in summary and summary['classifier_performance']:
        perf = summary['classifier_performance']
        logger.evaluation("Classifier Performance vs GPT-3.5 Ground Truth:")
        logger.evaluation(f"  Accuracy: {perf['accuracy']*100:.1f}%")
        if perf['auroc'] is not None:
            logger.evaluation(f"  AUROC: {perf['auroc']:.3f}")
        logger.evaluation(f"  F1 Score: {perf['f1']:.3f}")
        if 'custom_final_score' in perf:
            logger.evaluation(
                f"  Custom Score: {perf['custom_final_score']:.3f}")

    if SAVE_RESULTS:
        logger.info(
            f"Results saved to evaluation directory: {evaluation_output_dir}")
    else:
        logger.info("Results displayed only (not saved)")
        logger.info("To save results, set SAVE_RESULTS=True in configuration")

    # Display HTML report information
    logger.info(f"HTML Report Generated: {html_path}")
    logger.info(f"Evaluation Directory: {evaluation_output_dir}")
    logger.info("Open HTML report in browser to view detailed results")

    # Final memory cleanup
    if ENABLE_MEMORY_CLEARING:
        logger.info("Performing final memory cleanup...")
        clear_gpu_memory()

    # Display model information
    logger.info("Model Used: Llama 7B (local/downloaded)")
    logger.info(f"  Evaluation dataset: {DATASET_CSV_PATH}")
    logger.info(f"  Evaluation results directory: {evaluation_output_dir}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Physical GPU ID: {DEVICE_ID}")
    logger.info(
        f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"  Memory clearing enabled: {ENABLE_MEMORY_CLEARING}")
    logger.info(f"  Debug verbose: {DEBUG_VERBOSE}")

# ================================================================
# SCRIPT EXECUTION
# ================================================================
# This ensures the main function only runs when the script is executed directly,
# not when it's imported as a module


if __name__ == "__main__":
    main()
