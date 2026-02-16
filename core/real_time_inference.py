"""Real-Time Interactive Hallucination Detection.

Interactive command-line tool that:
1. Loads trained classifier from specified path
2. Discovers experiment configuration (model_name, model_id) from metadata
3. Prompts user for questions
4. Generates responses from the configured LLM
5. Captures activations at classifier's trained layer/hook/scheme
6. Predicts hallucination probability in real-time

CONFIGURATION DISCOVERY:
    - Reads config_metadata.json from experiment directory for model info
    - Falls back to classifier_config_metadata.json if needed
    - Automatically uses correct model that generated training data
    - Supports GPT-2, LLaMA 3.1, and other HuggingFace models

FEATURES:
    - Automatic model configuration from experiment metadata
    - GPU device selection via --device-id argument
    - Batch processing support for multiple questions
    - Dataset CSV override via --dataset-csv-path
    - Debug logging mode for detailed activation inspection

USAGE:
    Interactive mode:
        $ python real_time_inference.py --classifier-path ./data/classifier/classifier_run_1705432123/best_classifier_classifier_run_1705432123.pkl --device-id 0
    
    Batch mode (with dataset):
        $ python real_time_inference.py --classifier-path ... --dataset-csv-path ./data/squad_clean.csv

REQUIRED ARGUMENTS:
    --classifier-path: Path to saved classifier pkl file

OPTIONAL ARGUMENTS:
    --device-id: GPU device ID (default: 0)
    --max-answer-tokens: Max tokens to generate (default: 100)
    --dataset-csv-path: CSV file for batch mode (default: None for interactive)
    --debug: Enable debug logging

Example:
    $ python real_time_inference.py \\
        --classifier-path ./data/classifier/classifier_run_1705432123/best_classifier_classifier_run_1705432123.pkl \\
        --device-id 0 \\
        --max-answer-tokens 100
"""

# ================================================================
# REAL-TIME INTERACTIVE HALLUCINATION DETECTION
# ================================================================
# Interactive loop that:
# 1. Loads experiment configuration (model_name, huggingface_model_id) from classifier directory
# 2. Prompts user for a question
# 3. Generates response from the configured model (e.g., GPT-2 Medium, Llama 3.1 8B)
# 4. Captures activations at classifier's trained location
# 5. Predicts if response is a hallucination
#
# Configuration Discovery:
# - Reads ./data/activations/{EXPERIMENT_ID}/config_metadata.json for model info
# - Falls back to classifier_config_metadata.json if needed
# - Automatically uses the correct model that generated the training data
# ================================================================

from tqdm.auto import tqdm
from helpers.activation_utils import generate_and_capture_efficiently
from helpers.token_manager import TokenManager
from helpers.model_manager import ModelManager
import config
import os
import sys
import argparse
import torch
import numpy as np
import pickle
import json
from typing import Dict, List, Any, Tuple
import logging
import time

# --- CRITICAL: SET GPU DEVICE BEFORE TORCH IS IMPORTED ---
parser = argparse.ArgumentParser(
    description='Interactive hallucination detection with automatic model configuration from experiment')
parser.add_argument('--device-id', type=int, default=0, help='GPU device ID')
parser.add_argument('--classifier-path', type=str, required=True,
                    help='Path to the best saved classifier (e.g., ./data/classifier/classifier_run_1705432123/best_classifier_classifier_run_1705432123.pkl)')
parser.add_argument('--model-device-id', type=int,
                    default=0, help='GPU device ID for model')
parser.add_argument('--max-answer-tokens', type=int, default=100,
                    help='Maximum tokens to generate for answers')
parser.add_argument('--dataset-csv-path', type=str, default=None,
                    help='Path to the dataset CSV file to override config.DATASET_CSV_PATH (e.g., ./data/squad_clean.csv, ./data/dynabench.csv)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging')

args = parser.parse_args()

# Set GPU before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config after argument parsing

# Override config DATASET_CSV_PATH if provided via command line
if args.dataset_csv_path:
    logger.info(f"Overriding config.DATASET_CSV_PATH with: {args.dataset_csv_path}")
    config.DATASET_CSV_PATH = args.dataset_csv_path

# Import required modules

# ================================================================
# VALIDATION AND SETUP
# ================================================================


def load_experiment_config(classifier_path: str) -> Dict[str, Any]:
    """
    Loads the experiment configuration from config files in the classifier directory structure.
    
    Directory structure:
    ./data/classifier/
        └── classifier_run_{timestamp}/
            ├── config_metadata.json (generator config with model info)
            ├── classifier_config_metadata.json (classifier metadata)
            └── best_classifier_classifier_run_{timestamp}.pkl (classifier_path)
    
    Args:
        classifier_path (str): Path to the saved classifier pickle file.
        
    Returns:
        dict: Experiment configuration including model_name and huggingface_model_id.
    """
    # Get classifier directory
    classifier_dir = os.path.dirname(os.path.abspath(classifier_path))
    
    # Get experiment directory (parent of classifier directory)
    experiment_dir = os.path.dirname(classifier_dir)
    
    # Try to load config_metadata.json from experiment directory (has model info)
    config_metadata_path = os.path.join(experiment_dir, 'config_metadata.json')
    experiment_config = {}
    
    if os.path.exists(config_metadata_path):
        logger.info(f"Loading experiment config from: {config_metadata_path}")
        try:
            with open(config_metadata_path, 'r') as f:
                experiment_config = json.load(f)
            logger.info(f"  ✓ Found model_name: {experiment_config.get('model_name')}")
            logger.info(f"  ✓ Found huggingface_model_id: {experiment_config.get('huggingface_model_id')}")
        except Exception as e:
            logger.warning(f"Failed to load experiment config: {e}")
    else:
        logger.warning(f"Experiment config not found at: {config_metadata_path}")
        
    # Also load classifier_config_metadata.json for additional context
    classifier_metadata_path = os.path.join(classifier_dir, 'classifier_config_metadata.json')
    classifier_config = {}
    
    if os.path.exists(classifier_metadata_path):
        logger.info(f"Loading classifier config from: {classifier_metadata_path}")
        try:
            with open(classifier_metadata_path, 'r') as f:
                classifier_config = json.load(f)
            # Extract generator config if embedded
            if 'generator_config' in classifier_config:
                gen_config = classifier_config['generator_config']
                logger.info(f"  ✓ Found generator config in classifier metadata")
                # Fallback: use generator config if experiment config is missing model info
                if not experiment_config.get('model_name'):
                    experiment_config['model_name'] = gen_config.get('model_name')
                if not experiment_config.get('huggingface_model_id'):
                    experiment_config['huggingface_model_id'] = gen_config.get('huggingface_model_id')
        except Exception as e:
            logger.warning(f"Failed to load classifier config metadata: {e}")
    
    return {
        'experiment_config': experiment_config,
        'classifier_config': classifier_config,
        'experiment_dir': experiment_dir,
        'classifier_dir': classifier_dir,
    }


def validate_classifier_path(classifier_path: str) -> Dict[str, Any]:
    """
    Validates and loads the classifier from the specified path.

    Args:
        classifier_path (str): Path to the saved classifier pickle file.

    Returns:
        dict: Classifier package containing model, scaler, and metadata.

    Raises:
        FileNotFoundError: If classifier file doesn't exist.
        ValueError: If classifier format is invalid.
    """
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier not found at: {classifier_path}")

    logger.info(f"Loading classifier from: {classifier_path}")

    try:
        with open(classifier_path, 'rb') as f:
            classifier_package = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load classifier: {e}")

    # Validate package structure
    required_keys = ['model', 'scaler', 'metadata']
    for key in required_keys:
        if key not in classifier_package:
            raise ValueError(f"Classifier package missing required key: {key}")

    metadata = classifier_package['metadata']
    logger.info(f"Classifier Metadata:")
    logger.info(f"  - Layer: {metadata.get('layer')}")
    logger.info(f"  - Hook: {metadata.get('hook')}")
    logger.info(f"  - Scheme: {metadata.get('scheme')}")
    logger.info(f"  - Classifier: {metadata.get('classifier_name')}")
    logger.info(
        f"  - Optimal Threshold: {metadata.get('optimal_threshold', 0.5):.4f}")
    logger.info(
        f"  - Custom Score: {metadata.get('custom_score_at_optimal', 0.0):.4f}")
    logger.info(f"  - Features: {metadata.get('n_features')}")

    return classifier_package


def prepare_inference_config(classifier_package: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares the inference configuration based on classifier metadata.

    Args:
        classifier_package (dict): The loaded classifier package.

    Returns:
        dict: Inference configuration with hooks, schemes, and layers.
    """
    metadata = classifier_package['metadata']

    target_layer = metadata.get('layer')
    target_hook = metadata.get('hook')
    target_scheme = metadata.get('scheme')

    if not all([target_layer, target_hook, target_scheme]):
        raise ValueError("Classifier metadata missing layer, hook, or scheme")

    # Configure hooks and schemes for this specific classifier
    active_hooks = [target_hook]
    token_schemes = [target_scheme]

    inference_config = {
        'layer': target_layer,
        'hook': target_hook,
        'scheme': target_scheme,
        'active_hooks': active_hooks,
        'token_schemes': token_schemes,
        'n_features': metadata.get('n_features', 0),
        'optimal_threshold': metadata.get('optimal_threshold', 0.5),
    }

    logger.info(f"Inference Configuration:")
    logger.info(f"  - Active Hooks: {active_hooks}")
    logger.info(f"  - Token Schemes: {token_schemes}")
    logger.info(f"  - Layer Range: {target_layer} to {target_layer}")

    return inference_config


# ================================================================
# ACTIVATION EXTRACTION FOR INFERENCE
# ================================================================

def extract_activations_for_prediction(
    batch_activations: Dict[str, Any],
    positions_by_scheme: Dict[str, torch.Tensor],
    inference_config: Dict[str, Any],
    batch_size: int
) -> np.ndarray:
    """
    Extracts activations at the specified hook and position scheme for prediction.
    Uses the exact same mechanism as the classifier training.

    Args:
        batch_activations (dict): Activation data structure from generate_and_capture_efficiently.
                                  Format: {hook_key: {scheme: [list of activation dicts]}}
        positions_by_scheme (dict): Token positions for each scheme.
        inference_config (dict): Configuration with layer, hook, and scheme.
        batch_size (int): Batch size.

    Returns:
        np.ndarray: Features for classification (batch_size, n_features).
                    Each row is a flattened activation vector.
    """
    target_layer = inference_config['layer']
    target_hook = inference_config['hook']
    target_scheme = inference_config['scheme']

    # Build the hook key in the format used by activation_utils.py
    # Format: "layer_{layer}_{hook_name}"
    hook_key = f"layer_{target_layer}_{target_hook}"

    if hook_key not in batch_activations:
        available_hooks = list(batch_activations.keys())
        raise KeyError(
            f"Hook '{hook_key}' not found. Available hooks: {available_hooks}")

    if target_scheme not in batch_activations[hook_key]:
        available_schemes = list(batch_activations[hook_key].keys())
        raise KeyError(
            f"Scheme '{target_scheme}' not found. Available schemes: {available_schemes}")

    # Extract the list of activation dicts for this hook/scheme
    activation_list = batch_activations[hook_key][target_scheme]

    if not activation_list:
        raise ValueError(
            f"No activations captured for {hook_key}/{target_scheme}")

    if len(activation_list) != batch_size:
        logger.warning(
            f"Activation list size ({len(activation_list)}) != batch_size ({batch_size})")

    # Extract the numpy arrays from each activation dict
    # Each activation can be multi-dimensional (e.g., shape (d_model,) or (heads, d_head))
    # We need to flatten each one to 1D, then stack them to create 2D array
    features_list = []
    for act in activation_list:
        activation_array = act['activations']  # Could be any shape
        # Flatten to 1D
        flattened = activation_array.flatten()
        features_list.append(flattened)

    # Stack all flattened activations into a 2D array (batch_size, n_features)
    features = np.array(features_list)

    logger.debug(f"Extracted features shape: {features.shape}")
    logger.debug(
        f"Individual activation shape from first sample: {activation_list[0]['activations'].shape}")

    return features


# ================================================================
# INFERENCE PIPELINE
# ================================================================

def generate_llama_response(
    question: str,
    model_manager: ModelManager,
    token_manager: TokenManager,
    max_answer_tokens: int = 100,
    logger=None
) -> str:
    """
    Generates a response from Llama 3.1 8B Instruct for a given question.

    Args:
        question (str): The user's question.
        model_manager (ModelManager): Model manager instance.
        token_manager (TokenManager): Token manager instance.
        max_answer_tokens (int): Maximum tokens to generate.
        logger: Logger instance.

    Returns:
        str: The generated response text.
    """
    model = model_manager.get_model()
    device = next(model.parameters()).device

    # Format the prompt for Llama 3.1 Instruct
    prompt = f"Question: {question}\n\nAnswer:"

    # Tokenize the prompt
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]

    if logger:
        logger.debug(f"Prompt tokens shape: {tokens.shape}")

    # Generate response
    with torch.inference_mode():
        full_sequences = model.generate(
            input=tokens.unsqueeze(0),
            max_new_tokens=max_answer_tokens,
            do_sample=False,
            stop_at_eos=True,
            eos_token_id=model.tokenizer.eos_token_id,
            temperature=1.0,
            use_past_kv_cache=True,
            return_type='tokens',
            verbose=False
        )

    # Decode the generated text (skip the prompt tokens)
    generated_tokens = full_sequences[0, tokens.shape[0]:]
    response_text = model.to_string(generated_tokens).strip()

    return response_text


def run_inference_batch(
    questions: List[str],
    answers: List[str],
    model_manager: ModelManager,
    token_manager: TokenManager,
    classifier_package: Dict[str, Any],
    inference_config: Dict[str, Any],
    device_id: int = 0,
    max_answer_tokens: int = 50,
    debug_verbose: bool = False
) -> Dict[str, Any]:
    """
    Runs inference on a batch of Q&A pairs to detect hallucinations.

    Uses the same activation capture mechanism as generator.py:
    1. Tokenize Q&A prompts using token_manager
    2. Create LEFT-padded batch
    3. Generate completions and capture activations
    4. Extract features at specified positions
    5. Predict with loaded classifier

    Args:
        questions (list): List of question strings.
        answers (list): List of reference answer strings.
        model_manager (ModelManager): Model manager instance.
        token_manager (TokenManager): Token manager instance.
        classifier_package (dict): Loaded classifier package.
        inference_config (dict): Inference configuration.
        max_answer_tokens (int): Maximum tokens to generate.
        debug_verbose (bool): Enable verbose debugging.

    Returns:
        dict: Results including predictions, probabilities, and metadata.
    """
    model = model_manager.get_model()
    classifier = classifier_package['model']
    scaler = classifier_package['scaler']
    metadata = classifier_package['metadata']

    batch_size = len(questions)

    # Prepare prompts using the same mechanism as generator.py
    # Use simple Q&A format: "Context:\nQuestion: {q}\n\nAnswer: {a}"
    batch_prompts = []
    batch_info = []

    for idx, (q, a) in enumerate(zip(questions, answers)):
        # Create a simple context-free prompt (similar to generator.py's make_tokens_optimized)
        context_text = f"Question: {q}\n\nAnswer: {a}"
        tok_prompt = model.to_tokens(context_text, prepend_bos=True)[0]

        if tok_prompt is None or tok_prompt.numel() == 0:
            logger.warning(f"Failed to tokenize item {idx}")
            continue

        batch_prompts.append(tok_prompt)

        # Create a simple row object compatible with activation_utils.py
        # activation_utils expects batch_info[i]['row'] to have .question and .right_answer attributes
        batch_row = type('RowObject', (object,), {
            'question': q,
            'right_answer': a,  # For inference, this is the reference answer
        })()

        batch_item_info = {
            'row_idx': idx,
            'row': batch_row,  # Object with .question and .right_answer attributes
            'prompt_length': tok_prompt.shape[0],
            'padding_length': 0,
        }
        batch_info.append(batch_item_info)

    if not batch_prompts:
        logger.error("No valid prompts to process")
        return {}

    # Use TokenManager's LEFT-padding (identical to generator.py)
    batch_tensor, batch_info = token_manager.create_left_padded_batch(
        batch_prompts, batch_info)

    logger.debug(f"Batch tensor shape: {batch_tensor.shape}")
    logger.debug(f"Batch Info: {batch_info}")

    actual_batch_size = len(batch_prompts)

    # Generate and capture activations using the same mechanism as generator.py
    full_sequences, batch_activations, batch_results_list, positions_by_scheme = generate_and_capture_efficiently(
        model=model,
        token_manager=token_manager,
        batch_prompts_padded=batch_tensor,
        batch_info=batch_info,
        active_hooks=inference_config['active_hooks'],
        token_schemes=inference_config['token_schemes'],
        start_layer=inference_config['layer'],
        end_layer=inference_config['layer'],  # Single layer for classifier
        max_answer_tokens=max_answer_tokens,
        model_name=config.MODEL_NAME,
        logger=logger,
        debug_verbose=debug_verbose
    )

    # Extract features for classification
    features = extract_activations_for_prediction(
        batch_activations,
        positions_by_scheme,
        inference_config,
        actual_batch_size
    )

    logger.debug(f"Features shape before scaling: {features.shape}")

    # Scale features using the scaler from training
    features_scaled = scaler.transform(features)

    logger.debug(f"Features shape after scaling: {features_scaled.shape}")

    # Convert features to GPU tensor for XGBoost compatibility
    # XGBoost expects data on the same device as the model
    # Use cuda:0 since CUDA_VISIBLE_DEVICES is set to the correct physical GPU
    device = "cuda:0"
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    
    logger.debug(f"Features tensor device: {features_tensor.device}")
    logger.debug(f"Features tensor shape: {features_tensor.shape}")

    # Make predictions using GPU tensors
    predictions = classifier.predict(features_tensor)
    probabilities = classifier.predict_proba(features_tensor)

    # Apply optimal threshold from training
    optimal_threshold = metadata.get('optimal_threshold', 0.5)
    predictions_thresholded = (
        probabilities[:, 1] >= optimal_threshold).astype(int)

    # Prepare results
    results = {
        'questions': questions,
        'answers': answers,
        'predictions': predictions_thresholded.tolist(),
        'predictions_raw': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'confidence': np.max(probabilities, axis=1).tolist(),
        # Probability of hallucination
        'hallucination_scores': probabilities[:, 1].tolist(),
        'optimal_threshold': optimal_threshold,
        'metadata': {
            'layer': inference_config['layer'],
            'hook': inference_config['hook'],
            'scheme': inference_config['scheme'],
            'classifier_name': metadata.get('classifier_name'),
            'classifier_path': classifier_package.get('path', 'unknown'),
            'batch_size': actual_batch_size,
        }
    }

    return results


def main():
    """Main interactive hallucination detection loop."""

    # Load experiment configuration from classifier path
    logger.info("\n" + "="*80)
    logger.info("LOADING EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    try:
        config_data = load_experiment_config(args.classifier_path)
        experiment_config = config_data['experiment_config']
        
        # Extract model information
        model_name = experiment_config.get('model_name')
        huggingface_model_id = experiment_config.get('huggingface_model_id')
        
        if not model_name or not huggingface_model_id:
            raise ValueError("Model configuration not found in experiment config files")
            
        logger.info(f"Experiment configuration loaded successfully:")
        logger.info(f"  - Model Name: {model_name}")
        logger.info(f"  - HuggingFace Model ID: {huggingface_model_id}")
        logger.info(f"  - Experiment Directory: {config_data['experiment_dir']}")
        logger.info(f"  - Classifier Directory: {config_data['classifier_dir']}")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to load experiment configuration: {e}")
        logger.error("Falling back to config.py defaults")
        model_name = config.MODEL_NAME
        huggingface_model_id = config.HUGGINGFACE_MODEL_ID

    # Validate and load classifier
    try:
        classifier_package = validate_classifier_path(args.classifier_path)
        classifier_package['path'] = args.classifier_path
        
        # Update XGBoost classifier to use newer CUDA device configuration
        # Modern XGBoost uses 'device=cuda:X' and 'tree_method=hist' instead of 'tree_method=gpu_hist'
        classifier = classifier_package['model']
        if hasattr(classifier, 'set_params'):
            logger.info(f"Updating XGBoost classifier to use modern CUDA configuration on GPU {args.device_id}")
            try:
                # Set device to cuda:0 since CUDA_VISIBLE_DEVICES is set to args.device_id
                # This ensures XGBoost sees the correct GPU as device 0
                classifier.set_params(
                    device='cuda:0',  # Always use cuda:0 since CUDA_VISIBLE_DEVICES is set
                    tree_method='hist'
                )
                logger.info(f"  ✓ XGBoost device updated to: cuda:0 (via CUDA_VISIBLE_DEVICES={args.device_id})")
                logger.info(f"  ✓ XGBoost tree_method updated to: hist")
            except Exception as e:
                logger.warning(f"Could not update XGBoost device configuration: {e}")
                logger.warning("Proceeding with classifier's original configuration")
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Classifier validation failed: {e}")
        sys.exit(1)

    # Prepare inference configuration from classifier metadata
    try:
        inference_config = prepare_inference_config(classifier_package)
    except ValueError as e:
        logger.error(f"Configuration preparation failed: {e}")
        sys.exit(1)

    # Initialize model manager with loaded configuration
    logger.info("Initializing ModelManager...")
    model_config = {
        'DEVICE_ID': args.model_device_id,
        'HUGGINGFACE_MODEL_ID': huggingface_model_id,
        'TRANSFORMER_LENS_MODEL_NAME': huggingface_model_id,  # Use same as HF ID for TransformerLens
        'MODEL_NAME': model_name,
    }

    model_manager = ModelManager(model_config)
    model_manager.clear_gpu_memory()
    model_manager.load_model()
    model_manager.optimize_for_inference()

    logger.info(f"Model loaded: {model_name}")

    # Initialize token manager
    token_manager = TokenManager(
        model=model_manager.get_model(),
        max_answer_tokens=args.max_answer_tokens,
        model_dir=huggingface_model_id  # Use the loaded model ID
    )
    token_manager.setup_tokenizer_padding(model_manager.get_model())
    logger.info("TokenManager initialized")

    # Print banner
    print("\n" + "="*80)
    print("INTERACTIVE HALLUCINATION DETECTION")
    print("="*80)
    print(f"\nModel Configuration:")
    print(f"  Model Name: {model_name}")
    print(f"  HuggingFace ID: {huggingface_model_id}")
    print(f"\nClassifier Configuration:")
    print(f"  Layer:     {inference_config['layer']}")
    print(f"  Hook:      {inference_config['hook']}")
    print(f"  Scheme:    {inference_config['scheme']}")
    print(
        f"  Model:     {classifier_package['metadata'].get('classifier_name')}")
    print(
        f"  Threshold: {classifier_package['metadata'].get('optimal_threshold', 0.5):.4f}")
    print(f"\nDataset Configuration:")
    print(f"  Dataset CSV Path: {config.DATASET_CSV_PATH}")
    print(f"\nType 'quit', 'exit', or 'q' to exit")
    print("="*80 + "\n")

    # Interactive loop
    query_count = 0
    while True:
        try:
            # Get user question
            user_question = input(
                "\n📝 Enter your question (or 'quit' to exit): ").strip()

            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not user_question:
                print("⚠️  Question cannot be empty. Please try again.")
                continue

            query_count += 1
            print(f"\n[Query #{query_count}]")
            print("-" * 80)

            # Display user question
            print(f"\n📌 YOUR QUESTION:")
            print(f"   {user_question}")

            # Generate response from Llama
            print(f"\n🤖 GENERATING RESPONSE...")
            start_time = time.time()

            llama_response = generate_llama_response(
                question=user_question,
                model_manager=model_manager,
                token_manager=token_manager,
                max_answer_tokens=args.max_answer_tokens,
                logger=logger
            )

            generation_time = time.time() - start_time
            print(f"   [Generated in {generation_time:.2f}s]")

            print(f"\n🔤 LLAMA RESPONSE:")
            print(f"   {llama_response}")

            # Detect hallucination
            print(f"\n🔍 ANALYZING FOR HALLUCINATIONS...")
            analysis_start = time.time()

            results = run_inference_batch(
                questions=[user_question],
                answers=[llama_response],
                model_manager=model_manager,
                token_manager=token_manager,
                classifier_package=classifier_package,
                inference_config=inference_config,
                device_id=args.device_id,
                max_answer_tokens=args.max_answer_tokens,
                debug_verbose=args.debug
            )

            analysis_time = time.time() - analysis_start

            # Display results
            prediction = results['predictions'][0]
            hallucination_score = results['hallucination_scores'][0]
            confidence = results['confidence'][0]

            prediction_text = "HALLUCINATION ⚠️" if prediction == 1 else "ACCURATE ✓"

            print(f"\n✅ PREDICTION RESULT:")
            print(f"   Classification: {prediction_text}")
            print(f"   Hallucination Score: {hallucination_score:.4f}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   [Analysis time: {analysis_time:.2f}s]")

            # Additional context
            print(f"\n📊 CLASSIFIER DETAILS:")
            print(f"   Layer: {results['metadata']['layer']}")
            print(f"   Hook: {results['metadata']['hook']}")
            print(f"   Scheme: {results['metadata']['scheme']}")

            print("\n" + "-" * 80)

        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            print(f"❌ Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    print(f"\n📈 Session Summary: Processed {query_count} queries")
    print("="*80)


if __name__ == "__main__":
    main()
