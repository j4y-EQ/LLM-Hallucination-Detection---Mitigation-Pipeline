#!/usr/bin/env python
"""

Steering experiment script - runs steering experiments using saved baseline.

This script loads pre-generated baseline results and applies steering interventions
to test their effect. It runs multiple steering strengths and collects independent
hallucination statistics for each steering strength.

METRICS CALCULATION NOTES:
- Hallucination scores are 0 (no hallucination), 1 (hallucination), 2 (API failure)
- Per-strength metrics: hallucination count, non-hallucination count, API failure count
- Hallucination rate = count(score==1) / total samples (API failures excluded)
- TENTATIVE BEST STRENGTH is identified by highest non-hallucination count
- No baseline comparison metrics are calculated

Usage Examples:

1. Quick test run (10 samples, 2 strengths):
   python -m steer.steering_experiment \
       --device-id 0 \
       --baseline-dir ./data/baseline_results/ensembled_nqswap/BASELINE_ENSEMBLE_VOTED_20251215_013922 \
       --iti-config-path ./data/ITI/steering_vector/iti_intervention_config_top10.pkl \
       --steering-strength 0.5 1.0 \
       --output-dir ./data/ITI/test/evalonnqswap \
       --num-samples 10

2. Full experiment (multiple strengths, Llama-3):
   python -m steer.steering_experiment \
       --device-id 0 \
       --baseline-dir ./data/baseline_results_llama/ensembled_nqswap/BASELINE_20251215 \
       --iti-config-path ./data/ITI/steering_vector/llama/iti_intervention_config_top30.pkl \
       --steering-strength 0.5 1.0 2.0 2.5 3.0 5.0 10.0 \
       --output-dir ./data/ITI/steering_experiment_llama/nqswap \
       --num-samples 1000

3. Qwen model with fine-grained alpha sweep:
   python -m steer.steering_experiment \
       --device-id 1 \
       --baseline-dir ./data/baseline_results_qwen2.5_7b/ensembled_mmlu/BASELINE_20260128 \
       --iti-config-path ./data/ITI/steering_vector/qwen2.5_7b/iti_intervention_config_top15.pkl \
       --steering-strength 1.0 1.5 2.0 2.25 2.5 2.75 3.0 3.5 \
       --output-dir ./data/ITI/steering_experiment_qwen/mmlu

4. HellaSwag evaluation with Gemma:
   python -m steer.steering_experiment \
       --device-id 2 \
       --baseline-dir ./data/baseline_results_gemma/ensembled_hellaswag/BASELINE_20260119 \
       --iti-config-path ./data/ITI/steering_vector/gemma/iti_intervention_config_top13.pkl \
       --steering-strength 27.0 27.5 28.0 28.5 29.0 30.0 \
       --output-dir ./data/ITI/steering_experiment_gemma/hellaswag

Output Files:
- consolidated_results.json: All results + per-strength statistics
- independent_hallucination_analysis.csv: Statistical summary table
- independent_hallucination_viz.png: Bar chart of results
- tentative_best_strength.txt: Best alpha selection
- detailed_results_strength_X.X.txt: Human-readable per-sample results
- steering_experiment.log: Detailed execution log

"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set random seeds for reproducibility FIRST
np.random.seed(42)

# Parse arguments BEFORE torch is fully imported
parser = argparse.ArgumentParser(description='Apply ITI steering vectors using pre-generated baseline')
parser.add_argument('--device-id', type=int, default=0, help='GPU device ID to use')
parser.add_argument('--baseline-dir', type=str, required=True, help='Directory containing baseline results')
parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation')
parser.add_argument('--steering-strength', type=float, nargs='+', required=True,
                    help='Steering strength (alpha) values to test')
parser.add_argument('--iti-config-path', type=str, required=True,
                    help='Path to ITI intervention config (.pkl file)')
parser.add_argument('--output-dir', type=str, default='./data/steering_results', help='Directory to save results')
parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples in dataset')
parser.add_argument('--model', type=str, default='llama',
                    choices=['qwen', 'llama', 'gemma'],
                    help='Model to use: qwen, llama, or gemma (should match baseline)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)



from helpers.model_manager import ModelManager
from helpers.token_manager import TokenManager
from logger import consolidated_logger as logger
from steer.utils.dataset_formats import VALID_DATASET_FORMATS, MCQ_FORMATS
from steer.utils.steer_common_utils import (
    DatasetHandler,
    extract_left_padded_answer,
    load_baseline_config,
    MODEL_CONFIGS,
    validate_dataset_format,
    extract_answer_from_batch
)
from steer.utils.steer_files_utils import (
    load_baseline_data,
    create_independent_hallucination_analysis,
    create_independent_hallucination_visualization,
    save_tentative_best_strength,
    save_consolidated_results,
    save_detailed_text_results,
)
from tqdm import tqdm

# ================================================================
# CONFIGURATION
# ================================================================
MAX_ANSWER_TOKENS = args.max_tokens


def get_hook_name_for_layer(hook_base_name: str, layer_idx: int) -> str:
    """Convert base hook name to full hook name for specific layer"""
    if hook_base_name in ["hook_embed"]:
        return hook_base_name
    elif "." in hook_base_name:  # Sub-component hooks like attn.hook_z
        return f"blocks.{layer_idx}.{hook_base_name}"
    else:  # Direct layer hooks
        return f"blocks.{layer_idx}.{hook_base_name}"


def load_iti_config(config_path: str, device: str, dtype: torch.dtype) -> Tuple[Dict, str, int, int]:
    """
    Load ITI configuration from .pkl file and convert vectors to tensors.
    
    Args:
        config_path: Path to .pkl file
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Returns:
        Tuple containing:
        - dict: Intervention vectors, (e.g., {'layer_10': [...]})
        - str: Base hook name (e.g., "attn.hook_z")
        - int: Number of heads
        - int: Dimension of head
    """
    logger.info(f"Loading ITI intervention config from: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"ITI config file not found: {config_path}")
        raise FileNotFoundError(f"ITI config not found at {config_path}")
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    iti_vectors_by_layer = config.get('intervention_vectors', {})
    hook_name = config.get('hook', 'attn.hook_z')
    n_heads = config.get('n_heads', 32)
    d_head = config.get('d_head', 128)

    if not iti_vectors_by_layer:
        logger.warning(f"No 'intervention_vectors' found in config file: {config_path}")
    
    # Convert numpy arrays to tensors on the correct device
    total_vectors = 0
    for layer_key, head_infos in iti_vectors_by_layer.items():
        for head_info in head_infos:
            if 'direction' in head_info and isinstance(head_info['direction'], np.ndarray):
                head_info['direction'] = torch.from_numpy(head_info['direction']).to(device).to(dtype)
            if 'projection_std' in head_info:
                head_info['projection_std'] = float(head_info['projection_std'])
            total_vectors += 1

    logger.info(f"ITI config loaded: {total_vectors} intervention vectors across {len(iti_vectors_by_layer)} layers.")
    logger.info(f"Hook: {hook_name}, Head Dims: {n_heads}x{d_head}")
    
    return iti_vectors_by_layer, hook_name, n_heads, d_head


def create_iti_hook(antidote: torch.Tensor, head_idx: int, logger):
    """
    Create a steering hook function for a *single head*.
    """
    
    def iti_hook_fn(activation, hook):
        """
        Intervention hook that adds the antidote vector to a specific head
        at the last token position.
        
        Expected activation shape: [batch, seq, n_heads, d_head]
        """
        try:
            activation[:, -1, head_idx, :] = activation[:, -1, head_idx, :] + antidote
            
        except Exception as e:
            logger.warning(f"Error in ITI hook for head {head_idx}: {e}")
            logger.warning(f"Activation shape: {activation.shape}, Antidote shape: {antidote.shape}")
        
        return activation
    
    return iti_hook_fn


def generate_steered_batch_outputs(model, prompts: Dict, batch_size: int, max_tokens: int, desc: str = "Steered generation"):
    """
    Generate steered outputs using pre-computed prompts.
    
    Args:
        model: The model instance
        prompts: Dict of {idx: prompt_info} with 'qa_prompt' field
        batch_size: Batch size for processing
        max_tokens: Max tokens to generate
        desc: Progress bar description
        
    Returns:
        generated_texts: Dict {idx: extracted_answer_text}
        full_outputs: Dict {idx: full_generation_string}
    """
    generated_texts = {}
    full_outputs = {}
    
    prompt_indices = sorted(prompts.keys())
    
    for idx_batch in tqdm(range(0, len(prompt_indices), batch_size), desc=desc):
        batch_indices = prompt_indices[idx_batch:idx_batch + batch_size]
        batch_prompts = [prompts[idx]['qa_prompt'] for idx in batch_indices]
        
        # Tokenize with proper left-padding
        batch_prompts_tokenized = model.tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=model.cfg.n_ctx - max_tokens
        )
        batch_prompts_padded = batch_prompts_tokenized['input_ids'].to(model.cfg.device)
        attention_mask = batch_prompts_tokenized['attention_mask'].to(model.cfg.device)
        
        with torch.inference_mode():
            # Generate
            full_sequences = model.generate(
                input=batch_prompts_padded,
                max_new_tokens=max_tokens,
                do_sample=False,
                stop_at_eos=True,
                eos_token_id=model.tokenizer.eos_token_id,
                use_past_kv_cache=True,
                padding_side='left',
                return_type='tokens',
                verbose=False
            )
            
            # Get prompt lengths from attention mask
            prompt_lengths = batch_prompts_tokenized['attention_mask'].sum(dim=1)
        
        # Extract answers with proper left-padding accounting
        for i, idx in enumerate(batch_indices):
            answer_text, full_text = extract_left_padded_answer(
                full_sequences, prompt_lengths, batch_prompts_padded,
                model.tokenizer, i
            )
            generated_texts[idx] = answer_text
            full_outputs[idx] = full_text

    return generated_texts, full_outputs


def main():
    """Main entry point - load baseline and run steering experiments"""
    
    # Resolve paths to absolute paths (handle relative paths correctly)
    args.baseline_dir = os.path.abspath(args.baseline_dir)
    args.iti_config_path = os.path.abspath(args.iti_config_path)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory structure (same for both grid search and normal mode)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment subdirectory with descriptive name
    iti_config_basename = os.path.basename(args.iti_config_path).replace('.pkl', '')
    baseline_name = os.path.basename(args.baseline_dir.rstrip('/'))
    experiment_name = f"STEERING_{baseline_name}_{iti_config_basename}_{timestamp}"
    
    run_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for different output types
    results_dir = os.path.join(run_dir, "results")
    analysis_dir = os.path.join(run_dir, "analysis")
    config_dir = os.path.join(run_dir, "config")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # Initialize logger
    logger.set_output_directory(run_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info("ITI STEERING EXPERIMENT - USING SAVED BASELINE")
    logger.info(f"{'='*80}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Experiment Directory: {run_dir}")
    logger.info(f"Command line arguments:")
    logger.info(f"  device_id: {args.device_id}")
    logger.info(f"  baseline_dir: {args.baseline_dir}")
    logger.info(f"  max_tokens: {args.max_tokens}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  steering_strength (alphas): {args.steering_strength}")
    logger.info(f"  iti_config_path: {args.iti_config_path}")
    logger.info(f"  num_samples: {args.num_samples}")
    
    # --- LOAD BASELINE DATA ---
    logger.info(f"\n{'='*80}")
    logger.info("LOADING BASELINE DATA")
    logger.info(f"{'='*80}")
    
    logger.info(f"Looking for baseline directory at: {args.baseline_dir}")
    if not os.path.isdir(args.baseline_dir):
        logger.error(f"Baseline directory not found: {args.baseline_dir}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Does path exist? {os.path.exists(args.baseline_dir)}")
        raise FileNotFoundError(f"Baseline directory not found: {args.baseline_dir}")
    
    baseline_texts, baseline_outputs, prompts, baseline_evaluation = load_baseline_data(
        args.baseline_dir, logger, num_samples=args.num_samples
    )
    
    logger.info(f"✓ Loaded baseline data for {len(prompts)} samples")
    
    # Load and validate baseline config with strict dataset_format validation
    baseline_config = load_baseline_config(args.baseline_dir)
    if baseline_config is None:
        logger.error(f"baseline_config.json not found in baseline directory: {args.baseline_dir}")
        raise FileNotFoundError(f"baseline_config.json must exist in baseline directory. Ensure baseline was created with current pipeline version.")
    
    baseline_dataset_path = baseline_config.get('dataset_path')
    baseline_dataset_format = baseline_config.get('dataset_format')
    baseline_model = baseline_config.get('model')
    
    # Strict validation: dataset_format must be present and valid
    if baseline_dataset_format is None:
        logger.error("baseline_config.json must contain 'dataset_format' field")
        logger.error(f"Available keys in baseline_config.json: {list(baseline_config.keys())}")
        raise ValueError("baseline_config.json must contain 'dataset_format' field. Ensure baseline was created with current pipeline version.")
    
    # Use centralized validation function
    try:
        baseline_dataset_format = validate_dataset_format(baseline_dataset_format)
    except ValueError as e:
        logger.error(f"Dataset format validation failed: {e}")
        raise
    
    # Validate model from baseline config
    if baseline_model is None:
        logger.error("baseline_config.json must contain 'model' field")
        logger.error(f"Available keys in baseline_config.json: {list(baseline_config.keys())}")
        raise ValueError("baseline_config.json must contain 'model' field. Ensure baseline was created with current pipeline version.")
    
    if baseline_model not in MODEL_CONFIGS:
        logger.error(f"Invalid model in baseline_config.json: {baseline_model}")
        logger.error(f"Valid models are: {list(MODEL_CONFIGS.keys())}")
        raise ValueError(f"Invalid model in baseline_config.json: '{baseline_model}'. Must be one of: {list(MODEL_CONFIGS.keys())}")
    
    # Check if user-specified model matches baseline model
    if args.model != baseline_model:
        logger.warning(f"User specified model '{args.model}' but baseline was created with model '{baseline_model}'")
        logger.warning("This may cause inconsistencies. Consider using --model {baseline_model} to match baseline")
    
    # Get model configuration
    selected_model_config = MODEL_CONFIGS[baseline_model]
    MODEL_NAME = selected_model_config['model_name']
    HUGGINGFACE_MODEL_ID = selected_model_config['huggingface_model_id']
    TRANSFORMER_LENS_MODEL_NAME = selected_model_config['transformer_lens_model_name']
    
    baseline_use_mcq = (baseline_dataset_format in MCQ_FORMATS)
    
    # Save experiment config (after loading baseline config to get baseline_dataset_path)
    config_file = os.path.join(config_dir, "steering_config.json")
    config_data = {
        'experiment_name': experiment_name,
        'baseline_dir': args.baseline_dir,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'device_id': args.device_id,
        'iti_config_path': args.iti_config_path,
        'steering_strengths': args.steering_strength,
        'output_dir': run_dir,
        'num_samples': args.num_samples,
        'timestamp': timestamp,
        'baseline_dataset_path': baseline_dataset_path,
        'baseline_dataset_format': baseline_dataset_format,
        'model': baseline_model,
        'model_name': MODEL_NAME,
        'huggingface_model_id': HUGGINGFACE_MODEL_ID,
        'transformer_lens_model_name': TRANSFORMER_LENS_MODEL_NAME,
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"✓ Baseline config validated successfully")
    logger.info(f"✓ Baseline dataset format: {baseline_dataset_format}")
    logger.info(f"✓ Using MCQ mode: {baseline_use_mcq}")
    
    # --- LOAD MODEL ---
    logger.info(f"\n{'='*80}")
    logger.info("LOADING MODEL")
    logger.info(f"{'='*80}")
    
    config = {
        'DEVICE_ID': args.device_id,
        'HUGGINGFACE_MODEL_ID': HUGGINGFACE_MODEL_ID,
        'TRANSFORMER_LENS_MODEL_NAME': TRANSFORMER_LENS_MODEL_NAME,
        'MODEL_NAME': MODEL_NAME,
    }
    
    model_manager = ModelManager(config)
    model_manager.check_initial_gpu_memory()
    model_manager.clear_gpu_memory()
    
    logger.info(f"Loading {MODEL_NAME}...")
    model_manager.load_model()
    model_manager.optimize_for_inference()
    
    model = model_manager.get_model()
    logger.info("Model loaded successfully!")
    
    # --- LOAD ITI CONFIG ---
    logger.info(f"\n{'='*80}")
    logger.info("LOADING ITI CONFIGURATION")
    logger.info(f"{'='*80}")
    
    dtype = torch.bfloat16 if hasattr(model, 'cfg') and hasattr(model.cfg, 'dtype') else torch.float32
    device = model.cfg.device if hasattr(model, 'cfg') else "cuda:0"
    
    iti_vectors_by_layer, hook_base_name, n_heads, d_head = load_iti_config(
        args.iti_config_path, device, dtype
    )
    
    # --- INITIALIZE TOKEN MANAGER ---
    token_manager = TokenManager(
        model=model,
        max_answer_tokens=args.max_tokens,
        model_dir=TRANSFORMER_LENS_MODEL_NAME
    )
    token_manager.setup_tokenizer_padding(model)
    logger.info("TokenManager initialized")
    
    # --- CREATE DATASET HANDLER (REUSE FOR ALL STRENGTHS) ---
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING DATASET HANDLER")
    logger.info(f"{'='*80}")
    handler = DatasetHandler(use_mcq=baseline_use_mcq, dataset_format=baseline_dataset_format, logger=logger)
    logger.info(f"DatasetHandler initialized (use_mcq={baseline_use_mcq}, dataset_format={baseline_dataset_format})")
    
    # --- PROCESS STEERING EXPERIMENTS ---
    logger.info(f"\n{'='*80}")
    logger.info("PROCESSING STEERING EXPERIMENTS")
    logger.info(f"{'='*80}")
    
    all_results_by_strength = {}
    
    # Loop through each steering strength
    for steering_strength in args.steering_strength:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING STEERING STRENGTH (alpha): {steering_strength}")
        logger.info(f"{'='*80}")
        
        # Generate steered outputs for this strength
        logger.info(f"\nGenerating steered outputs for alpha={steering_strength}...")
        
        steered_texts = {}
        steered_outputs = {}
        
        try:
            # --- REGISTER ALL ITI HOOKS for this alpha ---
            hooks_to_add = []
            for layer_key, head_infos in iti_vectors_by_layer.items():
                try:
                    layer_idx = int(layer_key.split('_')[1])
                except Exception as e:
                    logger.warning(f"Could not parse layer index from key '{layer_key}': {e}. Skipping.")
                    continue
                
                # Get the full hook point name (e.g., "blocks.10.attn.hook_z")
                hook_point_name = get_hook_name_for_layer(hook_base_name, layer_idx)
                
                for head_info in head_infos:
                    head_idx = head_info['head']
                    direction_tensor = head_info['direction']
                    projection_std = head_info['projection_std']
                    alpha = steering_strength
                    
                    # Calculate the final ITI vector (antidote)
                    antidote = direction_tensor * projection_std * alpha
                    
                    # Create the hook function for this specific head
                    hook_fn = create_iti_hook(antidote, head_idx, logger)
                    
                    # Add it to our list
                    hooks_to_add.append((hook_point_name, hook_fn))
            
            logger.info(f"Registering {len(hooks_to_add)} ITI hooks for alpha={steering_strength}...")
            for name, fn in hooks_to_add:
                model.add_hook(name, fn)
            logger.info("All hooks registered.")
            # --- END HOOK REGISTRATION ---

            # Generate steered outputs
            steered_texts, steered_outputs = generate_steered_batch_outputs(
                model, prompts, args.batch_size, MAX_ANSWER_TOKENS,
                desc=f"Steered generation (α={steering_strength})"
            )
            
        finally:
            # Clean up: reset all hooks after this strength
            model.reset_hooks()
            logger.info(f"Hooks reset after alpha={steering_strength}")
        
        logger.info(f"✓ Generated {len(steered_texts)} steered outputs")
        
        # Evaluate steered outputs ONLY for this strength
        logger.info(f"\nEvaluating steered outputs for alpha={steering_strength}...")
        
        steered_evaluation_pairs = [(prompts[idx]['right_answer'], steered_texts[idx], {'idx': idx}) for idx in prompts.keys()]
        
        # Use handler's batch evaluation method (centralized logic)
        steered_evaluation_results = handler.batch_evaluate_answers(
            steered_evaluation_pairs, max_workers=10)
        
        steered_halluc_by_idx = {}
        steered_api_failures = 0
        for (gt, cand, metadata), eval_result in zip(steered_evaluation_pairs, steered_evaluation_results):
            halluc_score, _ = eval_result
            steered_halluc_by_idx[metadata['idx']] = halluc_score
            if halluc_score == 2:
                steered_api_failures += 1
        
        logger.info(f"✓ Completed evaluation for alpha={steering_strength}")
        if steered_api_failures > 0:
            steered_api_failure_rate = (steered_api_failures / len(steered_halluc_by_idx)) * 100
            logger.warning(f"Steered API Failures: {steered_api_failures}/{len(steered_halluc_by_idx)} samples ({steered_api_failure_rate:.1f}%)")
        
        # Aggregate results for this strength
        all_results = []
        for idx in prompts.keys():
            result = {
                'sample_idx': idx,
                'qa_prompt': prompts[idx]['qa_prompt'],
                'question': prompts[idx]['question'],
                'context': prompts[idx]['context'],
                'choices': prompts[idx].get('choices'),
                'answerKey': prompts[idx].get('answerKey', ''),
                'answer_text': prompts[idx].get('answer_text', ''),
                'right_answer': prompts[idx]['right_answer'],
                'baseline_answer': baseline_texts[idx],         # Extracted answer
                'steered_answer': steered_texts[idx],           # Extracted answer
                'baseline_output': baseline_outputs[idx],
                'steered_output': steered_outputs[idx],
                'baseline_hallucination': baseline_evaluation.get(idx, 2),
                'steered_hallucination': steered_halluc_by_idx.get(idx, 2),
            }
            all_results.append(result)
    
        all_results_by_strength[steering_strength] = all_results
    
    # --- INDEPENDENT HALLUCINATION ANALYSIS ---
    logger.info(f"\n{'='*80}")
    logger.info("INDEPENDENT HALLUCINATION ANALYSIS")
    logger.info(f"{'='*80}")
    
    analysis_data = create_independent_hallucination_analysis(all_results_by_strength, run_dir, logger)
    create_independent_hallucination_visualization(analysis_data, run_dir, logger)
    save_tentative_best_strength(analysis_data, run_dir, logger)
    
    
    # --- SAVE RESULTS FOR EACH STRENGTH ---
    logger.info(f"\n{'='*80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*80}")

    save_consolidated_results(all_results_by_strength, results_dir, args, logger, baseline_dataset_path)
    save_detailed_text_results(all_results_by_strength, results_dir, args, logger, baseline_dataset_path, baseline_dataset_format)
    logger.info(f"\n{'='*80}")
    logger.info("STEERING EXPERIMENT COMPLETE")
    logger.info(f"Experiment saved to: {run_dir}")
    logger.info(f"Results: {results_dir}/")
    logger.info(f"Analysis: {analysis_dir}/")
    logger.info(f"Config: {config_dir}/")
    logger.info(f"{'='*80}\n")
    
    return analysis_data

if __name__ == "__main__":
    results = main()
