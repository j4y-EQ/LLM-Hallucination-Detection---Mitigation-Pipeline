#!/usr/bin/env python
"""
Baseline generation script - runs ONCE and saves results for reuse.

This script generates baseline (no steering) outputs for a dataset and saves
all results persistently. Subsequent steering experiments can load this baseline
and compare against it.

METRICS CALCULATION NOTES:
- Hallucination scores are 0 (no hallucination), 1 (hallucination), 2 (API failure)
- Hallucination rate = count(score==1) / total samples
- API failures (score==2) are excluded from hallucination metrics
- Both absolute and relative reductions are calculated in steering experiments
- Relative reduction is the primary metric for selecting best parameter combinations

Usage:
    python -m steer.baseline_run \
        --device-id 0 \
        --dataset-path ./data/nq_swap.csv \
        --dataset-format non_mcq \
        --model llama \
        --batch-size 8 \
        --num-samples 1000 \
        --max-tokens 120 \
        --output-dir ./data/baseline_results_llama/individual/nqswap 

The script creates:
- baseline_config.json: Full configuration used for this run
- baseline_texts.pkl: Dict of {idx: answer_text}
- baseline_full_outputs.pkl: Dict of {idx: full_generation}
- baseline_prompts.pkl: Dict of {idx: prompt_info}
- baseline_evaluation.pkl: Dict of {idx: hallucination_score}
- baseline_results_detailed.json: Full results with all fields
"""

from tqdm import tqdm
from steer.utils.steer_files_utils import (
    save_baseline_data,
    save_baseline_detailed_results,
)
# Import validation functions early (before argparse, safe to import)
from steer.utils.steer_common_utils import (
    get_valid_dataset_formats,
    validate_dataset_format,
)
# Import other utilities after validation functions
from steer.utils.steer_common_utils import (
    parse_choices,
    DatasetHandler,
    extract_left_padded_answer,
    MODEL_CONFIGS,
    adapt_messages_for_model,
)
from steer.utils.eval_model_steer import set_eval_output_directory
from logger import consolidated_logger as logger
from helpers.token_manager import TokenManager
from helpers.model_manager import ModelManager
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set random seeds for reproducibility FIRST
np.random.seed(42)

# Parse arguments BEFORE torch is fully imported
parser = argparse.ArgumentParser(
    description='Generate baseline (no steering) outputs for a dataset')
parser.add_argument('--device-id', type=int, default=0,
                    help='GPU device ID to use')
parser.add_argument('--dataset-path', type=str, required=True,
                    help='Path to dataset CSV file')
parser.add_argument('--num-samples', type=int, default=1000,
                    help='Number of samples to evaluate')
parser.add_argument('--max-tokens', type=int, default=120,
                    help='Maximum tokens to generate')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size for generation')
parser.add_argument('--output-dir', type=str, default='./data/baseline_results',
                    help='Directory to save baseline results')
parser.add_argument('--dataset-format', type=str, required=True,
                    choices=get_valid_dataset_formats(),
                    help='Format of the input dataset (REQUIRED). Valid formats: mcq, non_mcq, mmlu, hellaswag')
parser.add_argument('--model', type=str, default='llama',
                    choices=['qwen', 'llama', 'gemma'],
                    help='Model to use: qwen, llama, or gemma')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

# ================================================================
# MODEL CONFIGURATION
# ================================================================
# Model configs are now imported from steer_common_utils

# Select model configuration
selected_model_config = MODEL_CONFIGS[args.model]
MODEL_NAME = selected_model_config['model_name']
HUGGINGFACE_MODEL_ID = selected_model_config['huggingface_model_id']
TRANSFORMER_LENS_MODEL_NAME = selected_model_config['transformer_lens_model_name']
MAX_ANSWER_TOKENS = args.max_tokens


def generate_batch_outputs(model, df_sample, batch_size, max_tokens, handler: DatasetHandler, dataset_format: str = 'mcq', desc="Generation"):
    """
    Generate baseline outputs without steering.

    Args:
        model: The model instance
        df_sample: DataFrame with samples
        batch_size: Batch size
        max_tokens: Max tokens to generate
        handler: DatasetHandler instance for formatting
        dataset_format: Format of the dataset ('mcq', 'non_mcq', or 'mmlu')
        desc: Progress bar description

    Returns:
        generated_texts: Dict {idx: extracted_answer_text}
        full_outputs: Dict {idx: full_generation_string}
        prompts_map: Dict {idx: prompt_info}
    """
    generated_texts = {}
    full_outputs = {}
    prompts_map = {}

    for idx_batch in tqdm(range(0, len(df_sample), batch_size), desc=desc):
        batch_indices = df_sample.index[idx_batch:idx_batch + batch_size]
        batch_prompts = []
        batch_idx_map = {}
        prompt_batch_idx = 0  # Track actual prompt index after filtering

        for i, idx in enumerate(batch_indices):
            row = df_sample.loc[idx]
            context = row.get('context', '')
            question = row.get('question', '')
            choices = parse_choices(row.get('choices', None))
            answer_key = row.get('answerKey', '')
            answer_text = row.get('answer', '')
            
            # Skip rows with invalid answer keys ONLY for MCQ/MMLU/HellaSwag formats
            # Non-MCQ format doesn't require answerKey
            is_mcq_format = (dataset_format in ['mcq', 'mmlu', 'hellaswag'])
            if is_mcq_format and (not answer_key or answer_key == ''):
                logger.warning(f"Skipping row {idx} with empty answer key")
                continue
            
            # Map actual prompt index in batch to original dataframe index
            batch_idx_map[prompt_batch_idx] = idx
            prompt_batch_idx += 1
            
            # Use handler to format right answer
            right_answer = handler.format_right_answer(answer_key, answer_text)

            # Use handler to create prompt
            qa_prompt_messages = handler.make_prompt(context, question, choices)
            
            # Adapt messages for model-specific requirements (e.g., Gemma doesn't support system role)
            adapted_messages = adapt_messages_for_model(qa_prompt_messages, args.model)
            
            # Apply the chat template to get the final string for the tokenizer
            final_prompt_str = model.tokenizer.apply_chat_template(
                adapted_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Store prompt info for later retrieval
            prompts_map[idx] = {
                'qa_prompt': final_prompt_str,
                'question': question,
                'context': context,
                'choices': choices,
                'right_answer': right_answer,
                'answerKey': answer_key,
                'answer_text': answer_text,
            }

            batch_prompts.append(final_prompt_str)

        # Skip batch if all rows were filtered out
        if not batch_prompts:
            continue

        # Tokenize with proper left-padding
        batch_prompts_tokenized = model.tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=model.cfg.n_ctx - max_tokens
        )
        batch_prompts_padded = batch_prompts_tokenized['input_ids'].to(
            model.cfg.device)
        attention_mask = batch_prompts_tokenized['attention_mask'].to(
            model.cfg.device)

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
        for i, idx in batch_idx_map.items():
            answer_text, full_text = extract_left_padded_answer(
                full_sequences, prompt_lengths, batch_prompts_padded,
                model.tokenizer, i
            )
            generated_texts[idx] = answer_text
            full_outputs[idx] = full_text

    return generated_texts, full_outputs, prompts_map


def main():
    """
    Main entry point - generate and save baseline results.
    
    Workflow:
    1. Validate dataset format argument (REQUIRED)
    2. Create timestamped output directory (BASELINE_{dataset}_{timestamp})
    3. Initialize logging to output directory
    4. Save experiment configuration (baseline_config.json)
    5. Load and validate dataset
    6. Initialize model and tokenizer (Llama/Qwen/Gemma)
    7. Generate baseline outputs without steering (batch processing)
    8. Evaluate hallucinations via Azure OpenAI API (gpt-4o)
    9. Save persistent artifacts:
       - baseline_texts.pkl: Extracted answer text per sample
       - baseline_full_outputs.pkl: Full model generations
       - baseline_prompts.pkl: Prompt components for reproducibility
       - baseline_evaluation.pkl: Hallucination scores (0/1/2)
       - baseline_results_detailed.json: Human-readable full results
    10. Calculate and log summary statistics
    
    The saved baseline serves as ground truth for all subsequent steering experiments,
    enabling fair comparison without re-running the expensive baseline generation.
    
    Raises:
        ValueError: If dataset_format validation fails
        FileNotFoundError: If dataset file doesn't exist
    """

    # --- STRICT VALIDATION: dataset_format is REQUIRED ---
    try:
        validate_dataset_format(args.dataset_format)
    except ValueError as e:
        logger.error(f"FATAL: Dataset format validation failed: {e}")
        raise

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment subdirectory
    dataset_basename = os.path.basename(args.dataset_path).replace('.csv', '')
    baseline_name = f"BASELINE_{dataset_basename}_{timestamp}"

    run_dir = os.path.join(args.output_dir, baseline_name)
    os.makedirs(run_dir, exist_ok=True)

    # Initialize logger
    logger.set_output_directory(run_dir)
    
    # Sync eval logger to same output directory
    set_eval_output_directory(run_dir)

    # Save configuration
    config_file = os.path.join(run_dir, "baseline_config.json")
    config_data = {
        'baseline_name': baseline_name,
        'dataset_path': args.dataset_path,
        'num_samples': args.num_samples,
        'max_tokens': args.max_tokens,
        'batch_size': args.batch_size,
        'device_id': args.device_id,
        'output_dir': run_dir,
        'timestamp': timestamp,
        'dataset_format': args.dataset_format,
        'model': args.model,
        'model_name': MODEL_NAME,
        'huggingface_model_id': HUGGINGFACE_MODEL_ID,
        'transformer_lens_model_name': TRANSFORMER_LENS_MODEL_NAME,
    }

    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("BASELINE GENERATION - NO STEERING")
    logger.info(f"{'='*80}")
    logger.info(f"Baseline Name: {baseline_name}")
    logger.info(f"Output Directory: {run_dir}")
    logger.info(f"Command line arguments:")
    logger.info(f"  device_id: {args.device_id}")
    logger.info(f"  dataset_path: {args.dataset_path}")
    logger.info(f"  num_samples: {args.num_samples}")
    logger.info(f"  max_tokens: {args.max_tokens}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  dataset_format: {args.dataset_format}")
    logger.info(f"  model: {args.model} ({HUGGINGFACE_MODEL_ID})")

    # --- CREATE DATASET HANDLER ---
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING DATASET HANDLER")
    logger.info(f"{'='*80}")
    
    # Determine use_mcq based on dataset_format
    use_mcq = (args.dataset_format in ['mcq', 'mmlu', 'hellaswag'])
    handler = DatasetHandler(use_mcq=use_mcq, dataset_format=args.dataset_format, logger=logger)
    logger.info(f"DatasetHandler initialized (use_mcq={use_mcq}, dataset_format={args.dataset_format})")

    # --- LOAD DATASET ---
    logger.info(f"\n{'='*80}")
    logger.info("LOADING DATASET")
    logger.info(f"{'='*80}")

    df = handler.load_dataset(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        dataset_format=args.dataset_format
    )
    
    df_sample = df

    logger.info(
        f"Using {len(df_sample)} samples for baseline")

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

    # --- INITIALIZE TOKEN MANAGER ---
    token_manager = TokenManager(
        model=model,
        max_answer_tokens=args.max_tokens,
        model_dir=TRANSFORMER_LENS_MODEL_NAME
    )
    token_manager.setup_tokenizer_padding(model)
    logger.info("TokenManager initialized")

    # --- GENERATE BASELINE OUTPUTS ---
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING BASELINE OUTPUTS FOR {len(df_sample)} SAMPLES")
    logger.info(f"{'='*80}")

    logger.info(f"\nGenerating baseline outputs...")
    baseline_texts, baseline_outputs, prompts = generate_batch_outputs(
        model, df_sample, args.batch_size, MAX_ANSWER_TOKENS, handler,
        dataset_format=args.dataset_format,
        desc="Baseline generation"
    )

    logger.info(f"✓ Generated {len(baseline_outputs)} baseline outputs")

    # --- EVALUATE BASELINE OUTPUTS ---
    logger.info(f"\nEvaluating baseline outputs...")
    baseline_evaluation_pairs = [(prompts[idx]['right_answer'], baseline_texts[idx], {
                                  'idx': idx}) for idx in prompts.keys()]
    
    # Use handler's batch evaluation method (centralized logic)
    baseline_evaluation_results = handler.batch_evaluate_answers(
        baseline_evaluation_pairs, max_workers=20)

    baseline_halluc_by_idx = {}
    api_failures = 0
    for (gt, cand, metadata), eval_result in zip(baseline_evaluation_pairs, baseline_evaluation_results):
        halluc_score, _ = eval_result
        baseline_halluc_by_idx[metadata['idx']] = halluc_score
        if halluc_score == 2:
            api_failures += 1
    logger.info(f"✓ Completed baseline evaluation")
    
    # Log API failures if any occurred
    if api_failures > 0:
        api_failure_rate = (api_failures / len(baseline_halluc_by_idx)) * 100
        logger.warning(f"API Evaluation Failures: {api_failures}/{len(baseline_halluc_by_idx)} samples ({api_failure_rate:.1f}%) - "
                       f"Caused by Azure API rate limits, content filters, or service errors")

    # --- SAVE BASELINE DATA ---
    logger.info(f"\n{'='*80}")
    logger.info("SAVING BASELINE DATA")
    logger.info(f"{'='*80}")

    save_baseline_data(
        baseline_texts=baseline_texts,
        baseline_outputs=baseline_outputs,
        prompts=prompts,
        baseline_evaluation=baseline_halluc_by_idx,
        run_dir=run_dir,
        logger=logger
    )

    save_baseline_detailed_results(
        baseline_texts=baseline_texts,
        baseline_outputs=baseline_outputs,
        prompts=prompts,
        baseline_evaluation=baseline_halluc_by_idx,
        run_dir=run_dir,
        logger=logger
    )

    # --- SUMMARY STATISTICS ---
    logger.info(f"\n{'='*80}")
    logger.info("BASELINE SUMMARY STATISTICS")
    logger.info(f"{'='*80}")

    baseline_halluc_count = sum(
        1 for score in baseline_halluc_by_idx.values() if score == 1)
    api_failures_summary = sum(
        1 for score in baseline_halluc_by_idx.values() if score == 2)
    valid_samples = len(baseline_halluc_by_idx) - api_failures_summary
    baseline_halluc_rate = baseline_halluc_count / valid_samples if valid_samples > 0 else 0

    logger.info(f"Total samples: {len(baseline_halluc_by_idx)}")
    logger.info(f"Hallucinations (score=1): {baseline_halluc_count}")
    logger.info(f"Hallucination rate: {baseline_halluc_rate*100:.2f}%")
    if api_failures_summary > 0:
        logger.warning(f"API Evaluation Failures (score=2): {api_failures_summary} - Excluded from hallucination metrics")

    logger.info(f"\n{'='*80}")
    logger.info("BASELINE GENERATION COMPLETE")
    logger.info(f"Baseline directory: {run_dir}")
    logger.info(f"All results saved and ready for steering experiments")
    logger.info(f"{'='*80}\n")

    return run_dir


if __name__ == "__main__":
    baseline_dir = main()
