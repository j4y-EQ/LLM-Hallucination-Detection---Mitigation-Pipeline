#!/usr/bin/env python
"""
Capture attention activations ('z' hook) for ITI experiments on faitheval dataset.

This script:
1. Loads the faitheval dataset (context, question, choices)
2. Generates outputs using the model
3. Captures attention activations at the 'z' hook for all layers
4. Saves activations in HDF5 format for downstream ITI processing

HDF5 Output Structure:
----------------------
activations_checkpoint_{idx}.h5
├── Attributes:
│   ├── checkpoint_idx: int - Checkpoint batch number
│   ├── num_samples: int - Total samples in checkpoint
│   ├── timestamp: str - Creation timestamp
│   └── dataset_format: str - Format type (mcq, non_mcq, mmlu, hellaswag)
│
├── sample_{row_idx}/  (one group per sample)
│   ├── metadata/
│   │   ├── Attributes:
│   │   │   ├── row_idx: int - Original dataset row index
│   │   │   ├── right_answer: str - Correct answer text
│   │   │   ├── generated_text: str - Model's generated answer
│   │   │   ├── prompt_length: int - Tokens in prompt
│   │   │   ├── last_gen_token_pos: int - Position of last generated token
│   │   │   ├── is_hallucination: bool - Hallucination flag (if evaluated)
│   │   │   ├── hallucination_status: str - Status message
│   │   │   └── prompt_info: str - JSON string with full prompt details
│   │   
│   └── activations_last_gen_token/
│       ├── layer_0: Dataset[n_heads, d_head] - Activations at layer 0
│       ├── layer_1: Dataset[n_heads, d_head] - Activations at layer 1
│       └── ... (one dataset per model layer)
│
└── sample_{next_row_idx}/ ...

Shape Details:
- Each layer dataset: [n_heads, d_head]
  * Llama-3-8B: [32, 128] (32 heads, 128-dim per head)
  * Qwen2.5-7B: [28, 128] (28 heads, 128-dim per head)
  * Gemma-2-9B: [16, 256] (16 heads, 256-dim per head)
- Only activations from the LAST generated token are stored (most predictive for steering)

Usage:
    python -m steer.grab_activation_ITI_attnhookz \
        --model llama \
        --device-id 0 \
        --dataset-path ./data/faitheval_counterfact.csv \
        --dataset-format mcq \
        --num-samples 1000 \
        --batch-size 8 \
        --max-tokens 120 \
        --output-dir ./data/ITI/activations

"""

import os
import sys
import argparse
import json
import pickle
import torch
import numpy as np
import pandas as pd
import h5py
import gc
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

# Import centralized validation function early (before argparse, safe to import)
from steer.utils.steer_common_utils import get_valid_dataset_formats, validate_dataset_format

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parse arguments BEFORE imports
parser = argparse.ArgumentParser(
    description='Capture attention activations (z hook) for ITI experiments')
parser.add_argument('--device-id', type=int, default=0,
                    help='GPU device ID to use')
parser.add_argument('--dataset-path', type=str,
                    default='./data/faitheval_counterfact.csv', help='Path to dataset CSV')
parser.add_argument('--num-samples', type=int, default=100,
                    help='Number of samples to evaluate')
parser.add_argument('--max-tokens', type=int, default=120,
                    help='Maximum tokens to generate')
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size for generation')
parser.add_argument('--output-dir', type=str, default='./data/ITI/activations',
                    help='Directory to save activations')
parser.add_argument('--checkpoint-freq', type=int, default=10,
                    help='Save and evaluate results every N batches')
parser.add_argument('--start-layer', type=int, default=None,
                    help='Start layer for activation capture')
parser.add_argument('--end-layer', type=int, default=None,
                    help='End layer for activation capture')
parser.add_argument('--dataset-format', type=str, required=True,
                    choices=get_valid_dataset_formats(),
                    help='Format of the input dataset (REQUIRED). Valid formats: mcq, non_mcq, mmlu, hellaswag')
parser.add_argument('--model', type=str, default='llama',
                    choices=['qwen', 'llama', 'gemma'],
                    help='Model to use: qwen, llama, or gemma')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

# Now import after CUDA_VISIBLE_DEVICES is set
from helpers.model_manager import ModelManager
from helpers.token_manager import TokenManager
from steer.utils.eval_model_steer import batch_judge_answers_mcq
from helpers.eval_model import batch_judge_answers
from logger import consolidated_logger as logger
from steer.utils.steer_common_utils import (
    parse_choices,
    make_qa_prompt,
    DatasetHandler,
    MODEL_CONFIGS,
    adapt_messages_for_model,
)

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
ATTENTION_HOOK = "z"  # Only capture at attention output hook


# ================================================================
# TERMINATOR TOKEN UTILITIES (MODEL-SPECIFIC)
# ================================================================

def build_terminators_for_model(model, model_type: str) -> list:
    """
    Build terminator token IDs based on the model type.
    
    Different models use different special tokens for generation termination:
    - Llama: Uses <|eot_id|> as a custom end-of-text token
    - Qwen: Uses standard eos_token_id (no special eot_id)
    - Gemma: Uses standard eos_token_id
    
    Args:
        model: The transformer-lens model
        model_type: One of 'llama', 'qwen', 'gemma'
        
    Returns:
        List of valid terminator token IDs (None values filtered out)
    """
    terminators = []
    
    # Always include the standard EOS token if it exists
    if model.tokenizer.eos_token_id is not None:
        terminators.append(model.tokenizer.eos_token_id)
    
    # Add model-specific special tokens
    if model_type == 'llama':
        # Llama 3 uses <|eot_id|> as a special end-of-text token
        eot_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id is not None and eot_token_id not in terminators:
            terminators.append(eot_token_id)
    elif model_type == 'qwen':
        # Qwen uses standard eos_token_id (no special eot_id)
        # Already added above
        pass
    elif model_type == 'gemma':
        # Gemma uses standard eos_token_id (no special eot_id)
        # Already added above
        pass
    
    # Filter out any None values and ensure we have at least one terminator
    terminators = [t for t in terminators if t is not None]
    
    if not terminators:
        logger.warning(f"No valid terminators found for model type {model_type}. Using pad token as fallback.")
        # Fallback: use pad_token_id if available
        if model.tokenizer.pad_token_id is not None:
            terminators = [model.tokenizer.pad_token_id]
        else:
            raise ValueError(f"Cannot determine valid terminator tokens for model {model_type}")
    
    logger.info(f"Built terminators for {model_type}: {terminators}")
    return terminators

# ================================================================
# BATCH CHECKPOINT MANAGER
# ================================================================

class BatchCheckpointManager:
    """
    Manages incremental batch checkpointing: accumulates results, triggers saves
    every N batches, and handles memory cleanup after checkpoint saves.
    """

    def __init__(self, checkpoint_freq: int):
        """
        Args:
            checkpoint_freq: Save and evaluate every N batches
        """
        if checkpoint_freq <= 0:
            raise ValueError(f"checkpoint_freq must be > 0, got {checkpoint_freq}")
        
        self.checkpoint_freq = checkpoint_freq
        self.batch_count = 0
        self.checkpoint_batch_count = 0
        self.current_checkpoint_results = {}  # Dict[row_idx -> result_entry]
        self.current_checkpoint_batch_results = []  # List of results for evaluation
        
        logger.info(f"BatchCheckpointManager initialized with checkpoint frequency: {checkpoint_freq} batches")

    def add_batch_result(self, batch_results: Dict[int, Dict]) -> None:
        """
        Add results from one batch to the current checkpoint accumulator.
        
        Args:
            batch_results: Dict mapping row_idx to result_entry dicts from one batch
        """
        self.current_checkpoint_results.update(batch_results)
        # Collect result entries for later evaluation (will be evaluated as batch)
        self.current_checkpoint_batch_results.extend(batch_results.values())
        self.batch_count += 1
        self.checkpoint_batch_count += 1

    def should_checkpoint(self) -> bool:
        """Check if we've accumulated enough batches to trigger a checkpoint save."""
        return self.checkpoint_batch_count >= self.checkpoint_freq

    def get_checkpoint_results(self) -> Tuple[Dict[int, Dict], List[Dict]]:
        """
        Return accumulated results for this checkpoint (for saving/evaluation).
        
        Returns:
            Tuple of (checkpoint_results_dict, checkpoint_batch_results_list)
        """
        return self.current_checkpoint_results, self.current_checkpoint_batch_results

    def clear_checkpoint(self) -> None:
        """Clear accumulated results and reset checkpoint counter after saving."""
        self.current_checkpoint_results.clear()
        self.current_checkpoint_batch_results.clear()
        self.checkpoint_batch_count = 0
        logger.debug("Checkpoint cleared, ready for next batch accumulation")

    def get_total_batches(self) -> int:
        """Get total number of batches processed so far."""
        return self.batch_count

    def get_checkpoint_batch_count(self) -> int:
        """Get number of batches in current checkpoint."""
        return self.checkpoint_batch_count


# ================================================================
# MEMORY MONITORING UTILITIES
# ================================================================

def aggressive_gpu_cleanup() -> None:
    """
    Aggressively clean GPU memory:
    - Delete references to large tensors
    - Force garbage collection
    - Clear CUDA cache
    - Synchronize to ensure cleanup is complete
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class AttentionActivationCapture:
    """
    Captures attention 'z' activations using transformer-lens run_with_cache.
    This is more efficient than manual hook registration as it avoids GPU memory overhead.
    """

    def __init__(self, model, start_layer: int = None, end_layer: int = None):
        self.model = model
        self.activations_by_layer = {}

        # Determine layer range
        n_layers = model.cfg.n_layers
        self.start_layer = start_layer if start_layer is not None else 0
        self.end_layer = end_layer if end_layer is not None else n_layers - 1

        if self.end_layer >= n_layers:
            self.end_layer = n_layers - 1

        logger.info(f"Will capture attention 'z' from layers {self.start_layer} to {self.end_layer}")
        logger.info(f"Using transformer-lens run_with_cache for efficient activation capture")

    def build_hook_name_filter(self) -> List[str]:
        """
        Build the list of hook names for run_with_cache to filter.
        Returns hook names like 'blocks.0.attn.hook_z', 'blocks.1.attn.hook_z', etc.
        """
        hook_names = []
        for layer_idx in range(self.start_layer, self.end_layer + 1):
            hook_names.append(f"blocks.{layer_idx}.attn.hook_z")
        return hook_names

    def capture_activations(self, full_sequences: torch.Tensor, attention_mask: torch.Tensor):
        """
        Capture activations using run_with_cache (efficient, single-pass analysis).
        
        Args:
            full_sequences: Full token sequences tensor [batch, seq_len]
            attention_mask: Attention mask tensor [batch, seq_len]
        """
        hook_name_filter = self.build_hook_name_filter()
        
        logger.info(f"Capturing with {len(hook_name_filter)} hooks using run_with_cache")
        logger.info(f"Hook name filter covers layers {self.start_layer} to {self.end_layer}")
        logger.debug(f"Hook names: {hook_name_filter}")

        with torch.inference_mode():
            # Validate sequences don't exceed context length
            logger.debug(f"Input sequence shape: {full_sequences.shape}, attention mask shape: {attention_mask.shape}")
            if full_sequences.shape[1] > self.model.cfg.n_ctx:
                logger.warning(
                    f"Generated sequence length {full_sequences.shape[1]} exceeds "
                    f"model context {self.model.cfg.n_ctx}, truncating"
                )
                full_sequences = full_sequences[:, :self.model.cfg.n_ctx]
                attention_mask = attention_mask[:, :self.model.cfg.n_ctx]

            # Run forward pass with cache, filtering to only attention 'z' hooks
            _, cache = self.model.run_with_cache(
                full_sequences,
                attention_mask=attention_mask,
                names_filter=hook_name_filter
            )
        
        logger.info(f"Cache returned {len(cache)} hooks (expected {len(hook_name_filter)})")
        logger.info(f"Cache keys: {list(cache.keys())}")

        # Extract activations from cache and store by layer index
        # Cache keys are like 'blocks.0.attn.hook_z', 'blocks.1.attn.hook_z', etc.
        layers_extracted = []
        for hook_name, activation_tensor in cache.items():
            # Parse layer index from hook name: 'blocks.X.attn.hook_z' -> X
            parts = hook_name.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    # Store as float32 to minimize memory while keeping precision
                    self.activations_by_layer[layer_idx] = activation_tensor.detach().float()
                    layers_extracted.append(layer_idx)
                    logger.debug(f"Extracted layer {layer_idx}, activation shape: {activation_tensor.shape}")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse layer index from hook name: {hook_name}")

        logger.info(f"Captured {len(self.activations_by_layer)} layer activations (expected {len(hook_name_filter)})")
        if len(self.activations_by_layer) != len(hook_name_filter):
            logger.warning(f"MISMATCH: Expected {len(hook_name_filter)} layers but captured {len(self.activations_by_layer)}")
            logger.warning(f"Expected layers: {list(range(self.start_layer, self.end_layer + 1))}")
            logger.warning(f"Captured layers: {sorted(layers_extracted)}")

        logger.info(f"Captured {len(self.activations_by_layer)} layer activations (expected {len(hook_name_filter)})")
        if len(self.activations_by_layer) != len(hook_name_filter):
            logger.warning(f"MISMATCH: Expected {len(hook_name_filter)} layers but captured {len(self.activations_by_layer)}")
            logger.warning(f"Expected layers: {list(range(self.start_layer, self.end_layer + 1))}")
            logger.warning(f"Captured layers: {sorted(layers_extracted)}")

    def clear_activations(self):
        """Clear stored activations."""
        self.activations_by_layer = {}

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Return captured activations."""
        return self.activations_by_layer


def save_checkpoint_to_hdf5(checkpoint_results: Dict[int, Dict], 
                            checkpoint_batch_results: List[Dict],
                            h5_file_path: str,
                            checkpoint_idx: int,
                            handler: DatasetHandler) -> Dict[int, Dict]:
    """
    Evaluate and save a checkpoint's batch results to HDF5 incrementally.
    
    This function:
    1. Evaluates hallucinations for all samples in the checkpoint batch
    2. Increments the results dict with evaluation labels
    3. Writes results to HDF5 file in append mode
    4. Returns the evaluated results (with is_hallucination labels)
    
    Args:
        checkpoint_results: Dict mapping row_idx to result_entry dicts
        checkpoint_batch_results: List of result_entry dicts from this checkpoint
        h5_file_path: Path to HDF5 file for incremental saves
        checkpoint_idx: Checkpoint number (for logging)
        handler: DatasetHandler instance (contains dataset_format and use_mcq)
        
    Returns:
        Updated checkpoint_results dict with is_hallucination labels added
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"CHECKPOINT {checkpoint_idx}: EVALUATING {len(checkpoint_batch_results)} SAMPLES")
    logger.info(f"{'='*80}")
    
    if not checkpoint_batch_results:
        logger.warning("No results to evaluate in this checkpoint")
        return checkpoint_results
    
    # --- BATCH EVALUATION: Judge all samples in this checkpoint ---
    evaluation_pairs = [
        (result['right_answer'], result['generated_text'], result)
        for result in checkpoint_batch_results
    ]
    
    # Use handler's batch evaluation method (centralized logic)
    evaluation_results = handler.batch_evaluate_answers(evaluation_pairs, max_workers=20)
    
    # Add evaluation results back to the checkpoint results dict
    for (hallucination_label, _), result_entry in zip(evaluation_results, checkpoint_batch_results):
        row_idx = result_entry['row_idx']
        # hallucination_label: 0=correct (non-hallucinated), 1=hallucination, 2=api_failure
        checkpoint_results[row_idx]['is_hallucination'] = hallucination_label
        
        # Also store human-readable interpretation
        if hallucination_label == 0:
            checkpoint_results[row_idx]['hallucination_status'] = 'correct'
        elif hallucination_label == 1:
            checkpoint_results[row_idx]['hallucination_status'] = 'hallucinated'
        else:  # 2
            checkpoint_results[row_idx]['hallucination_status'] = 'evaluation_failed'
    
    logger.info(f"✓ Evaluation complete for {len(checkpoint_batch_results)} samples in checkpoint {checkpoint_idx}")
    
    # --- SAVE TO HDF5 INCREMENTALLY (append mode) ---
    logger.info(f"Saving checkpoint {checkpoint_idx} to HDF5 (append mode)...")
    
    with h5py.File(h5_file_path, 'a') as h5f:
        # Store dataset_format at root level if this is the first checkpoint
        if 'dataset_format' not in h5f.attrs:
            h5f.attrs['dataset_format'] = handler.dataset_format
        
        for row_idx, result in checkpoint_results.items():
            group = h5f.create_group(f"sample_{row_idx}")

            # Store metadata
            meta_group = group.create_group("metadata")
            meta_group.attrs['row_idx'] = row_idx
            meta_group.attrs['right_answer'] = result['right_answer']
            meta_group.attrs['generated_text'] = result['generated_text']
            meta_group.attrs['prompt_length'] = result['prompt_length']
            meta_group.attrs['last_gen_token_pos'] = result['last_gen_token_pos']
            
            # Store hallucination evaluation results
            if 'is_hallucination' in result:
                meta_group.attrs['is_hallucination'] = result['is_hallucination']
                meta_group.attrs['hallucination_status'] = result.get('hallucination_status', 'unknown')

            # Store last-generated-token activations by layer (32 x 128)
            acts_last_gen_group = group.create_group("activations_last_gen_token")
            layers_saved = []
            for layer_idx, activation_last_gen in result['activations_last_gen_token'].items():
                # activation_last_gen shape: [n_heads, d_head] = [32, 128]
                acts_last_gen_group.create_dataset(f"layer_{layer_idx}", data=activation_last_gen)
                layers_saved.append(layer_idx)
            
            if row_idx == list(checkpoint_results.keys())[0]:  # Log for first sample only
                logger.info(f"Sample {row_idx}: Saved {len(layers_saved)} layers to HDF5: {sorted(layers_saved)}")

            # Store prompt info as JSON string
            prompt_info_str = json.dumps(result['prompt_info'], default=str)
            meta_group.attrs['prompt_info'] = prompt_info_str

    logger.info(f"✓ Checkpoint {checkpoint_idx} saved to HDF5")
    
    # --- POST-SAVE MEMORY CLEANUP ---
    logger.debug(f"Checkpoint {checkpoint_idx}: Cleaning up memory after save")
    
    # Explicitly clear all references
    for layer_idx in list(checkpoint_results.keys()):
        if 'activations_last_gen_token' in checkpoint_results[layer_idx]:
            # Clear the numpy arrays explicitly
            activations_dict = checkpoint_results[layer_idx]['activations_last_gen_token']
            for key in list(activations_dict.keys()):
                del activations_dict[key]
    
    del checkpoint_results
    del checkpoint_batch_results
    aggressive_gpu_cleanup()
    logger.debug(f"Checkpoint {checkpoint_idx}: Memory cleanup complete")
    
    return {}  # Return empty dict; results cleared from memory


def generate_and_capture_activations(model, token_manager, df_sample: pd.DataFrame,
                                     batch_size: int, max_tokens: int,
                                     activation_capturer, checkpoint_freq: int,
                                     handler: DatasetHandler,
                                     run_dir: str = None) -> str:
    """
    Generate outputs and capture attention activations for the faitheval dataset.
    
    Uses a two-pass approach with incremental checkpointing:
    1. GENERATE: Use transformer-lens generate() to produce full sequences
    2. ANALYZE: Run single forward pass with run_with_cache to capture activations from attention 'z'
    3. CHECKPOINT: Every N batches, evaluate hallucinations and save to HDF5, then clear memory
    4. TEXT SUMMARY: Generated from HDF5 file (NOT from in-memory accumulation)

    Args:
        checkpoint_freq: Save and evaluate every N batches
        run_dir: Directory to save HDF5 file to (typically the run output directory)

    Returns:
        Path to the saved HDF5 file
    """
    checkpoint_manager = BatchCheckpointManager(checkpoint_freq)
    # CHANGED: Do NOT accumulate all_results_for_summary - it causes memory buildup!
    # Results are already saved to HDF5, so we can load from there for text summary
    h5_file_path = None  # Will be set after HDF5 file is created
    checkpoint_idx = 0
    
    # Build terminators ONCE before the batch loop (not every batch!)
    terminators = build_terminators_for_model(model, args.model)
    
    total_batches = (len(df_sample) + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(tqdm(range(0, len(df_sample), batch_size),
                                                   desc="Generating & capturing activations",
                                                   total=total_batches)):
        batch_indices = df_sample.index[batch_start:batch_start + batch_size]
        batch_prompts = []
        batch_idx_map = {}
        batch_info = {}

        # Prepare batch
        for i, idx in enumerate(batch_indices):
            batch_idx_map[i] = idx
            row = df_sample.loc[idx]

            context = row.get('context', '')
            question = row.get('question', '')
            choices = parse_choices(row.get('choices', None))
            answer_key = row.get('answerKey', '')
            answer_text = row.get('answer', '')
            # Format ground truth using handler
            right_answer = handler.format_right_answer(answer_key, answer_text)

            qa_prompt_messages = handler.make_prompt(context, question, choices)
            
            # Adapt messages for model-specific requirements (e.g., Gemma doesn't support system role)
            adapted_messages = adapt_messages_for_model(qa_prompt_messages, args.model)
            
            # Apply the chat template to get the final string for the tokenizer
            final_prompt_str = model.tokenizer.apply_chat_template(
                adapted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(final_prompt_str)

            batch_info[i] = {
                'idx': idx,
                'question': question,
                'context': context,
                'choices': choices,
                'right_answer': right_answer,
                'qa_prompt': final_prompt_str, # Store the final templated string
                'answerKey': answer_key,
                'answer_text': answer_text,
            }

        # Tokenize
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

        # --- PASS 1: GENERATE TEXT ---
        logger.debug(f"Batch {batch_idx}: Generating text for {len(batch_idx_map)} samples")
        with torch.inference_mode():
            full_sequences = model.generate(
                input=batch_prompts_padded,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=terminators, # Use terminators here
                use_past_kv_cache=True,
                padding_side='left',
                return_type='tokens',
                verbose=False
            )
            prompt_lengths = batch_prompts_tokenized['attention_mask'].sum(dim=1)

        # --- PASS 2: CAPTURE ACTIVATIONS ---
        # Build attention mask for the full sequences (which may be longer than prompts)
        attention_mask_for_cache = (full_sequences != model.tokenizer.pad_token_id).long()
        
        logger.debug(f"Batch {batch_idx}: Capturing attention 'z' activations from full sequences")
        activation_capturer.clear_activations()
        activation_capturer.capture_activations(full_sequences, attention_mask_for_cache)

        # --- EXTRACT RESULTS ---
        batch_results = {}  # Results for this batch only
        batch_size_actual = len(batch_idx_map)
        for i, idx in batch_idx_map.items():
            prompt_len = prompt_lengths[i].item()
            
            # CRITICAL: Account for left-padding offset
            # With left-padding: [PAD...PAD][prompt][generated]
            # padding_length = total_width - prompt_len
            max_len = batch_prompts_padded.shape[1]
            padding_len = max_len - prompt_len
            
            # Absolute position where generation starts (after left-padding + prompt)
            start_of_generation = padding_len + prompt_len
            actual_seq_len = full_sequences.shape[1]
            
            # Debug logging for first sample in batch
            if i == 0:
                logger.debug(f"\n--- Answer Extraction Debug (Batch {batch_idx}, Sample {i}) ---")
                logger.debug(f"Prompt length (from attention mask): {prompt_len}")
                logger.debug(f"Max batch length (padded width): {max_len}")
                logger.debug(f"Padding length (left-pad amount): {padding_len}")
                logger.debug(f"Start of generation position: {start_of_generation}")
                logger.debug(f"Full sequence length: {actual_seq_len}")
            
            # Find where generation actually ends (look for any terminator token)
            generated_part_tokens = full_sequences[i, start_of_generation:actual_seq_len]
            first_eos_pos = -1
            for idx_token, token_id in enumerate(generated_part_tokens):
                token_id_val = token_id.item()
                if token_id_val in terminators:  # Check against ALL terminators
                    first_eos_pos = idx_token
                    break
            
            if first_eos_pos != -1:
                gen_len = first_eos_pos + 1
            else:
                gen_len = len(generated_part_tokens)
            
            if i == 0:
                logger.debug(f"Generation length (tokens): {gen_len}")
                logger.debug(f"EOS token found: {first_eos_pos != -1}")
            
            # Extract exact answer tokens
            end_of_generation = start_of_generation + gen_len
            answer_tokens = full_sequences[i, start_of_generation:end_of_generation]

            # Decode answer with skip_special_tokens=True to remove EOS/PAD tokens
            answer_text = model.tokenizer.decode(
                answer_tokens, skip_special_tokens=True).strip()
            
            # Capture detailed debug info for all samples (not just first in batch)
            answer_token_ids = answer_tokens.tolist()
            answer_token_strings = [model.tokenizer.decode([int(t)]) for t in answer_token_ids]
            
            # Check for special tokens
            special_token_ids = {model.tokenizer.eos_token_id, model.tokenizer.pad_token_id}
            has_special_tokens = any(int(t) in special_token_ids for t in answer_token_ids)
            
            # Decode without skip_special_tokens to see what's being filtered
            answer_text_with_special = model.tokenizer.decode(answer_tokens, skip_special_tokens=False)
            
            debug_info = {
                'answer_token_ids': answer_token_ids,
                'answer_token_strings': answer_token_strings,
                'answer_text_decoded': answer_text,
                'answer_text_with_special_tokens': answer_text_with_special,
                'num_tokens': len(answer_token_ids),
                'has_special_tokens': has_special_tokens,
                'prompt_length': int(prompt_len),
                'padding_length': int(padding_len),
                'generation_length': int(gen_len),
                'start_of_generation': int(start_of_generation),
                'end_of_generation': int(end_of_generation),
                'actual_seq_len': int(actual_seq_len),
                'max_batch_len': int(max_len),
                'eos_found_in_generation': first_eos_pos != -1,
            }
            
            if i == 0:
                logger.debug(f"Answer tokens shape: {answer_tokens.shape}")
                logger.debug(f"Decoded answer (first 100 chars): {answer_text[:100]}")
                logger.debug(f"Debug info: {debug_info}")
                logger.debug("--- End Debug ---\n")

            # Store activations and metadata with proper extraction tracking
            # CRITICAL: Convert GPU activations to CPU numpy immediately to prevent accumulation
            activations_last_gen_token = {}
            for layer_idx, activation_tensor in activation_capturer.get_activations().items():
                # activation_tensor shape: [batch, seq_len, n_heads, d_head]
                # Extract only the sample for this row and move to CPU immediately
                sample_activation = activation_tensor[i].cpu().detach().numpy()  # Shape: [seq_len, n_heads, d_head]
                # Extract LAST GENERATED token activation: [n_heads, d_head] = [32, 128]
                last_gen_token_pos = end_of_generation - 1
                activations_last_gen_token[layer_idx] = sample_activation[last_gen_token_pos]  # [32, 128]
            
            if i == 0:  # Log for first sample in batch
                logger.info(f"Batch {batch_idx}: Extracted activations for {len(activations_last_gen_token)} layers: {sorted(activations_last_gen_token.keys())}")

            result_entry = {
                'row_idx': idx,
                'activations_last_gen_token': activations_last_gen_token,
                'prompt_info': batch_info[i],
                'generated_text': answer_text,
                'right_answer': batch_info[i]['right_answer'],
                'prompt_length': prompt_len,
                'padding_length': padding_len,
                'generation_length': gen_len,
                'start_of_generation': start_of_generation,
                'end_of_generation': end_of_generation,
                'last_gen_token_pos': last_gen_token_pos,
                'debug_info': debug_info,
                'extraction_info': {
                    'method': 'left_padded_batch_run_with_cache',
                    'max_batch_len': max_len,
                    'actual_seq_len': actual_seq_len,
                    'eos_found': first_eos_pos != -1,
                    'activation_capture_method': 'transformer_lens_run_with_cache',
                }
            }
            
            batch_results[idx] = result_entry

        # --- AGGRESSIVE BATCH CLEANUP ---
        # CRITICAL: Delete all GPU tensors and clear cache immediately after extraction
        logger.debug(f"Batch {batch_idx}: Cleaning up GPU memory")
        
        # Explicitly delete all GPU-resident tensors
        del batch_prompts_tokenized
        del batch_prompts_padded
        del attention_mask
        del full_sequences
        del attention_mask_for_cache
        
        # Clear activation capturer's internal state
        activation_capturer.clear_activations()
        
        # Aggressive cleanup
        aggressive_gpu_cleanup()
        
        # --- ADD BATCH RESULTS TO CHECKPOINT MANAGER ---
        checkpoint_manager.add_batch_result(batch_results)
        
        # CRITICAL: Immediately clear batch_results to free memory
        del batch_results
        aggressive_gpu_cleanup()
        
        logger.debug(f"Batch {batch_idx}: Processed {len(batch_idx_map)} samples, "
                    f"checkpoint accumulation: {checkpoint_manager.get_checkpoint_batch_count()} batches")
        
        # --- CHECKPOINT: Evaluate and save every N batches ---
        if checkpoint_manager.should_checkpoint():
            checkpoint_idx += 1
            checkpoint_results, checkpoint_batch_results = checkpoint_manager.get_checkpoint_results()
            
            # Initialize HDF5 file on first checkpoint (save to run_dir)
            if h5_file_path is None:
                if run_dir is None:
                    raise ValueError("run_dir must be provided to save HDF5 file")
                h5_file_path = os.path.join(run_dir, "attention_z_activations.h5")
                # Create file in write mode on first checkpoint
                logger.info(f"Creating HDF5 file: {h5_file_path}")
                with h5py.File(h5_file_path, 'w') as f:
                    pass  # Just create the file
            
            # Save checkpoint to HDF5 and evaluate
            save_checkpoint_to_hdf5(checkpoint_results, checkpoint_batch_results, 
                                   h5_file_path, checkpoint_idx, handler=handler)
            
            # CHANGED: Don't accumulate results - they're already in HDF5!
            # Verify HDF5 has expected number of samples
            with h5py.File(h5_file_path, 'r') as h5f:
                samples_in_hdf5 = len(h5f)
                logger.info(f"Checkpoint {checkpoint_idx} verification: {samples_in_hdf5} samples now in HDF5 file")
            
            # Clear checkpoint for next batch accumulation
            checkpoint_manager.clear_checkpoint()

    # --- FINAL CHECKPOINT: Handle remaining batches if any ---
    if checkpoint_manager.get_checkpoint_batch_count() > 0:
        checkpoint_idx += 1
        checkpoint_results, checkpoint_batch_results = checkpoint_manager.get_checkpoint_results()
        
        # Initialize HDF5 file if not already done (edge case: fewer batches than checkpoint_freq)
        if h5_file_path is None:
            if run_dir is None:
                raise ValueError("run_dir must be provided to save HDF5 file")
            h5_file_path = os.path.join(run_dir, "attention_z_activations.h5")
            logger.info(f"Creating HDF5 file: {h5_file_path}")
            with h5py.File(h5_file_path, 'w') as f:
                pass
        
        logger.info(f"\nProcessing final checkpoint with {checkpoint_manager.get_checkpoint_batch_count()} remaining batches")
        save_checkpoint_to_hdf5(checkpoint_results, checkpoint_batch_results,
                               h5_file_path, checkpoint_idx, handler=handler)

    logger.info(f"\n{'='*80}")
    logger.info(f"All batches processed and checkpoints saved!")
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Total checkpoints: {checkpoint_idx}")
    
    # Verify final HDF5 file
    if h5_file_path and os.path.exists(h5_file_path):
        with h5py.File(h5_file_path, 'r') as h5f:
            total_samples = len(h5f)
            logger.info(f"Total samples in HDF5: {total_samples}")
    
    logger.info(f"HDF5 File: {h5_file_path}")
    logger.info(f"{'='*80}\n")

    return h5_file_path


def save_activations_to_hdf5(results: Dict[int, Dict], output_dir: str, model_name: str):
    """DEPRECATED: Use save_checkpoint_to_hdf5() instead.
    
    Kept for backwards compatibility. This function is no longer used in the main flow.
    Saves only last-generated-token activations (32 x 128 per layer).
    """
    os.makedirs(output_dir, exist_ok=True)

    h5_file = os.path.join(output_dir, f"{model_name}_attention_z_activations.h5")

    with h5py.File(h5_file, 'w') as h5f:
        for row_idx, result in results.items():
            group = h5f.create_group(f"sample_{row_idx}")

            # Store metadata
            meta_group = group.create_group("metadata")
            meta_group.attrs['row_idx'] = row_idx
            meta_group.attrs['right_answer'] = result['right_answer']
            meta_group.attrs['generated_text'] = result['generated_text']
            meta_group.attrs['prompt_length'] = result['prompt_length']
            meta_group.attrs['last_gen_token_pos'] = result['last_gen_token_pos']
            
            # Store hallucination evaluation results
            if 'is_hallucination' in result:
                meta_group.attrs['is_hallucination'] = result['is_hallucination']
                meta_group.attrs['hallucination_status'] = result.get('hallucination_status', 'unknown')

            # Store last-generated-token activations by layer (32 x 128)
            acts_last_gen_group = group.create_group("activations_last_gen_token")
            for layer_idx, activation_last_gen in result['activations_last_gen_token'].items():
                # activation_last_gen shape: [n_heads, d_head] = [32, 128]
                acts_last_gen_group.create_dataset(f"layer_{layer_idx}", data=activation_last_gen)

            # Store prompt info as JSON string
            prompt_info_str = json.dumps(result['prompt_info'], default=str)
            meta_group.attrs['prompt_info'] = prompt_info_str

    logger.info(f"Saved activations to {h5_file}")
    return h5_file


def save_results_to_text(h5_file_path: str, output_dir: str) -> str:
    """Save prompts, generated responses, and hallucination labels to a clean text file.
    
    Reads from HDF5 file instead of in-memory dict to avoid memory accumulation.
    Creates a nicely formatted text file with the full concatenated prompt and generated response.
    Includes detailed debugging information for diagnosing generation issues.
    
    Args:
        h5_file_path: Path to the HDF5 file with saved results
        output_dir: Directory to save the text file
        
    Returns:
        Path to the saved text file
    """
    text_file = os.path.join(output_dir, "results_summary.txt")
    
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("GENERATION AND HALLUCINATION EVALUATION RESULTS\n")
        f.write("="*90 + "\n\n")
        
        # Load results from HDF5 file
        with h5py.File(h5_file_path, 'r') as h5f:
            sample_keys = sorted(h5f.keys(), key=lambda x: int(x.split('_')[1]))
            
            for i, sample_key in enumerate(sample_keys, 1):
                group = h5f[sample_key]
                meta_group = group['metadata']
                
                # Read metadata attributes
                row_idx = meta_group.attrs['row_idx']
                right_answer = meta_group.attrs['right_answer']
                generated_text = meta_group.attrs['generated_text']
                prompt_length = meta_group.attrs['prompt_length']
                last_gen_token_pos = meta_group.attrs['last_gen_token_pos']
                is_hallucination = meta_group.attrs.get('is_hallucination', -1)
                hallucination_status = meta_group.attrs.get('hallucination_status', 'unknown')
                
                # Load prompt info from JSON string
                prompt_info_str = meta_group.attrs['prompt_info']
                prompt_info = json.loads(prompt_info_str)
                
                # Write sample header
                f.write(f"[Sample {i}] (Row Index: {row_idx})\n")
                f.write("-"*90 + "\n\n")
                
                # Full concatenated prompt (as it was given to the model)
                qa_prompt = prompt_info.get('qa_prompt', '')
                if qa_prompt:
                    f.write("PROMPT:\n")
                    f.write(f"{qa_prompt}\n\n")
                
                # Correct answer
                f.write("CORRECT ANSWER:\n")
                f.write(f"{right_answer}\n\n")
                
                # Generated response
                f.write("GENERATED RESPONSE:\n")
                f.write(f"{generated_text}\n\n")
                
                # Hallucination label
                f.write("EVALUATION:\n")
                f.write(f"  Status: {hallucination_status.upper()}\n")
                if is_hallucination == 0:
                    f.write("  Label: ✓ CORRECT (Non-hallucinated)\n")
                elif is_hallucination == 1:
                    f.write("  Label: ✗ HALLUCINATED\n")
                elif is_hallucination == 2:
                    f.write("  Label: ? EVALUATION FAILED\n")
                else:
                    f.write("  Label: PENDING EVALUATION\n")
                
                f.write("\n" + "="*90 + "\n\n")
        
        # Write summary statistics
        f.write("\nSUMMARY STATISTICS\n")
        f.write("-"*90 + "\n")
        
        with h5py.File(h5_file_path, 'r') as h5f:
            total_count = len(h5f)
            correct_count = 0
            hallucinated_count = 0
            failed_count = 0
            pending_count = 0
            
            for sample_key in h5f.keys():
                group = h5f[sample_key]
                meta_group = group['metadata']
                is_hallucination = meta_group.attrs.get('is_hallucination', -1)
                
                if is_hallucination == 0:
                    correct_count += 1
                elif is_hallucination == 1:
                    hallucinated_count += 1
                elif is_hallucination == 2:
                    failed_count += 1
                else:
                    pending_count += 1
        
        f.write(f"Total Samples:        {total_count}\n")
        f.write(f"Correct (non-hallucinated): {correct_count} ({100*correct_count/total_count:.1f}%)\n")
        f.write(f"Hallucinated:         {hallucinated_count} ({100*hallucinated_count/total_count:.1f}%)\n")
        f.write(f"Evaluation Failed:    {failed_count} ({100*failed_count/total_count:.1f}%)\n")
        f.write(f"Pending Evaluation:   {pending_count} ({100*pending_count/total_count:.1f}%)\n")
        f.write("="*90 + "\n")
    
    logger.info(f"Saved results summary to {text_file}")
    return text_file


def main():
    """Main entry point."""
    try:
        # --- STRICT VALIDATION: dataset_format is REQUIRED ---
        try:
            validate_dataset_format(args.dataset_format)
        except ValueError as e:
            logger.error(f"FATAL: Dataset format validation failed: {e}")
            raise

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_basename = os.path.basename(args.dataset_path).replace('.csv', '')
        run_name = f"ITI_ACTIVATIONS_{dataset_basename}_{timestamp}"
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        logger.set_output_directory(run_dir)

        logger.info(f"\n{'='*80}")
        logger.info("ATTENTION ACTIVATION CAPTURE FOR ITI")
        logger.info(f"{'='*80}")
        logger.info(f"Run Name: {run_name}")
        logger.info(f"Output Directory: {run_dir}")
        logger.info(f"Device ID: {args.device_id}")
        logger.info(f"Dataset: {args.dataset_path}")
        logger.info(f"Dataset Format: {args.dataset_format}")
        logger.info(f"Num Samples: {args.num_samples}")
        logger.info(f"Max Tokens: {args.max_tokens}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Checkpoint Frequency: {args.checkpoint_freq} batches")
        logger.info(f"Model: {args.model} ({HUGGINGFACE_MODEL_ID})")

        # --- LOAD DATASET ---
        try:
            logger.info(f"\n{'='*80}")
            logger.info("LOADING DATASET")
            logger.info(f"{'='*80}")

            if not os.path.exists(args.dataset_path):
                raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

            df = pd.read_csv(args.dataset_path)
            logger.info(f"Loaded dataset: {len(df)} total samples")

            if len(df) > args.num_samples:
                df_sample = df.head(args.num_samples)
            else:
                df_sample = df

            logger.info(f"Using {len(df_sample)} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
            print(f"ERROR: Failed to load dataset: {str(e)}")
            raise

        # --- CREATE DATASET HANDLER ---
        try:
            logger.info(f"\n{'='*80}")
            logger.info("INITIALIZING DATASET HANDLER")
            logger.info(f"{'='*80}")
            
            # Determine use_mcq based on dataset_format (like baseline_run.py)
            use_mcq = (args.dataset_format in ['mcq', 'mmlu', 'hellaswag'])
            handler = DatasetHandler(use_mcq=use_mcq, dataset_format=args.dataset_format, model_type=args.model, logger=logger)
            logger.info(f"DatasetHandler initialized (use_mcq={use_mcq}, dataset_format={args.dataset_format}, model_type={args.model})")
        except Exception as e:
            logger.error(f"Error initializing DatasetHandler: {str(e)}", exc_info=True)
            print(f"ERROR: Failed to initialize DatasetHandler: {str(e)}")
            raise

        # --- LOAD MODEL ---
        try:
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
            
            # Log model context length
            model_context_length = model.cfg.n_ctx
            logger.info(f"Model context length (n_ctx): {model_context_length} tokens")
            logger.info(f"Max tokens for generation (max_new_tokens): {args.max_tokens}")
            logger.info(f"Tokenization max_length: {model_context_length - args.max_tokens}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            print(f"ERROR: Failed to load model: {str(e)}")
            raise

        # --- INITIALIZE TOKEN MANAGER ---
        try:
            token_manager = TokenManager(
                model=model,
                max_answer_tokens=args.max_tokens,
                model_dir=TRANSFORMER_LENS_MODEL_NAME
            )
            token_manager.setup_tokenizer_padding(model)
            logger.info("TokenManager initialized")
        except Exception as e:
            logger.error(f"Error initializing TokenManager: {str(e)}", exc_info=True)
            print(f"ERROR: Failed to initialize TokenManager: {str(e)}")
            raise

        # --- INITIALIZE ACTIVATION CAPTURER ---
        try:
            logger.info(f"\n{'='*80}")
            logger.info("INITIALIZING ACTIVATION CAPTURE")
            logger.info(f"{'='*80}")

            activation_capturer = AttentionActivationCapture(
                model,
                start_layer=args.start_layer,
                end_layer=args.end_layer
            )
            logger.info(f"AttentionActivationCapture initialized (will use run_with_cache)")
        except Exception as e:
            logger.error(f"Error initializing activation capturer: {str(e)}", exc_info=True)
            print(f"ERROR: Failed to initialize activation capturer: {str(e)}")
            raise

        # --- GENERATE & CAPTURE ACTIVATIONS WITH INCREMENTAL CHECKPOINTING ---
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"GENERATING OUTPUTS & CAPTURING ACTIVATIONS (with checkpointing)")
            logger.info(f"{'='*80}")

            h5_file_path = generate_and_capture_activations(
                model, token_manager, df_sample, args.batch_size,
                MAX_ANSWER_TOKENS, activation_capturer, args.checkpoint_freq,
                handler=handler,
                run_dir=run_dir
            )

            # Get actual sample count from HDF5 file
            with h5py.File(h5_file_path, 'r') as h5f:
                num_samples_processed = len(h5f)
            
            logger.info(f"✓ Captured and evaluated activations for {num_samples_processed} samples IN THE HDF5 FILE")
            logger.info(f"✓ HDF5 file saved to: {h5_file_path}")
        except Exception as e:
            logger.error(f"Error generating and capturing activations: {str(e)}", exc_info=True)
            print(f"ERROR: Failed during activation generation: {str(e)}")
            raise

        # --- SAVE TEXT SUMMARY (deferred to end) ---
        try:
            logger.info(f"\n{'='*80}")
            logger.info("GENERATING FINAL TEXT SUMMARY")
            logger.info(f"{'='*80}")

            # CHANGED: Pass h5_file_path instead of results dict to avoid memory issues
            text_file = save_results_to_text(h5_file_path, run_dir)

            # Save metadata
            metadata = {
                'run_name': run_name,
                'dataset_path': args.dataset_path,
                'num_samples': args.num_samples,
                'max_tokens': args.max_tokens,
                'batch_size': args.batch_size,
                'checkpoint_freq': args.checkpoint_freq,
                'device_id': args.device_id,
                'model': args.model,
                'model_name': MODEL_NAME,
                'huggingface_model_id': HUGGINGFACE_MODEL_ID,
                'transformer_lens_model_name': TRANSFORMER_LENS_MODEL_NAME,
                'attention_hook': ATTENTION_HOOK,
                'h5_file': h5_file_path,
                'text_file': text_file,
                'timestamp': timestamp,
            }

            metadata_file = os.path.join(run_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"\n{'='*80}")
            logger.info("ACTIVATION CAPTURE COMPLETE")
            logger.info(f"Results directory: {run_dir}")
            logger.info(f"HDF5 file: {h5_file_path}")
            logger.info(f"Text summary: {text_file}")
            logger.info(f"Metadata: {metadata_file}")
            logger.info(f"{'='*80}\n")

            return run_dir
        except Exception as e:
            logger.error(f"Error saving text summary: {str(e)}", exc_info=True)
            print(f"ERROR: Failed to save text summary: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"FATAL ERROR in main execution: {str(e)}", exc_info=True)
        print(f"\n{'='*80}")
        print(f"FATAL ERROR: {str(e)}")
        print(f"{'='*80}")
        print("Check the error log file for full traceback details.")
        return None


if __name__ == "__main__":
    import gc
    try:
        main()
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
