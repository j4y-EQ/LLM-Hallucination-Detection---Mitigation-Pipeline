"""Activation Capture Utilities for Transformer Models.

Provides core functionality for capturing transformer activations during
text generation. Implements efficient batch processing with TransformerLens
hooks, token position management, and activation extraction strategies.

FUNCTIONALITY:
    - Batch generation with activation capture at multiple hooks
    - Token position calculation for different schemes (BOS, EOS, last token)
    - First-period truncation for controlled generation length
    - Activation extraction and formatting for classifier training
    - Memory-efficient processing for large-scale capture

TOKEN SCHEMES:
    - 'bos_token': Capture at beginning-of-sequence position
    - 'last_token': Capture at final generated token position
    - 'eos_token': Capture at end-of-sequence marker
    - 'first_period': Capture at first period token (if found)

USAGE:
    Used by generator.py and evaluate.py for activation capture during
    answer generation. Ensures consistent activation extraction between
    training and inference.

Dependencies:
    - TransformerLens: For hook-based activation capture
    - PyTorch: For tensor operations
"""

import os
import torch
import numpy as np
from typing import Dict, List, Any

# ================================================================
# ACTIVATION CAPTURE UTILITIES
# ================================================================

def truncate_at_first_period(text: str) -> str:
    """
    Truncate text at the first period (.), including the period.
    If no period is found, return the original text.
    
    Args:
        text: Input text to truncate
        
    Returns:
        Truncated text (up to and including first period), or original text if no period found
    """
    period_idx = text.find('.')
    if period_idx != -1:
        return text[:period_idx + 1]
    return text

def find_first_period_token_position(full_sequences, batch_idx: int, start_of_generation: int, 
                                    tokenizer, model_device: str) -> int:
    """
    Find the position of the first period token in the generated portion of a sequence.
    
    Args:
        full_sequences: The full token sequences tensor
        batch_idx: Index of the batch item
        start_of_generation: Position where generation starts
        tokenizer: The tokenizer to get period token ID
        model_device: Device the model is on
        
    Returns:
        Position of the first period token, or -1 if not found
    """
    period_token_id = tokenizer.encode('.', add_special_tokens=False)[0] if tokenizer.encode('.', add_special_tokens=False) else None
    
    if period_token_id is None:
        return -1
    
    generated_part = full_sequences[batch_idx, start_of_generation:]
    
    for idx, token_id in enumerate(generated_part):
        if token_id.item() == period_token_id:
            return start_of_generation + idx
    
    return -1

def apply_first_period_truncation(
    full_sequences: torch.Tensor,
    batch_idx: int,
    start_of_generation: int,
    original_gen_len: int,
    padding_len: int,
    prompt_len: int,
    tokenizer,
    model_device: str,
    token_scheme: str = "last_generated",
    first_period_truncation: bool = False,
    logger = None
) -> dict:
    """
    Unified function to apply first period truncation for both activation 
    position and text generation length.
    
    This ensures consistency between:
    - Activation extraction positions (for classifier input)
    - Text decoding ranges (for GPT-3.5 evaluation)
    
    Args:
        full_sequences: The full token sequences tensor [batch, seq_len]
        batch_idx: Index of the batch item to process
        start_of_generation: Absolute position where generation starts
        original_gen_len: Number of tokens originally generated
        padding_len: Number of left-padding tokens
        prompt_len: Length of the prompt (before generation)
        tokenizer: The tokenizer to identify period tokens
        model_device: Device the model is on
        token_scheme: Token position scheme (only applies to "last_generated")
        first_period_truncation: Whether to truncate at first period
        logger: Optional logger for debug messages
        
    Returns:
        dict with keys:
            - 'activation_position': Adjusted token position for activation extraction
            - 'adjusted_gen_len': Adjusted generation length (for text decoding)
            - 'truncated': Boolean indicating if truncation was applied
            - 'period_position': Position of period if found, else -1
    
    Example:
        Without truncation: gen_len=20, returns position at token 19
        With truncation (period at 5): gen_len=20, returns position at token 5, adjusted_gen_len=6
    """
    # Default: no truncation applied
    result = {
        'activation_position': padding_len + prompt_len + original_gen_len - 1,
        'adjusted_gen_len': original_gen_len,
        'truncated': False,
        'period_position': -1
    }
    
    # Only apply truncation for "last_generated" scheme when enabled
    if not first_period_truncation or token_scheme != "last_generated":
        return result
    
    # Find first period position in generated portion
    first_period_pos = find_first_period_token_position(
        full_sequences, batch_idx, start_of_generation, 
        tokenizer, model_device
    )
    
    if first_period_pos != -1:
        # Calculate how many tokens up to and including the period
        adjusted_gen_len = first_period_pos - start_of_generation + 1
        
        result.update({
            'activation_position': first_period_pos,
            'adjusted_gen_len': adjusted_gen_len,
            'truncated': True,
            'period_position': first_period_pos
        })
        
        if logger:
            logger.debug(
                f"Period truncation applied for batch_idx={batch_idx}: "
                f"original_gen_len={original_gen_len} → adjusted={adjusted_gen_len}, "
                f"period_pos={first_period_pos}"
            )
    else:
        if logger:
            logger.debug(
                f"No period found in generated text for batch_idx={batch_idx}, "
                f"using full generation length={original_gen_len}"
            )
    
    return result

def get_hook_name_for_layer(hook_base_name: str, layer_idx: int) -> str:
    """Convert base hook name to full hook name for specific layer"""
    if hook_base_name in ["hook_embed"]:
        return hook_base_name  # Layer-independent hooks
    elif "." in hook_base_name:  # Sub-component hooks like attn.hook_pattern
        return f"blocks.{layer_idx}.{hook_base_name}"
    else:  # Direct layer hooks like hook_resid_pre
        return f"blocks.{layer_idx}.{hook_base_name}"

def parse_hook_name(hook_name_full: str) -> tuple[int, str]:
    """
    Parses a full hook name like 'blocks.5.attn.hook_pattern' 
    into its layer and base name.
    
    Returns:
        tuple: (layer_idx, hook_base_name)
            - layer_idx (int): The layer index, or -1 for 
            layer-independent hooks.
            - hook_base_name (str): The base name (e.g., 'attn.
            hook_pattern', 'hook_resid_post').
    """
    if hook_name_full.startswith("blocks."):
        parts = hook_name_full.split('.')
        try:
            layer_idx = int(parts[1])
            hook_base_name = '.'.join(parts[2:])
            return layer_idx, hook_base_name
        except (ValueError, IndexError):
            return -1, hook_name_full
    else:
        # Layer-independent hooks like 'hook_embed'
        return -1, hook_name_full

def generate_and_capture_efficiently(
    model, 
    token_manager,
    batch_prompts_padded, 
    batch_info, 
    active_hooks: List[str],
    token_schemes: List[str],
    start_layer: int,
    end_layer: int,
    max_answer_tokens: int,
    model_name: str,
    logger,
    debug_verbose: bool = False,
    first_period_truncation: bool = False
):
    """
    Generates text and captures specified activations for an entire batch using an efficient, single-pass analysis.
    This function uses transformer-lens generate method directly.

    Args:
        model (HookedTransformer): The transformer-lens model.
        token_manager (TokenManager): The token manager instance.
        batch_prompts_padded (torch.Tensor): Padded tensor of input prompts.
        batch_info (list): List of metadata dicts for each item in the batch.
        active_hooks (List[str]): List of base hook names to capture.
        token_schemes (List[str]): List of token schemes to use for position calculation.
        start_layer (int): The starting layer index for activation capture.
        end_layer (int): The ending layer index for activation capture.
        max_answer_tokens (int): The maximum number of tokens to generate for an answer.
        model_name (str): The name of the model being used.
        logger: The logger instance.
        debug_verbose (bool): Flag for verbose debug output.
        first_period_truncation (bool): If True, truncate at first period for last_generated scheme.

    Returns:
        tuple: (full_sequences, dict_of_captured_activations, list_of_updated_batch_info, positions_by_scheme)
    """
    # --- 1. GENERATE FIRST: Use transformer-lens generate method directly ---
    logger.info("Step 1: Generating text for the batch deterministically using transformer-lens generate...")

    # Create the attention mask for later use in cache analysis and prompt length calculation
    attention_mask = (batch_prompts_padded != model.tokenizer.pad_token_id).long()

    with torch.inference_mode():
        full_sequences = model.generate(
            input=batch_prompts_padded,
            max_new_tokens=max_answer_tokens,
            do_sample=False,
            stop_at_eos=True,
            eos_token_id=model.tokenizer.eos_token_id,
            use_past_kv_cache=True,
            padding_side='left',
            return_type='tokens',
            verbose=False
        )
    logger.info("Generation complete.")

    # --- 2. ANALYZE SECOND: Run one forward pass on the complete sequences to get the cache. ---
    logger.info("Step 2: Running single analysis pass to capture all activations...")
    logger.debug(f"Full sequences shape before cache: {full_sequences.shape}")

    if torch.cuda.is_available() and debug_verbose:
        before_allocated = torch.cuda.memory_allocated() / (1024**3)
        before_reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.debug(f"\nGPU Memory before activation capture:")
        logger.debug(f"- Allocated: {before_allocated:.2f} GB")
        logger.debug(f"- Reserved:  {before_reserved:.2f} GB")

    # --- MEMORY FIX: Pre-filter hooks to only capture what's needed ---
    # This is the critical change to prevent GPU memory explosion.
    # We build a list of the exact hook names we want and pass it to run_with_cache.
    hook_name_filter = []
    for layer_idx in range(start_layer, end_layer + 1):
        for hook_base_name in active_hooks:
            if hook_base_name not in ["hook_embed"]:
                hook_name_filter.append(get_hook_name_for_layer(hook_base_name, layer_idx))
    if "hook_embed" in active_hooks:
        hook_name_filter.append("hook_embed")
    
    logger.info(f"Applying filter to run_with_cache for {len(hook_name_filter)} hooks.")

    with torch.inference_mode():
        attention_mask_for_cache = (full_sequences != model.tokenizer.pad_token_id).long()
        logger.debug(f"Attention mask for cache shape: {attention_mask_for_cache.shape}")
        logger.debug(f"Attention mask sample: {attention_mask_for_cache[0]}")   

        logger.debug(f"Model context length: {model.cfg.n_ctx}")
        if full_sequences.shape[1] > model.cfg.n_ctx:
            logger.warning(f"Generated sequence length {full_sequences.shape[1]} exceeds model context {model.cfg.n_ctx}")
            full_sequences = full_sequences[:, :model.cfg.n_ctx]
            attention_mask_for_cache = attention_mask_for_cache[:, :model.cfg.n_ctx]
            logger.warning(f"Truncated sequences to {full_sequences.shape[1]} tokens")

        _, cache = model.run_with_cache(
            full_sequences, 
            attention_mask=attention_mask_for_cache,
            names_filter=hook_name_filter  # Pass the filter here
        )
    logger.info("Activation capture complete.")

    if torch.cuda.is_available() and debug_verbose:
        after_allocated = torch.cuda.memory_allocated() / (1024**3)
        after_reserved = torch.cuda.memory_reserved() / (1024**3)
        diff_allocated = after_allocated - before_allocated
        diff_reserved = after_reserved - before_reserved
        logger.debug(f"\nGPU Memory after activation capture:")
        logger.debug(f"- Allocated: {after_allocated:.2f} GB (+ {diff_allocated:.2f} GB)")
        logger.debug(f"- Reserved:  {after_reserved:.2f} GB (+ {diff_reserved:.2f} GB)\n")

    # --- 3. EXTRACT: Calculate all necessary token positions for the batch. ---
    positions_by_scheme = {scheme: [] for scheme in token_schemes}
    generated_lengths = []
    batch_size = len(batch_info)
    prompt_lengths_from_mask = (attention_mask.sum(dim=1)).tolist()

    cache_seq_len = None
    for hook_name, tensor in cache.items():
        if len(tensor.shape) >= 2:  # Most tensors have [batch, seq, ...] shape
            cache_seq_len = tensor.shape[1]
            break
    
    if cache_seq_len is None:
        logger.error("Could not determine cache sequence length")
        return {}, []
    
    logger.info(f"Input sequence length: {attention_mask.shape[1]}")
    logger.info(f"Generated sequence length: {full_sequences.shape[1]}")

    for i in range(batch_size):
        # Calculate prompt length from attention mask (1s = real tokens, 0s = padding)
        # The prompt length is the number of 1s in the attention mask.
        # With LEFT-padding, padding is at the beginning, not the end.        
        prompt_len = prompt_lengths_from_mask[i]
        padding_len = attention_mask.shape[1] - prompt_len

        # Determine where generation starts in the sequence
        # With LEFT-padding: [padding_tokens | prompt_tokens | generated_tokens]
        # Example: padding_len=5, prompt_len=10 → generation starts at index 15
        # The prompt starts at position padding_len and ends at position padding_len + prompt_len - 1
        start_of_generation = padding_len + prompt_len

        # Make sure we don't exceed the actual sequence length we're analyzing
        actual_seq_len = full_sequences.shape[1]

        if start_of_generation >= actual_seq_len:
            # Edge case: Prompt takes up the entire sequence, no generation occurred
            gen_len = 0
            generated_lengths.append(gen_len)
            # Use positions for prompt-only (no generation)
            token_positions = token_manager.get_token_positions(prompt_len, gen_len, 
            padding_len)
            for scheme in token_schemes:
                positions_by_scheme[scheme].append(token_positions[scheme])
            continue
            
        # Extract just the generated portion of the sequence (after prompt)
        generated_part_tokens = full_sequences[i, start_of_generation:actual_seq_len]
        
        # Find first EOS token in generated text to determine actual generation length
        # Loop through generated tokens to find where model stopped generating
        first_stop_position_relative = -1
        for idx, token_id in enumerate(generated_part_tokens):
            if token_id.item() == model.tokenizer.eos_token_id:
                first_stop_position_relative = idx
                break

        if first_stop_position_relative != -1:
            # EOS found: generation length includes the EOS token position + 1
            # Example: EOS at index 7 → gen_len=8 (tokens 0-7, inclusive)
            gen_len = first_stop_position_relative + 1
        else:
            # No EOS token found: model generated full max_answer_tokens
            # Use entire generated sequence length
            gen_len = len(generated_part_tokens)
        
        generated_lengths.append(gen_len)

        # Use the calculated gen_len to get all token positions
        # With LEFT-padding, we need padding_len for position calculation
        token_positions = token_manager.get_token_positions(prompt_len, gen_len, padding_len)
        
        # Apply unified period truncation if enabled
        # This updates BOTH the activation position AND the generation length for text extraction
        if first_period_truncation and "last_generated" in token_schemes:
            truncation_result = apply_first_period_truncation(
                full_sequences=full_sequences,
                batch_idx=i,
                start_of_generation=start_of_generation,
                original_gen_len=gen_len,
                padding_len=padding_len,
                prompt_len=prompt_len,
                tokenizer=model.tokenizer,
                model_device=model.cfg.device,
                token_scheme="last_generated",
                first_period_truncation=True,
                logger=logger
            )
            
            # Update position for activation extraction
            token_positions["last_generated"] = truncation_result['activation_position']
            
            # Store adjusted gen_len in batch_info for text extraction later
            if truncation_result['truncated']:
                batch_info[i]["adjusted_gen_len"] = truncation_result['adjusted_gen_len']
                generated_lengths[i] = truncation_result['adjusted_gen_len']  # Update for consistency
            else:
                batch_info[i]["adjusted_gen_len"] = gen_len
        else:
            # No truncation - use original gen_len
            batch_info[i]["adjusted_gen_len"] = gen_len
        
        # CRITICAL FIX: Validate that all calculated positions are within the cache sequence length
        token_positions = token_manager.validate_and_adjust_positions(token_positions, 
        cache_seq_len, i)
        
        for scheme in token_schemes:
            positions_by_scheme[scheme].append(token_positions[scheme])

    # Convert lists to tensors
    for scheme, pos_list in positions_by_scheme.items():
        positions_by_scheme[scheme] = torch.tensor(pos_list, device=model.cfg.device, dtype=torch.long)

    # Generate text for all sequences first (needed for activation data)
    generated_texts = model.to_string(full_sequences)

    batch_activations = {}
    batch_indices = torch.arange(len(batch_prompts_padded), device=model.cfg.device)

    # Filter hooks to only those specified in ACTIVE_HOOKS within the desired layer range
    active_hook_keys = set()
    for layer_idx in range(start_layer, end_layer + 1):
        for hook_base_name in active_hooks:
            if hook_base_name not in ["hook_embed"]:
                active_hook_keys.add(get_hook_name_for_layer(hook_base_name, layer_idx))
    if "hook_embed" in active_hooks:
        active_hook_keys.add("hook_embed")

    logger.info(f"Extracting activations for {len(active_hook_keys)} specified hook points...")
    logger.debug(f"Cache contains {len(cache)} hooks total")
    logger.debug(f"First few cache hook names: {list(cache.keys())[:5]}")
    logger.debug(f"Expected active hooks (first 5): {list(active_hook_keys)[:5]}")

    # Check for hook name mismatches
    cache_hooks = set(cache.keys())
    active_hooks = set(active_hook_keys)
    matching_hooks = cache_hooks.intersection(active_hooks)
    missing_hooks = active_hooks - cache_hooks
    extra_hooks = cache_hooks - active_hooks
    
    logger.debug(f"Matching hooks: {len(matching_hooks)}")
    logger.debug(f"Missing hooks (expected but not in cache): {len(missing_hooks)}")
    logger.debug(f"Extra hooks (in cache but not expected): {len(extra_hooks)}")

    if debug_verbose and missing_hooks:
        logger.debug(f"All missing hooks ({len(missing_hooks)} total):")
        for i, hook in enumerate(sorted(missing_hooks), 1):
            logger.debug(f"  {i:3d}. {hook}")
    if debug_verbose and extra_hooks:
        logger.debug(f"All extra hooks ({len(extra_hooks)} total):")
        for i, hook in enumerate(sorted(extra_hooks), 1):
            logger.debug(f"  {i:3d}. {hook}")

    if len(matching_hooks) == 0:
        logger.error("CRITICAL ERROR: No hooks match between cache and expected hooks!")
        logger.error("This will result in no activations being captured!")
        return {}, []  # Return empty data to prevent HDF5 saving

    activations_captured = 0
    hooks_processed = 0
    for hook_name_full, activation_tensor in cache.items():
        hooks_processed += 1
        if hook_name_full not in active_hook_keys:
            if hooks_processed <= 3:  # Only log first few to avoid spam
                logger.debug(f"Skipping hook '{hook_name_full}' - not in active hooks")
            continue

        layer_idx, hook_base_name = parse_hook_name(hook_name_full)
        hook_key = f"layer_{layer_idx}_{hook_base_name}"
        if hook_key not in batch_activations:
            batch_activations[hook_key] = {scheme: [] for scheme in token_schemes}
        
        if activation_tensor.dtype == torch.float32:
            activation_tensor = activation_tensor.half()

        for scheme, positions in positions_by_scheme.items():
            if scheme not in token_schemes:
                continue

            try:
                # Proactive bounds checking to prevent IndexError
                # CRITICAL FIX: Different tensor shapes have sequence dimensions in different positions
                if len(activation_tensor.shape) == 3:  # [batch, seq, d_model]
                    seq_len = activation_tensor.shape[1]
                    extraction_method = "3d_standard"
                elif len(activation_tensor.shape) == 4 and hook_base_name in ["attn.hook_pattern", "attn.hook_attn_scores"]:
                    # Attention patterns/scores: [batch, heads, seq_query, seq_key]
                    # For these, we need to check both query and key sequence lengths
                    seq_query_len = activation_tensor.shape[2]
                    seq_key_len = activation_tensor.shape[3]
                    seq_len = min(seq_query_len, seq_key_len)  # Use the smaller one for safety
                    extraction_method = "4d_attention"
                elif len(activation_tensor.shape) == 4:  # [batch, seq, head, d_head]
                    seq_len = activation_tensor.shape[1]
                    extraction_method = "4d_standard"
                else:
                    logger.debug(f"Skipping {hook_key} - unsupported tensor shape: {activation_tensor.shape}")
                    continue
                
                # Check if any calculated position for this batch exceeds the tensor's sequence length.
                if torch.any(positions >= seq_len):
                    logger.debug(f"Bounds check failed for {hook_key}/{scheme}. Max position: {torch.max(positions).item()}, Seq len: {seq_len}. Tensor shape: {activation_tensor.shape}. Method: {extraction_method}. Skipped.")
                    continue  # Skip this hook/scheme if any position is invalid.

                # Vectorized extraction based on tensor shape
                if extraction_method == "3d_standard":  # [batch, seq, d_model]
                    extracted = activation_tensor[batch_indices, positions]
                elif extraction_method == "4d_attention":  # [batch, heads, seq_query, seq_key]
                    # For attention patterns, we want to extract the attention TO the specified position
                    # So we use positions as the key dimension (last dimension)
                    extracted = activation_tensor[batch_indices, :, :, positions]
                elif extraction_method == "4d_standard":  # [batch, seq, head, d_head]
                    extracted = activation_tensor[batch_indices, positions]
                else:
                    logger.debug(f"Skipping {hook_key} - unsupported extraction method: {extraction_method}")
                    continue

                activation_nps = extracted.cpu().float().numpy()

                # Validate extracted shape matches expected batch size
                expected_batch_size = len(batch_info)
                if activation_nps.shape[0] != expected_batch_size:
                    logger.error(f"Shape mismatch for {hook_key}/{scheme}. Expected batch_size={expected_batch_size}, got {activation_nps.shape[0]}")
                    continue

                for i in range(len(batch_info)):
                    # Validate that we have activation data for this sample
                    if i >= activation_nps.shape[0]:
                        logger.warning(f"Missing activation data for sample {i} in {hook_key}/{scheme}")
                        continue

                    # Extract text using the adjusted_gen_len (already truncated if needed)
                    padding_len = batch_info[i]['padding_length']
                    prompt_len = batch_info[i]['prompt_length']
                    gen_len = batch_info[i].get('adjusted_gen_len', generated_lengths[i])
                    
                    start_pos = padding_len + prompt_len
                    end_pos = start_pos + gen_len
                    answer_tokens = full_sequences[i, start_pos:end_pos]
                    trimmed_text = model.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                    
                    activation_data = {
                        'layer': layer_idx,
                        'hook_name': hook_base_name,
                        'token_scheme': scheme,
                        'token_position': int(positions[i]),
                        'row_idx': batch_info[i]['row_idx'],
                        'activations': activation_nps[i],
                        'activation_shape': activation_nps[i].shape,
                        # Include the full evaluation context
                        'question': batch_info[i]['row'].question,
                        'right_answer': batch_info[i]['row'].right_answer,
                        'gpt_answer': generated_texts[i] if i < len(generated_texts) else '',
                        'gpt_answer_trim': trimmed_text,
                        'is_correct': None,  # Will be filled during evaluation
                        'evaluator_response': None,  # Will be filled during evaluation
                        'model': model_name
                    }
                    batch_activations[hook_key][scheme].append(activation_data)
                    activations_captured += 1

            except Exception as e: # Broader catch for other potential issues
                logger.debug(f"Extraction failed for {hook_key}/{scheme} with unexpected error: {e}. Skipped.")
                continue

    logger.info(f"Successfully captured {activations_captured} activations total")
    logger.info(f"batch_activations has {len(batch_activations)} hook keys")

    if batch_activations:
        first_hook = list(batch_activations.keys())[0]
        logger.debug(f"First hook '{first_hook}' has schemes: {list(batch_activations[first_hook].keys())}")
        for scheme, activations in batch_activations[first_hook].items():
            logger.debug(f"Scheme '{scheme}' has {len(activations)} activations")

    # --- Explicit Memory Cleanup ---
    del cache
    del activation_tensor
    if 'extracted' in locals():
        del extracted
    
    # --- 4. EXTRACT TEXT FOR EVALUATION ---
    updated_batch_results = []
    for i, info in enumerate(batch_info):
        padding_len = info['padding_length']
        prompt_len = info['prompt_length']
        # Use adjusted_gen_len if period truncation was applied, otherwise use original
        gen_len = info.get('adjusted_gen_len', generated_lengths[i])

        # Extract the precise tokens that constitute the answer
        # With LEFT-padding, the answer starts after padding + prompt
        start_pos = padding_len + prompt_len
        end_pos = start_pos + gen_len
        answer_tokens = token_manager.extract_answer_tokens(full_sequences, i, start_pos, end_pos)

        # Decode the clean tokens
        generated_part = token_manager.decode_tokens_clean(answer_tokens)
        
        # No additional truncation needed - already handled by adjusted_gen_len
        trimmed_answer = generated_part.strip()

        result_item = {
            'row_idx': info['row_idx'],
            'model': model_name,
            'question': info['row'].question,
            'right_answer': info['row'].right_answer,
            'gpt_answer': generated_part,
            'gpt_answer_trim': trimmed_answer
        }
        updated_batch_results.append(result_item)

    return full_sequences, batch_activations, updated_batch_results, positions_by_scheme
