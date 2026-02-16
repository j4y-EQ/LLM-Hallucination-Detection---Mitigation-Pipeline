"""Centralized Token Management for QA Generation.

Handles all tokenization operations, token position calculations, and
tokenizer configuration for the hallucination detection pipeline. Ensures
consistent token handling between training (generator.py) and inference
(evaluate.py, real_time_inference.py).

FUNCTIONALITY:
    - Optimized QA prompt tokenization with caching
    - Left-padding for batch generation (required for causal LLMs)
    - Token position calculation for activation capture
    - Tokenizer loading with multiple fallback strategies
    - Padding token setup for LLaMA-style models

TOKEN SCHEMES:
    Calculates positions for different activation capture strategies:
    - 'bos_token': Beginning of sequence position
    - 'last_token': Final generated token position  
    - 'eos_token': End-of-sequence marker position
    - 'first_period': First period token (if truncation enabled)

KEY METHODS:
    - make_tokens_optimized(): Efficient QA prompt tokenization
    - create_left_padded_batch(): Batch creation with proper padding
    - calculate_token_positions(): Position calculation for capture
    - setup_tokenizer_padding(): LLaMA padding configuration

USAGE:
    token_manager = TokenManager(model, max_answer_tokens=100, model_dir='...')
    tokens = token_manager.make_tokens_optimized(context, question)
    batch, batch_info = token_manager.create_left_padded_batch(prompts, info)

Dependencies:
    - TransformerLens: Model and tokenizer interface
    - HuggingFace transformers: Tokenizer loading
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, Tuple
from logger import consolidated_logger as logger

class TokenManager:
    """
    Centralized token management for QA generation and activation capture.
    Handles tokenization, position calculations, and hook name utilities.
    """

    def __init__(self, model, max_answer_tokens: int, model_dir: str = None):
        """
        Initialize TokenManager with model and configuration.

        Args:
            model: The HookedTransformer model
            max_answer_tokens: Maximum number of tokens to generate
            model_dir: Directory containing model files (optional)
        """
        self.model = model
        self.max_answer_tokens = max_answer_tokens
        self.model_dir = model_dir
        self.max_ctx = model.cfg.n_ctx
        self.logger = logger  # Use the consolidated logger

        # Pre-compute constant tokenization parts for efficiency
        self._initialize_constant_tokens()

    def _initialize_constant_tokens(self):
        """Pre-compute and cache constant tokens for efficiency."""
        # Use the same device as the model
        self.device = next(self.model.parameters()).device

        # Pre-compute constant tokens
        self.context_prefix_tokens = self.model.to_tokens("Context:\n", prepend_bos=False)[0]
        self.answer_suffix_tokens = self.model.to_tokens("\nAnswer:", prepend_bos=False)[0]
        self.bos_token = torch.tensor([self.model.tokenizer.bos_token_id],
                                    device=self.device, dtype=torch.long)

    def make_tokens_optimized(self, knowledge: str, question: str):
        """
        Create optimized token sequence for QA prompt with efficient memory usage

        Args:
            knowledge (str): Context/knowledge text for the question
            question (str): The question to be answered

        Returns:
            torch.Tensor: Token sequence [prompt_length] ready for model input

        Output format: [BOS] + "Context:\n" + knowledge + "\n\nQuestion: " + question + "\nAnswer:"
        """
        # Tokenize variable parts (knowledge and question are different for each sample)
        # prepend_bos=False because we manually add BOS token
        question_tokens = self.model.to_tokens(f"\n\nQuestion: {question}", prepend_bos=False)[0]
        knowledge_tokens = self.model.to_tokens(knowledge, prepend_bos=False)[0]

        # Calculate space requirements using pre-computed constants
        # base_tokens = BOS + "Context:\n" + question + "\nAnswer:"
        base_tokens = (self.bos_token.numel() + self.context_prefix_tokens.numel() +
                      question_tokens.numel() + self.answer_suffix_tokens.numel())

        # Reserve space for generation: MAX_CTX - base_tokens - MAX_ANSWER_TOKENS = available for knowledge
        available_for_knowledge = self.max_ctx - base_tokens - self.max_answer_tokens

        # NEGATIVE TOKEN COUNT VALIDATION: Check if we have sufficient context space
        if available_for_knowledge <= 0:
            # Not enough space for any knowledge - this data point cannot be processed
            return None  # Signal that this data point should be skipped

        # Reject data point if knowledge tokens would need truncation
        if knowledge_tokens.numel() > available_for_knowledge:
            # Knowledge is too long - reject the data point without truncation
            if self.logger:
                self.logger.warning(
                    f"Knowledge tokens ({knowledge_tokens.numel()}) exceed available space "
                    f"({available_for_knowledge}). Rejecting data point without truncation."
                )
            else:
                print(
                    f"WARNING: Knowledge tokens ({knowledge_tokens.numel()}) exceed available space "
                    f"({available_for_knowledge}). Rejecting data point without truncation."
                )
            return None  # Signal that this data point should be skipped

        # Efficiently concatenate all tokens using pre-computed constant parts
        # Final structure: [BOS][Context:\n][trimmed_knowledge][\n\nQuestion: question][\nAnswer:]
        full_tokens = torch.cat([
            self.bos_token,                # [1] - Beginning of sequence token
            self.context_prefix_tokens,    # [2-3] - "Context:\n"
            knowledge_tokens,              # [variable] - Trimmed knowledge text
            question_tokens,               # [variable] - "\n\nQuestion: {question}"
            self.answer_suffix_tokens      # [2] - "\nAnswer:"
        ])

        # Safety truncation (should not be needed due to careful calculation above)
        return full_tokens[:self.max_ctx]

    def get_token_positions(self, prompt_length: int, generated_length: int, padding_length: int = 0) -> Dict[str, int]:
        """
        Calculate token positions for different extraction schemes, accounting for LEFT-padding

        Args:
            prompt_length (int): Length of the original prompt tokens (before padding)
            generated_length (int): Number of tokens generated so far (0 to MAX_ANSWER_TOKENS)
            padding_length (int): Number of LEFT-padding tokens added (0 if no padding needed)

        Returns:
            Dict[str, int]: Mapping of scheme names to absolute positions in padded tensor
                - "bos_token": Position of beginning-of-sequence token
                - "last_prompt_token": Position of last token in the prompt
                - "first_generated": Position of first generated token
                - "last_generated": Position of last generated token

        Example:
            With LEFT-padding: [PAD][PAD][PAD][BOS][Context:][knowledge][Question:][question][Answer:][gen1][gen2]
            - prompt_length = 6, generated_length = 2, padding_length = 3
            - bos_token = 3, last_prompt_token = 8, first_generated = 9, last_generated = 10
        """
        positions = {}

        # With LEFT-padding, actual content starts after padding tokens
        # Tensor structure: [PAD...PAD][BOS][prompt_tokens...][generated_tokens...]

        positions["bos_token"] = padding_length
        # BOS token is at the start of the actual content (after padding)

        positions["last_prompt_token"] = padding_length + prompt_length - 1
        # Last prompt token is at: padding_length + prompt_length - 1 (0-indexed)

        # FOR HALLUCINATION DETECTION: Capture activations AT the generated tokens themselves
        positions["first_generated"] = padding_length + prompt_length
        # First generated token immediately follows the prompt

        positions["last_generated"] = padding_length + prompt_length + generated_length - 1
        # Last generated token is at: padding_length + prompt_length + generated_length - 1 (0-indexed)
        # This is the position of the LAST MEANINGFUL generated token

        return positions

    def get_hook_name_for_layer(self, hook_base_name: str, layer_idx: int) -> str:
        """Convert base hook name to full hook name for specific layer"""
        if hook_base_name in ["hook_embed"]:
            return hook_base_name  # Layer-independent hooks
        elif "." in hook_base_name:  # Sub-component hooks like attn.hook_pattern
            return f"blocks.{layer_idx}.{hook_base_name}"
        else:  # Direct layer hooks like hook_resid_pre
            return f"blocks.{layer_idx}.{hook_base_name}"

    def load_tokenizer_with_fallbacks(self):
        """Load tokenizer with multiple fallback strategies for robustness.
        
        Implements progressive fallback sequence:
        1. Standard AutoTokenizer.from_pretrained()
        2. With trust_remote_code=True (for custom tokenizer implementations)
        3. With additional fallback parameters (for restricted environments)
        
        Each attempt validates that the loaded object has required encode/decode methods.
        
        Returns:
            transformers.PreTrainedTokenizer: Loaded and validated tokenizer.
            
        Raises:
            Exception: If all fallback strategies fail to load a valid tokenizer.
            
        Notes:
            - Validates tokenizer is not a boolean and has encode/decode methods
            - Logs each attempt and failure for debugging
            - Fallback strategies designed for corporate/restricted network environments
        """
        if self.logger:
            self.logger.info("   Loading tokenizer...")
        else:
            print("   Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            if self.logger:
                self.logger.info(f"   Tokenizer loaded successfully: {type(tokenizer)}")
            else:
                print(f"   Tokenizer loaded successfully: {type(tokenizer)}")

            # Validate tokenizer
            if isinstance(tokenizer, bool):
                raise ValueError(f"Tokenizer is a boolean ({tokenizer}), not a proper tokenizer object")
            if not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
                raise ValueError("Tokenizer missing required methods")
            return tokenizer

        except Exception as e:
            if self.logger:
                self.logger.error(f"   Tokenizer loading failed: {e}")
            else:
                print(f"   ERROR: Tokenizer loading failed: {e}")
            # Try with explicit trust_remote_code
            try:
                if self.logger:
                    self.logger.info("   Trying with trust_remote_code=True...")
                else:
                    print("   Trying with trust_remote_code=True...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
                if self.logger:
                    self.logger.info(f"   Tokenizer loaded with trust_remote_code: {type(tokenizer)}")
                else:
                    print(f"   Tokenizer loaded with trust_remote_code: {type(tokenizer)}")
                return tokenizer
            except Exception as e2:
                if self.logger:
                    self.logger.error(f"   Tokenizer loading failed even with trust_remote_code: {e2}")
                else:
                    print(f"   ERROR: Tokenizer loading failed even with trust_remote_code: {e2}")
                # Try with additional parameters for company environments
                try:
                    if self.logger:
                        self.logger.info("   Trying with additional fallback parameters...")
                    else:
                        print("   Trying with additional fallback parameters...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_dir,
                        trust_remote_code=True,
                        use_fast=False,  # Use slow tokenizer
                        local_files_only=True  # Don't try to download
                    )
                    if self.logger:
                        self.logger.info(f"   Tokenizer loaded with fallback parameters: {type(tokenizer)}")
                    else:
                        print(f"   Tokenizer loaded with fallback parameters: {type(tokenizer)}")
                    return tokenizer
                except Exception as e3:
                    if self.logger:
                        self.logger.error(f"   All tokenizer loading approaches failed: {e3}")
                    else:
                        print(f"   ERROR: All tokenizer loading approaches failed: {e3}")
                    raise e3

    def decode_tokens_clean(self, tokens):
        """Decode tokens to clean text, skipping special tokens."""
        return self.model.tokenizer.decode(tokens, skip_special_tokens=True)

    def decode_sequence_with_markers(self, full_sequences, i):
        """Decode full sequence and replace special tokens with readable markers."""
        try:
            full_text = self.model.to_string(full_sequences[i])
            if hasattr(self.model.tokenizer, 'eos_token') and self.model.tokenizer.eos_token:
                full_text = full_text.replace(self.model.tokenizer.eos_token, '[EOS]')
            if hasattr(self.model.tokenizer, 'pad_token') and self.model.tokenizer.pad_token:
                full_text = full_text.replace(self.model.tokenizer.pad_token, '[PAD]')
        except Exception as e:
            full_text = f"Error decoding sequence: {e}"
        return full_text

    def decode_individual_token_with_debug(self, generated_part_tokens, full_sequences, i, start_of_generation, idx):
        """Decode a single token at given position with special token handling for debug output."""
        token_id = generated_part_tokens[idx].item()
        try:
            # Decode individual token
            word = self.model.to_string(full_sequences[i, start_of_generation + idx:start_of_generation + idx + 1])
            # Clean up special tokens for display
            if hasattr(self.model.tokenizer, 'eos_token') and self.model.tokenizer.eos_token and word == self.model.tokenizer.eos_token:
                word = '<|eot_id|>'
            elif hasattr(self.model.tokenizer, 'pad_token') and self.model.tokenizer.pad_token and word == self.model.tokenizer.pad_token:
                word = '<|pad|>'
            elif hasattr(self.model.tokenizer, 'bos_token') and self.model.tokenizer.bos_token and word == self.model.tokenizer.bos_token:
                word = '<|bos|>'

            return f"    [{idx:2d}] Token ID: {token_id:5d} | Word: '{word}'"
        except Exception as e:
            return f"    [{idx:2d}] Token ID: {token_id:5d} | Word: <decode_error: {e}>"

    def validate_and_adjust_positions(self, token_positions, cache_seq_len, i):
        """Validate and adjust token positions to fit within cache sequence length.
        
        Checks if calculated token positions exceed the cached sequence length,
        which can happen with padding or generation artifacts. Automatically
        adjusts out-of-bounds positions to the last valid index.
        
        Args:
            token_positions (dict): Mapping of {scheme: position_index} for this sample.
            cache_seq_len (int): Maximum valid sequence length from cache.
            i (int): Sample index for logging.
            
        Returns:
            dict: Validated token_positions with any out-of-bounds positions adjusted.
            
        Notes:
            - Checks max position across all schemes against cache_seq_len
            - Adjusts out-of-bounds positions to (cache_seq_len - 1)
            - Logs warnings when adjustments are made
            - Prevents index errors during activation capture
        """
        max_position = max(token_positions.values())
        if max_position >= cache_seq_len:
            if self.logger:
                self.logger.warning(f"Calculated position {max_position} exceeds cache seq len {cache_seq_len} for sample {i}")
                self.logger.warning(f"  Token positions: {token_positions}")
            else:
                print(f"WARNING: Calculated position {max_position} exceeds cache seq len {cache_seq_len} for sample {i}")
                print(f"  Token positions: {token_positions}")

            # Adjust positions to fit within cache sequence length
            # Use the last valid position for any scheme that would exceed bounds
            adjusted_positions = {}
            for scheme, pos in token_positions.items():
                if pos >= cache_seq_len:
                    adjusted_positions[scheme] = cache_seq_len - 1  # Use last valid position
                    if self.logger:
                        self.logger.warning(f"  Adjusted {scheme} from {pos} to {cache_seq_len - 1}")
                    else:
                        print(f"WARNING: Adjusted {scheme} from {pos} to {cache_seq_len - 1}")
                else:
                    adjusted_positions[scheme] = pos
            token_positions = adjusted_positions

        return token_positions

    def setup_tokenizer_padding(self, model):
        """Setup tokenizer padding configuration for LLaMA models.
        
        LLaMA tokenizers don't have pad_token by default. This sets pad_token to
        eos_token (standard practice) and critically updates HookedTransformer's
        internal .cfg object, which is not updated by from_pretrained().
        
        Args:
            model (HookedTransformer): Model instance to configure.
            
        Returns:
            HookedTransformer: Model with properly configured pad_token_id.
            
        Notes:
            - Sets pad_token = eos_token if pad_token is None (LLaMA standard)
            - Updates both tokenizer.pad_token_id AND model.cfg.pad_token_id
            - Critical fix: HookedTransformer.cfg must be manually updated
            - Without this, padding tokens may not be properly ignored
        """
        # Llama tokenizers often do not have a padding token set by default.
        # For Llama models, use the EOS token as the padding token (standard practice)
        if model.tokenizer.pad_token is None:
            if self.logger:
                self.logger.info("Tokenizer `pad_token` not set. Using EOS token as pad token.")
            else:
                print("Tokenizer `pad_token` not set. Using EOS token as pad token.")
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
            # Ensure the model's config is updated with the pad token ID
            model.cfg.pad_token_id = model.tokenizer.pad_token_id

        # This is the critical fix. We must manually set the pad_token_id on the
        # HookedTransformer's internal config object, which is called `.cfg`.
        # This is because from_pretrained ignores the pad_token_id in the hf_model
        # config that is passed to it, and re-builds its own from a default.
        model.cfg.pad_token_id = model.tokenizer.pad_token_id

        return model

    def get_token_ids_debug_info(self, model):
        """Log debug information about token IDs."""
        if self.logger:
            self.logger.debug(f"EOS token ID: {model.tokenizer.eos_token_id}")
            self.logger.debug(f"PAD token ID: {model.tokenizer.pad_token_id}")
            self.logger.debug(f"BOS token ID: {model.tokenizer.bos_token_id}")
            if hasattr(model.tokenizer, 'unk_token_id'):
                self.logger.debug(f"UNK token ID: {model.tokenizer.unk_token_id}")
        else:
            print(f"EOS token ID: {model.tokenizer.eos_token_id}")
            print(f"PAD token ID: {model.tokenizer.pad_token_id}")
            print(f"BOS token ID: {model.tokenizer.bos_token_id}")
            if hasattr(model.tokenizer, 'unk_token_id'):
                print(f"UNK token ID: {model.tokenizer.unk_token_id}")

    def extract_answer_tokens(self, full_sequences, i, start_pos, end_pos):
        """Extract a range of tokens from a sequence."""
        return full_sequences[i, start_pos:end_pos]

    def extract_generated_part_tokens(self, full_sequences, i, start_of_generation, actual_seq_len):
        """Extract the generated part of tokens from a sequence."""
        return full_sequences[i, start_of_generation:actual_seq_len]

    def create_left_padded_batch(self, batch_prompts, batch_info):
        """
        Create LEFT-padded batch tensor from prompts with identical logic to both generator.py and evaluate.py
        
        Args:
            batch_prompts: List of tokenized prompts
            batch_info: List of batch item metadata dicts
            
        Returns:
            tuple: (batch_tensor, updated_batch_info)
        """
        # Efficient memory-optimized LEFT-padding. This is required for decoder-only models like Llama.
        prompt_lengths = [p.shape[0] for p in batch_prompts]
        max_len = max(prompt_lengths)
        batch_size = len(batch_prompts)
        
        # Pre-allocate with pad_token_id
        batch_tensor = torch.full(
            (batch_size, max_len), 
            self.model.tokenizer.pad_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        for i, prompt in enumerate(batch_prompts):
            prompt_len = prompt_lengths[i]
            # Place the prompt at the right end of the tensor (LEFT-padding)
            batch_tensor[i, -prompt_len:] = prompt.to(self.device)
            # Update padding length for position calculations
            batch_info[i]['padding_length'] = max_len - prompt_len
        
        return batch_tensor, batch_info