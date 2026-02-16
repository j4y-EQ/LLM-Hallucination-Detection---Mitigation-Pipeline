"""
Common utility functions for steering pipeline scripts.

This module consolidates duplicated functions across steering scripts to prevent
code duplication and improve maintainability.

Classes:
- DatasetHandler: Unified dataset loading, formatting, and evaluation

Functions:
- parse_choices(): Robustly parse choices column from dataset
- make_qa_prompt(): Create formatted QA prompts with MCQ options
- extract_answer_from_batch(): Extract answer text from generated sequences accounting for left-padding
- extract_left_padded_answer(): Extract answer text and full output with left-padding accounting
"""

import pandas as pd
import numpy as np
import torch
import logging
import os
import json
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from steer.utils.eval_model_steer import batch_judge_answers_mcq
from helpers.eval_model import batch_judge_answers
from steer.utils.dataset_formats import VALID_DATASET_FORMATS, MCQ_FORMATS

# ================================================================
# MODEL CONFIGURATION
# ================================================================
MODEL_CONFIGS = {
    'qwen': {
        'model_name': 'qwen2.5_7B_instruct',
        'huggingface_model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'transformer_lens_model_name': 'Qwen/Qwen2.5-7B-Instruct'
    },
    'llama': {
        'model_name': 'llamainstruct',
        'huggingface_model_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'transformer_lens_model_name': 'meta-llama/Meta-Llama-3-8B-Instruct'
    },
    'gemma': {
        'model_name': 'gemma2_9b_it',
        'huggingface_model_id': 'google/gemma-2-9b-it',
        'transformer_lens_model_name': 'google/gemma-2-9b-it'
    }
}

# ================================================================
# DATASET FORMAT VALIDATION (CENTRALIZED)
# ================================================================

def get_valid_dataset_formats() -> list:
    """
    Get the list of valid dataset formats for argparse choices.
    
    This is the SINGLE SOURCE OF TRUTH for valid dataset formats.
    All argparse definitions and validation should use this function.
    
    Returns:
        list: Valid dataset format strings ['mcq', 'non_mcq', 'mmlu', 'hellaswag']
    """
    return list(VALID_DATASET_FORMATS)


def validate_dataset_format(dataset_format: Optional[str]) -> str:
    """
    Validate dataset_format with strict enforcement.
    
    STRICT VALIDATION RULES:
    - dataset_format MUST be provided (cannot be None)
    - dataset_format MUST be one of the valid formats
    - No defaults, no inference, no fallback behavior
    - Raises ValueError immediately on validation failure
    
    Args:
        dataset_format: The dataset format to validate (can be None)
        
    Returns:
        str: The validated dataset_format string
        
    Raises:
        ValueError: If dataset_format is None, empty string, or not in valid formats
    """
    valid_formats = get_valid_dataset_formats()
    
    # STRICT: No defaults or inference
    if dataset_format is None:
        raise ValueError(
            f"dataset_format is REQUIRED and cannot be None. "
            f"Valid formats: {valid_formats}"
        )
    
    if isinstance(dataset_format, str):
        dataset_format = dataset_format.strip()
    
    if not dataset_format or dataset_format == '':
        raise ValueError(
            f"dataset_format cannot be empty. "
            f"Valid formats: {valid_formats}"
        )
    
    if dataset_format not in valid_formats:
        raise ValueError(
            f"Invalid dataset_format: '{dataset_format}'. "
            f"Valid formats: {valid_formats}"
        )
    
    return dataset_format


def load_baseline_config(baseline_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load baseline configuration from baseline_config.json.
    
    Args:
        baseline_dir: Directory containing baseline results
        
    Returns:
        Dict with baseline config, or None if file not found
    """
    baseline_config_path = os.path.join(baseline_dir, 'baseline_config.json')
    if os.path.exists(baseline_config_path):
        with open(baseline_config_path, 'r') as f:
            return json.load(f)
    return None


def adapt_messages_for_model(messages, model_name: str):
    """
    Adapt chat messages for models that don't support certain roles.
    
    This function handles model-specific requirements for chat templates:
    - Gemma: Doesn't support system role - merges system into first user message
    - Llama: Supports system role natively
    - Qwen: Supports system role natively
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Name of the model (e.g., 'gemma', 'llama', 'qwen')
    
    Returns:
        Adapted messages list compatible with the model's chat template
    """
    # Gemma doesn't support system role - merge system into first user message
    if model_name == 'gemma':
        adapted = []
        system_content = None
        
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                if system_content:
                    # Merge system message into user message
                    msg = {
                        'role': 'user',
                        'content': f"{system_content}\n\n{msg['content']}"
                    }
                    system_content = None  # Only merge once
                adapted.append(msg)
            else:
                adapted.append(msg)
        
        return adapted
    
    # Other models (Llama, Qwen) support system role as-is
    return messages


def parse_choices(choices_data):
    """
    Robustly parses the 'choices' column which might be a dict,
    or a string representation of a dict containing numpy arrays.
    
    This function handles various formats found in CSV files, including:
    - Direct Python dict objects
    - String representations of dicts with numpy arrays
    - Missing/null values
    
    Args:
        choices_data: The choices data from dataset, can be:
                     - dict: {'label': ['A', 'B', 'C'], 'text': ['opt1', 'opt2', 'opt3']}
                     - str: String representation of above dict
                     - None/NaN/empty: Returns None

    Returns:
        dict: Parsed choices dictionary with 'label' and 'text' keys
        None: If parsing fails or input is None/empty
        
    Example:
        >>> choices = parse_choices("{'label': array(['A', 'B']), 'text': array(['Yes', 'No'])}")
        >>> print(choices)
        {'label': ['A', 'B'], 'text': ['Yes', 'No']}
        
    Security Note:
        Uses restricted eval() with limited namespace to safely parse string representations.
        Only allows numpy array construction, no arbitrary code execution.
    """
    if isinstance(choices_data, dict):
        return choices_data

    if pd.isna(choices_data) or choices_data == "" or choices_data is None:
        return None

    try:
        context = {
            'array': np.array,
            'nan': np.nan,
            'dtype': None,
            'object': object,
            'int64': np.int64,
            'float64': np.float64
        }
        return eval(str(choices_data), {"__builtins__": {}}, context)
    except Exception as e:
        # Silently fail and return None - caller handles None gracefully
        return None


def make_qa_prompt(context: str, question: str, choices: dict = None, use_mcq: bool = True, dataset_format: str = None, model_type: str = None) -> list:
    """
    Create a list of messages for chat template (supports multiple model types).

    Args:
        context: The context/passage for the question
        question: The question to ask
        choices: Optional dict with 'label' and 'text' keys for multiple choice options
        use_mcq: If True, include MCQ options if provided. If False, use simple format.
        dataset_format: Format of the dataset ('mcq', 'non_mcq', 'mmlu', or 'hellaswag'). REQUIRED - must be provided.
        model_type: Type of model ('llama', 'qwen', 'gemma'). Used to handle model-specific prompt requirements.

    Returns:
        list: A list of message dictionaries for apply_chat_template
        
    Raises:
        ValueError: If dataset_format is None or not one of the valid formats
    """
    # Validate dataset_format - required parameter, no default
    if dataset_format is None:
        raise ValueError(f"dataset_format is REQUIRED and must not be None. Valid formats: {VALID_DATASET_FORMATS}")
    if dataset_format not in VALID_DATASET_FORMATS:
        raise ValueError(f"Invalid dataset_format: '{dataset_format}'. Must be one of: {VALID_DATASET_FORMATS}")
    
    # For MMLU, use simplified prompt without RAG system message
    if dataset_format == 'mmlu':
        user_content = f"Question:\n{question}\n"
        
        # Add MCQ options if provided
        if choices and isinstance(choices, dict):
            labels = choices.get('label', [])
            texts = choices.get('text', [])

            # Ensure labels and texts are lists (handle numpy arrays)
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
            if hasattr(texts, 'tolist'):
                texts = texts.tolist()

            # Only add options if we have both labels and texts
            if labels and texts and len(labels) == len(texts):
                user_content += "\nChoices:\n"
                for label, text in zip(labels, texts):
                    user_content += f"{label}. {text}\n"

        user_content += "\nAnswer:"
        
        messages = [
            {"role": "user", "content": user_content}
        ]
    elif dataset_format == 'hellaswag':
        # HellaSwag format: system instruction + context (incomplete sentence) + choices (endings)
        # No separate question - context is the prompt
        system_prompt = "Complete the following sentence by choosing the most logical continuation."
        
        user_content = f"Context:\n{context}\n"
        
        # Add MCQ options (endings)
        if choices and isinstance(choices, dict):
            labels = choices.get('label', [])
            texts = choices.get('text', [])

            # Ensure labels and texts are lists (handle numpy arrays)
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
            if hasattr(texts, 'tolist'):
                texts = texts.tolist()

            # Only add options if we have both labels and texts
            if labels and texts and len(labels) == len(texts):
                user_content += "\nChoices:\n"
                for label, text in zip(labels, texts):
                    user_content += f"{label}. {text}\n"

        user_content += "\nAnswer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    else:
        # RAG system prompt for non-MMLU/non-HellaSwag formats
        system_prompt = "You are an expert in retrieval-based question answering. Please respond with the exact answer, using only the information provided in the context."

        if use_mcq:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}\n"

            # Add MCQ options if provided
            if choices and isinstance(choices, dict):
                labels = choices.get('label', [])
                texts = choices.get('text', [])

                # Ensure labels and texts are lists (handle numpy arrays)
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                if hasattr(texts, 'tolist'):
                    texts = texts.tolist()

                # Only add options if we have both labels and texts
                if labels and texts and len(labels) == len(texts):
                    user_content += "\nOptions:\n"
                    for label, text in zip(labels, texts):
                        user_content += f"{label}. {text}\n"

            user_content += "\nAnswer:"

            # Handle Gemma: doesn't support system role, prepend to user content
            if model_type == 'gemma':
                user_content = f"{system_prompt}\n\n{user_content}"
                messages = [
                    {"role": "user", "content": user_content}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
        else:
            # Simple format without MCQ options
            user_content = f"Context: {context} Question: {question} Answer:"
            
            # Handle Gemma: doesn't support system role, prepend to user content
            if model_type == 'gemma':
                user_content = f"{system_prompt}\n\n{user_content}"
                messages = [
                    {"role": "user", "content": user_content}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
    
    return messages


def make_qa_prompt_with_answer(
    context: str,
    question: str,
    choices: dict,
    answer_key: str,
    use_mcq: bool = True,
    dataset_format: str = None,
    tokenizer = None
) -> str:
    """
    Create a QA prompt with the answer appended.
    
    This function builds on make_qa_prompt() by adding the selected answer
    at the end. Used for contrastive pair generation where we want to capture
    activations at the answer token position.
    
    Args:
        context: The context/passage for the question
        question: The question to ask
        choices: Dict with 'label' and 'text' keys for MCQ options
        answer_key: Which choice key to append as the answer (e.g., 'A', 'B')
        use_mcq: If True, format as "A. answer_text", else just "answer_text"
        dataset_format: Format of the dataset ('mcq', 'non_mcq', or 'mmlu'). Required.
        tokenizer: Optional tokenizer to apply chat template
    
    Returns:
        str: Complete prompt string with answer appended
        
    Raises:
        ValueError: If answer_key not found in choices
    """
    if dataset_format is None:
        dataset_format = 'mcq'  # Default to MCQ
    
    # Get base prompt messages
    messages = make_qa_prompt(context, question, choices, use_mcq, dataset_format)
    
    # Extract answer text from choices
    if not choices or 'label' not in choices or 'text' not in choices:
        raise ValueError("Choices must be a dict with 'label' and 'text' keys")
    
    labels = choices.get('label', [])
    texts = choices.get('text', [])
    
    # Handle numpy arrays
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
    
    # Find the answer text, handling label format mismatches (A/B/C/D vs 1/2/3/4)
    answer_idx = None
    actual_answer_key = answer_key
    
    # First try direct match
    if answer_key in labels:
        answer_idx = labels.index(answer_key)
    else:
        # If no direct match, try to interpret answer_key as position index
        try:
            # Check if it's a letter (A, B, C, D, etc.)
            if len(answer_key) == 1 and answer_key.isalpha():
                answer_idx = ord(answer_key.upper()) - ord('A')
            else:
                # Try to parse as integer
                answer_idx = int(answer_key) - 1  # Convert 1-indexed to 0-indexed
        except (ValueError, IndexError):
            raise ValueError(f"Answer key '{answer_key}' not found in choices {labels} and cannot be interpreted as position index")
    
    # Validate index is within bounds
    if answer_idx is None or answer_idx >= len(labels) or answer_idx < 0:
        raise ValueError(f"Answer key '{answer_key}' maps to invalid index {answer_idx} for {len(labels)} choices")
    
    try:
        answer_text = texts[answer_idx]
        actual_answer_key = labels[answer_idx]  # Use the actual key from choices
    except IndexError:
        raise ValueError(f"Answer key '{answer_key}' maps to index {answer_idx} which is out of range for choices")
    
    # Format answer based on use_mcq
    if use_mcq:
        answer_str = f"{actual_answer_key}. {answer_text}"
    else:
        answer_str = answer_text
    
    # Append answer to the last message
    if messages:
        last_message = messages[-1]
        if isinstance(last_message['content'], str):
            last_message['content'] = last_message['content'] + " " + answer_str
    
    # Apply chat template if tokenizer provided
    if tokenizer is not None:
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Build string manually from messages
        prompt_str = ""
        for msg in messages:
            prompt_str += f"{msg['content']}\n"
    
    return prompt_str


def extract_answer_from_batch(full_sequences: torch.Tensor,
                             prompt_lengths: torch.Tensor,
                             batch_prompts_padded: torch.Tensor,
                             tokenizer,
                             batch_idx: int = 0) -> Tuple[str, Dict]:
    """
    Extract answer text from generated sequences accounting for left-padding.

    Handles the critical left-padding offset calculation:
    [PAD...PAD][prompt][generated] -> extract [generated] portion only

    Args:
        full_sequences: Full generated sequences from model (shape: [batch_size, seq_len])
        prompt_lengths: Lengths of prompts from attention mask (shape: [batch_size])
        batch_prompts_padded: Original padded input sequences (shape: [batch_size, max_len])
        tokenizer: Tokenizer instance with eos_token_id
        batch_idx: Index within batch to extract (default: 0)

    Returns:
        Tuple of (answer_text, extraction_info_dict)
        - answer_text: The extracted and decoded answer string
        - extraction_info: Dict containing extraction metadata for debugging
    """
    prompt_len = prompt_lengths[batch_idx].item()

    # CRITICAL: Account for left-padding offset
    # With left-padding: [PAD...PAD][prompt][generated]
    # padding_length = total_width - prompt_len
    max_len = batch_prompts_padded.shape[1]
    padding_len = max_len - prompt_len

    # Absolute position where generation starts (after left-padding + prompt)
    start_of_generation = padding_len + prompt_len
    actual_seq_len = full_sequences.shape[1]

    # Find where generation actually ends (look for EOS token)
    generated_part_tokens = full_sequences[batch_idx, start_of_generation:actual_seq_len]
    first_eos_pos = -1
    for idx_token, token_id in enumerate(generated_part_tokens):
        if token_id.item() == tokenizer.eos_token_id:
            first_eos_pos = idx_token
            break

    if first_eos_pos != -1:
        gen_len = first_eos_pos + 1
    else:
        gen_len = len(generated_part_tokens)

    # Extract exact answer tokens
    end_of_generation = start_of_generation + gen_len
    answer_tokens = full_sequences[batch_idx, start_of_generation:end_of_generation]

    # Decode answer with skip_special_tokens=True to remove EOS/PAD tokens
    answer_text = tokenizer.decode(
        answer_tokens, skip_special_tokens=True).strip()

    extraction_info = {
        'prompt_length': prompt_len,
        'padding_length': padding_len,
        'start_of_generation': start_of_generation,
        'end_of_generation': end_of_generation,
        'generation_length': gen_len,
        'eos_found': first_eos_pos != -1,
        'max_batch_len': max_len,
        'actual_seq_len': actual_seq_len,
    }

    return answer_text, extraction_info


def extract_left_padded_answer(full_sequences: torch.Tensor,
                               prompt_lengths: torch.Tensor,
                               batch_prompts_padded: torch.Tensor,
                               tokenizer,
                               batch_idx: int = 0) -> Tuple[str, str]:
    """
    Extract both answer text and full output from left-padded sequences.
    
    Returns:
        Tuple of (answer_text, full_output_text)
        - answer_text: Just the generated answer (after prompt)
        - full_output_text: Full generation including prompt (without padding)
    """
    prompt_len = prompt_lengths[batch_idx].item()
    max_len = batch_prompts_padded.shape[1]
    padding_len = max_len - prompt_len
    
    # Calculate generation start position (after left-padding + prompt)
    start_of_generation = padding_len + prompt_len
    actual_seq_len = full_sequences.shape[1]
    
    # Find EOS token
    generated_part_tokens = full_sequences[batch_idx, start_of_generation:actual_seq_len]
    first_eos_pos = -1
    for idx_token, token_id in enumerate(generated_part_tokens):
        if token_id.item() == tokenizer.eos_token_id:
            first_eos_pos = idx_token
            break
    
    if first_eos_pos != -1:
        gen_len = first_eos_pos + 1
    else:
        gen_len = len(generated_part_tokens)
    
    # Extract answer (generated portion only)
    end_of_generation = start_of_generation + gen_len
    answer_tokens = full_sequences[batch_idx, start_of_generation:end_of_generation]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Extract full output (everything after padding, skip special tokens)
    full_text = tokenizer.decode(
        full_sequences[batch_idx, padding_len:], skip_special_tokens=True).strip()
    
    return answer_text, full_text


class DatasetHandler:
    """
    Unified dataset handling for steering experiments.
    
    Handles:
    - Loading datasets (CSV or MMLU)
    - Formatting right answers based on MCQ mode
    - Creating QA prompts (MCQ vs non-MCQ)
    - Evaluating answers using appropriate evaluator
    
    Attributes:
        use_mcq: Boolean flag controlling MCQ vs non-MCQ behavior
        logger: Logger instance
    """
    
    def __init__(self, use_mcq: bool = True, dataset_format: str = None, model_type: str = None, logger=None):
        """
        Initialize DatasetHandler.
        
        Args:
            use_mcq: If True, use MCQ format (letter + answer). 
                    If False, use simple text format.
            dataset_format: Format of the dataset ('mcq', 'non_mcq', or 'mmlu'). REQUIRED - must be provided.
            model_type: Type of model ('llama', 'qwen', 'gemma'). Used to handle model-specific prompt requirements.
            logger: Logger instance (creates default if not provided)
            
        Raises:
            ValueError: If dataset_format is None or not one of the valid formats
        """
        # Validate dataset_format - required parameter, no default
        if dataset_format is None:
            raise ValueError(f"dataset_format is REQUIRED and must not be None. Valid formats: {VALID_DATASET_FORMATS}")
        if dataset_format not in VALID_DATASET_FORMATS:
            raise ValueError(f"Invalid dataset_format: '{dataset_format}'. Must be one of: {VALID_DATASET_FORMATS}")
        
        self.use_mcq = use_mcq
        self.dataset_format = dataset_format
        self.model_type = model_type
        self.logger = logger or logging.getLogger(__name__)
    
    def load_dataset(self, dataset_path: str, 
                     num_samples: int = 5000,
                     dataset_format: str = 'mcq') -> pd.DataFrame:
        """
        Load dataset from CSV and normalize to standard format.
        
        Args:
            dataset_path: Path to CSV file
            num_samples: Max number of samples to load
            dataset_format: Format of the dataset:
                - 'mcq': Has 'knowledge'/'context', 'question', 'answerKey', 'answer', 'choices'
                - 'non_mcq': Has 'knowledge', 'question', 'answer' (no MCQ options)
                - 'mmlu': Has 'question', 'choices', 'answer' (no context, answer is choice index)
            
        Returns:
            pd.DataFrame: Standardized dataset with columns:
                        context, question, choices, answerKey, answer
        """
        df = self._load_csv(dataset_path, num_samples)
        df = self._normalize_dataset(df, dataset_format)
        return df
    
    def _load_csv(self, dataset_path: str, num_samples: int) -> pd.DataFrame:
        """Load CSV dataset."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        self.logger.info(f"Loading CSV dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        self.logger.info(f"Loaded {len(df)} total samples")
        
        if len(df) > num_samples:
            df = df.head(num_samples)
        
        self.logger.info(f"Using {len(df)} samples")
        return df
    
    def _normalize_dataset(self, df: pd.DataFrame, dataset_format: str) -> pd.DataFrame:
        """
        Normalize different dataset formats to standard format.
        
        Converts various dataset formats to a standard format with columns:
        context, question, choices, answerKey, answer
        
        Args:
            df: DataFrame to normalize
            dataset_format: Format of the input dataset
                - 'mcq': MCQ format with context, question, choices, answerKey, answer
                - 'non_mcq': No MCQ options, only context/knowledge and answer
                - 'mmlu': No context, choices are string array, answer is choice index
            
        Returns:
            Normalized DataFrame
        """
        # Helper to normalize answer column - use 'answer' if exists, else 'right_answer'
        def normalize_answer_column(df):
            """Ensure 'answer' column exists, using 'right_answer' if 'answer' doesn't exist."""
            if 'answer' not in df.columns and 'right_answer' in df.columns:
                df = df.rename(columns={'right_answer': 'answer'})
            return df
        
        if dataset_format == 'mcq':
            # MCQ format - normalize column names
            self.logger.info(f"Dataset format: mcq (context, question, choices, answerKey, answer)")
            
            # Normalize answer column first
            df = normalize_answer_column(df)
            
            # Rename 'knowledge' to 'context' if needed
            if 'knowledge' in df.columns and 'context' not in df.columns:
                df = df.rename(columns={'knowledge': 'context'})
            elif 'context' not in df.columns:
                df['context'] = ''
            
            # Ensure answerKey and answer columns exist
            if 'answerKey' not in df.columns:
                df['answerKey'] = ''
            if 'answer' not in df.columns:
                df['answer'] = ''
            if 'choices' not in df.columns:
                df['choices'] = None
            
            return df
        
        elif dataset_format == 'non_mcq':
            # Non-MCQ format: knowledge, question, answer (no MCQ)
            self.logger.info(f"Dataset format: non_mcq (knowledge, question, answer)")
            
            # Normalize answer column first
            df = normalize_answer_column(df)
            
            # Rename 'knowledge' to 'context' if needed
            if 'knowledge' in df.columns and 'context' not in df.columns:
                df = df.rename(columns={'knowledge': 'context'})
            elif 'context' not in df.columns:
                df['context'] = ''
            
            # Ensure required columns exist
            if 'question' not in df.columns:
                raise ValueError("Dataset must have 'question' column")
            if 'answer' not in df.columns:
                raise ValueError("Dataset must have 'answer' column (either 'answer' or 'right_answer')")
            
            # Set answerKey (empty for non-MCQ) and choices to None for non-MCQ format
            if 'answerKey' not in df.columns:
                df['answerKey'] = ''
            if 'choices' not in df.columns:
                df['choices'] = None
            
            return df[['context', 'question', 'choices', 'answerKey', 'answer']]
        
        elif dataset_format == 'mmlu':
            # MMLU format: question, choices (as string array of choice texts), answer (as index)
            self.logger.info(f"Dataset format: mmlu (question, choices as choice text array, answer as index)")
            
            # Normalize answer column first
            df = normalize_answer_column(df)
            
            if 'question' not in df.columns:
                raise ValueError("MMLU dataset must have 'question' column")
            if 'choices' not in df.columns:
                raise ValueError("MMLU dataset must have 'choices' column")
            if 'answer' not in df.columns:
                raise ValueError("MMLU dataset must have 'answer' column (either 'answer' or 'right_answer')")
            
            # No context in MMLU format
            df['context'] = ''
            
            # Parse choices string array and convert answer index to answerKey
            def parse_mmlu_row(row):
                choices_str = row['choices']
                answer_idx = row['answer']
                
                # Parse the string array format containing choice texts
                # Format: "['choice1' 'choice2' 'choice3' 'choice4']" (numpy-style with spaces)
                # Handles escaped quotes within strings
                choices_list = []
                try:
                    if isinstance(choices_str, str):
                        import re
                        
                        # Strategy: Parse numpy-style array with proper quote handling
                        # Format: ['item1' 'item2' "item3"] or with newlines: ['item1'\n 'item2']
                        
                        # First, remove the outer brackets
                        cleaned = choices_str.strip()
                        if cleaned.startswith('[') and cleaned.endswith(']'):
                            cleaned = cleaned[1:-1]
                        
                        # Use state machine to parse quoted strings, handling both ' and "
                        choices_list = []
                        current_choice = []
                        in_quote = False
                        quote_char = None  # Track which quote character opened the string
                        i = 0
                        
                        while i < len(cleaned):
                            char = cleaned[i]
                            
                            # Check if this is a quote character (either ' or ")
                            if char in ('"', "'") and (i == 0 or cleaned[i-1] != '\\'):
                                if not in_quote:
                                    # Opening quote
                                    in_quote = True
                                    quote_char = char
                                elif char == quote_char:
                                    # Closing quote (matching the opening)
                                    in_quote = False
                                    if current_choice:
                                        choices_list.append(''.join(current_choice))
                                        current_choice = []
                                    quote_char = None
                                else:
                                    # Different quote inside string - keep it
                                    current_choice.append(char)
                            elif in_quote:
                                # Inside quotes - collect the character (including newlines)
                                current_choice.append(char)
                            # else: outside quotes, skip (whitespace/newlines between items)
                            
                            i += 1
                        
                        # Handle any remaining choice
                        if current_choice:
                            choices_list.append(''.join(current_choice))
                        
                        # Clean up whitespace (but preserve internal structure)
                        choices_list = [c.strip() for c in choices_list if c.strip()]
                        
                    elif isinstance(choices_str, (list, tuple, np.ndarray)):
                        choices_list = [str(c).strip() for c in choices_str]
                    else:
                        choices_list = []
                except Exception as e:
                    self.logger.warning(f"Could not parse choices: {repr(choices_str)}, error: {e}")
                    choices_list = []
                
                # Convert answer index to letter (0->A, 1->B, 2->C, 3->D)
                try:
                    answer_idx_int = int(answer_idx) if not isinstance(answer_idx, int) else answer_idx
                    if 0 <= answer_idx_int < len(choices_list):
                        answerKey = chr(ord('A') + answer_idx_int)
                        answer_text = choices_list[answer_idx_int]
                    else:
                        self.logger.warning(
                            f"Answer index {answer_idx_int} out of range for {len(choices_list)} choices. "
                            f"Row index: {row.name}, Question: {row['question'][:80]}..., "
                            f"Choices: {choices_list}, Raw: {repr(choices_str)}"
                        )
                        answerKey = ''
                        answer_text = ''
                except Exception as e:
                    self.logger.warning(f"Could not convert answer index: {answer_idx}, error: {e}")
                    answerKey = ''
                    answer_text = ''
                
                # Format choices as dict with 'label' and 'text'
                labels = [chr(ord('A') + i) for i in range(len(choices_list))]
                choices_dict = {
                    'label': labels,
                    'text': choices_list
                }
                
                return pd.Series({
                    'context': '',
                    'question': row['question'],
                    'choices': choices_dict,
                    'answerKey': answerKey,
                    'answer': answer_text
                })
            
            df = df.apply(parse_mmlu_row, axis=1)
            
            # Filter out rows with invalid answer keys (empty answerKey means index was out of range)
            initial_count = len(df)
            df = df[df['answerKey'] != '']
            invalid_count = initial_count - len(df)
            if invalid_count > 0:
                self.logger.warning(f"Filtered out {invalid_count} rows with answer indices out of range for choices")
            
            return df[['context', 'question', 'choices', 'answerKey', 'answer']]
        
        elif dataset_format == 'hellaswag':
            # HellaSwag format: context (ctx), choices (as JSON array of ending texts), answer (as index 0-3)
            # No question column - context is the incomplete sentence/setup
            self.logger.info(f"Dataset format: hellaswag (ctx as context, choices as JSON array, answer as index)")
            
            # Rename 'ctx' to 'context' if needed
            if 'ctx' in df.columns and 'context' not in df.columns:
                df = df.rename(columns={'ctx': 'context'})
            elif 'context' not in df.columns:
                # Try to use activity_label as fallback context
                if 'activity_label' in df.columns:
                    df['context'] = df['activity_label']
                else:
                    df['context'] = ''
            
            # Validate required columns
            if 'choices' not in df.columns:
                raise ValueError("HellaSwag dataset must have 'choices' column")
            if 'answer' not in df.columns:
                raise ValueError("HellaSwag dataset must have 'answer' column")
            
            # Parse choices and convert answer index to answerKey
            def parse_hellaswag_row(row):
                choices_str = row['choices']
                answer_idx = row['answer']
                
                # Parse choices - HellaSwag stores as JSON array string ["item1", "item2", ...]
                choices_list = []
                try:
                    if isinstance(choices_str, str):
                        # Try to parse as JSON array first (most common for HellaSwag)
                        try:
                            import json as json_module
                            choices_list = json_module.loads(choices_str)
                            # Ensure all items are strings
                            choices_list = [str(c).strip() for c in choices_list]
                        except (json_module.JSONDecodeError, ValueError):
                            # Fallback: handle numpy-style array format ['item1' 'item2']
                            import re
                            
                            # Remove outer brackets
                            cleaned = choices_str.strip()
                            if cleaned.startswith('[') and cleaned.endswith(']'):
                                cleaned = cleaned[1:-1]
                            
                            # Use state machine to parse quoted strings
                            choices_list = []
                            current_choice = []
                            in_quote = False
                            quote_char = None
                            i = 0
                            
                            while i < len(cleaned):
                                char = cleaned[i]
                                
                                if char in ('"', "'") and (i == 0 or cleaned[i-1] != '\\'):
                                    if not in_quote:
                                        in_quote = True
                                        quote_char = char
                                    elif char == quote_char:
                                        in_quote = False
                                        if current_choice:
                                            choices_list.append(''.join(current_choice))
                                            current_choice = []
                                        quote_char = None
                                    else:
                                        current_choice.append(char)
                                elif in_quote:
                                    current_choice.append(char)
                                
                                i += 1
                            
                            if current_choice:
                                choices_list.append(''.join(current_choice))
                            
                            choices_list = [c.strip() for c in choices_list if c.strip()]
                        
                    elif isinstance(choices_str, (list, tuple, np.ndarray)):
                        choices_list = [str(c).strip() for c in choices_str]
                    else:
                        choices_list = []
                except Exception as e:
                    self.logger.warning(f"Could not parse HellaSwag choices: {repr(choices_str)}, error: {e}")
                    choices_list = []
                
                # Convert answer index to letter (0->A, 1->B, 2->C, 3->D)
                try:
                    answer_idx_int = int(answer_idx) if not isinstance(answer_idx, int) else answer_idx
                    if 0 <= answer_idx_int < len(choices_list):
                        answerKey = chr(ord('A') + answer_idx_int)
                        answer_text = choices_list[answer_idx_int]
                    else:
                        self.logger.warning(
                            f"HellaSwag answer index {answer_idx_int} out of range for {len(choices_list)} choices. "
                            f"Row index: {row.name}, Question: {row['question'][:80]}..., "
                            f"Choices: {choices_list}, Raw: {repr(choices_str)}"
                        )
                        answerKey = ''
                        answer_text = ''
                except Exception as e:
                    self.logger.warning(f"Could not convert HellaSwag answer index: {answer_idx}, error: {e}")
                    answerKey = ''
                    answer_text = ''
                
                # Format choices as dict with 'label' and 'text'
                labels = [chr(ord('A') + i) for i in range(len(choices_list))]
                choices_dict = {
                    'label': labels,
                    'text': choices_list
                }
                
                return pd.Series({
                    'context': row['context'],
                    'question': '',  # No question for HellaSwag
                    'choices': choices_dict,
                    'answerKey': answerKey,
                    'answer': answer_text
                })
            
            df = df.apply(parse_hellaswag_row, axis=1)
            
            # Filter out rows with invalid answer keys (empty answerKey means index was out of range)
            initial_count = len(df)
            df = df[df['answerKey'] != '']
            invalid_count = initial_count - len(df)
            if invalid_count > 0:
                self.logger.warning(f"Filtered out {invalid_count} HellaSwag rows with answer indices out of range for choices")
            
            return df[['context', 'question', 'choices', 'answerKey', 'answer']]
        
        else:
            raise ValueError(f"Unknown dataset_format: {dataset_format}. Choose from: 'mcq', 'non_mcq', 'mmlu', 'hellaswag'")
    
    def format_right_answer(self, answer_key: str, answer_text: str) -> str:
        """
        Format right answer based on MCQ mode.
        
        Args:
            answer_key: The answer key (e.g., 'A', 'B', 'C', 'D')
            answer_text: The answer text
            
        Returns:
            Formatted answer string:
            - MCQ mode: "A. answer_text"
            - Non-MCQ mode: "answer_text"
        """
        if self.use_mcq:
            return f"{answer_key}. {answer_text}"
        else:
            return answer_text
    
    def make_prompt(self, context: str, question: str, 
                   choices: dict = None) -> list:
        """
        Create QA prompt messages.
        
        Args:
            context: Context/passage
            question: Question text
            choices: Dict with 'label' and 'text' keys (or None)
            
        Returns:
            List of message dicts for chat template
        """
        return make_qa_prompt(context, question, choices, use_mcq=self.use_mcq, dataset_format=self.dataset_format, model_type=self.model_type)
    
    def evaluate_answer(self, right_answer: str, baseline_answer: str,
                       steered_answer: str, max_workers: int = 60) -> Tuple[int, int]:
        """
        Evaluate hallucination scores for baseline and steered answers.
        
        Uses appropriate evaluator based on MCQ mode:
        - MCQ: batch_judge_answers_mcq (checks if model picked correct letter)
        - Non-MCQ: batch_judge_answers (checks factual correctness)
        
        Args:
            right_answer: Correct answer (formatted by format_right_answer)
            baseline_answer: Model's baseline answer
            steered_answer: Model's steered answer
            max_workers: Max threads for evaluation
            
        Returns:
            Tuple of (baseline_score, steered_score) where score is 0 or 1
        """
        
        evaluation_pairs = [
            (right_answer, baseline_answer, {'type': 'baseline'}),
            (right_answer, steered_answer, {'type': 'steered'}),
        ]
        
        if self.use_mcq:
            self.logger.debug("Using MCQ evaluator (batch_judge_answers_mcq)")
            results = batch_judge_answers_mcq(evaluation_pairs, max_workers=max_workers)
        else:
            self.logger.debug("Using standard evaluator (batch_judge_answers)")
            results = batch_judge_answers(evaluation_pairs, max_workers=max_workers)
        
        scores = {}
        for (_, _, metadata), (score, _) in zip(evaluation_pairs, results):
            scores[metadata['type']] = score
        
        return scores['baseline'], scores['steered']
    
    def batch_evaluate_answers(self, evaluation_pairs: list, max_workers: int = 60) -> list:
        """
        Evaluate a batch of (ground_truth, candidate, metadata) tuples.
        
        Uses appropriate evaluator based on MCQ mode:
        - MCQ: batch_judge_answers_mcq (checks if model picked correct letter)
        - Non-MCQ: batch_judge_answers (checks factual correctness)
        
        Args:
            evaluation_pairs: List of (gt_answer, candidate_answer, metadata_dict) tuples
            max_workers: Max threads for parallel evaluation (default: 60)
            
        Returns:
            List of (hallucination_score, metadata) tuples in same order as input
            - hallucination_score: 0 = correct, 1 = hallucinated/incorrect
        """
        
        if self.use_mcq:
            self.logger.info(f"Using MCQ evaluator (batch_judge_answers_mcq)")
            results = batch_judge_answers_mcq(evaluation_pairs, max_workers=max_workers)
        else:
            self.logger.info(f"Using standard evaluator (batch_judge_answers)")
            results = batch_judge_answers(evaluation_pairs, max_workers=max_workers)
        
        return results
    
    def build_result_dict(self, idx: int, prompts_map: Dict, 
                         baseline_texts: Dict, steered_texts: Dict,
                         baseline_outputs: Dict, steered_outputs: Dict,
                         baseline_eval: Dict, steered_eval: Dict) -> Dict[str, Any]:
        """
        Build standardized result dictionary for a single sample.
        
        Args:
            idx: Sample index
            prompts_map: Dict mapping idx to prompt info
            baseline_texts: Dict of baseline extracted answers
            steered_texts: Dict of steered extracted answers
            baseline_outputs: Dict of baseline full outputs
            steered_outputs: Dict of steered full outputs
            baseline_eval: Dict of baseline evaluation scores
            steered_eval: Dict of steered evaluation scores
            
        Returns:
            Dict with all comparison data for this sample
        """
        return {
            'sample_idx': idx,
            'qa_prompt': prompts_map[idx]['qa_prompt'],
            'question': prompts_map[idx]['question'],
            'context': prompts_map[idx]['context'],
            'choices': prompts_map[idx].get('choices'),
            'right_answer': prompts_map[idx]['right_answer'],
            'baseline_answer': baseline_texts[idx],
            'steered_answer': steered_texts[idx],
            'baseline_output': baseline_outputs[idx],
            'steered_output': steered_outputs[idx],
            'baseline_hallucination': baseline_eval.get(idx, 2),
            'steered_hallucination': steered_eval.get(idx, 2),
            'hallucination_reduced': baseline_eval.get(idx, 2) > steered_eval.get(idx, 2),
            'steering_effect': baseline_eval.get(idx, 2) - steered_eval.get(idx, 2),
        }
