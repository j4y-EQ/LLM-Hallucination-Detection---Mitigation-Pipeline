"""
Centralized dataset format definitions for steering pipeline.

This module defines all valid dataset formats used across the pipeline in ONE place.
Any code that needs to validate or use dataset formats should import from here.

This prevents format inconsistencies across different scripts and modules.
"""

# ================================================================
# DATASET FORMATS - SINGLE SOURCE OF TRUTH
# ================================================================

# Valid dataset formats accepted by the steering pipeline
VALID_DATASET_FORMATS = ['mcq', 'non_mcq', 'mmlu', 'hellaswag']

# Formats that use MCQ (Multiple Choice Question) mode
MCQ_FORMATS = ['mcq', 'mmlu', 'hellaswag']

# Format descriptions
FORMAT_DESCRIPTIONS = {
    'mcq': 'MCQ format with context, question, choices (with labels and text), and answerKey',
    'non_mcq': 'Non-MCQ format with context, question, and answer text (no multiple choice options)',
    'mmlu': 'MMLU format with question and choices (as array), answer as choice index, no context',
    'hellaswag': 'HellaSwag format with context (incomplete sentence), choices (as endings), and answer (as index 0-3)',
}


def validate_dataset_format(dataset_format, raise_error=True):
    """
    Validate that the provided dataset format is valid.
    
    Args:
        dataset_format (str): The dataset format to validate
        raise_error (bool): If True, raise ValueError on invalid format. 
                           If False, return boolean result.
    
    Returns:
        bool: True if valid, False otherwise (only when raise_error=False)
        
    Raises:
        ValueError: If dataset_format is invalid and raise_error=True
    """
    if dataset_format not in VALID_DATASET_FORMATS:
        error_msg = (
            f"Invalid dataset_format: '{dataset_format}'. "
            f"Must be one of: {VALID_DATASET_FORMATS}"
        )
        if raise_error:
            raise ValueError(error_msg)
        else:
            return False
    return True


def is_mcq_format(dataset_format):
    """Check if a dataset format uses MCQ mode."""
    validate_dataset_format(dataset_format)
    return dataset_format in MCQ_FORMATS
