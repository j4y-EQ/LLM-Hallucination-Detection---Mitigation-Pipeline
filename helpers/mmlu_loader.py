"""
MMLU dataset loader with deterministic sampling and caching.

Loads MMLU (Massive Multitask Language Understanding) dataset from HuggingFace,
uniformly samples 1000 examples across 57 subjects, and caches results for
consistent future loads.

Usage:
    from helpers.mmlu_loader import load_mmlu_dataset
    df = load_mmlu_dataset(split='test', force_reload=False)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datasets import load_dataset


# MMLU configuration
MMLU_CACHE_DIR = "./data/mmlu_cache"
MMLU_CACHE_FILE = os.path.join(MMLU_CACHE_DIR, "mmlu_1000_stratified.pkl")
MMLU_METADATA_FILE = os.path.join(MMLU_CACHE_DIR, "mmlu_metadata.json")

# All 57 MMLU subjects
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

N_SUBJECTS = len(MMLU_SUBJECTS)
SAMPLES_PER_SUBJECT = 1000 // N_SUBJECTS  # ~17 per subject
TOTAL_SAMPLES = SAMPLES_PER_SUBJECT * N_SUBJECTS  # 952 samples (57 * 16 + 16)


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(MMLU_CACHE_DIR, exist_ok=True)


def _convert_mmlu_to_csv_format(df_mmlu: pd.DataFrame, subject: str) -> pd.DataFrame:
    """
    Convert MMLU format to expected CSV format.
    
    Args:
        df_mmlu: DataFrame loaded from MMLU dataset
        subject: Subject name (for reference)
        
    Returns:
        DataFrame with columns: context, question, choices, answerKey, answer
    """
    df = df_mmlu.copy()
    
    # MMLU has: question, choices (list of 4), answer (letter A-D)
    # Expected format: context, question, choices (dict), answerKey, answer (text)
    
    # Add empty context (MMLU has no context; model uses pretraining only)
    df['context'] = ''
    
    # Convert choices list to dict format
    df['choices'] = df.apply(
        lambda row: {
            'label': ['A', 'B', 'C', 'D'],
            'text': row['choices']
        },
        axis=1
    )
    
    # answerKey is the letter (A, B, C, D)
    df['answerKey'] = df['answer']
    
    # answer is the text of the correct answer
    df['answer'] = df.apply(
        lambda row: row['choices'][ord(row['answer']) - ord('A')],
        axis=1
    )
    
    # Keep only needed columns
    df = df[['context', 'question', 'choices', 'answerKey', 'answer']]
    
    # Add subject column for tracking
    df['subject'] = subject
    
    return df


def _load_and_sample_mmlu(split: str = 'test', random_seed: int = 42) -> pd.DataFrame:
    """
    Load MMLU dataset from HuggingFace and uniformly sample across subjects.
    
    Args:
        split: Dataset split ('test', 'dev', or 'auxiliary_train')
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with TOTAL_SAMPLES samples uniformly distributed across subjects
    """
    np.random.seed(random_seed)
    
    all_dfs = []
    
    for subject in MMLU_SUBJECTS:
        # Load this subject's data
        try:
            ds = load_dataset('cais/mmlu', subject, split=split, trust_remote_code=True)
            df = ds.to_pandas()
            
            # Randomly sample up to SAMPLES_PER_SUBJECT from this subject
            if len(df) > SAMPLES_PER_SUBJECT:
                sampled_indices = np.random.choice(len(df), SAMPLES_PER_SUBJECT, replace=False)
                df = df.iloc[sampled_indices].reset_index(drop=True)
            
            # Convert to expected format
            df = _convert_mmlu_to_csv_format(df, subject)
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Warning: Failed to load subject '{subject}': {e}")
            continue
    
    # Concatenate all subjects
    if not all_dfs:
        raise RuntimeError("Failed to load any MMLU subjects")
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    # Shuffle the combined dataset
    df_combined = df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    return df_combined


def _save_cache(df: pd.DataFrame, split: str):
    """Save sampled MMLU dataset and metadata to cache directory.
    
    Creates pickle cache of DataFrame and JSON metadata file with statistics
    about the cached dataset (split, sample counts, subject distribution).
    
    Args:
        df (pd.DataFrame): Sampled MMLU dataset to cache.
        split (str): Dataset split name ('test', 'dev', 'auxiliary_train').
        
    Returns:
        None (writes files to MMLU_CACHE_DIR)
        
    Notes:
        - Saves DataFrame as MMLU_CACHE_FILE (pickle format)
        - Saves metadata as MMLU_METADATA_FILE (JSON format)
        - Metadata includes: split, n_samples, n_subjects, subjects list, samples_per_subject
        - Creates cache directory if it doesn't exist
    """
    _ensure_cache_dir()
    
    # Save dataset
    with open(MMLU_CACHE_FILE, 'wb') as f:
        pickle.dump(df, f)
    
    # Save metadata
    metadata = {
        'split': split,
        'n_samples': len(df),
        'n_subjects': len(df['subject'].unique()),
        'subjects': df['subject'].unique().tolist(),
        'samples_per_subject': dict(df['subject'].value_counts().to_dict()),
    }
    
    with open(MMLU_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def _load_cache() -> Optional[pd.DataFrame]:
    """Load cached MMLU dataset if available.
    
    Attempts to load pickled DataFrame from cache directory. Returns None
    if cache doesn't exist or loading fails.
    
    Returns:
        pd.DataFrame | None: Cached dataset if available and valid, None otherwise.
        
    Notes:
        - Returns None if MMLU_CACHE_FILE doesn't exist
        - Returns None and warns if pickle loading fails (corrupted cache)
        - Silently returns None (not an error condition, just cache miss)
    """
    if os.path.exists(MMLU_CACHE_FILE):
        try:
            with open(MMLU_CACHE_FILE, 'rb') as f:
                df = pickle.load(f)
            return df
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    return None


def load_mmlu_dataset(split: str = 'test', force_reload: bool = False) -> pd.DataFrame:
    """
    Load MMLU dataset with caching.
    
    Loads MMLU from HuggingFace on first call and caches the result.
    Subsequent calls load from cache for consistency unless force_reload=True.
    
    Args:
        split: Dataset split ('test', 'dev', or 'auxiliary_train'). Default: 'test'
        force_reload: Force re-download and re-sample from HuggingFace. Default: False
        
    Returns:
        DataFrame with ~1000 stratified MMLU samples across 57 subjects
        Columns: context (empty), question, choices (dict), answerKey, answer, subject
        
    Example:
        >>> df = load_mmlu_dataset(split='test')
        >>> print(f"Loaded {len(df)} samples from {df['subject'].nunique()} subjects")
        Loaded 952 samples from 57 subjects
    """
    # Try to load from cache first (unless force_reload)
    if not force_reload:
        df = _load_cache()
        if df is not None:
            print(f"✓ Loaded MMLU dataset from cache: {len(df)} samples")
            return df
    
    print(f"Loading MMLU dataset from HuggingFace (split='{split}')...")
    
    # Load and sample from HuggingFace
    df = _load_and_sample_mmlu(split=split, random_seed=42)
    
    # Save to cache
    _save_cache(df, split)
    
    print(f"✓ Loaded and cached MMLU dataset: {len(df)} samples across {df['subject'].nunique()} subjects")
    print(f"  Cache location: {MMLU_CACHE_FILE}")
    
    return df


def clear_mmlu_cache():
    """Delete cached MMLU dataset and metadata files.
    
    Removes both the pickled dataset cache and JSON metadata file from
    the cache directory. Use this to force fresh download and re-sampling.
    
    Returns:
        None (deletes files and prints confirmation)
        
    Notes:
        - Deletes MMLU_CACHE_FILE (pickled DataFrame)
        - Deletes MMLU_METADATA_FILE (JSON metadata)
        - Prints confirmation messages for each deleted file
        - Safe to call even if cache doesn't exist (no error)
    """
    if os.path.exists(MMLU_CACHE_FILE):
        os.remove(MMLU_CACHE_FILE)
        print(f"Deleted cache file: {MMLU_CACHE_FILE}")
    
    if os.path.exists(MMLU_METADATA_FILE):
        os.remove(MMLU_METADATA_FILE)
        print(f"Deleted metadata file: {MMLU_METADATA_FILE}")


if __name__ == "__main__":
    # Test the loader
    print("Testing MMLU loader...")
    df = load_mmlu_dataset(split='test', force_reload=False)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst row:\n{df.iloc[0]}")
    print(f"\nSubjects distribution:\n{df['subject'].value_counts()}")
