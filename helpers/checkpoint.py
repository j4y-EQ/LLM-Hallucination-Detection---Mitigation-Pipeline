"""Checkpoint Management System

Provides comprehensive checkpoint operations for the hallucination detection pipeline,
including atomic file operations, corruption recovery, progress tracking, and buffered
saving strategies.

CORE FEATURES:
    - Atomic file operations with automatic corruption recovery
    - Comprehensive progress tracking with resume capability
    - Buffered saving for efficiency (results + activations)
    - Automatic verification of checkpoint completeness
    - Two-phase recovery (consolidated files, then individual batches)

CHECKPOINT TYPES:
    1. Batch checkpoints: batch_N_results.pkl (per-batch results)
    2. Consolidated results: consolidated_results_X_Y.pkl (merged batches)
    3. Activation files: batch_N_activations.h5 or consolidated_activations_X_Y.h5
    4. Progress state: progress.json (completed row indices, timestamps)

RECOVERY STRATEGY:
    - Fast path: Load consolidated_results_*.pkl first
    - Slow path: Load individual batch_*_results.pkl for non-consolidated data
    - Verification: Only consider samples complete if BOTH activations AND results exist
    - Automatic corruption handling: Skip corrupted files, log warnings

BUFFERING:
    - Accumulates multiple batches before writing to disk
    - Reduces filesystem operations for efficiency
    - Merges activation dictionaries preserving hook/scheme hierarchy
    - Configurable buffer size via RESULTS_BUFFER_SIZE

USAGE:
    manager = CheckpointManager(
        atomic_operation_manager=atomic_ops,
        OUT_DIR='./output',
        CHUNK_SIZE=5000,
        BATCH_SIZE=128,
        RESULTS_BUFFER_SIZE=5
    )
    
    # Save batch
    manager.save_batch_checkpoint(batch_results, batch_idx)
    
    # Recover on resume
    manager.recover_all_results()
    remaining_samples, start_batch = manager.filter_remaining_samples(...)

Classes:
    CheckpointManager: Main checkpoint management class

Dependencies:
    - AtomicOperationManager: For atomic file operations
    - consolidated_logger: For logging checkpoint operations
"""


# ================================================================
# CHECKPOINT MANAGER CLASS
# Industry-standard checkpoint management system
# ================================================================

# Required imports
import os
import glob
import time
import json
import gc
import h5py
import numpy as np
import torch
from tqdm.auto import tqdm  # Keep tqdm import
from typing import Dict, List, Any

# Import the consolidated logger
from logger import consolidated_logger

class CheckpointManager:
    """
    A comprehensive manager for checkpoint operations with configurable debug settings.

    This class provides all checkpoint-related functionality including scanning,
    saving, recovery, and progress tracking operations.
    """

    def __init__(self, atomic_operation_manager, debug_verbose: bool = False, **config):
        """
        Initialize the CheckpointManager.

        Args:
            atomic_operation_manager: The atomic operations manager instance
            debug_verbose (bool): Enable detailed logging for debugging. Defaults to False.
            **config: Configuration parameters (OUT_DIR, RUN_TIMESTAMP, etc.)
        """
        self.atomic_operation_manager = atomic_operation_manager

        # Use consolidated logger for checkpoint operations
        self.logger = consolidated_logger

        # Configuration parameters
        self.OUT_DIR = config.get('OUT_DIR', './output')
        self.RUN_TIMESTAMP = config.get('RUN_TIMESTAMP')
        self.CHUNK_SIZE = config.get('CHUNK_SIZE')
        self.BATCH_SIZE = config.get('BATCH_SIZE')
        self.RESULTS_BUFFER_SIZE = config.get('RESULTS_BUFFER_SIZE')
        self.DEVICE_ID = config.get('DEVICE_ID', 0)

        # Global state variables
        self.activation_storage = {}
        self.current_batch_info = {}
        self.captured_activations = set()
        self.global_error_log = []
        self.skipped_data_points = []
        self.buffer_flush_counter = 0
        self.results_buffer = []
        self.activation_buffer = []
        self.generation_results = []

        # Function dependencies (will be set by caller)
        self.batch_judge_answers = None
        self.tqdm = tqdm  # Keep tqdm as an attribute for progress bars

    def scan_checkpoint_files(self):
        """Scans all checkpoint files ONCE and returns sets of completed sample indices."""
        samples_with_activations = set()
        samples_with_results = set()
        samples_from_progress = set()

        # FAST PATH: Scan consolidated files first 
        consolidated_activation_files = glob.glob(os.path.join(self.OUT_DIR, "consolidated_activations_*.h5"))
        consolidated_result_files = glob.glob(os.path.join(self.OUT_DIR, "consolidated_results_*.pkl"))

        if consolidated_activation_files or consolidated_result_files:
            self.logger.debug(f"Found {len(consolidated_activation_files)} consolidated activation files and {len(consolidated_result_files)} consolidated result files")

        # Scan consolidated activation files
        for f in tqdm(consolidated_activation_files, desc="Scanning consolidated activation files"):
            try:
                with h5py.File(f, 'r') as h5f:
                    for hook in h5f.keys():
                        for scheme in h5f[hook].keys():
                            if 'row_indices' in h5f[hook][scheme]:
                                samples_with_activations.update(h5f[hook][scheme]['row_indices'][:])
            except Exception as e:
                self.logger.error(f"Could not read consolidated activation file {f}: {e}")
                continue

        # Scan consolidated result files
        for f in tqdm(consolidated_result_files, desc="Scanning consolidated result files"):
            try:
                batch_results = self.atomic_operation_manager.atomic_load(f)
                if batch_results is not None:
                    for result in batch_results:
                        if 'row_idx' in result:
                            samples_with_results.add(result['row_idx'])
            except Exception as e:
                self.logger.error(f"Could not read consolidated result file {f}: {e}")
                continue

        # SLOW PATH: Scan individual files for non-consolidated data
        activation_files = glob.glob(os.path.join(self.OUT_DIR, "*_batch_*_activations_*.h5"))
        for f in tqdm(activation_files, desc="Scanning individual activation files"):
            try:
                with h5py.File(f, 'r') as h5f:
                    for hook in h5f.keys():
                        for scheme in h5f[hook].keys():
                            if 'row_indices' in h5f[hook][scheme]:
                                samples_with_activations.update(h5f[hook][scheme]['row_indices'][:])
            except Exception as e:
                self.logger.error(f"Could not read activation file {f}: {e}")
                continue

        # Scan individual result files
        result_files = glob.glob(os.path.join(self.OUT_DIR, "batch_*_results.pkl"))
        for f in tqdm(result_files, desc="Scanning individual result files"):
            try:
                batch_results = self.atomic_operation_manager.atomic_load(f)
                if batch_results is not None:
                    for result in batch_results:
                        if 'row_idx' in result:
                            samples_with_results.add(result['row_idx'])
            except Exception as e:
                self.logger.error(f"Could not read result file {f}: {e}")
                continue

        # Scan progress.json
        progress_file = os.path.join(self.OUT_DIR, 'progress.json')
        if os.path.exists(progress_file):
            try:
                def load_json(filepath):
                    with open(filepath, 'r') as f:
                        return json.load(f)

                progress_data = self.atomic_operation_manager.atomic_load(progress_file, load_func=load_json)
                if progress_data and 'processed_row_indices' in progress_data:
                    samples_from_progress.update(progress_data['processed_row_indices'])
                    self.logger.debug(f"Loaded {len(progress_data['processed_row_indices'])} completed samples from progress.json")
            except Exception as e:
                self.logger.error(f"Could not read progress.json: {e}")

        return samples_with_activations, samples_with_results, samples_from_progress

    def get_completed_samples(self):
        """Scan activation files, batch result files, and progress.json to find completed samples"""
        completed = set()

        # Method 1: Check activation files (for samples that have activations saved)
        activation_files = glob.glob(os.path.join(self.OUT_DIR, "*_batch_*_activations_*.h5"))
        for f in tqdm(activation_files, desc="Scanning activation files"):
            try:
                with h5py.File(f, 'r') as h5f:
                    for hook in h5f.keys():
                        for scheme in h5f[hook].keys():
                            if 'row_indices' in h5f[hook][scheme]:
                                completed.update(h5f[hook][scheme]['row_indices'][:])
            except Exception as e:
                self.logger.error(f"Could not read activation file {f}: {e}")
                continue

        # Method 2: Check batch result files (for samples that have results saved)
        result_files = glob.glob(os.path.join(self.OUT_DIR, "batch_*_results.pkl"))
        for f in tqdm(result_files, desc="Scanning result files"):
            try:
                batch_results = self.atomic_operation_manager.atomic_load(f)
                if batch_results is not None:
                    for result in batch_results:
                        if 'row_idx' in result:
                            completed.add(result['row_idx'])
            except Exception as e:
                self.logger.error(f"Could not read result file {f}: {e}")
                continue

        # Method 3: Check progress.json (for samples that were tracked as completed)
        progress_file = os.path.join(self.OUT_DIR, 'progress.json')
        if os.path.exists(progress_file):
            try:
                def load_json(filepath):
                    with open(filepath, 'r') as f:
                        return json.load(f)

                progress_data = self.atomic_operation_manager.atomic_load(progress_file, load_func=load_json)
                if progress_data and 'processed_row_indices' in progress_data:
                    # Industry-standard integrity validation
                    if 'integrity_checksum' in progress_data:
                        expected_checksum = hash(tuple(sorted(progress_data['processed_row_indices'])))
                        if progress_data['integrity_checksum'] != expected_checksum:
                            self.logger.warning(f"WARNING: Progress.json integrity check failed - data may be corrupted")
                            self.logger.debug(f"Expected checksum: {expected_checksum}, Got: {progress_data['integrity_checksum']}")
                        else:
                            completed.update(progress_data['processed_row_indices'])
                            self.logger.debug(f"Loaded {len(progress_data['processed_row_indices'])} completed samples from progress.json")
                    else:
                        # Backward compatibility for old progress files
                        completed.update(progress_data['processed_row_indices'])
                        self.logger.debug(f"Loaded {len(progress_data['processed_row_indices'])} completed samples from progress.json (legacy format)")
            except Exception as e:
                self.logger.error(f"Could not read progress.json: {e}")

        return completed


    def save_batch_checkpoint(self, batch_results, batch_idx):
        """Atomically save batch results with immediate evaluation.
        
        Evaluates generated answers against ground truth using batch_judge_answers,
        then saves results atomically with corruption protection. Falls back to
        timestamped filename if primary save fails.
        
        Args:
            batch_results (list of dict): Generated results for this batch, each with:
                - 'right_answer': str, ground truth answer
                - 'gpt_answer_trim': str, generated answer
                - Additional generation metadata
            batch_idx (int): Batch number for filename.
            
        Returns:
            None (updates batch_results in-place with evaluation results)
            
        Notes:
            - Evaluates batch immediately before saving (inline evaluation)
            - Adds 'is_hallucination', 'is_correct', 'evaluator_response' to each result
            - Uses atomic_operation_manager for corruption-safe saves
            - Falls back to timestamped filename if standard filename conflicts
        """
        results_file = os.path.join(self.OUT_DIR, f"batch_{batch_idx}_results.pkl")

        # If evaluation is requested, evaluate this batch immediately
        # Use logger.progress() for logging, keep tqdm for progress bar
        self.logger.progress(f"Evaluating batch {batch_idx} ({len(batch_results)} results)...")
        evaluation_pairs = [(r["right_answer"], r["gpt_answer_trim"], r) for r in batch_results]
        evaluation_results = self.batch_judge_answers(evaluation_pairs, max_workers=10)  # Lower workers for batch-level

        # Add evaluation results to batch results
        for (eval_out, _), gen_result in zip(evaluation_results, batch_results):
            # eval_out is hallucination label: 1=hallucination, 0=non, 2=failure
            gen_result["is_hallucination"] = eval_out
            if eval_out in (0, 1):
                gen_result["is_correct"] = 1 - eval_out
            else:
                gen_result["is_correct"] = 2
            gen_result["evaluator_response"] = eval_out

        self.logger.progress(f"Batch {batch_idx} evaluation completed")

        success = self.atomic_operation_manager.atomic_save(batch_results, results_file)
        if not success:
            # Fallback: try timestamped filename
            alt_file = os.path.join(self.OUT_DIR, f"batch_{batch_idx}_results_{int(time.time())}.pkl")
            success = self.atomic_operation_manager.atomic_save(batch_results, alt_file)
            if success:
                self.logger.debug(f"Saved to alternative file: {alt_file}")

    def recover_all_results(self):
        """Recover all saved batch results with automatic corruption handling.
        
        Uses two-phase recovery strategy:
        1. Fast path: Load consolidated_results_*.pkl files first (pre-merged batches)
        2. Slow path: Load individual batch_*_results.pkl files for non-consolidated data
        
        Automatically skips corrupted files and reports corruption count.
        
        Returns:
            None (populates self.generation_results with recovered data)
            
        Notes:
            - Sets self.generation_results to combined list of all recovered results
            - Uses atomic_load for corruption-safe loading with backup fallback
            - Logs warning if any corrupted files are skipped
            - Consolidated files are processed first for efficiency
        """
        all_results = []
        corrupted_count = 0

        # FAST PATH: Load consolidated files first
        consolidated_result_files = glob.glob(os.path.join(self.OUT_DIR, "consolidated_results_*.pkl"))
        for result_file in sorted(consolidated_result_files):
            batch_results = self.atomic_operation_manager.atomic_load(result_file)
            if batch_results is not None:
                all_results.extend(batch_results)
            else:
                corrupted_count += 1

        # SLOW PATH: Load individual files for non-consolidated data
        result_files = glob.glob(os.path.join(self.OUT_DIR, "batch_*_results.pkl"))
        for result_file in sorted(result_files):
            batch_results = self.atomic_operation_manager.atomic_load(result_file)
            if batch_results is not None:
                all_results.extend(batch_results)
            else:
                corrupted_count += 1

        if corrupted_count > 0:
            self.logger.warning(f"Warning: {corrupted_count} corrupted result files skipped")

        self.generation_results = all_results


    def save_progress_state(self, completed_row_indices, batch_idx):
        """Atomically save progress checkpoint with industry-standard state tracking.
        
        Maintains persistent progress.json file with completed sample indices,
        allowing recovery after crashes. Uses incremental updates to preserve
        existing progress while adding new completions.
        
        Args:
            completed_row_indices (list of int): Row indices completed in this save.
            batch_idx (int): Current batch index.
            
        Returns:
            None (saves checkpoint to progress.json)
            
        Notes:
            - Merges new completions with existing processed_row_indices
            - Includes metadata: timestamp, total_completed, status, integrity checksum
            - Uses atomic save to prevent corruption during write
            - Version 1.0 format for future compatibility
        """
        checkpoint_file = os.path.join(self.OUT_DIR, 'progress.json')

        # Load existing progress
        def load_json(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)

        existing_data = self.atomic_operation_manager.atomic_load(checkpoint_file, load_func=load_json) or {}

        # Update with new completions
        existing_processed = set(existing_data.get('processed_row_indices', []))
        existing_processed.update(completed_row_indices)

        # Industry-standard checkpoint data structure
        checkpoint_data = {
            'processed_row_indices': list(existing_processed),
            'last_batch_idx': batch_idx,
            'run_timestamp': self.RUN_TIMESTAMP,
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_completed': len(existing_processed),
            'checkpoint_version': '1.0',  # Version for future compatibility
            'status': 'active' if len(existing_processed) < self.CHUNK_SIZE else 'completed',
            'integrity_checksum': hash(tuple(sorted(existing_processed)))  # Simple integrity check
        }

        def save_json(data, filepath):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        self.atomic_operation_manager.atomic_save(checkpoint_data, checkpoint_file, save_func=save_json)

    def filter_remaining_samples(self, sample_list, samples_with_activations, samples_with_results, samples_from_progress):
        """Remove already processed samples and calculate resume point.
        
        Implements robust verification that a sample is only considered complete if
        BOTH activations and results files exist. Identifies and warns about partially
        complete samples (e.g., has results but missing activations).
        
        Args:
            sample_list (list): All samples for this chunk.
            samples_with_activations (set of int): Row indices with saved activation files.
            samples_with_results (set of int): Row indices with saved result files.
            samples_from_progress (set of int): Row indices from progress.json.
            
        Returns:
            tuple: (remaining_samples, completed_batches) where:
                - remaining_samples: list of samples still needing processing
                - completed_batches: number of fully completed batches (for resume)
                
        Notes:
            - Verifies completeness by requiring both activation AND result files
            - Reports incomplete samples that will be re-processed
            - Logs detailed statistics about verified vs incomplete samples
            - Calculates completed_batches as verified_completed // BATCH_SIZE
        """

        # Combine all discovered samples to get a broad list of what might be complete
        # Set union: Aggregate samples from 3 sources (activation files, result files, progress state)
        # This casts a wide net - a sample might appear in only one source but still be included
        potentially_completed = samples_with_activations.union(samples_with_results).union(samples_from_progress)

        if not potentially_completed:
            self.logger.info("STARTING: Fresh run detected. No previously completed samples found.")
            return sample_list, 0

        self.logger.info(f"RESUMING: Found {len(potentially_completed)} potentially completed samples across all checkpoint files.")

        # Filter this list to only samples relevant to the current chunk
        # Set intersection: Keep only samples that are both "potentially complete" AND in current chunk
        # This prevents resuming with samples from a different chunk/dataset
        chunk_sample_indices = {s.Index for s in sample_list}
        completed_in_chunk = potentially_completed.intersection(chunk_sample_indices)

        if not completed_in_chunk:
            self.logger.info("No completed samples found for this specific chunk.")
            return sample_list, 0

        self.logger.info(f"Verifying {len(completed_in_chunk)} samples relevant to this chunk...")

        # --- VERIFICATION ---
        # A sample is only truly complete if it has BOTH activations and results saved.
        # Double intersection: sample must be in completed_in_chunk AND samples_with_activations AND samples_with_results
        # This strict verification prevents data corruption from partial save failures
        # We use the pre-scanned sets passed into this function.
        verified_completed = completed_in_chunk.intersection(samples_with_activations).intersection(samples_with_results)

        # Report on samples that are incomplete (e.g., have results but missing activations)
        # Set difference: completed_in_chunk - verified_completed = samples with partial save
        # These will be re-processed to ensure data integrity
        incomplete_samples = completed_in_chunk - verified_completed
        if incomplete_samples:
            self.logger.warning(f"WARNING: Found {len(incomplete_samples)} partially complete samples (e.g., missing activations or results). These will be re-processed.")
        
            for sample_idx in sorted(list(incomplete_samples))[:10]: # show first 10
                has_activation = sample_idx in samples_with_activations
                has_result = sample_idx in samples_with_results
                self.logger.warning(f"  - Sample {sample_idx} is incomplete - Activation file found: {has_activation}, Result file found: {has_result}")
            if len(incomplete_samples) > 10:
                self.logger.warning(f"  ... and {len(incomplete_samples) - 10} more incomplete samples.")

        # --- FINAL CALCULATION ---
        # Filter out verified_completed samples from the full sample list
        remaining = [s for s in sample_list if s.Index not in verified_completed]
        # Calculate how many complete batches we can skip on resume
        # Integer division: 153 verified samples // 50 batch_size = 3 complete batches
        # Resume will start at batch index 3 (skip batches 0, 1, 2)
        completed_batches = len(verified_completed) // self.BATCH_SIZE

        self.logger.info(f"VERIFIED COMPLETED: {len(verified_completed)} samples have both activations and results.")
        self.logger.info(f"REMAINING: {len(remaining)} samples will be processed in this run.")
        self.logger.info(f"RESUMING FROM: Batch index {completed_batches}.")

        return remaining, completed_batches

    def print_recovery_summary(self, completed_samples, sample_list):
        """Print detailed recovery summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("RECOVERY SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total samples in chunk: {len(sample_list)}")
        self.logger.info(f"Completed samples detected: {len(completed_samples)}")

        if completed_samples:
            chunk_sample_indices = {s.Index for s in sample_list}
            completed_in_chunk = completed_samples.intersection(chunk_sample_indices)
            self.logger.info(f"Completed samples in this chunk: {len(completed_in_chunk)}")
            self.logger.info(f"Remaining samples to process: {len(sample_list) - len(completed_in_chunk)}")

            if completed_in_chunk:
                self.logger.info(f"Completed sample indices: {sorted(completed_in_chunk)}")
        else:
            self.logger.info("No completed samples found - starting fresh run")

        self.logger.info("="*60)



    def process_and_save_batch_activations(self, activations_data: Dict, batch_results: List[Dict], model_name: str, batch_idx: int):
        """
        Process and save activations for current batch with atomic writes to prevent memory explosion

        Args:
            activations_data (Dict): Global activation_storage dict with structure:
                {hook_key: {scheme: [activation_data_list]}}
            batch_results (List[Dict]): Generation results for current batch only
            model_name (str): Model name for file naming (e.g., "gpt2-medium")
            batch_idx (int): Batch index for file naming (0, 1, 2, ...)

        Returns:
            str: Path to saved HDF5 file, or None if no data to save

        Side Effects:
            - Creates HDF5 file with compressed activation data using atomic writes
            - File structure: layer_L_hookname/scheme/activations, row_indices, token_positions
        """
        self.logger.debug(f"process_and_save_batch_activations called with batch_idx={batch_idx}")
        self.logger.debug(f"activations_data is None: {activations_data is None}")
        self.logger.debug(f"activations_data is empty: {not activations_data if activations_data else 'N/A'}")
        if activations_data:
            self.logger.debug(f"activations_data has {len(activations_data)} keys")

        # Skip if no activations were captured (shouldn't happen in normal operation)
        if not activations_data:
            self.logger.warning(f"WARNING: No activation data to save for batch {batch_idx}")
            return None

        # Validate activation data structure
        if not isinstance(activations_data, dict):
            self.logger.warning(f"ERROR: Invalid activation data type for batch {batch_idx}: {type(activations_data)}")
            return None

        # Check if activation data actually contains any data
        total_activations = 0
        for hook_key, scheme_data in activations_data.items():
            for scheme, activation_list in scheme_data.items():
                total_activations += len(activation_list)
        
        if total_activations == 0:
            self.logger.warning(f"WARNING: No activation data found in any hook/scheme for batch {batch_idx}")
            return None

        # Create unique filename with timestamp to avoid collisions
        timestamp = int(time.time())
        batch_h5_path = os.path.join(self.OUT_DIR, f"{model_name}_batch_{batch_idx}_activations_{timestamp}.h5")

        def save_hdf5_data(temp_filepath):
            """Internal function to write HDF5 data"""
            with h5py.File(temp_filepath, 'w') as h5f:
                debug_counter = 0  # Counter to limit debug output
                # Process each hook type (e.g., "layer_0_hook_resid_pre", "layer_5_attn.hook_pattern")
                for hook_key, scheme_data in activations_data.items():
                    # Only create group if this hook captured any data
                    if any(scheme_data.values()):
                        # Create group for this hook: "layer_0_hook_resid_pre"
                        layer_group = h5f.create_group(hook_key)

                        # Process each token scheme for this hook
                        for scheme, activation_list in scheme_data.items():
                            # Only create subgroup if we have activations for this scheme
                            if activation_list:  # activation_list is list of activation_data dicts
                                # Create subgroup for this scheme: "bos_token", "last_prompt_token", etc.
                                scheme_group = layer_group.create_group(scheme)

                                # Stack all activations for this scheme into numpy arrays
                                # activation_list = [{'activations': np.array, 'row_idx': int, ...}, ...]

                                # Only print debug info for first 3 hook/scheme combinations
                                if debug_counter < 3:
                                    self.logger.debug(f"DEBUG: Saving {len(activation_list)} samples for {hook_key}/{scheme}")
                                    debug_counter += 1

                                # Check for shape consistency before stacking
                                activation_arrays = [item['activations'] for item in activation_list]
                                shapes = [arr.shape for arr in activation_arrays]

                                if len(set(shapes)) > 1:
                                    # Handle inconsistent shapes - this can happen with attention patterns
                                    self.logger.warning(f"Shape mismatch in {hook_key}/{scheme}: {set(shapes)}")

                                    # Debug: Show pass types and tensor shapes for the mismatched activations
                                    for i, item in enumerate(activation_list):
                                        self.logger.debug(f"  Item {i}: shape={item['activations'].shape}, pass_type={item.get('pass_type', 'unknown')}, tensor_shape={item.get('tensor_shape', 'unknown')}")

                                    # For attention patterns/scores, we need to handle variable sequence lengths
                                    if any(pattern in hook_key for pattern in ["attn.hook_pattern", "attn.hook_attn_scores"]):
                                        # Find the maximum sequence length
                                        max_seq_len = max(shape[-1] for shape in shapes)

                                        # Pad all attention patterns to the same sequence length
                                        padded_arrays = []
                                        for arr in activation_arrays:
                                            if arr.shape[-1] < max_seq_len:
                                                # Pad the last dimension (key positions) with zeros
                                                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max_seq_len - arr.shape[-1])]
                                                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
                                                padded_arrays.append(padded_arr)
                                            else:
                                                padded_arrays.append(arr)

                                        activations_array = np.stack(padded_arrays)
                                    else:
                                        # For other hooks, skip if shapes don't match
                                        self.logger.warning(f"Skipping {hook_key}/{scheme} due to shape mismatch")
                                        continue
                                else:
                                    # All shapes match, proceed normally
                                    activations_array = np.stack(activation_arrays)  # [n_samples, activation_dim]

                                row_indices = np.array([item['row_idx'] for item in activation_list], dtype=np.int32)  # [n_samples]
                                token_positions = np.array([item['token_position'] for item in activation_list], dtype=np.int32)  # [n_samples]

                                # Save datasets with aggressive compression to minimize file size
                                scheme_group.create_dataset('activations', data=activations_array,
                                                          compression='gzip',      # GZIP compression
                                                          compression_opts=9,      # Maximum compression level
                                                          shuffle=True,            # Improve compression by reordering bytes
                                                          fletcher32=True)         # Add checksum for data integrity
                                scheme_group.create_dataset('row_indices', data=row_indices)      # Which samples these activations came from
                                scheme_group.create_dataset('token_positions', data=token_positions)  # Absolute positions in tensors

                                # Save metadata as attributes for easy access
                                scheme_group.attrs['n_samples'] = len(activation_list)              # Number of samples in this scheme
                                scheme_group.attrs['activation_shape'] = activations_array.shape[1:]  # Shape of individual activations
                                scheme_group.attrs['token_scheme'] = scheme                         # Which token position scheme
                                scheme_group.attrs['batch_idx'] = batch_idx                        # Which batch this came from

        # Use atomic save for HDF5
        self.logger.progress(f"\nAttempting to save batch {batch_idx} activations to {batch_h5_path}")
        self.logger.progress(f"Data summary before save:")
        self.logger.progress(f"- Total hooks: {len(activations_data)}")
        
        # Sample a few hooks for detailed logging
        sample_hooks = list(activations_data.keys())[:3]
        for hook in sample_hooks:
            schemes = activations_data[hook]
            total = sum(len(data) for data in schemes.values())
            self.logger.progress(f"- {hook}: {len(schemes)} schemes, {total} total samples")
            
            # Show first scheme's data shape
            first_scheme = next(iter(schemes.values()))
            if first_scheme:
                first_item = first_scheme[0]
                self.logger.progress(f"  First item shape: {first_item['activations'].shape}")
        
        success = self.atomic_operation_manager.atomic_save_hdf5(save_hdf5_data, batch_h5_path)

        if success:
            self.logger.progress(f"\nBatch {batch_idx} activations saved successfully: {os.path.basename(batch_h5_path)}")
            return batch_h5_path
        else:
            self.logger.critical(f"\nFAILED to save batch {batch_idx} activations")
            # Enhanced error logging for debugging
        
            self.logger.debug(f"DEBUG: Target path: {batch_h5_path}")
            self.logger.debug(f"DEBUG: Activation data summary:")
            self.logger.debug(f"- Total hooks: {len(activations_data) if activations_data else 'None'}")
        
            if activations_data:
                # Check for data integrity
                empty_hooks = []
                malformed_hooks = []
                for hook_key, scheme_data in activations_data.items():
                    if not scheme_data:
                        empty_hooks.append(hook_key)
                        continue
                        
                    total_activations = sum(len(activation_list) for activation_list in scheme_data.values())
                    if total_activations == 0:
                        empty_hooks.append(hook_key)
                    
                    # Check for malformed data
                    for scheme, activation_list in scheme_data.items():
                        if activation_list and not all('activations' in item for item in activation_list):
                            malformed_hooks.append(f"{hook_key}/{scheme}")
                
                if empty_hooks:
                    self.logger.warning(f"WARNING: Found {len(empty_hooks)} empty hooks")
                    self.logger.warning(f"First few empty hooks: {empty_hooks[:3]}")
                
                if malformed_hooks:
                    self.logger.warning(f"WARNING: Found {len(malformed_hooks)} malformed hooks")
                    self.logger.warning(f"First few malformed: {malformed_hooks[:3]}")
            
            return None

    def flush_buffers(self, model_name):
        """Flush accumulated buffers to disk in consolidated format.
        
        Combines multiple buffered batches into single consolidated files for
        efficiency. Merges results and activations from all buffered batches,
        handling nested activation dictionaries (hook_key -> scheme -> list).
        
        Args:
            model_name (str): Model identifier for activation filename.
            
        Returns:
            None (writes consolidated files to disk)
            
        Notes:
            - Merges all results_buffer entries into single result list
            - Carefully merges activation_buffer preserving hook/scheme hierarchy
            - Saves consolidated results as consolidated_results_X_Y.pkl
            - Saves consolidated activations as consolidated_activations_X_Y.h5
            - Clears buffers after successful flush
            - Logs detailed debug info about buffer merging process
        """
        if not self.results_buffer:
            return

        # Combine all buffered results
        all_buffered_results = []
        all_buffered_activations = {}

        for batch_idx, (batch_results, batch_activations_item) in enumerate(zip(self.results_buffer, self.activation_buffer)):
            self.logger.debug(f"Processing buffer batch {batch_idx}: {len(batch_results)} results, {len(batch_activations_item)} hooks")
            all_buffered_results.extend(batch_results)

            # Corrected Merging Logic: Build the merged dictionary step-by-step to be robust.
            for hook_key, scheme_data in batch_activations_item.items():
                # Ensure the hook_key (e.g., 'layer_7_hook_resid_pre') exists in the master dictionary.
                if hook_key not in all_buffered_activations:
                    all_buffered_activations[hook_key] = {}

                for scheme, activation_list in scheme_data.items():
                    # Ensure the scheme (e.g., 'bos_token') exists for the current hook.
                    if scheme not in all_buffered_activations[hook_key]:
                        all_buffered_activations[hook_key][scheme] = []

                    # Debug: show what we're adding
                    if hook_key == 'layer_7_hook_resid_pre' and scheme == 'bos_token':
                        self.logger.debug(f"DEBUG: Batch {batch_idx}: Adding {len(activation_list)} activations to {hook_key}/{scheme}, current total: {len(all_buffered_activations[hook_key][scheme])}")

                    # Now that we are certain the list exists, extend it.
                    all_buffered_activations[hook_key][scheme].extend(activation_list)

        # Calculate the batch index based on how many batches we've processed
        # buffer_flush_counter tracks how many times we've flushed RESULTS_BUFFER_SIZE batches
        # RESULTS_BUFFER_SIZE is the number of batches we accumulate before flushing
        # So buffer_batch_idx represents the starting batch index for this flush
        buffer_batch_idx = self.buffer_flush_counter * self.RESULTS_BUFFER_SIZE  # Each flush handles RESULTS_BUFFER_SIZE batches

        self.logger.debug(f"DEBUG: After merging - all_buffered_results: {len(all_buffered_results)} items")
        self.logger.debug(f"DEBUG: After merging - all_buffered_activations: {len(all_buffered_activations)} hooks")
        if all_buffered_activations:
            total_activation_count = sum(len(scheme_data) for hook_data in all_buffered_activations.values() for scheme_data in hook_data.values())
            self.logger.debug(f"DEBUG: Total activations after merging: {total_activation_count}")

            # Detailed debug: Show sample counts per hook/scheme
            for hook_key, scheme_data in list(all_buffered_activations.items())[:3]:  # Show first 3 hooks
                for scheme, activation_list in scheme_data.items():
                    self.logger.debug(f"DEBUG: {hook_key}/{scheme}: {len(activation_list)} samples")
                    if len(activation_list) != len(all_buffered_results):
                        self.logger.error(f"ERROR: Mismatch! Expected {len(all_buffered_results)} samples but got {len(activation_list)}")

            # Verify we have the expected number of samples per scheme
            expected_samples = len(all_buffered_results)
            self.logger.debug(f"DEBUG: Expected {expected_samples} samples per hook/scheme (should be RESULTS_BUFFER_SIZE * BATCH_SIZE={self.RESULTS_BUFFER_SIZE*self.BATCH_SIZE})")

        # Store original results before evaluation for updating generation_results
        original_results_before_eval = all_buffered_results.copy()

        self.save_batch_checkpoint(all_buffered_results, buffer_batch_idx)

        # Update progress state with ALL buffered samples
        completed_row_indices_buffer = [res['row_idx'] for res in all_buffered_results]
        self.save_progress_state(completed_row_indices_buffer, buffer_batch_idx)

        # Update generation_results with evaluated results
        # Create a temporary activation buffer copy for updating (before clearing)
        temp_activation_buffer = [all_buffered_activations] if all_buffered_activations else None
        self.update_generation_results_with_evaluation(original_results_before_eval, all_buffered_results, temp_activation_buffer)

        # Now save activations after evaluation data is available
        self.logger.debug(f"DEBUG: all_buffered_activations is empty: {not all_buffered_activations}")
        self.logger.debug(f"DEBUG: all_buffered_activations keys: {list(all_buffered_activations.keys()) if all_buffered_activations else 'EMPTY'}")
        if all_buffered_activations:
            self.process_and_save_batch_activations(all_buffered_activations, all_buffered_results, model_name, buffer_batch_idx)
        else:   
            self.logger.debug(f"DEBUG: Skipping HDF5 save because all_buffered_activations is empty!")

        # Clear buffers
        self.results_buffer.clear()
        self.activation_buffer.clear()

        # Increment counter for next flush
        self.buffer_flush_counter += 1

        # Count total activations captured across all hooks and schemes
        total_activations = sum(
            len(scheme_list)
            for hook_data in all_buffered_activations.values()
            for scheme_list in hook_data.values()
        )

        self.logger.progress(f"Flushed {len(all_buffered_results)} samples to disk (buffer set {self.buffer_flush_counter}) - {total_activations} activations captured")

        # Debug: Show evaluation results for first 20 samples
        self.logger.progress("\n" + "="*60)
        self.logger.progress(f"EVALUATION RESULTS SUMMARY (Buffer {self.buffer_flush_counter})")
        self.logger.progress("="*60)

        # Show first 20 evaluated results in minimal format
        for i, result in enumerate(all_buffered_results[:20]):
            eval_val = result.get('is_correct', None)
            if eval_val == 1:
                eval_str = '✓ CORRECT'
            elif eval_val == 0:
                eval_str = '✗ INCORRECT'
            else:
                eval_str = '? NOT EVAL'

            self.logger.progress(f"[{i+1:2d}] Row {result['row_idx']:4d} | {eval_str} | "
                            f"Ref: '{result['right_answer']}' | Ans: '{result['gpt_answer_trim']}'")

        if len(all_buffered_results) > 20:
            self.logger.progress(f"... and {len(all_buffered_results) - 20} more results")

        self.logger.progress("="*60)

    def update_generation_results_with_evaluation(self, original_results, evaluated_results, activation_buffer=None):
        """Update generation_results and activation_buffer with evaluation results.
        
        After buffered evaluation completes, this merges evaluation labels back into
        both the generation_results list and the activation_buffer using row_idx
        as the key for matching.
        
        Args:
            original_results (list): Original generation results (not used directly).
            evaluated_results (list of dict): Evaluation results with row_idx and labels:
                - 'row_idx': int, sample identifier
                - 'is_correct': int, correctness label (0, 1, or 2)
                - 'evaluator_response': int, evaluator response
            activation_buffer (list of dict, optional): Buffered activation data to update.
            
        Returns:
            None (updates self.generation_results and activation_buffer in-place)
            
        Notes:
            - Creates row_idx mapping for O(1) lookup
            - Updates is_correct and evaluator_response fields
            - Also updates activation data if activation_buffer provided
            - Activation buffer structure: [{hook_key: {scheme: [activation_dicts]}}]
        """

        # Create a mapping from row_idx to evaluated result for quick lookup
        evaluated_by_row_idx = {result['row_idx']: result for result in evaluated_results}

        # Update generation_results with evaluation data
        for i, result in enumerate(self.generation_results):
            row_idx = result['row_idx']
            if row_idx in evaluated_by_row_idx:
                evaluated_result = evaluated_by_row_idx[row_idx]
                # Update only the evaluation fields
                if 'is_correct' in evaluated_result:
                    self.generation_results[i]['is_correct'] = evaluated_result['is_correct']
                if 'evaluator_response' in evaluated_result:
                    self.generation_results[i]['evaluator_response'] = evaluated_result['evaluator_response']

        # Also update activation data with evaluation results
        if activation_buffer is not None:
            # The buffer now contains a single, merged dictionary of all activations
            all_buffered_activations = activation_buffer[0]
            for hook_key, scheme_data in all_buffered_activations.items():
                for scheme, activation_list in scheme_data.items():
                    for activation_data in activation_list:
                        row_idx = activation_data['row_idx']
                        if row_idx in evaluated_by_row_idx:
                            evaluated_result = evaluated_by_row_idx[row_idx]
                            # Update evaluation fields in activation data
                            if 'is_correct' in evaluated_result:
                                activation_data['is_correct'] = evaluated_result['is_correct']
                            if 'evaluator_response' in evaluated_result:
                                activation_data['evaluator_response'] = evaluated_result['evaluator_response']

    def log_skipped_data_point(self, reason, row_data=None, row_idx=None):
        """Log skipped data point for summary report"""
        import time

        skip_entry = {
            'timestamp': time.strftime('%H:%M:%S'),
            'reason': reason,
            'row_idx': row_idx,
            'row_data': row_data
        }
        self.skipped_data_points.append(skip_entry)
        self.logger.warning(f"[SKIPPED] Row {row_idx}: {reason}")

    def print_detailed_gpu_memory(self, model=None):
        """Print detailed breakdown of GPU memory usage

        Args:
            model: Optional model to analyze parameter dtypes. If provided, will check
                   parameter dtypes and count FP16/BF16 vs FP32 parameters.
        """
        if not torch.cuda.is_available():
            self.logger.critical("GPU not available")
            return

        # Check model dtype if model is provided
        if model is not None:
            try:
                # Check a few key parameters for dtype
                param_dtypes = []
                for name, param in list(model.named_parameters())[:5]:  # Check first 5 parameters
                    param_dtypes.append(f"{name}: {param.dtype}")

                self.logger.info(f"Model parameter dtypes: {param_dtypes}")

                # Count FP32 vs FP16 parameters
                fp32_count = 0
                fp16_count = 0
                for param in model.parameters():
                    if param.dtype == torch.float32:
                        fp32_count += 1
                    elif param.dtype in (torch.float16, torch.bfloat16):
                        fp16_count += 1

                self.logger.info(f"Model parameters: {fp16_count} FP16/BF16, {fp32_count} FP32")

            except Exception as e:
                self.logger.warning(f"Could not check model dtypes: {e}")

        self.logger.info("\n" + "="*80)
        self.logger.info("DETAILED GPU MEMORY BREAKDOWN")
        self.logger.info("="*80)

        # Basic memory stats
        # After setting CUDA_VISIBLE_DEVICES, the visible GPU appears as cuda:0
        visible_device_id = 0
        self.logger.debug(f"DEBUG: Using visible_device_id = {visible_device_id} (physical GPU {self.DEVICE_ID})")
        allocated = torch.cuda.memory_allocated(visible_device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(visible_device_id) / 1024**3
        total = torch.cuda.get_device_properties(visible_device_id).total_memory / 1024**3
        
        self.logger.info(f"Total GPU Memory: {total:.2f} GB")
        self.logger.info(f"Allocated Memory: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        self.logger.info(f"Reserved Memory: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        self.logger.info(f"Free Memory: {total - reserved:.2f} GB")
        
        # Memory fragmentation
        if reserved > allocated:
            fragmentation = (reserved - allocated) / reserved * 100
            self.logger.warning(f"Memory Fragmentation: {fragmentation:.1f}%")
        
        # Detailed tensor analysis
        self.logger.info(f"\nDETAILED TENSOR ANALYSIS:")
        self.logger.info("-" * 60)
        
        # Get all tensors in memory
        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensors.append({
                        'size': obj.element_size() * obj.nelement() / 1024**3,  # GB
                        'shape': obj.shape,
                        'dtype': obj.dtype,
                        'device': str(obj.device),
                        'requires_grad': obj.requires_grad,
                        'is_leaf': obj.is_leaf
                    })
            except:
                pass
        
        # Group tensors by type
        tensor_groups = {}
        for tensor in tensors:
            # Categorize by shape pattern
            if len(tensor['shape']) == 3 and tensor['shape'][-1] > 1000:
                group = "Large Activations"
            elif len(tensor['shape']) == 4:
                group = "Attention Tensors"
            elif tensor['requires_grad']:
                group = "Gradients"
            elif tensor['dtype'] == torch.float16:
                group = "FP16 Tensors"
            elif tensor['dtype'] == torch.float32:
                group = "FP32 Tensors"
            else:
                group = "Other Tensors"
            
            if group not in tensor_groups:
                tensor_groups[group] = []
            tensor_groups[group].append(tensor)
        
        # Print breakdown
        total_tensor_memory = 0
        for group, group_tensors in tensor_groups.items():
            group_memory = sum(t['size'] for t in group_tensors)
            total_tensor_memory += group_memory
            self.logger.info(f"{group}: {group_memory:.2f} GB ({len(group_tensors)} tensors)")
            
            # Show largest tensors in each group
            largest_tensors = sorted(group_tensors, key=lambda x: x['size'], reverse=True)[:3]
            for i, tensor in enumerate(largest_tensors):
                self.logger.info(f"  {i+1}. Shape: {tensor['shape']}, Size: {tensor['size']:.2f} GB, Dtype: {tensor['dtype']}")
        
        self.logger.info(f"\nTotal Tensor Memory: {total_tensor_memory:.2f} GB")
        self.logger.info(f"Non-Tensor Memory: {allocated - total_tensor_memory:.2f} GB")
        
        # Memory recommendations
        self.logger.info(f"\nMEMORY OPTIMIZATION RECOMMENDATIONS:")
        self.logger.info("-" * 60)
        
        if 'Gradients' in tensor_groups:
            grad_memory = sum(t['size'] for t in tensor_groups['Gradients'])
            self.logger.warning(f"WARNING: GRADIENTS DETECTED: {grad_memory:.2f} GB - Use torch.inference_mode()")
        
        if 'Large Activations' in tensor_groups:
            act_memory = sum(t['size'] for t in tensor_groups['Large Activations'])
            self.logger.info(f"INFO: LARGE ACTIVATIONS: {act_memory:.2f} GB - Consider compression")
        
        if 'FP32 Tensors' in tensor_groups:
            fp32_memory = sum(t['size'] for t in tensor_groups['FP32 Tensors'])
            self.logger.info(f"INFO: FP32 TENSORS: {fp32_memory:.2f} GB - Convert to FP16")
        
        if allocated > total * 0.9:
            self.logger.warning(f"CRITICAL: HIGH MEMORY USAGE: {allocated/total*100:.1f}% - Reduce batch size")
        
        self.logger.info("="*80)

    def print_error_summary(self):
        """Print comprehensive error summary at end of execution"""
        self.logger.info("\nEXECUTION SUMMARY")

        # Print GPU memory breakdown
        # self.print_detailed_gpu_memory()

        # Skipped data points summary
        if self.skipped_data_points:
            self.logger.critical(f"\nSKIPPED DATA POINTS: {len(self.skipped_data_points)} total")

            # Group by reason
            skip_reasons = {}
            for skip in self.skipped_data_points:
                reason = skip['reason']
                if reason not in skip_reasons:
                    skip_reasons[reason] = []
                skip_reasons[reason].append(skip)

            for reason, skips in skip_reasons.items():
                self.logger.warning(f"\n{reason}: {len(skips)} occurrences")
                for skip in skips[:5]:  # Show first 5 examples
                    self.logger.warning(f"  - Row {skip['row_idx']} at {skip['timestamp']}")
                if len(skips) > 5:
                    self.logger.warning(f"  ... and {len(skips) - 5} more")
        else:
            self.logger.info("\nSKIPPED DATA POINTS: None - all data points processed successfully!")

        # Error summary
        if not self.global_error_log:
            self.logger.info("\nERRORS: No errors detected during execution!")
            return

        # Group errors by type
        error_types = {}
        for error in self.global_error_log:
            error_type = error['error_type']
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)

        self.logger.error(f"\nERRORS: {len(self.global_error_log)} total issues detected")

        for error_type, errors in error_types.items():
            self.logger.error(f"\n[{error_type}] - {len(errors)} occurrences:")
            for i, error in enumerate(errors[:3], 1):  # Show first 3 examples
                self.logger.error(f"  {i}. Time: {error['timestamp']}")
                self.logger.error(f"     Message: {error['message']}")
                if error['location']:
                    self.logger.error(f"     Location: {error['location']}")
                if error['row_idx'] is not None:
                    self.logger.error(f"     Row: {error['row_idx']}")
            if len(errors) > 3:
                self.logger.error(f"  ... and {len(errors) - 3} more")

    def clear_activation_storage(self):
        """Clear activation storage to prevent memory accumulation"""
        self.activation_storage = {}
        self.captured_activations.clear()  # Clear duplicate tracking at the same time

        # # BUFFERED SAVING - Runtime buffers
        # self.results_buffer = []
        # self.activation_buffer = [] 

