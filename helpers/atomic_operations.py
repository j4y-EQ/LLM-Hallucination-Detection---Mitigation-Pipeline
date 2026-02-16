"""Atomic File Operations for Checkpoint Management.

Provides industrial-strength atomic file operations with automatic corruption
recovery, backup mechanisms, and support for multiple file formats (pickle,
JSON, HDF5, CSV). Ensures data integrity during saves and loads.

FEATURES:
    - Atomic write-then-rename strategy to prevent partial writes
    - Automatic backup creation and restoration on failure
    - Corruption detection with fallback to backup files
    - Support for pickle, JSON, HDF5, and CSV formats
    - Configurable debug logging for troubleshooting

ATOMIC SAVE STRATEGY:
    1. Write data to temporary file (.tmp)
    2. Create backup of existing file (.backup) if present
    3. Atomically rename temp file to target (OS-level atomic operation)
    4. Clean up backup after successful completion
    5. Rollback to backup if any step fails

ATOMIC LOAD STRATEGY:
    1. Try loading from main file
    2. If corrupted, fall back to .backup file
    3. If backup used successfully, restore as main file
    4. Return None if both files are corrupted (caller handles)

USAGE:
    atomic_ops = AtomicOperations(debug_verbose=False)
    atomic_ops.atomic_save(data, 'checkpoint.pkl')
    data = atomic_ops.atomic_load('checkpoint.pkl')

Classes:
    AtomicOperations: Main class for atomic file operations

Dependencies:
    - pickle: For serialization
    - h5py: For HDF5 support
    - pandas: For CSV operations
"""

# ================================================================
# ATOMIC OPERATIONS MODULE
# Industry-standard atomic file operations for checkpoint management
# ================================================================

# Imports
import os
import gc
import time
import torch
import pandas as pd
import h5py
import numpy as np
from tqdm.auto import tqdm
import pickle
import glob
import json
from typing import Dict, List, Any, Optional, Callable

# Import the consolidated logger
from logger import consolidated_logger

# ================================================================
# ATOMIC CHECKPOINT SYSTEM - INDUSTRY STANDARD
# ================================================================

class AtomicOperations:
    """
    A comprehensive manager for atomic file operations with configurable debug settings.

    This class provides atomic save and load operations with built-in error handling,
    backup mechanisms, and optional verbose logging.
    """

    def __init__(self, debug_verbose: bool = False):
        """
        Initialize the AtomicOperations.

        Args:
            debug_verbose (bool): Enable detailed logging for debugging. Defaults to False.
        """
        self.debug_verbose = debug_verbose
        # Use consolidated logger for atomic operations
        self.logger = consolidated_logger
    
    def _log(self, message: str):
        """
        Log messages if debug_verbose is enabled.

        Args:
            message (str): Message to log
        """
        if self.debug_verbose:
            self.logger.debug(message)
    
    def atomic_save(self, 
                    data: Any, 
                    filepath: str, 
                    save_func: Optional[Callable[[Any, str], None]] = None
                   ) -> bool:
        """
        Atomic save with temp file and backup mechanism.
        
        Args:
            data (Any): Data to save
            filepath (str): Target file path
            save_func (Optional[Callable]): Custom save function. If None, uses pickle.
        
        Returns:
            bool: True if save successful, False otherwise
        """
        temp_filepath = filepath + '.tmp'
        backup_filepath = filepath + '.backup'
        
        try:
            # Step 1: Write to temporary file
            if save_func:
                save_func(data, temp_filepath)
            else:
                with open(temp_filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Step 2: Backup existing file
            if os.path.exists(filepath):
                if os.path.exists(backup_filepath):
                    os.remove(backup_filepath)
                os.rename(filepath, backup_filepath)
            
            # Step 3: Atomic rename
            os.rename(temp_filepath, filepath)
            
            # Step 4: Clean up backup
            if os.path.exists(backup_filepath):
                os.remove(backup_filepath)
                
            return True
            
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            # Restore backup if needed
            if os.path.exists(backup_filepath) and not os.path.exists(filepath):
                os.rename(backup_filepath, filepath)
                
            self._log(f"Atomic save failed for {filepath}: {e}")
            return False
    
    def atomic_load(self, 
                    filepath: str, 
                    load_func: Optional[Callable[[str], Any]] = None
                   ) -> Optional[Any]:
        """
        Load with automatic corruption recovery.
        
        Args:
            filepath (str): Path to the file to load
            load_func (Optional[Callable]): Custom load function. If None, uses pickle.
        
        Returns:
            Optional[Any]: Loaded data or None if loading fails
        """
        backup_filepath = filepath + '.backup'
        
        # Try main file first
        if os.path.exists(filepath):
            try:
                if load_func:
                    return load_func(filepath)
                else:
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                self._log(f"Main file corrupted: {e}")
        
        # Fallback to backup
        if os.path.exists(backup_filepath):
            try:
                self._log(f"Recovering from backup: {backup_filepath}")
                if load_func:
                    data = load_func(backup_filepath)
                else:
                    with open(backup_filepath, 'rb') as f:
                        data = pickle.load(f)
                
                # Restore backup as main file
                self.atomic_save(data, filepath)
                return data
                
            except Exception as e:
                self._log(f"Backup also corrupted: {e}")
        
        return None
    
    def atomic_save_hdf5(self, 
                          save_function: Callable[[str], None], 
                          filepath: str
                         ) -> bool:
        """
        Atomic save for HDF5 files with integrity checking.
        
        Args:
            save_function (Callable): Function to save HDF5 file
            filepath (str): Target file path
        
        Returns:
            bool: True if save successful, False otherwise
        """
        temp_filepath = filepath + '.tmp'
        backup_filepath = filepath + '.backup'
        
        try:
            # Save to temp file
            self._log(f"Attempting to save to temp file: {temp_filepath}")
            save_function(temp_filepath)
            self._log("Initial save to temp file completed")
            
            # Verify HDF5 file integrity
            try:
                self._log("Verifying HDF5 file integrity...")
                with h5py.File(temp_filepath, 'r') as test_file:
                    keys = list(test_file.keys())
                    if len(keys) == 0:
                        raise ValueError("Empty HDF5 file created")
                    self._log(f"HDF5 file verified - contains {len(keys)} keys")
            except Exception as e:
                self._log(f"HDF5 integrity check failed with error: {str(e)}")
                raise ValueError(f"HDF5 integrity check failed: {e}")
            
            # Backup existing file
            if os.path.exists(filepath):
                if os.path.exists(backup_filepath):
                    os.remove(backup_filepath)
                os.rename(filepath, backup_filepath)
            
            # Atomic rename
            os.rename(temp_filepath, filepath)
            
            # Clean up backup
            if os.path.exists(backup_filepath):
                os.remove(backup_filepath)
                
            return True
            
        except Exception as e:
            # Cleanup on failure
            self._log(f"\nHDF5 save failed with error: {str(e)}")
            
            if os.path.exists(temp_filepath):
                self._log(f"Cleaning up temp file: {temp_filepath}")
                try:
                    os.remove(temp_filepath)
                    self._log("Temp file cleanup successful")
                except Exception as cleanup_e:
                    self._log(f"Failed to clean up temp file: {cleanup_e}")
            
            if os.path.exists(backup_filepath) and not os.path.exists(filepath):
                self._log(f"Attempting to restore from backup: {backup_filepath}")
                try:
                    os.rename(backup_filepath, filepath)
                    self._log("Backup restoration successful")
                except Exception as restore_e:
                    self._log(f"Failed to restore from backup: {restore_e}")
            
            self._log(f"HDF5 atomic save failed: {e}")
            return False






