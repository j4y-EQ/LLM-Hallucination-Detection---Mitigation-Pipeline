"""Model Lifecycle Management for Transformer Models.

Manages loading, optimization, and GPU memory management for transformer-based
language models using TransformerLens. Handles environment setup, zombie memory
detection, and multiple loading strategies with fallbacks.

FEATURES:
    - Automatic GPU detection and allocation
    - Zombie memory detection and cleanup
    - Multiple model loading strategies (direct, HuggingFace fallback)
    - Inference optimization (eval mode, no_grad)
    - Comprehensive error handling and retry logic
    - Memory monitoring and garbage collection

LOADING STRATEGIES:
    1. Direct TransformerLens loading (HookedTransformer.from_pretrained)
    2. HuggingFace transformers fallback (AutoModelForCausalLM + conversion)
    3. Local model loading if network unavailable

GPU MEMORY MANAGEMENT:
    - Pre-load zombie detection: Checks for unexpected GPU usage
    - Automatic GPU reset via nvidia-smi if zombies detected
    - Post-load memory monitoring and reporting
    - Garbage collection after model operations

USAGE:
    config = {'DEVICE_ID': 0, 'TRANSFORMER_LENS_MODEL_NAME': 'meta-llama/...', ...}
    manager = ModelManager(config)
    manager.load_model()
    manager.optimize_for_inference()
    model = manager.get_model()

Classes:
    ModelManager: Main model lifecycle management class

Dependencies:
    - TransformerLens: Primary model loading framework
    - HuggingFace transformers: Fallback loading
    - PyTorch: GPU management
"""

import os
import time
import torch
import gc
import subprocess
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from logger import consolidated_logger as logger

class ModelManager:
    """
    Manages the lifecycle of transformer language models.
    
    Handles environment setup, GPU memory management, model loading from HuggingFace,
    and optimization for inference. Supports both direct transformer-lens loading
    and fallback through HuggingFace transformers.
    
    Features:
        - Automatic GPU detection and memory monitoring
        - Zombie memory detection and automatic GPU reset
        - Multiple model loading strategies with fallbacks
        - Inference optimization (eval mode, gradient disabling)
        - Comprehensive error handling and logging
    
    Args:
        config (dict): Configuration dictionary containing:
            - 'DEVICE_ID': GPU device ID to use
            - 'TRANSFORMER_LENS_MODEL_NAME': Model identifier for transformer-lens
            - 'HUGGINGFACE_MODEL_ID': HuggingFace model identifier
            - 'MODEL_NAME': Human-readable model name for logging
    
    Attributes:
        config (dict): Configuration dictionary
        model (HookedTransformer): Loaded transformer-lens model instance
        device (str): Device string ('cuda:0' or 'cpu')
    
    Example:
        >>> config = {'DEVICE_ID': 0, 'TRANSFORMER_LENS_MODEL_NAME': 'gpt2', ...}
        >>> manager = ModelManager(config)
        >>> manager.load_model()
        >>> manager.optimize_for_inference()
        >>> model = manager.get_model()
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_model(self):
        """Returns the managed model instance."""
        return self.model

    def check_initial_gpu_memory(self):
        """Checks and logs the initial GPU memory state before model loading."""
        logger.debug(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.debug(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
            logger.debug(f"Current device: {torch.cuda.current_device()}")

            visible_device_id = 0 
            logger.debug(f"Using visible_device_id = {visible_device_id} (physical GPU {self.config['DEVICE_ID']})")

            try:
                allocated_memory = torch.cuda.memory_allocated(visible_device_id)
                total_memory = torch.cuda.get_device_properties(visible_device_id).total_memory
                logger.info(f"GPU Memory: {allocated_memory / 1024**3:.2f} GB allocated / {total_memory / 1024**3:.2f} GB total")

                if allocated_memory > 10 * 1024**3:
                    logger.info("Large model appears to be loaded on GPU")
                else:
                    logger.info("No large model detected on GPU")
            except Exception as e:
                logger.debug(f"Error accessing GPU memory: {e}")
                logger.debug("This might indicate a device mapping issue")
        else:
            logger.info("GPU not available")
        
        logger.debug("About to check GPU memory and availability")
        logger.debug(f"Current environment - CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")



    def clear_gpu_memory(self):
        """
        Forces garbage collection and clears the CUDA cache to free up GPU memory.
        Automatically detects and handles zombie memory situations by resetting the GPU.
        """
        logger.info("\n" + "="*60)
        logger.info("CLEARING GPU MEMORY")
        logger.info("="*60)
        logger.debug("clear_gpu_memory() called")

        if torch.cuda.is_available():
            visible_device_id = 0
            allocated = torch.cuda.memory_allocated(visible_device_id) / 1024**3
            if allocated > 20.0:
                logger.warning(f"High GPU memory usage detected: {allocated:.2f} GB")
                logger.info("Checking for zombie memory situation...")
                try:
                    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
                                         capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        logger.info("Active GPU processes found:")
                        logger.info(result.stdout.strip())
                    else:
                        logger.warning("No active GPU processes found - likely zombie memory situation")
                        logger.info("Attempting automatic GPU reset...")
                        try:
                            reset_result = subprocess.run(['nvidia-smi', '--gpu-reset'], capture_output=True, text=True, timeout=30)
                            if reset_result.returncode == 0:
                                logger.info("SUCCESS: GPU reset completed")
                                time.sleep(5)
                            else:
                                logger.error(f"GPU reset failed: {reset_result.stderr}")
                        except Exception as e:
                            logger.error(f"GPU reset failed with exception: {e}")
                except Exception as e:
                    logger.error(f"Could not check GPU processes: {e}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            visible_device_id = 0
            allocated = torch.cuda.memory_allocated(visible_device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(visible_device_id) / 1024**3
            logger.info("--- MEMORY STATE AFTER CLEARING ---")
            logger.info(f"GPU Memory after clearing: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved.")
            if allocated > 15.0:
                logger.warning(f"GPU memory still high ({allocated:.2f} GB) after clearing")
                logger.warning("You may need to manually run: sudo nvidia-smi --gpu-reset")
        else:
            logger.critical("GPU not available, skipping CUDA-specific clearing.")
        logger.info("="*60)

    def load_model(self):
        """Loads the model directly from HuggingFace using transformer-lens."""
        model_name = self.config['TRANSFORMER_LENS_MODEL_NAME']
        logger.info(f"\n=== Loading {self.config['MODEL_NAME']} ===")
        logger.info(f"Loading model from HuggingFace: {model_name}")
        logger.info("Note: First download may take several minutes...")

        try:
            logger.info(f"Loading with HookedTransformer.from_pretrained('{model_name}')...")
            self.model = HookedTransformer.from_pretrained(
                model_name, 
                device=self.device, 
                dtype=torch.bfloat16,
                fold_ln=False, 
                center_writing_weights=False, 
                center_unembed=False
            )
            logger.info(f"Successfully loaded model: {model_name}")
            return
        except Exception as e:
            logger.error(f"Approach 1 failed: {str(e)[:100]}...")

        # Approach 2: HuggingFace + transformer-lens (FALLBACK)
        if self.model is None:
            logger.info("Approach 2: Loading via HuggingFace + transformer-lens...")
            try:
                tokenizer = self._load_tokenizer_with_fallbacks()
                hf_model = self._load_hf_model()
                for model_ref_name in self.config['TRANSFORMER_LENS_MODEL_NAME']:
                    try:
                        logger.info(f"   Trying with model name: {model_ref_name}")
                        self.model = HookedTransformer.from_pretrained(
                            model_ref_name, hf_model=hf_model, tokenizer=tokenizer,
                            dtype=torch.bfloat16, fold_ln=False, center_writing_weights=False, center_unembed=False
                        )
                        logger.info(f"Success with model name: {model_ref_name}")
                        return
                    except Exception as e:
                        logger.error(f"   Failed with {model_ref_name}: {str(e)[:100]}...")
                        continue
            except Exception as e:
                logger.error(f"Approach 2 failed: {str(e)[:100]}...")

        if self.model is None:
            logger.critical("\n" + "="*80)
            logger.critical("MODEL LOADING FAILED")
            logger.critical("="*80)
            logger.critical(f"ERROR: Could not load model from HuggingFace: {model_name}")
            logger.critical(f"Exception: {str(e)}")
            logger.critical("\nPossible solutions:")
            logger.critical("1. Check your internet connection")
            logger.critical("2. Verify the model name is correct: " + model_name)
            logger.critical("3. Check that your HuggingFace token is valid (if model is private)")
            logger.critical("4. Ensure transformer-lens supports this model")
            logger.critical("\nStopping execution...")
            logger.critical("="*80)
            raise RuntimeError(f"Could not load model: {model_name}")
        
    def optimize_for_inference(self):
        """Optimizes the model for inference by disabling gradients and setting eval mode."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded yet.")
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Model optimized for inference - gradients disabled")

    def _download_llama_model(self):
        """Downloads the Llama 7B model from HuggingFace."""
        model_dir = self.config['MODEL_DIR']
        huggingface_id = self.config['HUGGINGFACE_MODEL_ID']
        
        try:
            logger.info("=" * 60)
            logger.info("LLAMA 7B MODEL DOWNLOADER")
            logger.info("=" * 60)
            logger.info(f"Downloading Llama 7B model to: {model_dir}")
            logger.info(f"Source: {huggingface_id}")
            logger.info("This may take several minutes depending on your internet connection...")
            logger.info("Model size: ~13.5 GB")

            os.makedirs(model_dir, exist_ok=True)

            logger.info("Step 1/2: Downloading tokenizer...")
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(
                huggingface_id,
                cache_dir=None,
                local_files_only=False
            )
            tokenizer.save_pretrained(model_dir)
            tokenizer_time = time.time() - start_time
            logger.info(f"Tokenizer downloaded successfully! ({tokenizer_time:.1f}s)")

            logger.info("Step 2/2: Downloading model weights...")
            logger.info("This will take longer - downloading ~13.5 GB...")
            model_start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                huggingface_id,
                cache_dir=None,
                local_files_only=False,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
            model.save_pretrained(model_dir)
            model_time = time.time() - model_start_time
            total_time = time.time() - start_time

            logger.info(f"Model weights downloaded successfully! ({model_time:.1f}s)")
            logger.info("=" * 60)
            logger.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Total download time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"Model saved to: {os.path.abspath(model_dir)}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False

    def _load_tokenizer_with_fallbacks(self):
        """Load tokenizer with multiple fallback strategies."""
        logger.info("   Loading tokenizer...")
        model_dir = self.config['MODEL_DIR']
        try:
            return AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            logger.error(f"   Tokenizer loading failed: {e}")
            logger.info("   Trying with trust_remote_code=True...")
            try:
                return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            except Exception as e2:
                logger.error(f"   Tokenizer loading failed even with trust_remote_code: {e2}")
                logger.info("   Trying with additional fallback parameters...")
                try:
                    return AutoTokenizer.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        use_fast=False,
                        local_files_only=True
                    )
                except Exception as e3:
                    logger.error(f"   All tokenizer loading approaches failed: {e3}")
                    raise e3

    def _load_hf_model(self):
        """Helper to load a HuggingFace model for transformer-lens."""
        logger.info("   Loading model weights for transformer-lens...")
        try:
            logger.debug(f"Loading model with device_map='{self.device}' (maps to physical GPU {self.config['DEVICE_ID']})")
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.config['MODEL_DIR'],
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            logger.info("   Model weights loaded successfully on GPU")
            return hf_model
        except torch.cuda.OutOfMemoryError as oom_e:
            logger.error(f"   GPU Out of Memory Error: {oom_e}")
            logger.error("   SOLUTION: Llama-7B requires ~14-16GB VRAM in bf16 mode")
            logger.error("   Try: 1) Close other GPU applications, 2) Use smaller model, or 3) Add more GPU memory")
            raise oom_e
        except Exception as e:
            logger.error(f"   Model loading failed: {e}")
            raise e
