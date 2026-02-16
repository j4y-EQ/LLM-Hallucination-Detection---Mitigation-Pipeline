"""Centralized Logging System for Hallucination Detection Pipeline.

Provides industrial-strength logging with multiple handlers, custom formatters,
and level-specific log files. Features console output with ANSI colors,
rotating file handlers, and specialized loggers for different components.

FEATURES:
    - Centralized logger (consolidated_logger) for most operations
    - Level-specific log files (errors.log, warnings.log, info.log, debug.log)
    - Rotating file handlers to prevent excessive disk usage
    - Console output with ANSI color coding
    - Custom log levels (PROGRESS, EVALUATION, SUCCESS)
    - Thread-safe buffering during initialization

LOG LEVELS:
    Standard Python logging levels:
    - DEBUG (10): Detailed debugging information
    - INFO (20): General informational messages
    - WARNING (30): Warning messages
    - ERROR (40): Error messages
    - CRITICAL (50): Critical errors
    
    Custom levels:
    - PROGRESS (25): Progress updates (between INFO and WARNING)
    - EVALUATION (21): Evaluation-specific messages
    - SUCCESS (22): Success confirmations

LOG FILES:
    - main.log: All messages (DEBUG and above)
    - errors.log: ERROR and CRITICAL only
    - warnings.log: WARNING and above
    - info.log: INFO and above
    - debug.log: Everything including DEBUG

USAGE:
    from logger import consolidated_logger as logger
    
    logger.info('Informational message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.progress('Progress update')  # Custom level

CONFIGURATION:
    Log rotation settings from config.py:
    - LOG_MAX_BYTES: Max size before rotation (default: 10MB)
    - LOG_BACKUP_COUNT: Number of backup files (default: 5)

Classes:
    - ColorFormatter: ANSI color formatting for console output
    - BufferingHandler: Memory buffer for early initialization
    - ConsolidatedLogger: Main logger with custom levels

Dependencies:
    - Python logging: Standard library logging
    - logging.handlers: RotatingFileHandler
"""

# ================================================================
# CENTRALIZED LOGGING SYSTEM
# Industry-standard logging with centralized and level-specific log files
# ================================================================

import os
import sys
import logging
import threading
from typing import Optional, Union
from datetime import datetime
from logging.handlers import RotatingFileHandler

class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds ANSI color codes to console log output.
    
    Applies different colors based on log level:
        - DEBUG: Blue
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Bold Red
    
    Args:
        Inherits all arguments from logging.Formatter
    """
    COLORS = {
        logging.DEBUG: '\033[94m',     # Blue
        logging.INFO: '\033[92m',      # Green
        logging.WARNING: '\033[93m',   # Yellow
        logging.ERROR: '\033[91m',     # Red
        logging.CRITICAL: '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with color"""
        log_message = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{log_message}{self.RESET}"

class BufferingHandler(logging.Handler):
    """
    Logging handler that buffers records in memory until file handlers are ready.
    
    Used during early initialization when output directory is not yet known.
    Records are held in memory and flushed to file handlers once they are created.
    This ensures no log messages are lost during startup.
    
    Attributes:
        buffer (list): List of buffered LogRecord objects
    """
    def __init__(self):
        super().__init__()
        self.buffer = []

    def emit(self, record):
        """Buffer the log record."""
        self.buffer.append(record)

    def flush_to_handlers(self, handlers):
        """Flush the buffered records to a list of new handlers."""
        self.acquire()
        try:
            for record in self.buffer:
                for handler in handlers:
                    if record.levelno >= handler.level:
                        handler.emit(record)
            self.buffer = []
        finally:
            self.release()

class CentralizedLogger:
    """
    Advanced logging system with lazy file initialization and multi-level output.
    
    Features:
        - Immediate console output with color-coding
        - In-memory buffering until output directory is set
        - Automatic file handler creation with rotation
        - Separate error-only log file
        - Custom log levels (PROGRESS, BATCH, EVALUATION)
        - Thread-safe operations
        - Captures third-party library warnings
    
    Log Files Created:
        - {name}_full_{timestamp}.log: All log levels (DEBUG and above)
        - {name}_errors_{timestamp}.log: Warnings and errors only
    
    Args:
        name (str): Base name for the logger and log files. Defaults to 'pipeline'.
        debug_verbose (bool): If True, show DEBUG messages on console. Defaults to False.
        output_dir (Optional[str]): Directory for log files. If None, buffering mode until set.
        max_bytes (int): Max bytes per full log file before rotation. Defaults to 50MB.
        backup_count (int): Number of rotated log files to keep. Defaults to 5.
        error_max_bytes (int): Max bytes per error log file before rotation. Defaults to 10MB.
        error_backup_count (int): Number of rotated error log files to keep. Defaults to 3.
    
    Example:
        >>> logger = CentralizedLogger(name='my_pipeline', debug_verbose=True)
        >>> logger.info('Starting process...')
        >>> logger.set_output_directory('./logs')
        >>> logger.error('An error occurred')
    """
    
    # Custom log levels
    PROGRESS = 15
    BATCH = 16
    EVALUATION = 17

    def __init__(
        self,
        name: str = 'pipeline',
        debug_verbose: bool = False,
        output_dir: Optional[str] = None,
        max_bytes: int = 50*1024*1024,
        backup_count: int = 5,
        error_max_bytes: int = 10*1024*1024,
        error_backup_count: int = 3
    ):
        self._lock = threading.Lock()
        
        # Store configuration for later use
        self.name = name
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.error_max_bytes = error_max_bytes
        self.error_backup_count = error_backup_count
        self.debug_verbose = debug_verbose
        
        # Create logger with a unique name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger_name = f"{self.name}_{self.timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent propagation to root logger
        
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Add custom log levels
        logging.addLevelName(self.PROGRESS, "PROGRESS")
        logging.addLevelName(self.BATCH, "BATCH")
        logging.addLevelName(self.EVALUATION, "EVALUATION")
        
        # Standard formatter for all handlers
        self.formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. Console Handler (always active)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColorFormatter(
            '%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG if debug_verbose else logging.INFO)
        self.logger.addHandler(console_handler)
        
        # 2. Buffering Handler (active until file handlers are set)
        self.buffering_handler = BufferingHandler()
        self.logger.addHandler(self.buffering_handler)
        
        self.file_handlers_initialized = False

    def set_output_directory(self, output_dir: str):
        """
        Initializes file-based logging. Creates log files in the specified directory,
        flushes any buffered logs to them, and replaces the buffer handler with file handlers.
        Also configures the root logger to capture third-party library warnings.
        """
        with self._lock:
            if self.file_handlers_initialized:
                return

            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            
            # 1. Centralized Log File Handler
            central_log_path = os.path.join(self.output_dir, f'{self.name}_full_{self.timestamp}.log')
            central_handler = RotatingFileHandler(
                central_log_path, mode='a', maxBytes=self.max_bytes, backupCount=self.backup_count
            )
            central_handler.setLevel(logging.DEBUG)
            central_handler.setFormatter(self.formatter)
            
            # 2. Error-only Log File Handler
            error_log_path = os.path.join(self.output_dir, f'{self.name}_errors_{self.timestamp}.log')
            error_handler = RotatingFileHandler(
                error_log_path, mode='a', maxBytes=self.error_max_bytes, backupCount=self.error_backup_count
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(self.formatter)

            # Flush buffered records to the new file handlers
            self.buffering_handler.flush_to_handlers([central_handler, error_handler])
            
            # Replace the buffering handler with the permanent file handlers
            self.logger.removeHandler(self.buffering_handler)
            self.logger.addHandler(central_handler)
            self.logger.addHandler(error_handler)
            
            # Configure root logger to capture warnings from third-party libraries
            # (e.g., HuggingFace, transformers, torch) and forward them to our logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            
            # Remove any existing root handlers to avoid duplicates
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Add our handlers to the root logger so third-party warnings are captured
            root_logger.addHandler(central_handler)
            root_logger.addHandler(error_handler)
            
            self.file_handlers_initialized = True
            
            # Replace the buffering handler with the permanent file handlers
            self.logger.removeHandler(self.buffering_handler)
            self.logger.addHandler(central_handler)
            self.logger.addHandler(error_handler)
            
            self.file_handlers_initialized = True
    
    def _log(
        self, 
        level: int, 
        msg: str, 
        *args, 
        **kwargs
    ):
        with self._lock:
            if level == logging.DEBUG and not self.debug_verbose:
                return
            
            self.logger.log(level, msg, *args, **kwargs)
            
            # Force immediate flushing for all handlers
            for handler in self.logger.handlers:
                try:
                    handler.flush()
                except Exception:
                    pass  # Ignore any flushing errors
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message (only when verbose)"""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log informational message"""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def progress(self, msg: str, *args, **kwargs):
        """Log progress updates"""
        self._log(self.PROGRESS, msg, *args, **kwargs)
    
    def batch_info(self, msg: str, *args, **kwargs):
        """Log batch processing information"""
        self._log(self.BATCH, msg, *args, **kwargs)
    
    def evaluation(self, msg: str, *args, **kwargs):
        """Log evaluation results"""
        self._log(self.EVALUATION, msg, *args, **kwargs)

# Global logger instance
_global_logger = None

# Create a single consolidated logger for all components
from config import ACTIVATIONS_BASE_DIR, DEBUG_VERBOSE
consolidated_logger = CentralizedLogger(
    name='logs',
    debug_verbose=DEBUG_VERBOSE,
    output_dir=ACTIVATIONS_BASE_DIR,
    max_bytes=100*1024*1024,  # Larger files for consolidated logging
    backup_count=10,
    error_max_bytes=20*1024*1024,
    error_backup_count=5
)

def clear_global_logger():
    """Clear the global logger instance"""
    global _global_logger
    if _global_logger is not None:
        for handler in _global_logger.logger.handlers[:]:
            _global_logger.logger.removeHandler(handler)
            handler.close()
        _global_logger = None

def get_logger(
    name: str = 'pipeline',
    debug_verbose: bool = False,
    output_dir: Optional[str] = None,
    max_bytes: int = 50*1024*1024,
    backup_count: int = 5,
    error_max_bytes: int = 10*1024*1024,
    error_backup_count: int = 3,
    force_reinit: bool = False
) -> CentralizedLogger:
    """
    Get a configured logger instance

    Args:
        name (str): Name of the logger
        debug_verbose (bool): Enable detailed debug logging
        output_dir (str, optional): Directory to store log files
        max_bytes (int): Maximum bytes per log file before rotation
        backup_count (int): Number of backup log files to keep
        error_max_bytes (int): Maximum bytes per error log file before rotation
        error_backup_count (int): Number of backup error log files to keep
        force_reinit (bool): Force reinitialization even if logger exists

    Returns:
        CentralizedLogger: Configured logger instance
    """
    global _global_logger
    
    # Create global logger if not exists, or reinitialize if force_reinit is True
    if _global_logger is None or force_reinit:
        # Clear existing handlers if reinitializing
        if _global_logger is not None:
            clear_global_logger()
        
        _global_logger = CentralizedLogger(
            name=name,
            debug_verbose=debug_verbose,
            output_dir=output_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
            error_max_bytes=error_max_bytes,
            error_backup_count=error_backup_count
        )
    
    return _global_logger