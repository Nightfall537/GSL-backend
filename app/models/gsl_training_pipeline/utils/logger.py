"""
Logging utility for GSL training pipeline
Provides consistent logging across all modules
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level: str = 'INFO', console_output: bool = True):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str):
    """
    Get existing logger or create new one with default settings
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Context manager for logging progress of long operations"""
    
    def __init__(self, logger, total: int, desc: str = "Processing"):
        """
        Initialize progress logger
        
        Args:
            logger: Logger instance
            total: Total number of items to process
            desc: Description of the operation
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting: {self.desc} (Total: {self.total})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"✅ Completed: {self.desc} in {elapsed:.2f}s")
        else:
            self.logger.error(f"❌ Failed: {self.desc} after {elapsed:.2f}s")
        return False
    
    def update(self, n: int = 1, message: str = None):
        """
        Update progress
        
        Args:
            n: Number of items processed
            message: Optional message to log
        """
        self.current += n
        progress = (self.current / self.total) * 100
        
        if message:
            self.logger.info(f"[{progress:.1f}%] {message}")
        elif self.current % max(1, self.total // 10) == 0:  # Log every 10%
            self.logger.info(f"[{progress:.1f}%] {self.desc}: {self.current}/{self.total}")
