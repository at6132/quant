import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"algorithm1_{timestamp}.log"
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatters
    file_formatter = logging.Formatter(log_format, date_format)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", date_format)
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create and configure project logger
    logger = logging.getLogger("algorithm1")
    logger.setLevel(logging.DEBUG)
    
    # Log initial information
    logger.info("=" * 50)
    logger.info("Starting Algorithm1 Pipeline")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 50)
    
    return logger 