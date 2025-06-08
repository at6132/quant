import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import os

def setup_logging(config: dict) -> logging.Logger:
    """Set up logging configuration."""
    # Get log level from config
    log_level = config.get('log_level', 'INFO')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # File gets all logs
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))  # Console gets config level
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial information
    logger.info("=" * 80)
    logger.info("Starting new pipeline run")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")
    logger.info("=" * 80)
    
    return logger

def log_config(cfg: dict, logger: logging.Logger):
    """Log configuration details."""
    logger.info("\nConfiguration:")
    logger.info("-" * 40)
    
    # Log data parameters
    logger.info("\nData Parameters:")
    logger.info(f"Data directory: {cfg.get('data_dir', 'Not specified')}")
    logger.info(f"Price columns: {cfg.get('price_cols', [])}")
    
    # Log feature engineering parameters
    logger.info("\nFeature Engineering Parameters:")
    fe_params = cfg.get('feature_engineering', {})
    logger.info(f"Rolling windows: {fe_params.get('rolling_windows', [])}")
    logger.info(f"Cross indicator pairs: {fe_params.get('cross_indicator_pairs', [])}")
    
    # Log labeling parameters
    logger.info("\nLabeling Parameters:")
    label_params = cfg.get('label', {})
    logger.info(f"Horizon minutes: {label_params.get('horizon_minutes')}")
    logger.info(f"Dollar threshold: ${label_params.get('dollar_threshold')}")
    
    # Log rule mining parameters
    logger.info("\nRule Mining Parameters:")
    rule_params = cfg.get('rule_mining', {})
    logger.info(f"Min precision: {rule_params.get('min_precision')}")
    logger.info(f"Min recall: {rule_params.get('min_recall')}")
    logger.info(f"Max indicators: {rule_params.get('max_indicators')}")
    
    # Log model parameters
    logger.info("\nModel Parameters:")
    model_params = cfg.get('models', {})
    for model_name, params in model_params.items():
        logger.info(f"\n{model_name.upper()}:")
        for param, value in params.items():
            logger.info(f"  {param}: {value}")
    
    # Log backtesting parameters
    logger.info("\nBacktesting Parameters:")
    backtest_params = cfg.get('backtesting', {})
    logger.info(f"Number of splits: {backtest_params.get('n_splits')}")
    
    logger.info("-" * 40)

def log_memory_usage(logger: logging.Logger):
    """Log current memory usage."""
    import psutil
    process = psutil.Process()
    
    logger.info("\nMemory Usage:")
    logger.info(f"RSS: {process.memory_info().rss / (1024**2):.2f} MB")
    logger.info(f"VMS: {process.memory_info().vms / (1024**2):.2f} MB")
    logger.info("")

def log_system_info(logger: logging.Logger):
    """Log system information."""
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

def log_pipeline_metrics(metrics: dict, logger: logging.Logger):
    """Log pipeline performance metrics."""
    logger.info("\nPipeline Metrics:")
    logger.info("-" * 40)
    
    for stage, timing in metrics.items():
        logger.info(f"{stage}: {timing:.2f} seconds")
    
    logger.info("-" * 40)

def save_log_metadata(logger: logging.Logger, metadata: dict):
    """Save additional metadata to the log file."""
    logger.info("\nPipeline Metadata:")
    logger.info("-" * 40)
    
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            logger.info(f"{key}: {json.dumps(value, indent=2)}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("-" * 40) 