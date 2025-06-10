import yaml, argparse, warnings, os
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import logging
import json
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna
from typing import Dict, List, Tuple
import traceback

# Import our modules
from src.utils.logging_config import (
    setup_logging, log_config, log_memory_usage,
    log_system_info, log_pipeline_metrics, save_log_metadata
)
from src.ingestion.load_multitf import load_all_frames, validate_frames
from src.cleaning.standardise_columns import standardize_frames
from src.cleaning.fill_gaps import fill_gaps_in_frames
from src.cleaning.drop_nan_cols import drop_nan_columns_in_frames
from src.feature_engineering.join_timeframes import join_timeframes
from src.feature_engineering.engineer_features import engineer_features
from src.labeling.generate_labels import generate_labels
from src.models.train_models import train_models
from src.backtesting.backtest import backtest
from src.utils.save_results import save_results

warnings.filterwarnings("ignore")

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file with proper encoding handling.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                cfg = yaml.safe_load(f)
                # Convert relative paths to absolute
                if 'data_dir' in cfg:
                    cfg['data_dir'] = str(Path(config_path).parent / cfg['data_dir'])
                return cfg
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise Exception(f"Error reading config file: {str(e)}")
    
    raise Exception(f"Could not read config file with any of the attempted encodings: {encodings}")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['artefacts', 'models', 'logs', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

def save_pipeline_metadata(cfg: dict, start_time: datetime, metrics: dict):
    """Save pipeline metadata including config and timing."""
    metadata = {
        'config': cfg,
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - start_time).total_seconds(),
        'metrics': metrics
    }
    
    with open('artefacts/pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories(config):
    """Create necessary directories if they don't exist"""
    dirs = [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['features_dir'],
        config['model']['models_dir']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the trading pipeline')
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("Starting new pipeline run")
    logger.info(f"Log file: logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"Log level: {config['logging']['level']}")
    logger.info("=" * 80)
    
    try:
        # Create necessary directories
        create_directories(config)
        
        # Step 1: Load and process data
        logger.info("\nStep 1: Loading data...")
        raw_data_dict = load_all_frames(config)
        validate_frames(raw_data_dict)
        # Select the 15Second DataFrame
        raw_data = raw_data_dict[config['timeframes'][0]]
        
        # Step 2: Engineer features
        logger.info("\nStep 2: Engineering features...")
        features = engineer_features(raw_data, config)
        
        # Step 3: Generate labels
        logger.info("\nStep 3: Generating labels...")
        labeled_data = generate_labels(features, config)
        
        # Step 4: Train models
        logger.info("\nStep 4: Training models...")
        models = train_models(labeled_data, config)
        
        logger.info("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
