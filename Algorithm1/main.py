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
from src.rule_mining.mine_rules import mine_rules
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

def main(cfg: Dict):
    """Main pipeline function."""
    # Set up logging first
    logger = setup_logging(cfg)
    
    try:
        logger.info("Starting pipeline...")
        
        # Log system info
        log_system_info(logger)
        
        # Step 1: Load data
        logger.info("\nStep 1: Loading data...")
        data = load_all_frames(cfg)
        
        # Step 2: Join timeframes
        logger.info("\nStep 2: Joining timeframes...")
        joined = join_timeframes(data)
        
        # Step 3: Engineer features
        logger.info("\nStep 3: Engineering features...")
        features = engineer_features(joined, cfg)
        
        # Step 4: Generate labels
        logger.info("\nStep 4: Generating labels...")
        labeled_data = generate_labels(features, cfg)
        
        # Step 5: Mine rules
        logger.info("\nStep 5: Mining rules...")
        rules = mine_rules(labeled_data, cfg)
        
        # Step 6: Train models
        logger.info("\nStep 6: Training models...")
        models = train_models(labeled_data, cfg)
        
        # Step 7: Backtest
        logger.info("\nStep 7: Running backtest...")
        results = backtest(labeled_data, models, rules, cfg)
        
        # Step 8: Save results
        logger.info("\nStep 8: Saving results...")
        save_results(results, cfg)
        
        logger.info("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the trading pipeline')
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Run pipeline
    main(cfg)
