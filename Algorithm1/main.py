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

# Import our modules
from src.utils.logging_config import setup_logging
from src.ingestion.load_multitf import load_all_frames, validate_frames
from src.cleaning.standardise_columns import standardize_frames
from src.cleaning.fill_gaps import fill_gaps_in_frames
from src.cleaning.drop_nan_cols import drop_nan_columns_in_frames
from src.feature_engineering.join_timeframes import join_timeframes
from src.feature_engineering.engineer_features import engineer_features
from src.labeling.generate_labels import generate_labels
from src.rule_mining.mine_rules import mine_rules
from src.models.train_lgbm import train_lgbm_model
from src.models.train_tft import train_tft_model
from src.backtesting.walk_forward import run_walk_forward
from src.visualization.plot_results import plot_results

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

def save_pipeline_metadata(cfg: dict, start_time: datetime):
    """Save pipeline metadata including config and timing."""
    metadata = {
        'config': cfg,
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - start_time).total_seconds()
    }
    
    with open('artefacts/pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def main(cfg):
    # Set up logging
    logger = setup_logging()
    logger.info("Configuration loaded: %s", cfg)
    
    # Create necessary directories
    ensure_directories()
    
    # Record start time
    start_time = datetime.now()
    
    try:
        # 1️⃣ Load & unify
        logger.info("Step 1: Loading data files...")
        start_time = time.time()
        raw = load_all_frames(cfg)
        validate_frames(raw)
        logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        
        # 2️⃣ Clean & standardize
        logger.info("Step 2: Cleaning and standardizing data...")
        start_time = time.time()
        standardized = standardize_frames(raw)
        logger.info(f"Column standardization completed in {time.time() - start_time:.2f} seconds")
        
        # 3️⃣ Fill gaps
        logger.info("Step 3: Filling data gaps...")
        start_time = time.time()
        filled = fill_gaps_in_frames(standardized, cfg['price_cols'])
        logger.info(f"Gap filling completed in {time.time() - start_time:.2f} seconds")
        
        # 4️⃣ Remove high-NaN columns
        logger.info("Step 4: Removing high-NaN columns...")
        start_time = time.time()
        cleaned = drop_nan_columns_in_frames(filled)
        logger.info(f"NaN column removal completed in {time.time() - start_time:.2f} seconds")
        
        # 5️⃣ Join timeframes
        logger.info("Step 5: Joining timeframes...")
        start_time = time.time()
        joined = join_timeframes(cleaned, base_tf='15Second', lookback=True)
        logger.info(f"Timeframe joining completed in {time.time() - start_time:.2f} seconds")
        
        # 6️⃣ Engineer features
        logger.info("Step 6: Engineering features...")
        start_time = time.time()
        features = engineer_features(joined)
        logger.info(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
        
        # 7️⃣ Generate labels
        logger.info("Step 7: Generating labels...")
        start_time = time.time()
        labeled = generate_labels(features, cfg)
        logger.info(f"Label generation completed in {time.time() - start_time:.2f} seconds")
        
        # Save feature matrix
        logger.info("Saving feature matrix...")
        labeled.to_parquet("artefacts/feature_matrix.parquet")
        
        # 8️⃣ Mine rules
        logger.info("Step 8: Mining rules...")
        start_time = time.time()
        rules = mine_rules(labeled, cfg)
        logger.info(f"Rule mining completed in {time.time() - start_time:.2f} seconds")
        
        # Save rules
        with open("artefacts/rules.json", 'w') as f:
            json.dump(rules, f, indent=2)
        
        # 9️⃣ Train models
        logger.info("Step 9: Training models...")
        start_time = time.time()
        
        # Train LightGBM
        lgbm_model = train_lgbm_model(labeled, cfg)
        lgbm_model.save_model("models/lgbm_model.txt")
        
        # Train TFT
        tft_model = train_tft_model(labeled, cfg)
        tft_model.save("models/tft_model.ckpt")
        
        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
        
        # 🔟 Run walk-forward backtest
        logger.info("Step 10: Running walk-forward backtest...")
        start_time = time.time()
        results = run_walk_forward(labeled, rules, lgbm_model, tft_model, cfg)
        logger.info(f"Backtesting completed in {time.time() - start_time:.2f} seconds")
        
        # Save results
        with open("artefacts/backtest_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        logger.info("Generating plots...")
        plot_results(results, "results")
        
        # Save pipeline metadata
        save_pipeline_metadata(cfg, start_time)
        
        logger.info("\nPipeline Summary:")
        logger.info(f"Total rows processed: {len(labeled)}")
        logger.info(f"Total features: {len(labeled.columns)}")
        logger.info(f"Rules found: {len(rules)}")
        logger.info(f"Results saved to artefacts/ and results/ directories")
        
    except Exception as e:
        logger.error("Error in main pipeline: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    
    try:
        cfg = load_config(args.config)
        main(cfg)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
