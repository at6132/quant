import yaml, argparse, warnings, os
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

from src.utils.logging_config import setup_logging
from src.ingestion.load_multitf import load_all_frames, validate_frames
from src.cleaning.standardise_columns import standardize_frames
from src.cleaning.fill_gaps import fill_gaps_in_frames
from src.cleaning.drop_nan_cols import drop_nan_columns_in_frames
from src.feature_engineering.join_timeframes import join_timeframes

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

def main(cfg):
    # Set up logging
    logger = setup_logging()
    logger.info("Configuration loaded: %s", cfg)
    
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
        
        # Save intermediate results
        logger.info("Saving processed data...")
        output_dir = Path("processed_data")
        output_dir.mkdir(exist_ok=True)
        joined.to_parquet(output_dir / "processed_data.parquet")
        logger.info(f"Processed data saved to {output_dir / 'processed_data.parquet'}")
        
        # Log summary statistics
        logger.info("\nData Processing Summary:")
        logger.info(f"Total rows: {len(joined)}")
        logger.info(f"Total columns: {len(joined.columns)}")
        logger.info(f"Memory usage: {joined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
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
