#!/usr/bin/env python3
import yaml
import argparse
import warnings
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from src.ingestion.load_multitf import load_all_frames, validate_frames
from src.feature_engineering.engineer_features import engineer_features  
from src.labeling.generate_labels import generate_labels
from src.models.train_models import train_models

warnings.filterwarnings("ignore")

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
    print("Starting pipeline...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the trading pipeline')
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
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
        print("Loading data...")
        raw_data_dict = load_all_frames(config)
        validate_frames(raw_data_dict)
        # Select the 15Second DataFrame
        raw_data = raw_data_dict[config['timeframes'][0]]
        print(f"Data loaded: {raw_data.shape}")
        
        # Step 2: Engineer features
        logger.info("\nStep 2: Engineering features...")
        print("Engineering features...")
        features = engineer_features(raw_data, config)
        print(f"Features engineered: {features.shape}")
        
        # Step 3: Generate labels
        logger.info("\nStep 3: Generating labels...")
        print("Generating labels...")
        labeled_data = generate_labels(features, config)
        print(f"Labels generated: {labeled_data.shape}")
        print(f"Label distribution:\n{labeled_data['label'].value_counts()}")
        
        # Step 4: Train models
        logger.info("\nStep 4: Training models...")
        print("\nStarting model training...")
        print("This may take a few minutes...")
        
        # Only train if we have valid labels
        if labeled_data['label'].nunique() > 1:
            models = train_models(labeled_data, config)
            logger.info("\nPipeline completed successfully!")
            print("\n✅ Pipeline completed successfully!")
            
            # Save the trained models
            if models:
                print(f"\nModels trained: {list(models.keys())}")
        else:
            logger.error("Not enough unique labels for training!")
            print("❌ Error: Not enough unique labels for training!")
            
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()