#!/usr/bin/env python3
"""
Complete Training Script for the Intelligent Trading System.
Trains the intuition model end-to-end with all components.
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_processor import MultiTimeframeDataProcessor
from data.label_generator import IntuitionLabelGenerator
from models.model_trainer import ModelTrainer
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """Load training configuration."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()

def get_default_config() -> dict:
    """Get default configuration."""
    return {
        'data': {
            'timeframes': ['15s', '1m', '5m', '15m', '1h', '4h'],
            'data_path': 'data/sample_data.csv',
            'indicators': ['it_foundation', 'smc_core', 'breaker_signals', 'ict_sm_trades']
        },
        'labeling': {
            'lookforward_periods': [5, 10, 20, 30],
            'threshold': 0.002,
            'min_move_size': 0.002,
            'max_holding_time': 30,
            'min_holding_time': 5
        },
        'model': {
            'input_dim': 10,
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'sequence_length': 50,
            'prediction_horizon': 1,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 100,
            'patience': 10
        },
        'training': {
            'validation_split': 0.2,
            'optimize_hyperparams': True,
            'n_trials': 20,
            'save_model': True,
            'plot_history': True
        },
        'risk': {
            'max_risk_per_trade': 0.02,
            'max_portfolio_risk': 0.06,
            'base_position_size': 0.01,
            'volatility_lookback': 60
        }
    }

def create_sample_data_if_needed(data_path: str, num_samples: int = 10000) -> pd.DataFrame:
    """Create sample data if it doesn't exist."""
    if os.path.exists(data_path):
        logger.info(f"Using existing data from {data_path}")
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    logger.info(f"Creating sample data with {num_samples} samples")
    
    # Create realistic sample data
    dates = pd.date_range('2024-01-01', periods=num_samples, freq='15S', tz='UTC')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    
    # Base price with trend
    base_price = 50000
    trend = np.linspace(0, 0.1, num_samples)  # 10% trend over period
    noise = np.random.normal(0, 0.001, num_samples)  # 0.1% noise
    
    close_prices = base_price * (1 + trend + noise)
    
    # Generate OHLCV data
    volatility = 0.002  # 0.2% volatility
    
    data = []
    for i, close in enumerate(close_prices):
        # Generate realistic OHLC
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = close * (1 + np.random.normal(0, volatility * 0.5))
        
        # Ensure OHLC relationship
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume with some correlation to price movement
        volume = np.random.uniform(100, 1000) * (1 + abs(close - open_price) / close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Save data
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path)
    logger.info(f"Sample data saved to {data_path}")
    
    return df

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train the intuition model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Skip hyperparameter optimization')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with reduced epochs and no optimization')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data:
        config['data']['data_path'] = args.data
    if args.epochs:
        config['model']['num_epochs'] = args.epochs
    if args.no_optimize:
        config['training']['optimize_hyperparams'] = False
    if args.quick:
        config['model']['num_epochs'] = 10
        config['training']['optimize_hyperparams'] = False
        config['training']['n_trials'] = 5
    
    logger.info("Starting intuition model training")
    logger.info(f"Configuration: {config}")
    
    try:
        # Step 1: Prepare data
        logger.info("Step 1: Preparing data...")
        data_path = config['data']['data_path']
        
        # Create sample data if needed
        raw_data = create_sample_data_if_needed(data_path)
        
        # Initialize data processor
        data_processor = MultiTimeframeDataProcessor(config)
        
        # Load and process data
        data_processor.load_data(data_path)
        multi_tf_data = data_processor.create_multi_timeframe_data()
        feature_matrix = data_processor.create_feature_matrix()
        
        logger.info(f"Feature matrix created: {feature_matrix.shape}")
        
        # Step 2: Generate labels
        logger.info("Step 2: Generating intuition labels...")
        label_generator = IntuitionLabelGenerator(config)
        labels = label_generator.generate_labels(feature_matrix)
        
        # Validate labels
        is_valid = label_generator.validate_labels(labels)
        if not is_valid:
            logger.error("Label validation failed!")
            return False
        
        logger.info(f"Labels generated: {labels.shape}")
        
        # Step 3: Train model
        logger.info("Step 3: Training intuition model...")
        trainer = ModelTrainer(config)
        
        # Prepare data for training
        train_loader, val_loader, test_loader = trainer.prepare_data(feature_matrix, labels)
        
        # Train model
        training_results = trainer.train_model(train_loader, val_loader, test_loader)
        
        # Step 4: Save results
        logger.info("Step 4: Saving results...")
        
        # Save training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history plot
        if config['training'].get('plot_history', True):
            plot_path = results_dir / "training_history.png"
            trainer.plot_training_history(str(plot_path))
        
        # Save model summary
        model_summary = trainer.get_model_summary()
        summary_path = results_dir / "model_summary.json"
        with open(summary_path, 'w') as f:
            import json
            json.dump(model_summary, f, indent=2, default=str)
        
        # Save configuration
        config_path = results_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print summary
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Model summary: {model_summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 