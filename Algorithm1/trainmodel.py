#!/usr/bin/env python3
"""
Comprehensive Training Script for Intelligent Trading System

This script:
1. Downloads 30 days of 1-second candle data from Binance
2. Processes the data through multi-timeframe analysis
3. Generates labels and features
4. Trains the intuition learning model
5. Extracts trading rules
6. Saves all artifacts to the artefacts/ directory
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from data.data_processor import MultiTimeframeDataProcessor
from models.intuition_model import IntuitionLearningModel
from features.label_generator import LabelGenerator
from features.feature_engineer import FeatureEngineer
from utils.logger import get_logger
from utils.config_validator import ConfigValidator
from utils.performance_monitor import PerformanceMonitor

# Binance API imports
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("Warning: python-binance not installed. Install with: pip install python-binance")
    Client = None

logger = get_logger(__name__)

class ModelTrainer:
    """
    Comprehensive model trainer for the intelligent trading system.
    """
    
    def __init__(self, config_path: str = "config/intelligent_config.yaml"):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.artifacts_dir = Path("artefacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = None
        self.label_generator = None
        self.feature_engineer = None
        self.model = None
        self.performance_monitor = PerformanceMonitor()
        
        # Training artifacts
        self.training_artifacts = {
            'model': None,
            'rules': None,
            'feature_importance': None,
            'training_metrics': None,
            'data_info': None,
            'config': None
        }
        
        logger.info("ModelTrainer initialized")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    async def download_binance_data(self, symbol: str = "BTCUSDT", days: int = 30) -> pd.DataFrame:
        """
        Download 1-second candle data from Binance's public data hub.
        
        Args:
            symbol: Trading symbol
            days: Number of days to download
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading {days} days of {symbol} data from Binance data hub")
            
            import requests
            import zipfile
            import io
            from datetime import datetime, timedelta
            
            all_data = []
            successful_downloads = 0
            
            # Calculate date range (go back from yesterday to avoid today's incomplete data)
            end_date = datetime.now() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=days-1)
            
            # Download data for each day
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Construct URL for Binance data hub
                url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1s/{symbol}-1s-{date_str}.zip"
                
                try:
                    logger.info(f"Downloading data for {date_str}")
                    
                    # Download the ZIP file
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Extract CSV from ZIP
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        # Get the CSV file name (should be the same as the ZIP but with .csv extension)
                        csv_filename = f"{symbol}-1s-{date_str}.csv"
                        
                        if csv_filename in zip_file.namelist():
                            with zip_file.open(csv_filename) as csv_file:
                                # Read CSV data
                                daily_data = pd.read_csv(
                                    csv_file,
                                    header=None,
                                    names=[
                                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                        'close_time', 'quote_asset_volume', 'number_of_trades',
                                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                                    ]
                                )
                                
                                all_data.append(daily_data)
                                successful_downloads += 1
                                logger.info(f"Downloaded {len(daily_data)} rows for {date_str}")
                        else:
                            logger.warning(f"CSV file {csv_filename} not found in ZIP for {date_str}")
                            
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        logger.warning(f"No data available for {date_str} (404)")
                    else:
                        logger.warning(f"HTTP error downloading data for {date_str}: {str(e)}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to download data for {date_str}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error processing data for {date_str}: {str(e)}")
                
                current_date += timedelta(days=1)
            
            if not all_data:
                raise ValueError(f"No data could be downloaded from Binance data hub for {symbol} over {days} days")
            
            if successful_downloads < days * 0.5:  # Require at least 50% of days
                logger.warning(f"Only downloaded {successful_downloads}/{days} days of data")
            
            # Combine all daily data
            df = pd.concat(all_data, ignore_index=True)
            
            # Debug: Check timestamp values
            logger.info(f"Sample timestamp values: {df['timestamp'].head()}")
            logger.info(f"Timestamp dtype: {df['timestamp'].dtype}")
            logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Convert timestamp to datetime (handle different timestamp formats)
            try:
                # First, check if timestamps are in milliseconds, seconds, or microseconds
                sample_timestamp = df['timestamp'].iloc[0]
                
                # If timestamp is very large (> 1e15), it's likely in microseconds
                # If timestamp is very large (> 1e12), it's likely in milliseconds
                # If timestamp is smaller, it might be in seconds
                if sample_timestamp > 1e15:
                    # Timestamp is in microseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
                elif sample_timestamp > 1e12:
                    # Timestamp is in milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    # Timestamp is in seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    
            except (OverflowError, ValueError) as e:
                logger.warning(f"Timestamp conversion error: {str(e)}")
                # Try alternative conversion methods
                try:
                    # Try treating as microseconds first
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', errors='coerce')
                except:
                    try:
                        # Try treating as seconds
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    except:
                        try:
                            # Try treating as milliseconds
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                        except:
                            # Last resort: try parsing as string
                            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Drop rows with invalid timestamps
                df = df.dropna(subset=['timestamp'])
                if len(df) == 0:
                    raise ValueError("No valid timestamps found in downloaded data")
            
            df.set_index('timestamp', inplace=True)
            
            # Convert price and volume columns to float
            price_cols = ['open', 'high', 'low', 'close']
            volume_cols = ['volume', 'quote_asset_volume']
            
            for col in price_cols + volume_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Ensure UTC timezone
            df.index = df.index.tz_localize('UTC')
            
            # Sort by time
            df = df.sort_index()
            
            # Remove any NaN values
            df = df.dropna()
            
            logger.info(f"Downloaded {len(df)} total rows of {symbol} data from {successful_downloads} days")
            logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data from Binance data hub: {str(e)}")
            raise ValueError(f"Failed to download real data from Binance. Error: {str(e)}")
            
    def _generate_synthetic_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing when real data is not available.
        
        Args:
            symbol: Trading symbol (for naming)
            days: Number of days to generate
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        logger.info(f"Generating {days} days of synthetic {symbol} data")
        
        # Calculate number of 1-second intervals
        total_seconds = days * 24 * 60 * 60
        
        # Generate time index
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=total_seconds)
        time_index = pd.date_range(start=start_time, end=end_time, freq='1S', tz='UTC')
        
        # Generate synthetic price data (random walk with trend)
        np.random.seed(42)  # For reproducible results
        
        # Start with a base price
        base_price = 50000.0  # BTC-like price
        
        # Generate price movements
        returns = np.random.normal(0, 0.0001, len(time_index))  # Small random returns
        trend = np.linspace(0, 0.001, len(time_index))  # Small upward trend
        returns += trend
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(time_index, prices)):
            # Add some noise to create realistic OHLC
            noise = np.random.normal(0, price * 0.0001)
            
            open_price = price + noise
            high_price = max(open_price, price + abs(noise) * 1.5)
            low_price = min(open_price, price - abs(noise) * 1.5)
            close_price = price + noise * 0.5
            
            # Generate volume (correlated with price movement)
            volume = np.random.exponential(1000) * (1 + abs(returns[i]) * 100)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=time_index)
        
        logger.info(f"Generated {len(df)} rows of synthetic {symbol} data")
        logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        return df
        
    async def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data through the complete pipeline.
        
        Args:
            raw_data: Raw OHLCV data
            
        Returns:
            Processed feature matrix
        """
        logger.info("Starting data processing pipeline")
        
        # Initialize data processor
        self.data_processor = MultiTimeframeDataProcessor(self.config)
        await self.data_processor.initialize()
        
        # Load the raw data into the processor
        self.data_processor.raw_data = raw_data
        
        # Create multi-timeframe data
        multi_tf_data = self.data_processor.create_multi_timeframe_data()
        
        # Compute indicators
        indicator_data = self.data_processor.compute_indicators()
        
        # Create feature matrix
        feature_matrix = self.data_processor.create_feature_matrix()
        
        logger.info(f"Created feature matrix with {len(feature_matrix)} rows and {len(feature_matrix.columns)} features")
        
        return feature_matrix
        
    async def generate_labels(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading labels from the feature matrix.
        
        Args:
            feature_matrix: Feature matrix
            
        Returns:
            Label matrix with multi-dimensional intuition labels
        """
        logger.info("Generating intuition-based trading labels")
        
        # Initialize label generator
        self.label_generator = LabelGenerator(self.config)
        await self.label_generator.initialize()
        
        # Generate intuition labels (multi-dimensional)
        labels = self.label_generator.generate_intuition_labels(feature_matrix)
        
        # Extract only the intuition label columns for training
        intuition_columns = [col for col in labels.columns if col.startswith('intuition_')]
        training_labels = labels[intuition_columns].copy()
        
        # Rename columns to match model expectations (remove 'intuition_' prefix)
        training_labels.columns = [col.replace('intuition_', '') for col in training_labels.columns]
        
        logger.info(f"Generated intuition labels with {len(training_labels)} rows and {len(training_labels.columns)} label types")
        logger.info(f"Label columns: {list(training_labels.columns)}")
        
        return training_labels
        
    async def engineer_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply advanced feature engineering.
        
        Args:
            feature_matrix: Initial feature matrix
            
        Returns:
            Enhanced feature matrix
        """
        logger.info("Applying feature engineering")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config)
        await self.feature_engineer.initialize()
        
        # Engineer features
        enhanced_features = self.feature_engineer.engineer_features(feature_matrix)
        
        logger.info(f"Enhanced features: {len(enhanced_features)} rows and {len(enhanced_features.columns)} features")
        
        return enhanced_features
        
    async def train_model(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        """
        Train the intuition learning model.
        
        Args:
            features: Feature matrix
            labels: Label matrix
            
        Returns:
            Training results and metrics
        """
        logger.info("Training intuition learning model")
        
        # Initialize model
        self.model = IntuitionLearningModel(self.config)
        await self.model.initialize()
        
        # Train model
        training_results = self.model.train_model(features, labels)
        
        logger.info("Model training completed")
        
        return training_results
        
    def extract_rules(self, model, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        """
        Extract trading rules from the trained model.
        
        Args:
            model: Trained model
            features: Feature matrix
            labels: Label matrix
            
        Returns:
            Extracted rules
        """
        logger.info("Extracting trading rules")
        
        # Get feature importance
        feature_importance = model.get_feature_importance(features)
        
        # Extract decision rules (simplified version)
        rules = {
            'feature_importance': feature_importance,
            'entry_thresholds': {
                'min_probability': self.config['trading']['min_entry_probability'],
                'min_confidence': self.config['trading']['min_entry_confidence']
            },
            'risk_parameters': {
                'max_position_size': self.config['risk']['max_position_size'],
                'max_drawdown': self.config['risk']['max_drawdown']
            },
            'extracted_patterns': self._extract_patterns(features, labels)
        }
        
        logger.info("Trading rules extracted")
        
        return rules
        
    def _extract_patterns(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        """
        Extract trading patterns from the data.
        
        Args:
            features: Feature matrix
            labels: Label matrix
            
        Returns:
            Extracted patterns
        """
        patterns = {}
        
        # Analyze entry patterns
        entry_labels = labels.get('entry_signal', pd.Series(0, index=labels.index))
        entry_indices = entry_labels[entry_labels == 1].index
        
        if len(entry_indices) > 0:
            # Get features at entry points
            entry_features = features.loc[entry_indices]
            
            # Calculate average values for key features
            key_features = ['rsi_14', 'macd', 'bb_position', 'volume_ma_ratio']
            available_features = [f for f in key_features if f in entry_features.columns]
            
            if available_features:
                patterns['entry_conditions'] = {
                    feature: {
                        'mean': float(entry_features[feature].mean()),
                        'std': float(entry_features[feature].std()),
                        'min': float(entry_features[feature].min()),
                        'max': float(entry_features[feature].max())
                    }
                    for feature in available_features
                }
        
        # Analyze market conditions
        patterns['market_conditions'] = {
            'total_samples': len(features),
            'entry_signals': int(entry_labels.sum()) if 'entry_signal' in labels.columns else 0,
            'avg_volatility': float(features.get('atr_14', pd.Series(0)).mean()) if 'atr_14' in features.columns else 0,
            'avg_volume': float(features.get('volume', pd.Series(0)).mean()) if 'volume' in features.columns else 0
        }
        
        return patterns
        
    def save_artifacts(self, model, rules: Dict, training_metrics: Dict, data_info: Dict):
        """
        Save all training artifacts to the artefacts directory.
        
        Args:
            model: Trained model
            rules: Extracted rules
            training_metrics: Training metrics
            data_info: Data information
        """
        logger.info("Saving training artifacts")
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.artifacts_dir / f"intuition_model_{timestamp}.pkl"
        model.save_model(str(model_path))
        logger.info(f"Saved model to {model_path}")
        
        # Save rules
        rules_path = self.artifacts_dir / f"trading_rules_{timestamp}.json"
        with open(rules_path, 'w') as f:
            json.dump(rules, f, indent=2, default=str)
        logger.info(f"Saved rules to {rules_path}")
        
        # Save training metrics
        metrics_path = self.artifacts_dir / f"training_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save data info
        data_info_path = self.artifacts_dir / f"data_info_{timestamp}.json"
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=2, default=str)
        logger.info(f"Saved data info to {data_info_path}")
        
        # Save configuration
        config_path = self.artifacts_dir / f"training_config_{timestamp}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved config to {config_path}")
        
        # Create latest symlinks
        self._create_latest_symlinks(timestamp)
        
        # Save training summary
        summary = {
            'timestamp': timestamp,
            'model_path': str(model_path),
            'rules_path': str(rules_path),
            'metrics_path': str(metrics_path),
            'data_info_path': str(data_info_path),
            'config_path': str(config_path),
            'training_duration': self.performance_monitor.get_elapsed_time(),
            'data_samples': data_info.get('total_samples', 0),
            'features_count': data_info.get('features_count', 0),
            'model_performance': training_metrics.get('final_metrics', {})
        }
        
        summary_path = self.artifacts_dir / f"training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved training summary to {summary_path}")
        
        # Update latest summary
        latest_summary_path = self.artifacts_dir / "latest_training_summary.json"
        with open(latest_summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info("All artifacts saved successfully")
        
    def _create_latest_symlinks(self, timestamp: str):
        """Create symlinks to latest artifacts."""
        try:
            # Create latest symlinks (if on Unix-like system)
            if os.name != 'nt':  # Not Windows
                latest_files = [
                    f"intuition_model_{timestamp}.pkl",
                    f"trading_rules_{timestamp}.json",
                    f"training_metrics_{timestamp}.json",
                    f"data_info_{timestamp}.json",
                    f"training_config_{timestamp}.yaml"
                ]
                
                for file in latest_files:
                    source = self.artifacts_dir / file
                    target = self.artifacts_dir / f"latest_{file.split('_', 1)[1]}"
                    
                    if target.exists():
                        target.unlink()
                    target.symlink_to(source.name)
                    
        except Exception as e:
            logger.warning(f"Could not create symlinks: {str(e)}")
            
    async def run_training_pipeline(self, symbol: str = "BTCUSDT", days: int = 30):
        """
        Run the complete training pipeline.
        
        Args:
            symbol: Trading symbol
            days: Number of days of data to use
        """
        try:
            logger.info("Starting complete training pipeline")
            self.performance_monitor.start()
            
            # Step 1: Download data
            logger.info("Step 1/6: Downloading data from Binance (0%)")
            raw_data = await self.download_binance_data(symbol, days)
            logger.info("Step 1/6: Downloading data from Binance (100%)")
            
            # Step 2: Process data
            logger.info("Step 2/6: Processing data (0%)")
            feature_matrix = await self.process_data(raw_data)
            logger.info("Step 2/6: Processing data (100%)")
            
            # Step 3: Generate labels
            logger.info("Step 3/6: Generating labels (0%)")
            labels = await self.generate_labels(feature_matrix)
            logger.info("Step 3/6: Generating labels (100%)")
            
            # Step 4: Engineer features
            logger.info("Step 4/6: Engineering features (0%)")
            enhanced_features = await self.engineer_features(feature_matrix)
            logger.info("Step 4/6: Engineering features (100%)")
            
            # Step 5: Train model
            logger.info("Step 5/6: Training model (0%)")
            training_results = await self.train_model(enhanced_features, labels)
            logger.info("Step 5/6: Training model (100%)")
            
            # Step 6: Extract rules
            logger.info("Step 6/6: Extracting rules (0%)")
            rules = self.extract_rules(self.model, enhanced_features, labels)
            logger.info("Step 6/6: Extracting rules (100%)")
            
            # Save artifacts
            logger.info("Saving training artifacts...")
            data_info = {
                'symbol': symbol,
                'days': days,
                'data_points': len(raw_data),
                'features': len(enhanced_features.columns),
                'labels': len(labels.columns),
                'data_range': f"{raw_data.index[0]} to {raw_data.index[-1]}",
                'price_range': f"${raw_data['low'].min():.2f} - ${raw_data['high'].max():.2f}"
            }
            
            self.save_artifacts(self.model, rules, training_results['final_metrics'], data_info)
            
            # Print summary
            self._print_training_summary(data_info, training_results, rules)
            
            # Log completion
            elapsed_time = self.performance_monitor.get_elapsed_time()
            logger.info(f"Training pipeline completed successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
            
    def _print_training_summary(self, data_info: Dict, training_results: Dict, rules: Dict):
        """Print a summary of the training results."""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        print(f"Symbol: {data_info['symbol']}")
        print(f"Data Period: {data_info['days']} days")
        print(f"Total Samples: {data_info['data_points']:,}")
        print(f"Features: {data_info['features']}")
        print(f"Labels: {data_info['labels']}")
        print(f"Processing Time: {self.performance_monitor.get_elapsed_time():.2f} seconds")
        
        if 'final_metrics' in training_results:
            metrics = training_results['final_metrics']
            print(f"\nModel Performance:")
            print(f"  Training Loss: {metrics.get('train_loss', 'N/A'):.4f}")
            print(f"  Validation Loss: {metrics.get('val_loss', 'N/A'):.4f}")
            print(f"  Training Accuracy: {metrics.get('train_accuracy', 'N/A'):.4f}")
            print(f"  Validation Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")
        
        if 'entry_conditions' in rules.get('extracted_patterns', {}):
            patterns = rules['extracted_patterns']['entry_conditions']
            print(f"\nExtracted Entry Patterns:")
            for feature, stats in patterns.items():
                print(f"  {feature}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"\nArtifacts saved to: {self.artifacts_dir}")
        print("="*80)

async def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the intelligent trading model")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download (default: 30)")
    parser.add_argument("--config", default="config/intelligent_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    # Run training pipeline
    await trainer.run_training_pipeline(args.symbol, args.days)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Run the training pipeline
    asyncio.run(main()) 