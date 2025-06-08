import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict

from datetime import timedelta
import yaml

logger = logging.getLogger(__name__)

def add_rolling_stats(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add rolling statistics for price and volume."""
    logger.info("Adding rolling statistics...")
    
    # Get price columns from config
    price_cols = config['price_cols']
    
    # Add rolling statistics for each price column
    for window in config['feature_engineering']['rolling_windows']:
        for col in price_cols:
            # Mean
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            # Std
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
            # Min
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
            # Max
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
            # Range
            df[f'{col}_roll_range_{window}'] = df[f'{col}_roll_max_{window}'] - df[f'{col}_roll_min_{window}']
    
    # Add volume statistics if volume column exists
    if 'volume' in df.columns:
        for window in config['feature_engineering']['rolling_windows']:
            df[f'volume_roll_mean_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_roll_std_{window}'] = df['volume'].rolling(window=window).std()
            df[f'volume_roll_sum_{window}'] = df['volume'].rolling(window=window).sum()
    else:
        logger.warning("Volume column not found in data, skipping volume features")
    
    return df

def add_time_since_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-since-event counters for various indicators."""
    logger.info("Adding time-since-event counters...")
    
    # Find columns that might indicate events
    event_cols = [col for col in df.columns if any(x in col.lower() for x in 
                 ['break', 'cross', 'signal', 'alert', 'pattern', 'bos', 'smc'])]
    
    for col in event_cols:
        # Create a mask for when the event occurs
        event_mask = df[col] != 0
        
        # Initialize counter
        counter = 0
        time_since = []
        
        # Count bars since last event
        for is_event in event_mask:
            if is_event:
                counter = 0
            else:
                counter += 1
            time_since.append(counter)
        
        df[f'time_since_{col}'] = time_since
    
    return df

def add_cross_indicator_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add features based on cross-indicator relationships."""
    logger.info("Adding cross features...")
    
    # Cross indicator pairs from config
    pairs = config['feature_engineering']['cross_indicator_pairs']
    
    for pair in pairs:
        if all(ind in df.columns for ind in pair):
            # Ratio
            df[f'{pair[0]}_{pair[1]}_ratio'] = df[pair[0]] / df[pair[1]]
            # Difference
            df[f'{pair[0]}_{pair[1]}_diff'] = df[pair[0]] - df[pair[1]]
            # Cross signal
            df[f'{pair[0]}_{pair[1]}_cross'] = (df[pair[0]] > df[pair[1]]).astype(int)
    
    # Add volume-based features only if volume exists
    if 'volume' in df.columns:
        # Volume-price ratio
        df['volume_price_ratio'] = df['volume'] / df['close']
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(window=20).mean() / df['volume'].rolling(window=100).mean()
    else:
        logger.warning("Volume column not found, skipping volume-based cross features")
    
    return df

def add_time_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add time-based features."""
    if not config['feature_engineering']['time_features']:
        return df
        
    logger.info("Adding time features...")
    
    # Extract time components
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    return df

def engineer_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Main feature engineering function."""
    logger.info("Starting feature engineering...")
    
    # Add all feature types
    df = add_rolling_stats(df, config)
    df = add_time_since_events(df)
    df = add_cross_indicator_features(df, config)
    df = add_time_features(df, config)
    
    # Drop rows with NaN values only in critical columns
    critical_cols = config['price_cols']  # Only drop if price data is missing
    df = df.dropna(subset=critical_cols)
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load the joined data
    logger.info("Loading joined data...")
    df = pd.read_parquet("processed_data/processed_data.parquet")
    
    # Engineer features
    df = engineer_features(df, cfg)
    
    # Save feature matrix
    output_dir = Path("artefacts")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Saving feature matrix...")
    df.to_parquet(output_dir / "feature_matrix.parquet")
    logger.info(f"Feature matrix saved to {output_dir / 'feature_matrix.parquet'}")
    
    # Print feature summary
    logger.info("\nFeature Summary:")
    logger.info(f"Total features: {len(df.columns)}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main() 