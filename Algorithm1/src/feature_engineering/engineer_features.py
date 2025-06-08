import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
from datetime import timedelta

logger = logging.getLogger(__name__)

def add_rolling_stats(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 100]) -> pd.DataFrame:
    """Add rolling statistics for price and volume columns."""
    logger.info("Adding rolling statistics...")
    
    # Price-based rolling stats
    for col in ['open', 'high', 'low', 'close']:
        for window in windows:
            # Rolling mean
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            
            # Rolling std
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
            
            # Rolling min/max
            df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
            df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
            
            # Rolling range
            df[f'{col}_roll_range_{window}'] = df[f'{col}_roll_max_{window}'] - df[f'{col}_roll_min_{window}']
    
    # Volume-based rolling stats
    for window in windows:
        df[f'volume_roll_mean_{window}'] = df['volume'].rolling(window=window).mean()
        df[f'volume_roll_std_{window}'] = df['volume'].rolling(window=window).std()
    
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

def add_cross_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features by combining different indicators."""
    logger.info("Adding cross-indicator features...")
    
    # Trend alignment features
    trend_cols = [col for col in df.columns if 'trend' in col.lower()]
    for i, col1 in enumerate(trend_cols):
        for col2 in trend_cols[i+1:]:
            df[f'trend_align_{col1}_{col2}'] = (df[col1] == df[col2]).astype(int)
    
    # Volume-price relationship
    df['volume_price_ratio'] = df['volume'] / df['close']
    df['volume_price_ratio_ma'] = df['volume_price_ratio'].rolling(window=20).mean()
    
    # Momentum features
    df['price_momentum'] = df['close'].pct_change(periods=5)
    df['volume_momentum'] = df['volume'].pct_change(periods=5)
    
    # Volatility features
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    logger.info("Adding time-based features...")
    
    # Extract time components
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    
    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to engineer all features."""
    logger.info("Starting feature engineering...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Add all feature groups
    df = add_rolling_stats(df)
    df = add_time_since_events(df)
    df = add_cross_indicator_features(df)
    df = add_time_features(df)
    
    # Drop any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill NaN values for most features
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    df[feature_cols] = df[feature_cols].ffill(limit=10)
    
    # Fill remaining NaNs with 0
    df = df.fillna(0)
    
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
    df = engineer_features(df)
    
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