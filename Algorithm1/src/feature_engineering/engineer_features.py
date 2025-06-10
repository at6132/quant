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
            # Get the column data
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]  # Take first column if it's a DataFrame
            
            # Mean
            df[f'{col}_roll_mean_{window}'] = col_data.rolling(window=window).mean()
            # Std
            df[f'{col}_roll_std_{window}'] = col_data.rolling(window=window).std()
            # Min
            df[f'{col}_roll_min_{window}'] = col_data.rolling(window=window).min()
            # Max
            df[f'{col}_roll_max_{window}'] = col_data.rolling(window=window).max()
            # Range
            df[f'{col}_roll_range_{window}'] = df[f'{col}_roll_max_{window}'] - df[f'{col}_roll_min_{window}']
            # Momentum
            df[f'{col}_momentum_{window}'] = col_data.pct_change(window)
            # Volatility
            df[f'{col}_volatility_{window}'] = df[f'{col}_roll_std_{window}'] / df[f'{col}_roll_mean_{window}']
    
    # Add volume statistics if volume column exists
    if 'volume' in df.columns:
        volume_data = df['volume']
        if isinstance(volume_data, pd.DataFrame):
            volume_data = volume_data.iloc[:, 0]  # Take first column if it's a DataFrame
            
        for window in config['feature_engineering']['rolling_windows']:
            df[f'volume_roll_mean_{window}'] = volume_data.rolling(window=window).mean()
            df[f'volume_roll_std_{window}'] = volume_data.rolling(window=window).std()
            df[f'volume_roll_sum_{window}'] = volume_data.rolling(window=window).sum()
            df[f'volume_momentum_{window}'] = volume_data.pct_change(window)
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
    
    # Print index info before deduplication
    print(f"[DEBUG] Index before deduplication: {df.index[:10]}")
    print(f"[DEBUG] Number of duplicate index labels: {df.index.duplicated().sum()}")
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    # Only reset index if not already a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)
    print(f"[DEBUG] Index after deduplication and reset: {df.index[:10]}")
    print(f"[DEBUG] Number of duplicate index labels after: {df.index.duplicated().sum()}")
    
    # Debug print for volume and close
    print(f"[DEBUG] df['volume'] type: {type(df['volume'])}, shape: {getattr(df['volume'], 'shape', None)}")
    print(f"[DEBUG] df['close'] type: {type(df['close'])}, shape: {getattr(df['close'], 'shape', None)}")
    
    # If close is a DataFrame, use only the first column
    close = df['close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Add price-based features
    df['price_range'] = df['high'] - df['low']
    df['price_range_pct'] = df['price_range'] / df['close']
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_size_pct'] = df['body_size'] / df['close']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Add volume-based features
    if 'volume' in df.columns:
        df['volume_price_ratio'] = df['volume'].to_numpy() / close.to_numpy()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_trend'] = df['volume'].rolling(window=20).mean() / df['volume'].rolling(window=100).mean()
        df['volume_volatility'] = df['volume'].rolling(window=20).std() / df['volume'].rolling(window=20).mean()
    
    # Cross indicator pairs from config
    pairs = config['feature_engineering']['cross_indicator_pairs']
    
    for pair in pairs:
        if all(ind in df.columns for ind in pair):
            # Ratio
            df[f'{pair[0]}_{pair[1]}_ratio'] = df[pair[0]].to_numpy() / df[pair[1]].to_numpy()
            # Difference
            df[f'{pair[0]}_{pair[1]}_diff'] = df[pair[0]].to_numpy() - df[pair[1]].to_numpy()
            # Cross signal
            df[f'{pair[0]}_{pair[1]}_cross'] = (df[pair[0]].to_numpy() > df[pair[1]].to_numpy()).astype(int)
            # Rolling correlation
            df[f'{pair[0]}_{pair[1]}_corr'] = df[pair[0]].rolling(20).corr(df[pair[1]])
    
    return df

def add_time_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add time-based features."""
    if not config['feature_engineering']['time_features']:
        return df
        
    logger.info("Adding time features...")
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame index is not datetime and no 'date' column found.")
    
    # Extract time components
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['second'] = df.index.second
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['second_sin'] = np.sin(2 * np.pi * df['second'] / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['second'] / 60)
    
    # Add time-based volatility features
    df['hourly_volatility'] = df.groupby('hour')['close'].transform(lambda x: x.pct_change().std())
    df['minute_volatility'] = df.groupby('minute')['close'].transform(lambda x: x.pct_change().std())
    
    return df

def engineer_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Main feature engineering function."""
    logger.info("Starting feature engineering...")
    
    # Ensure index is datetime at the very start
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame index is not datetime and no 'date' column found at start of engineer_features.")
    df = df[~df.index.duplicated(keep='first')]
    
    # Add all feature types
    df = add_rolling_stats(df, config)
    df = add_time_since_events(df)
    df = add_cross_indicator_features(df, config)
    df = add_time_features(df, config)
    
    # Drop rows with NaN values only in critical columns
    critical_cols = config['price_cols']  # Only drop if price data is missing
    df = df.dropna(subset=critical_cols)
    
    # Fill remaining NaN values with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
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