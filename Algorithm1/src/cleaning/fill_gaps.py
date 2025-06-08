import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def detect_missing_intervals(df: pd.DataFrame, expected_freq: str) -> pd.DataFrame:
    """
    Detect missing intervals in the DataFrame index.
    
    Args:
        df: Input DataFrame
        expected_freq: Expected frequency (e.g., '15S', '1T', '15T', '1H', '4H')
        
    Returns:
        DataFrame with missing intervals filled with NaN
    """
    # Create a complete index
    start_time = df.index.min()
    end_time = df.index.max()
    complete_index = pd.date_range(start=start_time, end=end_time, freq=expected_freq, tz='UTC')
    
    # Reindex the DataFrame
    df_reindexed = df.reindex(complete_index)
    
    # Count missing values
    missing_count = df_reindexed.isna().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing intervals")
    
    return df_reindexed

def forward_fill_ohlcv(df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
    """
    Forward fill OHLCV data except for price columns.
    
    Args:
        df: Input DataFrame
        price_cols: List of price column names to exclude from forward fill
        
    Returns:
        DataFrame with forward-filled values
    """
    df = df.copy()
    
    # Identify columns to forward fill
    fill_cols = [col for col in df.columns if col not in price_cols]
    
    # Forward fill non-price columns
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna(method='ffill')
    
    return df

def fill_gaps_in_frames(frames: Dict[str, pd.DataFrame], price_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fill gaps in all timeframe DataFrames.
    
    Args:
        frames: Dictionary of timeframe DataFrames
        price_cols: List of price column names
        
    Returns:
        Dictionary of gap-filled DataFrames
    """
    # Expected frequencies for each timeframe
    expected_freqs = {
        '15Second': '15S',
        '1minute': '1T',
        '15minute': '15T',
        '1hour': '1H',
        '4hours': '4H'
    }
    
    filled_frames = {}
    for tf, df in frames.items():
        logger.info(f"Filling gaps in {tf} data...")
        
        # Detect and fill missing intervals
        if tf in expected_freqs:
            df = detect_missing_intervals(df, expected_freqs[tf])
        
        # Forward fill non-price columns
        df = forward_fill_ohlcv(df, price_cols)
        
        filled_frames[tf] = df
        logger.info(f"Filled gaps in {tf} data: {len(df)} rows")
    
    return filled_frames 