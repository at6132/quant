import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def drop_high_nan_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Drop columns that have more than threshold% NaN values.
    
    Args:
        df: Input DataFrame
        threshold: Maximum allowed percentage of NaN values (default: 0.95)
        
    Returns:
        DataFrame with high-NaN columns removed
    """
    # Calculate NaN percentage for each column
    nan_percentages = df.isna().mean()
    
    # Identify columns to drop
    cols_to_drop = nan_percentages[nan_percentages > threshold].index.tolist()
    
    if cols_to_drop:
        logger.warning(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% NaN values")
        logger.debug(f"Dropped columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df

def drop_nan_columns_in_frames(frames: Dict[str, pd.DataFrame], threshold: float = 0.95) -> Dict[str, pd.DataFrame]:
    """
    Drop high-NaN columns from all timeframe DataFrames.
    
    Args:
        frames: Dictionary of timeframe DataFrames
        threshold: Maximum allowed percentage of NaN values
        
    Returns:
        Dictionary of cleaned DataFrames
    """
    cleaned_frames = {}
    for tf, df in frames.items():
        logger.info(f"Cleaning {tf} data...")
        
        # Drop high-NaN columns
        df = drop_high_nan_columns(df, threshold)
        
        cleaned_frames[tf] = df
        logger.info(f"Cleaned {tf} data: {len(df.columns)} columns remaining")
    
    return cleaned_frames 