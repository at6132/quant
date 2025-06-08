import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to snake_case and standardize common variations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Common column name mappings
    name_mappings = {
        # Price columns
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'VWAP': 'vwap',
        
        # Common indicator prefixes
        'RSI': 'rsi',
        'MACD': 'macd',
        'BB': 'bb',
        'ATR': 'atr',
        'EMA': 'ema',
        'SMA': 'sma',
        
        # Common suffixes
        '_Signal': '_signal',
        '_Histogram': '_hist',
        '_Upper': '_upper',
        '_Lower': '_lower',
        '_Middle': '_middle',
    }
    
    # Apply mappings
    new_columns = {}
    for col in df.columns:
        new_col = col
        for old, new in name_mappings.items():
            new_col = new_col.replace(old, new)
        # Convert to snake_case
        new_col = new_col.lower().replace(' ', '_')
        new_columns[col] = new_col
    
    df.columns = [new_columns[col] for col in df.columns]
    return df

def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean and text columns to int8 type.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with converted column types
    """
    df = df.copy()
    
    for col in df.columns:
        # Check if column is boolean or text
        if df[col].dtype == bool:
            df[col] = df[col].astype('int8')
        elif df[col].dtype == object:
            # Check if column contains only boolean-like values
            unique_vals = df[col].unique()
            if set(unique_vals).issubset({'True', 'False', True, False, 1, 0, '1', '0'}):
                df[col] = df[col].astype('int8')
    
    return df

def standardize_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Standardize column names and types across all timeframe DataFrames.
    
    Args:
        frames: Dictionary of timeframe DataFrames
        
    Returns:
        Dictionary of standardized DataFrames
    """
    standardized = {}
    for tf, df in frames.items():
        logger.info(f"Standardizing {tf} columns...")
        
        # Standardize column names
        df = standardize_column_names(df)
        
        # Convert boolean columns
        df = convert_boolean_columns(df)
        
        standardized[tf] = df
        logger.info(f"Standardized {tf} columns: {df.columns.tolist()}")
    
    return standardized 