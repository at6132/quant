import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def standardize_column_name(col: str) -> str:
    """Convert column names to snake_case and standardize common names."""
    # Convert to lowercase and replace spaces/dashes with underscores
    col = col.lower().replace(' ', '_').replace('-', '_')
    
    # Standardize common indicator names
    replacements = {
        'bullish': 'bull',
        'bearish': 'bear',
        'volume_imbalance': 'vol_imb',
        'breakerblock': 'breaker_block',
        'orderblock': 'order_block',
        'fairvaluegap': 'fvg',
        'change_of_character': 'choch',
        'break_of_structure': 'bos'
    }
    
    for old, new in replacements.items():
        col = col.replace(old, new)
    
    return col

def clean_dataframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Clean and standardize a single dataframe."""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert('UTC') if df.index.tz else df.index.tz_localize('UTC')
    
    # Sort by time
    df = df.sort_index()
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    # Standardize column names
    df.columns = [standardize_column_name(col) for col in df.columns]
    
    # Convert boolean columns to int8
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype('int8')
    
    # Forward fill gaps for non-price columns
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    non_price_cols = [col for col in df.columns if col not in price_cols]
    
    if non_price_cols:
        df[non_price_cols] = df[non_price_cols].ffill()
    
    # Drop columns with too many NaNs (>95%)
    nan_threshold = 0.95
    nan_ratio = df.isna().sum() / len(df)
    cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns from {timeframe} with >95% NaNs")
        df = df.drop(columns=cols_to_drop)
    
    return df

def load_frame(path: str, timeframe: str) -> pd.DataFrame:
    """Load and clean a single timeframe."""
    df = pd.read_parquet(path)
    df = clean_dataframe(df, timeframe)
    
    # Add timeframe suffix to non-price columns
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    df.columns = [
        col if col in price_cols else f"{col}__{timeframe}"
        for col in df.columns
    ]
    
    return df

def merge_timeframes(frames: Dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Merge multiple timeframes into a single dataframe aligned to 15-second candles."""
    # Start with the base 15-second timeframe
    base_tf = "15second"
    if base_tf not in frames:
        raise ValueError("15-second timeframe not found in data")
    
    merged = frames[base_tf].copy()
    
    # Merge higher timeframes
    for tf, df in frames.items():
        if tf == base_tf:
            continue
        
        # For higher timeframes, we need to align properly
        # Use forward fill to propagate higher TF values to lower TF
        higher_tf = df.copy()
        
        # Resample to 15-second frequency and forward fill
        higher_tf = higher_tf.resample('15s').ffill()
        
        # Shift by one period to avoid look-ahead bias
        non_price_cols = [col for col in higher_tf.columns 
                         if not any(pc in col for pc in ['open', 'high', 'low', 'close', 'volume'])]
        if non_price_cols:
            higher_tf[non_price_cols] = higher_tf[non_price_cols].shift(1)
        
        # Merge with the base dataframe
        merged = merged.join(higher_tf, how='left', rsuffix=f'_{tf}')
    
    # Final forward fill for any remaining NaNs
    merged = merged.ffill()
    
    return merged

def load_all_frames(cfg: dict) -> pd.DataFrame:
    """Load all timeframes and merge them."""
    base_dir = cfg["data_dir"]
    frames = {}
    
    print("Loading timeframes...")
    for tf in cfg["timeframes"]:
        # Convert timeframe format for file path
        tf_lower = tf.lower().replace(' ', '')
        fpath = os.path.join(base_dir, f"{tf_lower}.parquet")
        
        if not os.path.exists(fpath):
            # Try alternative naming
            fpath = os.path.join(base_dir, f"{tf}.parquet")
        
        if os.path.exists(fpath):
            print(f"Loading {tf} from {fpath}")
            frames[tf_lower] = load_frame(fpath, tf_lower)
        else:
            print(f"Warning: File not found for timeframe {tf} at {fpath}")
    
    if not frames:
        raise ValueError("No data files found!")
    
    print("Merging timeframes...")
    merged = merge_timeframes(frames, cfg)
    
    print(f"Final shape: {merged.shape}")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")
    
    return merged
