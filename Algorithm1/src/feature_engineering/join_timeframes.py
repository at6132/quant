import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def merge_higher_timeframes(base_df: pd.DataFrame, 
                          higher_tf_dfs: Dict[str, pd.DataFrame],
                          lookback: bool = True) -> pd.DataFrame:
    """
    Merge higher timeframe data onto the base timeframe DataFrame.
    
    Args:
        base_df: Base timeframe DataFrame (e.g., 15s)
        higher_tf_dfs: Dictionary of higher timeframe DataFrames
        lookback: If True, use previous bar's data to avoid look-ahead bias
        
    Returns:
        DataFrame with merged higher timeframe data
    """
    result_df = base_df.copy()
    
    # Sort timeframes by their period length
    tf_order = {
        '15Second': 15,
        '1minute': 60,
        '15minute': 900,
        '1hour': 3600,
        '4hours': 14400
    }
    
    sorted_tfs = sorted(higher_tf_dfs.keys(), 
                       key=lambda x: tf_order.get(x, float('inf')))
    
    for tf in sorted_tfs:
        logger.info(f"Merging {tf} data...")
        higher_df = higher_tf_dfs[tf]
        
        # Get the suffix for this timeframe
        suffix = f"_{tf.lower().replace('minute', 'm').replace('hour', 'h')}"
        
        # Merge with the base DataFrame
        if lookback:
            # Use previous bar's data to avoid look-ahead bias
            higher_df = higher_df.shift(1)
        
        # Merge using the index
        result_df = result_df.join(higher_df, rsuffix=suffix)
        
        logger.info(f"Merged {tf} data: {len(result_df.columns)} total columns")
    
    return result_df

def stream_merge_timeframes(frames: Dict[str, pd.DataFrame],
                          base_tf: str = '15Second',
                          lookback: bool = True,
                          chunk_size: int = 10000) -> pd.DataFrame:
    """
    Stream merge timeframes to handle large datasets efficiently.
    
    Args:
        frames: Dictionary of timeframe DataFrames
        base_tf: Base timeframe to merge onto
        lookback: If True, use previous bar's data
        chunk_size: Number of rows to process at once
        
    Returns:
        Merged DataFrame
    """
    if base_tf not in frames:
        raise ValueError(f"Base timeframe {base_tf} not found in frames")
    
    base_df = frames[base_tf]
    higher_tf_dfs = {tf: df for tf, df in frames.items() if tf != base_tf}
    
    # Initialize result DataFrame
    result_dfs = []
    
    # Process in chunks
    for i in tqdm(range(0, len(base_df), chunk_size), desc="Merging timeframes"):
        chunk = base_df.iloc[i:i + chunk_size]
        merged_chunk = merge_higher_timeframes(chunk, higher_tf_dfs, lookback)
        result_dfs.append(merged_chunk)
    
    # Combine all chunks
    result_df = pd.concat(result_dfs)
    
    return result_df

def join_timeframes(frames: Dict[str, pd.DataFrame],
                   base_tf: str = '15Second',
                   lookback: bool = True) -> pd.DataFrame:
    """
    Join all timeframes onto the base timeframe.
    
    Args:
        frames: Dictionary of timeframe DataFrames
        base_tf: Base timeframe to merge onto
        lookback: If True, use previous bar's data
        
    Returns:
        Merged DataFrame
    """
    logger.info(f"Joining timeframes onto {base_tf}...")
    
    # Use stream merge for large datasets
    result_df = stream_merge_timeframes(frames, base_tf, lookback)
    
    logger.info(f"Joined timeframes: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df 