import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Generate labels based on future price movements.
    
    For each 15-second candle:
    1. Look ahead for the specified horizon
    2. If price moves up by threshold before down by threshold -> LongSuccess
    3. If price moves down by threshold first -> ShortSuccess (or LongFail)
    4. Save the move size and time taken
    
    Args:
        df: DataFrame with price data
        cfg: Configuration dictionary with label parameters
        
    Returns:
        DataFrame with added label columns
    """
    logger.info("Generating labels...")
    
    # Get parameters from config
    horizon = cfg['label']['horizon_minutes']
    threshold = cfg['label']['dollar_threshold']
    
    # Calculate number of bars to look ahead (4 bars per minute for 15s data)
    lookahead_bars = horizon * 4
    
    # Initialize label columns
    df = df.copy()
    df['label'] = 0  # 0: No signal, 1: LongSuccess, -1: ShortSuccess
    df['move_size'] = 0.0
    df['time_to_move'] = 0.0
    
    # For each bar
    for i in range(len(df) - lookahead_bars):
        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+lookahead_bars+1]
        
        # Calculate price changes
        price_changes = future_prices - current_price
        
        # Find first up and down moves
        up_move = price_changes[price_changes >= threshold]
        down_move = price_changes[price_changes <= -threshold]
        
        def index_to_minutes(idx):
            # If index is integer, just subtract
            if isinstance(idx, (int, np.integer)):
                return idx * (15/60)
            # If index is Timestamp, subtract and convert to minutes
            elif hasattr(idx, 'to_pydatetime'):
                delta = (idx - df.index[i])
                return delta.total_seconds() / 60
            # If index is Timedelta, convert to minutes
            elif hasattr(idx, 'total_seconds'):
                return idx.total_seconds() / 60
            else:
                return float(idx) * (15/60)
        
        if len(up_move) > 0 and len(down_move) > 0:
            # Both moves occurred
            if up_move.index[0] < down_move.index[0]:
                # Up move happened first
                df.loc[df.index[i], 'label'] = 1  # LongSuccess
                df.loc[df.index[i], 'move_size'] = up_move.iloc[0]
                df.loc[df.index[i], 'time_to_move'] = index_to_minutes(up_move.index[0])
            else:
                # Down move happened first
                df.loc[df.index[i], 'label'] = -1  # ShortSuccess
                value = down_move.iloc[0]
                df.iloc[i, df.columns.get_loc('move_size')] = value
                df.loc[df.index[i], 'time_to_move'] = index_to_minutes(down_move.index[0])
        elif len(up_move) > 0:
            # Only up move occurred
            df.loc[df.index[i], 'label'] = 1  # LongSuccess
            df.loc[df.index[i], 'move_size'] = up_move.iloc[0]
            df.loc[df.index[i], 'time_to_move'] = index_to_minutes(up_move.index[0])
        elif len(down_move) > 0:
            # Only down move occurred
            df.loc[df.index[i], 'label'] = -1  # ShortSuccess
            value = down_move.iloc[0]
            df.iloc[i, df.columns.get_loc('move_size')] = value
            df.loc[df.index[i], 'time_to_move'] = index_to_minutes(down_move.index[0])
    
    # Log label distribution
    label_counts = df['label'].value_counts()
    logger.info("\nLabel Distribution:")
    logger.info(f"LongSuccess (1): {label_counts.get(1, 0)}")
    logger.info(f"ShortSuccess (-1): {label_counts.get(-1, 0)}")
    logger.info(f"No Signal (0): {label_counts.get(0, 0)}")
    
    # Log move statistics
    logger.info("\nMove Statistics:")
    logger.info(f"Average move size: ${df[df['label'] != 0]['move_size'].mean():.2f}")
    avg_time_to_move = df[df['label'] != 0]['time_to_move'].mean()
    logger.info(f"Average time to move: {float(avg_time_to_move):.2f} minutes")
    
    return df 