"""
SMC Core Indicator Implementation
Simplified version to handle DataFrame indexing properly and avoid scalar variable errors.
"""

import pandas as pd
import numpy as np
from typing import Generator, Dict, Any


def process_candles(df: pd.DataFrame) -> Generator[Dict[str, Any], None, None]:
    """
    Process candles for SMC Core indicator.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'time']
             Must have integer index starting from 0.
    
    Yields:
        Dict containing SMC indicator values for each bar
    """
    if df.empty:
        return
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Ensure proper integer index
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    
    # Initialize variables for SMC calculations
    n_bars = len(df)
    
    for i in range(n_bars):
        try:
            # Safely access DataFrame values using .iat for scalar access
            open_price = df.iat[i, df.columns.get_loc('open')]
            high_price = df.iat[i, df.columns.get_loc('high')]
            low_price = df.iat[i, df.columns.get_loc('low')]
            close_price = df.iat[i, df.columns.get_loc('close')]
            
            # Simple SMC calculations (simplified for demonstration)
            # In a full implementation, these would be more complex SMC calculations
            
            # Basic trend detection
            if i > 0:
                prev_close = df.iat[i-1, df.columns.get_loc('close')]
                internal_trend = 1 if close_price > prev_close else -1
            else:
                internal_trend = 0
            
            # Swing trend (using a simple 5-period lookback)
            lookback = min(5, i)
            if lookback > 0:
                recent_closes = df.iloc[max(0, i-lookback):i+1]['close'].values
                swing_trend = 1 if len(recent_closes) > 1 and recent_closes[-1] > recent_closes[0] else -1
            else:
                swing_trend = 0
            
            # Simple pivot detection
            is_pivot_high = False
            is_pivot_low = False
            
            if i >= 2 and i < n_bars - 2:
                # Check if current bar is a pivot high
                prev_high = df.iat[i-1, df.columns.get_loc('high')]
                next_high = df.iat[i+1, df.columns.get_loc('high')]
                is_pivot_high = high_price > prev_high and high_price > next_high
                
                # Check if current bar is a pivot low
                prev_low = df.iat[i-1, df.columns.get_loc('low')]
                next_low = df.iat[i+1, df.columns.get_loc('low')]
                is_pivot_low = low_price < prev_low and low_price < next_low
            
            # Order block detection (simplified)
            is_order_block = False
            if i >= 3:
                # Simple order block: significant price movement followed by retracement
                three_bars_ago_close = df.iat[i-3, df.columns.get_loc('close')]
                price_change = abs(close_price - three_bars_ago_close) / three_bars_ago_close
                is_order_block = price_change > 0.01  # 1% threshold
            
            # Yield the results for this bar
            yield {
                'internal_trend': internal_trend,
                'swing_trend': swing_trend,
                'pivot_high': 1 if is_pivot_high else 0,
                'pivot_low': 1 if is_pivot_low else 0,
                'order_block': 1 if is_order_block else 0,
                'range': high_price - low_price,
                'body_size': abs(close_price - open_price),
                'is_bullish': 1 if close_price > open_price else 0,
                'upper_wick': high_price - max(open_price, close_price),
                'lower_wick': min(open_price, close_price) - low_price
            }
            
        except (IndexError, KeyError) as e:
            # Handle indexing errors gracefully
            print(f"SMC Core: Index error at bar {i}: {e}")
            # Yield default values to maintain consistency
            yield {
                'internal_trend': 0,
                'swing_trend': 0,
                'pivot_high': 0,
                'pivot_low': 0,
                'order_block': 0,
                'range': 0.0,
                'body_size': 0.0,
                'is_bullish': 0,
                'upper_wick': 0.0,
                'lower_wick': 0.0
            }
        except Exception as e:
            print(f"SMC Core: Unexpected error at bar {i}: {e}")
            # Yield default values to maintain consistency
            yield {
                'internal_trend': 0,
                'swing_trend': 0,
                'pivot_high': 0,
                'pivot_low': 0,
                'order_block': 0,
                'range': 0.0,
                'body_size': 0.0,
                'is_bullish': 0,
                'upper_wick': 0.0,
                'lower_wick': 0.0
            }


def prepare_smc_core_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and process SMC Core data, handling index alignment properly.
    
    Args:
        df: Input DataFrame with OHLC data
        
    Returns:
        DataFrame with SMC Core indicators added
    """
    try:
        # Ensure we have the basic columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame must contain OHLC columns")
        
        # Create a copy for SMC processing with proper index
        smc_df = df[['open', 'high', 'low', 'close']].copy()
        
        # Add time column if it doesn't exist (SMC Core might need it)
        if 'time' not in smc_df.columns:
            smc_df['time'] = smc_df.index.astype(np.int64) // 10**6  # Convert to ms
        
        # Reset index to integer for SMC processing
        original_index = smc_df.index
        smc_df = smc_df.reset_index(drop=True)
        
        # Process through SMC Core
        smc_results = list(process_candles(smc_df))
        
        if not smc_results:
            # Return empty DataFrame with expected columns if no results
            empty_cols = ['internal_trend', 'swing_trend', 'pivot_high', 'pivot_low', 
                         'order_block', 'range', 'body_size', 'is_bullish', 
                         'upper_wick', 'lower_wick']
            empty_df = pd.DataFrame(index=original_index, columns=empty_cols)
            empty_df = empty_df.fillna(0)
            return empty_df
        
        # Convert results to DataFrame with original index
        result_df = pd.DataFrame(smc_results, index=original_index)
        
        return result_df
        
    except Exception as e:
        print(f"Error in prepare_smc_core_data: {e}")
        # Return empty DataFrame with expected columns
        empty_cols = ['internal_trend', 'swing_trend', 'pivot_high', 'pivot_low', 
                     'order_block', 'range', 'body_size', 'is_bullish', 
                     'upper_wick', 'lower_wick']
        empty_df = pd.DataFrame(index=df.index, columns=empty_cols)
        empty_df = empty_df.fillna(0)
        return empty_df