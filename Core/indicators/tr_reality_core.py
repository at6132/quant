"""
TR Reality Core Indicator - Simplified Implementation
"""

import pandas as pd
import numpy as np


def tr_reality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified TR Reality Core indicator implementation.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with TR Reality indicators
    """
    try:
        if df.empty:
            result = pd.DataFrame(index=df.index)
            result['tr_signal'] = 0
            result['tr_strength'] = 0.0
            return result
        
        result = pd.DataFrame(index=df.index)
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate ATR
        atr_period = 14
        atr = true_range.rolling(window=atr_period, min_periods=1).mean()
        
        # Simple trend detection
        price_change = df['close'] - df['close'].shift(1)
        trend_strength = price_change / atr
        
        # Generate signals based on trend strength
        result['tr_signal'] = np.where(trend_strength > 0.5, 1,
                                      np.where(trend_strength < -0.5, -1, 0))
        
        result['tr_strength'] = trend_strength.fillna(0)
        
        return result
        
    except Exception as e:
        print(f"Error in tr_reality: {e}")
        result = pd.DataFrame(index=df.index)
        result['tr_signal'] = 0
        result['tr_strength'] = 0.0
        return result