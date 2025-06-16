"""
BB OB Engine Indicator - Simplified Implementation
"""

import pandas as pd
import numpy as np


def process_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified BB OB Engine indicator implementation.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with BB OB indicators
    """
    try:
        if df.empty:
            result = pd.DataFrame(index=df.index)
            result['bb_signal'] = 0
            result['bb_upper'] = 0.0
            result['bb_lower'] = 0.0
            result['bb_middle'] = 0.0
            result['ob_signal'] = 0
            return result
        
        result = pd.DataFrame(index=df.index)
        
        # Calculate Bollinger Bands
        period = 20
        std_dev = 2
        
        bb_middle = df['close'].rolling(window=period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=period, min_periods=1).std()
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        
        # Generate BB signals
        bb_signal = np.where(df['close'] > bb_upper, 1,
                            np.where(df['close'] < bb_lower, -1, 0))
        
        # Simple order block detection
        # Look for significant price rejection at key levels
        high_wick = df['high'] - np.maximum(df['open'], df['close'])
        low_wick = np.minimum(df['open'], df['close']) - df['low']
        body_size = abs(df['close'] - df['open'])
        
        # Order block signal when wicks are large relative to body
        wick_threshold = 0.7
        ob_signal = np.where((high_wick > body_size * wick_threshold) & 
                            (df['close'] > bb_middle), -1,  # Bearish OB
                           np.where((low_wick > body_size * wick_threshold) & 
                                   (df['close'] < bb_middle), 1, 0))  # Bullish OB
        
        result['bb_signal'] = bb_signal
        result['bb_upper'] = bb_upper
        result['bb_lower'] = bb_lower
        result['bb_middle'] = bb_middle
        result['ob_signal'] = ob_signal
        
        return result
        
    except Exception as e:
        print(f"Error in bb_ob_engine: {e}")
        result = pd.DataFrame(index=df.index)
        result['bb_signal'] = 0
        result['bb_upper'] = 0.0
        result['bb_lower'] = 0.0
        result['bb_middle'] = 0.0
        result['ob_signal'] = 0
        return result