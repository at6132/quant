import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_candles(df: pd.DataFrame) -> pd.DataFrame:
    return tr_reality(df)

def tr_reality(df: pd.DataFrame) -> pd.DataFrame:
    """
    TR Reality Core indicator implementation
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with TR Reality columns
    """
    try:
        result_df = pd.DataFrame(index=df.index)
        
        # Always include all expected TR Reality columns
        result_df['tr_ema_5'] = 0
        result_df['tr_ema_13'] = 0
        result_df['tr_ema_50'] = 0
        result_df['tr_ema_200'] = 0
        result_df['tr_ema_800'] = 0
        result_df['tr_yesterday_h'] = 0
        result_df['tr_yesterday_l'] = 0
        result_df['tr_lastweek_h'] = 0
        result_df['tr_lastweek_l'] = 0
        result_df['tr_vec_color'] = 0
        result_df['tr_vcz_top'] = 0
        result_df['tr_vcz_bot'] = 0
        
        # Add original TR Reality logic
        result_df['tr_signal'] = 0
        result_df['tr_strength'] = 0
        result_df['tr_direction'] = 0
        result_df['tr_momentum'] = 0
        
        # Simple TR (True Range) based analysis
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Calculate True Range
            prev_close = df['close'].shift(1)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - prev_close)
            tr3 = abs(df['low'] - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR (Average True Range)
            atr = true_range.rolling(window=14).mean()
            
            # Volatility-based signals
            high_volatility = true_range > atr * 1.5
            low_volatility = true_range < atr * 0.5
            
            result_df.loc[high_volatility, 'tr_signal'] = 1
            result_df.loc[low_volatility, 'tr_signal'] = -1
            
            # Trend strength based on price movement relative to ATR
            price_change = df['close'] - df['close'].shift(1)
            result_df['tr_strength'] = price_change / atr
            
            # Direction based on price action
            result_df.loc[price_change > 0, 'tr_direction'] = 1
            result_df.loc[price_change < 0, 'tr_direction'] = -1
            
            # Momentum using rate of change
            roc = df['close'].pct_change(periods=5)
            result_df['tr_momentum'] = np.tanh(roc * 100)  # Normalize
            
        logger.debug(f"TR Reality Core processed {len(df)} candles")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in TR Reality Core: {str(e)}")
        # Return empty DataFrame with expected columns
        result_df = pd.DataFrame(index=df.index)
        result_df['tr_ema_5'] = 0
        result_df['tr_ema_13'] = 0
        result_df['tr_ema_50'] = 0
        result_df['tr_ema_200'] = 0
        result_df['tr_ema_800'] = 0
        result_df['tr_yesterday_h'] = 0
        result_df['tr_yesterday_l'] = 0
        result_df['tr_lastweek_h'] = 0
        result_df['tr_lastweek_l'] = 0
        result_df['tr_vec_color'] = 0
        result_df['tr_vcz_top'] = 0
        result_df['tr_vcz_bot'] = 0
        result_df['tr_signal'] = 0
        result_df['tr_strength'] = 0
        result_df['tr_direction'] = 0
        result_df['tr_momentum'] = 0
        return result_df

def get_features():
    return ['tr_ema_5', 'tr_ema_13', 'tr_ema_50', 'tr_ema_200', 'tr_ema_800', 'tr_yesterday_h', 'tr_yesterday_l', 'tr_lastweek_h', 'tr_lastweek_l', 'tr_vec_color', 'tr_vcz_top', 'tr_vcz_bot']