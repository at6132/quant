import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import ta

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features."""
    # Price changes and returns
    df['returns_15s'] = df['close'].pct_change()
    df['log_returns_15s'] = np.log1p(df['returns_15s'])
    
    # Price position within candle
    df['high_low_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['close_open_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
    
    # Volatility measures
    df['true_range'] = df['high'] - df['low']
    df['dollar_volume'] = df['close'] * df['volume']
    
    # Rolling statistics (various windows)
    for window in [20, 50, 100, 200]:  # ~5min, 12.5min, 25min, 50min for 15s data
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'std_{window}'] = df['close'].rolling(window).std()
        df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
        
        # Z-score
        df[f'zscore_{window}'] = (df['close'] - df[f'sma_{window}']) / (df[f'std_{window}'] + 1e-10)
        
        # Volume ratio
        df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_sma_{window}'] + 1e-10)
    
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators."""
    # RSI at different periods
    for period in [14, 28, 56]:
        df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Rate of change
    for period in [10, 20, 40]:
        df[f'roc_{period}'] = ta.momentum.roc(df['close'], window=period)
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    # Time of day features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    
    # Session indicators (rough approximation)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['session_overlap'] = ((df['london_session'] == 1) & (df['ny_session'] == 1)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on indicator events."""
    # Find all indicator columns that might be events
    event_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in [
            'bull', 'bear', 'bos', 'choch', 'breaker', 'fvg', 
            'order_block', 'liquidity', 'sweep', 'trap', 'signal'
        ]):
            # Ensure it's binary
            if df[col].dropna().isin([0, 1]).all():
                event_cols.append(col)
    
    print(f"Found {len(event_cols)} event columns")
    
    # Time since last event for each event type
    for col in event_cols:
        # Forward fill to propagate the cumsum properly
        event_cumsum = df[col].fillna(0).cumsum()
        
        # Time since last event (in bars)
        df[f'{col}_bars_since'] = df.groupby(event_cumsum).cumcount()
        
        # Cap at reasonable value
        df[f'{col}_bars_since'] = df[f'{col}_bars_since'].clip(upper=1000)
        
        # Also create a "recent" flag
        for window in [4, 20, 100]:  # 1min, 5min, 25min
            df[f'{col}_recent_{window}'] = (df[f'{col}_bars_since'] <= window).astype(int)
    
    # Count events in rolling windows
    for window in [20, 100, 400]:  # 5min, 25min, 100min
        for col in event_cols[:10]:  # Limit to avoid too many features
            df[f'{col}_count_{window}'] = df[col].fillna(0).rolling(window).sum()
    
    return df

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market microstructure features."""
    # Bid-ask proxy (using high-low)
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    
    # Kyle's lambda (price impact)
    df['kyle_lambda'] = df['returns_15s'].abs() / (df['volume'] + 1e-10)
    
    # Amihud illiquidity
    df['amihud_illiq'] = df['returns_15s'].abs() / (df['dollar_volume'] + 1e-10)
    
    # Volume imbalance
    df['volume_imbalance'] = df['volume'].diff()
    
    # High-low volatility (Parkinson)
    df['parkinson_vol'] = np.sqrt(np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2)))
    
    # Rolling measures
    for window in [20, 100]:
        df[f'spread_proxy_ma_{window}'] = df['spread_proxy'].rolling(window).mean()
        df[f'kyle_lambda_ma_{window}'] = df['kyle_lambda'].rolling(window).mean()
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between indicators."""
    # RSI extremes with volume
    df['rsi_oversold_high_volume'] = (
        (df.get('rsi_14', 0) < 30) & 
        (df.get('volume_ratio_20', 0) > 1.5)
    ).astype(int)
    
    df['rsi_overbought_high_volume'] = (
        (df.get('rsi_14', 0) > 70) & 
        (df.get('volume_ratio_20', 0) > 1.5)
    ).astype(int)
    
    # MACD crosses with trend
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)
    
    # Multi-timeframe alignment
    # Look for alignment of trends across timeframes
    trend_cols = [col for col in df.columns if 'trend' in col.lower() or 'sma' in col.lower()]
    
    if len(trend_cols) >= 2:
        # Simple trend alignment: price above multiple SMAs
        sma_cols = [col for col in df.columns if col.startswith('sma_')]
        if sma_cols:
            df['trend_alignment_bull'] = 1
            df['trend_alignment_bear'] = 1
            
            for col in sma_cols[:3]:  # Use first 3 SMAs
                df['trend_alignment_bull'] &= (df['close'] > df[col])
                df['trend_alignment_bear'] &= (df['close'] < df[col])
            
            df['trend_alignment_bull'] = df['trend_alignment_bull'].astype(int)
            df['trend_alignment_bear'] = df['trend_alignment_bear'].astype(int)
    
    return df

def build_feature_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the complete feature matrix."""
    print("Building feature matrix...")
    
    # Keep a copy of original columns
    original_cols = df.columns.tolist()
    
    # Add all feature groups
    print("Adding price features...")
    df = add_price_features(df)
    
    print("Adding momentum features...")
    df = add_momentum_features(df)
    
    print("Adding time features...")
    df = add_time_features(df)
    
    print("Adding event features...")
    df = add_event_features(df)
    
    print("Adding microstructure features...")
    df = add_microstructure_features(df)
    
    print("Creating interaction features...")
    df = create_interaction_features(df)
    
    # Drop any infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill NaN values for most features
    feature_cols = [col for col in df.columns if col not in original_cols]
    df[feature_cols] = df[feature_cols].ffill(limit=10)
    
    # Fill remaining NaNs with 0
    df = df.fillna(0)
    
    print(f"Feature matrix shape: {df.shape}")
    print(f"Added {len(df.columns) - len(original_cols)} new features")
    
    return df