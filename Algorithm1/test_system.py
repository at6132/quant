"""
Test script for Algorithm1 trading system.
Creates sample data to test the full pipeline.
"""

import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime, timedelta

def create_sample_data(n_rows=10000):
    """Create sample BTC data for testing."""
    # Create 15-second intervals
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq='15s', tz='UTC')
    
    # Generate realistic BTC price data around $106,000
    np.random.seed(42)
    base_price = 106000
    returns = np.random.normal(0, 0.0002, n_rows)  # Small returns
    price = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame(index=timestamps)
    df['close'] = price
    df['open'] = df['close'].shift(1).fillna(base_price)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.0005, n_rows))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.0005, n_rows))
    df['volume'] = np.random.lognormal(10, 1, n_rows)
    
    # Add some sample indicator columns
    # Simple moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Trend indicators
    df['trend_up__15second'] = (df['close'] > df['sma_20']).astype(int)
    df['trend_down__15second'] = (df['close'] < df['sma_20']).astype(int)
    
    # Volume indicators
    df['volume_spike__15second'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
    
    # Some fake signals
    df['bull_signal__15second'] = (np.random.random(n_rows) < 0.05).astype(int)
    df['bear_signal__15second'] = (np.random.random(n_rows) < 0.05).astype(int)
    
    # Market structure
    df['bos_bull__15second'] = (np.random.random(n_rows) < 0.02).astype(int)
    df['bos_bear__15second'] = (np.random.random(n_rows) < 0.02).astype(int)
    
    return df

def create_higher_timeframes(df_15s):
    """Create higher timeframe data from 15-second data."""
    dfs = {}
    
    # 1 minute
    df_1m = df_15s.resample('1min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Add indicators
    df_1m['trend_up__1minute'] = (df_1m['close'] > df_1m['close'].rolling(20).mean()).astype(int)
    df_1m['bull_signal__1minute'] = (np.random.random(len(df_1m)) < 0.05).astype(int)
    dfs['1minute'] = df_1m
    
    # 15 minute
    df_15m = df_15s.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    df_15m['trend_up__15minute'] = (df_15m['close'] > df_15m['close'].rolling(20).mean()).astype(int)
    df_15m['bull_signal__15minute'] = (np.random.random(len(df_15m)) < 0.05).astype(int)
    dfs['15minute'] = df_15m
    
    # 1 hour
    df_1h = df_15s.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    df_1h['trend_up__1hour'] = (df_1h['close'] > df_1h['close'].rolling(20).mean()).astype(int)
    df_1h['bull_signal__1hour'] = (np.random.random(len(df_1h)) < 0.05).astype(int)
    dfs['1hour'] = df_1h
    
    # 4 hour
    df_4h = df_15s.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    df_4h['trend_up__4hours'] = (df_4h['close'] > df_4h['close'].rolling(10).mean()).astype(int)
    df_4h['bull_signal__4hours'] = (np.random.random(len(df_4h)) < 0.05).astype(int)
    dfs['4hours'] = df_4h
    
    return dfs

def main():
    print("Creating sample data for testing...")
    
    # Create data directory
    os.makedirs('processed_data', exist_ok=True)
    
    # Generate sample data
    df_15s = create_sample_data(10000)  # About 42 hours of data
    
    # Save 15-second data
    df_15s.to_parquet('processed_data/15second.parquet')
    print(f"Created 15-second data: {df_15s.shape}")
    
    # Create and save higher timeframes
    higher_tfs = create_higher_timeframes(df_15s)
    
    for tf, df in higher_tfs.items():
        df.to_parquet(f'processed_data/{tf}.parquet')
        print(f"Created {tf} data: {df.shape}")
    
    print("\nSample data created successfully!")
    print("\nYou can now run the main algorithm with:")
    print("python main.py -c config.yaml")
    
    # Test loading the data
    print("\nTesting data loader...")
    cfg = yaml.safe_load(open('config.yaml'))
    
    from quantsys.data_loader import load_all_frames
    try:
        merged = load_all_frames(cfg)
        print(f"\nSuccessfully loaded and merged data: {merged.shape}")
        print(f"Columns: {list(merged.columns[:10])}... (showing first 10)")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main()