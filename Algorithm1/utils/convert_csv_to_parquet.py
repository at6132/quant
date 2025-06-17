import pandas as pd
import numpy as np
from pathlib import Path

# Read the CSV file
df = pd.read_csv('../btcusdt_15s_test.csv')

# Convert date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure UTC timezone
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

# Create processed_data directory if it doesn't exist
processed_data_dir = Path('processed_data')
processed_data_dir.mkdir(exist_ok=True)

# Define resampling rules for OHLCV data
def resample_ohlcv(df, freq):
    """Resample OHLCV data to a different frequency."""
    return df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

# Generate indicator columns for testing
def add_dummy_indicators(df):
    """Add dummy indicator columns for testing."""
    # Add some basic indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_sma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_sma + (bb_std * 2)
    df['bb_lower'] = bb_sma - (bb_std * 2)
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Add prefix to match expected format
    rename_dict = {}
    for col in df.columns:
        if col not in ['open', 'high', 'low', 'close', 'volume']:
            rename_dict[col] = f'indicator_{col}'
    
    df = df.rename(columns=rename_dict)
    
    # Add prefixed price columns (expected by the loader)
    df['pvsra_open'] = df['open']
    df['pvsra_high'] = df['high']
    df['pvsra_low'] = df['low']
    df['pvsra_close'] = df['close']
    df['pvsra_volume'] = df['volume']
    
    return df

# Save 15-second data with indicators
df_15s = add_dummy_indicators(df.copy())
df_15s.to_parquet(processed_data_dir / '15Second.parquet')
print(f"Saved 15Second.parquet: {len(df_15s)} rows")

# Create other timeframes
timeframe_map = {
    '1minute': '1T',
    '15minute': '15T',
    '1hour': '1H',
    '4hours': '4H'
}

for tf_name, freq in timeframe_map.items():
    # Resample to the desired frequency
    df_resampled = resample_ohlcv(df, freq)
    
    # Add indicators
    df_resampled = add_dummy_indicators(df_resampled)
    
    # Save to parquet
    df_resampled.to_parquet(processed_data_dir / f'{tf_name}.parquet')
    print(f"Saved {tf_name}.parquet: {len(df_resampled)} rows")

print("\nAll parquet files created successfully!")