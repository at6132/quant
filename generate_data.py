import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate realistic Bitcoin price data with significant movements
np.random.seed(42)
start_time = datetime(2024, 1, 1, 0, 0, 0)
num_periods = 10000  # Generate 10,000 15-second periods (about 41 hours of data)

# Generate timestamps
timestamps = [start_time + timedelta(seconds=15*i) for i in range(num_periods)]

# Generate realistic Bitcoin price movements with volatility
base_price = 42000
prices = []
current_price = base_price

# Parameters for realistic price generation
volatility = 0.002  # 0.2% volatility per 15 seconds
trend_strength = 0.00001
pump_probability = 0.001  # 0.1% chance of pump
dump_probability = 0.001  # 0.1% chance of dump

for i in range(num_periods):
    # Long-term trend
    trend = trend_strength * np.sin(i / 1000) * base_price
    
    # Random walk component
    random_change = np.random.normal(0, volatility * current_price)
    
    # Occasional large moves (pumps and dumps)
    if np.random.random() < pump_probability:
        random_change += current_price * np.random.uniform(0.003, 0.01)  # 0.3% to 1% pump
    elif np.random.random() < dump_probability:
        random_change -= current_price * np.random.uniform(0.003, 0.01)  # 0.3% to 1% dump
    
    # Update price
    current_price = max(current_price + random_change + trend, base_price * 0.8)
    current_price = min(current_price, base_price * 1.2)  # Keep within reasonable bounds
    prices.append(current_price)

# Create OHLCV data
data = []
for i in range(num_periods):
    close = prices[i]
    
    # Generate realistic OHLC values
    high_spread = abs(np.random.normal(0, 0.0002)) * close  # 0.02% average spread
    low_spread = abs(np.random.normal(0, 0.0002)) * close
    
    high = close + high_spread
    low = close - low_spread
    
    # Open is previous close with small gap
    if i > 0:
        gap = np.random.normal(0, 0.00005) * prices[i-1]  # Small gap
        open_price = prices[i-1] + gap
    else:
        open_price = close + np.random.normal(0, 0.0001) * close
    
    # Volume with some correlation to price movement
    price_change = abs(close - open_price) / open_price if open_price > 0 else 0
    base_volume = 100
    volume = base_volume * (1 + price_change * 50) * np.random.lognormal(0, 0.3)
    
    data.append({
        'date': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
        'open': round(open_price, 2),
        'high': round(high, 2),
        'low': round(low, 2),
        'close': round(close, 2),
        'volume': round(volume, 2)
    })

# Create DataFrame and save
df = pd.DataFrame(data)

# Save to parent directory as expected by config
df.to_csv('/workspace/15Second.csv', index=False)
df.to_parquet('/workspace/15Second.parquet', index=False)

print(f'Generated {len(df)} rows of Bitcoin 15-second data')
print(f'Date range: {df.date.iloc[0]} to {df.date.iloc[-1]}')
print(f'Price range: ${df.close.min():.2f} - ${df.close.max():.2f}')
print(f'Average volume: {df.volume.mean():.2f}')
print(f'Number of significant moves (>0.3%): {sum(abs(df.close.pct_change()) > 0.003)}')
print('\nFiles saved:')
print('- /workspace/15Second.csv')
print('- /workspace/15Second.parquet')