import pandas as pd
import numpy as np
from numba import jit

@jit(nopython=True)
def find_first_breach(prices, start_idx, horizon_bars, entry_price, threshold):
    """
    Find which threshold is breached first within the horizon.
    Returns: (label, bars_to_breach, price_at_breach)
    - label: 1 for long success (+threshold first), -1 for short success (-threshold first), 0 for no breach
    - bars_to_breach: number of bars until breach (or horizon if no breach)
    - price_at_breach: the price when threshold was breached
    """
    upper_target = entry_price + threshold
    lower_target = entry_price - threshold
    
    end_idx = min(start_idx + horizon_bars, len(prices))
    
    for i in range(start_idx + 1, end_idx):
        current_price = prices[i]
        
        # Check if upper threshold breached first
        if current_price >= upper_target:
            return 1, i - start_idx, current_price
        
        # Check if lower threshold breached first  
        elif current_price <= lower_target:
            return -1, i - start_idx, current_price
    
    # No threshold breached within horizon
    return 0, horizon_bars, prices[min(end_idx - 1, len(prices) - 1)]

def label_future_move(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Label each row based on future price movement.
    
    Labels:
    - 1: Long success (price hits +threshold before -threshold)
    - -1: Short success (price hits -threshold before +threshold)  
    - 0: No significant move within horizon
    """
    df = df.copy()
    
    # Extract parameters
    horizon_minutes = cfg['label']['horizon_minutes']
    dollar_threshold = cfg['label']['dollar_threshold']
    
    # Calculate horizon in bars (assuming 15-second base timeframe)
    horizon_bars = horizon_minutes * 4  # 4 bars per minute for 15-second data
    
    # Get close prices as numpy array for speed
    close_prices = df['close'].values
    
    # Initialize label columns
    labels = np.zeros(len(df), dtype=np.int8)
    bars_to_breach = np.zeros(len(df), dtype=np.int32)
    price_at_breach = np.zeros(len(df), dtype=np.float64)
    
    # Label each point
    print(f"Labelling {len(df)} rows with ${dollar_threshold} threshold over {horizon_minutes} minutes...")
    
    for i in range(len(df) - 1):
        if i % 10000 == 0:
            print(f"Progress: {i}/{len(df)} ({i/len(df)*100:.1f}%)")
        
        entry_price = close_prices[i]
        label, bars, breach_price = find_first_breach(
            close_prices, i, horizon_bars, entry_price, dollar_threshold
        )
        
        labels[i] = label
        bars_to_breach[i] = bars
        price_at_breach[i] = breach_price
    
    # Add columns to dataframe
    df['label'] = labels
    df['bars_to_breach'] = bars_to_breach
    df['price_at_breach'] = price_at_breach
    df['move_size'] = df['price_at_breach'] - df['close']
    df['time_to_breach_minutes'] = df['bars_to_breach'] / 4  # Convert to minutes
    
    # Calculate label statistics
    label_counts = df['label'].value_counts()
    print("\nLabel distribution:")
    print(f"Long success (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
    print(f"Short success (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(df)*100:.2f}%)")
    print(f"No move (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.2f}%)")
    
    return df
