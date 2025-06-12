import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import time

class DataProcessor:
    def __init__(self):
        self.price_history = []
        self.timestamp_history = []
        self.feature_cache = {}
        self.last_update = time.time()
        
    def process_live_data(self, price: float) -> Optional[Dict[str, float]]:
        """Process live price data and generate features."""
        current_time = time.time()
        
        # Add new price
        self.price_history.append(price)
        self.timestamp_history.append(current_time)
        
        # Keep only last 1000 prices (about 4 hours of 15s data)
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
            self.timestamp_history = self.timestamp_history[-1000:]
        
        # Only process every 15 seconds
        if current_time - self.last_update < 15:
            return None
            
        self.last_update = current_time
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': self.timestamp_history,
            'close': self.price_history
        })
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility (20-period)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Calculate momentum (5-period)
        df['momentum'] = df['close'].pct_change(5)
        
        # Calculate RSI (14-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Calculate price distance from BB
        df['bb_distance'] = (df['close'] - df['bb_middle']) / df['bb_std']
        
        # Calculate volume profile (using price changes as proxy)
        df['volume_profile'] = df['returns'].abs().rolling(20).mean()
        
        # Get latest features
        latest = df.iloc[-1].to_dict()
        
        # Clean up NaN values
        features = {k: 0.0 if pd.isna(v) else float(v) for k, v in latest.items()}
        
        return features 