import pandas as pd
import os
from log import logger

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process_data(self, symbol):
        # Load data
        df = pd.read_parquet(os.path.join(self.config['data']['raw_dir'], f"{symbol.lower()}_raw.parquet"))
        logger.info(f"Loaded data for {symbol}")

        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility (20-period rolling standard deviation of returns)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
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
        
        # Calculate price momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        
        # Calculate volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate price range features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Calculate time-based features
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Ensure all columns are numeric
        for col in df.columns:
            if col not in ['timestamp', 'hour', 'day_of_week']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any remaining NaN values
        df = df.dropna()
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        
        # Ensure timestamp is in the correct format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Save processed data
        output_path = os.path.join(self.config['data']['processed_dir'], f"{symbol.lower()}_processed.parquet")
        df.to_parquet(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return df 