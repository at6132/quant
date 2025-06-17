import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from label_generator import LabelGenerator
import ta
from typing import Dict, List
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    logger.info("Adding technical indicators...")
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    
    # Momentum
    df['momentum'] = ta.momentum.ROCIndicator(df['close']).roc()
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price ratios
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    # CCI
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    
    # MFI
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
    
    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # ADX
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # Ichimoku
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to the dataframe."""
    logger.info("Adding time features...")
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Extract time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    return df

def process_timeframe(data_path: str, config: Dict) -> pd.DataFrame:
    """Process a single timeframe of data."""
    logger.info(f"Processing data from {data_path}")
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add time features
    df = add_time_features(df)
    
    # Generate labels
    label_gen = LabelGenerator(config)
    df = label_gen.generate_action_labels(df)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Process each timeframe
    processed_data = {}
    for timeframe in config['timeframes']:
        # Fix path handling
        data_path = Path(config['data']['raw_dir'].replace('\\', '/')) / config['data']['timeframe_files'][timeframe]
        if data_path.exists():
            logger.info(f"Processing {timeframe} data...")
            df = process_timeframe(str(data_path), config)
            processed_data[timeframe] = df
            
            # Save processed data
            output_path = Path(config['data']['processed_dir']) / f"{timeframe}_processed.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path)
            logger.info(f"Saved processed {timeframe} data to {output_path}")
            
            # Clear memory
            del df
            gc.collect()
        else:
            logger.warning(f"Data file not found: {data_path}")
    
    logger.info("Data processing completed!")

if __name__ == "__main__":
    main() 