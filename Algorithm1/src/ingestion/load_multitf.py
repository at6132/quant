import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    return df

def extract_base_columns(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Extract base price columns from the DataFrame.
    Handles both simple column names (e.g. 'open') and suffixed names (e.g. '15Second_open').
    
    Args:
        df: Input DataFrame
        required_cols: List of required column names (e.g. ['open', 'high', 'low', 'close'])
        
    Returns:
        DataFrame with only the base price columns
    """
    base_cols = {}
    
    for req_col in required_cols:
        # Try exact match first
        if req_col in df.columns:
            base_cols[req_col] = df[req_col]
        else:
            # Try finding column ending with _req_col
            matching_cols = [col for col in df.columns if col.endswith(f"_{req_col}")]
            if matching_cols:
                base_cols[req_col] = df[matching_cols[0]]
            else:
                raise ValueError(f"Could not find column '{req_col}' or any column ending with '_{req_col}'")
    
    return pd.DataFrame(base_cols, index=df.index)

def load_all_frames(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Load all timeframe files and ensure they have proper UTC timestamps.
    
    Args:
        cfg: Configuration dictionary containing data_dir and timeframes
        
    Returns:
        Dictionary mapping timeframe names to DataFrames
    """
    data_dir = Path(cfg['data_dir'])
    timeframes = cfg['timeframes']
    
    frames = {}
    for tf in timeframes:
        # Try both parquet and csv files
        parquet_path = data_dir / f"{tf}.parquet"
        csv_path = data_dir / f"btcusdt_15s_test.csv" if tf == "15Second" else data_dir / f"{tf}.csv"
        
        if parquet_path.exists():
            file_path = parquet_path
            df = pd.read_parquet(file_path)
        elif csv_path.exists():
            file_path = csv_path
            df = pd.read_csv(file_path)
            # Set index to date column and ensure it's datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            # Remove any duplicate indices
            df = df[~df.index.duplicated(keep='first')]
        else:
            raise FileNotFoundError(f"Missing data file: {parquet_path} or {csv_path}")
            
        logger.info(f"Loading {tf} data...")
        
        # Normalize column names to lowercase
        df = normalize_column_names(df)
        
        # Extract base price columns
        try:
            base_df = extract_base_columns(df, cfg['price_cols'])
            df = pd.concat([base_df, df], axis=1)
            # Remove duplicate columns, keeping the first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
        except ValueError as e:
            logger.error(f"Available columns in {tf}: {df.columns.tolist()}")
            raise
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != 'UTC':
            df.index = df.index.tz_convert('UTC')
            
        # Validate required columns (case-insensitive)
        required_cols = [col.lower() for col in cfg['price_cols']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Available columns in {tf}: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns in {tf}: {missing_cols}")
            
        frames[tf] = df
        logger.info(f"Loaded {tf} data: {len(df)} rows, {len(df.columns)} columns")
        
    return frames

def validate_frames(frames: Dict[str, pd.DataFrame]) -> None:
    """
    Validate that all frames have consistent data types and no major issues.
    
    Args:
        frames: Dictionary of timeframe DataFrames
    """
    for tf, df in frames.items():
        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"{tf} has NaN values in columns: {nan_cols}")
            
        # Check for infinite values
        inf_cols = df.columns[df.isin([float('inf'), float('-inf')]).any()].tolist()
        if inf_cols:
            logger.warning(f"{tf} has infinite values in columns: {inf_cols}")
            
        # Check for duplicate indices
        if df.index.duplicated().any():
            logger.warning(f"{tf} has duplicate timestamps")
            
        # Check for gaps in index
        expected_freq = {
            '15Second': '15S',
            '1minute': '1T',
            '15minute': '15T',
            '1hour': '1H',
            '4hours': '4H'
        }
        if tf in expected_freq:
            gaps = df.index.to_series().diff().dt.total_seconds() != pd.Timedelta(expected_freq[tf]).total_seconds()
            if gaps.any():
                logger.warning(f"{tf} has gaps in timestamp sequence") 