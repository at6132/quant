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
    Extract base price columns from prefixed columns.
    For example, if we have 'pvsra_open', 'pvsra_high', etc., we'll use those as our base columns.
    
    Args:
        df: Input DataFrame with prefixed columns
        required_cols: List of required base column names
        
    Returns:
        DataFrame with base columns extracted
    """
    # Find the first occurrence of each required column with any prefix
    base_cols = {}
    for req_col in required_cols:
        # Look for columns ending with the required name
        matches = [col for col in df.columns if col.endswith(f"_{req_col}")]
        if matches:
            # Use the first match
            base_cols[req_col] = matches[0]
            logger.info(f"Using {matches[0]} as base column for {req_col}")
        else:
            raise ValueError(f"Could not find any column ending with _{req_col}")
    
    # Create new DataFrame with base columns
    result = pd.DataFrame(index=df.index)
    for base_name, prefixed_name in base_cols.items():
        result[base_name] = df[prefixed_name]
    
    return result

def load_all_frames(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Load all timeframe parquet files and ensure they have proper UTC timestamps.
    
    Args:
        cfg: Configuration dictionary containing data_dir and timeframes
        
    Returns:
        Dictionary mapping timeframe names to DataFrames
    """
    data_dir = Path(cfg['data_dir'])
    timeframes = cfg['timeframes']
    
    frames = {}
    for tf in timeframes:
        file_path = data_dir / f"{tf}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {file_path}")
            
        logger.info(f"Loading {tf} data...")
        df = pd.read_parquet(file_path)
        
        # Normalize column names to lowercase
        df = normalize_column_names(df)
        
        # Extract base price columns
        try:
            base_df = extract_base_columns(df, cfg['price_cols'])
            df = pd.concat([base_df, df], axis=1)
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