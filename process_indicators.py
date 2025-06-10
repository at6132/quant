import pandas as pd
import numpy as np
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc
import psutil
import logging
import time
import traceback
import json
from pathlib import Path

# ────────────────────────────────────────────────────────────────
#  In-house libs
# ────────────────────────────────────────────────────────────────
from Core.indicators.breaker_signals   import breaker_signals, BreakerEngine, EVENTS
from Core.indicators.liquidity_swings  import liquidity_swings
from Core.indicators.tr_reality_core   import tr_reality
from Core.indicators.smc_core          import process_candles   as smc_process
from Core.indicators.pvsra_vs          import pvsra_vs
from Core.indicators.sessions          import build_session_table
from Core.indicators.ict_sm_trades     import run               as ict_sm_trades_run
from Core.indicators.bb_ob_engine      import process_candles   as bb_process
from Core.indicators.IT_Foundation     import process           as it_foundation_process

# ────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'process_indicators_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _safe_join(base: pd.DataFrame,
               new: pd.DataFrame | None,
               prefix: str = "") -> pd.DataFrame:
    """
    Joins *new* onto *base* only if *new* is a non-empty DataFrame.
    """
    if new is None:
        return base

    # Handle generator output
    if hasattr(new, '__iter__') and not isinstance(new, pd.DataFrame):
        new = pd.DataFrame(list(new))

    if new.empty:
        return base

    if prefix:
        new = new.add_prefix(prefix)

    # Re-index to the base index once, fill forward if indicator is lower tf
    new = new.reindex(base.index, method="ffill")
    return base.join(new)

def _safe_run(fn, name: str, *args, **kwargs):
    """
    Wrap an indicator call so that a hard failure does not kill the run.
    """
    try:
        out = fn(*args, **kwargs)
        # Handle generator output
        if hasattr(out, '__iter__') and not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(list(out))
        return out
    except Exception as e:
        logger.warning("Indicator %s failed: %s", name, e, exc_info=True)
        return None

def get_chunk_size(df_size):
    """Calculate appropriate chunk size based on available memory"""
    available_memory = psutil.virtual_memory().available
    # Use 0.5% of available memory, but not more than 1,000 rows
    chunk_size = min(1000, max(500, int(df_size * 0.005)))
    return chunk_size

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def resample_dataframe(df, timeframe):
    """Resample DataFrame to a different timeframe"""
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    return resampled

def validate_dataframe(df, name=""):
    """Validate DataFrame has required columns and data types"""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Check columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} DataFrame missing columns: {missing_cols}")
    
    # Check data types
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{name} DataFrame column '{col}' is not numeric")
    
    # Check for NaN values
    nan_cols = df[required_cols].columns[df[required_cols].isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"{name} DataFrame has NaN values in columns: {nan_cols}")
    
    return True

def process_it_foundation(df, max_retries=3):
    """Process IT_Foundation separately with retries"""
    logger.info("Processing IT_Foundation separately...")
    log_memory_usage()
    
    # Validate input data
    try:
        validate_dataframe(df, "Input")
    except ValueError as e:
        logger.error(f"Data validation failed: {str(e)}")
        return pd.DataFrame()
    
    # Create multi-timeframe data
    logger.info("Creating multi-timeframe data...")
    try:
        frames = {
            '15s': df,
            '1m': resample_dataframe(df, '1min'),
            '5m': resample_dataframe(df, '5min'),
            '15m': resample_dataframe(df, '15min'),
            '1h': resample_dataframe(df, '1h')
        }
        
        # Validate all timeframes
        for tf, frame in frames.items():
            validate_dataframe(frame, f"{tf} timeframe")
            
    except Exception as e:
        logger.error(f"Error creating multi-timeframe data: {str(e)}")
        return pd.DataFrame()
    
    # Process entire dataset at once
    for retry in range(max_retries):
        try:
            logger.info(f"Processing IT_Foundation (attempt {retry + 1})")
            result = it_foundation_process(frames)
            if not result.empty:
                logger.info("IT_Foundation processing successful")
                return result
            else:
                logger.warning("IT_Foundation returned empty result")
        except Exception as e:
            if retry == max_retries - 1:
                logger.error(f"Failed to process IT_Foundation after {max_retries} retries: {str(e)}")
            else:
                logger.warning(f"Retry {retry + 1} for IT_Foundation")
                time.sleep(1)  # Wait before retry
    
    return pd.DataFrame()

def process_ict_trades(df, start_idx):
    """Process ICT trades for a chunk of data"""
    try:
        # Ensure we have a copy of the data
        df_copy = df.copy()
        
        # Process ICT trades
        logger.info(f"Processing ICT trades for chunk starting at index {start_idx}")
        result = ict_sm_trades_run(df_copy)
        
        # Clean up
        del df_copy
        gc.collect()
        
        return result
    except Exception as e:
        logger.error(f"Error in ICT trades processing: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def process_chunk(chunk_df, start_idx, full_df):
    """Process a chunk of data with all indicators"""
    try:
        # Ensure index is DatetimeIndex and UTC-localized
        if not isinstance(chunk_df.index, pd.DatetimeIndex):
            chunk_df.index = pd.to_datetime(chunk_df.index)
        if chunk_df.index.tz is None:
            chunk_df.index = chunk_df.index.tz_localize('UTC')
            
        # Initialize result DataFrame with chunk's index
        result_df = pd.DataFrame(index=chunk_df.index)
        
        # List to store any errors
        errors = []
        
        try:
            # 1. Process PVSRA
            logger.info(f"Processing PVSRA")
            pvsra_result = pvsra_vs(chunk_df)
            if not pvsra_result.empty:
                logger.info(f"PVSRA added columns: {pvsra_result.columns.tolist()}")
                pvsra_result = pvsra_result.add_prefix('pvsra_')
                result_df = pd.concat([result_df, pvsra_result], axis=1)
        except Exception as e:
            errors.append(f"Error in PVSRA: {str(e)}")
            logger.error(f"PVSRA error: {str(e)}")
            
        try:
            # 2. Process Sessions
            logger.info(f"Processing Sessions")
            sessions_result = build_session_table(chunk_df[['close']].copy())
            if not sessions_result.empty:
                logger.info(f"Sessions added columns: {sessions_result.columns.tolist()}")
                sessions_result = sessions_result.add_prefix('sessions_')
                result_df = pd.concat([result_df, sessions_result], axis=1)
        except Exception as e:
            errors.append(f"Error in Sessions: {str(e)}")
            logger.error(f"Sessions error: {str(e)}")
            
        try:
            # 3. Process ICT SM Trades
            logger.info(f"Processing ICT SM Trades")
            ict_result = ict_sm_trades_run(chunk_df)
            if not ict_result.empty:
                logger.info(f"ICT SM Trades added columns: {ict_result.columns.tolist()}")
                ict_result = ict_result.add_prefix('ict_')
                result_df = pd.concat([result_df, ict_result], axis=1)
        except Exception as e:
            errors.append(f"Error in ICT SM Trades: {str(e)}")
            logger.error(f"ICT SM Trades error: {str(e)}")
            
        try:
            # 4. Process Breaker Signals
            logger.info(f"Processing Breaker Signals")
            # Ensure required columns exist and are float type
            breaker_df = chunk_df[['open', 'high', 'low', 'close']].copy()
            breaker_df = breaker_df.astype(float)
            
            # Convert to numpy arrays for breaker signals
            o = breaker_df['open'].values
            h = breaker_df['high'].values
            l = breaker_df['low'].values
            c = breaker_df['close'].values
            
            # Create a new instance of BreakerEngine with default parameters
            eng = BreakerEngine()
            out = np.zeros((len(breaker_df), 22), dtype=bool)
            
            # Process each bar
            for i in range(len(breaker_df)):
                fired = eng.on_bar(i, o, h, l, c)  # Pass the full arrays
                out[i] = fired
            
            # Create the result DataFrame with the original datetime index
            breaker_result = pd.DataFrame(out, index=breaker_df.index, columns=EVENTS)
            if not breaker_result.empty:
                logger.info(f"Breaker Signals added columns: {breaker_result.columns.tolist()}")
                breaker_result = breaker_result.add_prefix('breaker_')
                result_df = pd.concat([result_df, breaker_result], axis=1)
        except Exception as e:
            errors.append(f"Error in Breaker Signals: {str(e)}")
            logger.error(f"Breaker Signals error: {str(e)}")
            
        try:
            # 5. Process Liquidity Swings
            logger.info(f"Processing Liquidity Swings")
            liq_df = chunk_df[['open', 'high', 'low', 'close', 'volume']].copy()
            liq_result = liquidity_swings(liq_df)
            if not liq_result.empty:
                logger.info(f"Liquidity Swings added columns: {liq_result.columns.tolist()}")
                liq_result = liq_result.add_prefix('liq_')
                result_df = pd.concat([result_df, liq_result], axis=1)
        except Exception as e:
            errors.append(f"Error in Liquidity Swings: {str(e)}")
            logger.error(f"Liquidity Swings error: {str(e)}")
            
        try:
            # 6. Process SMC Core
            logger.info(f"Processing SMC Core")
            smc_df = chunk_df[['open', 'high', 'low', 'close']].copy()
            smc_df['time'] = smc_df.index.astype(np.int64) // 10**6  # Convert to milliseconds
            smc_df = smc_df.reset_index(drop=True)  # Reset to integer index
            smc_result = list(smc_process(smc_df))
            if smc_result:
                smc_df = pd.DataFrame(smc_result, index=chunk_df.index)  # Set back to datetime index
                logger.info(f"SMC Core added columns: {smc_df.columns.tolist()}")
                smc_df = smc_df.add_prefix('smc_')
                result_df = pd.concat([result_df, smc_df], axis=1)
        except Exception as e:
            errors.append(f"Error in SMC Core: {str(e)}")
            logger.error(f"SMC Core error: {str(e)}")
            
        try:
            # 7. Process TR Reality Core
            logger.info(f"Processing TR Reality Core")
            tr_df = chunk_df[['open', 'high', 'low', 'close', 'volume']].copy()
            tr_df['time'] = tr_df.index.astype(np.int64) // 10**6  # Convert to milliseconds
            tr_result = tr_reality(tr_df)
            if not tr_result.empty:
                logger.info(f"TR Reality Core added columns: {tr_result.columns.tolist()}")
                tr_result = tr_result.add_prefix('tr_')
                result_df = pd.concat([result_df, tr_result], axis=1)
        except Exception as e:
            errors.append(f"Error in TR Reality Core: {str(e)}")
            logger.error(f"TR Reality Core error: {str(e)}")
            
        try:
            # 8. Process BB OB Engine
            logger.info(f"Processing BB OB Engine")
            bb_df = chunk_df.copy()
            # Add time column in milliseconds if not present
            if 'time' not in bb_df.columns:
                bb_df['time'] = bb_df.index.astype(np.int64) // 10**6  # Convert to milliseconds
            bb_df = bb_df.reset_index(drop=True)  # Reset to integer index
            bb_result = list(bb_process(bb_df))
            if bb_result:
                bb_df = pd.DataFrame(bb_result, index=chunk_df.index)  # Set back to datetime index
                logger.info(f"BB OB Engine added columns: {bb_df.columns.tolist()}")
                bb_df = bb_df.add_prefix('bb_')
                result_df = pd.concat([result_df, bb_df], axis=1)
        except Exception as e:
            errors.append(f"Error in BB OB Engine: {str(e)}")
            logger.error(f"BB OB Engine error: {str(e)}")
            
        # Clean up memory
        gc.collect()
        
        return result_df, errors
        
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return pd.DataFrame(), [str(e)]

def save_checkpoint(results, errors, chunk_index, total_chunks):
    """Save intermediate results and errors"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save results
    if not results.empty:
        results.to_csv(f"{checkpoint_dir}/results_chunk_{chunk_index}.csv")
    
    # Save errors
    with open(f"{checkpoint_dir}/errors_chunk_{chunk_index}.json", 'w') as f:
        json.dump(errors, f)
    
    # Save progress
    with open(f"{checkpoint_dir}/progress.json", 'w') as f:
        json.dump({
            'last_chunk': chunk_index,
            'total_chunks': total_chunks,
            'timestamp': datetime.now().isoformat()
        }, f)

def load_checkpoint():
    """Load the last checkpoint if available"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    progress_file = f"{checkpoint_dir}/progress.json"
    if not os.path.exists(progress_file):
        return None, 0
    
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    return progress['last_chunk'], progress['total_chunks']

def process_data(input_file, output_dir):
    """Process data and create parquet files for different timeframes"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the input file
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        
        # Define timeframes and their resampling rules
        timeframes = {
            '15Second': None,  # Original data
            '1minute': '1T',
            '15minute': '15T',
            '1hour': '1H',
            '4hours': '4H'
        }
        
        # Process each timeframe
        for tf_name, tf_rule in timeframes.items():
            logger.info(f"Processing {tf_name} timeframe")
            
            # Get the data for this timeframe
            if tf_rule is None:
                # Use original 15-second data
                tf_data = df.copy()
            else:
                # Resample the data
                tf_data = df.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            
            # Process indicators for this timeframe
            logger.info(f"Processing indicators for {tf_name}")
            processed_df, _ = process_chunk(tf_data, 0, tf_data)
            
            # Convert data types before saving
            print("\nConverting data types for parquet storage...")
            for col in processed_df.columns:
                if 'session_name' in col:
                    processed_df[col] = processed_df[col].astype(str)
                elif processed_df[col].dtype == 'object':
                    try:
                        processed_df[col] = processed_df[col].astype(float)
                    except:
                        processed_df[col] = processed_df[col].astype(str)
            
            # Downcast to float32 to reduce memory footprint
            processed_df = processed_df.astype('float32', errors='ignore')
            
            # Save the processed data
            output_file = os.path.join(output_dir, f'{tf_name}.parquet')
            logger.info(f"Saving {tf_name} data to: {output_file}")
            processed_df.to_parquet(output_file)
            
            # Clean up
            del tf_data, processed_df
            gc.collect()
            
        logger.info("All timeframes processed and saved successfully")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('process_indicators.log'),
            logging.StreamHandler()
        ]
    )
    
    # Process the data
    input_file = "BTCUSDT_15s_last7days.csv"  # Your input file
    output_dir = "processed_data"  # Directory to save parquet files
    process_data(input_file, output_dir) 