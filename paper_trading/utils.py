import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import traceback
import gc
from typing import Tuple, List
import sys
import os

# Add Core to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ────────────────────────────────────────────────────────────────
#  Core imports
# ────────────────────────────────────────────────────────────────
from Core.indicators.pvsra_vs import pvsra_vs as pvsra_process
from Core.indicators.sessions import build_session_table as sessions_process
from Core.indicators.ict_sm_trades import run as ict_sm_trades_process
from Core.indicators.breaker_signals import breaker_signals as breaker_signals_process
from Core.indicators.liquidity_swings import liquidity_swings as liquidity_swings_process
from Core.indicators.smc_core import process_candles as smc_core_process
from Core.indicators.tr_reality_core import tr_reality as tr_reality_core_process
from Core.indicators.bb_ob_engine import process_candles as bb_process

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def align_features(df, feature_names):
    missing = [f for f in feature_names if f not in df.columns]
    for m in missing:
        df[m] = 0
    return df[feature_names]

def process_chunk(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[pd.DataFrame, List[str]]:
    """Process a chunk of data through all indicators"""
    errors = []
    
    try:
        # Ensure DataFrame has a proper index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Process PVSRA
        logger.debug("Processing PVSRA...")
        try:
            pvsra_result = pvsra_process(df)
            if isinstance(pvsra_result, pd.DataFrame) and not pvsra_result.empty:
                # Ensure index alignment
                if not pvsra_result.index.equals(df.index):
                    pvsra_result = pvsra_result.reindex(df.index, fill_value=0)
                # Select only expected columns to avoid conflicts
                expected_cols = ['vec_color', 'gr_pattern']
                available_cols = [col for col in expected_cols if col in pvsra_result.columns]
                if available_cols:
                    df = pd.concat([df, pvsra_result[available_cols]], axis=1)
            else:
                logger.warning("PVSRA process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in PVSRA: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process Sessions
        logger.debug("Processing Sessions...")
        try:
            sessions_result = sessions_process(df)
            if isinstance(sessions_result, pd.DataFrame) and not sessions_result.empty:
                # Ensure index alignment
                if not sessions_result.index.equals(df.index):
                    sessions_result = sessions_result.reindex(df.index, fill_value=0)
                # Select only expected columns
                expected_cols = ['session_id', 'session_name', 'in_session', 'new_session', 'session_open', 'minutes_into']
                available_cols = [col for col in expected_cols if col in sessions_result.columns]
                if available_cols:
                    df = pd.concat([df, sessions_result[available_cols]], axis=1)
            else:
                logger.warning("Sessions process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in Sessions: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process ICT SM Trades
        logger.debug("Processing ICT SM Trades...")
        try:
            ict_result = ict_sm_trades_process(df)
            if isinstance(ict_result, pd.DataFrame) and not ict_result.empty:
                # Ensure index alignment
                if not ict_result.index.equals(df.index):
                    ict_result = ict_result.reindex(df.index, fill_value=0)
                df = pd.concat([df, ict_result], axis=1)
            else:
                logger.warning("ICT SM Trades process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in ICT SM Trades: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process Breaker Signals
        logger.debug("Processing Breaker Signals...")
        try:
            breaker_result = breaker_signals_process(df)
            if isinstance(breaker_result, pd.DataFrame) and not breaker_result.empty:
                # Ensure index alignment
                if not breaker_result.index.equals(df.index):
                    breaker_result = breaker_result.reindex(df.index, fill_value=0)
                df = pd.concat([df, breaker_result], axis=1)
            else:
                logger.warning("Breaker Signals process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in Breaker Signals: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process Liquidity Swings
        logger.debug("Processing Liquidity Swings...")
        try:
            swing_result = liquidity_swings_process(df)
            if isinstance(swing_result, pd.DataFrame) and not swing_result.empty:
                # Ensure index alignment
                if not swing_result.index.equals(df.index):
                    swing_result = swing_result.reindex(df.index, fill_value=0)
                df = pd.concat([df, swing_result], axis=1)
            else:
                logger.warning("Liquidity Swings process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in Liquidity Swings: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process SMC Core
        try:
            logger.debug("Processing SMC Core...")
            
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                error_msg = "SMC Core requires OHLC columns"
                logger.error(error_msg)
                errors.append(error_msg)
                return df, errors
            
            # Use the proper prepare_smc_core_data function from the smc_core module
            from Core.indicators.smc_core import prepare_smc_core_data
            
            smc_result_df = prepare_smc_core_data(df)
            
            if not smc_result_df.empty:
                # Add prefix to avoid column name conflicts
                smc_result_df = smc_result_df.add_prefix('smc_')
                logger.debug(f"SMC Core added columns: {smc_result_df.columns.tolist()}")
                
                # Ensure index alignment before concatenation
                if not smc_result_df.index.equals(df.index):
                    logger.warning("SMC Core index mismatch, attempting to align...")
                    smc_result_df = smc_result_df.reindex(df.index, fill_value=0)
                
                df = pd.concat([df, smc_result_df], axis=1)
            else:
                logger.warning("SMC Core returned empty DataFrame")
                
        except Exception as e:
            error_msg = f"Error in SMC Core: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process TR Reality Core
        logger.debug("Processing TR Reality Core...")
        try:
            tr_result = tr_reality_core_process(df)
            if isinstance(tr_result, pd.DataFrame) and not tr_result.empty:
                # Ensure index alignment
                if not tr_result.index.equals(df.index):
                    tr_result = tr_result.reindex(df.index, fill_value=0)
                df = pd.concat([df, tr_result], axis=1)
            else:
                logger.warning("TR Reality Core process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in TR Reality Core: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        # Process BB OB Engine
        logger.debug("Processing BB OB Engine...")
        try:
            bb_result = bb_process(df)
            if isinstance(bb_result, pd.DataFrame) and not bb_result.empty:
                # Ensure index alignment
                if not bb_result.index.equals(df.index):
                    bb_result = bb_result.reindex(df.index, fill_value=0)
                df = pd.concat([df, bb_result], axis=1)
            else:
                logger.warning("BB OB Engine process returned invalid or empty result")
        except Exception as e:
            error_msg = f"Error in BB OB Engine: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
        logger.debug("All indicators processed successfully")
        return df, errors
        
    except Exception as e:
        error_msg = f"Unexpected error in process_chunk: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return df, errors 