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
        logger.debug("Processing indicators...")
        pvsra_result = pvsra_process(df)
        if isinstance(pvsra_result, pd.DataFrame):
            df = pd.concat([df, pvsra_result[['vec_color', 'gr_pattern']]], axis=1)
        else:
            error_msg = f"pvsra_process returned type {type(pvsra_result)}: {pvsra_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process Sessions
        sessions_result = sessions_process(df)
        if isinstance(sessions_result, pd.DataFrame):
            df = pd.concat([df, sessions_result[['session_id', 'session_name', 'in_session', 'new_session', 'session_open', 'minutes_into']]], axis=1)
        else:
            error_msg = f"sessions_process returned type {type(sessions_result)}: {sessions_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process ICT SM Trades
        ict_result = ict_sm_trades_process(df)
        if isinstance(ict_result, pd.DataFrame):
            df = pd.concat([df, ict_result], axis=1)
        else:
            error_msg = f"ict_sm_trades_process returned type {type(ict_result)}: {ict_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process Breaker Signals
        breaker_result = breaker_signals_process(df)
        if isinstance(breaker_result, pd.DataFrame):
            df = pd.concat([df, breaker_result], axis=1)
        else:
            error_msg = f"breaker_signals_process returned type {type(breaker_result)}: {breaker_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process Liquidity Swings
        swing_result = liquidity_swings_process(df)
        if isinstance(swing_result, pd.DataFrame):
            df = pd.concat([df, swing_result], axis=1)
        else:
            error_msg = f"liquidity_swings_process returned type {type(swing_result)}: {swing_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process SMC Core
        try:
            print('SMC Core: input df shape:', df.shape)
            print('SMC Core: input df columns:', df.columns.tolist())
            print('SMC Core: input df head:\n', df.head())
            try:
                smc_df = df[['open', 'high', 'low', 'close']].copy()
                smc_df['time'] = smc_df.index.astype(np.int64) // 10**6  # Convert to ms
                smc_df = smc_df.reset_index(drop=True)  # Integer index for SMC
                print('SMC Core: smc_df shape:', smc_df.shape)
                print('SMC Core: smc_df columns:', smc_df.columns.tolist())
                print('SMC Core: smc_df head:\n', smc_df.head())
            except Exception as e:
                print('SMC Core: DataFrame preparation error:', e)
                error_msg = f"SMC Core DataFrame preparation error: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return df, errors
            try:
                smc_result = list(smc_core_process(smc_df))
                print('SMC Core: smc_result type:', type(smc_result), 'length:', len(smc_result))
                if smc_result:
                    smc_result_df = pd.DataFrame(smc_result, index=df.index)  # Re-align to datetime index
                    smc_result_df = smc_result_df.add_prefix('smc_')
                    print('SMC Core: smc_result_df shape:', smc_result_df.shape)
                    print('SMC Core: smc_result_df columns:', smc_result_df.columns.tolist())
                    print('SMC Core: smc_result_df head:\n', smc_result_df.head())
                    df = pd.concat([df, smc_result_df], axis=1)
                else:
                    error_msg = "SMC Core returned empty list"
                    print(error_msg)
                    logger.error(error_msg)
                    errors.append(error_msg)
                    return df, errors
            except Exception as e:
                print('SMC Core: process_candles error:', e)
                error_msg = f"Error in SMC Core: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return df, errors
        except Exception as e:
            print('SMC Core: unknown error:', e)
            error_msg = f"Error in SMC Core: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process TR Reality Core
        tr_result = tr_reality_core_process(df)
        if isinstance(tr_result, pd.DataFrame):
            df = pd.concat([df, tr_result], axis=1)
        else:
            error_msg = f"tr_reality_core_process returned type {type(tr_result)}: {tr_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        # Process BB OB Engine
        bb_result = bb_process(df)
        if isinstance(bb_result, pd.DataFrame):
            df = pd.concat([df, bb_result], axis=1)
        else:
            error_msg = f"bb_process returned type {type(bb_result)}: {bb_result}"
            logger.error(error_msg)
            errors.append(error_msg)
            return df, errors
            
        logger.debug("Indicators processed successfully")
        return df, errors
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Error processing indicators: %s", error_msg)
        errors.append(error_msg)
        return df, errors 