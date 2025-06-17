"""
Smart Money Concepts (SMC) indicators for the intelligent trading system.
"""

from .IT_Foundation import process as it_foundation_process
from .smc_core import process_candles as smc_process
from .breaker_signals import breaker_signals, process_candles as breaker_process_candles
from .ict_sm_trades import run as ict_sm_trades_run
from .liquidity_swings import liquidity_swings
from .tr_reality_core import tr_reality
from .pvsra_vs import pvsra_vs
from .bb_ob_engine import process_candles as bb_process
from .sessions import build_session_table

# Indicator registry for easy access
INDICATOR_REGISTRY = {
    'it_foundation': it_foundation_process,
    'smc_core': smc_process,
    'breaker_signals': breaker_signals,
    'ict_sm_trades': ict_sm_trades_run,
    'liquidity_swings': liquidity_swings,
    'tr_reality': tr_reality,
    'pvsra': pvsra_vs,
    'bb_ob_engine': bb_process,
    'sessions': build_session_table
}

def get_indicator(name):
    """Get indicator function by name."""
    return INDICATOR_REGISTRY.get(name)

def get_all_indicators():
    """Get all available indicators."""
    return list(INDICATOR_REGISTRY.keys())

__all__ = [
    'INDICATOR_REGISTRY',
    'get_indicator',
    'get_all_indicators',
    'it_foundation_process',
    'smc_process', 
    'breaker_signals',
    'breaker_process_candles',
    'ict_sm_trades_run',
    'liquidity_swings',
    'tr_reality',
    'pvsra_vs',
    'bb_process',
    'build_session_table'
]