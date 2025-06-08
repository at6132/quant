import sys
import gc
import logging
from pathlib import Path

import pandas as pd
import numpy as np

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
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("process_indicators_fixed.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────
def load_15s_csv(path: Path) -> pd.DataFrame:
    """
    Works for *both* your aggregated 15-s file **and** raw Binance ZIPs
    (if you ever feed one directly).

    - If first column is integer → treat as unix-ms.
    - Otherwise treat as ISO / RFC-like string.
    """
    df = pd.read_csv(path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Set date as index
    df = df.set_index('date')
    
    # Convert numeric columns to float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Sort by index
    df = df.sort_index()
    
    return df


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
    except Exception as e:                              # noqa: BLE001
        log.warning("Indicator %s failed: %s", name, e, exc_info=True)
        return None


# ────────────────────────────────────────────────────────────────
#  Pipeline
# ────────────────────────────────────────────────────────────────
def build_feature_set(df15: pd.DataFrame) -> pd.DataFrame:
    res = df15.copy()   # start with raw OHLCV
    
    # Add time column for indicators that need it
    res['time'] = res.index.astype(np.int64) // 10**6  # Convert to milliseconds

    # 1) PVSRA - runs first because several later indicators piggy-back on it
    res = _safe_join(res, _safe_run(pvsra_vs,          "PVSRA",            res),       "pvsra_")

    # 2) Sessions table (labels such as session_name, open_price …)
    res = _safe_join(res, _safe_run(build_session_table,"Sessions",        res[["close"]].copy()), "sess_")

    # 3) ICT SM trades
    res = _safe_join(res, _safe_run(ict_sm_trades_run, "ICT_SM",           res),       "ict_")

    # 4) Breaker signals - convert to numpy arrays for processing
    breaker_df = res[['open', 'high', 'low', 'close']].copy()
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
    breaker_result = pd.DataFrame(out, index=res.index, columns=EVENTS)
    res = _safe_join(res, breaker_result, "brk_")

    # 5) Liquidity swings
    res = _safe_join(res, _safe_run(liquidity_swings,  "Liquidity",        res),       "liq_")

    # 6) SMC core (generator) - needs integer index
    smc_df = res[['open', 'high', 'low', 'close', 'time']].copy()
    smc_df = smc_df.reset_index(drop=True)  # Reset to integer index
    smc_result = _safe_run(smc_process, "SMC_core", smc_df)
    if smc_result is not None:
        smc_result = pd.DataFrame(smc_result, index=res.index)  # Set back to datetime index
    res = _safe_join(res, smc_result, "smc_")

    # 7) TR Reality
    tr_df = res[['open', 'high', 'low', 'close', 'volume', 'time']].copy()
    res = _safe_join(res, _safe_run(tr_reality,        "TR_reality",       tr_df),       "tr_")

    # 8) BB OB engine (generator) - needs integer index
    bb_df = res.copy()
    bb_df = bb_df.reset_index(drop=True)  # Reset to integer index
    bb_result = _safe_run(bb_process, "BB_OB", bb_df)
    if bb_result is not None:
        bb_result = pd.DataFrame(bb_result, index=res.index)  # Set back to datetime index
    res = _safe_join(res, bb_result, "bb_")

    # 9) IT Foundation  (needs multiple TFs)
    # Reset index to integers for IT Foundation
    multi = {
        "15s": res.reset_index(drop=True),
        "1m" : res.resample("1min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).reset_index(drop=True),
        "5m" : res.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).reset_index(drop=True),
        "15m": res.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).reset_index(drop=True),
        "1h" : res.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).reset_index(drop=True)
    }
    it_result = _safe_run(it_foundation_process, "IT_Foundation", multi)
    if it_result is not None:
        it_result = pd.DataFrame(it_result, index=res.index)  # Set back to datetime index
    res = _safe_join(res, it_result, "it_")

    # final clean-up: drop columns that are *entirely* NaN (they never came back)
    res = res.dropna(axis=1, how="all")

    # guard against duplicated indices (just in case)
    res = res.loc[~res.index.duplicated(keep="first")]

    log.info("Final frame: %d rows  |  %d columns", len(res), res.shape[1])
    return res


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    src = Path("BTCUSDT_15s.csv").expanduser()
    dest = Path("sample_data.csv").expanduser()

    log.info("Loading %s …", src)
    df15 = load_15s_csv(src)
    # Take first 200 rows
    df15 = df15.iloc[:200]
    log.info("Loaded %d rows (%.2f-MB)", len(df15), df15.memory_usage(deep=True).sum()/2**20)

    log.info("Building feature set …")
    try:
        features = build_feature_set(df15)
        
        # Check if any indicators failed (columns with all NaN)
        nan_columns = features.columns[features.isna().all()].tolist()
        if nan_columns:
            log.error("The following indicators failed: %s", nan_columns)
            sys.exit(1)
            
        log.info("Saving %s", dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(dest)
        log.info("Done.")
    except Exception as e:
        log.error("Error processing data: %s", str(e))
        sys.exit(1)