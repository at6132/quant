# -------------------------------------------------------------
# IT-Foundation numeric replica – NO colours / graphics
# -------------------------------------------------------------
# Inputs: a dict of OHLCV DataFrames keyed by timeframe
# (e.g. {'15s': df_15s, '1m': df_1m, '1h': df_1h}).
# All timestamps must be UTC and index-aligned.
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from collections import defaultdict

EMA_SET = (20, 50, 200)          # hard-coded for now
ALIGN_MAP = {'bull':  1,         # all EMAs ascending
             'bear': -1,         # all EMAs descending
             'mix' :  0}         # anything else

def ema(df: pd.DataFrame, length: int) -> pd.Series:
    return df['close'].ewm(span=length, adjust=False).mean()

def alignment_state(row) -> int:
    if row['ema20'] > row['ema50'] > row['ema200']:
        return ALIGN_MAP['bull']
    if row['ema20'] < row['ema50'] < row['ema200']:
        return ALIGN_MAP['bear']
    return ALIGN_MAP['mix']

def detect_fvg(df: pd.DataFrame, keep=20) -> list[dict]:
    """
    Returns list of active gaps as dicts with numeric coords.
    "keep" = how many future bars to extend.
    """
    live, boxes = [], []
    for i in range(2, len(df)):
        # bullish gap
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            boxes.append({'start': i-2,
                          'end': i+keep,
                          'top': df['low'].iloc[i],
                          'bot': df['high'].iloc[i-2],
                          'bias':  1})          # +1 = bullish
        # bearish gap
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            boxes.append({'start': i-2,
                          'end': i+keep,
                          'top': df['low'].iloc[i-2],
                          'bot': df['high'].iloc[i],
                          'bias': -1})          # –1 = bearish

        # prune: filled or expired
        px_lo, px_hi = df['low'].iloc[i], df['high'].iloc[i]
        live = []
        for b in boxes:
            filled = (b['bias'] == 1 and px_lo  <= b['bot']) or \
                     (b['bias'] == -1 and px_hi >= b['top'])
            expired = i > b['end']
            if not (filled or expired):
                live.append(b)
        boxes = live
    return live

def tf_trend(df: pd.DataFrame, length=200) -> int:
    ema_ = df['close'].ewm(span=length, adjust=False).mean()
    if df['close'].iloc[-1] > ema_.iloc[-1]:
        return 1
    if df['close'].iloc[-1] < ema_.iloc[-1]:
        return -1
    return 0

def process(frames: dict[str, pd.DataFrame],
            main_tf: str = '15s',
            mtf_list=('1m','5m','15m','1h')):
    main = frames[main_tf].copy()

    # --- EMAs + alignment code (-1,0,+1) ---------------------
    for L in EMA_SET:
        main[f'ema{L}'] = ema(main, L)
    main['align'] = main.apply(alignment_state, axis=1).astype('int8')

    # --- active FVG boxes (list per last row) ----------------
    main['fvg_active'] = [None]*len(main)
    live_boxes = detect_fvg(main)
    if live_boxes:
        main.at[main.index[-1], 'fvg_active'] = live_boxes   # numeric only

    # --- multi-TF trend snapshot -----------------------------
    mtf_status = {tf: tf_trend(frames[tf]) for tf in mtf_list}
    main['mtf_trend'] = [mtf_status]*len(main)

    return main

def process_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    IT Foundation indicator implementation
    """
    try:
        # Provide all required timeframes as copies of df if missing
        frames = {'15s': df}
        for tf in ['1m', '5m', '15m', '1h']:
            if tf not in frames:
                frames[tf] = df.copy()
        
        # Call the original process function
        result = process(frames)
        
        # Ensure all expected features are present
        result_df = pd.DataFrame(index=df.index)
        result_df['it_align'] = result.get('align', 0)
        result_df['it_fvg_active'] = result.get('fvg_active', None)
        result_df['it_mtf_trend'] = result.get('mtf_trend', None)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in IT Foundation: {str(e)}")
        # Return empty DataFrame with expected columns
        result_df = pd.DataFrame(index=df.index)
        result_df['it_align'] = 0
        result_df['it_fvg_active'] = None
        result_df['it_mtf_trend'] = None
        return result_df

def get_features():
    return ['it_align', 'it_fvg_active', 'it_mtf_trend']

# Example (commented):
# candles = {'15s': df15s, '1m': df1m, '5m': df5m, '1h': df1h}
# out = process(candles)
# out.tail()[['close','align','mtf_trend','fvg_active']]
