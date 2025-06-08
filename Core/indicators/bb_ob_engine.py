# bb_ob_engine.py  – “Breaker/Order-Block Overlap” numeric core
#
# HOW TO USE ─────────────────────────────────────────────────────────────
#     import pandas as pd
#     from bb_ob_engine import process_candles
#
#     df = pd.read_csv("ohlc.csv", parse_dates=['time'])      # time, open, high, low, close
#     params = {           # copy/paste the same inputs you set in TV
#         "PP": 9,
#         "OB_valid": 500,
#         "mitigation_OB": "Proximal",    # or '50 % OB' / 'Distal'
#         "mitigation_BB": "Proximal",
#         "mitigation_OLB": "Proximal",
#     }
#
#     for bar in process_candles(df, **params):
#         if bar['alerts']['Alert_DMainM']:
#             print("Bullish overlap at", bar['ts'], "price", bar['close'])
# -----------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Any, Generator, Tuple

##############################################################################
# ──────────────────────────  HELPER FUNCTIONS  ──────────────────────────── #
##############################################################################

def _pivot_high(h: np.ndarray, left: int) -> np.ndarray:
    """ Boolean array: True where bar i is a high pivot with window `left` on each side """
    win = left * 2 + 1
    return (pd.Series(h)
            .rolling(win, center=True)
            .apply(lambda s: s[left] == s.max(), raw=True)
            .fillna(0)
            .astype(bool)
            .to_numpy())

def _pivot_low(l: np.ndarray, left: int) -> np.ndarray:
    win = left * 2 + 1
    return (pd.Series(l)
            .rolling(win, center=True)
            .apply(lambda s: s[left] == s.min(), raw=True)
            .fillna(0)
            .astype(bool)
            .to_numpy())

def _zigzag_labels(h_piv, l_piv, h, l, pp) -> Tuple[List[str], List[float], List[int]]:
    """
    Translate pivot booleans into the same advanced label/price/index
    arrays Pine produced (ArrayTypeAdv / ArrayValueAdv / ArrayIndexAdv).
    Logic is identical to the branch-fest in the original code,
    but expressed iteratively in Python for clarity.
    """
    lbl, val, idx = [], [], []
    last_type = None

    for i, (isH, isL) in enumerate(zip(h_piv, l_piv)):
        if not (isH or isL):
            continue

        if isH:
            t, v = "H", h[i]
        else:
            t, v = "L", l[i]

        # first point
        if not lbl:
            lbl.append(t)
            val.append(v)
            idx.append(i)
            last_type = t
            continue

        # replicate the “replace/extend” rules from the Pine script
        prev_t, prev_v, prev_i = lbl[-1], val[-1], idx[-1]

        if t == "H":
            if prev_t.startswith("L"):              # direction change
                t_out = "HL" if len(lbl) > 1 and val[-2] < v else "LL"
                lbl.append(t_out)
                val.append(v)
                idx.append(i)
            else:                                   # still highs: keep the higher one
                if v > prev_v:
                    val[-1] = v
                    idx[-1] = i
        else:  # current pivot is Low
            if prev_t.startswith("H"):
                t_out = "LH" if len(lbl) > 1 and val[-2] > v else "HH"
                lbl.append(t_out)
                val.append(v)
                idx.append(i)
            else:
                if v < prev_v:
                    val[-1] = v
                    idx[-1] = i

        last_type = t

    # Promote first two points to Major (“M”) so later rules work
    if len(lbl) >= 1: lbl[0] = "M" + lbl[0]
    if len(lbl) >= 2: lbl[1] = "M" + lbl[1]
    return lbl, val, idx


##############################################################################
# ─────────────────────────  PER-BAR CORE ENGINE  ────────────────────────── #
##############################################################################

def process_candles(df: pd.DataFrame,
                    PP: int = 9,
                    OB_valid: int = 500,
                    mitigation_OB: str = "Proximal",
                    mitigation_BB: str = "Proximal",
                    mitigation_OLB: str = "Proximal"
                    ) -> Generator[Dict[str, Any], None, None]:
    """
    Yield a dict for every bar containing the exact same primitives the Pine
    script relied on (pivots, trends, block coordinates, boolean flags).
    """

    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    o = df['open'].to_numpy()
    c = df['close'].to_numpy()
    n = len(df)

    # rolling pivot masks
    pivH = _pivot_high(h, PP)
    pivL = _pivot_low (l, PP)

    # State that must persist across bars
    arr_lbl: List[str]   = []
    arr_val: List[float] = []
    arr_idx: List[int]   = []

    external_trend = "No Trend"
    internal_trend = "No Trend"

    # queues for BoS / ChoCh indices (to replicate alert frequency control)
    break_lock_M = -1
    break_lock_m = -1

    # storage for refiner / overlap outputs (only last known values are needed
    # to build the bar dict; they are updated whenever a new block is found)
    block_state: Dict[str, Dict[str, float|int]] = {}

    for i in range(n):
        # ─── 1. Update zig-zag every bar  ─────────────────────────────────
        if pivH[i] or pivL[i]:
            lbl, val, idx = _zigzag_labels(pivH[:i+1], pivL[:i+1], h, l, PP)
            arr_lbl, arr_val, arr_idx = lbl, val, idx

        # ─── 2. Trend, BoS / ChoCh detection (major only for brevity) ────
        bull_major_bos  = False
        bear_major_bos  = False
        bull_major_cho  = False
        bear_major_cho  = False

        if len(arr_lbl) >= 2:
            last_type = arr_lbl[-1]
            last_price= arr_val[-1]
            last_idx  = arr_idx[-1]

            # pick counterpart level depending on type
            if last_type.endswith('HH') or last_type.endswith('LH'):
                major_high = last_price
                major_high_idx = last_idx
            else:
                major_high = max(arr_val[-2], arr_val[-1])  # fallback
                major_high_idx = max(arr_idx[-2], arr_idx[-1])

            if last_type.endswith('LL') or last_type.endswith('HL'):
                major_low = last_price
                major_low_idx = last_idx
            else:
                major_low = min(arr_val[-2], arr_val[-1])
                major_low_idx = min(arr_idx[-2], arr_idx[-1])

            # bullish break up
            if c[i] > major_high and break_lock_M != major_high_idx:
                if external_trend in ("No Trend", "Up Trend"):
                    bull_major_bos = True
                else:
                    bull_major_cho = True
                external_trend = "Up Trend"
                break_lock_M = major_high_idx

            # bearish break down
            if c[i] < major_low and break_lock_M != major_low_idx:
                if external_trend in ("No Trend", "Down Trend"):
                    bear_major_bos = True
                else:
                    bear_major_cho = True
                external_trend = "Down Trend"
                break_lock_M = major_low_idx

        # ─── 3. (Greatly Simplified) block / refiner demo ─────────────────
        # Here we only build one illustrative block per trigger; extend as
        # needed to cover every *_Trigger the Pine code produced.
        if bull_major_cho:
            key = "BuMChMain"
            block_state[key] = {
                "Xd1": arr_idx[-1],
                "Xd2": i,
                "Yd12": l[i],
                "Xp1": arr_idx[-1],
                "Xp2": i,
                "Yp12": (l[i] + h[i]) / 2
            }

        if bear_major_cho:
            key = "BeMChMain"
            block_state[key] = {
                "Xd1": arr_idx[-1],
                "Xd2": i,
                "Yd12": h[i],
                "Xp1": arr_idx[-1],
                "Xp2": i,
                "Yp12": (l[i] + h[i]) / 2
            }

        # ─── 4. Assemble bar dict identical to Pine variables ─────────────
        bar_out: Dict[str, Any] = {
            "ts": df.index[i] if df.index.name else df.loc[i, 'time'],
            "open":  o[i],
            "high":  h[i],
            "low":   l[i],
            "close": c[i],
            "signals": {
                "Bullish_Major_ChoCh": bull_major_cho,
                "Bearish_Major_ChoCh": bear_major_cho,
                "Bullish_Major_BoS":  bull_major_bos,
                "Bearish_Major_BoS":  bear_major_bos,
                # … add the other 18 boolean flags here …
            },
            "pivots": {
                "ArrayTypeAdv": arr_lbl.copy(),
                "ArrayValueAdv": arr_val.copy(),
                "ArrayIndexAdv": arr_idx.copy(),
                "ExternalTrend": external_trend,
                "InternalTrend": internal_trend,
            },
            "blocks": block_state.copy(),   # every key is a block name, value = coords
            "alerts": {
                "Alert_DMainM": bull_major_cho,   # sample mapping
                "Alert_SMainM": bear_major_cho,
                # map the remaining 16 alert flags as required
            }
        }

        yield bar_out
