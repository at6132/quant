# smc_core.py  –  Smart-Money-Concepts re-implementation
# Author: you 🫡   License: CC BY-NC-SA 4.0 (same as the Pine)

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from enum import IntEnum, auto

# ----------------------------------------------------------------------
# 1.  constants / enums  (mirror Pine - keeps the port 1-for-1)
# ----------------------------------------------------------------------
class Bias(IntEnum):
    BEAR = -1
    BULL = 1
class Leg(IntEnum):
    BEAR = 0
    BULL = 1

# ----------------------------------------------------------------------
# 2.  dataclasses to match Pine UDTs
# ----------------------------------------------------------------------
@dataclass
class Pivot:
    current: float = np.nan
    last: float    = np.nan
    crossed: bool  = False
    t: int         = 0        # epoch ms
    idx: int       = 0
@dataclass
class OrderBlock:
    hi: float
    lo: float
    t: int
    bias: Bias
@dataclass
class FVG:
    top: float
    bot: float
    bias: Bias
@dataclass
class Alerts:          # only the bools – easy to extend
    int_bull_BOS : bool = False
    int_bear_BOS : bool = False
    int_bull_CH  : bool = False
    int_bear_CH  : bool = False
    sw_bull_BOS  : bool = False
    sw_bear_BOS  : bool = False
    sw_bull_CH   : bool = False
    sw_bear_CH   : bool = False
    int_bull_OB  : bool = False
    int_bear_OB  : bool = False
    sw_bull_OB   : bool = False
    sw_bear_OB   : bool = False
    eqh          : bool = False
    eql          : bool = False
    bull_fvg     : bool = False
    bear_fvg     : bool = False

# ----------------------------------------------------------------------
# 3.  a helper that yields "bars" with every derived field
# ----------------------------------------------------------------------
def process_candles(df: pd.DataFrame,
                    swings_len: int = 50,
                    eq_len: int = 3,
                    thr_eq: float = 0.1,
                    atr_period: int = 200,
                    show_internal_ob: bool = True,
                    show_swing_ob: bool = True,
                    ob_filter: str = "Atr"):
    """
    df columns:  open, high, low, close, time  (time in ms / np.int64)
    Yields one dict per bar mirroring Pine state.
    """

    # ----- persistent state (equivalent to Pine `var` / `array.new`) -----
    p_swing_hi, p_swing_lo   = Pivot(), Pivot()
    p_int_hi,   p_int_lo     = Pivot(), Pivot()
    p_eqh,      p_eql        = Pivot(), Pivot()

    t_swing = Bias.BEAR     # unknown -> lean bear, will switch on first pivot
    t_int   = Bias.BEAR

    swing_OBs, int_OBs   = [], []      # lists of OrderBlock
    fvgs                 = []          # list of FVG

    trailing_hi = -np.inf
    trailing_lo =  np.inf
    trailing_t_hi = trailing_t_lo = 0
    trailing_idx_hi = trailing_idx_lo = 0

    # caches for "parsed" highs/lows (handles hi-vol bar swap)
    parsed_highs, parsed_lows, times = [], [], []

    # ATR / CMR running stats
    true_range = df["high"] - df["low"]
    atr = true_range.rolling(atr_period, min_periods=1).mean()

    # ------------------------------------------------------------------
    for i, (open_, high, low, close, t) in df[["open","high","low","close","time"]].iterrows():

        ### ===== section 0:  pre-parsed values =====
        hi_vol = (high - low) >= 2 * atr.iat[i] if ob_filter == "Atr" else False
        p_high = low if hi_vol else high
        p_low  = high if hi_vol else low

        parsed_highs.append(p_high)
        parsed_lows.append(p_low)
        times.append(t)

        ### ===== section 1:  leg detection =====
        def leg(size):
            seg_hi = df["high"].iloc[i-size+1:i+1].max()
            seg_lo = df["low"].iloc[i-size+1:i+1].min()
            new_hi = high > seg_hi
            new_lo = low  < seg_lo
            return Leg.BEAR if new_hi else (Leg.BULL if new_lo else leg_prev)
        # we need previous leg per frame:
        leg_prev = Leg.BEAR
        leg_sw   = leg(swings_len)
        leg_int  = leg(5)

        # convenience lambdas
        def new_pivot(l_now, l_prev): return l_now != l_prev
        def bull_pivot(l_now, l_prev): return l_prev==Leg.BEAR and l_now==Leg.BULL
        def bear_pivot(l_now, l_prev): return l_prev==Leg.BULL and l_now==Leg.BEAR

        ### ===== helpers to update a Pivot =====
        def set_pivot(p: Pivot, price: float):
            p.last, p.current = p.current, price
            p.crossed=False
            p.t = t
            p.idx = i

        ### ===== section 2:  swing pivots =====
        if new_pivot(leg_sw, leg_prev):
            if bull_pivot(leg_sw, leg_prev):     # new LOW pivot
                set_pivot(p_swing_lo, low)
                trailing_lo, trailing_t_lo, trailing_idx_lo = low, t, i
            elif bear_pivot(leg_sw, leg_prev):   # new HIGH pivot
                set_pivot(p_swing_hi, high)
                trailing_hi, trailing_t_hi, trailing_idx_hi = high, t, i

        ### ===== section 3:  internal pivots (5-bar) =====
        if new_pivot(leg_int, Leg.BEAR):  # reusing leg_prev=BEAR placeholder
            if bull_pivot(leg_int, Leg.BEAR):
                set_pivot(p_int_lo, low)
            elif bear_pivot(leg_int, Leg.BEAR):
                set_pivot(p_int_hi, high)

        ### ===== section 4:  equal highs/lows =====
        # simple distance check vs p_eqh/p_eql
        if abs(high - p_eqh.current) < thr_eq * atr.iat[i]:
            set_pivot(p_eqh, high)
        if abs(low  - p_eql.current) < thr_eq * atr.iat[i]:
            set_pivot(p_eql, low)

        ### ===== section 5:  structure breaks & order-block storage =====
        alerts = Alerts()

        def cross_up(price, pivot: Pivot):  return close > pivot.current and not pivot.crossed
        def cross_dn(price, pivot: Pivot):  return close < pivot.current and not pivot.crossed

        # ---- bullish break (close > swing high) ----
        if cross_up(close, p_swing_hi):
            tag = "CHOCH" if t_swing == Bias.BEAR else "BOS"
            alerts.sw_bull_CH = tag=="CHOCH"
            alerts.sw_bull_BOS = tag=="BOS"
            p_swing_hi.crossed = True
            t_swing = Bias.BULL
            # store bullish swing OB
            if show_swing_ob:
                # deepest parsed low between pivot.idx .. i
                seg = parsed_lows[p_swing_hi.idx:i+1]
                lo_price = min(seg)
                lo_idx   = seg.index(lo_price) + p_swing_hi.idx
                swing_OBs.insert(0, OrderBlock(parsed_highs[lo_idx], lo_price, times[lo_idx], Bias.BULL))
                swing_OBs = swing_OBs[:20]   # keep last 20

        # ---- bearish break (close < swing low) ----
        if cross_dn(close, p_swing_lo):
            tag = "CHOCH" if t_swing == Bias.BULL else "BOS"
            alerts.sw_bear_CH = tag=="CHOCH"
            alerts.sw_bear_BOS = tag=="BOS"
            p_swing_lo.crossed = True
            t_swing = Bias.BEAR
            if show_swing_ob:
                seg = parsed_highs[p_swing_lo.idx:i+1]
                hi_price = max(seg)
                hi_idx   = seg.index(hi_price) + p_swing_lo.idx
                swing_OBs.insert(0, OrderBlock(hi_price, parsed_lows[hi_idx], times[hi_idx], Bias.BEAR))
                swing_OBs = swing_OBs[:20]

        # ---- internal breaks (same logic but vs p_int_) ----
        if cross_up(close, p_int_hi):
            tag = "CHOCH" if t_int == Bias.BEAR else "BOS"
            alerts.int_bull_CH = tag=="CHOCH"
            alerts.int_bull_BOS = tag=="BOS"
            p_int_hi.crossed = True
            t_int = Bias.BULL
            if show_internal_ob:
                seg = parsed_lows[p_int_hi.idx:i+1]
                lo_price = min(seg)
                lo_idx = seg.index(lo_price)+p_int_hi.idx
                int_OBs.insert(0, OrderBlock(parsed_highs[lo_idx], lo_price, times[lo_idx], Bias.BULL))
                int_OBs = int_OBs[:20]

        if cross_dn(close, p_int_lo):
            tag = "CHOCH" if t_int == Bias.BULL else "BOS"
            alerts.int_bear_CH = tag=="CHOCH"
            alerts.int_bear_BOS = tag=="BOS"
            p_int_lo.crossed = True
            t_int = Bias.BEAR
            if show_internal_ob:
                seg = parsed_highs[p_int_lo.idx:i+1]
                hi_price = max(seg)
                hi_idx = seg.index(hi_price)+p_int_lo.idx
                int_OBs.insert(0, OrderBlock(hi_price, parsed_lows[hi_idx], times[hi_idx], Bias.BEAR))
                int_OBs = int_OBs[:20]

        ### ----- section 6:  OB mitigation (delete) -----
        def mitigate(ob_list, mitigation_src, bull):
            keep = []
            trig_bull = trig_bear = False
            for ob in ob_list:
                crossed = mitigation_src < ob.lo if bull else mitigation_src > ob.hi
                if crossed:
                    trig_bull |= bull and ob.bias == Bias.BULL
                    trig_bear |= (not bull) and ob.bias == Bias.BEAR
                else:
                    keep.append(ob)
            return keep, trig_bull, trig_bear

        int_OBs , bull_int_hit, bear_int_hit = mitigate(int_OBs , low,  True)
        swing_OBs, bull_sw_hit , bear_sw_hit = mitigate(swing_OBs, low,  True)
        int_OBs , tmp_bull , tmp_bear = mitigate(int_OBs , high, False)
        swing_OBs, tmp2_bull, tmp2_bear = mitigate(swing_OBs, high, False)
        bull_int_hit |= tmp_bull
        bear_int_hit |= tmp_bear
        bull_sw_hit  |= tmp2_bull
        bear_sw_hit  |= tmp2_bear
        alerts.int_bull_OB = bull_int_hit
        alerts.int_bear_OB = bear_int_hit
        alerts.sw_bull_OB  = bull_sw_hit
        alerts.sw_bear_OB  = bear_sw_hit

        ### ----- section 7:  (optional) FVG detection, equal-high/lows, zones … -----
        # -- left out for brevity, but you port exactly like above:
        #    inspect last2/last1/current candles, build FVGs, push to `fvgs`,
        #    set alerts.bull_fvg, alerts.bear_fvg accordingly.

        ### ----- section 8:  package result -----
        out = dict(
            t = int(t),
            open = float(open_),
            high = float(high),
            low  = float(low),
            close= float(close),
            trend_internal = int(t_int),
            trend_swing    = int(t_swing),

            pivots = dict(
                swing_hi = asdict(p_swing_hi),
                swing_lo = asdict(p_swing_lo),
                int_hi   = asdict(p_int_hi),
                int_lo   = asdict(p_int_lo),
                eqh      = asdict(p_eqh),
                eql      = asdict(p_eql),
            ),
            alerts = asdict(alerts),
            swing_OBs = [asdict(ob) for ob in swing_OBs],
            int_OBs   = [asdict(ob) for ob in int_OBs],
            # fvgs etc …
        )
        yield out

        # roll leg_prev for next bar
        leg_prev = leg_sw

class LiveSMC:
    def __init__(self, window_size: int = 1000, swings_len: int = 50,
                 eq_len: int = 3, thr_eq: float = 0.1,
                 atr_period: int = 200, show_internal_ob: bool = True,
                 show_swing_ob: bool = True, ob_filter: str = "Atr"):
        """
        Live version of SMC
        
        Parameters
        ----------
        window_size : int
            Number of bars to keep in memory
        swings_len : int
            Length for swing detection
        eq_len : int
            Length for equal highs/lows
        thr_eq : float
            Threshold for equal highs/lows
        atr_period : int
            Period for ATR calculation
        show_internal_ob : bool
            Whether to show internal order blocks
        show_swing_ob : bool
            Whether to show swing order blocks
        ob_filter : str
            Filter for order blocks
        """
        self.window_size = window_size
        self.data_buffer = []
        self.swings_len = swings_len
        self.eq_len = eq_len
        self.thr_eq = thr_eq
        self.atr_period = atr_period
        self.show_internal_ob = show_internal_ob
        self.show_swing_ob = show_swing_ob
        self.ob_filter = ob_filter
        self.callback = None
        
    def add_callback(self, callback):
        """Add a callback function to handle results"""
        self.callback = callback
        
    def process_data(self, df: pd.DataFrame) -> dict:
        """
        Process live data using SMC logic
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price data
            
        Returns
        -------
        dict
            Dictionary containing the latest SMC data
        """
        if len(df) < max(self.swings_len, self.atr_period):
            return {}
            
        # Convert the live data to required format
        ohlc_df = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'time': df.index.astype(np.int64) // 10**6  # Convert to milliseconds
        })
        
        # Process the data
        latest = None
        for result in process_candles(
            ohlc_df,
            swings_len=self.swings_len,
            eq_len=self.eq_len,
            thr_eq=self.thr_eq,
            atr_period=self.atr_period,
            show_internal_ob=self.show_internal_ob,
            show_swing_ob=self.show_swing_ob,
            ob_filter=self.ob_filter
        ):
            latest = result
            
        if latest is None:
            return {}
            
        return {
            'high': latest.get('swing_hi', None),
            'low': latest.get('swing_lo', None)
        }
