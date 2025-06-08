# tr_reality_core.py
# --------------------------------------------------------------------
#   Numerical re-implementation of the relevant bits of
#   "Traders Reality Main" (© TradersReality, MPL-2.0).
#   Outputs a pandas.DataFrame – one row per bar.
# --------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# ---------- 1. Settings that mirror the Pine defaults ------------- #
EMA_PERIODS      = (5, 13, 50, 200, 800)
VEC_WINDOW       = 10            # bars for volume/vol-spread history
VEC_HARD_RED_GRN = 2.0           # >= 200 % vol or max(spread*vol)
VEC_SOFT         = 1.5           # >= 150 % vol
# colour names kept for reference only
COL_RED, COL_GRN, COL_VIO, COL_BLU, COL_DN, COL_UP = range(6)

# ---------- 2. Dataclasses --------------------------------------- #
@dataclass
class VCandle:
    flag: int              # one of the colour enums above
    top: float             # zone top
    bot: float             # zone bottom
    idx_open: int          # index where zone starts

# ---------- 3. Core function ------------------------------------- #
def tr_reality(df: pd.DataFrame,
               keep_zone_boxes: int = 500
               ) -> pd.DataFrame:
    """
    Parameters
    ----------
    df  : DataFrame with columns [time, open, high, low, close, volume]
          time can be anything hashable; no plotting performed.
    Returns
    -------
    DataFrame with:
        ema_5 … ema_800,
        prev_day_H/L, prev_week_H/L,
        vec_color (int enum),
        vcz_top, vcz_bot  (NaN when no active zone)
    """
    out = pd.DataFrame(index=df.index)

    # --- 3.1   EMAs ------------------------------------------------ #
    for p in EMA_PERIODS:
        out[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    # --- 3.2   Yesterday / Last-Week levels ----------------------- #
    t = pd.to_datetime(df["time"], unit="ms") if np.issubdtype(df["time"].dtype, np.integer) else pd.to_datetime(df["time"])
    day   = t.dt.date
    week  = t.dt.isocalendar().week
    # shift so "yesterday" means the *previous* completed day
    y_hi = df.groupby(day)["high"].transform("max").shift(1)
    y_lo = df.groupby(day)["low" ].transform("min").shift(1)
    w_hi = df.groupby(week)["high"].transform("max").shift(1)
    w_lo = df.groupby(week)["low" ].transform("min").shift(1)
    out["yesterday_H"], out["yesterday_L"] = y_hi, y_lo
    out["lastweek_H"], out["lastweek_L"]   = w_hi, w_lo

    # --- 3.3   Vector-candle classification ----------------------- #
    # helpers
    spread = df["high"] - df["low"]
    vol    = df["volume"]
    vol_ma = vol.rolling(VEC_WINDOW, 1).mean()
    prod   = spread * vol
    max_prod = prod.rolling(VEC_WINDOW, 1).max()

    vec_flag = np.full(len(df), COL_DN)  # default bear/regular down

    bull = df["close"] > df["open"]
    bear = ~bull

    # hard vectors
    big = (vol >= VEC_HARD_RED_GRN * vol_ma) | (prod >= max_prod)
    vec_flag[np.where(big & bull)] = COL_GRN
    vec_flag[np.where(big & bear)] = COL_RED

    # soft vectors (only where not already hard)
    med = (vol >= VEC_SOFT * vol_ma) & ~big
    vec_flag[np.where(med & bull)] = COL_BLU
    vec_flag[np.where(med & bear)] = COL_VIO

    # remaining candles get "regular up/down"
    vec_flag[np.where(~big & ~med & bull)] = COL_UP

    out["vec_color"] = vec_flag

    # --- 3.4   Vector-Candle Zones (body-only, simple rules) ------ #
    tops  = []
    bots  = []
    active_above: list[VCandle] = []   # zones where close>open
    active_below: list[VCandle] = []   # close<open

    opens, closes = df["open"].to_numpy(), df["close"].to_numpy()

    for i, flag in enumerate(vec_flag):
        hi, lo = df["high"].iat[i], df["low"].iat[i]
        # ----- create zone on new vector candle -------------------
        if flag in (COL_GRN, COL_BLU):       # bullish vectors
            active_above.append(VCandle(flag, max(opens[i], closes[i]),
                                         min(opens[i], closes[i]), i))
        elif flag in (COL_RED, COL_VIO):     # bearish vectors
            active_below.append(VCandle(flag, max(opens[i], closes[i]),
                                         min(opens[i], closes[i]), i))

        # ----- trim old lists ------------------------------------
        if len(active_above) > keep_zone_boxes:
            active_above.pop(0)
        if len(active_below) > keep_zone_boxes:
            active_below.pop(0)

        # ----- clear zones ("body with wicks" logic) -------------
        for arr, bullish in ((active_above, True), (active_below, False)):
            kill = []
            for z in arr:
                hit = (lo < z.bot) if bullish else (hi > z.top)
                if hit:
                    kill.append(z)
            for z in kill:
                arr.remove(z)

        # ----- aggregate current extremes (for export) -----------
        tops.append(active_above[-1].top if active_above else np.nan)
        bots.append(active_below[-1].bot if active_below else np.nan)

    out["vcz_top"] = tops
    out["vcz_bot"] = bots
    return out

class LiveTRReality:
    def __init__(self, window_size: int = 1000, keep_zone_boxes: int = 500):
        """
        Live version of TR Reality
        
        Parameters
        ----------
        window_size : int
            Number of bars to keep in memory
        keep_zone_boxes : int
            Number of zone boxes to keep in memory
        """
        self.window_size = window_size
        self.data_buffer = []
        self.keep_zone_boxes = keep_zone_boxes
        self.callback = None
        
    def add_callback(self, callback):
        """Add a callback function to handle results"""
        self.callback = callback
        
    def process_data(self, df: pd.DataFrame) -> dict:
        """
        Process live data using TR Reality logic
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price data
            
        Returns
        -------
        dict
            Dictionary containing the latest TR Reality data
        """
        if len(df) < 2:
            return {}
            
        # Convert the live data to required format
        ohlc_df = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume'],
            'time': df.index
        })
        
        # Calculate TR Reality indicators
        tr_data = tr_reality(ohlc_df, keep_zone_boxes=self.keep_zone_boxes)
        
        # Get the latest values
        latest = tr_data.iloc[-1].to_dict()
        
        return {
            'high': latest.get('vcz_top', None),
            'low': latest.get('vcz_bot', None)
        }
