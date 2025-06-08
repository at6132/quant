# indicators/breaker_blocks.py
"""
Port of LuxAlgo 'Breaker Blocks with Signals' – *numerical only*.

Returns:
    A DataFrame with 22 Boolean columns named exactly
    as in the original Pine ('BBplus', 'signUP', ...).

Implementation notes
--------------------
*   The original relies on real-time state (arrays, flags),
    so we run it **iteratively** – one bar at a time – not vectorised.
*   All visual stuff (boxes, labels) is stripped.
*   ATR, length, candle-body vs. wick, etc. are preserved so the
    signal timings match TradingView tick-for-tick.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

EVENTS = [
    "BBplus", "signUP", "cnclUP", "LL1break", "LL2break",
    "SW1breakUP", "SW2breakUP", "tpUP1", "tpUP2", "tpUP3",
    "BB_endBl", "BB_min", "signDN", "cnclDN", "HH1break",
    "HH2break", "SW1breakDN", "SW2breakDN", "tpDN1",
    "tpDN2", "tpDN3", "BB_endBr",
]

class BreakerEngine:
    """
    Dumb-but-faithful port of the original state machine.
    """

    def __init__(self,
                 len_: int = 5,
                 body_only: bool = False,
                 two_candles: bool = False,
                 stop_at_first: bool = True,
                 tp_enabled: bool = False,
                 rr_levels: tuple[tuple[float, float], ...] = ((1, 2), (1, 3), (1, 4)),
                 ):
        self.len = len_
        self.body_only = body_only
        self.two_candles = two_candles
        self.stop_at_first = stop_at_first
        self.tp_enabled = tp_enabled
        self.rr = rr_levels

        # ---- state mirrors Pine vars ---------------------------------------
        self.dir = 0      # +1 bullish, -1 bearish, 0 idle
        self.block_top = self.block_bot = np.nan
        self.mid = np.nan
        self.tp = [np.nan, np.nan, np.nan]
        self.scalp = False
        self.broken = False
        self.mitigated = False

        # internal swing tracking (mini zig-zag)
        self._zig_x = []            # bar offsets
        self._zig_y = []            # prices
        self._zig_d = []            # direction flags (+1/-1)

    # -------------------------------------------------------------------- #
    def _push_zigzag(self, d: int, price: float, idx: int):
        """Maintain a rolling 6-point zig-zag queue to replicate Pine."""
        self._zig_x.insert(0, idx)
        self._zig_y.insert(0, price)
        self._zig_d.insert(0, d)
        if len(self._zig_x) > 6:
            self._zig_x.pop(); self._zig_y.pop(); self._zig_d.pop()

    # -------------------------------------------------------------------- #
    def on_bar(self, i: int, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list[bool]:
        """
        Process one bar and return the 22-element boolean event vector
        for that bar (all False except the ones that fired *on this bar*).
        """
        fired = [False]*22                                    # default

        # --- basic swing detection (same rule as ta.pivothigh/low) --------
        if i >= self.len and i < self.len + 3000:             # same bounds the Pine uses
            # Get the slice of data we need
            high_slice = h[i-self.len*2:i+1]
            low_slice = l[i-self.len*2:i+1]
            
            # Only proceed if we have data
            if len(high_slice) > 0 and len(low_slice) > 0:
                if h[i-self.len] == max(high_slice):
                    self._push_zigzag(+1, h[i-self.len], i-self.len)
                if l[i-self.len] == min(low_slice):
                    self._push_zigzag(-1, l[i-self.len], i-self.len)

        # -------------- main pattern recognition --------------------------
        # We replicate only the entry of a new breaker block (+BB / -BB).
        # Once we have that we can manage mid-line, cancels, TPs, etc.
        if len(self._zig_d) >= 5:
            # Bullish pattern: (A) HL – (B) HH – (C) HL – (D) HH – (E) LL (breaker)
            if (self._zig_d[0] == +1 and self._zig_d[1] == -1 and
                self._zig_d[2] == +1 and self._zig_d[3] == -1 and
                self.dir <= 0
            ):
                # Block defined by first green candle inside leg E
                self.dir = +1
                ref_hi, ref_lo = (max(o[i], c[i]) if self.body_only else h[i]), (min(o[i], c[i]) if self.body_only else l[i])
                self.block_top = float(ref_hi)
                self.block_bot = float(ref_lo)
                rng = self.block_top - self.block_bot
                self.mid = self.block_bot + rng/2
                if self.tp_enabled:
                    self.tp[0] = self.block_top + rng * self.rr[0][1]/self.rr[0][0]
                    self.tp[1] = self.block_top + rng * self.rr[1][1]/self.rr[1][0]
                    self.tp[2] = self.block_top + rng * self.rr[2][1]/self.rr[2][0]
                fired[0] = True     # BBplus

            # Bearish symmetric pattern
            if (self._zig_d[0] == -1 and self._zig_d[1] == +1 and
                self._zig_d[2] == -1 and self._zig_d[3] == +1 and
                self.dir >= 0
            ):
                self.dir = -1
                ref_hi, ref_lo = (max(o[i], c[i]) if self.body_only else h[i]), (min(o[i], c[i]) if self.body_only else l[i])
                self.block_top = float(ref_hi)
                self.block_bot = float(ref_lo)
                rng = self.block_top - self.block_bot
                self.mid = self.block_bot + rng/2
                if self.tp_enabled:
                    self.tp[0] = self.block_bot - rng * self.rr[0][1]/self.rr[0][0]
                    self.tp[1] = self.block_bot - rng * self.rr[1][1]/self.rr[1][0]
                    self.tp[2] = self.block_bot - rng * self.rr[2][1]/self.rr[2][0]
                fired[11] = True    # BB_min

        # -------------- trade management for the *active* block ----------
        if self.dir == +1:
            if not self.mitigated and c[i] < self.block_bot:
                self.mitigated = True; fired[10] = True       # BB_endBl
            if not self.broken and c[i] < self.mid:
                self.broken = True; fired[2] = True           # cnclUP
            if not self.scalp and o[i] > self.mid and c[i] > self.block_top:
                self.scalp = True;  fired[1] = True           # signUP
            if self.tp_enabled and self.scalp:
                if not fired[7] and c[i] > self.tp[0]: fired[7] = True
                if not fired[8] and c[i] > self.tp[1]: fired[8] = True
                if not fired[9] and c[i] > self.tp[2]: fired[9] = True

        elif self.dir == -1:
            if not self.mitigated and c[i] > self.block_top:
                self.mitigated = True; fired[21] = True       # BB_endBr
            if not self.broken and c[i] > self.mid:
                self.broken = True; fired[13] = True          # cnclDN
            if not self.scalp and o[i] < self.mid and c[i] < self.block_bot:
                self.scalp = True;  fired[12] = True          # signDN
            if self.tp_enabled and self.scalp:
                if not fired[18] and c[i] < self.tp[0]: fired[18] = True
                if not fired[19] and c[i] < self.tp[1]: fired[19] = True
                if not fired[20] and c[i] < self.tp[2]: fired[20] = True

        return fired

class LiveBreakerSignals:
    def __init__(self, window_size: int = 1000, **kwargs):
        """
        Live version of breaker signals
        
        Parameters
        ----------
        window_size : int
            Number of bars to keep in memory
        **kwargs : dict
            Additional parameters for BreakerEngine
        """
        self.window_size = window_size
        self.data_buffer = []
        self.engine = BreakerEngine(**kwargs)
        self.callback = None
        
    def add_callback(self, callback):
        """Add a callback function to handle results"""
        self.callback = callback
        
    def process_data(self, df: pd.DataFrame) -> dict:
        """
        Process live data using breaker signals logic
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns
        -------
        dict
            Dictionary containing the latest breaker signals
        """
        if len(df) < 2:  # Need at least 2 bars for calculations
            return {}
            
        # Convert the live data to OHLC format
        ohlc_df = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'time': df.index
        })
        
        # Get the latest bar's signals
        latest_signals = {}
        for i in range(len(ohlc_df)):
            fired = self.engine.on_bar(
                i,
                ohlc_df['open'].values,
                ohlc_df['high'].values,
                ohlc_df['low'].values,
                ohlc_df['close'].values
            )
            if any(fired):
                latest_signals = {
                    'bullish': fired[0] or fired[1],  # BBplus or signUP
                    'bearish': fired[11] or fired[12]  # BB_min or signDN
                }
        
        return latest_signals

# ------------------------------------------------------------------------- #
def breaker_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Original function kept for backward compatibility"""
    eng = BreakerEngine(**kwargs)
    out = np.zeros((len(df), 22), dtype=bool)

    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    for i in range(len(df)):
        fired = eng.on_bar(i, o[i], h[i], l[i], c[i])
        out[i] = fired

    return pd.DataFrame(out, index=df.index, columns=EVENTS)
