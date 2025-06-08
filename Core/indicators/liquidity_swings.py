# indicators/liq_swings.py
import pandas as pd
import numpy as np

class LiveLiquiditySwings:
    def __init__(self, window_size: int = 1000, length: int = 14,
                 area: str = "wick", count_filter: int = 0,
                 vol_filter: float = 0.):
        """
        Live version of liquidity swings
        
        Parameters
        ----------
        window_size : int
            Number of bars to keep in memory
        length : int
            Pivot look-back period
        area : str
            "wick" or "full" for zone calculation
        count_filter : int
            Minimum touch count to activate zone
        vol_filter : float
            Minimum volume to activate zone
        """
        self.window_size = window_size
        self.data_buffer = []
        self.length = length
        self.area = area
        self.count_filter = count_filter
        self.vol_filter = vol_filter
        self.callback = None
        
    def add_callback(self, callback):
        """Add a callback function to handle results"""
        self.callback = callback
        
    def process_data(self, df: pd.DataFrame) -> dict:
        """
        Process live data using liquidity swings logic
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price data
            
        Returns
        -------
        dict
            Dictionary containing the latest liquidity swing data
        """
        if len(df) < self.length * 2:
            return {}
            
        # Convert the live data to required format
        ohlc_df = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        })
        
        # Calculate liquidity swings
        swings = liquidity_swings(
            ohlc_df,
            length=self.length,
            area=self.area,
            count_filter=self.count_filter,
            vol_filter=self.vol_filter
        )
        
        # Get the latest values
        latest = swings.iloc[-1].to_dict()
        
        return {
            'high': latest.get('ph_level', None),
            'low': latest.get('pl_level', None)
        }

# Keep original function for backward compatibility
def liquidity_swings(df: pd.DataFrame,
                     length: int           = 14,
                     area: str             = "wick",   # "wick" | "full"
                     count_filter: int     = 0,
                     vol_filter: float     = 0.,
                    ) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with ['open','high','low','close','volume']
          indexed by datetime (5-second cadence).
    length : pivot look-back (matches Pine `length`)
    area   : "wick" = wick extremity, "full" = full candle range
    count_filter :   ≥N touch count required to activate zone (0 = disable)
    vol_filter   :   ≥V volume required to activate zone (0 = disable)

    Returns
    -------
    features DF with columns listed above.
    """
    hi, lo, cl, op, vol = (df[c] for c in ['high','low','close','open','volume'])
    n = len(df)
    # ------------------------------------------------------------------ #
    # Detect pivot highs / lows (fractal length, same rule as Pine)
    piv_hi = (hi.shift(length) > hi.shift(length).rolling(length*2+1).max().shift(1))           # True where hi[length] is highest in 2*len+1 window
    piv_lo = (lo.shift(length) < lo.shift(length).rolling(length*2+1).min().shift(1))
    # ensure proper boolean index
    piv_hi = piv_hi.reindex(df.index, fill_value=False)
    piv_lo = piv_lo.reindex(df.index, fill_value=False)

    # Containers
    ph_level  = np.full(n, np.nan)
    pl_level  = np.full(n, np.nan)
    ph_count  = np.zeros(n, dtype=int)
    pl_count  = np.zeros(n, dtype=int)
    ph_vol    = np.zeros(n)
    pl_vol    = np.zeros(n)
    ph_cross  = np.zeros(n, dtype=int)
    pl_cross  = np.zeros(n, dtype=int)

    # running state
    cur_ph_top = cur_ph_btm = np.nan
    cur_pl_top = cur_pl_btm = np.nan
    cur_ph_cross = False
    cur_pl_cross = False
    cur_ph_count = cur_pl_count = 0
    cur_ph_vol   = cur_pl_vol   = 0.0

    for i in range(n):
        # ---------------------------------------------------------------- pivot high logic
        if piv_hi.iloc[i]:  # new swing high found at i-length
            idx = i - length
            cur_ph_top = hi.iloc[idx]
            cur_ph_btm = max(cl.iloc[idx], op.iloc[idx]) if area=="wick" else lo.iloc[idx]
            cur_ph_cross = False
            cur_ph_count = 0
            cur_ph_vol   = 0.0

        # update counts if zone active
        if not np.isnan(cur_ph_top):
            # bar range overlap?
            if lo.iloc[i] < cur_ph_top and hi.iloc[i] > cur_ph_btm:
                cur_ph_count += 1
                cur_ph_vol   += vol.iloc[i]
            # crossed?
            if cl.iloc[i] > cur_ph_top:
                cur_ph_cross = True

        # write state to arrays
        ph_level[i]  = cur_ph_top
        ph_count[i]  = cur_ph_count
        ph_vol[i]    = cur_ph_vol
        ph_cross[i]  = int(cur_ph_cross)

        # ---------------------------------------------------------------- pivot low logic
        if piv_lo.iloc[i]:
            idx = i - length
            cur_pl_top = min(cl.iloc[idx], op.iloc[idx]) if area=="wick" else hi.iloc[idx]
            cur_pl_btm = lo.iloc[idx]
            cur_pl_cross = False
            cur_pl_count = 0
            cur_pl_vol   = 0.0

        if not np.isnan(cur_pl_btm):
            if lo.iloc[i] < cur_pl_top and hi.iloc[i] > cur_pl_btm:
                cur_pl_count += 1
                cur_pl_vol   += vol.iloc[i]
            if cl.iloc[i] < cur_pl_btm:
                cur_pl_cross = True

        pl_level[i]  = cur_pl_btm
        pl_count[i]  = cur_pl_count
        pl_vol[i]    = cur_pl_vol
        pl_cross[i]  = int(cur_pl_cross)

    out = pd.DataFrame({
        'ph_level': ph_level,
        'ph_count': ph_count,
        'ph_volume': ph_vol,
        'ph_crossed': ph_cross,
        'pl_level': pl_level,
        'pl_count': pl_count,
        'pl_volume': pl_vol,
        'pl_crossed': pl_cross,
    }, index=df.index)

    # optional filters (mimic Pine crossover logic)
    if count_filter:
        out.loc[out['ph_count'] < count_filter, ['ph_level','ph_crossed']] = np.nan, 0
        out.loc[out['pl_count'] < count_filter, ['pl_level','pl_crossed']] = np.nan, 0
    if vol_filter:
        out.loc[out['ph_volume'] < vol_filter, ['ph_level','ph_crossed']]  = np.nan, 0
        out.loc[out['pl_volume'] < vol_filter, ['pl_level','pl_crossed']]  = np.nan, 0

    return out
