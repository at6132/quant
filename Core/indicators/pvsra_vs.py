import pandas as pd
import numpy as np

def pvsra_vs(df, ma_len: int = 10):
    """
    Re-implementation of the TradingView PVSRA_VS pane.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, volume
    ma_len : look-back length used for the avg-volume & max(vol*spread)

    Returns
    -------
    DataFrame with extra columns:
        vec_color : 0=grey,1=green,2=red,3=blue,4=violet
        gr_pattern: bool, Green/Red or Red/Green two-bar pattern
    """

    out = df.copy()

    # rolling statistics -------------------------------------------------------
    avg_vol   = out['volume'].rolling(ma_len).mean()
    spread    = out['high'] - out['low']
    vol_x_spd = (out['volume'] * spread)
    max_vxs   = vol_x_spd.rolling(ma_len).max()

    # base colour --------------------------------------------------------------
    out['vec_color'] = 0  # default grey

    cond_hard = (out['volume'] >= 2.0 * avg_vol) | (vol_x_spd >= max_vxs)
    cond_soft = (out['volume'] >= 1.5 * avg_vol)

    # bullish / bearish flag from candle body
    bull = out['close'] > out['open']
    bear = ~bull

    # assign colours – order matters
    out.loc[cond_hard & bull, 'vec_color'] = 1   # green
    out.loc[cond_hard & bear, 'vec_color'] = 2   # red
    out.loc[cond_soft & bull & ~cond_hard, 'vec_color'] = 3   # blue
    out.loc[cond_soft & bear & ~cond_hard, 'vec_color'] = 4   # violet

    # two-bar pattern (“Green/Red” text)
    prev = out['vec_color'].shift(1)
    out['gr_pattern'] = (
        ((out['vec_color'] == 1) & (prev == 2)) |    # green after red
        ((out['vec_color'] == 2) & (prev == 1))      # red after green
    )

    return out

# -------------- example usage -----------------------------------------------
# df = pd.read_csv('your_15s_bars.csv')
# result = pvsra_vs(df)
# result[['vec_color', 'gr_pattern']].head()
