# ict_sm_trades.py  ───────────────────────────────────────────────────────────
"""
Numerical translation of the TradingView indicator
 ‘ICT SM Trades’ (pivot / session / D-W liquidity grabs, MSS & FVG).

How to use
----------
    import pandas as pd, ict_sm_trades as ict

    df      = ...  # your UTC-indexed OHLCV dataframe
    signals = ict.run(df,
                      atr_len      = 28,
                      lg_types     = ("pivot","daily","weekly","session"),
                      fvg          = True,
                      mss_source   = "close")        # or "highlow"
"""

from dataclasses import dataclass
import pandas as pd
import numpy  as np

# ───────────────────────────────────────────────────── configurable bits ─────
@dataclass
class Cfg:
    atr_len   : int                  = 28
    lg_types  : tuple[str,...]       = ("pivot","daily","weekly","session")
    mss_src   : str                  = "close"       # or "highlow"
    fvg       : bool                 = True
    fvg_mid   : bool                 = True          # middle line requested
    lookback  : int                  = 2000          # keep RAM in check

# ─────────────────────────────────────────── helper – swings & pivots ────────
def rolling_extrema(series: pd.Series, left:int, right:int, mode:str):
    """
    Returns True at swing-high / swing-low indices.
    mode : "high" or "low"
    """
    if mode == "high":
        return series == series.rolling(left+right+1, center=True).max()
    else:
        return series == series.rolling(left+right+1, center=True).min()

# ───────────────────────────────────── core routine ──────────────────────────
def run(df: pd.DataFrame, **user_cfg) -> pd.DataFrame:
    cfg               = Cfg(**user_cfg)              # merge defaults / user
    data              = df.copy().iloc[-cfg.lookback:].copy()

    hi, lo, cl        = data.high, data.low, data.close
    rng               = hi - lo
    atr               = rng.rolling(cfg.atr_len).mean()

    # ── 1.  swing pivots ────────────────────────────────────────────────
    piv_hi            = rolling_extrema(hi, 3, 3, "high")
    piv_lo            = rolling_extrema(lo, 3, 3, "low")

    # ── 2.  reference levels for LGs ────────────────────────────────────
    daily_hi  = hi.resample("1D").transform("max")
    daily_lo  = lo.resample("1D").transform("min")
    weekly_hi = hi.resample("1W-MON").transform("max")
    weekly_lo = lo.resample("1W-MON").transform("min")

    # helper that checks "took liquidity then rejected"
    def lg_mask(src_hi, src_lo):
        sweep_hi = (hi > src_hi.shift()) & (cl < src_hi.shift())
        sweep_lo = (lo < src_lo.shift()) & (cl > src_lo.shift())
        return sweep_hi, sweep_lo

    lg_masks = {}
    if "pivot"  in cfg.lg_types:
        lg_masks["LG_pivot_hi"], lg_masks["LG_pivot_lo"] = lg_mask(hi.where(piv_hi),
                                                                  lo.where(piv_lo))
    if "daily"  in cfg.lg_types:
        lg_masks["LG_daily_hi"], lg_masks["LG_daily_lo"] = lg_mask(daily_hi, daily_lo)
    if "weekly" in cfg.lg_types:
        lg_masks["LG_weekly_hi"],lg_masks["LG_weekly_lo"]= lg_mask(weekly_hi,weekly_lo)

    # ── 3.  MSS  (market-structure shift)  ------------------------------
    src_hi        = hi if cfg.mss_src=="highlow" else cl
    src_lo        = lo if cfg.mss_src=="highlow" else cl
    last_hh       = src_hi[piv_hi].reindex_like(src_hi).ffill()
    last_ll       = src_lo[piv_lo].reindex_like(src_lo).ffill()
    mss_up        = (cl > last_hh.shift())        # bullish shift
    mss_dn        = (cl < last_ll.shift())

    # ── 4.  Fair-Value-Gaps  (Wakabayashi 2020)  ------------------------
    if cfg.fvg:
        gap_up  = (lo.shift(1) > hi.shift(2))
        gap_dn  = (hi.shift(1) < lo.shift(2))
        fvg_top    = lo.shift(1).where(gap_up)
        fvg_bot    = hi.shift(2).where(gap_up)
        fvg_top_dn = hi.shift(1).where(gap_dn)
        fvg_bot_dn = lo.shift(2).where(gap_dn)
        if cfg.fvg_mid:
            fvg_mid_up = (fvg_top + fvg_bot)/2
            fvg_mid_dn = (fvg_top_dn + fvg_bot_dn)/2

    # ── 5.  compile result table  --------------------------------------
    out = pd.DataFrame(index=data.index)
    out["close"]    = cl
    out["atr"]      = atr

    for k,v in lg_masks.items():
        out[k] = v.astype(int)

    out["mss_up"]   = mss_up.astype(int)
    out["mss_dn"]   = mss_dn.astype(int)

    if cfg.fvg:
        out["fvg_up_high"]   = fvg_top
        out["fvg_up_low"]    = fvg_bot
        out["fvg_dn_high"]   = fvg_top_dn
        out["fvg_dn_low"]    = fvg_bot_dn
        if cfg.fvg_mid:
            out["fvg_up_mid"] = fvg_mid_up
            out["fvg_dn_mid"] = fvg_mid_dn

    return out.dropna(how="all", subset=out.columns.difference(["close","atr"]))
