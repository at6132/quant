import numpy as np, pandas as pd
from quantsys.utils import dollars_to_price_tick

def label_future_move(df, cfg):
    horizon = cfg["label"]["horizon_minutes"]
    dollar  = cfg["label"]["dollar_threshold"]
    tick_th = dollars_to_price_tick(dollar)   # ≈500 for BTCUSDT
    fwd = df["close"].shift(-horizon*4)       # 15-sec → 4 ticks/min
    move = fwd - df["close"]
    df["label_up_$500"]   = (move >=  tick_th).astype(int)
    df["label_down_$500"] = (move <= -tick_th).astype(int)
    return df
