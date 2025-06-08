#!/usr/bin/env python3
"""
Download the last 7 days of 1-second BTCUSDT spot klines from Binance,
resample into 15-second bars, and write both data-sets to CSV.

Outputs
-------
BTCUSDT_1s_last7days.csv   – raw 1-second data
BTCUSDT_15s_last7days.csv  – aggregated 15-second data
"""
import datetime as dt
import io
import os
import zipfile
from typing import List, Optional

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SYMBOL        = "BTCUSDT"
INTERVAL_1S   = "1s"
DAYS_BACK     = 7                        # last 7 **completed** days
# Start from yesterday and go back 7 days
END_DATE      = dt.datetime.utcnow().date() - dt.timedelta(days=1)  # yesterday
DATES         = [(END_DATE - dt.timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range(DAYS_BACK)]  # 7 days before yesterday
BASE_URL      = (
    "https://data.binance.vision/"
    "data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
)
OUT_1S        = f"{SYMBOL}_1s_last{DAYS_BACK}days.csv"
OUT_15S       = f"{SYMBOL}_15s_last{DAYS_BACK}days.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def download_daily_zip(symbol: str, interval: str, date: str) -> Optional[pd.DataFrame]:
    """Download one daily ZIP, return its data as a DataFrame (Open-High-Low-Close-Volume)."""
    url = BASE_URL.format(symbol=symbol, interval=interval, date=date)
    print(f"⇣  {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:      # 404? -> just skip that day
        print(f"⚠  {date} not yet available – skipped")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]          # only one file inside
        with zf.open(csv_name) as csv_file:
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
            ]
            df = pd.read_csv(
                csv_file,
                header=None,
                names=cols,
                dtype={"open_time": "int64"},
            )

    # keep only what we care about
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return df


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def main() -> None:
    # 1) download & concatenate 1-second klines
    frames = []
    for date in DATES:
        df = download_daily_zip(SYMBOL, INTERVAL_1S, date)
        if df is not None:
            frames.append(df)
    
    if not frames:
        print("No data available for the specified dates.")
        return
        
    df_1s = pd.concat(frames, ignore_index=True).sort_values("open_time")
    print(f"\nFetched {len(df_1s):,} rows of 1-second data.")

    # 2) convert epoch-ms to UTC datetime and set index
    # Convert timestamps using pd.Timestamp
    df_1s["date"] = df_1s["open_time"].apply(lambda x: pd.Timestamp(x, unit='ms', tz='UTC'))
    df_1s.set_index("date", inplace=True)
    df_1s.drop(columns="open_time", inplace=True)

    # 3) resample to 15-second bars
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_15s = (
        df_1s
        .resample("15S", label="right", closed="right")
        .agg(agg)
        .dropna(how="any")
    )
    print(f"Aggregated to {len(df_15s):,} rows of 15-second data.")

    # 4) save CSVs
    df_1s.to_csv(OUT_1S, float_format="%.10f")
    df_15s.to_csv(OUT_15S, float_format="%.10f")
    print(f"✔  Saved {OUT_1S} and {OUT_15S} to {os.getcwd()}")


if __name__ == "__main__":
    main()
