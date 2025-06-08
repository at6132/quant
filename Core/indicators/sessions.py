# sessions.py   ---------------------------------------------------------------
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────── CONFIG ─────
# Default institutional-FX style sessions (all UTC).  
# Feel free to change or add rows – the parser picks them up automatically.
SESSIONS = pd.DataFrame({
    "name"  : ["Asia",       "London",  "NewYork"],
    "open"  : ["23:00:00",   "07:00:00", "12:30:00"],     # session start
    "close" : ["07:00:00",   "15:30:00", "20:00:00"]      # session end
})

# If your data are not UTC: df.tz_localize('exchangeTZ').tz_convert('UTC') first
# ──────────────────────────────────────────────────────────────────────────────

def build_session_table(df: pd.DataFrame,
                        sessions: pd.DataFrame = SESSIONS
                       ) -> pd.DataFrame:
    """
    For every bar:
    ▸ session_id      – integer key (–1 = no session / weekend)
    ▸ session_name    – e.g. 'Asia'
    ▸ in_session      – 1 if bar is inside a session, else 0
    ▸ new_session     – 1 on the first bar of a session
    ▸ session_open    – price at session open (Close by default)
    ▸ minutes_into    – minutes elapsed since that session's open
    """

    out              = df.copy()
    out["session_id"]   = -1
    out["session_name"] = ""
    out["in_session"]   = 0
    out["new_session"]  = 0
    out["session_open"] = np.nan
    out["minutes_into"] = np.nan

    # Pre-compute today's boundaries for speed
    borders = []
    for idx, row in sessions.iterrows():
        borders.append((idx,
                        pd.Timedelta(row.open),  pd.Timedelta(row.close),
                        row.name))
    # vectorised day-offset (keeps things fast on millions of bars)
    day_zero = out.index.normalize()

    # Iterate once, fill arrays
    last_sid   = -1
    open_price = np.nan
    for i, (ts, row) in enumerate(out.iterrows()):
        hit = False
        for sid, t_open, t_close, label in borders:
            t = ts - day_zero[i]
            if t_open <= t < t_close:
                hit = True
                out.at[ts, "session_id"]   = sid
                out.at[ts, "session_name"] = label
                out.at[ts, "in_session"]   = 1

                if sid != last_sid:                    # first bar in session
                    open_price                    = df.at[ts, "close"]
                    out.at[ts, "new_session"]     = 1
                    out.at[ts, "session_open"]    = open_price
                    out.at[ts, "minutes_into"]    = 0
                else:
                    out.at[ts, "session_open"]    = open_price
                    delta = (ts - (ts.normalize() + t_open))
                    out.at[ts, "minutes_into"]    = delta.total_seconds() // 60
                last_sid = sid
                break
        if not hit:                     # weekend / dead hours
            last_sid   = -1
            open_price = np.nan
    return out
