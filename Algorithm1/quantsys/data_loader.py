import os, pandas as pd

def load_frame(path, alias):
    df = pd.read_parquet(path)
    df.columns = [f"{c}__{alias}" if c not in ("open","high","low","close","volume") else c 
                  for c in df.columns]
    return df

def load_all_frames(cfg):
    base_dir = cfg["data_dir"]
    merged   = None
    for tf in cfg["timeframes"]:
        fpath = os.path.join(base_dir, f"{tf}.parquet")
        alias = tf.lower()
        frame = load_frame(fpath, alias)
        if merged is None:
            merged = frame
        else:
            merged = merged.join(frame, how="outer")
    merged.sort_index(inplace=True)
    return merged
