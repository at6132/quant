import itertools, pandas as pd, numpy as np
from sklearn.metrics import precision_score, recall_score

def mine_rules(df, cfg):
    INDICATORS = [c for c in df.columns if "__" in c]
    BEST = []
    for a,b in itertools.combinations(INDICATORS, 2):
        sig = (df[a] > 0) & (df[b] > 0)             # toy rule: both positive
        p = precision_score(df["label_up_$500"], sig)
        r = recall_score(   df["label_up_$500"], sig)
        if p>0.6 and r>0.1:
            BEST.append((p*r, a, b, p, r))
    BEST.sort(reverse=True)
    top = BEST[:50]
    # build dataframe of signals
    signals = pd.Series(False, index=df.index)
    for _,a,b,_,_ in top:
        signals |= (df[a]>0)&(df[b]>0)
    return {"rules": top, "signals": signals}
