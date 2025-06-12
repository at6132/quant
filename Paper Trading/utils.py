def align_features(df, feature_names):
    missing = [f for f in feature_names if f not in df.columns]
    for m in missing:
        df[m] = 0
    return df[feature_names] 