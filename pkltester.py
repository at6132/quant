import lightgbm as lgb
import joblib            # or `pickle`
import json

# --- 1. Load the model -------------------------------------------------------
model = joblib.load("Algorithm1/artefacts/lgbm_model.pkl")         # or pickle.load(open(...,'rb'))

# --- 2. Basic metadata -------------------------------------------------------
is_classifier = hasattr(model, "classes_")
n_features    = len(model.feature_name())
n_trees       = model.num_trees()
params        = model.params

print(f"Classifier?   {is_classifier}")
print(f"# features:   {n_features}")
print(f"# trees:      {n_trees}")
print(json.dumps(params, indent=2))

# --- 3. Feature importance ---------------------------------------------------
import pandas as pd
fi_gain = model.feature_importance(importance_type="gain")
fi_split = model.feature_importance(importance_type="split")

fi = (pd.DataFrame({
        "feature": model.feature_name(),
        "gain"   : fi_gain,
        "split"  : fi_split
      })
      .sort_values("gain", ascending=False))

print("\nTop 20 Most Important Features:")
print(fi.head(20))

# --- 4. Global metrics (if available) ---------------------------------------
if hasattr(model, "best_score"):
    print(f"\nBest Score: {model.best_score}")
