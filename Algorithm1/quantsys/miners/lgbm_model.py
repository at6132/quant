import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for modeling."""
    # Exclude non-feature columns
    exclude_cols = ['label', 'bars_to_breach', 'price_at_breach', 'move_size', 
                   'time_to_breach_minutes', 'open', 'high', 'low', 'close', 'volume']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle any remaining NaN/inf values
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Labels
    y = df['label']
    
    return X, y

def lgb_objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for Optuna optimization."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,  # -1, 0, 1
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
        'verbosity': -1,
        'random_state': 42
    }
    
    # Adjust labels to 0, 1, 2 for LightGBM
    y_train_adj = y_train + 1
    y_val_adj = y_val + 1
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train_adj)
    val_data = lgb.Dataset(X_val, label=y_val_adj, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1) - 1  # Convert back to -1, 0, 1
    
    # We want to maximize the accuracy of non-zero predictions
    mask = y_val != 0
    if mask.sum() > 0:
        accuracy = accuracy_score(y_val[mask], y_pred_class[mask])
    else:
        accuracy = 0
    
    return accuracy

def run_lgbm(df: pd.DataFrame, cfg: dict) -> Dict:
    """Run LightGBM model with Bayesian optimization."""
    print("Running LightGBM model...")
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Filter to rows with actual signals for training
    # This helps the model focus on the important cases
    signal_mask = y != 0
    signal_ratio = signal_mask.sum() / len(y)
    print(f"Signal ratio: {signal_ratio:.2%}")
    
    # Use time series split for validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    best_params = None
    best_score = -1
    
    # If we have enough signals, do optimization
    if signal_mask.sum() > 1000:
        print("Running Bayesian optimization...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create study for this fold
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            
            study.optimize(
                lambda trial: lgb_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=cfg['ml']['lgbm']['max_evals'] // n_splits  # Distribute trials across folds
            )
            
            if study.best_value > best_score:
                best_score = study.best_value
                best_params = study.best_params
                
        print(f"\nBest validation score: {best_score:.4f}")
    
    # Train final model on all data with best params (or defaults)
    if best_params is None:
        best_params = {
            'num_leaves': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.1
        }
    
    # Final model parameters
    final_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        **best_params
    }
    
    # Train on full dataset
    print("\nTraining final model...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Adjust labels
    y_train_adj = y_train + 1
    y_test_adj = y_test + 1
    
    train_data = lgb.Dataset(X_train, label=y_train_adj)
    val_data = lgb.Dataset(X_test, label=y_test_adj, reference=train_data)
    
    model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(cfg['ml']['lgbm']['early_stopping']), lgb.log_evaluation(100)]
    )
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 most important features:")
    print(importance_df.head(20))
    
    # Generate predictions for the entire dataset
    predictions = model.predict(X, num_iteration=model.best_iteration)
    pred_classes = np.argmax(predictions, axis=1) - 1
    
    # Also get prediction probabilities
    pred_probs = predictions.max(axis=1)
    
    # Create signals based on high-confidence predictions
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['confidence'] = 0
    
    # Only take high confidence predictions
    confidence_threshold = 0.6
    high_conf_mask = pred_probs > confidence_threshold
    
    signals.loc[high_conf_mask, 'signal'] = pred_classes[high_conf_mask]
    signals.loc[high_conf_mask, 'confidence'] = pred_probs[high_conf_mask]
    
    # Test set performance
    y_test_pred = pred_classes[train_size:]
    test_mask = y_test != 0
    
    if test_mask.sum() > 0:
        test_accuracy = accuracy_score(y_test[test_mask], y_test_pred[test_mask])
        print(f"\nTest set accuracy (non-zero labels): {test_accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test[test_mask], y_test_pred[test_mask], 
                                  target_names=['Short', 'Long']))
    
    return {
        'model': model,
        'signals': signals,
        'feature_importance': importance_df,
        'params': final_params,
        'test_accuracy': test_accuracy if 'test_accuracy' in locals() else None
    }