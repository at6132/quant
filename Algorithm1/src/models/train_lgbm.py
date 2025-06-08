import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import optuna
import joblib

logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training."""
    # Get feature columns (exclude label and metadata columns)
    exclude_cols = ['label', 'move_size', 'time_to_move']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    return X, y

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,  # -1, 0, 1
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            param,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get validation score
        preds = model.predict(X_val)
        pred_labels = np.argmax(preds, axis=1) - 1  # Convert to -1, 0, 1
        score = np.mean(pred_labels == y_val)
        scores.append(score)
    
    return np.mean(scores)

def train_lgbm_model(df: pd.DataFrame, cfg: dict) -> lgb.Booster:
    """
    Train a LightGBM model with hyperparameter optimization.
    
    Args:
        df: DataFrame with features and labels
        cfg: Configuration dictionary with model parameters
        
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM model...")
    
    # Prepare features
    X, y = prepare_features(df, cfg)
    logger.info(f"Training on {len(X)} samples with {len(X.columns)} features")
    
    # Create study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=cfg['models']['lgbm']['n_trials'])
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        'verbosity': -1,
        'boosting_type': 'gbdt'
    })
    
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        verbose_eval=False
    )
    
    # Save feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 most important features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']}")
    
    # Save model
    model.save_model("models/lgbm_model.txt")
    joblib.dump(best_params, "models/lgbm_params.pkl")
    
    return model 