import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna
from typing import Dict, Tuple, List
import gc
from label_generator import LabelGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns, excluding non-feature columns."""
    exclude_cols = ['label', 'action_label', 'open', 'high', 'low', 'close', 'volume']
    return [col for col in df.columns if col not in exclude_cols]

def process_chunk(chunk: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Process a chunk of data, handling NaN and inf values."""
    X = chunk[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X

def prepare_features_chunked(df: pd.DataFrame, chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Preparing features in chunks...")
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        processed_chunk = process_chunk(chunk, feature_cols)
        chunks.append(processed_chunk)
        logger.info(f"Processed chunk {i//chunk_size + 1}/{(len(df) + chunk_size - 1)//chunk_size}")
        gc.collect()
    X = pd.concat(chunks, axis=0)
    y = df['action_label']
    return X, y

def lgb_objective(trial, X_train, y_train, X_val, y_val, n_classes):
    param = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.5)
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        param,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    return model.best_score['valid_0']['multi_logloss']

def train_model(df: pd.DataFrame, cfg: dict) -> Dict:
    logger.info("Preparing features...")
    X, y = prepare_features_chunked(df)
    n_classes = len(np.unique(y))
    logger.info(f"Number of classes: {n_classes}")
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    best_score = float('inf')
    if len(y) > 1000:
        logger.info("Running Bayesian optimization...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold + 1}/{n_splits}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: lgb_objective(trial, X_train, y_train, X_val, y_val, n_classes),
                n_trials=cfg['ml']['lgbm']['max_evals'] // n_splits
            )
            if study.best_value < best_score:
                best_score = study.best_value
                best_params = study.best_params
        logger.info(f"\nBest validation score: {best_score:.4f}")
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
    final_params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        **best_params
    }
    logger.info("\nTraining final model...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(cfg['ml']['lgbm']['early_stopping']), lgb.log_evaluation(100)]
    )
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    logger.info("\nTop 20 most important features:")
    logger.info(importance_df.head(20))
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model.save_model(output_dir / "lgbm_model.txt")
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    return {
        'model': model,
        'feature_importance': importance_df,
        'best_params': best_params
    }

def main():
    cfg = load_config("Algorithm1/config.yaml")
    logger.info("Loading processed data...")
    df = pd.read_parquet("processed_data/processed_data.parquet")
    logger.info("Generating action-based labels...")
    label_gen = LabelGenerator(cfg)
    df = label_gen.generate_action_labels(df)
    logger.info("Starting model training...")
    results = train_model(df, cfg)
    logger.info("Training complete!")
    logger.info(f"Model saved to models/lgbm_model.txt")
    logger.info(f"Feature importance saved to models/feature_importance.csv")

if __name__ == "__main__":
    main() 