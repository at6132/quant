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

def generate_labels(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Generate labels based on future price movements.
    Label is 1 if price moves up by threshold, -1 if down by threshold, 0 otherwise.
    """
    horizon = cfg['label']['horizon_minutes']
    threshold = cfg['label']['dollar_threshold']
    
    # Calculate future price changes
    future_close = df['close'].shift(-horizon * 4)  # 4 bars per minute for 15s data
    price_change = future_close - df['close']
    
    # Generate labels
    labels = pd.Series(0, index=df.index)
    labels[price_change >= threshold] = 1
    labels[price_change <= -threshold] = -1
    
    return labels

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns, excluding non-feature columns."""
    exclude_cols = ['label', 'open', 'high', 'low', 'close', 'volume']
    return [col for col in df.columns if col not in exclude_cols]

def process_chunk(chunk: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Process a chunk of data, handling NaN and inf values."""
    # Select only feature columns
    X = chunk[feature_cols].copy()
    
    # Replace inf values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    X = X.fillna(0)
    
    return X

def prepare_features_chunked(df: pd.DataFrame, chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels for modeling, processing in chunks."""
    logger.info("Preparing features in chunks...")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")
    
    # Process data in chunks
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        processed_chunk = process_chunk(chunk, feature_cols)
        chunks.append(processed_chunk)
        logger.info(f"Processed chunk {i//chunk_size + 1}/{(len(df) + chunk_size - 1)//chunk_size}")
        
        # Force garbage collection
        gc.collect()
    
    # Combine chunks
    X = pd.concat(chunks, axis=0)
    y = df['label']
    
    return X, y

def lgb_objective(trial, X_train, y_train, X_val, y_val):
    """LightGBM objective function for Optuna."""
    param = {
        'objective': 'multiclass',
        'num_class': 3,
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
    
    # Adjust labels for LightGBM (0, 1, 2 instead of -1, 0, 1)
    y_train_adj = y_train + 1
    y_val_adj = y_val + 1
    
    train_data = lgb.Dataset(X_train, label=y_train_adj)
    val_data = lgb.Dataset(X_val, label=y_val_adj, reference=train_data)
    
    model = lgb.train(
        param,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    return model.best_score['valid_0']['multi_logloss']

def train_model(df: pd.DataFrame, cfg: dict) -> Dict:
    """Train LightGBM model with Bayesian optimization."""
    logger.info("Preparing features...")
    X, y = prepare_features_chunked(df)
    
    # Filter to rows with actual signals for training
    signal_mask = y != 0
    signal_ratio = signal_mask.sum() / len(y)
    logger.info(f"Signal ratio: {signal_ratio:.2%}")
    
    # Use time series split for validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    best_params = None
    best_score = -1
    
    # If we have enough signals, do optimization
    if signal_mask.sum() > 1000:
        logger.info("Running Bayesian optimization...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create study for this fold
            study = optuna.create_study(direction='minimize')
            
            study.optimize(
                lambda trial: lgb_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=cfg['ml']['lgbm']['max_evals'] // n_splits
            )
            
            if study.best_value > best_score:
                best_score = study.best_value
                best_params = study.best_params
                
        logger.info(f"\nBest validation score: {best_score:.4f}")
    
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
    logger.info("\nTraining final model...")
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
    
    logger.info("\nTop 20 most important features:")
    logger.info(importance_df.head(20))
    
    # Save model and feature importance
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
    # Load config
    cfg = load_config("config.yaml")
    
    # Load processed data
    logger.info("Loading processed data...")
    df = pd.read_parquet("processed_data/processed_data.parquet")
    
    # Generate labels
    logger.info("Generating labels...")
    df['label'] = generate_labels(df, cfg)
    
    # Train model
    logger.info("Starting model training...")
    results = train_model(df, cfg)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to models/lgbm_model.txt")
    logger.info(f"Feature importance saved to models/feature_importance.csv")

if __name__ == "__main__":
    main() 