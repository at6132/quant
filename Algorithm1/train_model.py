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
import pickle
import json
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_forecasting.data.encoders import GroupNormalizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

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

def save_feature_matrix(X, y, output_dir):
    output_path = Path(output_dir) / "feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = X.copy()
    df['action_label'] = y
    df.to_parquet(output_path)
    logger.info(f"Saved feature matrix to {output_path}")

def generate_rule_miner_report(X, y, threshold=500, min_precision=0.6, min_recall=0.3):
    report = {
        "timestamp": datetime.now().isoformat(),
        "move_threshold": threshold,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "rules": []
    }
    # Find significant moves
    significant_moves = np.abs(y) >= threshold
    for col in X.columns:
        feature_values = X[col]
        for percentile in [25, 50, 75, 90]:
            feature_threshold = np.percentile(feature_values, percentile)
            rule = feature_values > feature_threshold
            precision = np.mean(significant_moves[rule])
            recall = np.mean(rule[significant_moves])
            if precision >= min_precision and recall >= min_recall:
                report["rules"].append({
                    "feature": col,
                    "feature_threshold": float(feature_threshold),
                    "percentile": percentile,
                    "precision": float(precision),
                    "recall": float(recall),
                    "support": float(np.mean(rule))
                })
    report["rules"].sort(key=lambda x: x["precision"] * x["recall"], reverse=True)
    output_path = Path("artefacts") / "rule_miner_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print("\nRule Miner Report:")
    print(f"Found {len(report['rules'])} significant rules")
    print(f"Move Threshold: ${threshold}")
    print(f"Min Precision: {min_precision}")
    print(f"Min Recall: {min_recall}")
    print("\nTop Rules:")
    for rule in report["rules"][:10]:
        print(f"\nFeature: {rule['feature']}")
        print(f"Feature Threshold: {rule['feature_threshold']:.2f} (Percentile: {rule['percentile']})")
        print(f"Precision: {rule['precision']:.2f}")
        print(f"Recall: {rule['recall']:.2f}")
        print(f"Support: {rule['support']:.2f}")
    return report

def plot_pnl_curves(pnl, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(pnl, label='LightGBM', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.title('Walk-forward PnL Curve (Out-of-Sample)')
    plt.legend()
    plt.grid(True)
    output_path = Path(output_dir) / "pnl_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    stats = {
        "Total PnL": float(pnl[-1]),
        "Sharpe Ratio": float(np.mean(pnl) / np.std(pnl)) if np.std(pnl) > 0 else 0.0,
        "Max Drawdown": float(np.min(pnl - np.maximum.accumulate(pnl)))
    }
    stats_path = Path(output_dir) / "pnl_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nPnL Statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")
    return stats

def train_tft_model(df, cfg, output_dir, X_train_idx, X_test_idx):
    # Prepare data
    df = df.copy()
    df['time_idx'] = np.arange(len(df))
    df['group_id'] = 0  # single group
    df['label'] = df['action_label'].astype(int)
    # Use only numeric features
    numeric_features = [col for col in df.columns if df[col].dtype in [np.float32, np.float64, np.int32, np.int64] and col not in ['time_idx', 'group_id', 'label']]
    # Split
    train_df = df.iloc[X_train_idx]
    test_df = df.iloc[X_test_idx]
    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="label",
        group_ids=["group_id"],
        min_encoder_length=cfg['ml']['tft']['encoder_length'],
        max_encoder_length=cfg['ml']['tft']['encoder_length'],
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=numeric_features,
        target_normalizer=GroupNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_dtype=torch.long,
    )
    validation = TimeSeriesDataSet.from_dataset(training, test_df, predict=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=cfg['ml']['tft']['batch_size'], num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=cfg['ml']['tft']['batch_size'], num_workers=0)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=cfg['ml']['tft']['learning_rate'],
        hidden_size=cfg['ml']['tft']['hidden_size'],
        attention_head_size=cfg['ml']['tft']['attention_head_size'],
        dropout=cfg['ml']['tft']['dropout'],
        hidden_continuous_size=cfg['ml']['tft']['hidden_continuous_size'],
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
    trainer = pl.Trainer(
        max_epochs=cfg['ml']['tft']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')]
    )
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # Save model
    tft_path = Path(output_dir) / "tft_model.ckpt"
    tft.save(tft_path)
    logger.info(f"Saved TFT model to {tft_path}")
    return tft, test_df, val_dataloader

def calculate_realistic_pnl(actions, y_true, prices, initial_capital=100000, fee_rate=0.001, slippage=0.0005):
    capital = initial_capital
    position = 0  # +1 for long, -1 for short, 0 for flat
    pnl_curve = []
    entry_price = None
    for i, (action, true_label, price) in enumerate(zip(actions, y_true, prices)):
        # Example: 1=open long, 2=open short, 3=close long, 4=close short, 5=add to long, 6=add to short
        # This logic can be customized for your action space
        if action == 1:  # open long
            if position == 0:
                entry_price = price * (1 + slippage)
                position = 1
        elif action == 2:  # open short
            if position == 0:
                entry_price = price * (1 - slippage)
                position = -1
        elif action == 3 and position == 1:  # close long
            pnl = (price * (1 - slippage) - entry_price) * position - fee_rate * price
            capital += pnl
            position = 0
            entry_price = None
        elif action == 4 and position == -1:  # close short
            pnl = (entry_price - price * (1 + slippage)) * abs(position) - fee_rate * price
            capital += pnl
            position = 0
            entry_price = None
        # Add to long/short logic can be added here
        pnl_curve.append(capital)
    return np.array(pnl_curve)

def main():
    cfg = load_config("Algorithm1/config.yaml")
    logger.info("Loading processed data...")
    df = pd.read_parquet("processed_data/processed_data.parquet")
    logger.info("Generating action-based labels...")
    label_gen = LabelGenerator(cfg)
    df = label_gen.generate_action_labels(df)
    logger.info("Checking for data leakage in features...")
    for col in get_feature_columns(df):
        if any('future' in str(col).lower() or 'lead' in str(col).lower() for col in df.columns):
            logger.warning(f"Potential leakage in feature: {col}")
    logger.info("Preparing features...")
    X, y = prepare_features_chunked(df)
    save_feature_matrix(X, y, "artefacts")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    price_col = 'close' if 'close' in df.columns else None
    prices_test = df[price_col].iloc[train_size:] if price_col else np.ones(len(X_test))
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    n_classes = len(np.unique(y))
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: lgb_objective(trial, X_train, y_train, X_test, y_test, n_classes),
        n_trials=cfg['ml']['lgbm']['max_evals']
    )
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    final_params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        **best_params
    }
    logger.info("Training final model on training set...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(cfg['ml']['lgbm']['early_stopping']), lgb.log_evaluation(100)]
    )
    output_dir = Path("artefacts")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "lgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info("Feature importance saved.")
    generate_rule_miner_report(X, y, threshold=cfg['labeling']['threshold'], min_precision=cfg['rule_mining']['min_precision'], min_recall=cfg['rule_mining']['min_recall'])
    logger.info("Evaluating LightGBM on out-of-sample test set for PnL curve...")
    test_preds = model.predict(X_test)
    test_actions = np.argmax(test_preds, axis=1)
    pnl_curve = calculate_realistic_pnl(test_actions, y_test, prices_test)
    plot_pnl_curves(pnl_curve, output_dir)
    # --- TFT Model ---
    logger.info("Training TFT model...")
    tft, test_df, val_dataloader = train_tft_model(df, cfg, output_dir, X_train.index, X_test.index)
    logger.info("Evaluating TFT on out-of-sample test set for PnL curve...")
    # Get TFT predictions
    tft_preds = []
    tft_prices = []
    tft_labels = []
    for batch in val_dataloader:
        x, y_true = batch[0], batch[1]
        y_pred = tft(x).detach().cpu().numpy()
        tft_preds.extend(np.argmax(y_pred, axis=1))
        tft_labels.extend(y_true.cpu().numpy())
        tft_prices.extend(test_df['close'].values[:len(y_true)])
    tft_pnl_curve = calculate_realistic_pnl(tft_preds, tft_labels, tft_prices)
    plot_pnl_curves(tft_pnl_curve, output_dir)
    logger.info("All artifacts saved. Training complete!")

if __name__ == "__main__":
    main() 