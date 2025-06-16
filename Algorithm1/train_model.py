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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
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

def prepare_data(df: pd.DataFrame, cfg: Dict) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Prepare data for TFT model training."""
    logger.info("Preparing data for TFT model...")
    
    # Add time index and group ID
    df = df.copy()
    df['time_idx'] = np.arange(len(df))
    df['group_id'] = 0  # single group
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['time_idx', 'group_id', 'action_label', 'label']]
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="action_label",
        group_ids=["group_id"],
        min_encoder_length=cfg['models']['tft']['encoder_length'],
        max_encoder_length=cfg['models']['tft']['encoder_length'],
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True)
    
    return training, validation

def train_tft_model(training: TimeSeriesDataSet, validation: TimeSeriesDataSet, cfg: Dict) -> Tuple[TemporalFusionTransformer, pl.Trainer]:
    """Train TFT model with improved configuration."""
    logger.info("Training TFT model...")
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=cfg['models']['tft']['batch_size'],
        num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=cfg['models']['tft']['batch_size'],
        num_workers=0
    )
    
    # Initialize model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=cfg['models']['tft']['learning_rate'],
        hidden_size=cfg['models']['tft']['hidden_size'],
        attention_head_size=cfg['models']['tft']['attention_head_size'],
        dropout=cfg['models']['tft']['dropout'],
        hidden_continuous_size=cfg['models']['tft']['hidden_continuous_size'],
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
    
    # Set up logging
    loggers = []
    if cfg['logging']['tensorboard']:
        loggers.append(TensorBoardLogger("logs/"))
    if cfg['logging']['wandb']:
        wandb_logger = WandbLogger(project="trading_model")
        loggers.append(wandb_logger)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg['models']['tft']['train']['patience'],
            mode="min"
        ),
        ModelCheckpoint(
            dirpath=cfg['model']['models_dir'],
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            save_top_k=cfg['model']['save_top_k'],
            monitor="val_loss",
            mode="min"
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg['models']['tft']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=cfg['models']['tft']['train']['gradient_clip_val'],
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=cfg['models']['tft']['train']['accumulate_grad_batches']
    )
    
    # Train model
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    return tft, trainer

def evaluate_model(model: TemporalFusionTransformer, val_dataloader: torch.utils.data.DataLoader) -> Dict:
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    
    # Get predictions
    predictions = model.predict(val_dataloader)
    actuals = torch.cat([y for x, y in val_dataloader])
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(actuals, predictions),
        "precision": precision_score(actuals, predictions, average='weighted'),
        "recall": recall_score(actuals, predictions, average='weighted'),
        "f1_score": f1_score(actuals, predictions, average='weighted')
    }
    
    logger.info("Model evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def save_model_artifacts(model: TemporalFusionTransformer, trainer: pl.Trainer, metrics: Dict, cfg: Dict):
    """Save model artifacts and metadata."""
    logger.info("Saving model artifacts...")
    
    # Create artifacts directory
    artifacts_dir = Path("artefacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = artifacts_dir / "tft_model.ckpt"
    trainer.save_checkpoint(model_path)
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "tft",
        "version": cfg['model']['version'],
        "metrics": metrics,
        "config": {
            "learning_rate": cfg['models']['tft']['learning_rate'],
            "hidden_size": cfg['models']['tft']['hidden_size'],
            "attention_head_size": cfg['models']['tft']['attention_head_size'],
            "dropout": cfg['models']['tft']['dropout'],
            "encoder_length": cfg['models']['tft']['encoder_length'],
            "batch_size": cfg['models']['tft']['batch_size']
        }
    }
    
    metadata_path = artifacts_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model artifacts saved to {artifacts_dir}")

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
    """Main training pipeline."""
    # Load configuration
    cfg = load_config("config.yaml")
    
    # Load and prepare data
    df = pd.read_parquet("processed_data/features.parquet")
    
    # Generate labels
    label_generator = LabelGenerator(cfg)
    df = label_generator.generate_action_labels(df)
    
    # Prepare data for training
    training, validation = prepare_data(df, cfg)
    
    # Train model
    model, trainer = train_tft_model(training, validation, cfg)
    
    # Evaluate model
    metrics = evaluate_model(model, validation.to_dataloader(train=False, batch_size=cfg['models']['tft']['batch_size']))
    
    # Save artifacts
    save_model_artifacts(model, trainer, metrics, cfg)
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main() 