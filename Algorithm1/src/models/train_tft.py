import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiLoss
from pytorch_forecasting.data import GroupNormalizer
import optuna
import joblib

logger = logging.getLogger(__name__)

def prepare_tft_data(df: pd.DataFrame, cfg: dict) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Prepare data for TFT model training."""
    # Create time index
    df = df.copy()
    df['time_idx'] = np.arange(len(df))
    
    # Get feature columns
    exclude_cols = ['label', 'move_size', 'time_to_move']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create training dataset
    training = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="label",
        group_ids=["time_idx"],  # Each row is its own group
        min_encoder_length=cfg['models']['tft']['encoder_length'],
        max_encoder_length=cfg['models']['tft']['encoder_length'],
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=feature_cols,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["label"],
        target_normalizer=GroupNormalizer(groups=["time_idx"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    return training, validation

def objective(trial: optuna.Trial, training: TimeSeriesDataSet, validation: TimeSeriesDataSet) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    param = {
        'hidden_size': trial.suggest_int('hidden_size', 16, 128),
        'attention_head_size': trial.suggest_int('attention_head_size', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'hidden_continuous_size': trial.suggest_int('hidden_continuous_size', 8, 64),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
    }
    
    # Create model
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=param['learning_rate'],
        hidden_size=param['hidden_size'],
        attention_head_size=param['attention_head_size'],
        dropout=param['dropout'],
        hidden_continuous_size=param['hidden_continuous_size'],
        loss=MultiLoss([torch.nn.MSELoss(), torch.nn.L1Loss()]),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='auto',
        gradient_clip_val=0.1,
    )
    
    # Train model
    trainer.fit(model, train_dataloaders=training.to_dataloader(), val_dataloaders=validation.to_dataloader())
    
    # Get validation score
    predictions = model.predict(validation.to_dataloader())
    actuals = torch.cat([y for x, y in validation.to_dataloader()])
    score = torch.mean((predictions - actuals) ** 2).item()
    
    return -score  # Negative because we want to maximize

def train_tft_model(df: pd.DataFrame, cfg: dict) -> TemporalFusionTransformer:
    """
    Train a Temporal Fusion Transformer model with hyperparameter optimization.
    
    Args:
        df: DataFrame with features and labels
        cfg: Configuration dictionary with model parameters
        
    Returns:
        Trained TFT model
    """
    logger.info("Training TFT model...")
    
    # Prepare data
    training, validation = prepare_tft_data(df, cfg)
    logger.info(f"Training on {len(training)} samples with {len(training.variables)} features")
    
    # Create study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, training, validation), n_trials=cfg['models']['tft']['n_trials'])
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=best_params['learning_rate'],
        hidden_size=best_params['hidden_size'],
        attention_head_size=best_params['attention_head_size'],
        dropout=best_params['dropout'],
        hidden_continuous_size=best_params['hidden_continuous_size'],
        loss=MultiLoss([torch.nn.MSELoss(), torch.nn.L1Loss()]),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        gradient_clip_val=0.1,
    )
    
    trainer.fit(model, train_dataloaders=training.to_dataloader(), val_dataloaders=validation.to_dataloader())
    
    # Save model and parameters
    model.save("models/tft_model.ckpt")
    joblib.dump(best_params, "models/tft_params.pkl")
    
    return model 