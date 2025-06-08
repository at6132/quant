import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import lightgbm as lgb
import optuna
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

def train_lgbm_model(df: pd.DataFrame, config: Dict) -> lgb.Booster:
    """Train LightGBM model with hyperparameter optimization."""
    logger.info("Training LightGBM model...")
    
    # Prepare data - only use numeric columns
    feature_cols = [col for col in df.columns 
                   if col not in ['label', 'open', 'high', 'low', 'close'] 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    logger.info(f"Using {len(feature_cols)} numeric features for training")
    X = df[feature_cols]
    y = df['label']
    
    # Encode labels to be in [0, num_classes)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Get number of classes
    n_classes = len(label_encoder.classes_)
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=config['backtesting']['n_splits'])
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'objective': 'multiclass',
            'num_class': n_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
            'verbose': -1  # Disable logging
        }
        
        # Cross validation
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Create callbacks
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=config['models']['lgbm']['early_stopping_rounds'],
                    verbose=False
                ),
                lgb.log_evaluation(period=0)  # Disable logging
            ]
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=config['models']['lgbm']['num_boost_round'],
                callbacks=callbacks
            )
            
            # Evaluate
            y_pred = model.predict(X_val).argmax(axis=1)
            score = f1_score(y_val, y_pred, average='weighted')
            scores.append(score)
        
        return np.mean(scores)
    
    # Optimize hyperparameters
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config['models']['lgbm']['n_trials'])
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1  # Disable logging
    })
    
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=config['models']['lgbm']['num_boost_round']
    )
    
    # Store label encoder in model for later use
    final_model.label_encoder = label_encoder
    
    logger.info("LightGBM training complete")
    return final_model

def train_tft_model(df: pd.DataFrame, config: Dict) -> Any:
    """Train Temporal Fusion Transformer model."""
    logger.info("Training TFT model...")
    
    # Prepare data - only use numeric columns and handle missing values
    feature_cols = [col for col in df.columns 
                   if col not in ['label', 'open', 'high', 'low', 'close'] 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    # Fill missing values with 0
    df = df.copy()
    df[feature_cols] = df[feature_cols].fillna(0)
    
    logger.info(f"Using {len(feature_cols)} numeric features for TFT training")
    
    # Create time index
    df['time_idx'] = np.arange(len(df))
    
    # Create training dataset
    training = TimeSeriesDataSet(
        data=df,
        time_idx='time_idx',
        target='label',
        group_ids=['time_idx'],  # No grouping needed for single series
        min_encoder_length=config['models']['tft']['encoder_length'] // 2,
        max_encoder_length=config['models']['tft']['encoder_length'],
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=['time_idx']),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=len(df) // 2)
    
    # Create dataloaders
    batch_size = config['models']['tft']['batch_size']
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # Define model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=config['models']['tft']['max_epochs'],
        accelerator='auto',
        enable_model_summary=True,
        gradient_clip_val=0.1,
    )
    
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    logger.info("TFT training complete")
    return tft

def train_models(df: pd.DataFrame, config: Dict) -> Dict:
    """Train all models."""
    logger.info("Starting model training...")
    
    # Train LightGBM
    lgbm_model = train_lgbm_model(df, config)
    
    # Train TFT
    tft_model = train_tft_model(df, config)
    
    return {
        'lgbm': lgbm_model,
        'tft': tft_model
    } 