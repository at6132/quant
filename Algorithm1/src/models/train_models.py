import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import lightgbm as lgb
import optuna
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss, CrossEntropy
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

def train_lgbm_model(df: pd.DataFrame, config: Dict) -> Any:
    """Train LightGBM model with hyperparameter optimization"""
    try:
        # Prepare data
        df = df.copy()
        # Ensure DataFrame has a proper index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Get numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in ["label", "time_idx", "group_id"]]
        logger.info(f"Using {len(numeric_features)} numeric features for training")
        
        # Prepare data for LightGBM
        X = df[numeric_features]
        y = df["label"]
        
        # Encode labels with sklearn's LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Create study for hyperparameter optimization
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            params = {
                "objective": "multiclass",
                "num_class": len(le.classes_),
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0)
            }
            
            # Create and train model
            model = lgb.LGBMClassifier(**params)
            
            # Use TimeSeriesSplit for validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
                # Skip split if not all classes are present in training set
                if len(np.unique(y_train)) < len(le.classes_):
                    continue
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='multi_logloss',
                    callbacks=[lgb.early_stopping(50)]
                )
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)
            
            if len(scores) == 0:
                return 0.0
            return np.mean(scores)
        
        # Run optimization
        study.optimize(objective, n_trials=config["models"]["lgbm"]["n_trials"])
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            "objective": "multiclass",
            "num_class": len(le.classes_),
            "metric": "multi_logloss",
            "boosting_type": "gbdt"
        })
        
        # Split data for final training
        validation_size = int(config["models"]["lgbm"]["validation_size"] * len(X))
        for val_size in range(validation_size, 0, -1):
            X_train = X.iloc[:-val_size]
            X_val = X.iloc[-val_size:]
            y_train = y_encoded[:-val_size]
            y_val = y_encoded[-val_size:]
            if len(np.unique(y_train)) == len(le.classes_):
                break
        else:
            # If no split found, use all data for training and skip validation
            X_train = X
            y_train = y_encoded
            X_val = None
            y_val = None
        
        final_model = lgb.LGBMClassifier(**best_params)
        if X_val is not None and y_val is not None and len(X_val) > 0:
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(50)]
            )
        else:
            final_model.fit(X_train, y_train)
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('artefacts/feature_importance.csv', index=False)
        logger.info("Feature importance saved to artefacts/feature_importance.csv")
        
        return final_model
        
    except Exception as e:
        logger.error(f"Error in LightGBM training: {str(e)}")
        raise e

def train_tft_model(df: pd.DataFrame, config: Dict) -> Any:
    """Train TFT model with GPU support and error handling"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Configure CUDA settings
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
        
        # Prepare data
        df = df.copy()
        print(f"Initial DataFrame shape: {df.shape}")
        
        # Create timestamp column first
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
        else:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='15S')
        
        # Clean data
        df = df.replace('', np.nan)
        print(f"After replacing empty strings: {df.shape}")

        # Replace inf/-inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Convert object columns to numeric where possible
        object_cols = df.select_dtypes(include=['object']).columns
        print(f"Found {len(object_cols)} object columns")
        for col in object_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric")
            except Exception as e:
                print(f"Failed to convert {col} to numeric: {str(e)}")
        
        print(f"After converting to numeric: {df.shape}")
        print(f"NaN count after numeric conversion: {df.isna().sum().sum()}")
        
        # Fill NaN values with 0 for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        print(f"After filling NaN values: {df.shape}")
        print(f"NaN count after filling: {df.isna().sum().sum()}")
        
        # Drop any remaining NaN rows
        df = df.dropna()
        print(f"After dropping NaN rows: {df.shape}")
        
        # Get numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Found {len(numeric_features)} numeric features")
        
        if len(numeric_features) == 0:
            raise ValueError("No valid numeric features found after cleaning")
        
        # Create time_idx and group_id
        df['time_idx'] = np.arange(len(df))
        df['group_id'] = 0  # Single group for now
        
        # Ensure label is int64 for PyTorch CrossEntropy
        df['label'] = df['label'].astype(np.int64)
        
        # Split data
        validation_size = config['models']['tft']['validation_size']
        if len(df) <= validation_size:
            validation_size = max(1, len(df) // 2)
        train_data = df.iloc[:-validation_size]
        val_data = df.iloc[-validation_size:]
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        
        # Create TimeSeriesDataSet
        training = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="label",
            group_ids=["group_id"],
            min_encoder_length=config['models']['tft']['encoder_length'],
            max_encoder_length=config['models']['tft']['encoder_length'],
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
        
        validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=True)
        
        # Create dataloaders
        batch_size = config['models']['tft']['batch_size']
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        
        # Create model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=config['models']['tft']['learning_rate'],
            hidden_size=config['models']['tft']['hidden_size'],
            attention_head_size=config['models']['tft']['attention_head_size'],
            dropout=config['models']['tft']['dropout'],
            hidden_continuous_size=config['models']['tft']['hidden_continuous_size'],
            loss=CrossEntropy(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config['models']['tft']['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    mode='min'
                )
            ]
        )
        
        # Train model
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return tft
        
    except Exception as e:
        print(f"Error in TFT training: {str(e)}")
        raise e

def train_tft_model_cpu(df, config):
    """Train TFT model on CPU with reduced complexity"""
    try:
        # Prepare data
        df = df.copy()
        # Ensure DataFrame has a proper index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure label is int64 for PyTorch CrossEntropy
        df['label'] = df['label'].astype(np.int64)
        
        df["time_idx"] = np.arange(len(df))
        df["group_id"] = df.index.date.astype(str)
        
        # Clean data: replace empty strings with NaN, convert to numeric where possible
        df = df.replace('', np.nan)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='ignore')
        df = df.fillna(method="ffill").fillna(method="bfill")
        df = df.dropna()
        
        # Create training and validation datasets
        training_cutoff = df["time_idx"].max() - config["models"]["tft"]["encoder_length"]
        
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="label",
            group_ids=["group_id"],
            min_encoder_length=config["models"]["tft"]["encoder_length"],
            max_encoder_length=config["models"]["tft"]["encoder_length"],
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[col for col in df.columns if col not in ["time_idx", "group_id", "label"]],
            target_normalizer=GroupNormalizer(groups=["group_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_dtype=torch.long,
        )
        
        validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.time_idx > training_cutoff], predict=True, stop_randomization=True)
        
        # Create dataloaders with smaller batch size
        batch_size = min(16, config["models"]["tft"]["batch_size"])
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        
        # Create model with reduced complexity
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=config["models"]["tft"]["learning_rate"],
            hidden_size=min(64, config["models"]["tft"]["hidden_size"]),
            attention_head_size=min(4, config["models"]["tft"]["attention_head_size"]),
            dropout=config["models"]["tft"]["dropout"],
            hidden_continuous_size=min(32, config["models"]["tft"]["hidden_continuous_size"]),
            loss=CrossEntropy(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # Create trainer for CPU
        trainer = pl.Trainer(
            max_epochs=config["models"]["tft"]["max_epochs"],
            accelerator="cpu",
            devices=1,
            gradient_clip_val=0.1,
            limit_train_batches=50,
            callbacks=[EarlyStopping(monitor="val_loss")]
        )
        
        # Train model
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return tft
        
    except Exception as e:
        logger.error(f"Error in CPU training: {str(e)}")
        raise e

def train_models(df: pd.DataFrame, config: Dict) -> Dict:
    """Train both LGBM and TFT models"""
    try:
        logger.info("Starting model training...")
        
        # Train LGBM model
        logger.info("Training LightGBM model...")
        lgbm_model = train_lgbm_model(df, config)
        
        # Train TFT model
        logger.info("Training TFT model...")
        tft_model = train_tft_model(df, config)
        
        # Save models
        logger.info("Saving models...")
        lgbm_model.save_model('models/lgbm_model.txt')
        torch.save(tft_model.state_dict(), 'models/tft_model.pt')
        
        return {
            'lgbm': lgbm_model,
            'tft': tft_model
        }
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise e 