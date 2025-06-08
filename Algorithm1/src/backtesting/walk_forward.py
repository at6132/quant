import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from pytorch_forecasting import TemporalFusionTransformer
import torch
import json

logger = logging.getLogger(__name__)

def evaluate_rule_signals(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """Evaluate rule-based signals."""
    signals = pd.DataFrame(index=df.index)
    
    for rule in rules:
        # Create signal for this rule
        rule_signal = df[rule['indicators']].all(axis=1)
        signals[f"rule_{rule['indicators'][0]}"] = rule_signal
    
    return signals

def evaluate_lgbm_signals(df: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
    """Evaluate LightGBM model signals."""
    # Get feature columns
    exclude_cols = ['label', 'move_size', 'time_to_move']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Get predictions
    preds = model.predict(df[feature_cols])
    pred_labels = np.argmax(preds, axis=1) - 1  # Convert to -1, 0, 1
    
    # Create signals
    signals = pd.DataFrame(index=df.index)
    signals['lgbm_long'] = pred_labels == 1
    signals['lgbm_short'] = pred_labels == -1
    
    return signals

def evaluate_tft_signals(df: pd.DataFrame, model: TemporalFusionTransformer) -> pd.DataFrame:
    """Evaluate TFT model signals."""
    # Prepare data
    df = df.copy()
    df['time_idx'] = np.arange(len(df))
    
    # Get feature columns
    exclude_cols = ['label', 'move_size', 'time_to_move']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create dataset
    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="label",
        group_ids=["time_idx"],
        min_encoder_length=model.hparams.encoder_length,
        max_encoder_length=model.hparams.encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=feature_cols,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["label"],
        target_normalizer=model.target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Get predictions
    preds = model.predict(dataset.to_dataloader())
    pred_labels = torch.round(preds).int().numpy() - 1  # Convert to -1, 0, 1
    
    # Create signals
    signals = pd.DataFrame(index=df.index)
    signals['tft_long'] = pred_labels == 1
    signals['tft_short'] = pred_labels == -1
    
    return signals

def calculate_pnl(signals: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate PnL for each signal."""
    pnl = pd.DataFrame(index=signals.index)
    
    for col in signals.columns:
        # Get signal direction
        if 'long' in col:
            direction = 1
        elif 'short' in col:
            direction = -1
        else:
            direction = 1  # Default to long for rules
        
        # Calculate PnL
        signal = signals[col]
        price_change = df['close'].diff()
        pnl[col] = signal.shift(1) * direction * price_change
    
    return pnl

def run_walk_forward(df: pd.DataFrame, rules: List[Dict], lgbm_model: lgb.Booster, tft_model: TemporalFusionTransformer, cfg: dict) -> Dict:
    """
    Run walk-forward backtesting for all models.
    
    Args:
        df: DataFrame with features and labels
        rules: List of mined rules
        lgbm_model: Trained LightGBM model
        tft_model: Trained TFT model
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with backtest results
    """
    logger.info("Running walk-forward backtest...")
    
    # Initialize results
    results = {
        'rules': {},
        'lgbm': {},
        'tft': {}
    }
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=cfg['backtesting']['n_splits'])
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        logger.info(f"\nFold {i+1}/{cfg['backtesting']['n_splits']}")
        
        # Split data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Evaluate signals
        rule_signals = evaluate_rule_signals(test_df, rules)
        lgbm_signals = evaluate_lgbm_signals(test_df, lgbm_model)
        tft_signals = evaluate_tft_signals(test_df, tft_model)
        
        # Combine signals
        all_signals = pd.concat([rule_signals, lgbm_signals, tft_signals], axis=1)
        
        # Calculate PnL
        pnl = calculate_pnl(all_signals, test_df)
        
        # Calculate statistics for each signal
        for col in all_signals.columns:
            signal_pnl = pnl[col]
            
            # Calculate metrics
            total_return = signal_pnl.sum()
            sharpe = signal_pnl.mean() / signal_pnl.std() * np.sqrt(252 * 24 * 4)  # Annualized
            win_rate = (signal_pnl > 0).mean()
            profit_factor = abs(signal_pnl[signal_pnl > 0].sum() / signal_pnl[signal_pnl < 0].sum())
            max_drawdown = (signal_pnl.cumsum() - signal_pnl.cumsum().cummax()).min()
            
            # Store results
            if 'rule' in col:
                results['rules'][col] = {
                    'total_return': float(total_return),
                    'sharpe': float(sharpe),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor),
                    'max_drawdown': float(max_drawdown)
                }
            elif 'lgbm' in col:
                results['lgbm'][col] = {
                    'total_return': float(total_return),
                    'sharpe': float(sharpe),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor),
                    'max_drawdown': float(max_drawdown)
                }
            elif 'tft' in col:
                results['tft'][col] = {
                    'total_return': float(total_return),
                    'sharpe': float(sharpe),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor),
                    'max_drawdown': float(max_drawdown)
                }
    
    # Log results
    logger.info("\nBacktest Results:")
    for model_type, model_results in results.items():
        logger.info(f"\n{model_type.upper()}:")
        for signal, metrics in model_results.items():
            logger.info(f"\n{signal}:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    with open("artefacts/backtest_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results 