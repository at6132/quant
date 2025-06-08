import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    """Evaluate prediction performance."""
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    precision = (y_true & y_pred).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    recall = (y_true & y_pred).sum() / y_true.sum() if y_true.sum() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def backtest(df: pd.DataFrame, models: Dict, rules: List[Dict], config: Dict) -> Dict:
    """Run walk-forward backtest."""
    logger.info("Starting backtest...")
    
    # Get parameters
    n_splits = config['backtesting']['n_splits']
    initial_capital = config['backtesting']['initial_capital']
    
    # Initialize results
    results = {
        'splits': [],
        'overall': {},
        'rules': [],
        'models': {}
    }
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Track portfolio value
    portfolio_values = []
    current_capital = initial_capital
    
    # Walk-forward testing
    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        logger.info(f"\nSplit {i+1}/{n_splits}")
        
        # Split data
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        # Get predictions from models
        model_predictions = {}
        for name, model in models.items():
            if name == 'lgbm':
                # LightGBM predictions
                X_test = test_data.drop(['label', 'open', 'high', 'low', 'close'], axis=1)
                preds = model.predict(X_test)
                model_predictions[name] = (preds > 0.5).astype(int)
            elif name == 'tft':
                # TFT predictions
                # TODO: Implement TFT prediction
                pass
        
        # Get predictions from rules
        rule_predictions = []
        for rule in rules:
            # Create rule signal
            signal = 1
            for ind in rule['indicators']:
                ind_signal = (test_data[ind] > 0).astype(int)
                signal = signal & ind_signal
            
            # Calculate rule performance
            if signal.sum() > 0:
                precision = (test_data['label'] & signal).sum() / signal.sum()
                recall = (test_data['label'] & signal).sum() / test_data['label'].sum()
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                rule_predictions.append({
                    'indicators': rule['indicators'],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'signals': int(signal.sum())
                })
        
        # Combine predictions (simple majority vote)
        combined_preds = pd.DataFrame(model_predictions).mean(axis=1) > 0.5
        
        # Calculate returns
        returns = test_data['close'].pct_change()
        strategy_returns = returns * combined_preds.shift(1)
        
        # Update portfolio value
        portfolio_values.extend((1 + strategy_returns) * current_capital)
        current_capital = portfolio_values[-1]
        
        # Store split results
        split_results = {
            'train_size': len(train_data),
            'test_size': len(test_data),
            'model_performance': {
                name: evaluate_predictions(test_data['label'], preds)
                for name, preds in model_predictions.items()
            },
            'rule_performance': rule_predictions,
            'portfolio_value': current_capital
        }
        results['splits'].append(split_results)
    
    # Calculate overall metrics
    results['overall'] = {
        'initial_capital': initial_capital,
        'final_capital': current_capital,
        'total_return': (current_capital / initial_capital - 1) * 100,
        'portfolio_values': portfolio_values
    }
    
    # Store best rules
    results['rules'] = sorted(rule_predictions, 
                            key=lambda x: x['f1_score'], 
                            reverse=True)[:10]
    
    # Store model performance
    results['models'] = {
        name: evaluate_predictions(df['label'], preds)
        for name, preds in model_predictions.items()
    }
    
    logger.info("\nBacktest Results:")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Final Capital: ${current_capital:,.2f}")
    logger.info(f"Total Return: {results['overall']['total_return']:.2f}%")
    
    return results 