import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from itertools import combinations
from sklearn.metrics import precision_score, recall_score
import json

logger = logging.getLogger(__name__)

def find_indicator_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that might be indicators (containing specific keywords)."""
    indicator_keywords = ['break', 'cross', 'signal', 'alert', 'pattern', 'bos', 'smc']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in indicator_keywords)]

def evaluate_rule(df: pd.DataFrame, indicators: List[str], min_precision: float = 0.7, min_recall: float = 0.1) -> Dict:
    """
    Evaluate a combination of indicators as a potential rule.
    
    Args:
        df: DataFrame with indicator columns and labels
        indicators: List of indicator column names to evaluate
        min_precision: Minimum precision required for a valid rule
        min_recall: Minimum recall required for a valid rule
        
    Returns:
        Dictionary with rule details if valid, None otherwise
    """
    # Create rule signal (all indicators must be True)
    rule_signal = df[indicators].all(axis=1)
    
    # Calculate metrics
    precision = precision_score(df['label'] != 0, rule_signal)
    recall = recall_score(df['label'] != 0, rule_signal)
    
    if precision >= min_precision and recall >= min_recall:
        # Calculate additional statistics
        rule_df = df[rule_signal]
        success_rate = (rule_df['label'] != 0).mean()
        avg_move = rule_df['move_size'].mean()
        avg_time = rule_df['time_to_move'].mean()
        
        return {
            'indicators': indicators,
            'precision': float(precision),
            'recall': float(recall),
            'success_rate': float(success_rate),
            'avg_move_size': float(avg_move),
            'avg_time_to_move': float(avg_time),
            'signal_count': int(rule_signal.sum())
        }
    
    return None

def mine_rules(df: pd.DataFrame, cfg: dict) -> List[Dict]:
    """
    Mine indicator combinations that precede large price moves.
    
    Args:
        df: DataFrame with indicator columns and labels
        cfg: Configuration dictionary with rule mining parameters
        
    Returns:
        List of valid rules with their statistics
    """
    logger.info("Mining rules...")
    
    # Get parameters from config
    min_precision = cfg['rule_mining']['min_precision']
    min_recall = cfg['rule_mining']['min_recall']
    max_indicators = cfg['rule_mining']['max_indicators']
    
    # Find indicator columns
    indicator_cols = find_indicator_columns(df)
    logger.info(f"Found {len(indicator_cols)} potential indicator columns")
    
    # Initialize results
    valid_rules = []
    
    # Try combinations of 2 to max_indicators
    for n in range(2, min(max_indicators + 1, len(indicator_cols) + 1)):
        logger.info(f"Trying combinations of {n} indicators...")
        
        for combo in combinations(indicator_cols, n):
            rule = evaluate_rule(df, list(combo), min_precision, min_recall)
            if rule:
                valid_rules.append(rule)
    
    # Sort rules by precision * recall
    valid_rules.sort(key=lambda x: x['precision'] * x['recall'], reverse=True)
    
    # Log results
    logger.info(f"\nFound {len(valid_rules)} valid rules")
    if valid_rules:
        logger.info("\nTop 5 rules:")
        for i, rule in enumerate(valid_rules[:5], 1):
            logger.info(f"\nRule {i}:")
            logger.info(f"Indicators: {', '.join(rule['indicators'])}")
            logger.info(f"Precision: {rule['precision']:.3f}")
            logger.info(f"Recall: {rule['recall']:.3f}")
            logger.info(f"Success Rate: {rule['success_rate']:.3f}")
            logger.info(f"Avg Move Size: ${rule['avg_move_size']:.2f}")
            logger.info(f"Avg Time to Move: {rule['avg_time_to_move']:.2f} minutes")
            logger.info(f"Signal Count: {rule['signal_count']}")
    
    return valid_rules 