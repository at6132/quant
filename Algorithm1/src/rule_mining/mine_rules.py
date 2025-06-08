import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from itertools import combinations
from sklearn.metrics import precision_score, recall_score
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

logger = logging.getLogger(__name__)

def find_indicator_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that might be indicators (containing specific keywords)."""
    indicator_keywords = ['break', 'cross', 'signal', 'alert', 'pattern', 'bos', 'smc']
    return [col for col in df.columns if any(keyword in col.lower() for keyword in indicator_keywords)]

def evaluate_rule(df: pd.DataFrame, indicators: List[str], label_col: str, 
                 min_precision: float, min_recall: float) -> Dict:
    """Evaluate a single rule combination."""
    # Create rule signal using vectorized operations
    signal = pd.Series(True, index=df.index)
    for ind in indicators:
        # Handle both numeric and string indicators
        if pd.api.types.is_numeric_dtype(df[ind]):
            signal &= (df[ind] > 0)
        else:
            # For string indicators, check if they're non-empty
            signal &= (df[ind].notna() & (df[ind] != ''))
    
    # Skip if too few signals
    signal_sum = signal.sum()
    if signal_sum < 10:  # Minimum 10 signals to consider
        return None
    
    # Calculate metrics using vectorized operations
    true_positives = (df[label_col] & signal).sum()
    precision = true_positives / signal_sum
    recall = true_positives / df[label_col].sum()
    
    # Check if rule meets minimum criteria
    if precision >= min_precision and recall >= min_recall:
        return {
            'indicators': indicators,
            'precision': float(precision),  # Convert to float for JSON serialization
            'recall': float(recall),
            'support': float(signal_sum / len(df))
        }
    return None

def mine_rules(df: pd.DataFrame, config: Dict) -> List[Dict]:
    """Mine trading rules from the data."""
    logger.info("Mining rules...")
    
    # Get parameters from config
    min_precision = config['rule_mining']['min_precision']
    min_recall = config['rule_mining']['min_recall']
    max_indicators = config['rule_mining']['max_indicators']
    label_col = 'label'  # Assuming binary classification label
    
    # Get all indicator columns and their types
    indicator_cols = [col for col in df.columns 
                     if col not in ['open', 'high', 'low', 'close', 'label']]
    logger.info(f"Found {len(indicator_cols)} indicator columns")
    
    # Log column types for debugging
    for col in indicator_cols[:5]:  # Log first 5 columns
        logger.info(f"Column {col} type: {df[col].dtype}")
    
    # Initialize results
    rules = []
    max_rules = 100  # Limit total number of rules
    
    # Try different numbers of indicators
    for n in range(2, max_indicators + 1):
        if len(rules) >= max_rules:
            logger.info(f"Reached maximum number of rules ({max_rules})")
            break
            
        logger.info(f"Trying combinations of {n} indicators...")
        
        # Use ProcessPoolExecutor for parallel processing
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Process in smaller chunks
        chunk_size = 100  # Process 100 combinations at a time
        total_combs = len(list(combinations(indicator_cols, n)))
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            # Process combinations in chunks
            for i in range(0, total_combs, chunk_size):
                # Get chunk of combinations
                chunk_combs = list(combinations(indicator_cols, n))[i:i + chunk_size]
                
                # Submit chunk for processing
                futures = []
                for indicators in chunk_combs:
                    futures.append(
                        executor.submit(evaluate_rule, df, list(indicators), 
                                     label_col, min_precision, min_recall)
                    )
                
                # Collect results with progress bar
                for future in tqdm(futures, total=len(futures), 
                                 desc=f"Testing {n}-indicator rules ({i}/{total_combs})"):
                    result = future.result()
                    if result is not None:
                        rules.append(result)
                        if len(rules) >= max_rules:
                            break
                
                if len(rules) >= max_rules:
                    break
    
    # Sort rules by precision * recall
    rules.sort(key=lambda x: x['precision'] * x['recall'], reverse=True)
    
    # Log results
    logger.info(f"Found {len(rules)} valid rules")
    if rules:
        logger.info("Top 5 rules:")
        for i, rule in enumerate(rules[:5], 1):
            logger.info(f"{i}. Indicators: {rule['indicators']}")
            logger.info(f"   Precision: {rule['precision']:.3f}, Recall: {rule['recall']:.3f}")
    
    return rules 