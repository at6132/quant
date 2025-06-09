from typing import List, Tuple, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def evaluate_rules_parallel(df: pd.DataFrame, 
                          rules: List[Tuple[str, str]], 
                          batch_size: int = 1000,
                          max_workers: int = 4) -> pd.DataFrame:
    """
    Evaluate rules in parallel with memory-efficient batching.
    
    Args:
        df: DataFrame with price and indicator data
        rules: List of (indicator1, indicator2) tuples
        batch_size: Number of rules to process in each batch
        max_workers: Maximum number of parallel workers
    
    Returns:
        DataFrame with rule evaluation results
    """
    logger.info(f"Evaluating {len(rules)} rules in parallel with batch size {batch_size}")
    
    # Process rules in batches to manage memory
    all_results = []
    for i in range(0, len(rules), batch_size):
        batch_rules = rules[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(rules) + batch_size - 1)//batch_size}")
        
        # Create a pool of workers for this batch
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch of rules for evaluation
            futures = [
                executor.submit(evaluate_rule, df.copy(), rule)
                for rule in batch_rules
            ]
            
            # Collect results as they complete
            batch_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating rule: {e}")
                    continue
        
        # Add batch results to overall results
        all_results.extend(batch_results)
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Combine all results
    if not all_results:
        logger.warning("No valid rules found")
        return pd.DataFrame()
    
    results_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"Successfully evaluated {len(results_df)} rules")
    return results_df

def evaluate_rule(df: pd.DataFrame, rule: Tuple[str, str]) -> Optional[pd.DataFrame]:
    """
    Evaluate a single rule with memory-efficient operations.
    
    Args:
        df: DataFrame with price and indicator data
        rule: Tuple of (indicator1, indicator2)
    
    Returns:
        DataFrame with rule evaluation results or None if invalid
    """
    try:
        ind1, ind2 = rule
        
        # Create rule name
        rule_name = f"{ind1}_{ind2}"
        
        # Calculate rule signal
        df[rule_name] = (df[ind1] > df[ind2]).astype(int)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate rule performance
        rule_returns = df[rule_name].shift(1) * df['returns']
        
        # Calculate metrics
        total_return = rule_returns.sum()
        sharpe = rule_returns.mean() / rule_returns.std() if rule_returns.std() != 0 else 0
        win_rate = (rule_returns > 0).mean()
        
        # Create result row
        result = pd.DataFrame({
            'rule': [rule_name],
            'total_return': [total_return],
            'sharpe': [sharpe],
            'win_rate': [win_rate]
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating rule {rule}: {e}")
        return None

def generate_rules(indicators: List[str], max_combinations: int = 1000) -> List[Tuple[str, str]]:
    """
    Generate rules by combining indicators.
    
    Args:
        indicators: List of indicator names
        max_combinations: Maximum number of combinations to generate
    
    Returns:
        List of (indicator1, indicator2) tuples
    """
    rules = []
    for i, ind1 in enumerate(indicators):
        for ind2 in indicators[i+1:]:
            rules.append((ind1, ind2))
            if len(rules) >= max_combinations:
                return rules
    return rules 