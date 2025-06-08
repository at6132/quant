import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple

def evaluate_rule(df: pd.DataFrame, conditions: List[str], label_col: str = 'label') -> Dict:
    """Evaluate a rule's performance."""
    # Build the combined condition
    combined_condition = ' & '.join([f"({cond})" for cond in conditions])
    
    try:
        # Find where rule triggers
        mask = df.eval(combined_condition)
        triggered = df[mask]
        
        if len(triggered) == 0:
            return None
        
        # Calculate metrics
        total_triggers = len(triggered)
        
        # For multi-class labels (1=long, -1=short, 0=no move)
        long_success = (triggered[label_col] == 1).sum()
        short_success = (triggered[label_col] == -1).sum()
        no_move = (triggered[label_col] == 0).sum()
        
        # Win rate for the dominant direction
        if long_success >= short_success:
            direction = 'long'
            win_rate = long_success / total_triggers if total_triggers > 0 else 0
            signal_type = 1
        else:
            direction = 'short'
            win_rate = short_success / total_triggers if total_triggers > 0 else 0
            signal_type = -1
        
        # Skip rules with low win rate
        if win_rate < 0.55:  # 55% minimum win rate
            return None
        
        # Average move size when successful
        successful_moves = triggered[triggered[label_col] == signal_type]['move_size'].abs()
        avg_move = successful_moves.mean() if len(successful_moves) > 0 else 0
        
        return {
            'rule': ' AND '.join(conditions),
            'conditions': conditions,
            'direction': direction,
            'signal_type': signal_type,
            'triggers': total_triggers,
            'win_rate': win_rate,
            'long_success': long_success,
            'short_success': short_success,
            'no_move': no_move,
            'avg_move': avg_move,
            'coverage': total_triggers / len(df)
        }
        
    except Exception as e:
        return None

def generate_simple_conditions(df: pd.DataFrame) -> List[str]:
    """Generate simple conditions based on the available features."""
    conditions = []
    
    # Price-based conditions
    for window in [20, 50, 100]:
        if f'zscore_{window}' in df.columns:
            conditions.extend([
                f'zscore_{window} > 2',
                f'zscore_{window} < -2',
                f'zscore_{window} > 1.5',
                f'zscore_{window} < -1.5'
            ])
        
        if f'volume_ratio_{window}' in df.columns:
            conditions.extend([
                f'volume_ratio_{window} > 2',
                f'volume_ratio_{window} > 1.5'
            ])
    
    # RSI conditions
    for period in [14, 28]:
        if f'rsi_{period}' in df.columns:
            conditions.extend([
                f'rsi_{period} < 30',
                f'rsi_{period} > 70',
                f'rsi_{period} < 25',
                f'rsi_{period} > 75'
            ])
    
    # MACD conditions
    if 'macd_diff' in df.columns:
        conditions.extend([
            'macd_diff > 0',
            'macd_diff < 0',
            'macd_cross_up == 1',
            'macd_cross_down == 1'
        ])
    
    # Event-based conditions
    event_cols = [col for col in df.columns if '_recent_' in col and col.endswith('_4')]
    for col in event_cols[:20]:  # Limit to avoid explosion
        conditions.append(f'{col} == 1')
    
    # Microstructure conditions
    if 'spread_proxy' in df.columns:
        conditions.extend([
            'spread_proxy > spread_proxy_ma_20 * 1.5',
            'spread_proxy < spread_proxy_ma_20 * 0.5'
        ])
    
    # Session conditions
    for session in ['asian_session', 'london_session', 'ny_session', 'session_overlap']:
        if session in df.columns:
            conditions.append(f'{session} == 1')
    
    # Multi-timeframe conditions
    for tf in ['1minute', '15minute', '1hour']:
        # Look for specific indicator patterns from higher timeframes
        tf_cols = [col for col in df.columns if f'__{tf}' in col]
        
        for col in tf_cols[:10]:
            if 'bull' in col.lower():
                conditions.append(f'{col} == 1')
            elif 'bear' in col.lower():
                conditions.append(f'{col} == 1')
    
    return list(set(conditions))  # Remove duplicates

def mine_rules(df: pd.DataFrame, cfg: dict) -> Dict:
    """Mine trading rules from the feature matrix."""
    print("Mining trading rules...")
    
    # Generate candidate conditions
    conditions = generate_simple_conditions(df)
    print(f"Generated {len(conditions)} candidate conditions")
    
    # Evaluate single conditions first
    single_rules = []
    for cond in conditions:
        result = evaluate_rule(df, [cond])
        if result and result['triggers'] >= 50:  # Minimum 50 triggers
            single_rules.append(result)
    
    # Sort by win rate * coverage
    single_rules.sort(key=lambda x: x['win_rate'] * x['coverage'], reverse=True)
    
    print(f"Found {len(single_rules)} viable single-condition rules")
    
    # Try combining top conditions
    top_conditions = [rule['conditions'][0] for rule in single_rules[:20]]
    combined_rules = []
    
    # Try 2-condition combinations
    for cond1, cond2 in combinations(top_conditions, 2):
        result = evaluate_rule(df, [cond1, cond2])
        if result and result['triggers'] >= 30:
            combined_rules.append(result)
    
    # Try 3-condition combinations from the best 2-condition rules
    if combined_rules:
        best_2_conds = sorted(combined_rules, key=lambda x: x['win_rate'], reverse=True)[:10]
        
        for rule in best_2_conds:
            base_conds = rule['conditions']
            for extra_cond in top_conditions[:10]:
                if extra_cond not in base_conds:
                    result = evaluate_rule(df, base_conds + [extra_cond])
                    if result and result['triggers'] >= 20 and result['win_rate'] > rule['win_rate']:
                        combined_rules.append(result)
    
    # Combine all rules and sort
    all_rules = single_rules + combined_rules
    all_rules.sort(key=lambda x: x['win_rate'] * np.sqrt(x['triggers']), reverse=True)
    
    # Select top rules
    selected_rules = all_rules[:50]
    
    print(f"\nTop 10 rules:")
    for i, rule in enumerate(selected_rules[:10]):
        print(f"{i+1}. {rule['direction'].upper()} - WR: {rule['win_rate']:.2%}, "
              f"Triggers: {rule['triggers']}, Rule: {rule['rule'][:100]}...")
    
    # Generate signals from rules
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    # Apply rules in order of quality
    for rule in selected_rules:
        mask = df.eval(' & '.join([f"({cond})" for cond in rule['conditions']]))
        signals.loc[mask, 'signal'] = rule['signal_type']
        signals.loc[mask, 'rule_id'] = selected_rules.index(rule)
    
    return {
        'rules': selected_rules,
        'signals': signals,
        'summary': pd.DataFrame(selected_rules)
    }
