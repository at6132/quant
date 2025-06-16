"""
Sessions Indicator - Simplified Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime


def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified sessions indicator implementation.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with session-related columns
    """
    try:
        if df.empty:
            result = pd.DataFrame(index=df.index)
            result['session_id'] = 0
            result['session_name'] = 'none'
            result['in_session'] = 0
            result['new_session'] = 0
            result['session_open'] = 0
            result['minutes_into'] = 0
            return result
        
        result = pd.DataFrame(index=df.index)
        
        # Simple session detection based on hour
        if hasattr(df.index, 'hour'):
            hours = df.index.hour
        else:
            # If no datetime index, create dummy values
            hours = np.zeros(len(df))
        
        # Define simple sessions
        # London: 8-16, New York: 13-21, Asian: 21-5
        session_id = np.where((hours >= 8) & (hours < 16), 1,  # London
                             np.where((hours >= 13) & (hours < 21), 2,  # New York
                                     np.where((hours >= 21) | (hours < 5), 3, 0)))  # Asian
        
        session_names = ['none', 'london', 'newyork', 'asian']
        
        result['session_id'] = session_id
        result['session_name'] = [session_names[sid] for sid in session_id]
        result['in_session'] = (session_id > 0).astype(int)
        
        # Detect new sessions (simplified)
        result['new_session'] = (result['session_id'].diff() != 0).astype(int)
        result.loc[result.index[0], 'new_session'] = 1  # First bar is always new session
        
        result['session_open'] = result['new_session']
        result['minutes_into'] = 0  # Placeholder
        
        return result
        
    except Exception as e:
        print(f"Error in build_session_table: {e}")
        result = pd.DataFrame(index=df.index)
        result['session_id'] = 0
        result['session_name'] = 'none'
        result['in_session'] = 0
        result['new_session'] = 0
        result['session_open'] = 0
        result['minutes_into'] = 0
        return result