import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config['labeling']['threshold']
        self.lookforward = config['labeling']['lookforward']
        self.min_move_size = config['labeling']['min_move_size']
        self.max_holding_time = config['labeling']['max_holding_time']
        self.min_holding_time = config['labeling']['min_holding_time']
        self.stop_loss = config['labeling']['stop_loss']
        self.take_profit = config['labeling']['take_profit']

    async def initialize(self):
        """Initialize the label generator (no-op for compatibility)"""
        pass

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading labels based on future returns with improved signal generation
        """
        logger.info("Generating trading labels...")
        
        # Calculate future returns
        future_returns = df['close'].shift(-self.lookforward) / df['close'] - 1
        
        # Generate binary labels with improved thresholds
        df['label'] = np.where(future_returns > self.threshold, 1,
                             np.where(future_returns < -self.threshold, -1, 0))
        
        # Drop rows with NaN values
        df = df.dropna()
        
        logger.info(f"Generated labels. Distribution: {df['label'].value_counts().to_dict()}")
        return df

    def generate_action_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate simplified action-based labels (buy, sell, hold)
        with improved signal generation logic
        """
        logger.info("Generating action-based labels...")
        
        # Calculate future returns and volatility
        future_returns = df['close'].shift(-self.lookforward) / df['close'] - 1
        volatility = df['close'].rolling(window=20).std() / df['close']
        
        # Initialize action label column
        df['action_label'] = 0  # hold by default
        
        # Generate signals based on multiple conditions
        for i in range(len(df) - self.lookforward):
            if pd.isna(future_returns.iloc[i]) or pd.isna(volatility.iloc[i]):
                continue
                
            # Calculate dynamic thresholds based on volatility
            vol_factor = min(2.0, max(0.5, volatility.iloc[i] * 100))
            dynamic_threshold = self.threshold * vol_factor
            
            # Generate signals
            if future_returns.iloc[i] > dynamic_threshold:
                # Strong buy signal
                df.iloc[i, df.columns.get_loc('action_label')] = 1
            elif future_returns.iloc[i] < -dynamic_threshold:
                # Strong sell signal
                df.iloc[i, df.columns.get_loc('action_label')] = -1
                
        # Apply position management rules
        df = self._apply_position_rules(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        logger.info(f"Generated action labels. Distribution: {df['action_label'].value_counts().to_dict()}")
        return df

    def _apply_position_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position management rules to the signals
        """
        position = 0
        entry_price = 0
        entry_time = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = i
            
            # Check for stop loss or take profit if in position
            if position != 0:
                holding_time = current_time - entry_time
                price_change = (current_price - entry_price) / entry_price
                
                # Check stop loss
                if (position == 1 and price_change < -self.stop_loss) or \
                   (position == -1 and price_change > self.stop_loss):
                    df.iloc[i, df.columns.get_loc('action_label')] = 0
                    position = 0
                    continue
                
                # Check take profit
                if (position == 1 and price_change > self.take_profit) or \
                   (position == -1 and price_change < -self.take_profit):
                    df.iloc[i, df.columns.get_loc('action_label')] = 0
                    position = 0
                    continue
                
                # Check max holding time
                if holding_time >= self.max_holding_time:
                    df.iloc[i, df.columns.get_loc('action_label')] = 0
                    position = 0
                    continue
            
            # Check for new entry signals
            if position == 0 and df['action_label'].iloc[i] != 0:
                position = df['action_label'].iloc[i]
                entry_price = current_price
                entry_time = current_time
            elif position != 0 and df['action_label'].iloc[i] == -position:
                # Close position if opposite signal
                df.iloc[i, df.columns.get_loc('action_label')] = 0
                position = 0
        
        return df

    def get_signal_metadata(self, df: pd.DataFrame) -> Dict:
        """
        Generate metadata about the signals for analysis
        """
        signals = df[df['action_label'] != 0]
        total_signals = len(signals)
        
        if total_signals == 0:
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "avg_holding_time": 0,
                "win_rate": 0
            }
        
        buy_signals = len(signals[signals['action_label'] == 1])
        sell_signals = len(signals[signals['action_label'] == -1])
        
        # Calculate average holding time
        holding_times = []
        current_position = 0
        entry_time = 0
        
        for i in range(len(df)):
            if df['action_label'].iloc[i] != 0:
                if current_position == 0:
                    current_position = df['action_label'].iloc[i]
                    entry_time = i
                elif df['action_label'].iloc[i] == -current_position:
                    holding_times.append(i - entry_time)
                    current_position = 0
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        return {
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "avg_holding_time": avg_holding_time,
            "win_rate": buy_signals / total_signals if total_signals > 0 else 0
        }

    def generate_multi_class_labels(self, df):
        """
        Generate multi-class labels based on return thresholds
        """
        logger.info("Generating multi-class labels...")
        
        # Calculate future returns
        future_returns = df['close'].shift(-self.lookforward) / df['close'] - 1
        
        # Define thresholds for multi-class labels
        thresholds = {
            'strong_buy': self.threshold,
            'buy': self.threshold/2,
            'neutral': 0,
            'sell': -self.threshold/2,
            'strong_sell': -self.threshold
        }
        
        # Generate multi-class labels
        conditions = [
            (future_returns > thresholds['strong_buy']),
            (future_returns > thresholds['buy']),
            (future_returns > thresholds['neutral']),
            (future_returns > thresholds['sell']),
            (future_returns > thresholds['strong_sell'])
        ]
        choices = [2, 1, 0, -1, -2]
        
        df['multi_class_label'] = np.select(conditions, choices, default=-2)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Ensure no duplicate column names
        df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col 
                     for i, col in enumerate(df.columns)]
        
        logger.info(f"Generated multi-class labels. Distribution: {df['multi_class_label'].value_counts().to_dict()}")
        return df 

    def generate_intuition_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-dimensional intuition labels for advanced trading decisions.
        Creates 10 different labels that match our intuition model outputs.
        """
        logger.info("Generating intuition-based multi-dimensional labels...")
        
        # Calculate key metrics
        future_returns = df['close'].shift(-self.lookforward) / df['close'] - 1
        volatility = df['close'].rolling(window=20).std() / df['close']
        momentum = df['close'].pct_change(10)
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        
        # Initialize all label columns
        labels = {}
        
        # 1. Entry probability (0-1): How likely is a profitable trade?
        labels['entry_probability'] = np.where(
            np.abs(future_returns) > self.threshold,
            np.clip(np.abs(future_returns) / (self.threshold * 2), 0, 1),
            0.0
        )
        
        # 2. Entry direction (-1, 0, 1): Long, short, or no trade
        labels['entry_direction'] = np.where(
            future_returns > self.threshold, 1.0,
            np.where(future_returns < -self.threshold, -1.0, 0.0)
        )
        
        # 3. Entry confidence (0-1): Confidence in the signal
        labels['entry_confidence'] = np.clip(
            np.abs(future_returns) / (self.threshold * 1.5) * 
            (1 + volume_ratio) / 2, 0, 1
        )
        
        # 4. Position size multiplier (0-1): Risk sizing based on confidence
        labels['position_size_multiplier'] = np.clip(
            labels['entry_confidence'] * (1 - volatility), 0, 1
        )
        
        # 5. Take profit distance (0-1): Distance to take profit
        labels['tp_distance'] = np.clip(
            np.abs(future_returns) * 0.8, 0, 1
        )
        
        # 6. Stop loss distance (0-1): Distance to stop loss
        labels['sl_distance'] = np.clip(
            volatility * 2, 0, 1
        )
        
        # 7. Trail aggressiveness (0-1): How aggressively to trail
        labels['trail_aggressiveness'] = np.clip(
            momentum * 10, 0, 1
        )
        
        # 8. Hold duration (0-1): How long to hold (normalized)
        labels['hold_duration'] = np.clip(
            np.abs(future_returns) / (self.threshold * 3), 0, 1
        )
        
        # 9. Exit probability (0-1): Probability of exit
        labels['exit_probability'] = np.where(
            np.abs(future_returns) < self.threshold * 0.5,
            0.8,  # High exit probability if small move expected
            0.2   # Low exit probability if big move expected
        )
        
        # 10. Exit urgency (0-1): How urgently to exit
        labels['exit_urgency'] = np.clip(
            volatility * 3, 0, 1
        )
        
        # Add all labels to DataFrame
        for label_name, label_values in labels.items():
            df[f'intuition_{label_name}'] = label_values
        
        # Create combined label for backward compatibility
        df['intuition_label'] = labels['entry_direction']
        
        # Drop rows with NaN values
        df = df.dropna()
        
        logger.info(f"Generated intuition labels with {len(df)} samples")
        logger.info(f"Entry direction distribution: {df['intuition_entry_direction'].value_counts().to_dict()}")
        
        return df 