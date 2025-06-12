import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelGenerator:
    def __init__(self, config):
        self.config = config
        self.threshold = config['labeling']['threshold']
        self.lookforward = config['labeling']['lookforward']

    def generate_labels(self, df):
        """
        Generate trading labels based on future returns
        """
        logger.info("Generating trading labels...")
        
        # Calculate future returns
        future_returns = df['close'].shift(-self.lookforward) / df['close'] - 1
        
        # Generate binary labels
        df['label'] = np.where(future_returns > self.threshold, 1,
                             np.where(future_returns < -self.threshold, -1, 0))
        
        # Drop rows with NaN values (last few rows due to lookforward)
        df = df.dropna()
        
        # Ensure no duplicate column names
        df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col 
                     for i, col in enumerate(df.columns)]
        
        logger.info(f"Generated labels. Distribution: {df['label'].value_counts().to_dict()}")
        return df

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

    def generate_action_labels(self, df):
        """
        Generate action-based multi-class labels for open/close/add long/short and hold.
        0: hold, 1: open long, 2: open short, 3: close long, 4: close short, 5: add to long, 6: add to short
        """
        logger.info("Generating action-based multi-class labels...")
        future_returns = df['close'].shift(-self.lookforward) / df['close'] - 1
        # Example logic (customize as needed):
        df['action_label'] = 0  # hold by default
        # Open long
        df.loc[future_returns > self.threshold, 'action_label'] = 1
        # Open short
        df.loc[future_returns < -self.threshold, 'action_label'] = 2
        # Add to long (example: very strong move)
        df.loc[future_returns > 2 * self.threshold, 'action_label'] = 5
        # Add to short (example: very strong move)
        df.loc[future_returns < -2 * self.threshold, 'action_label'] = 6
        # Close long (example: after a run up, price falls back)
        df.loc[(future_returns < 0) & (df['action_label'] == 1), 'action_label'] = 3
        # Close short (example: after a run down, price rises back)
        df.loc[(future_returns > 0) & (df['action_label'] == 2), 'action_label'] = 4
        df = df.dropna()
        df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col 
                     for i, col in enumerate(df.columns)]
        logger.info(f"Generated action labels. Distribution: {df['action_label'].value_counts().to_dict()}")
        return df 