import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple, Optional
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model inference system."""
        self.cfg = self._load_config(config_path)
        self.model = None
        self.training = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """Load the trained TFT model and its configuration."""
        logger.info("Loading model...")
        
        # Load model metadata
        metadata_path = Path("artefacts/model_metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError("Model metadata not found. Please train the model first.")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model
        model_path = Path("artefacts/tft_model.ckpt")
        if not model_path.exists():
            raise FileNotFoundError("Model file not found. Please train the model first.")
        
        # Create dummy dataset for model loading
        dummy_data = pd.DataFrame({
            'time_idx': np.arange(1000),
            'group_id': 0,
            'action_label': np.zeros(1000)
        })
        
        # Add dummy features
        for i in range(10):
            dummy_data[f'feature_{i}'] = np.random.randn(1000)
        
        self.training = TimeSeriesDataSet(
            dummy_data,
            time_idx="time_idx",
            target="action_label",
            group_ids=["group_id"],
            min_encoder_length=self.cfg['models']['tft']['encoder_length'],
            max_encoder_length=self.cfg['models']['tft']['encoder_length'],
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[f'feature_{i}' for i in range(10)],
            target_normalizer=GroupNormalizer(),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Load model
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            model_path,
            map_location=self.device
        )
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model inference."""
        # Ensure required columns exist
        required_cols = ['time_idx', 'group_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in input data")
        
        # Add time index if not present
        if 'time_idx' not in df.columns:
            df['time_idx'] = np.arange(len(df))
        
        # Add group ID if not present
        if 'group_id' not in df.columns:
            df['group_id'] = 0
        
        return df
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Generate predictions for the input data.
        Returns:
            Tuple[np.ndarray, float]: (predictions, confidence)
        """
        logger.info("Generating predictions...")
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Create dataset
        dataset = TimeSeriesDataSet.from_dataset(
            self.training,
            df,
            predict=True
        )
        
        # Create dataloader
        dataloader = dataset.to_dataloader(
            train=False,
            batch_size=self.cfg['models']['tft']['batch_size'],
            num_workers=0
        )
        
        # Generate predictions
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = x.to(self.device)
                
                # Get model predictions
                output = self.model(x)
                probs = torch.softmax(output, dim=1)
                
                # Get predicted class and confidence
                pred_class = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                
                predictions.extend(pred_class.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Calculate average confidence
        avg_confidence = float(np.mean(confidences))
        
        logger.info(f"Generated predictions with average confidence: {avg_confidence:.4f}")
        return predictions, avg_confidence
    
    def get_trading_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal from model predictions.
        Returns:
            Dict: Trading signal with action and metadata
        """
        predictions, confidence = self.predict(df)
        
        # Get the latest prediction
        latest_pred = predictions[-1]
        latest_confidence = confidence
        
        # Map prediction to trading action
        action_map = {
            0: "HOLD",
            1: "BUY",
            -1: "SELL"
        }
        
        # Generate signal
        signal = {
            "timestamp": datetime.now().isoformat(),
            "action": action_map[latest_pred],
            "confidence": latest_confidence,
            "metadata": {
                "model_version": self.metadata["version"],
                "prediction_threshold": self.cfg['labeling']['threshold'],
                "min_confidence": self.cfg['trading']['paper_trading']['min_confidence']
            }
        }
        
        # Add position management rules
        signal["position_management"] = {
            "stop_loss": self.cfg['labeling']['stop_loss'],
            "take_profit": self.cfg['labeling']['take_profit'],
            "max_holding_time": self.cfg['labeling']['max_holding_time']
        }
        
        logger.info(f"Generated trading signal: {signal['action']} with confidence {signal['confidence']:.4f}")
        return signal
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate if the trading signal meets the criteria for execution.
        """
        # Check confidence threshold
        min_confidence = self.cfg['trading']['paper_trading']['min_confidence']
        if signal['confidence'] < min_confidence:
            logger.info(f"Signal rejected: confidence {signal['confidence']:.4f} below threshold {min_confidence}")
            return False
        
        # Check if action is valid
        valid_actions = ["BUY", "SELL", "HOLD"]
        if signal['action'] not in valid_actions:
            logger.warning(f"Invalid action in signal: {signal['action']}")
            return False
        
        return True

def main():
    """Test the model inference system."""
    # Initialize inference system
    inference = ModelInference()
    
    # Load test data
    df = pd.read_parquet("processed_data/features.parquet")
    
    # Generate trading signal
    signal = inference.get_trading_signal(df)
    
    # Validate signal
    if inference.validate_signal(signal):
        logger.info("Signal validated successfully")
    else:
        logger.info("Signal validation failed")

if __name__ == "__main__":
    main() 