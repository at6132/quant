import logging
from datetime import datetime
from typing import Dict, List, Union
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize data processor with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load data processing settings
        self.data_settings = self.config.get('data_processing', {})
        self.trading_settings = self.config.get('trading', {})
        self.timeframes = self.trading_settings.get('timeframes', ['15s', '1m', '5m', '15m', '1h', '4h'])
        
        # Initialize state
        self.price_data = {}
        self.indicators = {}
        self.signals = {}
        self.running = False
        
        logger.info("Data processor initialized")
        
    def start(self):
        """Start the data processor"""
        if self.running:
            logger.warning("Data processor is already running")
            return
            
        try:
            self.running = True
            logger.info("Data processor started")
        except Exception as e:
            logger.error("Error starting data processor: %s", str(e))
            self.running = False
            raise
            
    def stop(self):
        """Stop the data processor"""
        if not self.running:
            logger.warning("Data processor is not running")
            return
            
        try:
            self.running = False
            logger.info("Data processor stopped")
        except Exception as e:
            logger.error("Error stopping data processor: %s", str(e))
            raise
            
    def update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price data for a symbol"""
        try:
            if symbol not in self.price_data:
                self.price_data[symbol] = []
                
            # Create a proper DataFrame with OHLCV data
            new_data = {
                'timestamp': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1.0  # Default volume
            }
            
            # Update existing data if we have it
            if self.price_data[symbol]:
                last_data = self.price_data[symbol][-1]
                if last_data['timestamp'] == timestamp:
                    # Update the last candle
                    last_data['high'] = max(last_data['high'], price)
                    last_data['low'] = min(last_data['low'], price)
                    last_data['close'] = price
                else:
                    # Add new candle
                    self.price_data[symbol].append(new_data)
            else:
                # First data point
                self.price_data[symbol].append(new_data)
                
            # Keep only recent data
            lookback = self.data_settings['lookback_period']
            if len(self.price_data[symbol]) > lookback:
                self.price_data[symbol] = self.price_data[symbol][-lookback:]
                
            # Update indicators and signals
            self._update_indicators(symbol)
            self._generate_signals(symbol)
            
            logger.info("Updated price data for %s: %.2f", symbol, price)
            
        except Exception as e:
            logger.error("Error updating price data: %s", str(e))
            
    def get_price_data(self, symbol: str) -> List[Dict]:
        """Get price data for a symbol"""
        return self.price_data.get(symbol, [])
        
    def get_indicators(self, symbol: str) -> Dict:
        """Get technical indicators for a symbol"""
        return self.indicators.get(symbol, {})
        
    def get_signals(self, symbol: str) -> Dict:
        """Get trading signals for a symbol"""
        return self.signals.get(symbol, {})
        
    def _update_indicators(self, symbol: str):
        """Update technical indicators"""
        try:
            if symbol not in self.price_data:
                return
                
            # Convert to DataFrame with proper index
            prices = pd.DataFrame(self.price_data[symbol])
            prices.set_index('timestamp', inplace=True)
            
            if len(prices) < 2:
                return
                
            # Calculate indicators
            indicators = {}
            
            # Simple Moving Averages
            for period in self.data_settings['indicators']['sma_periods']:
                if len(prices) >= period:
                    indicators[f'sma_{period}'] = prices['close'].rolling(period).mean()
                    
            # RSI
            rsi_period = self.data_settings['indicators']['rsi_period']
            if len(prices) >= rsi_period:
                delta = prices['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs))
                
            # Bollinger Bands
            bb_period = self.data_settings['indicators']['bb_period']
            bb_std = self.data_settings['indicators']['bb_std']
            if len(prices) >= bb_period:
                sma = prices['close'].rolling(bb_period).mean()
                std = prices['close'].rolling(bb_period).std()
                indicators['bb_upper'] = sma + (std * bb_std)
                indicators['bb_middle'] = sma
                indicators['bb_lower'] = sma - (std * bb_std)
                
            # Convert indicators to DataFrame
            indicators_df = pd.DataFrame(indicators, index=prices.index)
            self.indicators[symbol] = indicators_df
            
        except Exception as e:
            logger.error("Error updating indicators: %s", str(e))
            
    def _generate_signals(self, symbol: str):
        """Generate trading signals based on indicators"""
        try:
            if symbol not in self.indicators:
                return
                
            indicators = self.indicators[symbol]
            signals = pd.DataFrame(index=indicators.index)
            
            # Moving Average Crossover
            if 'sma_20' in indicators.columns and 'sma_50' in indicators.columns:
                signals['ma_crossover'] = np.where(indicators['sma_20'] > indicators['sma_50'], 'buy',
                                                 np.where(indicators['sma_20'] < indicators['sma_50'], 'sell', None))
                
            # RSI Signals
            if 'rsi' in indicators.columns:
                signals['rsi'] = np.where(indicators['rsi'] < 30, 'buy',
                                        np.where(indicators['rsi'] > 70, 'sell', None))
                
            # Bollinger Bands Signals
            if all(k in indicators.columns for k in ['bb_upper', 'bb_middle', 'bb_lower']):
                current_price = self.price_data[symbol][-1]['close']
                signals['bb'] = np.where(current_price < indicators['bb_lower'], 'buy',
                                       np.where(current_price > indicators['bb_upper'], 'sell', None))
                
            self.signals[symbol] = signals
            
        except Exception as e:
            logger.error("Error generating signals: %s", str(e))
            
    def calculate_volatility(self, symbol: str) -> float:
        """Calculate price volatility"""
        try:
            if symbol not in self.price_data:
                return 0.0
                
            prices = pd.DataFrame(self.price_data[symbol])
            if len(prices) < 2:
                return 0.0
                
            returns = prices['close'].pct_change().dropna()
            return returns.std()
            
        except Exception as e:
            logger.error("Error calculating volatility: %s", str(e))
            return 0.0 