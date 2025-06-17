import logging
from datetime import datetime
from typing import Dict, List, Union, Optional
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
import websocket
import json
import threading
import time
import os

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
        self.features = {}
        self.predictions = {}
        self.running = False
        
        # Load LGBM model
        try:
            model_path = Path("../Algorithm1/artefacts/lgbm_model.pkl")
            if os.path.exists(model_path):
                try:
                    # Try loading as joblib first
                    self.model = joblib.load(model_path)
                    logger.info("Loaded model with joblib successfully")
                except:
                    # If that fails, try loading as LightGBM Booster
                    try:
                        self.model = lgb.Booster(model_file=str(model_path))
                        logger.info("Loaded LGBM Booster model successfully")
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}")
                        self.model = None
                
                # Print model feature names for debugging
                # if self.model is not None:
                #     if hasattr(self.model, 'feature_name'):
                #         feature_names = self.model.feature_name()
                #         logger.info(f"Model expects {len(feature_names)} features: {feature_names}")
                #     elif hasattr(self.model, 'feature_names_'):
                #         feature_names = self.model.feature_names_
                #         logger.info(f"Model expects {len(feature_names)} features: {feature_names}")
                #     else:
                #         logger.info("Model does not have feature_name or feature_names_ attribute")
            else:
                logger.warning("Model file not found: %s", model_path)
                self.model = None
        except Exception as e:
            logger.error("Error loading LGBM model: %s", str(e))
            self.model = None
            
        # Initialize Kraken WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.symbol = 'BTC/USD'  # Only trade BTC
            
        logger.info("Data processor initialized")
        
    def start(self):
        """Start the data processor"""
        try:
            # Initialize WebSocket connection
            self.ws = websocket.WebSocketApp(
                "wss://ws.kraken.com",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info("Connected to Kraken WebSocket")
            logger.info("Data processor started")
            
        except Exception as e:
            logger.error("Error starting data processor: %s", str(e))
            
    def stop(self):
        """Stop the data processor"""
        if not self.running:
            logger.warning("Data processor is not running")
            return
            
        try:
            self.running = False
            
            # Disconnect WebSocket
            if self.ws:
                self.ws.close()
                
            logger.info("Data processor stopped")
        except Exception as e:
            logger.error("Error stopping data processor: %s", str(e))
            raise
            
    def _on_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                # Handle ticker data
                if len(data) > 1 and isinstance(data[1], dict):
                    ticker = data[1]
                    if 'c' in ticker:  # Current price
                        price = float(ticker['c'][0])  # First element is the current price
                        timestamp = datetime.now()
                        
                        # Update price data
                        self.update_price_data(
                            self.symbol,
                            price,
                            timestamp,
                            {
                                'open': float(ticker['o'][0]),
                                'high': float(ticker['h'][0]),
                                'low': float(ticker['l'][0]),
                                'close': price,
                                'volume': float(ticker['v'][0])
                            }
                        )
                
                # Handle trade data
                elif len(data) > 1 and isinstance(data[1], list):
                    trades = data[1]
                    for trade in trades:
                        if len(trade) >= 3:  # Ensure we have at least price, volume, and time
                            price = float(trade[0])
                            volume = float(trade[1])
                            # Convert timestamp to float if it's a string
                            timestamp = float(trade[2]) if isinstance(trade[2], str) else trade[2]
                            timestamp = datetime.fromtimestamp(timestamp)
                            
                            # Update price data
                            self.update_price_data(
                                self.symbol,
                                price,
                                timestamp,
                                {
                                    'open': price,
                                    'high': price,
                                    'low': price,
                                    'close': price,
                                    'volume': volume
                                }
                            )
            elif isinstance(data, dict):
                # Handle system messages
                if data.get('event') == 'systemStatus':
                    logger.info("System status: %s", data.get('status'))
                elif data.get('event') == 'subscriptionStatus':
                    status = data.get('status')
                    if status == 'error':
                        logger.error("Subscription error: %s", data.get('errorMessage'))
                elif data.get('event') == 'error':
                    logger.error("WebSocket error: %s", data.get('errorMessage'))
                    
        except Exception as e:
            logger.error("Error handling WebSocket message: %s", str(e))
            
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error("WebSocket error: %s", str(error))
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.info("WebSocket connection closed")
        
    def _on_open(self, ws):
        """Handle WebSocket open"""
        try:
            # Subscribe to BTC/USD trades
            subscribe_msg = {
                "event": "subscribe",
                "pair": ["XBT/USD"],
                "subscription": {
                    "name": "trade"
                }
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to BTC/USD trades")
            
            # Also subscribe to ticker for continuous price updates
            ticker_msg = {
                "event": "subscribe",
                "pair": ["XBT/USD"],
                "subscription": {
                    "name": "ticker"
                }
            }
            ws.send(json.dumps(ticker_msg))
            logger.info("Subscribed to BTC/USD ticker")
            
        except Exception as e:
            logger.error("Error subscribing to trades: %s", str(e))
            
    def update_price_data(self, symbol: str, price: float, timestamp: datetime, ohlcv: Optional[Dict] = None):
        """Update price data for a symbol"""
        try:
            if symbol not in self.price_data:
                self.price_data[symbol] = []
                
            # Create a proper DataFrame with OHLCV data
            if ohlcv:
                new_data = {
                    'timestamp': timestamp,
                    'open': float(ohlcv['open']),
                    'high': float(ohlcv['high']),
                    'low': float(ohlcv['low']),
                    'close': float(ohlcv['close']),
                    'volume': float(ohlcv['volume'])
                }
            else:
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
                    last_data['high'] = max(last_data['high'], new_data['high'])
                    last_data['low'] = min(last_data['low'], new_data['low'])
                    last_data['close'] = new_data['close']
                    last_data['volume'] += new_data['volume']
                else:
                    # Add new candle
                    self.price_data[symbol].append(new_data)
            else:
                # First data point
                self.price_data[symbol].append(new_data)
                
            # Keep only recent data
            lookback = self.data_settings.get('lookback_period', 1000)
            if len(self.price_data[symbol]) > lookback:
                self.price_data[symbol] = self.price_data[symbol][-lookback:]
                
            # Update indicators, features, and signals
            self._update_indicators(symbol)
            self._generate_features(symbol)
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
        
    def get_features(self, symbol: str) -> pd.DataFrame:
        """Get feature set for a symbol"""
        return self.features.get(symbol, pd.DataFrame())
        
    def get_predictions(self, symbol: str) -> Dict:
        """Get model predictions for a symbol"""
        return self.predictions.get(symbol, {})
        
    def get_signals(self, symbol: str) -> Dict:
        """Get trading signals for a symbol"""
        return self.signals.get(symbol, {})
        
    def _update_indicators(self, symbol: str):
        """Update technical indicators"""
        try:
            if symbol not in self.price_data:
                return
                
            # Convert price data to DataFrame
            df = pd.DataFrame(self.price_data[symbol])
            df.set_index('timestamp', inplace=True)
            
            # Generate comprehensive features instead of relying on individual indicator modules
            comprehensive_features = self._generate_comprehensive_features(df)
            
            # Import and calculate specific indicators for additional features
            from Core.indicators import (
                bb_ob_engine,
                breaker_signals,
                ict_sm_trades,
                IT_Foundation,
                liquidity_swings,
                pvsra_vs,
                sessions,
                smc_core,
                tr_reality_core
            )
            
            # Calculate specific indicators and merge them with comprehensive features
            indicator_dfs = [comprehensive_features]
            
            # Add specific indicator features
            bb_df = bb_ob_engine.process_candles(df)
            if isinstance(bb_df, pd.DataFrame):
                indicator_dfs.append(bb_df)
            
            breaker_df = breaker_signals.process_candles(df)
            if isinstance(breaker_df, pd.DataFrame):
                indicator_dfs.append(breaker_df)
            
            ict_df = ict_sm_trades.process_candles(df)
            if isinstance(ict_df, pd.DataFrame):
                indicator_dfs.append(ict_df)
            
            it_df = IT_Foundation.process_candles(df)
            if isinstance(it_df, pd.DataFrame):
                indicator_dfs.append(it_df)
            
            liquidity_df = liquidity_swings.process_candles(df)
            if isinstance(liquidity_df, pd.DataFrame):
                indicator_dfs.append(liquidity_df)
            
            pvsra_df = pvsra_vs.process_candles(df)
            if isinstance(pvsra_df, pd.DataFrame):
                indicator_dfs.append(pvsra_df)
            
            session_df = sessions.process_candles(df)
            if isinstance(session_df, pd.DataFrame):
                indicator_dfs.append(session_df)
            
            smc_df = smc_core.process_candles(df)
            if isinstance(smc_df, pd.DataFrame):
                indicator_dfs.append(smc_df)
            
            tr_df = tr_reality_core.process_candles(df)
            if isinstance(tr_df, pd.DataFrame):
                indicator_dfs.append(tr_df)
            
            # Concatenate all indicator DataFrames
            if indicator_dfs:
                indicators_df = pd.concat(indicator_dfs, axis=1)
                # Remove duplicate columns (keep the first occurrence)
                indicators_df = indicators_df.loc[:, ~indicators_df.columns.duplicated()]
                
                # Debug: Log what features we have
                # logger.info(f"Generated {len(indicators_df.columns)} total features")
                # logger.info(f"Feature columns: {list(indicators_df.columns)}")
                
                self.indicators[symbol] = indicators_df
            else:
                # Create empty DataFrame with correct index
                self.indicators[symbol] = pd.DataFrame(index=df.index)
            
        except Exception as e:
            logger.error("Error updating indicators: %s", str(e))
            
    def _generate_features(self, symbol: str):
        """Generate feature set for model input"""
        try:
            if symbol not in self.indicators:
                return
                
            # Get latest indicators (which now include comprehensive features)
            indicators = self.indicators[symbol]
            
            # Get required features
            required_features = self._get_required_features()
            
            # Check if we have all required features
            missing_features = set(required_features) - set(indicators.columns)
            
            if missing_features:
                logger.warning("Missing features: %s", missing_features)
                # Add missing features with default values
                missing_features_dict = {feature: 0.0 for feature in missing_features}
                indicators = pd.concat([indicators, pd.DataFrame(missing_features_dict, index=indicators.index)], axis=1)
                self.indicators[symbol] = indicators
                    
            # Select only required features in correct order
            features = indicators[required_features]
            
            # Store features
            self.features[symbol] = features
            
            # Generate predictions if model is loaded
            if self.model is not None and len(features) > 0:
                latest_features = features.iloc[-1:].values
                
                # Debug: Log feature statistics
                non_zero_features = np.count_nonzero(latest_features[0])
                total_features = len(latest_features[0])
                logger.info(f"Feature stats: {non_zero_features}/{total_features} non-zero features")
                
                # Debug: Log some sample feature values
                sample_features = latest_features[0][:10]  # First 10 features
                logger.info(f"Sample features (first 10): {sample_features}")
                
                # Use predict_proba if available, else use predict
                if hasattr(self.model, 'predict_proba'):
                    prediction = self.model.predict_proba(latest_features)[0]
                    probability = float(prediction[1])
                    logger.info(f"Raw predict_proba output: {prediction}, probability: {probability}")
                else:
                    prediction = self.model.predict(latest_features)[0]
                    probability = float(prediction)  # For Booster, this is the probability of class 1
                    logger.info(f"Raw predict output: {prediction}, probability: {probability}")
                    
                self.predictions[symbol] = {
                    'probability': probability,
                    'prediction': int(probability > 0.5),
                    'timestamp': features.index[-1]
                }
                
        except Exception as e:
            logger.error("Error generating features: %s", str(e))
            
    def _generate_signals(self, symbol: str):
        """Generate trading signals based on model predictions"""
        try:
            if symbol not in self.predictions:
                return
                
            prediction = self.predictions[symbol]
            signals = pd.DataFrame(index=[prediction['timestamp']])
            
            # Generate signals based on model probability
            signals['signal'] = np.where(
                prediction['probability'] > 0.7, 'buy',
                np.where(prediction['probability'] < 0.3, 'sell', 'hold')
            )
            
            # Add probability and prediction
            signals['probability'] = prediction['probability']
            signals['prediction'] = prediction['prediction']
            
            self.signals[symbol] = signals
            
        except Exception as e:
            logger.error("Error generating signals: %s", str(e))
            
    def _get_required_features(self) -> List[str]:
        """Get list of required features for the model"""
        # If we have a model with feature names, use those exactly
        if self.model is not None:
            if hasattr(self.model, 'feature_names_'):
                return list(self.model.feature_names_)
            elif hasattr(self.model, 'feature_name'):
                return self.model.feature_name()
        
        # Fallback to hardcoded list if no model
        return [
            'open', 'high', 'low', 'close', 'volume', 'pvsra_open', 'pvsra_high', 'pvsra_low', 'pvsra_close', 'pvsra_volume', 'pvsra_vec_color', 'pvsra_gr_pattern', 'sessions_close', 'sessions_session_id', 'sessions_in_session', 'sessions_new_session', 'sessions_session_open', 'sessions_minutes_into', 'ict_close', 'ict_atr', 'ict_lg_pivot_hi', 'ict_lg_pivot_lo', 'ict_lg_daily_hi', 'ict_lg_daily_lo', 'ict_lg_weekly_hi', 'ict_lg_weekly_lo', 'ict_mss_up', 'ict_mss_dn', 'ict_fvg_up_high', 'ict_fvg_up_low', 'ict_fvg_dn_high', 'ict_fvg_dn_low', 'ict_fvg_up_mid', 'ict_fvg_dn_mid', 'breaker_bbplus', 'breaker_signup', 'breaker_cnclup', 'breaker_ll1break', 'breaker_ll2break', 'breaker_sw1breakup', 'breaker_sw2breakup', 'breaker_tpup1', 'breaker_tpup2', 'breaker_tpup3', 'breaker_bb_endbl', 'breaker_bb_min', 'breaker_signdn', 'breaker_cncldn', 'breaker_hh1break', 'breaker_hh2break', 'breaker_sw1breakdn', 'breaker_sw2breakdn', 'breaker_tpdn1', 'breaker_tpdn2', 'breaker_tpdn3', 'breaker_bb_endbr', 'liq_ph_level', 'liq_ph_count', 'liq_ph_volume', 'liq_ph_crossed', 'liq_pl_level', 'liq_pl_count', 'liq_pl_volume', 'liq_pl_crossed', 'smc_t', 'smc_open', 'smc_high', 'smc_low', 'smc_close', 'smc_trend_internal', 'smc_trend_swing', 'tr_ema_5', 'tr_ema_13', 'tr_ema_50', 'tr_ema_200', 'tr_ema_800', 'tr_yesterday_h', 'tr_yesterday_l', 'tr_lastweek_h', 'tr_lastweek_l', 'tr_vec_color', 'tr_vcz_top', 'tr_vcz_bot', 'bb_ts', 'bb_open', 'bb_high', 'bb_low', 'bb_close', 'open_roll_mean_10', 'open_roll_std_10', 'open_roll_min_10', 'open_roll_max_10', 'open_roll_range_10', 'open_momentum_10', 'open_volatility_10', 'high_roll_mean_10', 'high_roll_std_10', 'high_roll_min_10', 'high_roll_max_10', 'high_roll_range_10', 'high_momentum_10', 'high_volatility_10', 'low_roll_mean_10', 'low_roll_std_10', 'low_roll_min_10', 'low_roll_max_10', 'low_roll_range_10', 'low_momentum_10', 'low_volatility_10', 'close_roll_mean_10', 'close_roll_std_10', 'close_roll_min_10', 'close_roll_max_10', 'close_roll_range_10', 'close_momentum_10', 'close_volatility_10', 'volume_roll_mean_10', 'volume_roll_std_10', 'volume_roll_min_10', 'volume_roll_max_10', 'volume_roll_range_10', 'volume_momentum_10', 'volume_volatility_10', 'open_roll_mean_20', 'open_roll_std_20', 'open_roll_min_20', 'open_roll_max_20', 'open_roll_range_20', 'open_momentum_20', 'open_volatility_20', 'high_roll_mean_20', 'high_roll_std_20', 'high_roll_min_20', 'high_roll_max_20', 'high_roll_range_20', 'high_momentum_20', 'high_volatility_20', 'low_roll_mean_20', 'low_roll_std_20', 'low_roll_min_20', 'low_roll_max_20', 'low_roll_range_20', 'low_momentum_20', 'low_volatility_20', 'close_roll_mean_20', 'close_roll_std_20', 'close_roll_min_20', 'close_roll_max_20', 'close_roll_range_20', 'close_momentum_20', 'close_volatility_20', 'volume_roll_mean_20', 'volume_roll_std_20', 'volume_roll_min_20', 'volume_roll_max_20', 'volume_roll_range_20', 'volume_momentum_20', 'volume_volatility_20', 'open_roll_mean_50', 'open_roll_std_50', 'open_roll_min_50', 'open_roll_max_50', 'open_roll_range_50', 'open_momentum_50', 'open_volatility_50', 'high_roll_mean_50', 'high_roll_std_50', 'high_roll_min_50', 'high_roll_max_50', 'high_roll_range_50', 'high_momentum_50', 'high_volatility_50', 'low_roll_mean_50', 'low_roll_std_50', 'low_roll_min_50', 'low_roll_max_50', 'low_roll_range_50', 'low_momentum_50', 'low_volatility_50', 'close_roll_mean_50', 'close_roll_std_50', 'close_roll_min_50', 'close_roll_max_50', 'close_roll_range_50', 'close_momentum_50', 'close_volatility_50', 'volume_roll_mean_50', 'volume_roll_std_50', 'volume_roll_min_50', 'volume_roll_max_50', 'volume_roll_range_50', 'volume_momentum_50', 'volume_volatility_50', 'open_roll_mean_100', 'open_roll_std_100', 'open_roll_min_100', 'open_roll_max_100', 'open_roll_range_100', 'open_momentum_100', 'open_volatility_100', 'high_roll_mean_100', 'high_roll_std_100', 'high_roll_min_100', 'high_roll_max_100', 'high_roll_range_100', 'high_momentum_100', 'high_volatility_100', 'low_roll_mean_100', 'low_roll_std_100', 'low_roll_min_100', 'low_roll_max_100', 'low_roll_range_100', 'low_momentum_100', 'low_volatility_100', 'close_roll_mean_100', 'close_roll_std_100', 'close_roll_min_100', 'close_roll_max_100', 'close_roll_range_100', 'close_momentum_100', 'close_volatility_100', 'volume_roll_mean_100', 'volume_roll_std_100', 'volume_roll_min_100', 'volume_roll_max_100', 'volume_roll_range_100', 'volume_momentum_100', 'volume_volatility_100', 'open_roll_mean_200', 'open_roll_std_200', 'open_roll_min_200', 'open_roll_max_200', 'open_roll_range_200', 'open_momentum_200', 'open_volatility_200', 'high_roll_mean_200', 'high_roll_std_200', 'high_roll_min_200', 'high_roll_max_200', 'high_roll_range_200', 'high_momentum_200', 'high_volatility_200', 'low_roll_mean_200', 'low_roll_std_200', 'low_roll_min_200', 'low_roll_max_200', 'low_roll_range_200', 'low_momentum_200', 'low_volatility_200', 'close_roll_mean_200', 'close_roll_std_200', 'close_roll_min_200', 'close_roll_max_200', 'close_roll_range_200', 'close_momentum_200', 'close_volatility_200', 'volume_roll_mean_200', 'volume_roll_std_200', 'volume_roll_min_200', 'volume_roll_max_200', 'volume_roll_range_200', 'volume_momentum_200', 'volume_volatility_200', 'open_roll_mean_500', 'open_roll_std_500', 'open_roll_min_500', 'open_roll_max_500', 'open_roll_range_500', 'open_momentum_500', 'open_volatility_500', 'high_roll_mean_500', 'high_roll_std_500', 'high_roll_min_500', 'high_roll_max_500', 'high_roll_range_500', 'high_momentum_500', 'high_volatility_500', 'low_roll_mean_500', 'low_roll_std_500', 'low_roll_min_500', 'low_roll_max_500', 'low_roll_range_500', 'low_momentum_500', 'low_volatility_500', 'close_roll_mean_500', 'close_roll_std_500', 'close_roll_min_500', 'close_roll_max_500', 'close_roll_range_500', 'close_momentum_500', 'close_volatility_500', 'volume_roll_mean_500', 'volume_roll_std_500', 'volume_roll_min_500', 'volume_roll_max_500', 'volume_roll_range_500', 'volume_momentum_500', 'volume_volatility_500', 'open_roll_mean_1000', 'open_roll_std_1000', 'open_roll_min_1000', 'open_roll_max_1000', 'open_roll_range_1000', 'open_momentum_1000', 'open_volatility_1000', 'high_roll_mean_1000', 'high_roll_std_1000', 'high_roll_min_1000', 'high_roll_max_1000', 'high_roll_range_1000', 'high_momentum_1000', 'high_volatility_1000', 'low_roll_mean_1000', 'low_roll_std_1000', 'low_roll_min_1000', 'low_roll_max_1000', 'low_roll_range_1000', 'low_momentum_1000', 'low_volatility_1000', 'close_roll_mean_1000', 'close_roll_std_1000', 'close_roll_min_1000', 'close_roll_max_1000', 'close_roll_range_1000', 'close_momentum_1000', 'close_volatility_1000', 'volume_roll_mean_1000', 'volume_roll_std_1000', 'volume_roll_min_1000', 'volume_roll_max_1000', 'volume_roll_range_1000', 'volume_momentum_1000', 'volume_volatility_1000', 'volume_roll_sum_10', 'volume_roll_sum_20', 'volume_roll_sum_50', 'volume_roll_sum_100', 'volume_roll_sum_200', 'volume_roll_sum_500', 'volume_roll_sum_1000', 'time_since_pvsra_gr_pattern', 'time_since_breaker_bbplus', 'time_since_breaker_signup', 'time_since_breaker_cnclup', 'time_since_breaker_ll1break', 'time_since_breaker_ll2break', 'time_since_breaker_sw1breakup', 'time_since_breaker_sw2breakup', 'time_since_breaker_tpup1', 'time_since_breaker_tpup2', 'time_since_breaker_tpup3', 'time_since_breaker_bb_endbl', 'time_since_breaker_bb_min', 'time_since_breaker_signdn', 'time_since_breaker_cncldn', 'time_since_breaker_hh1break', 'time_since_breaker_hh2break', 'time_since_breaker_sw1breakdn', 'time_since_breaker_sw2breakdn', 'time_since_breaker_tpdn1', 'time_since_breaker_tpdn2', 'time_since_breaker_tpdn3', 'time_since_breaker_bb_endbr', 'time_since_liq_ph_crossed', 'time_since_liq_pl_crossed', 'time_since_smc_t', 'time_since_smc_open', 'time_since_smc_high', 'time_since_smc_low', 'time_since_smc_close', 'time_since_smc_trend_internal', 'time_since_smc_trend_swing', 'time_since_smc_pivots', 'time_since_smc_alerts', 'time_since_smc_swing_obs', 'time_since_smc_int_obs', 'time_since_bb_signals', 'time_since_bb_alerts', 'price_range', 'price_range_pct', 'body_size', 'body_size_pct', 'upper_shadow', 'lower_shadow', 'volume_price_ratio', 'volume_change', 'volume_trend', 'volume_volatility', 'volume_close_ratio', 'volume_close_diff', 'volume_close_cross', 'volume_close_corr', 'hour', 'minute', 'second', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'second_sin', 'second_cos', 'hourly_volatility', 'minute_volatility'
        ]
        
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

    def _generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set including all rolling windows and technical indicators"""
        try:
            # Create all features in dictionaries first to avoid fragmentation
            feature_dicts = []
            
            # Basic OHLCV features
            basic_features = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    basic_features[col] = df[col]
                else:
                    basic_features[col] = 0.0
            feature_dicts.append(basic_features)
            
            # Rolling window features for different periods
            periods = [10, 20, 50, 100, 200, 500, 1000]
            
            for period in periods:
                period_features = {}
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        # Rolling statistics
                        period_features[f'{col}_roll_mean_{period}'] = df[col].rolling(window=period, min_periods=1).mean()
                        period_features[f'{col}_roll_std_{period}'] = df[col].rolling(window=period, min_periods=1).std().fillna(0)
                        period_features[f'{col}_roll_min_{period}'] = df[col].rolling(window=period, min_periods=1).min()
                        period_features[f'{col}_roll_max_{period}'] = df[col].rolling(window=period, min_periods=1).max()
                        
                        # Momentum
                        period_features[f'{col}_momentum_{period}'] = df[col].pct_change(periods=period).fillna(0)
                        
                        # Volatility
                        period_features[f'{col}_volatility_{period}'] = df[col].rolling(window=period, min_periods=1).std().fillna(0) / df[col].rolling(window=period, min_periods=1).mean().fillna(1)
                    else:
                        # Default values
                        period_features[f'{col}_roll_mean_{period}'] = 0.0
                        period_features[f'{col}_roll_std_{period}'] = 0.0
                        period_features[f'{col}_roll_min_{period}'] = 0.0
                        period_features[f'{col}_roll_max_{period}'] = 0.0
                        period_features[f'{col}_momentum_{period}'] = 0.0
                        period_features[f'{col}_volatility_{period}'] = 0.0
                
                # Calculate ranges after max/min are available
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        period_features[f'{col}_roll_range_{period}'] = period_features[f'{col}_roll_max_{period}'] - period_features[f'{col}_roll_min_{period}']
                    else:
                        period_features[f'{col}_roll_range_{period}'] = 0.0
                
                feature_dicts.append(period_features)
            
            # Volume rolling features
            volume_features = {}
            if 'volume' in df.columns:
                for period in periods:
                    volume_features[f'volume_roll_mean_{period}'] = df['volume'].rolling(window=period, min_periods=1).mean()
                    volume_features[f'volume_roll_std_{period}'] = df['volume'].rolling(window=period, min_periods=1).std().fillna(0)
                    volume_features[f'volume_roll_min_{period}'] = df['volume'].rolling(window=period, min_periods=1).min()
                    volume_features[f'volume_roll_max_{period}'] = df['volume'].rolling(window=period, min_periods=1).max()
                    volume_features[f'volume_momentum_{period}'] = df['volume'].pct_change(periods=period).fillna(0)
                    volume_features[f'volume_volatility_{period}'] = df['volume'].rolling(window=period, min_periods=1).std().fillna(0) / df['volume'].rolling(window=period, min_periods=1).mean().fillna(1)
                    volume_features[f'volume_roll_sum_{period}'] = df['volume'].rolling(window=period, min_periods=1).sum()
            else:
                for period in periods:
                    volume_features[f'volume_roll_mean_{period}'] = 0.0
                    volume_features[f'volume_roll_std_{period}'] = 0.0
                    volume_features[f'volume_roll_min_{period}'] = 0.0
                    volume_features[f'volume_roll_max_{period}'] = 0.0
                    volume_features[f'volume_momentum_{period}'] = 0.0
                    volume_features[f'volume_volatility_{period}'] = 0.0
                    volume_features[f'volume_roll_sum_{period}'] = 0.0
            
            # Calculate volume ranges
            for period in periods:
                if 'volume' in df.columns:
                    volume_features[f'volume_roll_range_{period}'] = volume_features[f'volume_roll_max_{period}'] - volume_features[f'volume_roll_min_{period}']
                else:
                    volume_features[f'volume_roll_range_{period}'] = 0.0
            
            feature_dicts.append(volume_features)
            
            # Time-based features
            time_features = {}
            if isinstance(df.index, pd.DatetimeIndex):
                time_features['hour'] = df.index.hour
                time_features['minute'] = df.index.minute
                time_features['second'] = df.index.second
                time_features['day_of_week'] = df.index.dayofweek
                time_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
                
                # Cyclical encoding
                time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24)
                time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24)
                time_features['minute_sin'] = np.sin(2 * np.pi * time_features['minute'] / 60)
                time_features['minute_cos'] = np.cos(2 * np.pi * time_features['minute'] / 60)
                time_features['second_sin'] = np.sin(2 * np.pi * time_features['second'] / 60)
                time_features['second_cos'] = np.cos(2 * np.pi * time_features['second'] / 60)
                
                # Volatility by time
                time_features['hourly_volatility'] = df['close'].rolling(window=60, min_periods=1).std().fillna(0) if 'close' in df.columns else 0.0
                time_features['minute_volatility'] = df['close'].rolling(window=4, min_periods=1).std().fillna(0) if 'close' in df.columns else 0.0
            else:
                # Default time values
                time_features['hour'] = 0
                time_features['minute'] = 0
                time_features['second'] = 0
                time_features['day_of_week'] = 0
                time_features['is_weekend'] = 0
                time_features['hour_sin'] = 0.0
                time_features['hour_cos'] = 1.0
                time_features['minute_sin'] = 0.0
                time_features['minute_cos'] = 1.0
                time_features['second_sin'] = 0.0
                time_features['second_cos'] = 1.0
                time_features['hourly_volatility'] = 0.0
                time_features['minute_volatility'] = 0.0
            
            feature_dicts.append(time_features)
            
            # Price action features
            price_features = {}
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                price_features['price_range'] = df['high'] - df['low']
                price_features['price_range_pct'] = price_features['price_range'] / df['close']
                price_features['body_size'] = abs(df['close'] - df['open'])
                price_features['body_size_pct'] = price_features['body_size'] / df['close']
                price_features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
                price_features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            else:
                price_features['price_range'] = 0.0
                price_features['price_range_pct'] = 0.0
                price_features['body_size'] = 0.0
                price_features['body_size_pct'] = 0.0
                price_features['upper_shadow'] = 0.0
                price_features['lower_shadow'] = 0.0
            
            feature_dicts.append(price_features)
            
            # Volume features
            volume_derived_features = {}
            if 'volume' in df.columns and 'close' in df.columns:
                volume_derived_features['volume_price_ratio'] = df['volume'] / df['close']
                volume_derived_features['volume_change'] = df['volume'].pct_change().fillna(0)
                volume_derived_features['volume_trend'] = df['volume'].rolling(window=20, min_periods=1).mean() / df['volume'].rolling(window=100, min_periods=1).mean().fillna(1)
                volume_derived_features['volume_volatility'] = df['volume'].rolling(window=20, min_periods=1).std().fillna(0) / df['volume'].rolling(window=20, min_periods=1).mean().fillna(1)
                volume_derived_features['volume_close_ratio'] = df['volume'] / df['close']
                volume_derived_features['volume_close_diff'] = df['volume'] - df['close']
                volume_derived_features['volume_close_cross'] = (df['volume'] > df['close']).astype(int)
                volume_derived_features['volume_close_corr'] = df['volume'].rolling(window=20, min_periods=1).corr(df['close']).fillna(0)
            else:
                volume_derived_features['volume_price_ratio'] = 0.0
                volume_derived_features['volume_change'] = 0.0
                volume_derived_features['volume_trend'] = 0.0
                volume_derived_features['volume_volatility'] = 0.0
                volume_derived_features['volume_close_ratio'] = 0.0
                volume_derived_features['volume_close_diff'] = 0.0
                volume_derived_features['volume_close_cross'] = 0
                volume_derived_features['volume_close_corr'] = 0.0
            
            feature_dicts.append(volume_derived_features)
            
            # Time-since features (simplified - all set to 0 for now)
            time_since_features = [
                'time_since_pvsra_gr_pattern', 'time_since_breaker_bbplus', 'time_since_breaker_signup',
                'time_since_breaker_cnclup', 'time_since_breaker_ll1break', 'time_since_breaker_ll2break',
                'time_since_breaker_sw1breakup', 'time_since_breaker_sw2breakup', 'time_since_breaker_tpup1',
                'time_since_breaker_tpup2', 'time_since_breaker_tpup3', 'time_since_breaker_bb_endbl',
                'time_since_breaker_bb_min', 'time_since_breaker_signdn', 'time_since_breaker_cncldn',
                'time_since_breaker_hh1break', 'time_since_breaker_hh2break', 'time_since_breaker_sw1breakdn',
                'time_since_breaker_sw2breakdn', 'time_since_breaker_tpdn1', 'time_since_breaker_tpdn2',
                'time_since_breaker_tpdn3', 'time_since_breaker_bb_endbr', 'time_since_liq_ph_crossed',
                'time_since_liq_pl_crossed', 'time_since_smc_t', 'time_since_smc_open', 'time_since_smc_high',
                'time_since_smc_low', 'time_since_smc_close', 'time_since_smc_trend_internal',
                'time_since_smc_trend_swing', 'time_since_smc_pivots', 'time_since_smc_alerts',
                'time_since_smc_swing_obs', 'time_since_smc_int_obs', 'time_since_bb_signals',
                'time_since_bb_alerts'
            ]
            
            time_since_dict = {feature: 0.0 for feature in time_since_features}
            feature_dicts.append(time_since_dict)
            
            # Additional indicator features (simplified)
            additional_features = [
                'pvsra_open', 'pvsra_high', 'pvsra_low', 'pvsra_close', 'pvsra_volume', 'pvsra_vec_color', 'pvsra_gr_pattern',
                'breaker_bbplus', 'breaker_signup', 'breaker_cnclup', 'breaker_ll1break', 'breaker_ll2break',
                'breaker_sw1breakup', 'breaker_sw2breakup', 'breaker_tpup1', 'breaker_tpup2', 'breaker_tpup3',
                'breaker_bb_endbl', 'breaker_bb_min', 'breaker_signdn', 'breaker_cncldn', 'breaker_hh1break',
                'breaker_hh2break', 'breaker_sw1breakdn', 'breaker_sw2breakdn', 'breaker_tpdn1', 'breaker_tpdn2',
                'breaker_tpdn3', 'breaker_bb_endbr', 'liq_ph_level', 'liq_ph_count', 'liq_ph_volume',
                'liq_ph_crossed', 'liq_pl_level', 'liq_pl_count', 'liq_pl_volume', 'liq_pl_crossed',
                'smc_t', 'smc_open', 'smc_high', 'smc_low', 'smc_close', 'smc_trend_internal', 'smc_trend_swing',
                'tr_ema_5', 'tr_ema_13', 'tr_ema_50', 'tr_ema_200', 'tr_ema_800', 'tr_yesterday_h', 'tr_yesterday_l',
                'tr_lastweek_h', 'tr_lastweek_l', 'tr_vec_color', 'tr_vcz_top', 'tr_vcz_bot',
                'bb_ts', 'bb_open', 'bb_high', 'bb_low', 'bb_close'
            ]
            
            additional_dict = {}
            for feature in additional_features:
                if feature.startswith('pvsra_') and feature != 'pvsra_vec_color' and feature != 'pvsra_gr_pattern':
                    # Map PVSRA features to OHLCV
                    base_col = feature.split('_', 1)[1]
                    if base_col in df.columns:
                        additional_dict[feature] = df[base_col]
                    else:
                        additional_dict[feature] = 0.0
                elif feature.startswith('bb_') and feature != 'bb_ts':
                    # Map BB features to OHLCV
                    base_col = feature.split('_', 1)[1]
                    if base_col in df.columns:
                        additional_dict[feature] = df[base_col]
                    else:
                        additional_dict[feature] = 0.0
                else:
                    # Default values for other features
                    additional_dict[feature] = 0.0
            
            # Set categorical features
            additional_dict['pvsra_vec_color'] = 0
            additional_dict['pvsra_gr_pattern'] = 0
            additional_dict['tr_vec_color'] = 0
            additional_dict['bb_ts'] = 0
            
            feature_dicts.append(additional_dict)
            
            # Combine all features at once using pd.concat
            result_df = pd.concat([pd.DataFrame(feature_dict, index=df.index) for feature_dict in feature_dicts], axis=1)
            
            logger.debug(f"Generated {len(result_df.columns)} comprehensive features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating comprehensive features: {str(e)}")
            return pd.DataFrame(index=df.index) 