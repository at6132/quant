import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import yaml
import redis
import json
import websocket
import pandas as pd
from Core.datafeed_kraken import KrakenDataFeed
from paper_trading.data_processor import DataProcessor
from paper_trading.paper_trader import PaperTrader
from paper_trading.risk_manager import RiskManager
from paper_trading.monitoring import MonitoringSystem
from paper_trading.portfolio_analytics import PortfolioAnalytics
from paper_trading.utils import process_chunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingLoop:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize trading loop with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.paper_trader = PaperTrader(self.config)
        self.risk_manager = RiskManager(self.config)
        self.portfolio_analytics = PortfolioAnalytics(self.config)
        self.monitoring = MonitoringSystem(self.config)
        
        # Set monitoring system dependencies
        self.monitoring.set_dependencies(
            self.paper_trader,
            self.portfolio_analytics,
            self.data_processor
        )
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.config['monitoring']['redis']['host'],
            port=self.config['monitoring']['redis']['port'],
            db=self.config['monitoring']['redis']['db']
        )
        
        # Initialize state
        self.running = False
        self.paused = False
        
        # Initialize websocket
        self.ws = None
        
        logger.info("Trading loop initialized")
        
    def start(self):
        """Start the trading loop"""
        if self.running:
            logger.warning("Trading loop is already running")
            return
            
        try:
            logger.info("Initializing trading loop...")
            
            # Set running flag first
            self.running = True
            
            # Connect to Kraken websocket
            self.ws = websocket.WebSocketApp(
                self.config['data']['kraken']['websocket_url'],
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start websocket in a separate thread
            logger.info("Starting websocket thread...")
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Start monitoring in a separate thread
            logger.info("Starting monitoring thread...")
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            # Start timeframe processing in a separate thread
            logger.info("Starting timeframe processing thread...")
            self.timeframe_thread = threading.Thread(target=self._timeframe_loop)
            self.timeframe_thread.daemon = True
            self.timeframe_thread.start()
            
            # Wait for websocket to connect
            logger.info("Waiting for websocket connection...")
            time.sleep(2)
            
            logger.info("Trading loop started successfully")
            
        except Exception as e:
            logger.error("Error starting trading loop: %s", str(e))
            self.running = False
            self.stop()
            raise
            
    def _on_open(self, ws):
        """Handle websocket connection open"""
        try:
            # Subscribe to ticker for each symbol
            for symbol in self.config['trading']['symbols']:
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": [symbol],
                    "subscription": {
                        "name": "ticker"
                    }
                }
                ws.send(json.dumps(subscribe_msg))
                
            logger.info("Websocket connected and subscribed to symbols")
            
        except Exception as e:
            logger.error("Error in websocket on_open: %s", str(e))
            
    def _on_message(self, ws, message):
        """Handle incoming websocket message"""
        try:
            data = json.loads(message)
            
            # Handle ticker data
            if isinstance(data, list) and len(data) > 1:
                ticker_data = data[1]
                if isinstance(ticker_data, dict):
                    # Extract price data
                    price_data = {
                        'symbol': data[3],
                        'price': float(ticker_data['c'][0]),
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    # Update data processor
                    self._handle_market_data(price_data)
                    
        except Exception as e:
            logger.error("Error in websocket on_message: %s", str(e))
            
    def _on_error(self, ws, error):
        """Handle websocket error"""
        logger.error("Websocket error: %s", str(error))
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle websocket connection close"""
        logger.info("Websocket connection closed")
        
    def _handle_market_data(self, data: Dict):
        """Handle incoming market data"""
        try:
            # Extract symbol from data
            symbol = data.get('symbol', self.config['trading']['symbols'][0])
            
            # Convert timestamp to datetime
            try:
                # Try to parse timestamp as milliseconds
                timestamp = datetime.fromtimestamp(data['timestamp'] / 1000)
            except (ValueError, TypeError):
                # If that fails, try to parse as seconds
                timestamp = datetime.fromtimestamp(data['timestamp'])
            
            # Round timestamp to nearest 15-second interval
            rounded_timestamp = timestamp.replace(
                second=(timestamp.second // 15) * 15,
                microsecond=0
            )
            
            # Update data processor
            self.data_processor.update_price_data(
                symbol,
                data['price'],
                timestamp
            )
            
            # Store raw price data in Redis
            self.redis_client.hset(
                'market_data:raw',
                timestamp.isoformat(),
                json.dumps({
                    'price': float(data['price']),
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol
                })
            )
            
            # For each timeframe, aggregate the raw data into OHLCV candles
            for tf in self.config['trading']['timeframes']:
                # Get existing candle data for this timeframe
                tf_data = self.redis_client.hgetall(f'market_data:{tf}')
                if tf_data:
                    tf_data = {
                        k.decode(): json.loads(v)
                        for k, v in tf_data.items()
                    }
                else:
                    tf_data = {}
                    
                # Get or create candle for current interval
                candle_key = rounded_timestamp.isoformat()
                if candle_key not in tf_data:
                    tf_data[candle_key] = {
                        'open': float(data['price']),
                        'high': float(data['price']),
                        'low': float(data['price']),
                        'close': float(data['price']),
                        'volume': float(data.get('volume', 0)),
                        'timestamp': candle_key,
                        'symbol': symbol
                    }
                else:
                    # Update existing candle
                    candle = tf_data[candle_key]
                    candle['high'] = max(candle['high'], float(data['price']))
                    candle['low'] = min(candle['low'], float(data['price']))
                    candle['close'] = float(data['price'])
                    candle['volume'] += float(data.get('volume', 0))
                    
                # Store updated candle in Redis
                self.redis_client.hset(
                    f'market_data:{tf}',
                    candle_key,
                    json.dumps(tf_data[candle_key])
                )
            
        except Exception as e:
            logger.error("Error handling market data: %s", str(e))
            logger.debug("Market data: %s", data)  # Log the data for debugging
            
    def _timeframe_loop(self):
        """Background thread for processing timeframes"""
        logger.info("Starting timeframe processing loop")
        while self.running:
            try:
                logger.debug("Processing timeframes...")
                self._process_timeframes()
                time.sleep(15)  # Process every 15 seconds
            except Exception as e:
                logger.error("Error in timeframe loop: %s", str(e))
                time.sleep(1)  # Wait a bit before retrying
                
    def _process_timeframes(self):
        """Process all timeframes and indicators"""
        try:
            # Get data for all timeframes
            timeframes = self.config['trading']['timeframes']
            data = {}
            
            for tf in timeframes:
                # Get data from Redis
                tf_data = self.redis_client.hgetall(f'market_data:{tf}')
                logger.debug(f"Raw Redis data for {tf}: {tf_data}")
                if tf_data:
                    # Convert to DataFrame with proper index
                    df_data = []
                    for k, v in tf_data.items():
                        row = json.loads(v)
                        df_data.append({
                            'timestamp': pd.to_datetime(row['timestamp']),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        })
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        data[tf] = df
                    else:
                        logger.error(f"No valid OHLCV data for timeframe {tf}, skipping.")
                        
            # Process indicators for each timeframe
            for tf, df in data.items():
                if len(df) > 0:
                    try:
                        # Process indicators
                        result = process_chunk(df, 0, len(df))
                        
                        # Handle tuple output (processed_df, errors)
                        if isinstance(result, tuple) and len(result) == 2:
                            processed_df, errors = result
                            
                            # Log any errors
                            if errors:
                                for error in errors:
                                    logger.error("Error processing indicators: %s", error)
                            
                            # Process the DataFrame if it exists
                            if isinstance(processed_df, pd.DataFrame) and not processed_df.empty:
                                # Ensure unique column names
                                processed_df.columns = [f"{col}_{i}" if processed_df.columns.tolist().count(col) > 1 
                                                     else col for i, col in enumerate(processed_df.columns)]
                                
                                # Store processed data in Redis
                                processed_data = processed_df.to_dict(orient='index')
                                for timestamp, row in processed_data.items():
                                    self.redis_client.hset(
                                        f'processed_data:{tf}',
                                        timestamp.isoformat(),
                                        json.dumps(row)
                                    )
                                
                                # Log successful processing
                                logger.info(f"Processed {len(processed_df)} rows for {tf} timeframe")
                                
                                # Print model labels for 15s timeframe
                                if tf == '15s':
                                    logger.info("=== Model Labels for 15s Candle ===")
                                    logger.info("Price: %.2f", df['close'].iloc[-1])
                                    
                                    # Get signals
                                    signals = self.data_processor.get_signals(tf)
                                    if signals:
                                        logger.info("Signals: %s", json.dumps(signals, indent=2))
                                    
                                    # Check positions
                                    positions = self.paper_trader.get_positions()
                                    if positions:
                                        logger.info("=== Risk Model Response ===")
                                        for symbol, position in positions.items():
                                            # Get risk metrics
                                            risk_metrics = self.risk_manager.get_risk_metrics()
                                            logger.info("Position: %s", json.dumps(position, indent=2))
                                            logger.info("Risk Metrics: %s", json.dumps(risk_metrics, indent=2))
                                            
                                            # Get portfolio analytics
                                            portfolio_metrics = self.portfolio_analytics.calculate_portfolio_metrics(
                                                positions,
                                                self.paper_trader.get_trade_history(),
                                                self.data_processor.get_price_data(symbol)
                                            )
                                            logger.info("Portfolio Metrics: %s", json.dumps(portfolio_metrics, indent=2))
                                    
                                    logger.info("=== End of 15s Update ===")
                            else:
                                logger.error(f"Invalid or empty processed DataFrame for timeframe {tf}")
                        else:
                            logger.error(f"Unexpected process_chunk return type: {type(result)}")
                            
                    except Exception as e:
                        logger.error("Error processing timeframe %s: %s", tf, str(e))
                        logger.debug("DataFrame shape: %s", df.shape)
                        logger.debug("DataFrame columns: %s", df.columns.tolist())
                    
        except Exception as e:
            logger.error("Error processing timeframes: %s", str(e))
            logger.debug("Data: %s", data)  # Log the data for debugging
            
    def _monitoring_loop(self):
        """Background thread for monitoring"""
        while self.running:
            try:
                self._update_monitoring()
                time.sleep(self.config['monitoring']['metrics_interval'])
            except Exception as e:
                logger.error("Error in monitoring loop: %s", str(e))
                
    def _update_monitoring(self):
        """Update monitoring system with latest metrics"""
        try:
            # Get current metrics
            metrics = self.portfolio_analytics.calculate_portfolio_metrics(
                self.paper_trader.get_positions(),
                self.paper_trader.get_trade_history(),
                self.data_processor.get_price_data(self.config['trading']['symbols'][0])
            )
            
            # Log metrics
            self.monitoring.log_metrics(metrics)
            
            # Check risk limits
            if not self.monitoring.check_risk_limits(metrics):
                logger.warning("Risk limits exceeded, pausing trading")
                self.paused = True
            else:
                self.paused = False
                
        except Exception as e:
            logger.error("Error updating monitoring: %s", str(e))
            
    def stop(self):
        """Stop the trading loop"""
        if not self.running:
            logger.warning("Trading loop is not running")
            return
            
        try:
            # Set running flag to False
            self.running = False
            
            # Close websocket
            if self.ws:
                self.ws.close()
                
            # Wait for threads to finish
            if hasattr(self, 'ws_thread'):
                self.ws_thread.join(timeout=5)
            if hasattr(self, 'monitoring_thread'):
                self.monitoring_thread.join(timeout=5)
            if hasattr(self, 'timeframe_thread'):
                self.timeframe_thread.join(timeout=5)
                
            logger.info("Trading loop stopped")
            
        except Exception as e:
            logger.error("Error stopping trading loop: %s", str(e))
            raise 