import logging
import time
import json
import redis
import threading
from typing import Dict, List, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime

from paper_trading.data_processor import DataProcessor
from paper_trading.trading_loop import TradingLoop
from paper_trading.portfolio_analytics import PortfolioAnalytics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringSystem:
    def __init__(self, config: Dict, data_processor: Any = None,
                 trading_loop: Any = None, portfolio_analytics: Any = None):
        """Initialize monitoring system with configuration and components
        
        Args:
            config: Configuration dictionary
            data_processor: Data processor instance
            trading_loop: Trading loop instance
            portfolio_analytics: Portfolio analytics instance
        """
        self.config = config
        self.data_processor = data_processor
        self.trading_loop = trading_loop
        self.portfolio_analytics = portfolio_analytics
        self.running = False
        self.monitoring_thread = None
        
        # Load monitoring settings
        self.monitoring_settings = self.config.get('monitoring', {})
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.monitoring_settings['redis']['host'],
            port=self.monitoring_settings['redis']['port'],
            db=self.monitoring_settings['redis']['db']
        )
        
        # Initialize state
        self.metrics = {}
        self.alerts = []
        
        logger.info("Monitoring system initialized")
        
    def start(self):
        """Start the monitoring system in a separate thread"""
        if self.running:
            logger.warning("Monitoring system is already running")
            return
            
        try:
            self.running = True
            logger.info("Starting monitoring system...")
            
            # Start monitoring in a separate thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Monitoring system started in background thread")
            
        except Exception as e:
            logger.error("Error starting monitoring system: %s", str(e))
            self.running = False
            raise
            
    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            logger.warning("Monitoring system is not running")
            return
            
        try:
            self.running = False
            
            # Wait for the monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)  # Wait up to 5 seconds
                
            logger.info("Monitoring system stopped")
        except Exception as e:
            logger.error("Error stopping monitoring system: %s", str(e))
            raise
            
    def _update_metrics(self):
        """Update monitoring metrics"""
        try:
            # Get portfolio metrics if portfolio_analytics is available
            if self.portfolio_analytics and hasattr(self.portfolio_analytics, 'get_metrics'):
                try:
                    portfolio_metrics = self.portfolio_analytics.get_metrics()
                    self.metrics.update(portfolio_metrics)
                except Exception as e:
                    logger.debug(f"Could not get portfolio metrics: {e}")
            
            # Get trading metrics if trading_loop and data_processor are available
            if self.trading_loop and self.data_processor:
                try:
                    # Get symbols from trading loop
                    symbols = getattr(self.trading_loop, 'symbols', [self.trading_loop.symbol])
                    
                    for symbol in symbols:
                        # Get latest data
                        price_data = self.data_processor.get_price_data(symbol)
                        if not price_data:
                            continue
                            
                        # Get latest bar
                        latest_bar = price_data[-1]
                        
                        # Get indicators
                        try:
                            indicators = self.data_processor.get_indicators(symbol)
                        except:
                            indicators = None
                        
                        # Get predictions
                        predictions = self.data_processor.get_predictions(symbol)
                        if not predictions:
                            continue
                            
                        # Update symbol metrics
                        self.metrics[f"{symbol}_price"] = latest_bar['close']
                        self.metrics[f"{symbol}_volume"] = latest_bar.get('volume', 0)
                        self.metrics[f"{symbol}_prediction"] = predictions['probability']
                        
                        # Add key indicators if available
                        if indicators is not None:
                            for indicator in self.monitoring_settings.get('key_indicators', []):
                                if indicator in indicators.columns:
                                    self.metrics[f"{symbol}_{indicator}"] = indicators[indicator].iloc[-1]
                except Exception as e:
                    logger.debug(f"Could not get trading metrics: {e}")
                        
        except Exception as e:
            logger.debug("Error updating metrics: %s", str(e))
            
    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            # Check portfolio alerts if portfolio_analytics is available
            if self.portfolio_analytics and hasattr(self.portfolio_analytics, 'check_alerts'):
                try:
                    portfolio_alerts = self.portfolio_analytics.check_alerts()
                    self.alerts.extend(portfolio_alerts)
                except Exception as e:
                    logger.debug(f"Could not check portfolio alerts: {e}")
            
            # Check trading alerts if trading_loop is available
            if self.trading_loop:
                try:
                    # Get symbols from trading loop
                    symbols = getattr(self.trading_loop, 'symbols', [self.trading_loop.symbol])
                    
                    for symbol in symbols:
                        # Get current metrics
                        price = self.metrics.get(f"{symbol}_price")
                        prediction = self.metrics.get(f"{symbol}_prediction")
                        
                        if price is None or prediction is None:
                            continue
                            
                        # Check price alerts
                        if price > self.monitoring_settings.get('price_alert_threshold', float('inf')):
                            self.alerts.append({
                                'type': 'price_alert',
                                'symbol': symbol,
                                'message': f"Price {price} exceeds threshold"
                            })
                            
                        # Check prediction alerts
                        if prediction > self.monitoring_settings.get('prediction_alert_threshold', 0.95):
                            self.alerts.append({
                                'type': 'prediction_alert',
                                'symbol': symbol,
                                'message': f"Prediction {prediction} exceeds threshold"
                            })
                except Exception as e:
                    logger.debug(f"Could not check trading alerts: {e}")
                    
            # Keep only recent alerts
            max_alerts = self.monitoring_settings.get('max_alerts', 100)
            if len(self.alerts) > max_alerts:
                self.alerts = self.alerts[-max_alerts:]
                
        except Exception as e:
            logger.debug("Error checking alerts: %s", str(e))
            
    def _publish_metrics(self):
        """Publish metrics to Redis"""
        try:
            # Publish metrics
            for key, value in self.metrics.items():
                self.redis_client.set(f"metrics:{key}", value)
                
            # Publish alerts
            if self.alerts:
                self.redis_client.lpush("alerts", *[json.dumps(alert) for alert in self.alerts])
                self.alerts.clear()
                
        except Exception as e:
            logger.error("Error publishing metrics: %s", str(e))
            
    def get_metrics(self) -> Dict:
        """Get current metrics
        
        Returns:
            Dictionary of current metrics
        """
        return self.metrics.copy()
        
    def get_alerts(self) -> List[Dict]:
        """Get current alerts
        
        Returns:
            List of current alerts
        """
        return self.alerts.copy()

    def log_metrics(self, metrics: Dict):
        """Log metrics to Redis"""
        try:
            # Store metrics in Redis
            self.redis_client.hset(
                'metrics',
                datetime.now().isoformat(),
                json.dumps(metrics)
            )
            
            logger.info("Logged metrics: %s", metrics)
            
        except Exception as e:
            logger.error("Error logging metrics: %s", str(e))
            
    def set_dependencies(self, paper_trader, portfolio_analytics, data_processor):
        """Set dependencies for monitoring system"""
        self.paper_trader = paper_trader
        self.portfolio_analytics = portfolio_analytics
        self.data_processor = data_processor

    def check_risk_limits(self, *args, **kwargs):
        """Check if any risk limits have been exceeded"""
        try:
            if not all([self.paper_trader, self.portfolio_analytics, self.data_processor]):
                logger.warning("Dependencies not set, skipping risk checks")
                return True
                
            # Get current positions and trade history
            positions = self.paper_trader.get_positions()
            trade_history = self.paper_trader.get_trade_history()
            
            # Skip risk checks if no trades have been made
            if not trade_history:
                logger.debug("No trades yet, skipping risk checks")
                return True
                
            # Calculate portfolio metrics
            portfolio_metrics = self.portfolio_analytics.calculate_portfolio_metrics(
                positions,
                trade_history,
                self.data_processor.get_price_data(self.config['trading']['symbols'][0])
            )
            
            # Check Sharpe ratio
            if portfolio_metrics['sharpe_ratio'] < self.config['risk']['min_sharpe_ratio']:
                logger.warning("Sharpe ratio too low: %.2f < %.2f",
                             portfolio_metrics['sharpe_ratio'],
                             self.config['risk']['min_sharpe_ratio'])
                return False
                
            # Check Sortino ratio
            if portfolio_metrics['sortino_ratio'] < self.config['risk']['min_sortino_ratio']:
                logger.warning("Sortino ratio too low: %.2f < %.2f",
                             portfolio_metrics['sortino_ratio'],
                             self.config['risk']['min_sortino_ratio'])
                return False
                
            # Check win rate
            if portfolio_metrics['win_rate'] < self.config['risk']['min_win_rate']:
                logger.warning("Win rate too low: %.2f < %.2f",
                             portfolio_metrics['win_rate'],
                             self.config['risk']['min_win_rate'])
                return False
                
            # Check max drawdown
            if portfolio_metrics['max_drawdown'] > self.config['risk']['max_drawdown']:
                logger.warning("Max drawdown too high: %.2f > %.2f",
                             portfolio_metrics['max_drawdown'],
                             self.config['risk']['max_drawdown'])
                return False
                
            return True
            
        except Exception as e:
            logger.error("Error checking risk limits: %s", str(e))
            return False
            
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        try:
            while self.running:
                try:
                    # Update metrics
                    self._update_metrics()
                    
                    # Check for alerts
                    self._check_alerts()
                    
                    # Publish metrics to Redis
                    self._publish_metrics()
                    
                    # Sleep for monitoring interval
                    time.sleep(self.monitoring_settings.get('interval', 1))
                    
                except Exception as e:
                    logger.error("Error in monitoring loop: %s", str(e))
                    time.sleep(1)  # Brief pause on error
                    
        except Exception as e:
            logger.error("Error in monitoring thread: %s", str(e))
            self.running = False
            
    def log_trade(self, trade: Dict):
        """Log trade to Redis"""
        try:
            timestamp = datetime.now().isoformat()
            self.redis_client.hset('trades', timestamp, json.dumps(trade))
            logger.info("Logged trade: %s", trade)
        except Exception as e:
            logger.error("Error logging trade: %s", str(e))
            
    def log_position(self, position: Dict):
        """Log position to Redis"""
        try:
            timestamp = datetime.now().isoformat()
            self.redis_client.hset('positions', timestamp, json.dumps(position))
            logger.info("Logged position: %s", position)
        except Exception as e:
            logger.error("Error logging position: %s", str(e))
            
    def send_alert(self, alert: str):
        """Send alert to Redis"""
        try:
            # Store alert in Redis
            self.redis_client.hset(
                'alerts',
                datetime.now().isoformat(),
                json.dumps({
                    'message': alert,
                    'level': 'warning'
                })
            )
            
            logger.warning("Alert: %s", alert)
            
        except Exception as e:
            logger.error("Error sending alert: %s", str(e))
            
    def generate_daily_report(self):
        """Generate daily report from logged metrics"""
        try:
            # Get all metrics for today
            today = datetime.now().date().isoformat()
            metrics = self.redis_client.hgetall('metrics')
            
            if not metrics:
                logger.warning("No metrics found for today")
                return
                
            # Process metrics
            daily_metrics = {}
            for timestamp, value in metrics.items():
                if timestamp.decode().startswith(today):
                    daily_metrics[timestamp.decode()] = json.loads(value)
                    
            # Store report
            report = {
                'date': today,
                'metrics': daily_metrics
            }
            
            self.redis_client.hset('reports', today, json.dumps(report))
            logger.info("Generated daily report for %s", today)
            
        except Exception as e:
            logger.error("Error generating daily report: %s", str(e))
            
    def get_metrics(self) -> Dict:
        """Get latest metrics from Redis"""
        try:
            # Get latest metrics
            metrics = self.redis_client.hgetall('metrics')
            if metrics:
                latest = max(metrics.keys(), key=lambda x: x.decode())
                return json.loads(metrics[latest])
            return {}
            
        except Exception as e:
            logger.error("Error getting metrics: %s", str(e))
            return {}
            
    def get_alerts(self) -> List[Dict]:
        """Get latest alerts from Redis"""
        try:
            # Get latest alerts
            alerts = self.redis_client.hgetall('alerts')
            if alerts:
                return [
                    json.loads(v)
                    for k, v in sorted(alerts.items(), key=lambda x: x[0].decode())
                ]
            return []
            
        except Exception as e:
            logger.error("Error getting alerts: %s", str(e))
            return [] 