import logging
from datetime import datetime
from typing import Dict, List, Union
import yaml
import redis
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringSystem:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize monitoring system with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load monitoring settings
        self.monitoring_settings = self.config.get('monitoring', {})
        self.risk_settings = self.config.get('risk_management', {})
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.monitoring_settings.get('redis', {}).get('host', 'localhost'),
            port=self.monitoring_settings.get('redis', {}).get('port', 6379),
            db=self.monitoring_settings.get('redis', {}).get('db', 0)
        )
        
        # Initialize state
        self.metrics_history = []
        self.alerts = []
        self.running = False
        
        # Initialize dependencies
        self.paper_trader = None
        self.portfolio_analytics = None
        self.data_processor = None
        
        logger.info("Monitoring system initialized")
        
    def start(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitoring system is already running")
            return
            
        try:
            self.running = True
            logger.info("Monitoring system started")
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
            logger.info("Monitoring system stopped")
        except Exception as e:
            logger.error("Error stopping monitoring system: %s", str(e))
            raise
            
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
        """Background thread for monitoring system"""
        while self.running:
            try:
                # Check risk limits
                if not self.check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    self.paper_trader.pause_trading()
                else:
                    # Only resume if we have trades
                    if self.paper_trader.get_trade_history():
                        self.paper_trader.resume_trading()
                    
                # Sleep for monitoring interval
                time.sleep(self.config['monitoring']['interval'])
                
            except Exception as e:
                logger.error("Error in monitoring loop: %s", str(e))
                time.sleep(1)  # Wait a bit before retrying
            
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