import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
import smtplib
from email.mime.text import MIMEText
import requests
import redis
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringSystem:
    def __init__(self, config_path: str):
        """Initialize monitoring system with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize Redis connection
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Set up logging
        self.logger = logging.getLogger('monitoring')
        self.logger.setLevel(self.config['monitoring']['log_level'])
        
        # Initialize metrics storage
        self.metrics_key = 'trading:metrics'
        self.trades_key = 'trading:trades'
        self.alerts_key = 'trading:alerts'
        self.performance_key = 'trading:performance'
        
        # Load configuration
        self.risk_limits = self.config['monitoring']['risk_limits']
        self.performance_config = self.config['monitoring']['performance']
        self.regime_config = self.config['monitoring']['regime']
        self.alert_config = self.config['monitoring']['alerts']
        
    def log_trade(self, trade: Dict) -> None:
        """Log trade to Redis and file"""
        # Add timestamp if not present
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.now().isoformat()
            
        # Store in Redis with expiry
        trade_key = f"{self.trades_key}:{trade['id']}"
        self.redis.setex(
            trade_key,
            timedelta(days=7),  # Keep trades for 7 days
            json.dumps(trade)
        )
        
        # Add to trades list
        self.redis.lpush(self.trades_key, trade_key)
        self.redis.ltrim(self.trades_key, 0, 999)  # Keep last 1000 trades
        
        # Log to file
        self.logger.info(f"Trade executed: {json.dumps(trade)}")
        
    def log_risk_metrics(self, metrics: Dict) -> None:
        """Log risk metrics to Redis"""
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Store in Redis with expiry
        metrics_key = f"{self.metrics_key}:{metrics['timestamp']}"
        self.redis.setex(
            metrics_key,
            timedelta(days=1),  # Keep metrics for 1 day
            json.dumps(metrics)
        )
        
        # Add to metrics list
        self.redis.lpush(self.metrics_key, metrics_key)
        self.redis.ltrim(self.metrics_key, 0, 1439)  # Keep last 24 hours of metrics
        
    def check_risk_limits(self, metrics: Dict) -> List[str]:
        """Check if any risk limits are breached"""
        alerts = []
        
        # Check drawdown
        if metrics.get('drawdown', 0) > self.risk_limits['max_drawdown']:
            alerts.append(f"CRITICAL: Drawdown {metrics['drawdown']:.2%} exceeds limit {self.risk_limits['max_drawdown']:.2%}")
            
        # Check position size
        if metrics.get('position_size', 0) > self.risk_limits['max_position_size']:
            alerts.append(f"WARNING: Position size {metrics['position_size']:.2%} exceeds limit {self.risk_limits['max_position_size']:.2%}")
            
        # Check daily loss
        if metrics.get('daily_pnl', 0) < -self.risk_limits['max_daily_loss']:
            alerts.append(f"CRITICAL: Daily loss {abs(metrics['daily_pnl']):.2%} exceeds limit {self.risk_limits['max_daily_loss']:.2%}")
            
        # Check leverage
        if metrics.get('leverage', 0) > self.risk_limits['max_leverage']:
            alerts.append(f"WARNING: Leverage {metrics['leverage']:.1f}x exceeds limit {self.risk_limits['max_leverage']:.1f}x")
            
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < self.risk_limits['min_sharpe']:
            alerts.append(f"WARNING: Sharpe ratio {metrics['sharpe_ratio']:.2f} below minimum {self.risk_limits['min_sharpe']:.2f}")
            
        # Check correlation
        if metrics.get('market_correlation', 0) > self.risk_limits['max_correlation']:
            alerts.append(f"WARNING: Market correlation {metrics['market_correlation']:.2f} exceeds limit {self.risk_limits['max_correlation']:.2f}")
            
        # Store alerts in Redis
        if alerts:
            for alert in alerts:
                self.redis.lpush(self.alerts_key, alert)
                self.redis.ltrim(self.alerts_key, 0, 999)  # Keep last 1000 alerts
                
        return alerts
        
    def check_performance_degradation(self, metrics: Dict) -> List[str]:
        """Check for performance degradation"""
        alerts = []
        
        # Get historical performance from Redis
        performance_data = self.get_performance_history()
        if not performance_data:
            return alerts
            
        # Calculate metrics
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        drawdown_duration = metrics.get('drawdown_duration', 0)
        
        # Check win rate
        if win_rate < self.performance_config['min_win_rate']:
            alerts.append(f"WARNING: Win rate {win_rate:.2%} below minimum {self.performance_config['min_win_rate']:.2%}")
            
        # Check profit factor
        if profit_factor < self.performance_config['min_profit_factor']:
            alerts.append(f"WARNING: Profit factor {profit_factor:.2f} below minimum {self.performance_config['min_profit_factor']:.2f}")
            
        # Check drawdown duration
        if drawdown_duration > self.performance_config['max_drawdown_duration']:
            alerts.append(f"WARNING: Drawdown duration {drawdown_duration} days exceeds maximum {self.performance_config['max_drawdown_duration']} days")
            
        return alerts
        
    def check_regime_change(self, metrics: Dict) -> List[str]:
        """Check for market regime changes"""
        alerts = []
        
        # Get historical volatility from Redis
        volatility_data = self.get_volatility_history()
        if not volatility_data:
            return alerts
            
        # Calculate current volatility
        current_vol = metrics.get('volatility', 0)
        
        # Check volatility threshold
        if current_vol > self.regime_config['volatility_threshold']:
            alerts.append(f"INFO: High volatility regime detected: {current_vol:.2%}")
            
        # Check correlation threshold
        if metrics.get('market_correlation', 0) > self.regime_config['correlation_threshold']:
            alerts.append(f"INFO: High correlation regime detected: {metrics['market_correlation']:.2f}")
            
        return alerts
        
    def send_alert(self, level: str, message: str) -> None:
        """Send alert through configured channels"""
        # Store alert in Redis
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.redis.lpush(self.alerts_key, json.dumps(alert))
        self.redis.ltrim(self.alerts_key, 0, 999)  # Keep last 1000 alerts
        
        # Send email alert
        if self.alert_config['email']['enabled']:
            self._send_email_alert(level, message)
            
        # Send Slack alert
        if self.alert_config['slack']['enabled']:
            self._send_slack_alert(level, message)
            
    def _send_email_alert(self, level: str, message: str) -> None:
        """Send email alert"""
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"Trading Alert - {level}"
            msg['From'] = self.alert_config['email']['sender_email']
            msg['To'] = ', '.join(self.alert_config['email']['recipient_emails'])
            
            with smtplib.SMTP(self.alert_config['email']['smtp_server'], 
                            self.alert_config['email']['smtp_port']) as server:
                server.starttls()
                server.login(self.alert_config['email']['sender_email'],
                           self.alert_config['email']['sender_password'])
                server.send_message(msg)
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            
    def _send_slack_alert(self, level: str, message: str) -> None:
        """Send Slack alert"""
        try:
            payload = {
                'text': f"*{level}*: {message}",
                'username': 'Trading Bot',
                'icon_emoji': ':robot_face:'
            }
            requests.post(self.alert_config['slack']['webhook_url'], json=payload)
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            
    def generate_daily_report(self) -> Dict:
        """Generate daily performance report"""
        # Get today's metrics from Redis
        today = datetime.now().date()
        metrics = self.get_metrics_for_date(today)
        
        # Get today's trades from Redis
        trades = self.get_trades_for_date(today)
        
        # Calculate performance metrics
        report = {
            'date': today.isoformat(),
            'trades': len(trades),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'returns': self._calculate_returns(trades),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'drawdown': metrics.get('drawdown', 0),
            'position_sizes': self._get_position_sizes(trades),
            'risk_metrics': self._get_risk_metrics(metrics)
        }
        
        # Store report in Redis
        report_key = f"report:{today.isoformat()}"
        self.redis.setex(
            report_key,
            timedelta(days=30),  # Keep reports for 30 days
            json.dumps(report)
        )
        
        return report
        
    def get_metrics_for_date(self, date: datetime.date) -> Dict:
        """Get metrics for a specific date from Redis"""
        metrics = {}
        keys = self.redis.lrange(self.metrics_key, 0, -1)
        
        for key in keys:
            data = json.loads(self.redis.get(key))
            if datetime.fromisoformat(data['timestamp']).date() == date:
                metrics.update(data)
                
        return metrics
        
    def get_trades_for_date(self, date: datetime.date) -> List[Dict]:
        """Get trades for a specific date from Redis"""
        trades = []
        keys = self.redis.lrange(self.trades_key, 0, -1)
        
        for key in keys:
            data = json.loads(self.redis.get(key))
            if datetime.fromisoformat(data['timestamp']).date() == date:
                trades.append(data)
                
        return trades
        
    def get_performance_history(self) -> List[Dict]:
        """Get performance history from Redis"""
        history = []
        keys = self.redis.lrange(self.performance_key, 0, -1)
        
        for key in keys:
            data = json.loads(self.redis.get(key))
            history.append(data)
            
        return history
        
    def get_volatility_history(self) -> List[float]:
        """Get volatility history from Redis"""
        volatility = []
        keys = self.redis.lrange(self.metrics_key, 0, -1)
        
        for key in keys:
            data = json.loads(self.redis.get(key))
            if 'volatility' in data:
                volatility.append(data['volatility'])
                
        return volatility
        
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return winning_trades / len(trades)
        
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from trades"""
        if not trades:
            return 0.0
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
    def _calculate_returns(self, trades: List[Dict]) -> float:
        """Calculate total returns from trades"""
        return sum(t.get('pnl', 0) for t in trades)
        
    def _get_position_sizes(self, trades: List[Dict]) -> Dict:
        """Get position size statistics"""
        if not trades:
            return {'min': 0, 'max': 0, 'avg': 0}
            
        sizes = [t.get('position_size_pct', 0) for t in trades]
        return {
            'min': min(sizes),
            'max': max(sizes),
            'avg': sum(sizes) / len(sizes)
        }
        
    def _get_risk_metrics(self, metrics: Dict) -> Dict:
        """Get risk metrics"""
        return {
            'volatility': metrics.get('volatility', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'drawdown': metrics.get('drawdown', 0),
            'market_correlation': metrics.get('market_correlation', 0)
        } 