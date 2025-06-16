import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import yaml
import redis
import json
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from paper_trading.trading_loop import TradingLoop

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebApp:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize webapp with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load settings
        self.webapp_settings = self.config.get('webapp', {})
        self.monitoring_settings = self.config.get('monitoring', {})
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'your-secret-key'  # Change this in production
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.monitoring_settings.get('redis', {}).get('host', 'localhost'),
            port=self.monitoring_settings.get('redis', {}).get('port', 6379),
            db=self.monitoring_settings.get('redis', {}).get('db', 0)
        )
        
        # Initialize state
        self.running = False
        self.trading_loop = None
        
        # Set up routes
        self._setup_routes()
        
        logger.info("Webapp initialized")
        
    def _setup_routes(self):
        """Set up Flask routes"""
        @self.app.route('/')
        def index():
            """Render the main dashboard."""
            return render_template('index.html')
            
        @self.app.route('/api/portfolio')
        def get_portfolio():
            """Get latest portfolio metrics"""
            try:
                # Get latest portfolio metrics
                metrics = self.redis_client.hgetall('portfolio_metrics')
                if metrics:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in metrics.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting portfolio data: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/performance')
        def get_performance():
            """Get latest performance metrics"""
            try:
                # Get latest performance metrics
                metrics = self.redis_client.hgetall('metrics')
                if metrics:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in metrics.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting performance data: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get latest alerts"""
            try:
                # Get latest alerts
                alerts = self.redis_client.hgetall('alerts')
                if alerts:
                    data = [
                        json.loads(v)
                        for k, v in sorted(alerts.items(), key=lambda x: x[0].decode())
                    ]
                    return jsonify(data)
                    
                return jsonify([])
                
            except Exception as e:
                logger.error("Error getting alerts: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get latest metrics"""
            try:
                # Get latest metrics
                metrics = self.redis_client.hgetall('metrics')
                if metrics:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in metrics.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting metrics: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/trades')
        def get_trades():
            """Get latest trades"""
            try:
                # Get latest trades
                trades = self.redis_client.hgetall('trades')
                if trades:
                    data = [
                        json.loads(v)
                        for k, v in sorted(trades.items(), key=lambda x: x[0].decode())
                    ]
                    return jsonify(data)
                    
                return jsonify([])
                
            except Exception as e:
                logger.error("Error getting trades: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions"""
            try:
                # Get current positions
                positions = self.redis_client.hgetall('positions')
                if positions:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in positions.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting positions: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/market_data')
        def get_market_data():
            """Get latest market data"""
            try:
                # Get latest market data
                market_data = self.redis_client.hgetall('market_data')
                if market_data:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in market_data.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting market data: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/indicators')
        def get_indicators():
            """Get latest indicators"""
            try:
                # Get latest indicators
                indicators = self.redis_client.hgetall('indicators')
                if indicators:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in indicators.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting indicators: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/signals')
        def get_signals():
            """Get latest signals"""
            try:
                # Get latest signals
                signals = self.redis_client.hgetall('signals')
                if signals:
                    data = {
                        k.decode(): json.loads(v)
                        for k, v in signals.items()
                    }
                    return jsonify(data)
                    
                return jsonify({})
                
            except Exception as e:
                logger.error("Error getting signals: %s", str(e))
                return jsonify({'error': str(e)}), 500
                
    def start(self):
        """Start the webapp"""
        if self.running:
            logger.warning("Webapp is already running")
            return
            
        try:
            # Initialize trading loop
            self.trading_loop = TradingLoop(self.config)
            
            # Start trading loop in a separate thread
            self.trading_thread = threading.Thread(target=self.trading_loop.start)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # Start Flask app
            self.app.run(
                host=self.webapp_settings.get('host', '0.0.0.0'),
                port=self.webapp_settings.get('port', 5000),
                debug=self.webapp_settings.get('debug', False)
            )
            
            self.running = True
            logger.info("Webapp started")
            
        except Exception as e:
            logger.error("Error starting webapp: %s", str(e))
            self.stop()
            raise
            
    def stop(self):
        """Stop the webapp"""
        if not self.running:
            logger.warning("Webapp is not running")
            return
            
        try:
            # Stop trading loop
            if self.trading_loop:
                self.trading_loop.stop()
                
            self.running = False
            logger.info("Webapp stopped")
            
        except Exception as e:
            logger.error("Error stopping webapp: %s", str(e))
            raise

def load_config(config_path: str = "paper_trading_config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_redis_connection(config: Dict) -> redis.Redis:
    """Create Redis connection"""
    return redis.Redis(
        host=config['monitoring']['redis']['host'],
        port=config['monitoring']['redis']['port'],
        db=config['monitoring']['redis']['db']
    )

def start_webapp(config_path: str = "paper_trading_config.yaml"):
    """Start the Flask web application"""
    try:
        config = load_config(config_path)
        app.config['config'] = config
        
        host = config['webapp']['host']
        port = config['webapp']['port']
        debug = config['webapp']['debug']
        
        logger.info("Starting web application on %s:%d", host, port)
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error("Error starting web application: %s", str(e))
        raise

def emit_updates():
    """Emit real-time updates to connected clients."""
    try:
        # Get latest data from Redis
        metrics = json.loads(redis_client.get('portfolio_metrics') or '{}')
        positions = json.loads(redis_client.get('current_positions') or '{}')
        trades = json.loads(redis_client.get('recent_trades') or '[]')
        alerts = json.loads(redis_client.get('recent_alerts') or '[]')
        
        # Emit updates through WebSocket
        socketio.emit('metrics_update', metrics)
        socketio.emit('positions_update', positions)
        socketio.emit('trades_update', trades)
        socketio.emit('alerts_update', alerts)
    except Exception as e:
        logger.error(f"Error emitting updates: {str(e)}")

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit_updates()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

@app.route('/api/report', methods=['GET'])
def get_daily_report():
    """Get daily performance report"""
    try:
        config = load_config()
        redis_client = get_redis_connection(config)
        
        # Get today's metrics
        today = datetime.now().date()
        metrics = []
        
        for key in redis_client.hkeys("metrics"):
            metric = json.loads(redis_client.hget("metrics", key))
            metric_date = datetime.fromisoformat(metric['timestamp']).date()
            
            if metric_date == today:
                metrics.append(metric)
                
        if not metrics:
            return jsonify({})
            
        # Calculate daily statistics
        latest_metric = metrics[-1]
        
        report = {
            'date': today.isoformat(),
            'portfolio_value': latest_metric['portfolio_value'],
            'risk_metrics': latest_metric['risk_metrics'],
            'performance_metrics': latest_metric['performance_metrics'],
            'trade_statistics': latest_metric['trade_statistics'],
            'pnl_metrics': latest_metric['pnl_metrics']
        }
        
        return jsonify(report)
        
    except Exception as e:
        logger.error("Error getting daily report: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    webapp = WebApp()
    webapp.start() 
    