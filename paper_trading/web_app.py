import logging
import json
import yaml
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import redis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebApp:
    def __init__(self, config: dict):
        """Initialize web application
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.app = Flask(__name__)
        self.redis_client = None
        
        # Initialize Redis connection
        redis_config = self.config.get('monitoring', {}).get('redis', {})
        try:
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0)
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Set up routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up Flask routes"""
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/portfolio')
        def get_portfolio():
            """Get comprehensive portfolio data"""
            try:
                if self.redis_client:
                    # Get portfolio data from Redis
                    portfolio_data = self.redis_client.get('portfolio_data')
                    if portfolio_data:
                        data = json.loads(portfolio_data)
                    else:
                        # Fallback to individual keys
                        data = self._get_portfolio_from_individual_keys()
                    
                    return jsonify({
                        'status': 'success',
                        'data': data
                    })
                else:
                    # Fallback to config values
                    account_balance = self.config.get('account', {}).get('initial_balance', 1000000.0)
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'portfolio_value': account_balance,
                            'account_balance': account_balance,
                            'total_pnl': 0.0,
                            'daily_pnl': 0.0,
                            'positions': {},
                            'trade_history': [],
                            'timestamp': datetime.now().isoformat()
                        }
                    })
            except Exception as e:
                logger.error(f"Error getting portfolio data: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
                
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get real-time metrics from trading system"""
            try:
                if self.redis_client:
                    # Get portfolio value
                    portfolio_value = self.redis_client.get('portfolio_value')
                    portfolio_value = float(portfolio_value) if portfolio_value else 1000000.0
                    
                    # Get daily PnL
                    daily_pnl = self.redis_client.get('daily_pnl')
                    daily_pnl = float(daily_pnl) if daily_pnl else 0.0
                    
                    # Get total PnL
                    total_pnl = self.redis_client.get('total_pnl')
                    total_pnl = float(total_pnl) if total_pnl else 0.0
                    
                    # Get positions
                    positions = self.redis_client.get('positions')
                    positions = json.loads(positions) if positions else {}
                    
                    # Get account balance
                    account_balance = self.redis_client.get('account_balance')
                    account_balance = float(account_balance) if account_balance else 1000000.0
                    
                    # Get last update time
                    last_update = self.redis_client.get('last_update')
                    last_update = last_update.decode('utf-8') if last_update else datetime.now().isoformat()
                    
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'portfolio_value': portfolio_value,
                            'daily_pnl': daily_pnl,
                            'total_pnl': total_pnl,
                            'account_balance': account_balance,
                            'positions': positions,
                            'last_update': last_update,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                else:
                    # Fallback to config values
                    account_balance = self.config.get('account', {}).get('initial_balance', 1000000.0)
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'portfolio_value': account_balance,
                            'daily_pnl': 0.0,
                            'total_pnl': 0.0,
                            'account_balance': account_balance,
                            'positions': {},
                            'last_update': datetime.now().isoformat(),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
            
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions from trading system"""
            try:
                if self.redis_client:
                    positions = self.redis_client.get('positions')
                    positions = json.loads(positions) if positions else {}
                else:
                    positions = {}
                    
                return jsonify({
                    'status': 'success',
                    'data': positions
                })
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
                
        @self.app.route('/api/trades')
        def get_trades():
            """Get recent trade history"""
            try:
                if self.redis_client:
                    trade_history = self.redis_client.get('trade_history')
                    trades = json.loads(trade_history) if trade_history else []
                else:
                    trades = []
                    
                return jsonify({
                    'status': 'success',
                    'data': trades
                })
            except Exception as e:
                logger.error(f"Error getting trades: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
                
        @self.app.route('/api/market-data')
        def get_market_data():
            """Get current market data"""
            try:
                if self.redis_client:
                    # Get latest price data
                    price_data = self.redis_client.get('market_data:15s')
                    if price_data:
                        data = json.loads(price_data)
                        latest = data[-1] if data else {}
                    else:
                        latest = {}
                        
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'current_price': latest.get('close', 0),
                            'volume': latest.get('volume', 0),
                            'timestamp': latest.get('timestamp', datetime.now().isoformat())
                        }
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'current_price': 0,
                            'volume': 0,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
            except Exception as e:
                logger.error(f"Error getting market data: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
                
        @self.app.route('/api/performance')
        def get_performance():
            """Get performance metrics"""
            try:
                if self.redis_client:
                    # Get trade history for performance calculation
                    trade_history = self.redis_client.get('trade_history')
                    trades = json.loads(trade_history) if trade_history else []
                    
                    # Calculate performance metrics
                    total_trades = len(trades)
                    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
                    losing_trades = total_trades - winning_trades
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    total_pnl = sum(t.get('pnl', 0) for t in trades)
                    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
                    
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'total_trades': total_trades,
                            'winning_trades': winning_trades,
                            'losing_trades': losing_trades,
                            'win_rate': win_rate,
                            'total_pnl': total_pnl,
                            'avg_trade': avg_trade
                        }
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0,
                            'win_rate': 0,
                            'total_pnl': 0,
                            'avg_trade': 0
                        }
                    })
            except Exception as e:
                logger.error(f"Error getting performance: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
    
    def _get_portfolio_from_individual_keys(self):
        """Get portfolio data from individual Redis keys"""
        try:
            portfolio_value = self.redis_client.get('portfolio_value')
            portfolio_value = float(portfolio_value) if portfolio_value else 1000000.0
            
            account_balance = self.redis_client.get('account_balance')
            account_balance = float(account_balance) if account_balance else 1000000.0
            
            total_pnl = self.redis_client.get('total_pnl')
            total_pnl = float(total_pnl) if total_pnl else 0.0
            
            daily_pnl = self.redis_client.get('daily_pnl')
            daily_pnl = float(daily_pnl) if daily_pnl else 0.0
            
            positions = self.redis_client.get('positions')
            positions = json.loads(positions) if positions else {}
            
            trade_history = self.redis_client.get('trade_history')
            trade_history = json.loads(trade_history) if trade_history else []
            
            return {
                'portfolio_value': portfolio_value,
                'account_balance': account_balance,
                'total_pnl': total_pnl,
                'daily_pnl': daily_pnl,
                'positions': positions,
                'trade_history': trade_history,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio from individual keys: {e}")
            return {
                'portfolio_value': 1000000.0,
                'account_balance': 1000000.0,
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'positions': {},
                'trade_history': [],
                'timestamp': datetime.now().isoformat()
            }
        
    def start(self):
        """Start the web application"""
        try:
            port = self.config.get('web_app', {}).get('port', 5000)
            self.app.run(host='0.0.0.0', port=port, debug=False)
        except Exception as e:
            logger.error(f"Error starting web app: {e}")
            raise
            
    def stop(self):
        """Stop the web application"""
        try:
            # Flask doesn't have a built-in stop method
            # This would need to be implemented with a proper shutdown mechanism
            logger.info("Web app stopped")
        except Exception as e:
            logger.error(f"Error stopping web app: {e}") 