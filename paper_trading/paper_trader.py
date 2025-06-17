import logging
from typing import Dict, List, Union
import yaml
import time
from datetime import datetime
from .order_manager import OrderManager
from .risk_manager import RiskManager
from .portfolio_analytics import PortfolioAnalytics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize paper trader with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load trading settings
        self.trading_settings = self.config.get('trading', {})
        self.account_settings = self.config.get('account', {})
        
        # Initialize state
        self.positions = {}
        self.orders = []
        self.running = False
        
        logger.info("Paper trader initialized")
        
        try:
            # Initialize components
            self.order_manager = OrderManager(config)
            self.risk_manager = RiskManager(config)
            
            # Initialize portfolio analytics with default settings if not in config
            if 'portfolio_analytics' not in self.config:
                logger.warning("Portfolio analytics configuration not found, using default settings")
                self.config['portfolio_analytics'] = {
                    'risk_free_rate': 0.02,
                    'performance_thresholds': {
                        'min_sharpe': 1.0,
                        'min_sortino': 1.5,
                        'max_drawdown': 0.15,
                        'min_win_rate': 0.5
                    },
                    'metrics': {
                        'update_interval': 60,
                        'lookback_period': 252
                    },
                    'reporting': {
                        'daily_report': True,
                        'weekly_report': True,
                        'monthly_report': True,
                        'report_time': "23:59"
                    }
                }
            self.portfolio_analytics = PortfolioAnalytics(config)
            
        except Exception as e:
            logger.error("Error initializing paper trader: %s", str(e))
            raise
            
    def start(self):
        """Start the paper trader"""
        if self.running:
            logger.warning("Paper trader is already running")
            return
            
        try:
            # Start components
            self.risk_manager.start()
            
            self.running = True
            logger.info("Paper trader started")
            
        except Exception as e:
            logger.error("Error starting paper trader: %s", str(e))
            self.stop()
            raise
            
    def stop(self):
        """Stop the paper trader"""
        if not self.running:
            logger.warning("Paper trader is not running")
            return
            
        try:
            # Stop components
            self.risk_manager.stop()
            
            self.running = False
            logger.info("Paper trader stopped")
            
        except Exception as e:
            logger.error("Error stopping components: %s", str(e))
            raise
            
    def place_order(self, symbol: str, side: str, size: float, price: float) -> Dict:
        """Place a paper trade order
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            size: Order size
            price: Order price
            
        Returns:
            Order information
        """
        try:
            # Create order
            order = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'timestamp': datetime.now(),
                'status': 'filled'  # Paper trading fills immediately
            }
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = 0.0
                
            if side == 'buy':
                self.positions[symbol] += size
            else:
                self.positions[symbol] -= size
                
            # Store order
            self.orders.append(order)
            
            logger.info("Placed %s order for %s: %.2f @ %.2f", 
                       side, symbol, size, price)
            
            return order
            
        except Exception as e:
            logger.error("Error placing order: %s", str(e))
            return None
            
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        return self.positions.get(symbol, 0.0)
        
    def get_positions(self) -> Dict[str, float]:
        """Get all current positions
        
        Returns:
            Dictionary of symbol to position size
        """
        return self.positions.copy()
        
    def get_orders(self) -> List[Dict]:
        """Get order history
        
        Returns:
            List of order dictionaries
        """
        return self.orders.copy()
        
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics"""
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            self.positions,
            self.orders,
            self.positions
        )
        return self.portfolio_analytics.generate_portfolio_report(metrics)
        
    def get_daily_report(self) -> Dict:
        """Get daily performance report"""
        return self.portfolio_analytics.generate_daily_report() 