import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import yaml
from .order_manager import OrderManager
from .risk_manager import RiskManager
from .monitoring import MonitoringSystem
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
        self.trade_history = []
        self.balance = self.account_settings.get('initial_balance', 100000.0)
        self.running = False
        
        logger.info("Paper trader initialized")
        
        try:
            # Initialize components
            self.order_manager = OrderManager(config)
            self.risk_manager = RiskManager(config)
            self.monitoring = MonitoringSystem(config)
            
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
            
            # Initialize state
            self.current_prices = {}
            
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
            self.monitoring.start()
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
            self.monitoring.stop()
            self.risk_manager.stop()
            
            self.running = False
            logger.info("Paper trader stopped")
            
        except Exception as e:
            logger.error("Error stopping components: %s", str(e))
            raise
            
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices"""
        self.current_prices = prices
        self.risk_manager.update_prices(prices)
        
    def execute_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Execute a trade"""
        try:
            # Create order
            order = self.order_manager.create_order(symbol, side, size, price)
            
            # Process order
            filled_order = self.order_manager.process_order(order)
            
            if filled_order:
                # Update positions
                self.positions = self.order_manager.get_positions()
                    
                # Update risk metrics
                self.risk_manager.update_positions(self.positions)
                
                # Log trade
                self.monitoring.log_trade(filled_order)
                
                # Update trade history
                self.trade_history.append(filled_order)
                
                return True
                
            return False
                    
        except Exception as e:
            logger.error("Error executing trade: %s", str(e))
            return False
            
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.positions
        
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history
        
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics"""
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            self.positions,
            self.trade_history,
            self.current_prices
        )
        return self.portfolio_analytics.generate_portfolio_report(metrics)
        
    def get_daily_report(self) -> Dict:
        """Get daily performance report"""
        return self.monitoring.generate_daily_report() 