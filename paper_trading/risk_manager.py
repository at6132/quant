import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import yaml
import numpy as np
from .risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize risk manager with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load settings
        self.risk_settings = self.config.get('risk_management', {})
        self.account_settings = self.config.get('account', {})
        
        # Initialize state
        self.risk_metrics = {}
        self.alerts = []
        
        # Load risk management settings
        self.max_leverage = self.risk_settings['max_leverage']
        self.min_margin_level = self.risk_settings['min_margin_level']
        self.max_drawdown = self.risk_settings['max_drawdown']
        self.max_var_95 = self.risk_settings['max_var_95']
        self.position_limits = self.risk_settings['position_limits']
        
        # Initialize state
        self.positions = {}
        self.trades = []
        self.current_prices = {}
        self.running = False
        
        logger.info("Risk manager initialized")
        
    def start(self):
        """Start the risk manager"""
        self.running = True
        logger.info("Risk manager started")
        
    def stop(self):
        """Stop the risk manager"""
        self.running = False
        logger.info("Risk manager stopped")
        
    def update_positions(self, positions: Dict):
        """Update current positions"""
        self.positions = positions
        
    def update_trades(self, trades: List[Dict]):
        """Update trade history"""
        self.trades = trades
        
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices"""
        self.current_prices = prices
        
    def check_risk_limits(self) -> List[str]:
        """Check if any risk limits are exceeded"""
        if not self.running:
            return []
            
        alerts = []
        
        try:
            # Calculate metrics
            metrics = self._calculate_risk_metrics()
            
            # Check leverage
            if metrics['leverage'] > self.max_leverage:
                alerts.append(f"Leverage {metrics['leverage']:.2f} exceeds limit {self.max_leverage}")
                
            # Check margin level
            if metrics['margin_level'] < self.min_margin_level:
                alerts.append(f"Margin level {metrics['margin_level']:.2f}% below limit {self.min_margin_level}%")
                
            # Check max drawdown
            if metrics['max_drawdown'] > self.max_drawdown:
                alerts.append(f"Max drawdown {metrics['max_drawdown']:.2%} exceeds limit {self.max_drawdown:.2%}")
                
            # Check VaR
            if metrics['var_95'] > self.max_var_95:
                alerts.append(f"VaR(95%) {metrics['var_95']:.2%} exceeds limit {self.max_var_95:.2%}")
                
            # Check position limits
            long_positions = len([p for p in self.positions.values() if p['side'] == 'long'])
            short_positions = len([p for p in self.positions.values() if p['side'] == 'short'])
            
            if long_positions > self.position_limits['max_long_positions']:
                alerts.append(f"Number of long positions ({long_positions}) exceeds limit ({self.position_limits['max_long_positions']})")
                
            if short_positions > self.position_limits['max_short_positions']:
                alerts.append(f"Number of short positions ({short_positions}) exceeds limit ({self.position_limits['max_short_positions']})")
                
            # Check position size limits
            for symbol, position in self.positions.items():
                if position['quantity'] > self.position_limits['max_position_size']:
                    alerts.append(f"Position size for {symbol} ({position['quantity']}) exceeds limit ({self.position_limits['max_position_size']})")
                    
        except Exception as e:
            logger.error("Error checking risk limits: %s", str(e))
            
        return alerts
        
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        try:
            # Calculate total position value
            position_value = sum(
                pos['quantity'] * self.current_prices.get(symbol, 0)
                for symbol, pos in self.positions.items()
            )
            
            # Calculate margin used
            margin_used = sum(
                abs(pos['quantity'] * self.current_prices.get(symbol, 0) / self.max_leverage)
                for symbol, pos in self.positions.items()
            )
            
            # Calculate leverage
            leverage = margin_used / position_value if position_value > 0 else 0
            
            # Calculate margin level
            margin_level = (position_value / margin_used * 100) if margin_used > 0 else 100
            
            # Calculate max drawdown
            cumulative_pnl = np.cumsum([trade.get('pnl', 0) for trade in self.trades])
            max_dd = 0
            peak = cumulative_pnl[0] if len(cumulative_pnl) > 0 else 0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                dd = (peak - pnl) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
            # Calculate VaR
            returns = [trade.get('pnl', 0) for trade in self.trades]
            var_95 = np.percentile(returns, 5) if returns else 0
            
            return {
                'leverage': leverage,
                'margin_level': margin_level,
                'max_drawdown': max_dd,
                'var_95': abs(var_95)
            }
            
        except Exception as e:
            logger.error("Error calculating risk metrics: %s", str(e))
            return {
                'leverage': 0,
                'margin_level': 100,
                'max_drawdown': 0,
                'var_95': 0
            }
            
    def check_trade(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Check if a trade is allowed based on risk limits"""
        if not self.running:
            return False
            
        try:
            # Check position limits
            if side == 'buy' and len([p for p in self.positions.values() if p['side'] == 'long']) >= self.position_limits['max_long_positions']:
                logger.warning("Trade rejected: Maximum long positions reached")
                return False
                
            if side == 'sell' and len([p for p in self.positions.values() if p['side'] == 'short']) >= self.position_limits['max_short_positions']:
                logger.warning("Trade rejected: Maximum short positions reached")
                return False
                
            # Check position size
            if quantity > self.position_limits['max_position_size']:
                logger.warning("Trade rejected: Position size exceeds limit")
                return False
                
            # Check margin requirements
            trade_value = quantity * price
            margin_required = trade_value / self.max_leverage
            
            # Calculate current margin used
            current_margin = sum(
                abs(pos['quantity'] * self.current_prices.get(symbol, 0) / self.max_leverage)
                for symbol, pos in self.positions.items()
            )
            
            # Check if adding this trade would exceed margin limits
            if current_margin + margin_required > self.risk_settings['max_margin_used']:
                logger.warning("Trade rejected: Would exceed maximum margin used")
                return False
                
            return True
            
        except Exception as e:
            logger.error("Error checking trade: %s", str(e))
            return False

    def get_position_size(self, action_prob: float, account) -> Tuple[float, Dict]:
        """Calculate position size using the advanced risk engine."""
        # Get current state
        capital = account.get_capital()
        price = account.get_last_price()
        
        # Calculate position size using risk engine
        btc_size, debug_info = self.risk_engine.calculate_position_size(
            equity=capital,
            price=price,
            model_prob=action_prob
        )
        
        return btc_size, debug_info
    
    def update_after_trade(self, trade_result: Dict):
        """Update risk engine state after a trade."""
        if trade_result:
            # Update risk engine with trade result
            self.risk_engine.update_state(
                price=trade_result['price'],
                pnl=trade_result.get('net_pnl', 0.0)
            )
            
            # Store trade for history
            self.position_history.append(trade_result)
            self.last_trade = trade_result
            
            # Keep only last 100 trades
            if len(self.position_history) > 100:
                self.position_history = self.position_history[-100:]
    
    def execute_trade(self, action: str, action_prob: float, price: float) -> Optional[Dict]:
        """Execute a trade with proper risk management."""
        # Get position size from risk engine
        size, debug_info = self.get_position_size(action_prob, self.account)
        
        # Check if we have enough capital
        if size <= 0:
            return None
        
        # Execute trade based on action
        trade_result = None
        if action == 'OPEN_LONG':
            trade_result = self.account.open_trade(price, size, 'long')
        elif action == 'OPEN_SHORT':
            trade_result = self.account.open_trade(price, size, 'short')
        elif action == 'CLOSE_LONG':
            trade_result = self.account.close_long_trades(price)
        elif action == 'CLOSE_SHORT':
            trade_result = self.account.close_short_trades(price)
        elif action == 'ADD_LONG':
            trade_result = self.account.add_to_long_trades(price, size)
        elif action == 'ADD_SHORT':
            trade_result = self.account.add_to_short_trades(price, size)
        
        # Update risk engine if trade was executed
        if trade_result:
            self.update_after_trade(trade_result)
            # Add debug info to trade result
            trade_result['risk_debug'] = debug_info
        
        return trade_result 