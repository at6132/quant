from typing import Dict, Optional, Tuple
import numpy as np
from risk_engine import RiskEngine

class RiskManager:
    def __init__(self, account):
        self.account = account
        self.risk_engine = RiskEngine()
        self.position_history = []
        self.last_trade = None
        
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