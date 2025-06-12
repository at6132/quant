from typing import Dict, Optional
import numpy as np

class RiskManager:
    def __init__(self, account):
        self.account = account
        self.max_position_size = 0.1  # Max 10% of capital per trade
        self.min_position_size = 0.01  # Min 1% of capital per trade
        self.kelly_fraction = 0.5  # Conservative Kelly
        self.win_rate = 0.5  # Initial win rate estimate
        self.profit_factor = 1.5  # Initial profit factor estimate
        self.max_drawdown = 0.2  # Max 20% drawdown
        self.position_history = []
        
    def get_position_size(self, action_prob: float, account) -> float:
        """Calculate position size based on Kelly Criterion and action probability."""
        # Get current capital
        capital = account.get_capital()
        
        # Calculate Kelly fraction
        kelly = self._calculate_kelly()
        
        # Adjust position size based on action probability
        position_size = capital * kelly * action_prob
        
        # Apply position limits
        position_size = min(position_size, capital * self.max_position_size)
        position_size = max(position_size, capital * self.min_position_size)
        
        return position_size
    
    def _calculate_kelly(self) -> float:
        """Calculate Kelly Criterion fraction."""
        # Kelly = (p * b - q) / b
        # where p = win rate, q = loss rate, b = profit factor
        p = self.win_rate
        q = 1 - p
        b = self.profit_factor
        
        kelly = (p * b - q) / b
        kelly = max(0, kelly)  # Don't take negative positions
        kelly *= self.kelly_fraction  # Conservative Kelly
        
        return kelly
    
    def update_after_trade(self, trade_result: Dict):
        """Update win rate and profit factor after each trade."""
        self.position_history.append(trade_result)
        
        if len(self.position_history) >= 10:
            # Calculate win rate
            wins = sum(1 for t in self.position_history if t['net_pnl'] > 0)
            self.win_rate = wins / len(self.position_history)
            
            # Calculate profit factor
            gross_profit = sum(t['net_pnl'] for t in self.position_history if t['net_pnl'] > 0)
            gross_loss = abs(sum(t['net_pnl'] for t in self.position_history if t['net_pnl'] < 0))
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
            
            # Keep only last 100 trades
            self.position_history = self.position_history[-100:]
    
    def execute_trade(self, action: str, action_prob: float, price: float) -> Optional[Dict]:
        """Execute a trade with proper risk management."""
        # Get position size
        size = self.get_position_size(action_prob, self.account)
        
        # Check if we have enough capital
        if size > self.account.get_capital():
            return None
        
        # Execute trade based on action
        if action == 'OPEN_LONG':
            return self.account.open_trade(price, size, 'long')
        elif action == 'OPEN_SHORT':
            return self.account.open_trade(price, size, 'short')
        elif action == 'CLOSE_LONG':
            return self.account.close_long_trades(price)
        elif action == 'CLOSE_SHORT':
            return self.account.close_short_trades(price)
        elif action == 'ADD_LONG':
            return self.account.add_to_long_trades(price, size)
        elif action == 'ADD_SHORT':
            return self.account.add_to_short_trades(price, size)
        
        return None 