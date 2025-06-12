from typing import Dict, List, Optional
import os
import json
import time

class Account:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.open_positions = []
        self.trade_log = []
        self.position_counter = 0
        self.last_price = None
        
        # Create analytics directory if it doesn't exist
        self.analytics_dir = 'analytics'
        os.makedirs(self.analytics_dir, exist_ok=True)
        self.analytics_file = os.path.join(self.analytics_dir, 'trades.jsonl')
    
    def get_capital(self) -> float:
        """Get current account capital."""
        return self.capital
    
    def get_last_price(self) -> float:
        """Get last known price."""
        return self.last_price if self.last_price is not None else 0.0
    
    def get_open_trades(self) -> List[Dict]:
        """Get list of open trades."""
        return self.open_positions
    
    def get_trade_log(self) -> List[Dict]:
        """Get list of all trades."""
        return self.trade_log
    
    def open_trade(self, price: float, size: float, direction: str) -> Optional[Dict]:
        """Open a new trade."""
        if size <= 0 or size > self.capital:
            return None
        
        # Update last price
        self.last_price = price
        
        # Create new position
        position = {
            'id': self.position_counter,
            'direction': direction,
            'entry_price': price,
            'size': size,
            'entry_time': time.time(),
            'pnl': 0.0
        }
        
        # Update account
        self.capital -= size
        self.open_positions.append(position)
        self.position_counter += 1
        
        # Log trade
        trade = {
            'timestamp': time.time(),
            'action': f'OPEN_{direction.upper()}',
            'price': price,
            'size': size,
            'capital': self.capital,
            'reason': 'model-signal'
        }
        self.trade_log.append(trade)
        
        # Save to analytics
        self._save_trade(trade)
        
        return trade
    
    def close_trade(self, position: Dict, price: float, reason: str = 'model-signal') -> Optional[Dict]:
        """Close an existing trade."""
        if position not in self.open_positions:
            return None
        
        # Update last price
        self.last_price = price
        
        # Calculate PnL
        if position['direction'] == 'long':
            pnl = (price - position['entry_price']) * position['size']
        else:  # short
            pnl = (position['entry_price'] - price) * position['size']
        
        # Update account
        self.capital += position['size'] + pnl
        self.open_positions.remove(position)
        
        # Log trade
        trade = {
            'timestamp': time.time(),
            'action': f'CLOSE_{position["direction"].upper()}',
            'price': price,
            'size': position['size'],
            'pnl': pnl,
            'net_pnl': pnl,  # No fees in paper trading
            'capital': self.capital,
            'reason': reason
        }
        self.trade_log.append(trade)
        
        # Save to analytics
        self._save_trade(trade)
        
        return trade
    
    def close_long_trades(self, price: float) -> Optional[Dict]:
        """Close all long positions."""
        trades = []
        for pos in self.open_positions[:]:
            if pos['direction'] == 'long':
                trade = self.close_trade(pos, price)
                if trade:
                    trades.append(trade)
        return trades[0] if trades else None
    
    def close_short_trades(self, price: float) -> Optional[Dict]:
        """Close all short positions."""
        trades = []
        for pos in self.open_positions[:]:
            if pos['direction'] == 'short':
                trade = self.close_trade(pos, price)
                if trade:
                    trades.append(trade)
        return trades[0] if trades else None
    
    def add_to_long_trades(self, price: float, size: float) -> Optional[Dict]:
        """Add to existing long positions."""
        if size <= 0 or size > self.capital:
            return None
        
        # Update last price
        self.last_price = price
        
        # Find existing long position
        long_pos = next((p for p in self.open_positions if p['direction'] == 'long'), None)
        if not long_pos:
            return None
        
        # Update position
        long_pos['size'] += size
        self.capital -= size
        
        # Log trade
        trade = {
            'timestamp': time.time(),
            'action': 'ADD_LONG',
            'price': price,
            'size': size,
            'capital': self.capital,
            'reason': 'model-signal'
        }
        self.trade_log.append(trade)
        
        # Save to analytics
        self._save_trade(trade)
        
        return trade
    
    def add_to_short_trades(self, price: float, size: float) -> Optional[Dict]:
        """Add to existing short positions."""
        if size <= 0 or size > self.capital:
            return None
        
        # Update last price
        self.last_price = price
        
        # Find existing short position
        short_pos = next((p for p in self.open_positions if p['direction'] == 'short'), None)
        if not short_pos:
            return None
        
        # Update position
        short_pos['size'] += size
        self.capital -= size
        
        # Log trade
        trade = {
            'timestamp': time.time(),
            'action': 'ADD_SHORT',
            'price': price,
            'size': size,
            'capital': self.capital,
            'reason': 'model-signal'
        }
        self.trade_log.append(trade)
        
        # Save to analytics
        self._save_trade(trade)
        
        return trade
    
    def _save_trade(self, trade: Dict):
        """Save trade to analytics file."""
        with open(self.analytics_file, 'a') as f:
            f.write(json.dumps(trade) + '\n')

    def check_tp_sl(self, price, timestamp):
        closed = []
        for pos in self.open_positions[:]:
            hit_tp = (pos['direction'] == 'long' and price >= pos['tp']) or (pos['direction'] == 'short' and price <= pos['tp'])
            hit_sl = (pos['direction'] == 'long' and price <= pos['sl']) or (pos['direction'] == 'short' and price >= pos['sl'])
            if hit_tp:
                closed.append(self.close_trade(pos, pos['tp'], timestamp, reason='take-profit'))
            elif hit_sl:
                closed.append(self.close_trade(pos, pos['sl'], timestamp, reason='stop-loss'))
        return closed 